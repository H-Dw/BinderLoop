#!/usr/bin/env python3
"""Offline tests for LLM hotspot selection, isolation, and memorization scoring."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.agents.hotspot_selection_agent import (
    HotspotSelectionAgent,
    anonymize_hotspot_prompt,
    prompt_contains_identity,
)
from binderloop.analysis.hotspot_compare import (
    compare_run_to_prior,
    jaccard_index,
    load_prior_hotspots,
)
from binderloop.analysis.hotspot_descriptors import (
    build_target_residue_table,
    deterministic_surface_hotspots,
    sanitize_hotspot_tokens,
)
from binderloop.analysis.hotspot_memorization import (
    classify_memorization,
    literature_keyword_hits,
    score_probe_response,
)
from binderloop.config import ConfigError, HotspotSelectionSpec, load_config
from binderloop.llm import (
    LLMConfigError,
    LLMSettings,
    ModelEndpoint,
    OpenAICompatibleClient,
    reject_online_web_search_model,
    strip_web_search_payload,
)


def _atom(serial: int, name: str, resname: str, chain: str, resseq: int, x: float, y: float, z: float) -> str:
    return (
        "ATOM  "
        + f"{serial:>5d}"
        + " "
        + f"{name:<4s}"
        + " "
        + f"{resname:>3s}"
        + " "
        + chain
        + f"{resseq:>4d}"
        + "    "
        + f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
        + "  1.00  0.00           C"
    )


def _synthetic_target_pdb() -> str:
    residues = [
        (1, "ALA", 0.0, 0.0, 0.0),
        (2, "VAL", 3.8, 0.2, 0.1),
        (3, "PHE", 7.6, 0.0, 0.4),
        (4, "LEU", 11.4, 0.3, 0.0),
        (5, "LYS", 15.2, 8.0, 6.0),
        (6, "GLU", 19.0, 8.5, 6.2),
        (7, "TRP", 22.8, 0.1, 0.2),
        (8, "TYR", 26.6, 0.0, 0.0),
        (9, "SER", 30.4, 0.4, 0.3),
        (10, "ILE", 34.2, 0.2, 0.1),
    ]
    lines = []
    for index, (resseq, resname, x, y, z) in enumerate(residues, start=1):
        lines.append(_atom(index, "CA", resname, "A", resseq, x, y, z))
    return "\n".join(lines) + "\n"


def _owner_yaml(*, tmp: Path, hotspots=None, binding_types=None, boltzgen_input=None, hotspot_selection=None, task_name="anon_task") -> Path:
    structure = tmp / "target.pdb"
    structure.write_text(_synthetic_target_pdb(), encoding="utf-8")
    hotspot_block = ""
    if hotspots:
        hotspot_block = "    hotspots:\n" + "".join("    - %s\n" % item for item in hotspots)
    binding_block = ""
    if binding_types:
        binding_block = "    target_binding_types: %s\n" % json.dumps(binding_types)
    boltzgen_block = ""
    if boltzgen_input:
        boltzgen_block = "    boltzgen_input_path: %s\n" % boltzgen_input
    hs = hotspot_selection or {}
    hs_lines = ["    hotspot_selection:"]
    hs_lines.append("      enabled: %s" % str(bool(hs.get("enabled", False))).lower())
    hs_lines.append("      allow_web_search: %s" % str(bool(hs.get("allow_web_search", False))).lower())
    hs_lines.append("      require_llm: %s" % str(bool(hs.get("require_llm", False))).lower())
    hs_lines.append("      min_hotspots: %s" % int(hs.get("min_hotspots", 3)))
    hs_lines.append("      max_hotspots: %s" % int(hs.get("max_hotspots", 6)))
    text = "\n".join([
        "schema_version: 1",
        "owner:",
        "  task_hard_constraints:",
        "    task_name: %s" % task_name,
        "    target_structure_path: target.pdb",
        "    target_chain_id: A",
        hotspot_block.rstrip("\n"),
        binding_block.rstrip("\n"),
        boltzgen_block.rstrip("\n"),
        "    binder_length_range: [60, 80]",
        "    binder_length_step: 10",
        "    num_designs: 4",
        "  boltzgen_design_native:",
        "    protocol: protein-anything",
        "  filtering_budget:",
        "    budget: 10",
        "  active_learning_and_rollback:",
        "    max_rounds: 2",
        "    branch_width: 2",
        "  runtime_resources:",
        "    resource:",
        "      backend: dry_run",
        "  llm_context_learning:",
        "\n".join(hs_lines),
        "",
    ])
    text = "\n".join(line for line in text.splitlines() if line.strip() != "")
    path = tmp / "task.yaml"
    path.write_text(text + "\n", encoding="utf-8")
    return path


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.last_kwargs = {}
        self.settings = LLMSettings(
            default_model="test",
            endpoints={
                "test": ModelEndpoint(
                    name="test",
                    base_url="https://example.invalid/v1",
                    api_key="k",
                    extra_body={"plugins": [{"id": "web"}], "tools": [{"type": "web_search"}]},
                )
            },
            enabled=True,
        )

    def available(self):
        return True

    def chat_json(self, **kwargs):
        self.last_kwargs = dict(kwargs)
        return dict(self.payload)


class ConfigIsolationTests(unittest.TestCase):
    def test_enabled_rejects_yaml_hotspots(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _owner_yaml(tmp=Path(directory), hotspots=["A:3"], hotspot_selection={"enabled": True})
            with self.assertRaises(ConfigError) as caught:
                load_config(path)
            self.assertIn("forbids owner.task_hard_constraints.hotspots", str(caught.exception))

    def test_enabled_rejects_binding_types(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _owner_yaml(
                tmp=Path(directory),
                binding_types=[{"chain": {"id": "A", "binding": "3,7"}}],
                hotspot_selection={"enabled": True},
            )
            with self.assertRaises(ConfigError) as caught:
                load_config(path)
            self.assertIn("target_binding_types", str(caught.exception))

    def test_enabled_rejects_boltzgen_input_binding_types(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bg = root / "bg.yaml"
            bg.write_text(
                "entities:\n- file:\n    path: target.pdb\n    binding_types:\n    - chain:\n        id: A\n        binding: 3,7\n",
                encoding="utf-8",
            )
            path = _owner_yaml(tmp=root, boltzgen_input="bg.yaml", hotspot_selection={"enabled": True})
            with self.assertRaises(ConfigError) as caught:
                load_config(path)
            self.assertIn("boltzgen_input binding_types", str(caught.exception))

    def test_enabled_empty_hotspots_ok(self):
        with tempfile.TemporaryDirectory() as directory:
            path = _owner_yaml(tmp=Path(directory), hotspot_selection={"enabled": True, "require_llm": False})
            cfg = load_config(path)
            self.assertTrue(cfg.hotspot_selection.enabled)
            self.assertFalse(cfg.hotspot_selection.allow_web_search)
            self.assertEqual(cfg.target.hotspots, [])


class WebSearchStripTests(unittest.TestCase):
    def test_strip_tools_and_plugins(self):
        cleaned = strip_web_search_payload({
            "model": "x",
            "tools": [{"type": "web_search"}],
            "plugins": [{"id": "web"}],
            "temperature": 0.1,
            "extra": [{"id": "web"}, {"id": "keep"}],
        })
        self.assertNotIn("tools", cleaned)
        self.assertNotIn("plugins", cleaned)
        self.assertEqual(cleaned["temperature"], 0.1)
        self.assertEqual(cleaned["extra"], [{"id": "keep"}])

    def test_reject_online_model(self):
        with self.assertRaises(LLMConfigError):
            reject_online_web_search_model("openai/gpt-4o:online")
        reject_online_web_search_model("gpt-4o-mini")

    def test_create_chat_completion_strips_web_search(self):
        class _Response:
            def __enter__(self):
                return self
            def __exit__(self, *_args):
                return False
            def read(self):
                return json.dumps({"choices": [{"message": {"content": "{\"ok\": true}"}}], "usage": {}}).encode("utf-8")

        endpoint = ModelEndpoint(
            name="test",
            base_url="https://example.invalid/v1",
            api_key="test-key",
            extra_body={"plugins": [{"id": "web"}], "tools": [{"type": "web_search_preview"}]},
            retry_backoff_seconds=0,
            max_retries=1,
        )
        client = OpenAICompatibleClient(LLMSettings(default_model="test", endpoints={"test": endpoint}, enabled=True))
        with patch("binderloop.llm.urllib.request.urlopen", return_value=_Response()) as mocked:
            client.create_chat_completion(
                messages=[{"role": "user", "content": "x"}],
                allow_web_search=False,
            )
        payload = json.loads(mocked.call_args.args[0].data.decode("utf-8"))
        self.assertNotIn("plugins", payload)
        self.assertNotIn("tools", payload)


class DescriptorAndSanitizeTests(unittest.TestCase):
    def test_residue_table_from_synthetic_pdb(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t.pdb"
            path.write_text(_synthetic_target_pdb(), encoding="utf-8")
            table = build_target_residue_table(path, chain_id="A")
            self.assertEqual(table.chain_id, "A")
            self.assertEqual(table.residue_count, 10)
            self.assertTrue(table.sequence.startswith("AVF"))
            tokens = table.tokens()
            self.assertIn("A:3", tokens)
            phe = table.by_token()["A:3"]
            self.assertTrue(phe.aromatic)
            self.assertGreater(phe.hydrophobicity, 0)

    def test_sanitize_and_inertia(self):
        allowed = ["A:1", "A:2", "A:3", "A:4", "A:5", "A:6"]
        cleaned, notes = sanitize_hotspot_tokens(
            ["A:1", "B:99", "A:2", "A:3", "A:4"],
            allowed_tokens=allowed,
            chain_id="A",
            min_hotspots=3,
            max_hotspots=4,
            previous=["A:1", "A:2", "A:3"],
            max_change_per_round=1,
        )
        self.assertLessEqual(len(cleaned), 4)
        self.assertTrue(all(token in allowed for token in cleaned))
        self.assertTrue(any("dropped_wrong_chain" in note or "dropped_unknown" in note for note in notes) or "inertia_capped_changes" in notes)

    def test_deterministic_fallback_count(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t.pdb"
            path.write_text(_synthetic_target_pdb(), encoding="utf-8")
            table = build_target_residue_table(path, chain_id="A")
            selected = deterministic_surface_hotspots(table, min_hotspots=3, max_hotspots=4)
            self.assertGreaterEqual(len(selected), 3)
            self.assertLessEqual(len(selected), 4)


class AnonymityAndAgentTests(unittest.TestCase):
    def test_anonymize_strips_paths_and_names(self):
        payload = anonymize_hotspot_prompt({
            "task_name": "PD-L1_len50",
            "structure_path": "/data/PD-L1.cif",
            "pdb_id": "5o45",
            "residue_table": {"residues": [{"token": "A:40", "aa": "Y"}]},
        })
        hits = prompt_contains_identity(payload, forbidden_tokens=["PD-L1", "5o45", "/data/PD-L1.cif"])
        self.assertEqual(hits, [])
        self.assertNotIn("task_name", payload)
        self.assertNotIn("structure_path", payload)
        self.assertEqual(payload["residue_table"]["residues"][0]["token"], "A:40")

    def test_agent_uses_llm_and_disables_web_search(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t.pdb"
            path.write_text(_synthetic_target_pdb(), encoding="utf-8")
            table = build_target_residue_table(path, chain_id="A")
            fake = FakeLLM({"hotspots": ["A:3", "A:7", "A:8"], "rationale": "exposed aromatic patch", "expected_signal_next_round": "higher hotspot_contact", "changes_from_previous": []})
            agent = HotspotSelectionAgent(fake, spec=HotspotSelectionSpec(enabled=True, require_llm=False, min_hotspots=3, max_hotspots=4), require_llm=False)
            result = agent.select(residue_table=table, previous_hotspots=["A:3", "A:4", "A:7"])
            self.assertTrue(result.llm_used)
            self.assertEqual(fake.last_kwargs.get("allow_web_search"), False)
            user = fake.last_kwargs.get("user") or {}
            hits = prompt_contains_identity(user, forbidden_tokens=["PD-L1", "target.pdb"])
            self.assertEqual(hits, [])
            self.assertGreaterEqual(len(result.hotspots), 3)

    def test_agent_fallback_and_round_update(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t.pdb"
            path.write_text(_synthetic_target_pdb(), encoding="utf-8")
            table = build_target_residue_table(path, chain_id="A")
            agent = HotspotSelectionAgent(None, spec=HotspotSelectionSpec(enabled=True, require_llm=False, min_hotspots=3, max_hotspots=4), require_llm=False)
            first = agent.select(residue_table=table)
            self.assertFalse(first.llm_used)
            fake = FakeLLM({"hotspots": first.hotspots[:2] + ["A:8"], "rationale": "shift toward aromatic", "expected_signal_next_round": "iptm", "changes_from_previous": ["+A:8"]})
            agent.llm = fake
            second = agent.select(
                residue_table=table,
                previous_hotspots=first.hotspots,
                round_evidence={"evaluation": {"success_count": 0, "total_candidates": 8, "tag_counts": {"hotspot_miss": 5}}},
            )
            self.assertTrue(second.llm_used)
            self.assertLessEqual(len(set(second.hotspots) ^ set(first.hotspots)), 2)


class CompareAndMemorizationTests(unittest.TestCase):
    def test_compare_best_round_to_prior(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prior_path = root / "prior.yaml"
            prior_path.write_text("hotspots: ['A:40', 'A:99', 'A:107']\n", encoding="utf-8")
            for round_id, hotspots, success, rank in (
                (0, ["A:3", "A:7", "A:8"], 0, [0, 0, 0, 0, 0]),
                (1, ["A:40", "A:99", "A:110"], 2, [2, 0.1, 0.5, -8.0, -2.0]),
            ):
                round_dir = root / ("round_%02d" % round_id)
                round_dir.mkdir()
                (round_dir / "llm_hotspot_selection.json").write_text(json.dumps({
                    "hotspots": hotspots,
                    "round_metrics": {"success_count": success, "total_candidates": 10, "success_rate": success / 10, "round_rank_key": rank},
                }), encoding="utf-8")
            prior = load_prior_hotspots(prior_path)
            payload = compare_run_to_prior(root, prior)
            self.assertEqual(payload["best_round"]["round_id"], 1)
            self.assertGreater(payload["hotspot_comparison"]["jaccard_residue_numbers"], 0.4)

    def test_memorization_scoring(self):
        scored = score_probe_response(
            ["A:40", "A:99", "A:107"],
            ["A:40", "A:99", "A:107"],
            rationale="From the PD-1/PD-L1 crystal complex PDB 5O45.",
            condition="identity_only",
        )
        self.assertEqual(scored["jaccard_residue_numbers"], 1.0)
        self.assertTrue(literature_keyword_hits(scored["rationale"]))
        verdict = classify_memorization(
            identity_jaccard=1.0,
            structure_jaccard=0.1,
            identity_keyword_hits=scored["literature_keyword_hits"],
            identity_overlap=3,
        )
        self.assertEqual(verdict, "likely_memorized")
        structure_only = classify_memorization(
            identity_jaccard=0.0,
            structure_jaccard=0.67,
            identity_keyword_hits=[],
            identity_overlap=0,
        )
        self.assertEqual(structure_only, "likely_structure_reasoned")

    def test_jaccard_empty(self):
        self.assertEqual(jaccard_index([], []), 1.0)
        self.assertEqual(jaccard_index(["A:1"], []), 0.0)


if __name__ == "__main__":
    unittest.main()
