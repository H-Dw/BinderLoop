#!/usr/bin/env python3
"""Config schema cleanup and hard-constraint freeze regressions."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.agents.config_parameter_contract import invalid_config_value_keys, supported_config_changes
from binderloop.config import ConfigError, load_config
from binderloop.orchestration.orchestrator import BinderDesignOrchestrator


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


class ConfigCleanupTests(unittest.TestCase):
    def _config(self, root: Path, extra: str = "") -> Path:
        return _write(root / "task.yaml", "\n".join([
            "schema_version: 1", "owner:", "  task_hard_constraints:",
            "    task_name: demo", "    target_structure_path: t.cif",
            "    binder_length_range: [60, 100]", "    binder_length_step: 20",
            "    num_designs: 12", "  sampler_bounds:",
            "    noise_scale: {min: 0.6, max: 0.9}",
            "    step_scale: {min: 0.6, max: 1.0}",
            "    alpha: {min: 0.001, max: 0.05}",
            "  filtering_budget:", "    budget: 20",
            "    run_filtering: true", "  boltzgen_design_native:",
            "    protocol: protein-anything", "  harness_search_space:",
            "    model_order: [boltzgen]", extra, ""]))

    def test_owner_schema_compiles_to_legacy_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "t.cif").write_text("cif")
            cfg = load_config(self._config(root))
            self.assertEqual(cfg.search_space.binder_lengths, [60, 80, 100])
            self.assertEqual(cfg.search_space.num_designs_per_round, 12)
            self.assertEqual(cfg.search_space.boltzgen["num_designs"], 12)
            self.assertEqual(cfg.search_space.boltzgen["budget"], 20)
            self.assertEqual(cfg.owner.task_hard_constraints.num_designs, 12)
            self.assertEqual(cfg.owner.sampler_bounds.noise_scale, {"min": 0.6, "max": 0.9})

    def test_use_kernels_yaml_bool_false_is_canonicalized(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "t.cif").write_text("cif")
            text = self._config(root).read_text().replace(
                "    protocol: protein-anything",
                "    protocol: protein-anything\n    use_kernels: false",
            )
            cfg = load_config(_write(root / "kernels.yaml", text))
            self.assertEqual(cfg.search_space.boltzgen["use_kernels"], "false")

    def test_use_kernels_yaml_string_false_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "t.cif").write_text("cif")
            text = self._config(root).read_text().replace(
                "    protocol: protein-anything",
                "    protocol: protein-anything\n    use_kernels: 'false'",
            )
            cfg = load_config(_write(root / "kernels.yaml", text))
            self.assertEqual(cfg.search_space.boltzgen["use_kernels"], "false")

    def test_use_kernels_invalid_value_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "t.cif").write_text("cif")
            text = self._config(root).read_text().replace(
                "    protocol: protein-anything",
                "    protocol: protein-anything\n    use_kernels: maybe",
            )
            with self.assertRaisesRegex(ConfigError, "use_kernels"):
                load_config(_write(root / "bad.yaml", text))

    def test_schema_version_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = _write(Path(tmp) / "x.yaml", "owner: {}\n")
            with self.assertRaisesRegex(ConfigError, "schema_version"):
                load_config(p)

    def test_num_designs_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = _write(Path(tmp) / "x.yaml", "schema_version: 1\nowner:\n  task_hard_constraints:\n    target_structure_path: t.cif\n    binder_length_range: [60, 80]\n")
            with self.assertRaisesRegex(ConfigError, "num_designs"):
                load_config(p)

    def test_legacy_user_fields_are_rejected(self) -> None:
        cases = [
            "task:\n  max_binders_per_round: 8\n",
            "search_space:\n  boltzgen: {}\n",
            "target:\n  structure_path: t.cif\n",
        ]
        for legacy in cases:
            with self.subTest(legacy=legacy), tempfile.TemporaryDirectory() as tmp:
                p = _write(Path(tmp) / "x.yaml", "schema_version: 1\nowner: {}\n" + legacy)
                with self.assertRaises(ConfigError): load_config(p)

    def test_unknown_owner_and_native_fields_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/"t.cif").write_text("cif")
            with self.assertRaises(ConfigError): load_config(self._config(root, "  mystery_owner: {}"))
            text=self._config(root).read_text().replace("    protocol: protein-anything", "    protocol: protein-anything\n    mystery_knob: 1")
            p=_write(root/"bad.yaml",text)
            with self.assertRaises(ConfigError): load_config(p)

    def test_sampler_bounds_are_typed_and_ordered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/"t.cif").write_text("cif")
            text=self._config(root).read_text().replace("{min: 0.6, max: 0.9}", "{min: 0.9, max: 0.6}")
            with self.assertRaisesRegex(ConfigError, "min cannot exceed max"):
                load_config(_write(root/"bad.yaml",text))

    def test_formal_configs_use_owner_schema_examples_stay_native(self) -> None:
        root=Path(__file__).resolve().parents[1]
        formal=[p for p in (root/"configs").glob("*.yaml")  ]
        self.assertGreaterEqual(len(formal), 14)
        for p in formal:
            data=__import__('yaml').safe_load(p.read_text())
            self.assertEqual(data["schema_version"], 1)
            self.assertIn("owner", data)
            self.assertIn("num_designs", data["owner"]["task_hard_constraints"])


if __name__ == "__main__":
    unittest.main()
