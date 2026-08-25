from __future__ import annotations

import json
from pathlib import Path

from experiments.llm_3d_hotspot_validation_deepseek.src.deepseek_api import (
    APIConfig,
    ChatResponse,
)
from experiments.llm_3d_hotspot_validation_deepseek.src.deepseek_runner import (
    DeepSeekBenchmarkError,
    RateLimiter,
    RunSpec,
    _compact_features,
    load_api_config,
    run_one,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_SCHEMA = (
    REPO_ROOT
    / "experiments"
    / "llm_3d_hotspot_validation"
    / "process"
    / "prediction_schema.json"
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, RunSpec, dict[str, object]]:
    root = tmp_path / "experiment"
    spec = RunSpec(
        run_id="run_0123456789abcdef0123",
        case_id="case_deadbeef",
        condition="anonymous_no_web",
        replicate=1,
        task_path="runs/run_0123456789abcdef0123",
    )
    input_dir = root / spec.task_path / "input"
    input_dir.mkdir(parents=True)
    schema = json.loads(SOURCE_SCHEMA.read_text(encoding="utf-8"))
    _write_json(root / "process" / "prediction_schema.json", schema)
    _write_json(root / "process" / "identity_output_blocklist.json", {"terms": ["pd-l1"]})
    _write_json(input_dir / "prediction_schema.json", schema)
    residues = [
        {
            "token": f"T1:{index}",
            "residue_name": "ALA",
            "relative_sasa": 0.5,
            "residue_sasa_angstrom2": 40.0,
            "ca_angstrom": [float(index), 0.0, 0.0],
            "sidechain_heavy_atom_centroid_angstrom": [float(index), 1.0, 0.0],
            "per_atom_sasa": [{"atom": "CA", "sasa": 10.0}],
        }
        for index in range(1, 7)
    ]
    features = {"residues": residues, "coordinate_frame": "anonymous"}
    _write_json(input_dir / "features.json", features)
    _write_json(input_dir / "model_features.json", _compact_features(features))
    (input_dir / "prompt.md").write_text("Return a valid JSON envelope.\n", encoding="utf-8")
    (input_dir / "structure.cif").write_text("data_anonymous\n#\n", encoding="utf-8")
    prediction = {
        "schema_version": "1.0",
        "case_id": spec.case_id,
        "condition": spec.condition,
        "replicate": spec.replicate,
        "primary_hotspots": ["T1:1", "T1:2", "T1:3"],
        "alternate_hotspots": ["T1:4", "T1:5", "T1:6"],
        "pocket_groups": [["T1:1", "T1:2", "T1:3"], ["T1:4", "T1:5", "T1:6"]],
        "structural_rationale": "Exposed contiguous surface patch with mixed chemistry.",
        "recognition_status": "none",
    }
    return root, spec, prediction


class _FakeClient:
    def __init__(self, contents: list[str]) -> None:
        self.contents = contents
        self.calls = 0

    def chat(self, _messages: object) -> ChatResponse:
        content = self.contents[self.calls]
        self.calls += 1
        return ChatResponse(
            content=content,
            response_id=f"mock-{self.calls}",
            model="deepseek-v4-pro",
            finish_reason="stop",
            usage={"prompt_tokens": 1, "completion_tokens": 1},
            public_response={"content": content},
            transport_attempts=1,
        )


def test_compact_features_omits_redundant_per_atom_sasa() -> None:
    compact = _compact_features(
        {
            "coordinate_frame": "anonymous",
            "residues": [
                {
                    "token": "T1:1",
                    "residue_name": "TYR",
                    "relative_sasa": 0.8,
                    "per_atom_sasa": [{"atom": "CZ", "sasa": 5.0}],
                }
            ],
        }
    )
    assert compact["residues"][0]["token"] == "T1:1"
    assert compact["residues"][0]["rSASA"] == 0.8
    assert "per_atom_sasa" not in compact["residues"][0]


def test_run_one_repairs_schema_once_and_freezes_audit_under_scratch(tmp_path: Path) -> None:
    root, spec, prediction = _fixture(tmp_path)
    valid = json.dumps(
        {
            "prediction": prediction,
            "process_markdown": "Ranked an exposed geometric pocket using local structure only.",
        }
    )
    client = _FakeClient(["{}", valid])
    config = APIConfig(api_key="secret-never-persisted", base_url="https://api.deepseek.com")

    result = run_one(
        root,
        spec,
        client,
        config,
        RateLimiter(0),
        max_input_bytes=1_000_000,
        dry_run=False,
    )

    assert result.status == "prediction"
    assert result.schema_repair_used is True
    assert client.calls == 2
    assert (root / spec.task_path / "output" / "prediction.json").is_file()
    audit = root / spec.task_path / "scratch" / "audit"
    assert (audit / "request_manifest.json").is_file()
    assert (audit / "response_manifest.json").is_file()
    assert len(list((audit / "attempts").glob("*.json"))) == 2
    persisted = "\n".join(
        path.read_text(encoding="utf-8")
        for path in root.rglob("*")
        if path.is_file()
    )
    assert "secret-never-persisted" not in persisted


def test_identity_leak_fails_after_single_repair(tmp_path: Path) -> None:
    root, spec, prediction = _fixture(tmp_path)
    leaked = json.dumps(
        {
            "prediction": prediction,
            "process_markdown": "I recognized PD-L1 from memory.",
        }
    )
    client = _FakeClient([leaked, leaked])
    config = APIConfig(api_key="x", base_url="https://api.deepseek.com")

    result = run_one(
        root,
        spec,
        client,
        config,
        RateLimiter(0),
        max_input_bytes=1_000_000,
        dry_run=False,
    )

    assert result.status == "terminal_invalid_output"
    assert client.calls == 2
    assert not (root / spec.task_path / "output" / "prediction.json").exists()
    assert (root / "process" / "failures" / f"{spec.run_id}.md").is_file()


def test_llm_endpoint_config_resolves_secret_and_endpoint_defaults(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "llm_endpoints.test.json"
    _write_json(
        config_path,
        {
            "enabled": True,
            "default_model": "deepseek_test",
            "secrets": {"API_KEY": {"value": "file-secret"}},
            "endpoints": {
                "deepseek_test": {
                    "provider": "deepseek",
                    "base_url": "https://api.deepseek.test/v1",
                    "model": "deepseek-v4-pro",
                    "api_key_env": "API_KEY",
                    "thinking": "high",
                    "timeout_seconds": 321,
                    "max_retries": 2,
                    "retry_backoff_seconds": 1.5,
                    "max_output_tokens": 12345,
                }
            },
        },
    )
    monkeypatch.delenv("API_KEY", raising=False)

    config = load_api_config(llm_config=config_path)

    assert config.api_key == "file-secret"
    assert config.base_url == "https://api.deepseek.test/v1"
    assert config.model == "deepseek-v4-pro"
    assert config.thinking is True
    assert config.reasoning_effort == "high"
    assert config.timeout_seconds == 321
    assert config.transport_retries == 2
    assert config.backoff_base_seconds == 1.5
    assert config.max_tokens == 12345
    assert config.endpoint_key == "deepseek_test"
    assert config.credential_source == "llm_config:llm_endpoints.test.json:deepseek_test"
    assert "file-secret" not in repr(config)


def test_llm_endpoint_environment_secret_overrides_file(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "llm_endpoints.test.json"
    _write_json(
        config_path,
        {
            "enabled": True,
            "default_model": "deepseek",
            "secrets": {"API_KEY": "file-secret"},
            "endpoints": {
                "deepseek": {
                    "base_url": "https://api.deepseek.com",
                    "model": "deepseek-v4-pro",
                    "api_key_env": "API_KEY",
                }
            },
        },
    )
    monkeypatch.setenv("API_KEY", "environment-secret")

    config = load_api_config(llm_config=config_path)

    assert config.api_key == "environment-secret"


def test_llm_endpoint_config_requires_resolvable_key(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "llm_endpoints.test.json"
    _write_json(
        config_path,
        {
            "enabled": True,
            "default_model": "deepseek",
            "secrets": {},
            "endpoints": {
                "deepseek": {
                    "base_url": "https://api.deepseek.com",
                    "model": "deepseek-v4-pro",
                    "api_key_env": "API_KEY",
                }
            },
        },
    )
    monkeypatch.delenv("API_KEY", raising=False)

    try:
        load_api_config(llm_config=config_path)
    except DeepSeekBenchmarkError as exc:
        assert "no API key resolved" in str(exc)
    else:
        raise AssertionError("missing endpoint API key should be rejected")
