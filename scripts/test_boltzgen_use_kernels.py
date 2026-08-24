#!/usr/bin/env python3
"""YAML/config -> BoltzGen --use_kernels false must emit a lowercase CLI token."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.agents.config_parameter_contract import canonicalize_config_parameter_value
from binderloop.agents.config_validation_agent import ConfigValidationAgent
from binderloop.agents.design_parameter_agent import DesignParameterAgent
from binderloop.config import load_config
from binderloop.models.boltzgen_renderer import render_boltzgen_command


ROOT = Path(__file__).resolve().parents[1]


def _flag_value(command, flag):
    tokens = list(map(str, command))
    index = tokens.index(flag)
    return tokens[index + 1]


class BoltzGenUseKernelsTests(unittest.TestCase):
    def test_renderer_coerces_python_bool_false_to_lowercase_token(self) -> None:
        command = render_boltzgen_command(
            spec_path=Path("configs/boltzgen_design_spec.yaml"),
            output_dir=Path("outputs/boltzgen_output"),
            params={"num_designs": 8, "budget": 10, "use_kernels": False},
        )
        self.assertEqual(_flag_value(command, "--use_kernels"), "false")
        self.assertNotIn("False", command)

    def test_renderer_preserves_lowercase_string_false(self) -> None:
        command = render_boltzgen_command(
            spec_path=Path("configs/boltzgen_design_spec.yaml"),
            output_dir=Path("outputs/boltzgen_output"),
            params={"num_designs": 8, "budget": 10, "use_kernels": "false"},
        )
        self.assertEqual(_flag_value(command, "--use_kernels"), "false")

    def test_pre_submit_validation_normalizes_bool_false(self) -> None:
        result = ConfigValidationAgent().validate_for_submission(
            {"num_designs": 8, "budget": 20, "use_kernels": False}
        )
        self.assertTrue(result.is_valid)
        self.assertEqual(result.corrected_config["use_kernels"], "false")
        self.assertEqual(result.validated_partition["runner"]["use_kernels"], "false")

    def test_contract_canonicalizes_bool_and_rejects_unknown_tokens(self) -> None:
        self.assertEqual(canonicalize_config_parameter_value("use_kernels", False), "false")
        self.assertEqual(canonicalize_config_parameter_value("use_kernels", "AUTO"), "auto")
        self.assertEqual(canonicalize_config_parameter_value("use_kernels", "maybe"), "maybe")

    def test_pdl1_owner_config_renders_use_kernels_false(self) -> None:
        cfg = load_config(ROOT / "configs/pdl1_structured_task_notemp_iptm035_simple.yaml")
        self.assertEqual(cfg.search_space.boltzgen["use_kernels"], "false")
        params = DesignParameterAgent().choose_boltzgen_parameters(cfg)
        self.assertEqual(params["use_kernels"], "false")
        command = render_boltzgen_command(
            spec_path=Path("configs/boltzgen_design_spec.yaml"),
            output_dir=Path("outputs/boltzgen_output"),
            params=params,
        )
        self.assertEqual(_flag_value(command, "--use_kernels"), "false")

    def test_design_parameter_agent_accepts_yaml_bool_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "t.cif").write_text("cif", encoding="utf-8")
            config_path = root / "task.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "schema_version: 1",
                        "owner:",
                        "  task_hard_constraints:",
                        "    task_name: demo",
                        "    target_structure_path: t.cif",
                        "    binder_length_range: [60, 100]",
                        "    binder_length_step: 20",
                        "    num_designs: 12",
                        "  filtering_budget:",
                        "    budget: 20",
                        "    run_filtering: true",
                        "  boltzgen_design_native:",
                        "    protocol: protein-anything",
                        "    use_kernels: false",
                        "  harness_search_space:",
                        "    model_order: [boltzgen]",
                    ]
                ),
                encoding="utf-8",
            )
            cfg = load_config(config_path)
            params = DesignParameterAgent().choose_boltzgen_parameters(cfg)
            self.assertEqual(params["use_kernels"], "false")


if __name__ == "__main__":
    unittest.main()
