#!/usr/bin/env python3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from binderloop.agents.config_validation_agent import ConfigValidationAgent


def main() -> None:
    params = {
        "protocol": "protein-anything",
        "num_designs": 100,
        "budget": 99999,
        "run_filtering": True,
        "additional_filters": ["iptm>0.35"],
        "config_overrides": ["filtering", "iptm_threshold=0.25"],
    }

    validation = ConfigValidationAgent().validate_for_submission(params)

    assert validation.is_valid is True
    assert validation.corrected_config["additional_filters"] == ["iptm>0.35"]
    assert validation.corrected_config["config_overrides"] == []
    assert any("iptm_threshold" in issue.get("correction", "") for issue in validation.issues)

    print("OK: BoltzGen iPTM threshold stays in additional_filters and unsupported iptm_threshold override is dropped")


if __name__ == "__main__":
    main()
