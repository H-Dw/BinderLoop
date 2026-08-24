
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class DesignJob:
    job_id: str
    target_structure: str
    chain_id: str
    hotspots: List[str]
    binder_length: int
    # ``seed`` is an internal, defaulted field retained only for adapters that
    # accept a seed argument (e.g. ODesign). BoltzGen does not support seed
    # control, so the closed loop never multiplies jobs by seed and never sets
    # this from user configuration; one round maps to a single design job.
    seed: int = 0
    params: Dict[str, Any] = field(default_factory=dict)
    output_dir: str = "outputs/job"


class ModelAdapter:
    name: str

    def build_command(self, job: DesignJob) -> List[str]:
        raise NotImplementedError

    def expected_outputs(self, job: DesignJob) -> Dict[str, str]:
        return {"output_dir": job.output_dir}
