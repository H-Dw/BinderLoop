
import json
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Union


def conda_run_command(
    command: List[str],
    *,
    env_name: str,
    conda_executable: str = "conda",
) -> List[str]:
    """Wrap a command for deterministic execution in a named Conda environment."""
    env_name = str(env_name or "").strip()
    conda_executable = str(conda_executable or "").strip()
    if not env_name:
        raise ValueError("env_name is required for direct Conda execution")
    if not conda_executable:
        raise ValueError("conda_executable is required for direct Conda execution")
    return [conda_executable, "run", "--no-capture-output", "-n", env_name, *map(str, command)]


@dataclass
class CommandResult:
    name: str
    command: List[str]
    cwd: str
    returncode: Optional[int]
    stdout: str = ""
    stderr: str = ""
    dry_run: bool = True


class Runner:
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run

    def run(self, name: str, command: List[str], cwd: Union[str, Path]) -> CommandResult:
        cwd = str(cwd)
        if self.dry_run:
            return CommandResult(name=name, command=command, cwd=cwd, returncode=None, dry_run=True)
        proc = subprocess.run(command, cwd=cwd, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return CommandResult(name=name, command=command, cwd=cwd, returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr, dry_run=False)

    @staticmethod
    def save_results(results: List[CommandResult], path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2), encoding="utf-8")
