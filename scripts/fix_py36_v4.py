#!/usr/bin/env python3
"""Final pass: fix remaining X|Y unions (not X|None already handled)."""
from __future__ import print_function
import re
from pathlib import Path

ROOT = Path("/aceph/daweihuang/program/binder-harness")

FILES = [
    "binderloop/analysis/parsers.py",
    "binderloop/analysis/structure_features.py",
    "binderloop/analysis/scoring.py",
    "binderloop/memory.py",
    "binderloop/communication.py",
    "binderloop/llm.py",
    "binderloop/visualization/iteration_metrics_plot.py",
    "binderloop/secrets.py",
    "binderloop/agents/run_monitor_agent.py",
    "binderloop/agents/structure_evaluation_agent.py",
    "binderloop/agents/binder_quality_analysis_agent.py",
    "binderloop/agents/config_validation_agent.py",
    "binderloop/agents/fragment_template_mining_agent.py",
    "binderloop/agents/evaluation_agent.py",
    "binderloop/agents/design_parameter_agent.py",
    "binderloop/agents/diagnostic_coach_agent.py",
    "binderloop/agents/taiji_execution_agent.py",
    "binderloop/agents/input_configuration_agent.py",
    "binderloop/agents/result_ingestion_agent.py",
    "binderloop/agents/active_learning_policy_agent.py",
    "binderloop/models/boltzgen_adapter.py",
    "binderloop/orchestration/runner.py",
    "binderloop/orchestration/orchestrator.py",
    "scripts/run_closed_loop_orchestrator.py",
    "scripts/run_boltzgen_complete_path_test.py",
]

# Key substitutions - order matters!
SUBSTITUTIONS = [
    # str | Path -> Union[str, Path]
    (r'\bstr\s*\|\s*Path\b', 'Union[str, Path]'),
    # Path | str -> Union[Path, str]  
    (r'\bPath\s*\|\s*str\b', 'Union[Path, str]'),
    # str | Union[  -> Union[str, Union[
    (r'\bstr\s*\|\s*Union\[', 'Union[str, Union['),
    # Path | Union[ -> Union[Path, Union[
    (r'\bPath\s*\|\s*Union\[', 'Union[Path, Union['),
    # str | Optional[Path] -> Union[str, Optional[Path]]
    (r'\bstr\s*\|\s*Optional\[Path\]', 'Union[str, Optional[Path]]'),
    # Path | Optional[str] -> Union[Path, Optional[str]]
    (r'\bPath\s*\|\s*Optional\[str\]', 'Union[Path, Optional[str]]'),
    # Optional[str] | Optional[Path] -> Union[Optional[str], Optional[Path]]
    (r'Optional\[str\]\s*\|\s*Optional\[Path\]', 'Union[Optional[str], Optional[Path]]'),
    # "ClassName" | None -> Optional["ClassName"]
    (r'"([^"]+)"\s*\|\s*None\b', r'Optional["\1"]'),
    # "ClassName" | "ClassName2" -> Union["ClassName", "ClassName2"]
    (r'"([^"]+)"\s*\|\s*"([^"]+)"', r'Union["\1", "\2"]'),
    # Last resort: any_identifier | any_identifier that's NOT inside Union/Optional/None
    # We do these carefully, line by line
]

def fix_file(filepath):
    content = filepath.read_text(encoding="utf-8")
    original = content
    
    # Apply substitutions
    for pattern, replacement in SUBSTITUTIONS:
        content = re.sub(pattern, replacement, content)
    
    # Also fix lines that still have | between simple types
    # For each line containing :, check if it has remaining |
    lines = content.split('\n')
    new_lines = []
    for line in lines:
        if '|' in line and (': ' in line or '->' in line):
            # Try to fix remaining pipe patterns
            # Match: word | word where words are type-like
            # This handles cases like: BaseException | None
            fixed = re.sub(
                r'\b([A-Za-z_]\w*)\s*\|\s*([A-Za-z_]\w*)\b',
                lambda m: 'Optional[{}]'.format(m.group(1)) if m.group(2) == 'None'
                     else 'Union[{}, {}]'.format(m.group(1), m.group(2)),
                line
            )
            new_lines.append(fixed)
        else:
            new_lines.append(line)
    
    content = '\n'.join(new_lines)
    
    if content != original:
        filepath.write_text(content, encoding="utf-8")
        return True
    return False

fixed = 0
for f in FILES:
    fp = ROOT / f
    if fp.exists() and fix_file(fp):
        print("FIXED: {}".format(f))
        fixed += 1

print("\nFixed {} files.".format(fixed))
