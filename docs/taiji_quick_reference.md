# Taiji Submission Quick Reference
## Key Functions & Paths at a Glance

### 1. LOCAL GENERATION
```python
# File: binderloop/agents/design_spec_agent.py
agent = DesignSpecAgent(boltzgen_root)
run_spec = agent.create_boltzgen_run_spec(job, params=params)

# Output structure created:
# {output_dir}/project_package/
# ├─ inputs/IL-17A.cif
# ├─ configs/boltzgen_design_spec.yaml
# ├─ configs/boltzgen_parameter_plan.yaml
# ├─ scripts/run_boltzgen_full.sh
# ├─ scripts/run_boltzgen_analysis_local.sh (if needed)
# ├─ logs/
# └─ outputs/boltzgen_output/

# run_spec now has LOCAL paths
# run_spec.package_dir = "/tmp/round0_len50_seed0/project_package"
# run_spec.run_script_path = ".../scripts/run_boltzgen_full.sh"
```

### 2. SYNC TO CEPH
```python
# File: scripts/run_boltzgen_complete_path_test.py (lines 152-159)
from pathlib import Path
import shutil

TAIJI_REMOTE_RUN_ROOT = Path("/aceph/daweihuang/dataset/proteo_benchmark/boltzgen_harness_tests")

def sync_package_to_remote_run_dir(package_dir: str | Path, task_flag: str) -> Path:
    package_dir = Path(package_dir)
    remote_package_dir = TAIJI_REMOTE_RUN_ROOT / task_flag / "project_package"
    remote_package_dir.parent.mkdir(parents=True, exist_ok=True)
    if remote_package_dir.exists():
        shutil.rmtree(remote_package_dir)  # Destructive!
    shutil.copytree(package_dir, remote_package_dir, symlinks=True)
    return remote_package_dir

# Example result:
# remote_package_dir = Path("/aceph/.../boltzgen_harness_tests/binder_boltzgen_complete_path_len50/project_package")
```

### 3. UPDATE RUNSPEC TO REMOTE PATHS
```python
# File: scripts/run_boltzgen_complete_path_test.py (lines 162-187)
def point_run_spec_to_package(run_spec, package_dir: Path) -> None:
    # Mutates run_spec in-place to point to remote paths
    run_spec.package_dir = str(package_dir)
    run_spec.design_spec_path = str(package_dir / "configs" / "boltzgen_design_spec.yaml")
    run_spec.run_script_path = str(package_dir / "scripts" / "run_boltzgen_full.sh")
    run_spec.output_dir = str(package_dir / "outputs" / "boltzgen_output")
    run_spec.log_file = str(package_dir / "logs" / "boltzgen_full.log")
    run_spec.expected_outputs = {
        "package_dir": str(package_dir),
        "target_file": str(package_dir / "inputs" / "IL-17A.cif"),
        "boltzgen_output_dir": str(run_spec.output_dir),
        "steps_manifest": str(run_spec.output_dir / "steps.yaml"),
        "log_file": str(run_spec.log_file),
    }

# After this, all paths in run_spec point to Ceph
# run_spec.package_dir = "/aceph/.../project_package"
```

### 4. BUILD TAIJI OPTIONS
```python
# File: scripts/run_boltzgen_complete_path_test.py (lines 276-291)
taiji_options = {
    "business_flag": "pathology_gpu_chongqing",
    "project_id": 192631,
    "image_full_name": "mirrors.tencent.com/davedwhuang/boltzgen:cu118",
    "GPUName": "V100",
    "host_gpu_num": 1,
    "host_num": 1,
    "cuda_version": "11.0",
    "priority_level": "HIGH",
    "quota_type": "public",
    "location": "cq",
    "version": "v2.0",
    "dataset_id": "8b1d82389dfc7401019dfd3046540076",
    "model_id": "8b1d81e89dfc747f019dfd304ccf0080",
    "remote_project_dir": str(remote_package_dir),  # ⭐ CRITICAL!
}

# Optional: Add CEPH_SECRET
ceph_secret = secret_store.ceph_secret()
if ceph_secret:
    taiji_options["envs"] = {"CEPH_SECRET": ceph_secret}
```

### 5. CREATE TAIJI SUBMIT SPEC
```python
# File: binderloop/agents/taiji_execution_agent.py (lines 59-149)
from binderloop.agents import TaijiExecutionAgent

agent = TaijiExecutionAgent(dry_run=False)
submit_spec = agent.create_boltzgen_taiji_spec(
    run_spec,
    template_json=template,
    output_json=out_dir / "02_taiji_simple_config.json",
    task_flag=task_flag,
    taiji_options=taiji_options,  # Contains remote_project_dir
)

# Returns: TaijiSubmitSpec
# submit_spec.simple_config_path
# submit_spec.submit_command = "taiji_client start -scfg <path>"
# submit_spec.simple_config (dict with start_cmd embedded)
```

### 6. SUBMIT TO TAIJI
```python
# File: binderloop/agents/taiji_execution_agent.py (lines 151-186)
agent = TaijiExecutionAgent(dry_run=False)
submission = agent.submit(submit_spec)

# Returns: TaijiSubmissionRecord
# submission.taiji_job_id  ← Instance ID for monitoring
# submission.returncode    ← 0 for success
# submission.stdout / stderr ← Taiji output
```

### 7. MONITOR EXECUTION
```python
# File: binderloop/agents/run_monitor_agent.py
from binderloop.agents import RunMonitorAgent

monitor = RunMonitorAgent()
snapshot = monitor.check_once(
    task_flag=submit_spec.task_flag,
    instance_id=submission.taiji_job_id,
    expected_outputs=run_spec.expected_outputs,  # Now point to remote paths!
    simple_config_path=submit_spec.simple_config_path,
    config_path=submit_spec.full_config_path,
)

# Checks file existence at remote paths (Ceph)
# snapshot.state = "running" | "completed" | "failed"
# snapshot.found_files / missing_files
```

---

## Step Selection Logic

### DEFAULT_GPU_STEPS
```python
["design", "inverse_folding", "folding", "design_folding"]
```
- **Used when:** `analysis_location="local"` (default)
- **Location:** Taiji (GPU, fast: minutes)
- **Output:** `intermediate_designs/`, `intermediate_designs_inverse_folded/`

### DEFAULT_FULL_STEPS
```python
["design", "inverse_folding", "folding", "design_folding", "analysis", "filtering"]
```
- **Used when:** `analysis_location="taiji"` 
- **Location:** Taiji (GPU, slow: hours)
- **Output:** `final_ranked_designs/`, `*.csv` metrics

### LOCAL_ANALYSIS_STEPS
```python
["analysis", "filtering"]
```
- **Used when:** `analysis_location="local"` (after GPU generation completes)
- **Location:** Local workstation (CPU, fast: minutes)
- **Input:** `intermediate_designs/` from Taiji
- **Output:** `final_ranked_designs/`, `*.csv` metrics
- **Script:** `scripts/run_boltzgen_analysis_local.sh`

---

## Path Transformation Timeline

```
LOCAL GENERATION                  REMOTE SYNC              REMOTE EXECUTION
─────────────────                 ───────────              ────────────────

package_dir                        ──copytree──>            [Taiji container]
/tmp/round0_len50_seed0/          
  project_package/                                   Mount /aceph
                                                           
run_spec.package_dir             Updated by              WORKSPACE =
  = /tmp/...                      point_run_spec_to_      /aceph/.../project_package
                                  package()
                                                           cd $WORKSPACE
run_spec.run_script_path                                   bash scripts/run_boltzgen_full.sh
  = /tmp/.../scripts/...         ──mutate──>              
                                                           ✓ All relative paths work!
                                  = /aceph/.../scripts/    ✓ inputs/, configs/, outputs/
                                                           ✓ reproducible execution
```

---

## Configuration Files Generated

| File | Location | Purpose |
|------|----------|---------|
| `boltzgen_design_spec.yaml` | `configs/` | BoltzGen specification |
| `boltzgen_parameter_plan.yaml` | `configs/` | All parameters used |
| `run_boltzgen_full.sh` | `scripts/` | GPU execution script (executable) |
| `run_boltzgen_analysis_local.sh` | `scripts/` | Local analysis script (executable, conditional) |
| `taiji_simple_config.json` | `scripts/` | Taiji submission config (after create_boltzgen_taiji_spec) |
| `taiji_simple_config.redacted.json` | `scripts/` | Same, but with secrets masked |
| `taiji_submit_manifest.json` | `scripts/` | Submission metadata |
| `boltzgen_run_manifest.json` | `.` | BoltzGenRunSpec serialized |

---

## Critical Variables

### TAIJI_REMOTE_RUN_ROOT
```python
Path("/aceph/daweihuang/dataset/proteo_benchmark/boltzgen_harness_tests")
```
- Base directory for all Taiji packages on Ceph
- Final path: `{TAIJI_REMOTE_RUN_ROOT}/{task_flag}/project_package/`

### remote_project_dir
- **Type:** `str | Path`
- **Value:** Full path to synced package on Ceph
- **Example:** `/aceph/daweihuang/dataset/proteo_benchmark/boltzgen_harness_tests/binder_boltzgen_complete_path_len50/project_package`
- **Usage:** Passed in `taiji_options["remote_project_dir"]`
- **Critical:** Used by `_build_start_cmd()` to set WORKSPACE in Taiji container

### task_flag
- **Type:** `str`
- **Format:** `"binder_boltzgen_complete_path_len50_<timestamp>"` (if submitting)
- **Purpose:** Unique identifier for this task
- **Usage:** Determines path structure on Ceph

---

## Taiji Startup Command (Generated)

The `start_cmd` is embedded in `taiji_simple_config.json`:

```bash
bash -lc 'set -euo pipefail
mkdir -p /aceph/daweihuang
if ! mountpoint -q /aceph/daweihuang 2>/dev/null; then
  if [[ -n "${CEPH_SECRET:-}" ]]; then
    mount -t ceph 11.18.83.17:6789,11.18.83.31:6789,11.18.83.32:6789:/fandiwu/buddy1/daweihuang \
      /aceph/daweihuang -o name=fandiwubuddy1,secret="${CEPH_SECRET}"
  else
    echo "[HARNESS][ERROR] /aceph/daweihuang is not mounted and CEPH_SECRET is not set" >&2
    exit 16
  fi
fi
WORKSPACE='\''/aceph/daweihuang/dataset/proteo_benchmark/boltzgen_harness_tests/{task_flag}/project_package'\''
cd "$WORKSPACE"
if [[ ! -f scripts/run_boltzgen_full.sh ]]; then
  if [[ -f project_package/scripts/run_boltzgen_full.sh ]]; then
    cd project_package
  elif [[ -f taiji_project_package/scripts/run_boltzgen_full.sh ]]; then
    cd taiji_project_package
  fi
fi
echo "[HARNESS] workspace=$(pwd)"
bash scripts/run_boltzgen_full.sh
'
```

**Key points:**
1. Mounts Ceph using CEPH_SECRET
2. Sets WORKSPACE = remote_project_dir
3. Has fallback logic for nested `project_package/` then legacy `taiji_project_package/`
4. Executes run_boltzgen_full.sh from package root (all relative paths work)

---

## Error Exit Codes (from run_boltzgen_full.sh)

| Code | Meaning |
|------|---------|
| 11 | Target file missing or empty |
| 12 | Checkpoint missing |
| 13 | Cache directory missing |
| 14 | Failed to mount /aceph/daweihuang with CEPH_SECRET |
| 15 | Moldir missing |
| 16 | Ceph not mounted and CEPH_SECRET not provided |
| 127 | BoltzGen CLI not found |

---

## Common Issues & Debugging

### Issue: "CEPH_SECRET is not set"
**Solution:** Ensure `secret_store.ceph_secret()` returns valid secret, passed to `taiji_options["envs"]`

### Issue: "remote_project_dir not in start_cmd"
**Solution:** Check that `taiji_options["remote_project_dir"]` is set before calling `create_boltzgen_taiji_spec()`

### Issue: "Script not found" in Taiji
**Solution:** Verify package was synced with `sync_package_to_remote_run_dir()` before submission

### Issue: "Output files not found" in monitor
**Solution:** Check that `point_run_spec_to_package()` was called to update `run_spec.expected_outputs` to remote paths

---

## One-Liner Examples

```python
# Entire flow:
remote_pkg = sync_package_to_remote_run_dir(run_spec.package_dir, task_flag)
point_run_spec_to_package(run_spec, remote_pkg)
taiji_opts = {..., "remote_project_dir": str(remote_pkg)}
submit_spec = TaijiExecutionAgent().create_boltzgen_taiji_spec(run_spec, taiji_options=taiji_opts)
submission = TaijiExecutionAgent(dry_run=False).submit(submit_spec)
```

