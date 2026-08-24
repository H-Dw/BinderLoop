# Taiji Submission Pattern Report
## Binder-Harness BoltzGen Complete Flow

**Analysis Date:** 2026-05-26  
**Scope:** Understanding the complete Taiji submission pattern including package sync, config generation, and remote execution.

---

## 1. Key Constants & Path Configuration

### TAIJI_REMOTE_RUN_ROOT
**Location:** `/scripts/run_boltzgen_complete_path_test.py:37`
```python
TAIJI_REMOTE_RUN_ROOT = Path("/aceph/daweihuang/dataset/proteo_benchmark/boltzgen_harness_tests")
```
- **Purpose:** Base directory on Ceph where all Taiji project packages are synced
- **Structure:** `{TAIJI_REMOTE_RUN_ROOT}/{task_flag}/taiji_project_package/`
- **Usage:** Serves as the centralized location for all remote execution artifacts

---

## 2. Package Generation & Sync Flow

### 2.1 sync_package_to_remote_run_dir()
**Location:** `/scripts/run_boltzgen_complete_path_test.py:152-159`

**Function Signature:**
```python
def sync_package_to_remote_run_dir(package_dir: str | Path, task_flag: str) -> Path
```

**Implementation:**
```python
def sync_package_to_remote_run_dir(package_dir: str | Path, task_flag: str) -> Path:
    package_dir = Path(package_dir)
    remote_package_dir = TAIJI_REMOTE_RUN_ROOT / task_flag / "taiji_project_package"
    remote_package_dir.parent.mkdir(parents=True, exist_ok=True)
    if remote_package_dir.exists():
        shutil.rmtree(remote_package_dir)
    shutil.copytree(package_dir, remote_package_dir, symlinks=True)
    return remote_package_dir
```

**Flow:**
1. Takes locally-generated `package_dir` and `task_flag` as input
2. Creates destination path: `/aceph/daweihuang/dataset/proteo_benchmark/boltzgen_harness_tests/{task_flag}/taiji_project_package/`
3. Creates parent directories if needed
4. **Replaces existing package** if one exists (clean slate)
5. Recursively copies entire package tree, preserving symlinks
6. Returns remote package directory path

**Critical Detail:** This is **destructive** — it removes and recreates the entire remote package directory.

### 2.2 point_run_spec_to_package()
**Location:** `/scripts/run_boltzgen_complete_path_test.py:162-187`

**Function Signature:**
```python
def point_run_spec_to_package(run_spec, package_dir: Path) -> None
```

**Purpose:** Updates the `run_spec` object to point to remote Ceph paths instead of local paths, enabling:
- Remote result collection via monitoring
- Expected output validation against synced location
- Log file tracking on remote system

**Updated Fields:**
```python
run_spec.package_dir = str(package_dir)                    # Remote root
run_spec.design_spec_path = str(package_dir / "configs" / "boltzgen_design_spec.yaml")
run_spec.run_script_path = str(package_dir / "scripts" / "run_boltzgen_full.sh")
run_spec.output_dir = str(package_dir / "outputs" / "boltzgen_output")
run_spec.log_file = str(package_dir / "logs" / "boltzgen_full.log")
run_spec.expected_outputs = {...}  # All paths point to remote package_dir
```

---

## 3. Taiji Configuration Generation

### 3.1 taiji_options Dictionary
**Location:** `/scripts/run_boltzgen_complete_path_test.py:276-291`

**Key Structure:**
```python
taiji_options = {
    # Taiji infrastructure parameters
    "business_flag": "pathology_gpu_chongqing",
    "project_id": 192631,
    "image_full_name": "mirrors.tencent.com/davedwhuang/boltzgen:cu118",
    
    # GPU/Resource allocation
    "GPUName": "V100",
    "host_gpu_num": 1,
    "host_num": 1,
    "cuda_version": "11.0",
    
    # Job priority & quota
    "priority_level": "HIGH",
    "quota_type": "public",
    "location": "cq",  # Chongqing
    
    # Taiji version & dataset/model IDs (v2.0 configuration)
    "version": "v2.0",
    "dataset_id": "8b1d82389dfc7401019dfd3046540076",
    "model_id": "8b1d81e89dfc747f019dfd304ccf0080",
    
    # **CRITICAL: Remote package path**
    "remote_project_dir": str(remote_package_dir),  # Points to synced location
}
```

**The `remote_project_dir` Key:**
- **Value:** Full path to synced package on Ceph
- **Example:** `/aceph/daweihuang/dataset/proteo_benchmark/boltzgen_harness_tests/binder_boltzgen_complete_path_len50/taiji_project_package`
- **Used By:** `TaijiExecutionAgent.create_boltzgen_taiji_spec()` to build the startup command
- **Purpose:** Tells Taiji where to find and execute the project code

**Optional: CEPH_SECRET**
```python
ceph_secret = secret_store.ceph_secret()
if ceph_secret:
    taiji_options["envs"] = {"CEPH_SECRET": ceph_secret}
```
- Passed as environment variable for Ceph mounting in Taiji container

### 3.2 TaijiExecutionAgent.create_boltzgen_taiji_spec()
**Location:** `/binderloop/agents/taiji_execution_agent.py:59-149`

**How `remote_project_dir` is Handled (Lines 91-100):**

```python
# Extract remote_project_dir from taiji_options
prefix_cmd = config.pop("start_cmd_prefix", taiji_options.get("start_cmd_prefix", ""))
remote_project_dir = config.pop("remote_project_dir", taiji_options.get("remote_project_dir"))
package_dir = run_spec.package_dir or str(Path(run_spec.run_script_path).parents[1])
run_script_rel = str(Path(run_spec.run_script_path).relative_to(package_dir))

# Build the startup command that uses remote_project_dir
config["start_cmd"] = self._build_start_cmd(
    run_script_rel=run_script_rel,
    prefix_cmd=prefix_cmd,
    remote_project_dir=remote_project_dir,
)
```

**The _build_start_cmd() Method (Lines 221-249):**

The generated startup command:
1. **Ensures Ceph mount** at `/aceph/daweihuang` (using CEPH_SECRET if needed)
2. **Sets WORKSPACE** to the `remote_project_dir`
3. **Changes to workspace:** `cd "$WORKSPACE"`
4. **Fallback logic:** If script not found, tries `taiji_project_package/` subdirectory
5. **Executes script:** `bash scripts/run_boltzgen_full.sh`

```bash
set -euo pipefail
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
WORKSPACE='{remote_project_dir}'
cd "$WORKSPACE"
if [[ ! -f scripts/run_boltzgen_full.sh && -f taiji_project_package/scripts/run_boltzgen_full.sh ]]; then
  cd taiji_project_package
fi
echo "[HARNESS] workspace=$(pwd)"
bash scripts/run_boltzgen_full.sh
```

---

## 4. DesignSpecAgent Step Management

**Location:** `/binderloop/agents/design_spec_agent.py:33-44`

### Step Definitions:
```python
DEFAULT_GPU_STEPS = ["design", "inverse_folding", "folding", "design_folding"]
DEFAULT_FULL_STEPS = ["design", "inverse_folding", "folding", "design_folding", "analysis", "filtering"]
LOCAL_ANALYSIS_STEPS = ["analysis", "filtering"]
```

### Step Categories:

| Step Category | Steps | Location | Use Case |
|---|---|---|---|
| **GPU_STEPS** | design, inverse_folding, folding, design_folding | Taiji (remote) | Fast GPU-accelerated generation |
| **FULL_STEPS** | GPU_STEPS + analysis, filtering | Taiji (remote) | Full end-to-end on GPU |
| **LOCAL_ANALYSIS** | analysis, filtering | Local workstation | Post-processing after GPU generation |

### Analysis Location Determination
**Location:** `/binderloop/agents/design_spec_agent.py:252-262`

```python
def _effective_steps(self, params: Mapping[str, Any]) -> List[str]:
    analysis_location = str(params.get("analysis_location", "local")).lower()
    if analysis_location in {"taiji", "remote"} or params.get("run_analysis_on_taiji"):
        # Run all steps including analysis on Taiji (GPU)
        steps = list(params.get("steps") or self.DEFAULT_FULL_STEPS)
    else:
        # Run only GPU-intensive generation on Taiji; analysis locally
        steps = [step for step in list(params.get("steps") or self.DEFAULT_GPU_STEPS) 
                 if step not in {"analysis", "filtering"}]
        if not steps:
            steps = list(self.DEFAULT_GPU_STEPS)
    if not params.get("run_filtering", True):
        steps = [s for s in steps if s != "filtering"]
    return steps
```

**Decision Logic:**
- **`analysis_location = "local"` (default):** 
  - Taiji runs: `["design", "inverse_folding", "folding", "design_folding"]`
  - Local runs: `["analysis", "filtering"]` via `run_boltzgen_analysis_local.sh`
- **`analysis_location = "taiji"`:**
  - Taiji runs: `["design", "inverse_folding", "folding", "design_folding", "analysis", "filtering"]`
  - No local analysis needed

---

## 5. Local Analysis Script Generation

**Location:** `/binderloop/agents/design_spec_agent.py:119-140`

### Script Generation Condition:
```python
if "analysis" not in steps:  # Only if Taiji is NOT doing analysis
    local_analysis_command = self._ensure_complete_pipeline(
        self._build_packaged_command(..., params={**merged_params, "reuse": True}),
        {..., "steps": self.LOCAL_ANALYSIS_STEPS, ...},
    )
    analysis_script = script_dir / "run_boltzgen_analysis_local.sh"
    # Write local analysis script
```

### Generated Script Path:
```
{package_dir}/scripts/run_boltzgen_analysis_local.sh
```

**Script is generated only when `analysis_location="local"` (default)**

### Script Execution
**Location:** `/scripts/run_boltzgen_complete_path_test.py:328-341`

```python
local_analysis_script = Path(run_spec.package_dir or Path(run_spec.run_script_path).parents[1]) / "scripts/run_boltzgen_analysis_local.sh"
if args.run_local_analysis:
    if not local_analysis_script.exists():
        report["local_analysis"] = {"status": "missing_script", "path": str(local_analysis_script)}
    else:
        proc = subprocess.run(["bash", str(local_analysis_script)], ...)
```

---

## 6. Complete Taiji Submission Flow

### End-to-End Workflow

```
STEP 1: GENERATE LOCAL PACKAGE
├─ DesignSpecAgent.create_boltzgen_run_spec()
├─ Output structure:
│  ├─ {output_dir}/taiji_project_package/
│  ├─ inputs/                    (target structure)
│  ├─ configs/                   (design spec YAML)
│  ├─ scripts/run_boltzgen_full.sh           (executable)
│  ├─ scripts/run_boltzgen_analysis_local.sh (if local analysis)
│  ├─ logs/                      (empty initially)
│  └─ outputs/boltzgen_output/   (empty initially)
├─ Returns: BoltzGenRunSpec with LOCAL paths
└─ Confirmed generated file location found

            ↓

STEP 2: SYNC PACKAGE TO REMOTE CEPH
├─ sync_package_to_remote_run_dir()
├─ Source: {local package_dir}
├─ Destination: /aceph/daweihuang/dataset/proteo_benchmark/boltzgen_harness_tests/{task_flag}/taiji_project_package
├─ Action: shutil.copytree() with symlinks=True
├─ Cleans existing if present
└─ Returns: remote_package_dir Path

            ↓

STEP 3: UPDATE RUNSPEC TO REMOTE PATHS
├─ point_run_spec_to_package()
├─ Mutates run_spec:
│  ├─ run_spec.package_dir → {remote_package_dir}
│  ├─ run_spec.design_spec_path → {remote_package_dir}/configs/...
│  ├─ run_spec.run_script_path → {remote_package_dir}/scripts/...
│  ├─ run_spec.output_dir → {remote_package_dir}/outputs/boltzgen_output
│  └─ run_spec.log_file → {remote_package_dir}/logs/boltzgen_full.log
└─ Updates expected_outputs: all point to remote_package_dir

            ↓

STEP 4: BUILD TAIJI OPTIONS
├─ Populate taiji_options dictionary
├─ **CRITICAL**: remote_project_dir = str(remote_package_dir)
└─ Optional: envs.CEPH_SECRET

            ↓

STEP 5: CREATE TAIJI SUBMIT SPEC
├─ TaijiExecutionAgent.create_boltzgen_taiji_spec()
├─ Extracts remote_project_dir from taiji_options
├─ Builds start_cmd via _build_start_cmd():
│  ├─ WORKSPACE = {remote_project_dir}
│  ├─ Mounts /aceph/daweihuang (CEPH_SECRET)
│  ├─ cd "$WORKSPACE"
│  └─ bash scripts/run_boltzgen_full.sh
├─ Generates: taiji_simple_config.json
└─ Returns: TaijiSubmitSpec

            ↓

STEP 6: SUBMIT TO TAIJI
├─ TaijiExecutionAgent.submit()
├─ Executes: taiji_client start -scfg <config_path>
├─ Parses output for taiji_job_id
└─ Returns: TaijiSubmissionRecord

            ↓

STEP 7: MONITOR REMOTE EXECUTION
├─ RunMonitorAgent.check_once()
├─ Checks remote_package_dir files
└─ Validates expected_outputs

            ↓

STEP 8: OPTIONAL LOCAL ANALYSIS
├─ bash run_boltzgen_analysis_local.sh
├─ Runs locally if analysis_location="local"
└─ Produces final_ranked_designs/, *.csv

            ↓

STEP 9: RESULT INGESTION
├─ ResultIngestionAgent.ingest_boltzgen_output()
├─ Reads from run_spec.output_dir (remote)
└─ Collects candidates

            ↓

STEP 10: EVALUATION
├─ EvaluationAgent.evaluate_candidates()
└─ Scores and ranks designs

            ↓

STEP 11: ACTIVE LEARNING PROPOSAL
├─ ActiveLearningPolicyAgent.propose_next_boltzgen_params()
└─ Generates next-round parameters
```

---

## 7. Critical Data Flow

**Path Transformation:**

```
LOCAL GENERATION
├─ package_dir = /tmp/round0_len50_seed0/taiji_project_package
├─ run_spec.package_dir = /tmp/round0_len50_seed0/taiji_project_package
└─ run_spec.run_script_path = /tmp/round0_len50_seed0/taiji_project_package/scripts/run_boltzgen_full.sh

         ↓ sync_package_to_remote_run_dir() ↓

REMOTE SYNC
├─ remote_package_dir = /aceph/daweihuang/dataset/proteo_benchmark/boltzgen_harness_tests/{task_flag}/taiji_project_package
└─ Entire tree copied to remote

         ↓ point_run_spec_to_package() ↓

REMOTE PATHS
├─ run_spec.package_dir = /aceph/daweihuang/dataset/proteo_benchmark/boltzgen_harness_tests/{task_flag}/taiji_project_package
└─ run_spec.run_script_path = /aceph/daweihuang/dataset/proteo_benchmark/boltzgen_harness_tests/{task_flag}/taiji_project_package/scripts/run_boltzgen_full.sh

         ↓ create_boltzgen_taiji_spec() ↓

TAIJI CONFIG
├─ remote_project_dir = (same as above)
├─ start_cmd = "bash -lc 'cd {remote_project_dir} && bash scripts/run_boltzgen_full.sh'"
└─ taiji_simple_config.json written

         ↓ Taiji Execution ↓

REMOTE EXECUTION
├─ WORKSPACE = {remote_project_dir}
├─ cd $WORKSPACE
├─ bash scripts/run_boltzgen_full.sh
└─ Results written to: outputs/boltzgen_output/
```

---

## 8. Analysis Location Strategy

### Strategy 1: GPU Generation + Local Analysis (Default)
```
Taiji (Remote)
├─ Steps: ["design", "inverse_folding", "folding", "design_folding"]
├─ Output: intermediate_designs/, intermediate_designs_inverse_folded/
└─ Task Completes

     ↓ (Async, user triggered)

Workstation (Local)
├─ Script: scripts/run_boltzgen_analysis_local.sh
├─ Steps: ["analysis", "filtering"]
├─ Output: final_ranked_designs/, *.csv metrics
```

**Advantages:**
- Taiji resources freed immediately after GPU generation
- Local analysis can be retried without re-running GPU expensive steps
- Developers can analyze outputs incrementally

### Strategy 2: Full Pipeline on GPU
```
Taiji (Remote)
├─ Steps: ["design", "inverse_folding", "folding", "design_folding", "analysis", "filtering"]
├─ Output: final_ranked_designs/, *.csv metrics
```

**Advantages:**
- Single batch job simplifies orchestration
- All computation on GPU resources
- No local hardware needed

---

## 9. Key Data Structures

### BoltzGenRunSpec
After `point_run_spec_to_package()`, points to remote:
```python
package_dir: "/aceph/.../taiji_project_package"
design_spec_path: "/aceph/.../taiji_project_package/configs/boltzgen_design_spec.yaml"
run_script_path: "/aceph/.../taiji_project_package/scripts/run_boltzgen_full.sh"
output_dir: "/aceph/.../taiji_project_package/outputs/boltzgen_output"
log_file: "/aceph/.../taiji_project_package/logs/boltzgen_full.log"
expected_outputs: {all paths point to remote package_dir}
```

### taiji_options
```python
{
    "business_flag": "pathology_gpu_chongqing",
    "project_id": 192631,
    "version": "v2.0",
    "remote_project_dir": "/aceph/.../taiji_project_package",  # CRITICAL
    ...other parameters...
}
```

### start_cmd Flow
```
Input: remote_project_dir, run_script_rel
Output: bash -lc '
  # Mount Ceph if needed
  WORKSPACE={remote_project_dir}
  cd "$WORKSPACE"
  bash {run_script_rel}
'
```

---

## 10. Summary

The Taiji submission pattern implements a **three-phase workflow**:

1. **Generation Phase** (Local):
   - Create project package with scripts, configs, inputs
   - Generate `BoltzGenRunSpec` with local paths
   - Generate both GPU script and optional local analysis script

2. **Sync & Configure Phase** (Local → Ceph):
   - `sync_package_to_remote_run_dir()` copies entire package to Ceph
   - `point_run_spec_to_package()` updates all paths to remote locations
   - Build `taiji_options` with `remote_project_dir` pointing to synced location
   - `TaijiExecutionAgent` builds Taiji config with startup command using `remote_project_dir`

3. **Remote Execution Phase** (Ceph/Taiji):
   - Taiji mounts Ceph, sets `WORKSPACE` to `remote_project_dir`
   - Executes `scripts/run_boltzgen_full.sh` from package root
   - BoltzGen uses relative paths for portability
   - Optional: local post-processing via `run_boltzgen_analysis_local.sh`

**Key Design Principles:**
- **Reproducibility:** All artifacts (specs, scripts, inputs) packaged together
- **Portability:** Relative paths work identically local → Taiji
- **Modularity:** Analysis can be deferred to local execution
- **Transparency:** All generated configs written as JSON/YAML for inspection
