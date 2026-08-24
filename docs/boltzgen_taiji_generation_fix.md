# BoltzGen 脚本与 Taiji Config 生成修复说明

## 问题分析

Taiji 日志中的 `exit code: 1` 发生在 BoltzGen 命令启动后：

```text
[HARNESS] command=boltzgen run /aceph/.../boltzgen_design_spec.yaml ...
[TAIJI] user exit code: 1
```

原生成逻辑存在几个风险点：

1. **目标文件路径是硬编码远端路径**

```text
/aceph/daweihuang/dataset/proteo_benchmark/target_region_cif_boltzgen/IL-17A.cif
```

该路径不属于当前 Harness 项目包，Taiji 容器中是否存在不可审计；一旦不存在，BoltzGen 会直接失败。

2. **脚本通过 start_cmd 内嵌 heredoc 写 spec**

多层 JSON → shell → heredoc → YAML 引号转义复杂，日志中出现大量：

```text
<<'"'"'YAML'"'"'
'"'"''"'"'
```

虽然 shell 可能能解析，但可维护性和错误定位都差。

3. **输入、spec、输出不在同一个项目目录**

旧逻辑将 spec/output 写到 `/aceph/.../boltzgen_harness_tests/...`，target 又指向另一处 `/aceph/.../target_region...`，不利于 Harness 做 provenance 和 result ingestion。

4. **失败时日志不一定写 end status**

原脚本使用 `set -e`，BoltzGen 失败会直接退出，未必写入统一的：

```text
[HARNESS] end_time=... status=...
```

## 修复方案

### 1. 改为项目包模式

`DesignSpecAgent` 现在会生成完整项目包：

```text
project_package/
  inputs/
    IL-17A.cif
  configs/
    boltzgen_design_spec.yaml
    boltzgen_parameter_plan.yaml
  scripts/
    run_boltzgen_full.sh
  outputs/
    boltzgen_output/
  logs/
    boltzgen_full.log
  boltzgen_run_manifest.json
```

也就是说：

- 输入 target 在当前项目包中。
- 生成 spec 在当前项目包中。
- 输出目录在当前项目包中。
- 日志在当前项目包中。

### 2. YAML spec 不再通过 heredoc 写入

spec 由本地 Python 在提交前写入：

```text
configs/boltzgen_design_spec.yaml
```

Taiji 运行时只执行：

```bash
bash scripts/run_boltzgen_full.sh
```

不再在 `start_cmd` 中内联大段 YAML。

### 3. BoltzGen 命令使用相对路径

脚本会进入项目包根目录：

```bash
cd "$(dirname "$0")/.."
```

然后执行：

```bash
boltzgen run configs/boltzgen_design_spec.yaml \
  --output outputs/boltzgen_output \
  --steps design inverse_folding folding design_folding analysis filtering
```

### 4. target 缺失会提前报 Harness 级错误

脚本启动时检查：

```bash
if [[ ! -s "$TARGET_FILE" ]]; then
  echo "[HARNESS][ERROR] target file missing or empty: $TARGET_FILE"
  exit 11
fi
```

这样如果 target 没被打包，不会等到 BoltzGen 内部才报模糊错误。

### 5. 失败也写统一 end status

BoltzGen 命令前临时关闭 `set -e`：

```bash
set +e
boltzgen run ...
status=$?
set -e
echo "[HARNESS] end_time=$(date -Is) status=$status"
exit "$status"
```

这样监控模块可以通过日志直接判断 exit status。

## Taiji Config 生成修复

`TaijiExecutionAgent` 现在：

1. 将 `model_local_file_path` 默认设置为 `run_spec.package_dir`。
2. `start_cmd` 改为：

```bash
cd ${JIZHI_WORKSPACE_PATH:-.} && bash scripts/run_boltzgen_full.sh
```

3. 对 Taiji v2：
   - 若提供 `dataset_id + model_id`，使用：

```json
"code_path": "/",
"dataset_path": "/"
```

   - 否则保留 `model_local_file_path` 上传项目包。
   - 移除 TensorBoard 相关字段，避免触发存储配额兼容问题。

## 监控模块修复

`RunMonitorAgent` 新增：

```json
"needs_followup": true,
"recommended_followup_seconds": 300
```

当状态为：

```text
PENDING / QUEUED / WAITING / RUNNING / UNKNOWN
```

时，监控结果会提示需要后续检查，而不是只读一次就结束。

## 验证

已执行：

```bash
python3 -m py_compile binderloop/agents/design_spec_agent.py \
  binderloop/agents/taiji_execution_agent.py \
  binderloop/agents/run_monitor_agent.py \
  scripts/run_boltzgen_complete_path_test.py

python3 scripts/test_boltzgen_taiji_agents.py
python3 scripts/run_boltzgen_complete_path_test.py
```

生成的脚本示例：

```text
outputs/boltzgen_complete_path_test/round0_len50_seed0/project_package/scripts/run_boltzgen_full.sh
```

关键路径均在项目包内：

```text
inputs/IL-17A.cif
configs/boltzgen_design_spec.yaml
outputs/boltzgen_output
logs/boltzgen_full.log
```
