# Taiji 提交字段修改指南

本文说明在当前 `binderloop` 代码中，如何让 agent 提交到 Taiji 平台的任务使用不同的 `business_flag`、`task_flag`、`GPUName` 和 `host_gpu_num`。

## 当前代码路径

标准闭环入口是：

```bash
python scripts/run_closed_loop_orchestrator.py \
  --config configs/<your_task>.yaml \
  --backend taiji
```

Taiji 提交字段的数据流如下：

```text
configs/<task>.yaml
  -> binderloop.config.ResourceSpec
  -> ResourceSpec.to_taiji_options()
  -> scripts/run_closed_loop_orchestrator.py::_build_taiji_executor()
  -> TaijiExecutionAgent.create_boltzgen_taiji_spec()
  -> <job.output_dir>/taiji_simple_config.json
  -> taiji_client start -scfg <taiji_simple_config.json>
```

关键代码位置：

- `binderloop/config.py`：定义 `ResourceSpec`，并把 YAML 的 `resource` 段转成 `taiji_options`。
- `scripts/run_closed_loop_orchestrator.py`：构造闭环 Taiji executor，生成 `task_flag`，合并远程 package 路径和 secret。
- `binderloop/agents/taiji_execution_agent.py`：合并模板 JSON、`taiji_options` 和运行参数，写出 Taiji simple config，并执行 `taiji_client start`。
- `examples/bg_example/benchmark_v100.template.json`：示例 Taiji simple-config 模板，可提供 `business_flag`、`project_id`、镜像、quota 等默认字段。

## 推荐修改方式：改任务 YAML

对于闭环 orchestrator，优先在任务 YAML 的 `resource` 段修改资源和业务字段。示例：

```yaml
task_name: binder_my_target

resource:
  backend: taiji
  host_num: 1
  host_gpu_num: 4
  gpu_name: H800
  max_parallel_jobs: 1
  template_json: examples/bg_example/benchmark_v100.template.json
  image_full_name: mirrors.tencent.com/davedwhuang/boltzgen:cu118
  timeout_seconds: 7200
  taiji_options:
    business_flag: your_business_flag
    project_id: 192631
    location: cq
    quota_type: public
    priority_level: HIGH
```

字段含义和落点：

- `resource.host_gpu_num` 会写入 Taiji 的 `host_gpu_num`，同时也会进入 BoltzGen 参数里的 `devices`，用于多 GPU sharding。
- `resource.gpu_name` 会经 `ResourceSpec.to_taiji_options()` 映射为 Taiji simple config 的 `GPUName`。
- `resource.taiji_options.business_flag` 会覆盖模板 JSON 中的同名字段，并写入最终的 `taiji_simple_config.json`。
- `resource.image_full_name` 会写入 `image_full_name`；也可以放在 `taiji_options.image_full_name`，但推荐用顶层字段保持一致。

注意：如果只在 `resource.taiji_options` 里写 `host_gpu_num` 或 `GPUName`，会影响 Taiji simple config，但闭环内部生成 BoltzGen run spec 时仍主要读取顶层 `resource.host_gpu_num` 和 `resource.gpu_name`。为了避免 Taiji 申请资源和 BoltzGen `--devices` 不一致，建议把 GPU 数量和型号写在顶层：

```yaml
resource:
  host_gpu_num: 8
  gpu_name: V100
```

## `task_flag` 如何修改

闭环脚本不会直接使用模板 JSON 或 `taiji_options` 里的 `task_flag`。当前代码会在 `scripts/run_closed_loop_orchestrator.py` 中调用 `_task_flag()` 动态生成：

```text
<prefix>_<job_id_sha1前10位>_try<attempt>_<unix_timestamp>
```

prefix 的来源是：

```text
--taiji-task-prefix 参数优先；否则使用 YAML 的 task_name
```

因此推荐两种方式：

```yaml
# 方式 1：在 YAML 中修改 task_name，作为默认 task_flag 前缀
task_name: binder_my_target_h800
```

```bash
# 方式 2：运行时覆盖前缀
python scripts/run_closed_loop_orchestrator.py \
  --config configs/<your_task>.yaml \
  --backend taiji \
  --taiji-task-prefix binder_my_target_h800
```

如果你需要完全固定的 `task_flag`，而不是带 hash、attempt 和时间戳的动态值，当前闭环入口没有提供 YAML 字段。需要改 `scripts/run_closed_loop_orchestrator.py`，例如新增一个 CLI 参数或从 `resource.taiji_options.task_flag` 读取，并在调用 `create_boltzgen_taiji_spec()` 时传入该值。改动点是 `_build_taiji_executor()` 中这行逻辑：

```python
task_flag = _task_flag(args.taiji_task_prefix or cfg.task_name, job.job_id, attempt)
```

但不建议闭环批量任务使用固定 `task_flag`，因为多轮、多 attempt 或恢复重跑时容易覆盖远程 package 目录：

```text
<taiji_remote_run_root>/<task_flag>/project_package
```

## 模板 JSON 何时需要改

`template_json` 适合保存平台侧稳定默认值，例如：

```json
{
  "business_flag": "pathology_gpu_chongqing",
  "project_id": 192631,
  "host_num": 1,
  "host_gpu_num": 8,
  "GPUName": "V100",
  "location": "cq",
  "image_full_name": "mirrors.tencent.com/davedwhuang/boltzgen:cu118",
  "quota_type": "public"
}
```

不过在 `TaijiExecutionAgent.create_boltzgen_taiji_spec()` 中，合并顺序是：

```text
template_json
  -> taiji_options 覆盖模板同名字段
  -> 显式 task_flag 参数最终覆盖 task_flag
  -> 缺省值补齐 host_num、host_gpu_num、GPUName 等
```

所以推荐做法是：

- 多个任务共享同一套 Taiji 平台默认参数时，改或新增一个模板 JSON。
- 单个实验临时换 `business_flag`、GPU 型号或 GPU 数量时，优先改 YAML 的 `resource` 段。
- 不要把真实 `Token` 或 Ceph secret 提交到模板 JSON；占位符形如 `<SET_...>` 的值会被 `TaijiExecutionAgent` 丢弃，secret 应通过本地 ignored config 或环境变量注入。

## 直接调用 `TaijiExecutionAgent` 的场景

脚本或测试如果不走闭环 YAML，而是直接调用 `TaijiExecutionAgent.create_boltzgen_taiji_spec()`，就需要在调用处传入 `task_flag` 和 `taiji_options`：

```python
submit_spec = TaijiExecutionAgent(dry_run=True).create_boltzgen_taiji_spec(
    run_spec,
    template_json="examples/bg_example/benchmark_v100.template.json",
    output_json=out_dir / "taiji_simple_config.json",
    task_flag="binder_my_fixed_or_generated_task_flag",
    taiji_options={
        "business_flag": "your_business_flag",
        "project_id": 192631,
        "GPUName": "H800",
        "host_gpu_num": 4,
        "host_num": 1,
        "image_full_name": "mirrors.tencent.com/davedwhuang/boltzgen:cu118",
        "location": "cq",
    },
)
```

相关示例脚本中也存在硬编码 `taiji_options` 的路径，例如 `scripts/run_boltzgen_complete_path_test.py` 和部分历史 `run_il17a_*` 脚本。如果运行这些脚本，需要同步修改对应脚本里的 `taiji_options`，而不是只改闭环 YAML。

## 修改后如何确认

先 dry-run 生成配置，不提交：

```bash
python scripts/run_closed_loop_orchestrator.py \
  --config configs/<your_task>.yaml \
  --backend taiji \
  --taiji-task-prefix binder_my_target_h800
```

检查输出目录中的文件：

```text
<out>/round_*/<job>/taiji_simple_config.json
<out>/round_*/<job>/taiji_simple_config.redacted.json
<out>/round_*/<job>/taiji_submit_manifest.json
```

确认 `taiji_simple_config.json` 里包含期望值：

```json
{
  "business_flag": "your_business_flag",
  "task_flag": "binder_my_target_h800_<hash>_try1_<timestamp>",
  "GPUName": "H800",
  "host_gpu_num": 4
}
```

真实提交时再加 `--submit`：

```bash
python scripts/run_closed_loop_orchestrator.py \
  --config configs/<your_task>.yaml \
  --backend taiji \
  --taiji-task-prefix binder_my_target_h800 \
  --submit
```

如果 `host_gpu_num > 1`，当前 orchestrator 会把本地并行提交数限制为 1，因为单个 BoltzGen job 会在一个 Taiji 任务内使用所有请求的 GPU。这样可以避免同一轮同时提交多个 8-GPU 任务导致资源被重复预留。
