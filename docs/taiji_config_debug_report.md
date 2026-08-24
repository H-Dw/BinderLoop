# Taiji Config 调试记录

## 1. 目标

用户要求：

- 调试 Taiji config 代码。
- 将 `model_local_file_path` 调整为当前服务器中存储执行命令脚本的文件夹位置。
- 允许多轮 Taiji 任务提交测试。
- 但同一时间上一轮返回结果出来之前仅可提交 1 个任务。

本次调试严格按顺序提交；每轮 `taiji_client start` 返回后才进入下一轮，没有并发提交。

输出目录：

```text
outputs/boltzgen_complete_path_test_real_taiji/
```

## 2. 关键修复

### 2.1 修复 `model_local_file_path`

创建当前服务器可见目录：

```text
outputs/boltzgen_complete_path_test_real_taiji/model_package_current_server/
```

并在 Taiji simple config 中设置：

```json
"model_local_file_path": "/projects/design_harness/BinderLoop/outputs/boltzgen_complete_path_test_real_taiji/model_package_current_server"
```

该轮结果：

```text
create dataset success
create model success
```

说明 `model_local_file_path` 已从“本地不可见路径”修复为“当前服务器可见路径”。

### 2.2 发现 TensorBoard 存储字段问题

修复 model path 后，Taiji client 进入下一阶段，但报：

```text
构造 TensorBoard 存储配额信息失败，请检查 tensorboard_business_flag 和 tensorboard_container_path
```

尝试添加：

- `tensorboard_business_flag`
- `tensorboard_container_path`
- `tensorboard_custom_path`

仍失败。判断该问题来自旧版 simple config 与当前 Taiji v2 客户端的字段兼容性。

### 2.3 查询 Taiji 官方 simple config 文档

通过 iWiki 文档 `docid=456384856` 查到：

- 新版 simple config 推荐：`"version": "v2.0"`
- v2 中训练代码使用：`code_path`
- 数据可使用：`dataset_id` / `dataset_path`
- 已有模型可使用：`model_id`

### 2.4 切换为 v2 simple config

最终可提交的配置使用：

```json
"version": "v2.0"
"code_path": "/"
"dataset_path": "/"
"dataset_id": "8b1d82389dfc7401019dfd3046540076"
"model_id": "8b1d81e89dfc747f019dfd304ccf0080"
```

其中：

- `dataset_id` 和 `model_id` 来自已有任务 `binder_benchmark_boltzgen_test_v100` 的 `task_detail`。
- `code_path=/` 表示使用镜像内路径；实际逻辑由 `start_cmd` 自包含脚本完成。
- 这样绕开旧版 `model_local_file_path` 上传链路和 TensorBoard 存储字段校验问题。

## 3. 成功提交的任务

成功提交记录：

```text
outputs/boltzgen_complete_path_test_real_taiji/taiji_start_record_v2_ids.json
```

任务信息：

```text
task_flag: binder_boltzgen_complete_path_len50_v2ids_1779707913
instance_id: 8b1d81969e5e4ef3019e5edba94d019c
```

返回中包含：

```text
[info][start success]
```

说明任务已成功创建实例。

## 4. 当前运行状态

监控文件：

```text
outputs/boltzgen_complete_path_test_real_taiji/taiji_monitor_v2_ids_correct_instance.json
```

当前状态：

```text
state: PENDING
```

说明任务已经提交成功，目前在等待资源/调度，还未进入 BoltzGen 实际运行日志阶段。

## 5. 已修复 Agent 代码

修改：

```text
binderloop/agents/taiji_execution_agent.py
```

主要修复：

1. `TaijiExecutionAgent` 支持 v2 simple config：
   - `version=v2.0` 时优先设置 `code_path` / `dataset_path`。
   - 避免误走旧版 `model_local_file_path` 上传链路。
   - 移除 TensorBoard 相关字段，避免触发存储配额校验。

2. 修复 instance id 解析顺序：
   - 优先解析 `instance_id`。
   - 再解析 `job_id`。
   - 最后才解析 `task_id`。

之前误把 `task_id` 类字段当成 instance id，导致监控指向错误实例；已修复。

## 6. 后续监控

已安排一次 10 分钟后的 OpenClaw cron 检查，不会提交新任务，只做：

```text
taiji_client instance_detail
taiji_client logs --tail 200
```

并将结果写入：

```text
outputs/boltzgen_complete_path_test_real_taiji/taiji_monitor_followup.json
```

## 7. 推荐后续配置策略

对于当前 Taiji v2 客户端，建议 Harness 的默认提交策略改为：

### 优先模式：复用 dataset_id + model_id

```json
"version": "v2.0",
"dataset_id": "...",
"model_id": "...",
"code_path": "/",
"dataset_path": "/"
```

优点：

- 避免本地路径上传失败。
- 避免 TensorBoard 存储配额字段兼容问题。
- 更适合当前服务器已有 benchmark 任务环境。

### 备选模式：model_local_file_path

仅当本地目录确实存在，且 Taiji client 的上传/存储字段配置完整时使用。

提交前必须 preflight：

```text
Path(model_local_file_path).exists()
```

否则不提交。
