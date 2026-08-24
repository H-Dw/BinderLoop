# BoltzGen 真实 Taiji 单任务提交测试报告

## 1. 测试约束

用户确认允许使用：

```text
examples/bg_example/boltzgen_test_v100.json
```

中的 Taiji Token 和 mount secret，并要求：

```text
单次仅可以提交一个任务，限制为当前的服务器设置
```

因此本次仅执行了一次：

```bash
taiji_client start -scfg <generated_json>
```

未再次提交第二个任务。

## 2. 生成文件

输出目录：

```text
outputs/boltzgen_complete_path_test_real_taiji/
```

生成的 Taiji JSON：

```text
outputs/boltzgen_complete_path_test_real_taiji/taiji_complete_path_single_task.json
```

注意：该文件基于用户授权的示例 JSON 生成，包含提交所需敏感字段，不应外传。

脱敏 manifest：

```text
outputs/boltzgen_complete_path_test_real_taiji/submission_manifest_redacted.json
```

提交记录：

```text
outputs/boltzgen_complete_path_test_real_taiji/taiji_start_record.json
```

## 3. 本次生成的 BoltzGen 流程

本次任务的远端计划目录：

```text
/aceph/daweihuang/dataset/proteo_benchmark/boltzgen_harness_tests/<run_id>/
```

计划运行的 BoltzGen 命令包含完整流程：

```bash
--steps design inverse_folding folding design_folding analysis filtering
```

本轮测试参数：

- Target：IL-17A
- Binder chain：D
- Binder length：50
- Hotspot：A:67, A:89, B:49
- Protocol：protein-anything
- num_designs：4
- budget：2
- diffusion_batch_size：1
- inverse_fold_num_sequences：1
- filtering：开启

## 4. Taiji 提交结果

提交命令已执行一次：

```bash
taiji_client start -scfg outputs/boltzgen_complete_path_test_real_taiji/taiji_complete_path_single_task.json
```

返回：

```text
returncode=0
taiji_job_id=null
```

Taiji stdout 关键信息：

```text
create dataset ing...
create dataset success
create model ing...
[error][lstat /aceph/daweihuang/dataset/proteo_benchmark/design_scripts/: no such file or directory]
```

## 5. 运行监测结论

本次没有获得 Taiji instance/job id，因此没有进入真实 GPU 运行阶段，也无法进一步通过：

```bash
taiji_client logs <task_flag> <instance_id>
```

观察 BoltzGen 运行日志。

失败发生在 Taiji client 提交/创建模型阶段，而不是 BoltzGen 运行阶段。

## 6. 当前失败原因分析

本次失败原因是：

```text
model_local_file_path=/aceph/daweihuang/dataset/proteo_benchmark/design_scripts/
```

在当前提交客户端所在机器上不可见，Taiji client 在创建 model 时执行本地 `lstat`，因此报错：

```text
no such file or directory
```

这说明示例 JSON 中的 `model_local_file_path` 不能直接用于当前本地客户端提交，虽然该路径可能是在 Taiji 任务容器内部或挂载后才可见。

## 7. 对 Harness/Agent 的改进建议

`TaijiExecutionAgent` 需要增加提交前 preflight：

1. 如果 simple config 包含 `model_local_file_path`：
   - 检查该路径在当前提交机器是否存在。
   - 若不存在，不直接提交，先提示需要改用可本地访问路径、已有 `model_id`，或先同步/打包提交目录。

2. 支持三种 Taiji 模式：
   - `model_local_file_path` 模式：本地路径必须存在。
   - `model_id` 模式：复用已上传模型。
   - `start_cmd only` / 远端挂载模式：不依赖本地 model path，但需要确认 Taiji simple config 是否允许不填 model 字段。

3. `RunMonitorAgent` 在没有 job id 时应标记：

```text
state=submit_failed_before_instance
failure_hint=model_local_file_path_missing
```

4. 对于 BoltzGen 完整路径测试，建议先解决 Taiji config 的 model source 问题，再提交下一次任务。

## 8. 下一步建议

由于用户限制单次只提交一个任务，本次不再提交第二个任务。

下一次真实运行前建议先确认以下任一方案：

### 方案 A：提供当前机器可见的 `model_local_file_path`

例如一个当前提交客户端能 `ls` 到的目录。

### 方案 B：提供已有可复用的 `model_id`

让 Taiji JSON 使用 `model_id`，避免本地上传路径校验。

### 方案 C：修改 Taiji JSON 使其不依赖 `model_local_file_path`

如果 Taiji 平台允许仅通过 `image_full_name + start_cmd` 运行，则可以移除 `model_local_file_path`。但这需要先用 Taiji schema 或平台文档确认。

在确定上述方案后，再进行下一次单任务提交，并由 `RunMonitorAgent` 跟踪 instance 日志和 BoltzGen 输出。
