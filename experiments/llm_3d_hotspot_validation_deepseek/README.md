# DeepSeek-V4-Pro 三维结构 hotspot 盲测后端

本目录把已冻结的 8 靶点 × 3 条件 × 3 重复设计，迁移到本地脚本驱动的
DeepSeek OpenAI-compatible Chat Completions API。它会创建 **72 个全新、无状态、
彼此隔离的预测**，不会复用 GPT 预测结果。

## 隔离边界

- `prepare` 先验证原实验 prediction freeze，再只复制白名单结构输入；新 run ID
  与 GPT run ID 不同。
- DeepSeek 只收到匿名 CIF、紧凑结构特征、输出 schema、盲测 prompt，以及该条件
  唯一允许的 identity card 或冻结通用资料包。
- 不发送标签、local↔author 私有映射、其他 run 输出、仓库路径或评分反馈。
- 不启用 web search、function calling、代码执行或任何 server-side tool。
- API key 通过指定的 `llm_endpoints.*.json` endpoint 解析；同名环境变量优先于
  JSON 的 `secrets` 值。密钥绝不写入请求日志、预测冻结或结果文件。
- 全部 72 个 terminal outcome 完成并冻结后，才允许执行 `unseal-labels`。

注意：结构和提示内容会被上传到所配置的 DeepSeek API 服务。正式调用会产生费用；
请在运行前确认服务条款、数据处理要求、账户余额和速率限制。

## 推荐执行顺序

以下命令从仓库根目录执行。默认读取 `configs/llm_endpoints.ds.json`，也可通过
`--llm-config` 指定其他被 `.gitignore` 排除的 `llm_endpoints.*.json`。配置中的
endpoint 用 `api_key_env` 引用 `secrets`：

```json
{
  "enabled": true,
  "default_model": "deepseek",
  "secrets": {"API_KEY": {"value": "<your-key>"}},
  "endpoints": {
    "deepseek": {
      "provider": "deepseek",
      "base_url": "https://api.deepseek.com",
      "model": "deepseek-v4-pro",
      "api_key_env": "API_KEY",
      "thinking": "high"
    }
  }
}
```

若当前进程设置了 `$env:API_KEY`，它会覆盖 JSON 内的 `secrets.API_KEY`。CLI
不会显示解析出的值。

1. 从已经冻结的 GPT 实验生成独立且无标签的 DeepSeek 输入：

```powershell
python experiments/llm_3d_hotspot_validation_deepseek/run_benchmark.py prepare
```

2. 不连接 API，检查全部 72 个请求的白名单、哈希和大小：

```powershell
python experiments/llm_3d_hotspot_validation_deepseek/run_benchmark.py run --dry-run --llm-config configs/llm_endpoints.ds.json --llm-model deepseek
python experiments/llm_3d_hotspot_validation_deepseek/run_benchmark.py audit-inputs
```

3. 先做一个不接触标签的单波 smoke test，再恢复执行全部 24 波：

```powershell
python experiments/llm_3d_hotspot_validation_deepseek/run_benchmark.py run --llm-config configs/llm_endpoints.ds.json --llm-model deepseek --max-waves 1 --workers 3
python experiments/llm_3d_hotspot_validation_deepseek/run_benchmark.py run --llm-config configs/llm_endpoints.ds.json --llm-model deepseek --workers 3
```

可用 `--requests-per-minute N` 限流。运行可安全恢复：已有合法预测和终止失败会跳过；
网络/限流错误不会生成伪预测，下次运行会重试。格式不合法时只允许一次不含标签或
评分信息的 schema repair。

4. 检查状态并冻结所有输入、输出、代码和审计记录：

```powershell
python experiments/llm_3d_hotspot_validation_deepseek/run_benchmark.py status
python experiments/llm_3d_hotspot_validation_deepseek/run_benchmark.py validate
python experiments/llm_3d_hotspot_validation_deepseek/run_benchmark.py freeze
```

5. 只有 freeze 成功后才解封用户标签并运行原始冻结评分器：

```powershell
python experiments/llm_3d_hotspot_validation_deepseek/run_benchmark.py unseal-labels
python experiments/llm_3d_hotspot_validation_deepseek/run_benchmark.py evaluate
```

最终 Markdown 报告写入 `results/`；逐 run 的结构分析写入各自 `output/process.md`，
请求/响应摘要、重试记录和校验状态写入各自 `scratch/audit/` 并进入 prediction freeze。

## API 默认值

- Base URL：`https://api.deepseek.com`
- Endpoint：`/chat/completions`
- Model：`deepseek-v4-pro`
- Thinking：enabled
- Reasoning effort：`high`
- Response format：`json_object`
- 单次输出上限：本地默认 `32768` tokens
- HTTP timeout：本地默认 `900` 秒
- Transport retry：最多 3 次，指数退避

默认优先采用所选 endpoint 的 base URL、model、thinking、output limit、timeout
和 retry 配置；`run --help` 中的显式参数可覆盖它们。未提供 JSON 配置时，底层
加载器仍支持 `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL` 和 `DEEPSEEK_MODEL`
环境变量。不要提交任何包含真实密钥的 `llm_endpoints.*.json`。

## 定向运行

`run` 支持 `--run-id`、`--case-id`、`--condition`、`--replicate` 和
`--max-waves`。筛选只改变此次调度范围，不改变冻结 run plan。正式实验仍要求 72 个
run 全部形成合法预测、排除预测或记录在案的终止失败后才能 freeze。
