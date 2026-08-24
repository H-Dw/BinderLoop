# Task to compare the performance of different filters
# 1. No filter (sc2rbd_structured_task.yaml)
# 2. iptm>0.35 (sc2rbd_structured_task_iptm035.yaml)
# 3. interaction_pae<10 (sc2rbd_structured_task_ipae10.yaml)
# 4. filter_rmsd_design<2.5 (sc2rbd_structured_task_rmsd25.yaml) # not run (has set this filtering)
# 5. Automatic filtering by buddget (sc2rbd_structured_task_budget20.yaml)

# 1. No filter
```bash
nohup python scripts/run_closed_loop_orchestrator.py \
  --config configs/sc2rbd_structured_task.yaml \
  --out outputs/sc2rbd_closed_loop_llm_100s_9r_v15 \
  --max-rounds 9 \
  --llm-config configs/llm_endpoints.local.json \
  --require-llm \
  --submit \
  --boltzgen-heartbeat-seconds 360 \
  --taiji-poll-seconds 120 \
  --taiji-wait-timeout 7200 \
  > sc2rbd_closed_loop_100s_9r_v15.log 2>&1 &
```
# 2. iptm>0.35
```bash
nohup python scripts/run_closed_loop_orchestrator.py \
  --config configs/sc2rbd_structured_task_extend_iptm035.yaml \
  --out outputs/sc2rbd_closed_loop_llm_100s_9r_v15_extend_iptm035 \
  --max-rounds 9 \
  --llm-config configs/llm_endpoints.local.json \
  --require-llm \
  --submit \
  --boltzgen-heartbeat-seconds 360 \
  --taiji-poll-seconds 120 \
  --taiji-wait-timeout 7200 \
  > sc2rbd_closed_loop_100s_9r_v15_extend_iptm035.log 2>&1 &
```
# 3. interaction_pae<10
```bash
nohup python scripts/run_closed_loop_orchestrator.py \
  --config configs/sc2rbd_structured_task_ipae10.yaml \
  --out outputs/sc2rbd_closed_loop_llm_100s_9r_v15_ipae10 \
  --max-rounds 9 \
  --llm-config configs/llm_endpoints.local.json \
  --require-llm \
  --submit \
  --boltzgen-heartbeat-seconds 360 \
  --taiji-poll-seconds 120 \
  --taiji-wait-timeout 7200 \
  > sc2rbd_closed_loop_100s_9r_v15_ipae10.log 2>&1 &
```
# 4. filter_rmsd_design<2.5
```bash
nohup python scripts/run_closed_loop_orchestrator.py \
  --config configs/sc2rbd_structured_task_rmsd25.yaml \
  --out outputs/sc2rbd_closed_loop_llm_100s_9r_v15_rmsd25 \
  --max-rounds 9 \
  --llm-config configs/llm_endpoints.local.json \
  --require-llm \
  --submit \
  --boltzgen-heartbeat-seconds 360 \
  --taiji-poll-seconds 120 \
  --taiji-wait-timeout 7200 \
  > sc2rbd_closed_loop_100s_9r_v15_rmsd25.log 2>&1 &
```
# 5. buddget = 20

<!-- 先做硬过滤：无 X、filter_rmsd、filter_rmsd_design、designfolding-filter_rmsd，以及可选 filter_bindingsite、氨基酸组成偏置等。
再按默认 ranking metrics 排序：design_to_target_iptm、design_ptm、neg_min_design_to_target_pae、plip_hbonds_refolded、plip_saltbridge_refolded、delta_sasa_refolded。
每个设计按 (num_filters_passed, metric) 排名，取各 metric 中最差的 rank 作为质量分组。
最后用 alpha 在质量和序列多样性之间折中，选出 budget 个 final designs。 -->

```bash
nohup python scripts/run_closed_loop_orchestrator.py \
  --config configs/sc2rbd_structured_task_budget20.yaml \
  --out outputs/sc2rbd_closed_loop_llm_100s_9r_v15_budget20 \
  --max-rounds 9 \
  --llm-config configs/llm_endpoints.local.json \
  --require-llm \
  --submit \
  --boltzgen-heartbeat-seconds 360 \
  --taiji-poll-seconds 120 \
  --taiji-wait-timeout 7200 \
  > sc2rbd_closed_loop_100s_9r_v15_budget20.log 2>&1 &
```
```bash
nohup python scripts/run_closed_loop_orchestrator.py \
  --config configs/sc2rbd_structured_task_budget50.yaml \
  --out outputs/sc2rbd_closed_loop_llm_100s_9r_v15_budget50 \
  --max-rounds 9 \
  --llm-config configs/llm_endpoints.local.json \
  --require-llm \
  --submit \
  --boltzgen-heartbeat-seconds 360 \
  --taiji-poll-seconds 120 \
  --taiji-wait-timeout 7200 \
  > sc2rbd_closed_loop_100s_9r_v15_budget50.log 2>&1 &
```