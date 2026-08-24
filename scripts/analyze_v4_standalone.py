#!/usr/bin/env python3
"""Standalone v4 analysis script - Python 3.6 compatible."""
import json
import os
from pathlib import Path

OUT_DIR = Path("outputs/sc2rbd_closed_loop_llm_10r_v4")

print("=" * 80)
print("BinderHarness v4 闭环运行结果分析")
print("=" * 80)

rounds_data = []

for round_idx in range(10):
    round_dir = OUT_DIR / "round_{:02d}".format(round_idx)
    eval_file = round_dir / "evaluation_summary.json"
    
    if not eval_file.exists():
        print("\nRound {}: evaluation_summary.json NOT FOUND".format(round_idx))
        continue
    
    with open(eval_file) as f:
        data = json.load(f)
    
    total = data.get("total_candidates", 0)
    success = data.get("success_count", 0)
    failure = data.get("failure_count", 0)
    tag_counts = data.get("tag_counts", {})
    top_candidates = data.get("top_candidates", [])
    
    # Extract iPTM and other metrics from top candidate
    best_iptm = 0
    best_ptm = 0
    if top_candidates:
        best = top_candidates[0]
        raw = best.get("raw", {})
        best_iptm = float(raw.get("design_to_target_iptm", 0))
        best_ptm = float(raw.get("design_ptm", 0))
    
    # Compile round summary
    rounds_data.append({
        "round": round_idx,
        "total": total,
        "success": success,
        "failure": failure,
        "success_rate": success / total * 100 if total > 0 else 0,
        "best_iptm": best_iptm,
        "best_ptm": best_ptm,
        "tag_counts": tag_counts,
    })

# Print summary table
print("\n{:<8} {:<10} {:<10} {:<10} {:<12} {:<10} {:<10}".format(
    "Round", "Total", "Success", "Fail", "Rate(%)", "Best iPTM", "Best pTM"))
print("-" * 80)

for rd in rounds_data:
    print("{:<8} {:<10} {:<10} {:<10} {:<12.1f} {:<10.4f} {:<10.4f}".format(
        rd["round"], rd["total"], rd["success"], rd["failure"],
        rd["success_rate"], rd["best_iptm"], rd["best_ptm"]))

print("-" * 80)
total_all = sum(r["total"] for r in rounds_data)
success_all = sum(r["success"] for r in rounds_data)
print("{:<8} {:<10} {:<10} {:<10} {:<12.1f}".format(
    "TOTAL", total_all, success_all, total_all - success_all,
    success_all / total_all * 100 if total_all > 0 else 0))

# Check boltzgen template usage
print("\n" + "=" * 80)
print("BoltzGen Template (binder_template) 实施情况检查")
print("=" * 80)

for round_idx in range(10):
    round_dir = OUT_DIR / "round_{:02d}".format(round_idx)
    
    # Check fragment_templates.json
    frag_file = round_dir / "fragment_templates.json"
    has_frag = frag_file.exists()
    
    # Check next_round_config.yaml for binder_template
    config_file = round_dir / "next_round_config.yaml"
    has_template = False
    template_mode = "N/A"
    exploit_fragments = []
    avoid_fragments = []
    
    if config_file.exists():
        with open(config_file) as f:
            for line in f:
                if "binder_template:" in line and "null" not in line and "None" not in line:
                    has_template = True
                if "mode:" in line and has_template:
                    template_mode = line.split("mode:")[-1].strip()
    
    # Check next_round_parameter_proposal.json for exploit/avoid modules
    proposal_file = round_dir / "next_round_parameter_proposal.json"
    if proposal_file.exists():
        with open(proposal_file) as f:
            try:
                prop = json.load(f)
                exploit_fragments = prop.get("exploit_fragment_modules", [])
                avoid_fragments = prop.get("avoid_fragment_modules", [])
            except:
                pass
    
    print("\nRound {}: ".format(round_idx), end="")
    print("fragment_templates: {} | ".format("YES({} bytes)".format(frag_file.stat().st_size) if has_frag else "NO"), end="")
    print("binder_template in next_config: {} | ".format("YES" if has_template else "NO"), end="")
    print("exploit_fragments: {} | avoid_fragments: {}".format(len(exploit_fragments), len(avoid_fragments)))

# Check rollback decisions
print("\n" + "=" * 80)
print("Rollback 决策总结")
print("=" * 80)
for round_idx in range(10):
    rb_file = OUT_DIR / "round_{:02d}".format(round_idx) / "rollback_decision.json"
    if rb_file.exists():
        with open(rb_file) as f:
            rb = json.load(f)
        decision = rb.get("decision", "unknown")
        reason = rb.get("reason", "")
        print("Round {}: decision={} | reason={}".format(round_idx, decision, reason[:120]))

# Check memory for cross-round template library
print("\n" + "=" * 80)
print("跨轮 Template Library 检查")
print("=" * 80)
mem_file = OUT_DIR / "memory" / "experiment_memory.json"
if mem_file.exists():
    # Read just the template_library related parts
    with open(mem_file) as f:
        # Read first 50KB to check structure
        mem_content = f.read(50000)
    
    if "template_library" in mem_content:
        # Count template entries
        tpl_count = mem_content.count('"pdb_id"')  
        print("Template library EXISTS (found template entries)")
        print("Estimated template entries (in first 50KB): {}".format(tpl_count))
    else:
        print("Template library NOT FOUND in memory (first 50KB)")
else:
    print("experiment_memory.json NOT FOUND")

print("\n" + "=" * 80)
print("分析完成")
print("=" * 80)
