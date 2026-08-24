#!/usr/bin/env python3
"""Compare v3 vs v4 per-round metrics. Python 3.6 compatible."""
import json
from pathlib import Path

def analyze_run(out_dir, n_rounds, label):
    out_dir = Path(out_dir)
    print("\n" + "=" * 90)
    print("{}  ({})".format(label, out_dir))
    print("=" * 90)
    print("{:<6} {:<8} {:<8} {:<8} {:<10} {:<10} {:<10} {:<14}".format(
        "Rnd", "Total", "Succ", "Fail", "BestiPTM", "MeaniPTM", "BestpTM", "Rollback"))
    print("-" * 90)

    best_overall_iptm = 0.0
    best_overall_round = -1
    rollback_count = 0
    produced_rounds = 0
    iptm_series = []

    for i in range(n_rounds):
        rd = out_dir / "round_{:02d}".format(i)
        ev = rd / "evaluation_summary.json"
        rb = rd / "rollback_decision.json"
        if not ev.exists():
            continue
        with open(ev) as f:
            data = json.load(f)
        total = data.get("total_candidates", 0)
        succ = data.get("success_count", 0)
        fail = data.get("failure_count", 0)
        tops = data.get("top_candidates", [])
        best_iptm = 0.0
        best_ptm = 0.0
        all_iptm = []
        for c in tops:
            raw = c.get("raw", {})
            v = float(raw.get("design_to_target_iptm", 0) or 0)
            all_iptm.append(v)
            if v > best_iptm:
                best_iptm = v
                best_ptm = float(raw.get("design_ptm", 0) or 0)
        mean_iptm = sum(all_iptm) / len(all_iptm) if all_iptm else 0.0

        rb_action = ""
        if rb.exists():
            with open(rb) as f:
                rbd = json.load(f)
            rb_action = rbd.get("decision", {}).get("action", "")
            if rb_action in {"replay_best", "branch_from_best"}:
                rollback_count += 1

        if total > 0:
            produced_rounds += 1
            iptm_series.append((i, best_iptm))
        if best_iptm > best_overall_iptm:
            best_overall_iptm = best_iptm
            best_overall_round = i

        print("{:<6} {:<8} {:<8} {:<8} {:<10.4f} {:<10.4f} {:<10.4f} {:<14}".format(
            i, total, succ, fail, best_iptm, mean_iptm, best_ptm, rb_action))

    print("-" * 90)
    print("SUMMARY[{}]: produced_rounds={}, rollback_count={}, best_overall_iPTM={:.4f} @round{}".format(
        label, produced_rounds, rollback_count, best_overall_iptm, best_overall_round))
    return iptm_series

s3 = analyze_run("outputs/sc2rbd_closed_loop_llm_30r_v3", 30, "V3 (30 rounds)")
s4 = analyze_run("outputs/sc2rbd_closed_loop_llm_10r_v4", 10, "V4 (10 rounds, template+rollback)")
sN = analyze_run("outputs/sc2rbd_closed_loop_llm_10r_v4_old_notemplate", 10, "V4 no-template (baseline)")

print("\n" + "=" * 90)
print("V3 iPTM trajectory (best per produced round):")
print("  " + " -> ".join("r{}:{:.3f}".format(r, v) for r, v in s3))
print("V4 iPTM trajectory:")
print("  " + " -> ".join("r{}:{:.3f}".format(r, v) for r, v in s4))
print("=" * 90)
