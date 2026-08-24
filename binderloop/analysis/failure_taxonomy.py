
from typing import Dict, List, Optional


def classify_failures(metrics: Dict[str, float], thresholds: Optional[Dict[str, float]] = None) -> List[str]:
    """Map normalized candidate metrics to interpretable binder-design failure tags.

    Metrics are expected in [0, 1] except penalties where larger is worse.
    """
    t = {
        "binder_plddt": 0.65,
        "interface_confidence": 0.55,
        "hotspot_contact": 0.50,
        "clash_penalty": 0.30,
        "diversity": 0.20,
        **(thresholds or {}),
    }
    tags: List[str] = []
    if float(metrics.get("binder_plddt", 0.0)) < t["binder_plddt"]:
        tags.append("folding_failure")
    if float(metrics.get("interface_confidence", 0.0)) < t["interface_confidence"]:
        tags.append("binding_pose_failure")
    if float(metrics.get("hotspot_contact", 0.0)) < t["hotspot_contact"]:
        tags.append("hotspot_miss")
    if float(metrics.get("clash_penalty", 0.0)) > t["clash_penalty"]:
        tags.append("clash")
    if float(metrics.get("diversity", 0.0)) < t["diversity"]:
        tags.append("diversity_collapse")
    if not tags:
        tags.append("pass_compute_gate")
    return tags
