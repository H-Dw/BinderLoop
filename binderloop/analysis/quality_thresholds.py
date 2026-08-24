"""Canonical whole-binder success thresholds shared across evidence producers."""

from typing import Dict


SUCCESS_IPTM_MIN = 0.50
SUCCESS_PAE_MAX_ANGSTROM = 10.0
SUCCESS_PTM_MIN = 0.70
SUCCESS_RMSD_MAX_ANGSTROM = 2.5


def success_thresholds() -> Dict[str, float]:
    """Return a fresh mapping suitable for evaluation and prompt evidence."""

    return {
        "design_to_target_iptm": SUCCESS_IPTM_MIN,
        "min_design_to_target_pae": SUCCESS_PAE_MAX_ANGSTROM,
        "design_ptm": SUCCESS_PTM_MIN,
        "designfolding_filter_rmsd": SUCCESS_RMSD_MAX_ANGSTROM,
    }
