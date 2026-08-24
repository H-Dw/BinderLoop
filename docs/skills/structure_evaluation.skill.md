---
name: binder-structure-evaluation
version: 0.1.0
description: Extract coordinate-level Binder/interface features for seed reliability and next-round direction selection.
---

# Binder Structure Evaluation Skill

Features: interface contacts/residues, hotspot coverage/min distances, clash density, hydrogen-bond-like/salt-bridge-like/hydrophobic contacts, binder radius of gyration, end-to-end distance, contact-order proxy, chain breaks, interface hydrophobic/polar composition, reliability score/tags.

Rules: tiny interface -> `weak_or_tiny_interface`; high clash density -> `interface_clash_risk`; missing hotspot -> `hotspot_not_covered`; discontinuity -> `binder_chain_break`; suspicious geometry -> `binder_geometry_suspicious`; overly hydrophobic interface -> `over_hydrophobic_interface`.
