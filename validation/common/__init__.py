"""Shared utilities for the CKINO validation suite."""
from .seed import set_global_seed
from .metrics import (
    relative_l2,
    rollout,
    energy_drift_curve,
    symplectic_defect_2d,
    mass_conservation_error,
)
from .baselines import (
    MLPResidualODE,
    SympNetODE,
    NonSympODE,
    FNO1D,
    DeepONet1D,
    TinyTransformer1D,
    CKINO1DNoSymplectic,
)

__all__ = [
    "set_global_seed",
    "relative_l2",
    "rollout",
    "energy_drift_curve",
    "symplectic_defect_2d",
    "mass_conservation_error",
    "MLPResidualODE",
    "SympNetODE",
    "NonSympODE",
    "FNO1D",
    "DeepONet1D",
    "TinyTransformer1D",
    "CKINO1DNoSymplectic",
]
