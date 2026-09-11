"""CKINO: Chebyshev Kernel-Integral Neural Operator.

A novel physics-based neural operator with:
  * Chebyshev–rational spectral basis (no periodicity assumption).
  * Low-rank learnable Green's-function kernel integral.
  * Symplectic residual integrator (structural energy conservation).
  * Lie-generator equivariant lifting (sample-efficient).
  * Hypernetwork meta-conditioning for PDE families.
"""
from .basis import ChebyshevBasis, chebyshev_nodes, chebyshev_diff_matrix
from .kernel import LowRankKernelIntegral
from .symplectic import SymplecticBlock
from .hypernet import HyperNet
from .equivariance import LieLifting
from .model import CKINO
from .nd import (
    SeparableKernelIntegralND,
    SymplecticBlockND,
    LieLiftingND,
    CKINO_ND,
    CKINO2D,
    CKINO3D,
    cheb_eval_matrix,
    clenshaw_curtis_weights,
)

__all__ = [
    "ChebyshevBasis",
    "chebyshev_nodes",
    "chebyshev_diff_matrix",
    "LowRankKernelIntegral",
    "SymplecticBlock",
    "HyperNet",
    "LieLifting",
    "CKINO",
    "SeparableKernelIntegralND",
    "SymplecticBlockND",
    "LieLiftingND",
    "CKINO_ND",
    "CKINO2D",
    "CKINO3D",
    "cheb_eval_matrix",
    "clenshaw_curtis_weights",
]
