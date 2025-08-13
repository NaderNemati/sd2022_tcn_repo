"""
src package for the Smartphone Decimeter 2022 — TCN Residual Corrector.

Convenience imports are exposed so you can do:
    from src import TCN, add_features, compute_gsdc_score
"""

from .model_tcn import TCN
from .metrics import (
    compute_gsdc_score,
    to_common_columns,
    haversine_m,
    geodetic_to_ecef,
    ecef_to_enu,
    enu_to_ecef,
    ecef_to_geodetic,
)
from .data import add_features, WindowedDataset, prepare_training_tables

__all__ = [
    "TCN",
    "compute_gsdc_score",
    "to_common_columns",
    "haversine_m",
    "geodetic_to_ecef",
    "ecef_to_enu",
    "enu_to_ecef",
    "ecef_to_geodetic",
    "add_features",
    "WindowedDataset",
    "prepare_training_tables",
]

__version__ = "0.1.0"

