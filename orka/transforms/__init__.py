"""Pre-VQ pipeline transforms.

Pack order: normalize -> rotate -> outlier-extract.
Decode reverses: outlier-inject -> un-rotate -> un-normalize.
"""

from orka.transforms.normalize import (
    BLOCK_SCALE_NORMALIZATIONS,
    _apply_block_max_scales,
    _apply_block_max_scales_numpy,
    _apply_col_l2_scales,
    _apply_col_l2_scales_numpy,
    _apply_normalization,
    stores_block_scales,
)
from orka.transforms.outliers import _extract_outliers
from orka.transforms.rotate import (
    _fwht_numpy,
    _fwht_torch,
    _generate_orthogonal_numpy,
    _rotate_tensor_to_2d,
    _tensor_rotation_seed,
    _unrotate_flat,
)
