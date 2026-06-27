"""Public facade: VQLinear (runtime layer) + build_vq_linear (loader).

Implementation split into _vq_core (the layer) and _vq_build (the loader)."""
from orka.inference._vq_core import VQLinear  # noqa: F401
from orka.inference._vq_build import (  # noqa: F401
    build_vq_linear, _register_layer_buffers, _build_csr_correction, _to_group_major,
)
