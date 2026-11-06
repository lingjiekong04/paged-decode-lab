"""CPU-only options for the fixed-batch Qwen3.5 runner."""
from .spec import BACKENDS

# The nano-vLLM scheduler can change live batch/table geometry. Its backend list
# deliberately excludes the capacity-aligned fixed-geometry path.
HYBRID_BACKENDS = (*BACKENDS, "triton_matched")


def validate_hybrid_options(backend, mode, batch, max_context):
    if backend not in HYBRID_BACKENDS:
        raise ValueError(f"unknown Qwen3.5 backend {backend!r}; choose {HYBRID_BACKENDS}")
    if mode not in ("eager", "graph"):
        raise ValueError("execution mode must be eager or graph")
    if type(batch) is not int or batch < 1:
        raise ValueError("batch must be a positive integer")
    if type(max_context) is not int or max_context < 1:
        raise ValueError("max_context must be a positive integer")
    if mode == "graph" and max_context < 3:
        raise ValueError("Graph warmup needs context capacity >=3")
