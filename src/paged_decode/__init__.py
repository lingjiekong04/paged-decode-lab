"""GPU imports are lazy so the CPU specification can run without torch/Triton."""

from .spec import DecodeSpec

__all__ = ["DecodeSpec"]
