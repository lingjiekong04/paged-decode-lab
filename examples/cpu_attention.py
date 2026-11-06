"""Synthetic NumPy attention demo. No GPU kernel or model inference occurs."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from paged_decode.reference import dense_reference, streaming_reference

rng = np.random.default_rng(7)
q = rng.standard_normal((1, 4, 64)).astype(np.float32)
k = rng.standard_normal((3, 256, 2, 64)).astype(np.float32)
v = rng.standard_normal(k.shape).astype(np.float32)
table = np.array([[2, 0, 1]], dtype=np.int32)
lengths = np.array([257], dtype=np.int32)
expected = dense_reference(q, k, v, table, lengths)
actual = streaming_reference(q, k, v, table, lengths)
assert np.allclose(actual, expected, atol=2e-5, rtol=2e-5)
print("Synthetic CPU reference passed; no GPU kernel/model was executed.")
print("Logical token 256 -> physical page", int(table[0, 256 // 256]), "slot", 256 % 256)
print("Output shape:", actual.shape, "max abs error:", float(np.max(np.abs(actual - expected))))
