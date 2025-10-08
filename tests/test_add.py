import numpy as np
from tinytorch_hip import add_numpy

def test_add_numpy():
    x = np.random.randn(1024).astype(np.float32)
    y = np.random.randn(1024).astype(np.float32)
    out = add_numpy(x, y)
    assert np.allclose(out, x + y, atol=1e-6)
