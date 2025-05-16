import numpy as np
import pytest
import fastgl
from scipy.special import roots_legendre

@pytest.mark.parametrize("n", [4, 8, 16, 32, 64, 1000, 1024, 2048, 4096, 8192, 10000, 16384])
def test_fastgl_matches_scipy(n):

    x, weight = fastgl.roots_legendre(n)
    x2, weight2 = fastgl.roots_legendre_brute(n)
    xs_scipy, ws_scipy = roots_legendre(n)

    assert np.allclose(x, xs_scipy, atol=1e-14)
    assert np.allclose(weight, ws_scipy, atol=1e-14)
    assert np.allclose(x2, xs_scipy, atol=1e-14)
    assert np.allclose(weight2, ws_scipy, atol=1e-14)
