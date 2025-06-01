import numpy as np
import pytest
import fastgl
from scipy.special import roots_legendre
from contextlib import contextmanager
import time

@contextmanager
def timer(name="Block"):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    print(f"[{name}] Elapsed: {(end - start)*1000:.2f} ms")


# FastGL agrees with Scipy up to N=10,000
@pytest.mark.parametrize("n", [4, 8, 16, 32, 64, 1000, 1024, 2048, 4096, 8192, 10000])
def test_fastgl_matches_scipy(n):

    with timer(f"fastgl n={n}"):
        x, weight = fastgl.roots_legendre(n)
    with timer(f"scipy n={n}"):
        xs_scipy, ws_scipy = roots_legendre(n)

    assert np.allclose(x, xs_scipy, atol=1e-14)
    assert np.allclose(weight, ws_scipy, atol=1e-14)


# Classical iterative agrees with Scipy up to N=10,000
@pytest.mark.parametrize("n", [4, 8, 16, 32, 64, 1000, 1024, 2048, 4096, 8192, 10000])
def test_fastgl_brute_matches_scipy(n):

    x2, weight2 = fastgl.roots_legendre_brute(n)
    xs_scipy, ws_scipy = roots_legendre(n)

    assert np.allclose(x2, xs_scipy, atol=1e-14)
    assert np.allclose(weight2, ws_scipy, atol=1e-14)
    
# Classical iterative and FastGL agree *at least* up to N=32,768
@pytest.mark.parametrize("n", [4, 8, 16, 32, 64, 1000, 1024, 2048, 4096, 8192, 10000, 12000, 16384, 32768])
def test_fastgl_internal(n):

    with timer(f"fastgl brute n={n}"):
        x2, weight2 = fastgl.roots_legendre_brute(n)
    with timer(f"fastgl n={n}"):
        x, weight = fastgl.roots_legendre(n)

    assert np.allclose(x2, x, atol=1e-14)
    assert np.allclose(weight2, weight, atol=1e-14)
