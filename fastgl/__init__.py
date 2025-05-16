"""
fastgl 
======


Typical use
-----------
>>> import fastgl
>>> N = 1000
>>> mu, w_mu = fastgl.roots_legendre(N)
>>> mu, w_mu = fastgl.roots_legendre_brute(N)
"""

from importlib import import_module as _import_module
import sys as _sys

_fastgl = _import_module('fastgl._fastgl')

roots_legendre_brute   = _fastgl.roots_legendre_brute
roots_legendre   = _fastgl.roots_legendre

__all__ = ["roots_legendre", "roots_legendre_brute"]

# try:
#     _import_module('.fastgl', package=__name__)
# except ModuleNotFoundError:
#     # The helper module is optional; continue silently if it's missing.
#     pass

try:
    from importlib.metadata import version as _get_version
    __version__ = _get_version(__name__)
except Exception:          # pragma: no cover
    __version__ = "0.0.0"
