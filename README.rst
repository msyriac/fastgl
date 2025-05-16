FastGL
======

FastGL computes Gauss-Legendre quadrature nodes and weights O(1000)x faster than ``scipy.special.roots_legendre``. It does so by implementing an iteration-free algorithm developed in `Bogaert (2014) <https://epubs.siam.org/doi/abs/10.1137/140954969>`_. This Python package is a thin wrapper around the C++ code from that paper.  A classical iterative algorithm from Kendrick Smith is also implemented, which is around 20x faster than the SciPy implementation. Both are OpenMP parallelized. 


* Free software: BSD license

Usage
-----

This module contains functions to compute the sample points and weights for Gauss-Legendre
quadrature given the quadrature order. 

.. code-block:: python

		>>> from fastgl import roots_legendre
		>>> N = 100
		>>> mu, w_mu = roots_legendre(N) # FastGL calculation
		>>> mu, w_mu = roots_legendre_brute(N) # Classical Iterative calculation

Here, ``mu`` is a numpy array containing the cosine of the sample points (ranging from -1 to 1) and ``w_mu`` is a numpy array containing the corresponding quadrature weights.



Installing
----------

Make sure your ``pip`` tool is up-to-date. To install ``fastgl``, run:

.. code-block:: console
		
   $ pip install fastgl --user

This will install a pre-compiled binary suitable for your system (only Linux and Mac OS X with Python>=3.10 are supported). 

If you require more control over your installation, e.g. using Intel compilers, please see the section below on compiling from source.

Compiling from source (advanced / development workflow)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The easiest way to install from source is to use the ``pip`` tool,
with the ``--no-binary`` flag. This will download the source distribution
and compile it for you. Don't forget to make sure you have CC and FC set
if you have any problems.

For all other cases, below are general instructions.

First, download the source distribution or ``git clone`` this repository. You
can work from ``master`` or checkout one of the released version tags (see the
Releases section on Github). Then change into the cloned/source directory.

Once downloaded, you can install using ``pip install .`` inside the project
directory. We use the ``meson`` build system, which should be understood by
``pip`` (it will build in an isolated environment).

We suggest you then test the installation by running the unit tests. You
can do this by running ``pytest``.

To run an editable install, you will need to do so in a way that does not
have build isolation (as the backend build system, `meson` and `ninja`, actually
perform micro-builds on usage in this case):

.. code-block:: console
   
   $ pip install --upgrade pip meson ninja meson-python cython numpy pybind11
   $ pip install  --no-build-isolation --editable .



   
Contributions
-------------

If you have write access to this repository, please:

1. create a new branch
2. push your changes to that branch
3. merge or rebase to get in sync with master
4. submit a pull request on github

If you do not have write access, create a fork of this repository and proceed as described above.
  
