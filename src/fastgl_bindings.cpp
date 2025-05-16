/*

Copyright (c) 2025, Mathew S. Madhavacheril
// Licensed under the BSD 2-Clause License (see LICENSE file).

*/
// fastgl_bindings.cpp
// pybind11 glue for FastGL.

#include <pybind11/pybind11.h>
#include "fastgl.hpp"
#include <pybind11/numpy.h>
#include <memory>
#include <utility>
#include <vector>

namespace py = pybind11;
namespace fg = fastgl;

PYBIND11_MODULE(_fastgl, m) {
    m.doc() = "Python bindings for FastGL (Gauss-Legendre utilities)";

    /*------------------------------------------------------------
     * QuadPair  –– simple struct with two doubles and a helper
     *-----------------------------------------------------------*/
    py::class_<fg::QuadPair>(m, "QuadPair", R"pbdoc(
A single Gauss–Legendre node/weight pair.

Attributes
----------
theta : float   – angle in radians  (0 <= theta <= pi)
weight: float   – quadrature weight
)pbdoc")
        /* Constructors --------------------------------------------------- */
        .def(py::init<double, double>(),
             py::arg("theta"), py::arg("weight"),
             "Create a QuadPair from (theta, weight).")
        .def(py::init<>())                                     // default ctor
        /* Read-only data members ---------------------------------------- */
        .def_property_readonly("theta",
            [](const fg::QuadPair& q) { return q.theta; },
            "Angle theta of the node (radians).")
        .def_property_readonly("weight",
            [](const fg::QuadPair& q) { return q.weight; },
            "Quadrature weight for this node.")
        /* Methods -------------------------------------------------------- */
        .def("x", &fg::QuadPair::x,
             "Return cos(theta): the node mapped to x-space.")
        /* Nicety for debugging ------------------------------------------ */
        .def("__repr__", [](const fg::QuadPair& q) {
            return "<QuadPair theta=" + std::to_string(q.theta) +
                   ", weight=" + std::to_string(q.weight) + ">";
        });

    /*------------------------------------------------------------
     * free function: GLPair
     *-----------------------------------------------------------*/
    m.def("GLPair",
          &fg::GLPair,
          py::arg("n"), py::arg("k"),
          R"pbdoc(
Return the *k*-th Gauss–Legendre node/weight pair for *n* points.

Parameters
----------
n : int
    Quadrature order (number of nodes).
k : int
    1-based index of the desired node (1 ≤ k ≤ n).

Returns
-------
QuadPair
)pbdoc");




m.def(
    "roots_legendre",
    [](std::size_t n) {
        /* ---------------------------------------------------------
         * Run the parallel C++ routine -> two std::vector<double>
         * --------------------------------------------------------- */
        auto [theta_vec, weight_vec] = fg::roots_legendre(n);

        /* ---------------------------------------------------------
         * Move both vectors into a heap‑allocated struct so we
         * can give NumPy a capsule that owns the memory.
         * --------------------------------------------------------- */
        using VecPair = std::pair<std::vector<double>, std::vector<double>>;
        VecPair* vp   = new VecPair(std::move(theta_vec), std::move(weight_vec));

        /* Create a capsule that will delete the vectors
           when *all* arrays that reference them are gone. */
        py::capsule owner(vp, [](void* p) { delete static_cast<VecPair*>(p); });

        const std::vector<double>& theta_ref  = vp->first;
        const std::vector<double>& weight_ref = vp->second;

        /* ---------------------------------------------------------
         * Build two NumPy 1‑D arrays that *view* the vectors.
         *    Shape  : (n,)
         *    Stride : sizeof(double)
         *    Owner  : capsule  -> keeps vp alive
         * --------------------------------------------------------- */
        py::array theta(
            {static_cast<py::ssize_t>(theta_ref.size())},          // shape
            {static_cast<py::ssize_t>(sizeof(double))},            // stride
            theta_ref.data(),                                      // data ptr
            owner);                                                // base/owner

        py::array weight(
            {static_cast<py::ssize_t>(weight_ref.size())},
            {static_cast<py::ssize_t>(sizeof(double))},
            weight_ref.data(),
            owner);

        /* Return as a 2‑tuple */
        return py::make_tuple(std::move(theta), std::move(weight));
    },
    py::arg("n"),
    R"pbdoc(
Return all Gauss–Legendre nodes and weights for order *n*.

Returns
-------
(theta, weight) : tuple(ndarray, ndarray)
    Two 1‑D NumPy arrays of length *n* (zero‑copy).
)pbdoc");


 m.def("roots_legendre_brute",
      [](int n) {
          std::vector<double> x(n), w(n);
          fastgl::roots_legendre_brute(n, x.data(), w.data());
          return py::make_tuple(py::array(n, x.data()),
                                py::array(n, w.data()));
      },
      py::arg("n"),
      "Return (x, w) as NumPy arrays for Gauss–Legendre rule of order n.");


}
