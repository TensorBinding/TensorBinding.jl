**TensorBinding.jl**  
*Compressing Condensed Matter Problems with Tensor Networks*

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://TiagoAntao2.github.io/TensorBinding/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://TiagoAntao2.github.io/TensorBinding/dev/)
[![Build Status](https://github.com/TiagoAntao2/TensorBinding/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/TiagoAntao2/TensorBinding/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/TiagoAntao2/TensorBinding/branch/master/graph/badge.svg)](https://codecov.io/gh/TiagoAntao2/TensorBinding)
 
**TensorBinding.jl** is a Julia package for solving condensed matter physics problems using tensor network methods. It compresses tight-binding Hamiltonians into **Matrix Product Operator (MPO)** representations using two complementary approaches:
- **Quantics Tensor Cross Interpolation (QTCI)** — exponential compression via a quantics grid encoding
- **Exact MPO construction** — direct analytical MPO building for structured Hamiltonians

Both reduce a system of *2<sup>L</sup>* sites to an effective *L*-site pseudo-spin system, enabling efficient simulation of ultra-large real-space systems. The MPO framework is then leveraged to accelerate a broad range of physical observables and many-body calculations.

---

### **Key Features**  
✅ **Mean-field theory**  
- Self-consistent calculations for arbitrary systems with on-site Hubbard interactions  

✅ **Topological analysis**  
- Computation of topological invariants for selected systems

✅ **Kernel Polynomial Method (KPM)**  
- Chebyshev expansion of spectral quantities for large-scale systems

✅ **Random Phase Approximation (RPA)**  
- Susceptibility calculations for interacting systems

✅ **Time evolution**  
- Real- and imaginary-time evolution via the TDVP algorithm

✅ **Purification**  
- Finite-temperature density matrix purification

---

### **Why TensorBinding.jl?**  
| Traditional Methods | TensorBinding.jl Approach |
|---------------------|----------------------|
| Scale as *O(2<sup>L</sup>)* | Scales as *O(poly(L))* |
| Memory-intensive | Memory-optimized via tensor compression |
| Limited to small systems | Designed for ultra-large systems | 

The repo is still under heavy development.
