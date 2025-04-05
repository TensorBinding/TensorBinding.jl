**CMTensor.jl**  
*Compressing Condensed Matter Problems with Tensor Networks*

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://YITAOSUN42.github.io/MyPackageName.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://YITAOSUN42.github.io/MyPackageName.jl/dev/)
[![Build Status](https://github.com/YITAOSUN42/MyPackageName.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/YITAOSUN42/MyPackageName.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/YITAOSUN42/MyPackageName.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/YITAOSUN42/MyPackageName.jl)
 
**CMTensor.jl** is a Julia package for solving condensed matter physics problems using tensor network methods. At its core, it applys **Quantics Tensor Cross Interpolation (QTCI)** algorithm to achieve exponential compression of quantum systems:  
- Converts matrix-form tight-binding Hamiltonians into compressed **Matrix Product Operator (MPO)** representations  
- Reduces a system of *2<sup>L</sup>* sites to an effective *L*-site pseudo-spin system  
- Enables efficient simulation of ultra-large real-space systems with boosted computational performance   
QTCI is also further applied to accelerate calculations of physical observables.

---

### **Key Features**  
✅ **Mean-field theory**  
- Self-consistent calculations for arbitrary systems with on-site Hubbard interactions  

✅ **Topological analysis**  
- Computation of topological invariants for selected systems
  
---

### **Why CMTensor.jl?**  
| Traditional Methods | CMTensor.jl Approach |
|---------------------|----------------------|
| Scale as *O(2<sup>L</sup>)* | Scales as *O(poly(L))* |
| Memory-intensive | Memory-optimized via tensor compression |
| Limited to small systems | Designed for ultra-large systems | 

The repo is still under heavy development.
