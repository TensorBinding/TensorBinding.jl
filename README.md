**TensorBinding.jl**  
*Compressing Condensed Matter Problems with Tensor Networks*

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://TiagoAntao2.github.io/TensorBinding/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://TiagoAntao2.github.io/TensorBinding/dev/)
[![Build Status](https://github.com/TiagoAntao2/TensorBinding/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/TiagoAntao2/TensorBinding/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/TiagoAntao2/TensorBinding/branch/master/graph/badge.svg)](https://codecov.io/gh/TiagoAntao2/TensorBinding)

**TensorBinding.jl** is a Julia package for constructing and manipulating tight-binding Hamiltonians as **Matrix Product Operators (MPOs)** in the *quantics binary* (QTT) representation. A system of *N = 2<sup>L</sup>* sites is encoded in *L* qubit sites, making the bond dimension of many physically relevant Hamiltonians small (typically ≤ 10) and enabling compression via **Quantics Tensor Cross Interpolation (QTCI)**.

---

### Installation

```julia
using Pkg
Pkg.add(url="https://github.com/TiagoAntao2/TensorBinding")
```

Requires [ITensors.jl](https://github.com/ITensor/ITensors.jl), [ITensorMPS.jl](https://github.com/ITensor/ITensorMPS.jl), and [QuanticsTCI.jl](https://github.com/tensor4all/QuanticsTCI.jl).

---

### Quick Start

```julia
using TensorBinding

# 1D chain, 2^8 = 256 sites
H = get_Hamiltonian("chain_1d", 1.0; L=8, scale=4.5)

# KPM density of states
Tn, scale, center = KPM_Tn(H.mpo / H.scale, 200, H.sites; maxdim=100)
# omega_r = (omega_phys - center) / scale  must be in (-1, 1)
A = get_ldos_w_from_Tn(Tn, 200, 0.0)        # spectral MPO at ω=0
dos = real(tr(A))

# Density matrix at half-filling
rho = get_density_from_Tn(Tn, 200; fermi=0.0, maxdim=100)
```

---

### Key Features

**Hamiltonian construction**
- 1D: nearest-neighbour chain, SSH, Aubry–André–Harper quasicrystal
- 2D: square, triangular, and honeycomb lattices
- Bilayer and multilayer systems with AA or Bernal (AB) stacking
- Twisted multilayer with exponentially decaying interlayer coupling (TCI-compressed)
- Arbitrary hopping matrices `f(i,j)` compressed automatically via QTCI

**Spin and Nambu (BdG) extensions**  
Supported: Ising SOC, Rashba SOC, uniform or site-dependent Zeeman, singlet *s*-wave and custom pairing.

**Kernel Polynomial Method (KPM)**
- Chebyshev expansion of spectral functions, Green's functions, and density matrices
- Kernels: Jackson (default), Lorentz, Fejér, Dirichlet, HODC
- Spectral function *A(k,ω)* via QFT conjugation (`get_bands`)

**Density matrix purification**
- `mcweeny_purify` — cubic map, quadratic convergence
- `sp2_purify` — linear map, 1 MPO product per step, electron-number controlled

**Real-time evolution**
- Pure states: TDVP (fixed and time-dependent *H*)
- Density matrices: RK4 integration of *dρ/dt = −i[H(t), ρ]*

**Many-body methods**
- DMRG ground state and LDoS via DMRG (`dmrg_gs`, `dmrg_spectral`)
- Random Phase Approximation (RPA) for susceptibility via Dyson equation

**Topological invariants**
- Chern number and local Chern marker for Haldane and general 2D models
- Winding number for SSH and general 1D models

**Non-hermitian system**

**Mean-field calculation**

The package is under active development. A full function reference is available in `TensorBinding_overview.txt` and example notebooks are in `examples/`.
