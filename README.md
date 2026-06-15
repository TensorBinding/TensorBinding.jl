**TensorBinding.jl**  
*Compressing Condensed Matter Problems with Tensor Networks*

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://TiagoAntao2.github.io/TensorBinding/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://TiagoAntao2.github.io/TensorBinding/dev/)
[![Build Status](https://github.com/TiagoAntao2/TensorBinding/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/TiagoAntao2/TensorBinding/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/TiagoAntao2/TensorBinding/branch/master/graph/badge.svg)](https://codecov.io/gh/TiagoAntao2/TensorBinding)

**TensorBinding.jl** is a Julia package for constructing and studying tight-binding Hamiltonians as **Matrix Product Operators (MPOs)** in the *quantics binary* (QTT) representation. A system of *N = 2<sup>L</sup>* sites is encoded in *L* qubit sites, keeping bond dimensions small (typically ≤ 10) for physically relevant models. Arbitrary hopping matrices are compressed automatically via **Quantics Tensor Cross Interpolation (QTCI)**.

---

### Installation

```julia
using Pkg
Pkg.add(url="https://github.com/TensorBinding/TensorBinding")
```

Requires [ITensors.jl](https://github.com/ITensor/ITensors.jl), [ITensorMPS.jl](https://github.com/ITensor/ITensorMPS.jl), and [QuanticsTCI.jl](https://github.com/tensor4all/QuanticsTCI.jl).

---

### Quick Start

```julia
using TensorBinding

# 1D chain, 2^7 = 128 sites
H = get_Hamiltonian("chain_1d", 1.0; L=7)

# Density of states via KPM
omega = range(-2.5, 2.5; length=200)
dos = get_dos(H, 100, collect(omega); maxdim=40)

# Density matrix at half-filling
rho = mcweeny_purify(H; maxdim=50)

# Band structure (1D → momentum space)
kvals, Ek = get_bands(H, 200; nk=256)
```

---

### Key Features

**Hamiltonian construction**
- 1D: nearest-neighbour chain, SSH (uniform and sublattice-explicit), Aubry–André–Harper quasicrystal, uniform with on-site potential
- 2D: square, triangular, honeycomb, kagomé, Lieb, and dice lattices — including sublattice-explicit models with an explicit unit-cell index
- Generic *n*th-nearest-neighbour hopping on any 2D geometry (`add_hopping_2D!`): uniform, direction-dependent, site-dependent, or fully position+direction-dependent amplitude functions
- Arbitrary hopping matrix `f(i,j)` compressed via QTCI (`hopping2MPO`)
- T- and Y-junction geometries via a dim-3 junction auxiliary index

**Bilayer and multilayer**
- Commensurate AA and Bernal (AB) stacking, exact interlayer coupling (no QTCI)
- Twisted multilayer with exponentially decaying interlayer coupling, QTCI-compressed

**Spin, Nambu (BdG), and SOC extensions**
- Prepend or postpend spin-½ and Nambu indices (`add_spin!`, `add_superconductivity!`)
- Zeeman coupling, Ising SOC, Rashba SOC; singlet *s*-wave, *p*-wave (Kitaev), and arbitrary custom pairing (`type=:custom`)
- Auxiliary indices placeable at front (`:pre`) or back (`:post`) of the site chain

**Domain masking and geometry**
- Smooth domain walls via signed-distance-function (SDF) masks: disk, rectangle, half-plane, interval — QTCI-compressed sigmoid-shaped diagonal MPOs (`Flake_tk.jl`)
- Real-space geometry functions for all lattices; geometric centroid helpers

**Kernel Polynomial Method (KPM)**
- Chebyshev expansion of spectral functions, LDOS, Green's functions, and density matrices
- Kernels: Jackson (default), Lorentz, Fejér, Dirichlet, HODC
- Three complementary modes: MPO (full operator), diagonal/online (memory-efficient LDOS), MPS (reference-state propagation)
- Band structure *A(k,ω)* via QFT conjugation (`get_bands`); supports spin, BdG, layer, and sublattice projections via `aux_proj`

**Density matrix purification**
- `mcweeny_purify` — cubic map, quadratic convergence, two MPO products per step
- `sp2_purify` — second-order spectral projection, one MPO product per step, electron-number controlled

**Real-time evolution**
- Pure states: TDVP (time-independent and time-dependent *H*); compressed propagator MPO via QTCI
- Density matrices (Hermitian): RK4 integration of *dρ/dt = −i[H(t), ρ]*
- Density matrices (non-Hermitian): RK4 integration of *dρ/dt = −i(Hρ − ρH†)*

**Topological invariants**
- Real-space Chern marker (2D) and winding-number density (1D) via KPM or purification
- Quenched and flat position operators; compatible with all lattice geometries and auxiliary DOFs

**Non-Hermitian systems**
- Hermitization into a 2×2 block form with a placeable auxiliary index (`:pre`/`:post`)
- Four KPM spectral algorithms on a complex energy grid (`nh_spectrum_grid`):
  - `:scalar` — MPO×MPO partial recursion, total DOS
  - `:diag` — same recursion + site-resolved diagonal LDOS MPS
  - `:mps` — dual-chain MPS at a single probe site, O(χ_H·χ_ψ)
  - `:stochastic` — Monte Carlo trace with random product-state probes, no MPO×MPO products
- Complex on-site potentials, spatially modulated loss/gain, and non-reciprocal skin-effect hopping

**Quasiparticle Interference (QPI)**
- Single on-site impurity via exact rank-1 projector; LDOS difference + QFT gives *δA(k,ω)*
- Sigmoid apodization window to suppress edge ringing

**Exciton / two-particle systems**
- Electron–hole Hamiltonian on an interleaved 2*L*-site quantics chain
- Contact interaction; MPS probes in real and momentum space

**Many-body methods**
- DMRG ground state (`dmrg_gs`) and spectral DMRG for site-resolved DOS
- Random Phase Approximation (RPA): polarization bubble and Dyson susceptibility inversion
- Krylov/Haydock retarded Green's function *G(ω+iη)* via vectorized linear solves
- Self-consistent mean-field (Hubbard): Hartree/CDW, magnetic, and BdG pairing channels

**GPU acceleration**
- CUDA-accelerated counterparts for KPM, band structure, topology, SCF, and two-particle LDOS
- Setup (Hamiltonian construction, k-path bookkeeping) stays on CPU; Chebyshev recurrence and MPO products offloaded to GPU

---

A full function reference is in `docs/src/TensorBinding_overview.txt` and example notebooks are in `examples/`.
