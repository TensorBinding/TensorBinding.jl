```@meta
CurrentModule = TensorBinding
```

# TensorBinding.jl

TensorBinding is a Julia library for tight-binding physics on matrix-product-state (MPS/MPO) representations. It targets large-scale 1D and 2D lattice models where exact diagonalisation is infeasible, combining tensor-network methods (DMRG, TDVP, KPM, TCI) with GPU acceleration.

## Installation

TensorBinding is not yet registered. Install directly from the GitHub repository:

```julia
using Pkg
Pkg.add(url="https://github.com/TensorBinding/TensorBinding")
```

## Quick start

```julia
using TensorBinding

# Build a 1D tight-binding chain with 2^10 sites
sys = TBSystem(10)                   # 2^10 = 1024 sites, MPS representation
add_hopping!(sys, -1.0)             # nearest-neighbour hopping t = -1

# Compute the density of states via KPM (N = 200 Chebyshev moments)
dos = KPM_DOS(sys; N=200, η=0.02)
```

```@docs
TensorBinding
```

## Package organisation

| Section | Contents |
|---------|----------|
| [Core](api/core.md) | `TBSystem`, Hamiltonian builders, low-level utilities |
| [Lattice](api/lattice.md) | 2D shift operators, multilayer, twisted, flake, junction geometries |
| [Solvers](api/solvers.md) | KPM, Krylov Green's function, DMRG, time evolution |
| [Physics](api/physics.md) | SCF, RPA, topology, purification, two-particle, non-Hermitian, QPI, QFT, superconductivity |
| [GPU](api/gpu.md) | CUDA-accelerated mirrors of the CPU entry points |
