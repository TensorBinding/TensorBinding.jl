```@meta
CurrentModule = TensorBinding
```

# TensorBinding.jl

TensorBinding is a Julia library for tight-binding physics on matrix-product-state (MPS/MPO) representations. It targets large-scale 1D and 2D lattice models where exact diagonalisation is infeasible, combining tensor-network methods (DMRG, TDVP, KPM, TCI) with GPU acceleration.

## Installation

TensorBinding is not yet registered. Install directly from the GitHub repository:

```julia
using Pkg
Pkg.add(url="https://github.com/TensorBinding/TensorBinding.jl")
```


```@docs
TensorBinding
```

## Package organisation

| Section | Contents | Source files (under `src/`) |
|---------|----------|-----------------------------|
| [Core](api/core.md) | `TBSystem`, Hamiltonian builders, low-level utilities, MPO plumbing, auxiliary DOFs, model registry | `core/`: `Utils.jl`, `MPOTools.jl`, `Hamiltonian.jl`, `TBSystem.jl`, `AuxDOF.jl`, `ModelRegistry.jl` |
| [Position Spaces](api/position_spaces.md) | Projected Fibonacci, metallic-mean and k-bonacci quasicrystal registers | `position_spaces/`: `Fibonacci.jl`, `MetallicMean.jl`, `KBonacci.jl` |
| [Lattice](api/lattice.md) | 2D shift operators, multilayer, twisted, flake, junction geometries | `lattice/`: `masks2d.jl`, `hopping2d.jl`, `presets.jl`, `sublattice.jl`, `geometry.jl`, `NNNeighbor.jl`, `Bilayer.jl`, `Twisted.jl`, `Flake.jl`, `TJunction.jl` |
| [Solvers](api/solvers.md) | KPM, time evolution, DMRG, Krylov Green's function | `solvers/`: `kpm/*.jl`, `Krylov.jl`, `DMRG.jl`, `Timeev.jl` |
| [Physics](api/physics.md) | SCF, QFT, superconductivity, purification, non-Hermitian, topology, two-particle, RPA, QPI | `physics/`: `SCF.jl`, `rpa/*.jl`, `Topology.jl`, `Purification.jl`, `TwoParticle.jl`, `nh/*.jl`, `QPI.jl`, `qft/*.jl`, `Supercond.jl` |
| [GPU](api/gpu.md) | CUDA-accelerated mirrors of the CPU entry points | `gpu/*.jl` |
