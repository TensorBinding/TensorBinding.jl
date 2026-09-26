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

## Public API

`using TensorBinding` brings the main entry points into scope, listed below by area.
The rest of the API, including the types (`TBHamiltonian`, the position spaces,
`NonHermitianHamiltonian`, …), is not exported: call it as `TensorBinding.name` or import
it with `using TensorBinding: name`. TensorBinding exports only names specific to it, so
that loading it next to other packages does not make them ambiguous; generic names such as
`truncate!`, `hermitize` or `get_density` stay qualified. The six re-exported ITensors
names are the exception: a package that exports its own `OpSum`, `inner`, `expect`, `MPO`
or `MPS` (XDiag exports `OpSum` and `inner`, for example) clashes with them exactly as it
does with `using ITensors`; write `ITensors.OpSum` and so on in that case. The GPU
functions need CUDA.jl to be loaded (see the README).

- ITensors names, re-exported: `expect`, `inner`, `MPO`, `MPS`, `OpSum`, `siteinds`
- Model construction and mutation: `add_hopping!`, `add_hopping_2D!`, `add_interaction!`,
  `add_onsite!`, `add_soc!`, `add_spin!`, `add_superconductivity!`, `add_tjunction!`,
  `add_zeeman!`, `get_Hamiltonian`
- Spectral functions: `get_dos_stochastic`, `get_dos_trace`, `get_green_krylov`,
  `get_ldos_online`, `get_ldos_spatial`, `get_ldos_spectrum`, `KPM_Tn`, `KPM_Tn_mps`
- Band structure and quasiparticle interference: `get_bands`, `get_qpi`
- Density-matrix purification: `mcweeny_purify`, `sp2_purify`
- Mean-field self-consistency: `get_scf`
- Topology: `get_thouless_pump`, `get_valley_operator`
- RPA response: `get_bubble_mpo`, `get_rpa_susceptibility`, `get_rpa_susceptibility_wynn`
- Non-Hermitian models and spectra: `add_nh_nonreciprocal_hopping!`, `add_nh_onsite!`,
  `add_nh_skin_hopping!`, `nh_spectral_function`, `nh_spectrum_grid`
- Excitons: `get_exciton_bands`, `get_exciton_continuum`, `get_exciton_ldos_spatial`
- Time evolution: `build_tdvp_propagator_mpo`, `evolve_rk4_dm_nh`, `evolve_rk4_dm_timedep`,
  `evolve_with_propagator`, `evolve_with_tdvp`, `evolve_with_tdvp_timedep`
- Sampling plans: `fibonacci_ldos_sampling_plan`, `kspace_sampling_plan`,
  `spatial_sampling_plan`
- GPU: `get_bands_gpu`, `get_dos_stochastic_gpu`, `get_exciton_ldos_spatial_gpu`,
  `get_ldos_spatial_gpu`, `get_ldos_spatial_mps_gpu`, `get_nh_density_trajectory_gpu`,
  `get_nh_dos_grid_gpu`, `get_nh_dos_points_gpu`, `get_scf_bands_gpu`,
  `get_scf_magnetization_gpu`, `get_state_amplitude_trajectory_gpu`, `KPM_Tn_gpu`,
  `scf_magnetic_hubbard_gpu`

If you load the source rather than the installed package, with
`include("src/TensorBinding.jl"); using .TensorBinding` as the example notebooks do, run
that setup once per session. Running it again creates a second `TensorBinding` module
with the same exports, and a bare call such as `get_bands(...)` then fails with an
`UndefVarError` that calls the name ambiguous. Loading both the installed package
(`using TensorBinding`) and the source into one session does the same. To pick up edits
to the source, restart the session before running the setup again, or call the functions
qualified (`TensorBinding.get_bands(...)` always means the latest include), or develop
the package with `] dev` and Revise and load it with `using TensorBinding`.
