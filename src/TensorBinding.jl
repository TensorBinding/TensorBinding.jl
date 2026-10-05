# TensorBinding.jl — the package module: its dependencies, the export list, the
# source map below (what each file holds and calls into) and the include order. The
# API reference is in docs/src/api/.

"""
TensorBinding provides tight-binding physics on MPS/MPO tensor-network representations.
It combines DMRG, KPM, TDVP, and TCI with GPU acceleration for large 1D/2D lattice models.

Authors: Tiago V.C. Antão, Anouar Moustaj, Yitao Sun
"""
module TensorBinding

using LinearAlgebra
using Random
using ITensors
using NDTensors
using ITensorMPS
using Quantics
using QuanticsTCI
using QuanticsGrids
using TensorCrossInterpolation
import TensorCrossInterpolation as TCI
using FFTW
using Base.Threads

# Exports: the main entry points in groups, one comment line per group, alphabetical
# inside each group (ignoring case). A name is exported only when it is specific to
# TensorBinding; generic names (`truncate!`, `hermitize`, `get_matrix`, `get_density`,
# …), the lattice builders `*_hamiltonian`, names Tier 3 will rename, and all types and
# constants stay unexported and are called as `TensorBinding.name`. Types stay unexported
# because error messages print them with `typeof`, which drops the `TensorBinding.` prefix
# once a type is visible in Main, and the golden tests pin those messages. test/exports.jl
# checks the list against the exports of the dependencies and the standard libraries, and
# against the "Public API" list in docs/src/index.md. Running
# `include("src/TensorBinding.jl"); using .TensorBinding` twice in one session makes a
# second module with the same exports, so bare calls to them fail as ambiguous; the
# "Public API" section says how to avoid it.

# ITensors/ITensorMPS names every workflow needs (re-exported, the same objects)
export expect, inner, MPO, MPS, OpSum, siteinds
# Model construction and mutation
export add_hopping!, add_hopping_2D!, add_interaction!, add_onsite!, add_soc!, add_spin!,
       add_superconductivity!, add_tjunction!, add_zeeman!, get_Hamiltonian
# Spectral functions: KPM moments, LDOS, DOS, Krylov Green's function
export get_dos_stochastic, get_dos_trace, get_green_krylov, get_ldos_online,
       get_ldos_spatial, get_ldos_spectrum, KPM_Tn, KPM_Tn_mps
# Band structure and quasiparticle interference
export get_bands, get_qpi
# Density-matrix purification
export mcweeny_purify, sp2_purify
# Mean-field self-consistency
export get_scf
# Topology
export get_thouless_pump, get_valley_operator
# RPA response
export get_bubble_mpo, get_rpa_susceptibility, get_rpa_susceptibility_wynn
# Non-Hermitian models and spectra
export add_nh_nonreciprocal_hopping!, add_nh_onsite!, add_nh_skin_hopping!,
       nh_spectral_function, nh_spectrum_grid
# Excitons
export get_exciton_bands, get_exciton_continuum, get_exciton_ldos_spatial
# Time evolution
export build_tdvp_propagator_mpo, evolve_rk4_dm_nh, evolve_rk4_dm_timedep,
       evolve_with_propagator, evolve_with_tdvp, evolve_with_tdvp_timedep
# Sampling plans
export fibonacci_ldos_sampling_plan, kspace_sampling_plan, spatial_sampling_plan
# GPU entry points (need `using CUDA`)
export get_bands_gpu, get_dos_stochastic_gpu, get_exciton_ldos_spatial_gpu,
       get_ldos_spatial_gpu, get_ldos_spatial_mps_gpu, get_nh_density_trajectory_gpu,
       get_nh_dos_grid_gpu, get_nh_dos_points_gpu, get_scf_bands_gpu,
       get_scf_magnetization_gpu, get_state_amplitude_trajectory_gpu, KPM_Tn_gpu,
       scf_magnetic_hubbard_gpu

# Source map, in include order: one entry per file, what it holds on the first line
# and, after →, the files it calls into (the files that define a function, type or
# constant it uses).
#
# The → lists are derived from the code, not from intent: every name defined in the
# package that the file uses, plus the ITensors op names it builds operators from
# ("sigma_d", "FibRaise", …) and the builders ModelRegistry looks up by Symbol.
# Files are named by stem, with kpm/, rpa/, nh/, qft/, gpu/ for the split folders.
# "TBSystem" stands for the TBHamiltonian type and the position-space interface
# (physical_projector, physical_site_state, …), which dispatches to the projected
# spaces of position_spaces/. "—" means no other file of the package.
#
# A * marks a callee included LATER than its caller. That is legal: Julia resolves a
# call inside a function body when the call runs, not when the file is loaded. The
# order is therefore not a strict layering; a * says the callee is further down.
#
# core/Utils.jl              qubit ops, basis MPS, eval_mps, diagonal/shift MPOs, sampling plans,
#                            dense MPO matrices
#                            → Fibonacci*
# core/MPOTools.jl           mpo_kron, interleave_mpo & co., compose_power, _site_projector_mpo,
#                            sum_mpos, _checked_sum_mpos
#                            → Utils
# core/Hamiltonian.jl        1D kinetic MPOs, the QTCI hopping builder hopping2MPO
#                            → Utils
# core/TBSystem.jl           TBHamiltonian, position spaces, get_Hamiltonian, add_hopping!/add_onsite!
#                            → Utils, Hamiltonian, geometry*, ModelRegistry*, NNNeighbor*
# core/AuxDOF.jl             spin/Nambu indices, add_spin! & co., project_aux, sector projectors,
#                            the aux projection AuxProjection/_project_aux_sectors, probe_state
#                            → Utils, Hamiltonian, TBSystem, hopping2d*, Supercond*
# lattice/geometry.jl        i → position closures, *_positions tables, _resolve_2d_geometry
#                            → TBSystem
# position_spaces/Fibonacci.jl  Zeckendorf space, Fib* ops, constructors, conumbering
#                            → Utils, TBSystem, geometry
# position_spaces/MetallicMean.jl  metallic-mean (A → AᵐB) spaces on Qudit registers
#                            → Utils, TBSystem, geometry
# position_spaces/KBonacci.jl  k-bonacci spaces (no k consecutive ones)
#                            → Utils, TBSystem, Fibonacci (Fib* ops), geometry
# lattice/masks2d.jl         row-break/select and checkerboard mask MPOs
#                            → Utils (sigma_d/u ops)
# lattice/hopping2d.jl       binary shift MPOs (generate_kin_u/d), masked_shift_hopping and
#                            the 2D kinetic builders over it
#                            → Utils, masks2d
# lattice/presets.jl         preset QTCI model Hamiltonians H*
#                            → Utils, MPOTools, Hamiltonian, hopping2d
# core/ModelRegistry.jl      model registry MODELS (MODEL_REGISTRY), build_hamiltonian, the
#                            per-kind builders, the KPM scale maker estimate_scale
#                            → Utils, TBSystem, geometry, Fibonacci, MetallicMean, KBonacci,
#                              presets, sublattice*, DMRG*
# lattice/sublattice.jl      kagome/Lieb/honeycomb/dice/SSH sublattice builders
#                            → Utils, MPOTools, TBSystem, masks2d
# lattice/NNNeighbor.jl      nth-neighbour hopping add_hopping_2D!, get_shell_disps
#                            → Utils, TBSystem, geometry, masks2d, hopping2d
# lattice/Flake.jl           signed-distance functions, QTCI flake masks
#                            → Utils, TBSystem
# lattice/Twisted.jl         twisted multilayer builders
#                            → Utils, MPOTools, Hamiltonian, TBSystem, geometry, ModelRegistry
# lattice/Bilayer.jl         commensurate bilayer/multilayer builders
#                            → Utils, MPOTools, TBSystem, geometry, sublattice, Twisted
# lattice/TJunction.jl       T/Y-junction builders
#                            → Utils, MPOTools, Hamiltonian, TBSystem, geometry, masks2d, NNNeighbor
# solvers/DMRG.jl            dmrg_gs, dmrg_spectral, the KPM spectral bounds (_ensure_scale!)
#                            → Utils, TBSystem
# solvers/kpm/kernels.jl     damping kernels, KPM/HODC/DOS weights, moment-column LDOS, the
#                            energy grid _kpm_energy_grid
#                            → TBSystem
# solvers/kpm/recursion.jl   rescaled H̃ (_scaled_hamiltonian), the three-term recursion
#                            chebyshev_foreach, KPM_Tn, KPM_Tn_mps, online MPS recursion
#                            _run_kpm_mps!
#                            → TBSystem, DMRG
# solvers/kpm/cached.jl      LDOS, density, Green's functions from cached T_n / μ_n; the
#                            weighted T_n sum _chebyshev_sum
#                            → Utils, TBSystem, kpm/kernels
# solvers/kpm/ldos.jl        get_ldos_online, get_ldos_spatial
#                            → Utils, TBSystem, AuxDOF, TJunction, DMRG, kpm/kernels,
#                              kpm/recursion
# solvers/kpm/dos.jl         stochastic and trace DOS
#                            → Utils, TBSystem, AuxDOF, DMRG, kpm/kernels, kpm/recursion
# solvers/kpm/exciton.jl     exciton LDOS
#                            → Utils, TBSystem, DMRG, kpm/kernels, kpm/recursion
# solvers/Krylov.jl          get_green_krylov, Haydock recursion (haydock_cf & co.)
#                            → Utils, MPOTools, TBSystem
# solvers/Timeev.jl          TDVP/RK4 step and trajectory kernels, propagator MPO,
#                            density-matrix RK4 and observables
#                            → Utils, Hamiltonian, TBSystem
# physics/Purification.jl    McWeeny, SP2, get_density and its dispatcher _density_matrix
#                            → Utils, TBSystem, DMRG, kpm/recursion, kpm/cached
# physics/Supercond.jl       pairing MPOs, spin/BdG assemblers
#                            → Utils (sigma_± ops), MPOTools, Hamiltonian, AuxDOF
# physics/SCF.jl             mean-field SCF loops and drivers, BdG and profile helpers
#                            → Utils, TBSystem, AuxDOF, DMRG, Purification, Supercond
# physics/TwoParticle.jl     exciton Hamiltonian, momentum-space exciton MPS probes
#                            → Utils, MPOTools, TBSystem
# physics/qft/conjugation.jl conjugate_by_qft(_exciton), aux-site embedding, get_spect_k
#                            → Utils, MPOTools, TBSystem
# physics/qft/kpath.jl       high-symmetry k-paths
#                            → —
# physics/qft/bands.jl       get_bands; the overview of physics/qft/
#                            → Utils, TBSystem, AuxDOF, masks2d, DMRG, kpm/kernels,
#                              kpm/recursion, qft/conjugation, qft/kpath
# physics/qft/exciton_spectra.jl  exciton bands and continuum
#                            → Utils, TBSystem, DMRG, kpm/kernels, kpm/recursion, TwoParticle
# physics/rpa/bubble.jl      polarization bubble Π₀(ω) via KPM, Krylov or Haydock
#                            → Utils, MPOTools, TBSystem, DMRG, kpm/recursion, kpm/cached,
#                              Krylov, Purification
# physics/rpa/cheb2d.jl      double-Chebyshev bubbles (full MPO, k-space diagonal)
#                            → Utils, TBSystem, DMRG, kpm/kernels, kpm/recursion, kpm/cached,
#                              rpa/bubble, qft/conjugation
# physics/rpa/dyson.jl       RPA Dyson solve, Wynn series, magnon channel
#                            → MPOTools, TBSystem, AuxDOF, Krylov, rpa/bubble, qft/conjugation
# physics/Topology.jl        Chern/winding markers, valley operators, Thouless pump
#                            → Utils, TBSystem, NNNeighbor, kpm/recursion, Purification
# physics/nh/model.jl        NonHermitianHamiltonian, hermitize, add_loss!/add_nh_*
#                            → Utils, TBSystem
# physics/nh/kpm.jl          NH KPM, spectral function, nh_spectrum_grid
#                            → Utils, TBSystem, AuxDOF, DMRG, kpm/kernels, kpm/recursion,
#                              nh/model
# physics/QPI.jl             quasiparticle interference (LDOS difference + QFT)
#                            → Utils, TBSystem, Flake, DMRG, kpm/kernels, kpm/recursion
# gpu/device.jl              CUDA bridge, the upload _to_gpu, type resolution, residency checks;
#                            GPU toolkit overview
#                            → —
# gpu/primitives.jl          delta, MPS evaluation, diagonals, aux projection, QFT sandwich
#                            → Utils, AuxDOF, gpu/device
# gpu/kpm.jl                 KPM_Tn_gpu, spatial LDOS, stochastic DOS
#                            → Utils, TBSystem, AuxDOF, TJunction, DMRG, kpm/kernels,
#                              kpm/recursion, kpm/dos, gpu/device, gpu/primitives
# gpu/bands.jl               get_bands_gpu
#                            → Utils, TBSystem, AuxDOF, masks2d, DMRG, kpm/kernels,
#                              kpm/recursion, qft/kpath, qft/bands, gpu/device, gpu/primitives
# gpu/topology.jl            get_C_gpu
#                            → TBSystem, DMRG, Topology, Purification, gpu/device
# gpu/purification.jl        McWeeny purification
#                            → TBSystem, Purification, gpu/device
# gpu/scf.jl                 scf_magnetic_hubbard_gpu and its observables
#                            → Utils, TBSystem, SCF, gpu/device, gpu/primitives, gpu/bands,
#                              gpu/purification
# gpu/exciton.jl             exciton LDOS, Chebyshev convergence
#                            → Utils, TBSystem, DMRG, kpm/kernels, kpm/recursion, gpu/device
# gpu/nh.jl                  NH KPM density of states
#                            → TBSystem, AuxDOF, kpm/kernels, kpm/recursion, nh/model, nh/kpm,
#                              gpu/device, gpu/primitives
# gpu/timeev.jl              NH density and TDVP amplitude trajectories
#                            → Utils, TBSystem, Timeev, gpu/device, gpu/primitives
# gpu/conductivity.jl        conductivity-only Tucker/QFT/Hadamard helpers
#                            → Utils, TBSystem, rpa/bubble, qft/conjugation, gpu/device,
#                              gpu/primitives

include("core/Utils.jl")
include("core/MPOTools.jl")
include("core/Hamiltonian.jl")
include("core/TBSystem.jl")
include("core/AuxDOF.jl")
include("lattice/geometry.jl")
include("position_spaces/Fibonacci.jl")
include("position_spaces/MetallicMean.jl")
include("position_spaces/KBonacci.jl")
include("lattice/masks2d.jl")
include("lattice/hopping2d.jl")
include("lattice/presets.jl")
include("core/ModelRegistry.jl")
include("lattice/sublattice.jl")
include("lattice/NNNeighbor.jl")
include("lattice/Flake.jl")
include("lattice/Twisted.jl")
include("lattice/Bilayer.jl")
include("lattice/TJunction.jl")
include("solvers/DMRG.jl")
include("solvers/kpm/kernels.jl")
include("solvers/kpm/recursion.jl")
include("solvers/kpm/cached.jl")
include("solvers/kpm/ldos.jl")
include("solvers/kpm/dos.jl")
include("solvers/kpm/exciton.jl")
include("solvers/Krylov.jl")
include("solvers/Timeev.jl")
include("physics/Purification.jl")
include("physics/Supercond.jl")
include("physics/SCF.jl")
include("physics/TwoParticle.jl")
include("physics/qft/conjugation.jl")
include("physics/qft/kpath.jl")
include("physics/qft/bands.jl")
include("physics/qft/exciton_spectra.jl")
include("physics/rpa/bubble.jl")
include("physics/rpa/cheb2d.jl")
include("physics/rpa/dyson.jl")
include("physics/Topology.jl")
include("physics/nh/model.jl")
include("physics/nh/kpm.jl")
include("physics/QPI.jl")
include("gpu/device.jl")
include("gpu/primitives.jl")
include("gpu/kpm.jl")
include("gpu/bands.jl")
include("gpu/topology.jl")
include("gpu/purification.jl")
include("gpu/scf.jl")
include("gpu/exciton.jl")
include("gpu/nh.jl")
include("gpu/timeev.jl")
include("gpu/conductivity.jl")

end
