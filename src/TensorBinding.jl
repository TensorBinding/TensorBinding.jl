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

export MPO, MPS, OpSum, expect, inner, siteinds

# Load order matters:
#   core/Utils.jl          — binary/index helpers, diagonal MPO construction,
#                             shift/Hadamard operators, and every sampling plan
#                             (spatial, k-space, Fibonacci) shared by the CPU and
#                             GPU solvers (no deps)
#   core/Hamiltonian.jl    — 1D/2D kinetic operator and QTCI MPO builders,
#                             preset model Hamiltonians (uses Utils)
#   core/TBSystem.jl       — position-space policy types, TBHamiltonian struct,
#                             get_Hamiltonian, add_*! mutators and the
#                             position-space interface (uses Utils, Hamiltonian)
#   position_spaces/Fibonacci.jl — projected Fibonacci space, automata,
#                                   constructors, and conumbering (uses TBSystem)
#   position_spaces/MetallicMean.jl — projected metallic-mean spaces (A -> A^m B,
#                                      B -> A) on (m+1)-dimensional Qudit
#                                      registers (uses TBSystem, Utils)
#   position_spaces/KBonacci.jl — projected k-bonacci spaces (Tribonacci,
#                                   Tetranacci, …) on the binary register with
#                                   no k consecutive ones (uses TBSystem, Fibonacci)
#   lattice/masks2d.jl         — sigma_d/sigma_u projectors and diagonal row/column/
#                                 checkerboard mask MPOs (no deps)
#   lattice/hopping2d.jl       — binary shift MPOs and 2D kinetic/hopping MPO
#                                 builders (uses Utils, masks2d)
#   lattice/presets.jl         — preset QTCI model Hamiltonians H* (uses
#                                 Hamiltonian, hopping2d)
#   lattice/sublattice.jl      — kagome/Lieb/honeycomb/dice/SSH sublattice
#                                 Hamiltonians and positions (uses Utils,
#                                 TBSystem, masks2d, hopping2d)
#   lattice/model_registry.jl  — MODEL_REGISTRY, build_hamiltonian and
#                                 _geom_positions (uses presets, sublattice)
#   lattice/NNNeighbor_tk.jl   — generic nth-neighbor hopping accumulator
#                                 add_hopping_2D! (uses TBSystem, masks2d,
#                                 hopping2d, sublattice)
#   lattice/Flake_tk.jl        — smooth flake masking via QTCI SDFs (uses TBSystem)
#   lattice/Bilayer_tk.jl      — bilayer/multilayer commensurate-stacking
#                                 Hamiltonians (uses TBSystem, hopping2d, sublattice)
#   lattice/Twisted_tk.jl      — twisted multilayer Hamiltonians (uses TBSystem,
#                                 Bilayer_tk)
#   lattice/TJunction_tk.jl    — T/Y-junction geometries (uses TBSystem, Hamiltonian)
#   solvers/kpm/           — Chebyshev kernel polynomial method, "KPM_tk" below
#                             (uses TBSystem): kernels.jl (damping kernels, HODC
#                             and DOS weights), recursion.jl (spectral bounds,
#                             KPM_Tn/KPM_Tn_mps, online MPS recursion), cached.jl
#                             (LDOS/Green's functions from cached T_n or μ_n),
#                             ldos.jl (online and spatial LDOS), dos.jl (stochastic
#                             and trace DOS), exciton.jl (exciton LDOS)
#   solvers/Krylov_tk.jl   — Green's function via vectorized linsolve (uses TBSystem)
#   solvers/DMRG_tk.jl     — ground-state and spectral DMRG (uses TBSystem)
#   solvers/Timeev_tk.jl   — time evolution: TDVP, propagator MPO, density-matrix
#                             dynamics (uses Hamiltonian, TBSystem)
#   physics/SCF_tk.jl      — self-consistent mean-field SCF loop (uses KPM_tk,
#                             Purification_tk, TBSystem)
#   physics/rpa/plumbing.jl — MPO kron/interleave site plumbing
#   physics/rpa/bubble.jl   — polarization bubble Π₀(ω) (get_bubble_mpo, Haydock)
#   physics/rpa/cheb2d.jl   — double-Chebyshev bubbles (full MPO, k-space diagonal)
#   physics/rpa/dyson.jl    — RPA susceptibility: Dyson solve, Wynn series, magnon
#                              channel (rpa/ uses KPM_tk, QFT_tk, TwoParticle_tk)
#   physics/Topology_tk.jl — topological invariants: Chern marker, winding
#                             number, Thouless pump (uses KPM_tk, Purification_tk)
#   physics/Purification_tk.jl — density matrix purification: McWeeny, SP2
#                                 (uses KPM_tk)
#   physics/TwoParticle_tk.jl  — exciton/two-particle Hamiltonian and MPS
#                                 basis-state probes (uses TBSystem, Hamiltonian, Utils)
#   physics/nh/model.jl    — non-Hermitian model: NonHermitianHamiltonian,
#                             hermitization, add_loss!/add_nh_* builders
#                             (uses TBSystem, Utils)
#   physics/nh/kpm.jl      — non-Hermitian KPM: nh_kpm_*, spectral function,
#                             nh_spectrum_grid (uses nh/model, KPM_tk, Utils)
#   physics/QPI_tk.jl      — quasiparticle interference via KPM LDOS difference
#                             + QFT (uses KPM_tk, physics/qft)
#   physics/qft/           — QFT conjugation (conjugation.jl), band structure
#                             get_bands (bands.jl), high-symmetry k-paths
#                             (kpath.jl), exciton spectra (exciton_spectra.jl),
#                             aux-index projection (aux_projection.jl)
#                             (uses TBSystem, KPM_tk)
#   physics/Supercond_tk.jl — spin/Nambu extensions: add_spin!,
#                              add_superconductivity! (uses TBSystem, Utils)
#   gpu/*.jl               — GPU production toolkit: _gpu mirrors of KPM/QFT/
#                             Topology/SCF/TwoParticle entry points (uses CUDA,
#                             KPM_tk, QFT_tk, Topology_tk, SCF_tk, TwoParticle_tk):
#     gpu/device.jl        — CUDA bridge, CPU/GPU transfers, residency checks
#     gpu/primitives.jl    — GPU-safe delta/one-hot, MPS evaluation, diagonal
#                             extraction, aux projection, QFT sandwich
#     gpu/kpm.jl           — KPM_Tn_gpu, spatial LDOS, stochastic DOS
#     gpu/bands.jl         — get_bands_gpu
#     gpu/topology.jl      — get_C_gpu
#     gpu/purification.jl  — GPU McWeeny purification
#     gpu/scf.jl           — scf_magnetic_hubbard_gpu and its observables
#     gpu/exciton.jl       — exciton LDOS and Chebyshev convergence
#     gpu/nh.jl            — non-Hermitian KPM density of states
#     gpu/timeev.jl        — NH density and TDVP amplitude trajectories
#     gpu/conductivity.jl  — conductivity-only Tucker/QFT/Hadamard helpers

include("core/Utils.jl")
include("core/Hamiltonian.jl")
include("core/TBSystem.jl")
include("position_spaces/Fibonacci.jl")
include("position_spaces/MetallicMean.jl")
include("position_spaces/KBonacci.jl")
include("lattice/masks2d.jl")
include("lattice/hopping2d.jl")
include("lattice/presets.jl")
include("lattice/sublattice.jl")
include("lattice/model_registry.jl")
include("lattice/NNNeighbor_tk.jl")
include("lattice/Flake_tk.jl")
include("lattice/Bilayer_tk.jl")
include("lattice/Twisted_tk.jl")
include("lattice/TJunction_tk.jl")
include("solvers/kpm/kernels.jl")
include("solvers/kpm/recursion.jl")
include("solvers/kpm/cached.jl")
include("solvers/kpm/ldos.jl")
include("solvers/kpm/dos.jl")
include("solvers/kpm/exciton.jl")
include("solvers/Krylov_tk.jl")
include("solvers/DMRG_tk.jl")
include("solvers/Timeev_tk.jl")
include("physics/SCF_tk.jl")
include("physics/rpa/plumbing.jl")
include("physics/rpa/bubble.jl")
include("physics/rpa/cheb2d.jl")
include("physics/rpa/dyson.jl")
include("physics/Topology_tk.jl")
include("physics/Purification_tk.jl")
include("physics/TwoParticle_tk.jl")
include("physics/nh/model.jl")
include("physics/nh/kpm.jl")
include("physics/QPI_tk.jl")
include("physics/qft/conjugation.jl")
include("physics/qft/bands.jl")
include("physics/qft/kpath.jl")
include("physics/qft/exciton_spectra.jl")
include("physics/qft/aux_projection.jl")
include("physics/Supercond_tk.jl")
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
