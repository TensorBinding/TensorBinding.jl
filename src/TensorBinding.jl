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
#   core/Utils.jl          — Qubit op extensions (sigma_plus/minus, sigma_d/u),
#                             binary/index helpers and product-state MPS
#                             (incl. mpsexciton), MPS evaluation (eval_mps,
#                             _eval_diag_mps), diagonal MPO construction (incl.
#                             qtt_mpo), shift/Hadamard operators, prepend_op/
#                             postpend_op and the layer prepend helpers, and every
#                             sampling plan (spatial, k-space, Fibonacci) shared
#                             by the CPU and GPU solvers (no deps)
#   core/MPOTools.jl       — MPO Kronecker product, interleave/pair-collapse
#                             site plumbing, compose_power, _site_projector_mpo
#                             (uses Utils)
#   core/Hamiltonian.jl    — 1D kinetic operators (kinetic_1d_nn, kineticNNN)
#                             and the QTCI hopping MPO builders (uses Utils)
#   core/TBSystem.jl       — position-space policy types, TBHamiltonian struct,
#                             get_Hamiltonian, add_hopping!/add_onsite!/
#                             add_interaction! and the position-space
#                             interface (uses Utils, Hamiltonian)
#   core/AuxDOF.jl         — auxiliary DOFs: spin/Nambu indices and op tables,
#                             prepend/postpend_spin/nambu, add_spin!/add_zeeman!/
#                             add_superconductivity!/add_soc!, project_aux,
#                             aux_site, sector projectors, KPM aux setup
#                             (uses Utils, TBSystem)
#   position_spaces/Fibonacci.jl — projected Fibonacci space, automata,
#                                   constructors, and conumbering (uses TBSystem)
#   position_spaces/MetallicMean.jl — projected metallic-mean spaces (A -> A^m B,
#                                      B -> A) on (m+1)-dimensional Qudit
#                                      registers (uses TBSystem, Utils)
#   position_spaces/KBonacci.jl — projected k-bonacci spaces (Tribonacci,
#                                   Tetranacci, …) on the binary register with
#                                   no k consecutive ones (uses TBSystem, Fibonacci)
#   lattice/geometry.jl        — real-space geometry: i -> position closures and
#                                 _preset_geometry, the *_positions tables (preset,
#                                 sublattice and T/Y-junction lattices),
#                                 _geom_positions, lattice_positions and
#                                 _resolve_2d_geometry (uses TBSystem)
#   lattice/masks2d.jl         — diagonal row/column/checkerboard mask MPOs for
#                                 the 2D row-major layout (uses Utils' sigma_d/u)
#   lattice/hopping2d.jl       — binary shift MPOs and 2D kinetic/hopping MPO
#                                 builders (uses Utils, masks2d)
#   lattice/presets.jl         — preset QTCI model Hamiltonians H* (uses
#                                 Hamiltonian, hopping2d)
#   core/ModelRegistry.jl      — MODEL_REGISTRY, _parse_param_string and the
#                                 build_hamiltonian dispatcher (uses presets)
#   lattice/sublattice.jl      — kagome/Lieb/honeycomb/dice/SSH sublattice
#                                 Hamiltonians (uses Utils, TBSystem, masks2d,
#                                 hopping2d, geometry)
#   lattice/NNNeighbor_tk.jl   — generic nth-neighbor hopping accumulator
#                                 add_hopping_2D! (uses TBSystem, masks2d,
#                                 hopping2d, sublattice, geometry)
#   lattice/Flake_tk.jl        — smooth flake masking via QTCI SDFs (uses TBSystem)
#   lattice/Bilayer_tk.jl      — bilayer/multilayer commensurate-stacking
#                                 Hamiltonians (uses TBSystem, hopping2d, sublattice,
#                                 geometry)
#   lattice/Twisted_tk.jl      — twisted multilayer Hamiltonians (uses TBSystem,
#                                 Bilayer_tk, geometry)
#   lattice/TJunction_tk.jl    — T/Y-junction geometries (uses TBSystem, Hamiltonian,
#                                 MPOTools, geometry)
#   solvers/DMRG_tk.jl     — ground-state and spectral DMRG, and the DMRG spectral
#                             bounds _estimate_spectral_bounds/_ensure_scale! that
#                             every KPM caller uses (uses TBSystem)
#   solvers/kpm/           — Chebyshev kernel polynomial method, "KPM_tk" below
#                             (uses TBSystem, AuxDOF, DMRG_tk): kernels.jl (damping
#                             kernels, KPM/HODC/DOS weights, moment-column
#                             reconstruction), recursion.jl (KPM_Tn/KPM_Tn_mps, online
#                             MPS recursion), cached.jl (LDOS/Green's functions from
#                             cached T_n or μ_n), ldos.jl (online and spatial LDOS),
#                             dos.jl (stochastic and trace DOS), exciton.jl (exciton LDOS)
#   solvers/Krylov_tk.jl   — Green's function via vectorized linsolve and the
#                             Haydock recursion haydock_cf (uses TBSystem, MPOTools)
#   solvers/Timeev_tk.jl   — time evolution: TDVP, propagator MPO, density-matrix
#                             dynamics (uses Hamiltonian, TBSystem)
#   physics/SCF_tk.jl      — self-consistent mean-field SCF loop (uses KPM_tk,
#                             Purification_tk, TBSystem, AuxDOF)
#   physics/rpa/bubble.jl   — polarization bubble Π₀(ω) (get_bubble_mpo,
#                              get_bubble_mpo_haydock; uses Krylov_tk)
#   physics/rpa/cheb2d.jl   — double-Chebyshev bubbles (full MPO, k-space diagonal)
#   physics/rpa/dyson.jl    — RPA susceptibility: Dyson solve, Wynn series, magnon
#                              channel (rpa/ uses MPOTools, KPM_tk, QFT_tk,
#                              TwoParticle_tk, AuxDOF)
#   physics/Topology_tk.jl — topological invariants: Chern marker, winding
#                             number, Thouless pump (uses KPM_tk, Purification_tk)
#   physics/Purification_tk.jl — density matrix purification: McWeeny, SP2
#                                 (uses KPM_tk)
#   physics/TwoParticle_tk.jl  — exciton/two-particle Hamiltonian and momentum-space
#                                 MPS probes (uses TBSystem, Hamiltonian, Utils,
#                                 MPOTools)
#   physics/nh/model.jl    — non-Hermitian model: NonHermitianHamiltonian,
#                             hermitization, add_loss!/add_nh_* builders
#                             (uses TBSystem, Utils)
#   physics/nh/kpm.jl      — non-Hermitian KPM: nh_kpm_*, spectral function,
#                             nh_spectrum_grid (uses nh/model, KPM_tk, Utils)
#   physics/QPI_tk.jl      — quasiparticle interference via KPM LDOS difference
#                             + QFT (uses KPM_tk, physics/qft)
#   physics/qft/           — QFT conjugation and the k-space diagonal get_spect_k
#                             (conjugation.jl), band structure
#                             get_bands (bands.jl), high-symmetry k-paths
#                             (kpath.jl), exciton spectra (exciton_spectra.jl)
#                             (uses TBSystem, AuxDOF, KPM_tk, MPOTools)
#   physics/Supercond_tk.jl — pairing MPO builders (pairingNNN, pairing2MPO) and
#                              spin/BdG assemblers (uses AuxDOF, Hamiltonian, Utils,
#                              MPOTools)
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
include("core/MPOTools.jl")
include("core/Hamiltonian.jl")
include("core/TBSystem.jl")
include("core/AuxDOF.jl")
include("position_spaces/Fibonacci.jl")
include("position_spaces/MetallicMean.jl")
include("position_spaces/KBonacci.jl")
include("lattice/geometry.jl")
include("lattice/masks2d.jl")
include("lattice/hopping2d.jl")
include("lattice/presets.jl")
include("core/ModelRegistry.jl")
include("lattice/sublattice.jl")
include("lattice/NNNeighbor.jl")
include("lattice/Flake.jl")
include("lattice/Bilayer.jl")
include("lattice/Twisted.jl")
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
include("physics/SCF.jl")
include("physics/rpa/bubble.jl")
include("physics/rpa/cheb2d.jl")
include("physics/rpa/dyson.jl")
include("physics/Topology.jl")
include("physics/Purification.jl")
include("physics/TwoParticle.jl")
include("physics/nh/model.jl")
include("physics/nh/kpm.jl")
include("physics/QPI.jl")
include("physics/qft/conjugation.jl")
include("physics/qft/bands.jl")
include("physics/qft/kpath.jl")
include("physics/qft/exciton_spectra.jl")
include("physics/Supercond.jl")
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
