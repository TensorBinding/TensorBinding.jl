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
#                             shift/Hadamard operators (no deps)
#   core/Hamiltonian.jl    — 1D/2D kinetic operator and QTCI MPO builders,
#                             preset model Hamiltonians (uses Utils)
#   core/TBSystem.jl       — TBHamiltonian struct, get_Hamiltonian, add_*!
#                             mutators (uses Utils, Hamiltonian)
#   lattice/2Dlattice_tk.jl    — 2D shift operators, lattice hoppings, geometry
#                                 positions (uses Utils, Hamiltonian, TBSystem)
#   lattice/NNNeighbor_tk.jl   — generic nth-neighbor hopping accumulator
#                                 add_hopping_2D! (uses TBSystem, 2Dlattice_tk)
#   lattice/Flake_tk.jl        — smooth flake masking via QTCI SDFs (uses TBSystem)
#   lattice/Bilayer_tk.jl      — bilayer/multilayer commensurate-stacking
#                                 Hamiltonians (uses TBSystem, 2Dlattice_tk)
#   lattice/Twisted_tk.jl      — twisted multilayer Hamiltonians (uses TBSystem,
#                                 Bilayer_tk)
#   lattice/TJunction_tk.jl    — T/Y-junction geometries (uses TBSystem, Hamiltonian)
#   solvers/KPM_tk.jl      — Chebyshev kernel polynomial method (uses TBSystem)
#   solvers/Krylov_tk.jl   — Green's function via vectorized linsolve (uses TBSystem)
#   solvers/DMRG_tk.jl     — ground-state and spectral DMRG (uses TBSystem)
#   solvers/Timeev_tk.jl   — time evolution: TDVP, propagator MPO, density-matrix
#                             dynamics (uses Hamiltonian, TBSystem)
#   physics/SCF_tk.jl      — self-consistent mean-field SCF loop (uses KPM_tk,
#                             Purification_tk, TBSystem)
#   physics/RPA_tk.jl      — RPA susceptibility (uses KPM_tk, QFT_tk, TwoParticle_tk)
#   physics/Topology_tk.jl — topological invariants: Chern marker, winding
#                             number, Thouless pump (uses KPM_tk, Purification_tk)
#   physics/Purification_tk.jl — density matrix purification: McWeeny, SP2
#                                 (uses KPM_tk)
#   physics/TwoParticle_tk.jl  — exciton/two-particle Hamiltonian and MPS
#                                 basis-state probes (uses TBSystem, Hamiltonian, Utils)
#   physics/NH_tk.jl       — non-Hermitian extensions: hermitization, NH KPM
#                             spectral function (uses TBSystem, KPM_tk)
#   physics/QPI_tk.jl      — quasiparticle interference via KPM LDOS difference
#                             + QFT (uses KPM_tk, QFT_tk)
#   physics/QFT_tk.jl      — QFT conjugation and band structure get_bands
#                             (uses TBSystem, KPM_tk)
#   physics/Supercond_tk.jl — spin/Nambu extensions: add_spin!,
#                              add_superconductivity! (uses TBSystem, Utils)
#   gpu/GPU_tk.jl          — GPU production toolkit: _gpu mirrors of KPM/QFT/
#                             Topology/SCF/TwoParticle entry points (uses CUDA,
#                             KPM_tk, QFT_tk, Topology_tk, SCF_tk, TwoParticle_tk)

include("core/Utils.jl")
include("core/Hamiltonian.jl")
include("core/TBSystem.jl")
include("lattice/2Dlattice_tk.jl")
include("lattice/NNNeighbor_tk.jl")
include("lattice/Flake_tk.jl")
include("lattice/Bilayer_tk.jl")
include("lattice/Twisted_tk.jl")
include("lattice/TJunction_tk.jl")
include("solvers/KPM_tk.jl")
include("solvers/Krylov_tk.jl")
include("solvers/DMRG_tk.jl")
include("solvers/Timeev_tk.jl")
include("physics/SCF_tk.jl")
include("physics/RPA_tk.jl")
include("physics/Topology_tk.jl")
include("physics/Purification_tk.jl")
include("physics/TwoParticle_tk.jl")
include("physics/NH_tk.jl")
include("physics/QPI_tk.jl")
include("physics/QFT_tk.jl")
include("physics/Supercond_tk.jl")
include("gpu/GPU_tk.jl")

end
