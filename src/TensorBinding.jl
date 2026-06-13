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
#   utils.jl      — binary/index helpers, diagonal MPO construction (no deps)
#   Hamiltonian.jl — hopping MPO builders (uses utils)
#   2D_lattice.jl  — 2D shift operators and lattice hoppings (uses utils, Hamiltonian)
#   QFT_tk.jl      — QFT conjugation (uses utils)
#   TBSystem.jl    — TBHamiltonian struct and constructor (uses utils, Hamiltonian)
#   Flake_tk.jl    — smooth flake masking via QTCI SDFs (uses TBSystem)
#   QFT_tk.jl      — QFT conjugation (uses TBSystem for high-level overload)
#   KPM_tk.jl      — Chebyshev kernel (uses TBSystem for high-level overload)
#   Topology_tk.jl — topological invariants (uses KPM_tk, Hamiltonian)
#   Purification_tk.jl — density matrix purification (uses KPM_tk)
#   Meanfi_tk.jl   — mean-field SCF loop (uses KPM_tk, TBSystem)
#   RPA_tk.jl      — RPA susceptibility (uses KPM_tk)
#   krylov_tk.jl   — Green's function via vectorized linsolve (uses RPA_tk for interleave_mpo)
#   Timeev_tk.jl   — time evolution (uses Hamiltonian)
#   twoparticle_tk.jl — exciton / two-particle system construction (uses TBSystem, Hamiltonian)

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
#include("RSI_tk.jl")
include("gpu/GPU_tk.jl")

end
