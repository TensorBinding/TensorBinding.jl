module TensorBinding

using LinearAlgebra
using Random
using ITensors
using NDTensors
using ITensorMPS
using Quantics
using QuanticsTCI
using QuanticsGrids
using TCIITensorConversion
using TensorCrossInterpolation
import TensorCrossInterpolation as TCI
using PyCall
using PyPlot
using Plots
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

include("utils.jl")
include("Hamiltonian.jl")
include("2D_lattice.jl")
include("TBSystem.jl")
include("nnneighbor_tk.jl")
include("Flake_tk.jl")
include("QFT_tk.jl")
include("KPM_tk.jl")
include("Topology_tk.jl")
include("Purification_tk.jl")
include("SCF_tk.jl")
include("RPA_tk.jl")
include("NH_tk.jl")
#include("RSI_tk.jl")
include("krylov_tk.jl")
include("Timeev_tk.jl")
include("twisted_tk.jl")
include("bilayer_tk.jl")
include("Supercond_tk.jl")
include("DMRG_tk.jl")
include("twoparticle_tk.jl")

end
