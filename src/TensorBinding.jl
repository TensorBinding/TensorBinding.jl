module TensorBinding

using LinearAlgebra
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
using Base.Threads

export MPO, MPS, OpSum, expect, inner, siteinds

# Load order matters:
#   utils.jl      — binary/index helpers, diagonal MPO construction (no deps)
#   Geometry.jl   — Python geometry helper (no Julia deps)
#   Hamiltonian.jl — hopping MPO builders (uses utils)
#   2D_lattice.jl  — 2D shift operators and lattice hoppings (uses utils, Hamiltonian)
#   QFT_tk.jl      — QFT conjugation (uses utils)
#   KPM_tk.jl      — Chebyshev kernel (uses utils, Hamiltonian)
#   Topology_tk.jl — topological invariants (uses KPM_tk, Hamiltonian)
#   Purification_tk.jl — density matrix purification (uses KPM_tk)
#   TBSystem.jl    — TBHamiltonian struct and constructor (uses all of the above)
#   Meanfi_tk.jl   — mean-field SCF loop (uses KPM_tk, TBSystem)
#   RPA_tk.jl      — RPA susceptibility (uses KPM_tk)
#   Timeev_tk.jl   — time evolution (uses Hamiltonian)

include("utils.jl")
include("Geometry.jl")
include("Hamiltonian.jl")
include("2D_lattice.jl")
include("QFT_tk.jl")
include("KPM_tk.jl")
include("Topology_tk.jl")
include("Purification_tk.jl")
include("TBSystem.jl")
include("Meanfi_tk.jl")
include("RPA_tk.jl")
include("Timeev_tk.jl")

end
