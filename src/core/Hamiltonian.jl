# Hamiltonian.jl - MPO construction for tight-binding Hamiltonians
#
# Functions here build Hamiltonian MPOs from hopping functions or
# lattice parameters.  Low-level tensor utilities live in utils.jl.

# ============================================================
# 1D nearest-neighbour kinetic MPO (quantics binary encoding)
# ============================================================

function _tb_periodic_boundary(boundary::Symbol)
    boundary in (:open, :obc, :OBC) && return false
    boundary in (:periodic, :pbc, :PBC) && return true
    error("Unsupported boundary :$boundary. Use :open or :periodic.")
end

"""
    kinetic_1d_nn(L, sites; boundary=:open) -> MPO

Build the nearest-neighbour hopping MPO for a 1D chain of 2^L sites
in the quantics binary representation. Hopping amplitude = 1; scale by
multiplying the result. The default `boundary=:open` preserves the package's
open-chain convention; `boundary=:periodic` adds the wrap-around bond.
"""
function kinetic_1d_nn(L, sites; boundary::Symbol=:open, bc=nothing)
    @assert L == length(sites) "L must equal length(sites)"
    bc === nothing || (boundary = Symbol(bc))
    K, Kdag = shift_pair_mpos(sites, 1; cyclic=_tb_periodic_boundary(boundary))
    return +(K, Kdag; cutoff=1e-12)
end


# ============================================================
# General QTCI-based hopping MPO
# ============================================================

"""
    hopping2MPO(f, N, sites; tol=1e-8, initial_positions=[], type=Float64,
                unfoldingscheme=:interleaved, nrandominitpivot=5,
                nsearchglobalpivot=5) -> MPO

Compress an arbitrary NxN hopping matrix `H[i,j] = f(i,j)` into an
MPO using Quantics Tensor Cross Interpolation on a 2D quantics grid
(N must be a power of 2).

`unfoldingscheme` controls the bit ordering of the 2D quantics grid:
  - `:interleaved` (default): row and column bits alternate: r_L c_L ... r_1 c_1
  - `:fused`: all row bits first, then column bits: r_L ... r_1 c_L ... c_1

`initial_positions` seeds the TCI pivots; useful when the matrix has
known structure (e.g. near-diagonal for short-time propagators).

`nrandominitpivot` (extra random initial pivots) and `nsearchglobalpivot` (random
global-pivot search points per sweep) are passed to QuanticsTCI; the defaults are
QuanticsTCI's own. Both draw from the global RNG, so the result depends on its state.
With structural `initial_positions`, setting both to `0` gives a deterministic build
that leaves the global RNG untouched.
"""
function hopping2MPO(f, N, sites; tol=1e-8, initial_positions=[], type=Float64,
                     unfoldingscheme=:interleaved, nrandominitpivot::Int=5,
                     nsearchglobalpivot::Int=5)
    L     = Int(log2(N))
    qgrid = QuanticsGrids.DiscretizedGrid{2}(
        L, (1, 1), (N, N);
        includeendpoint=true,
        unfoldingscheme=unfoldingscheme,
    )
    qkw = (; tolerance=tol, nrandominitpivot, nsearchglobalpivot)
    if length(initial_positions) >= 1
        # QuanticsTCI takes the pivots positionally, as grid indices (Vector{Int})
        initialpivots = [collect(QuanticsGrids.origcoord_to_grididx(qgrid, Tuple(Float64.(pos))))
                         for pos in initial_positions]
        ci, _, _ = quanticscrossinterpolate(type, f, qgrid, initialpivots; qkw...)
    else
        ci, _, _ = quanticscrossinterpolate(type, f, qgrid; qkw...)
    end
    citt = TensorCrossInterpolation.TensorTrain(ci.tci)
    mps  = MPS(citt) # modified from ITensors.MPS to MPS 
    @debug "hopping2MPO: QTCI tensor train converted to MPS"
    mpo  = unfoldingscheme == :fused ? fused_mpo(mps, sites) : custom_mpo(mps, sites)
    @debug "hopping2MPO: MPS turned into MPO"
    ITensorMPS.truncate!(mpo; cutoff=1e-8)
    return mpo
end


"""
    qtci_matrix_to_MPO(A_fun, L, sites; tol=1e-8, type=Float64,
                       initial_positions=[]) -> MPO

Like `hopping2MPO` but works with a (2^L)x(2^L) matrix function and
applies an extra truncation step with `maxdim=20`.
"""
function qtci_matrix_to_MPO(A_fun, L, sites;
                             tol=1e-8, type=Float64, initial_positions=[])
    Nc    = Int(2^L)
    qgrid = QuanticsGrids.DiscretizedGrid{2}(
        L, (1, 1), (Nc, Nc);
        includeendpoint=true,
        unfoldingscheme=:interleaved,
    )
    @debug "qtci_matrix_to_MPO: quantics grid built"
    if !isempty(initial_positions)
        # QuanticsTCI takes the pivots positionally, as grid indices (Vector{Int})
        initialpivots = [collect(QuanticsGrids.origcoord_to_grididx(qgrid, Tuple(Float64.(pos))))
                         for pos in initial_positions]
        ci, _, _ = quanticscrossinterpolate(type, A_fun, qgrid, initialpivots;
                                            tolerance=tol)
    else
        ci, _, _ = quanticscrossinterpolate(type, A_fun, qgrid; tolerance=tol)
    end
    @debug "qtci_matrix_to_MPO: QTCI done"
    citt = TensorCrossInterpolation.TensorTrain(ci.tci)
    mps  = ITensors.MPS(citt)
    @debug "qtci_matrix_to_MPO: MPS built"
    mpo  = custom_mpo(mps, sites)
    @debug "qtci_matrix_to_MPO: MPO built"
    ITensorMPS.truncate!(mpo; maxdim=20, cutoff=1e-8)
    return mpo
end


# ============================================================
# General NNN 1D kinetic MPO (spatially varying hopping)
# ============================================================

"""
    kineticNNN(L, sites, hopping, nn; apply_kwargs=NamedTuple()) -> MPO

Build a kinetic MPO for a 1D chain with a **spatially varying hopping field**
encoded as the diagonal MPO `hopping`, and a neighbor reach controlled by `nn`.

Construction uses the shared shift-hopping helper:
`hopping * shift(nn) + shift(nn)' * dag(hopping)`.

`boundary=:open` is the default. Use `boundary=:periodic` (or `bc=:pbc`) for a
cyclic shift on the quantics chain.
"""
function kineticNNN(L, sites, hopping::MPO, nn::Integer;
                    apply_kwargs = NamedTuple(),
                    boundary::Symbol = :open,
                    bc = nothing)
    @assert L == length(sites) "L must equal length(sites)"
    @assert nn >= 1 "nn must be >= 1"
    bc === nothing || (boundary = Symbol(bc))
    return shift_hopping_mpo(hopping, sites, nn;
                             cyclic=_tb_periodic_boundary(boundary),
                             cutoff=1e-12,
                             apply_kwargs=apply_kwargs)
end
