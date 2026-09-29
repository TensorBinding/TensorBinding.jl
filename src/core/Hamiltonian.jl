# Hamiltonian.jl — 1D kinetic MPOs and the QTCI compressor for hopping matrices.
#
# Contents: kinetic_1d_nn (uniform nearest-neighbour chain) with its boundary
# parser _tb_periodic_boundary, hopping2MPO and qtci_matrix_to_MPO (compress an
# arbitrary hopping matrix f(i, j) by 2D QTCI; hopping2MPO(...; check=true) compares the
# result with f on a fixed sample and rebuilds it from structural pivots when it is
# wrong: _hopping2MPO, _hopping_samples, _hopping_mismatch, _hopping_pivots,
# _mpo_site_arrays, _mpo_array_entry), and kineticNNN (a chain with a spatially varying nn-th-neighbour hopping field).
#
# Main entry points: hopping2MPO, kinetic_1d_nn, kineticNNN.
#
# Depends on: Utils (shift_pair_mpos, shift_hopping_mpo, custom_mpo, fused_mpo).
# The 2D kinetic builders are in lattice/hopping2d.jl.

# ============================================================
# 1. 1D nearest-neighbour kinetic MPO (quantics binary encoding)
# ============================================================

function _tb_periodic_boundary(boundary::Symbol)
    boundary in (:open, :obc, :OBC) && return false
    boundary in (:periodic, :pbc, :PBC) && return true
    error("Unsupported boundary :$boundary. Use :open or :periodic.")
end

"""
    kinetic_1d_nn(L, sites; boundary=:open, bc=nothing) -> MPO

Build the nearest-neighbour hopping MPO for a 1D chain of 2^L sites
in the quantics binary representation. Hopping amplitude = 1; scale by
multiplying the result. The default `boundary=:open` preserves the package's
open-chain convention; `boundary=:periodic` adds the wrap-around bond. `bc`
(e.g. `:pbc`), when given, overrides `boundary`.
"""
function kinetic_1d_nn(L, sites; boundary::Symbol=:open, bc=nothing)
    @assert L == length(sites) "L must equal length(sites)"
    bc === nothing || (boundary = Symbol(bc))
    K, Kdag = shift_pair_mpos(sites, 1; cyclic=_tb_periodic_boundary(boundary))
    return +(K, Kdag; cutoff=1e-12)
end


# ============================================================
# 2. General QTCI-based hopping MPO
# ============================================================

"""
    hopping2MPO(f, N, sites; tol=1e-8, initial_positions=[], type=Float64,
                unfoldingscheme=:interleaved, nrandominitpivot=5,
                nsearchglobalpivot=5, check=false) -> MPO

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

`check=true` guards against the weakness of QTCI on sparse `f`: from its default pivot
(the corner entry) and a few random ones it can miss whole bond classes, giving a wrong
MPO or "maxsamplevalue is zero!", depending on the RNG. The build then runs as without
`check`, and its MPO is compared with `f` on a fixed sample of entries (a spread of rows,
column offsets 0, ±1, ±2, ±3, ±2^k and ±(2^k ± 1)); if it passes, it is returned as it is.
Otherwise (or if QTCI threw "maxsamplevalue is zero!") the MPO is rebuilt from the
nonzero sampled entries as pivots (plus `initial_positions`) without random pivots, and
checked again; a second failure, or `f` vanishing on every sample, is an error. An entry
fails when it is off by more than half of `|f(i, j)|`, or by more than `1e-3` of the
largest sampled `|f|`: a missed or spurious bond, not the small deviations of an
approximate compression. The sample calls `f` a few thousand times and draws nothing
from the RNG. `get_Hamiltonian("custom", f)`, `add_hopping!(H, f)` with a two-argument
`f`, `add_superconductivity!(H, Δ; type=:custom)` (through `pairing2MPO`) and
`add_soc!(H, λ; type=:custom)` with a two-argument `λ` turn the check on (their keyword
`check`); the default here is `false`.
"""
function hopping2MPO(f, N, sites; tol=1e-8, initial_positions=[], type=Float64,
                     unfoldingscheme=:interleaved, nrandominitpivot::Int=5,
                     nsearchglobalpivot::Int=5, check::Bool=false)
    build(pivots, nrand, nsearch) =
        _hopping2MPO(f, N, sites; tol, initial_positions=pivots, type, unfoldingscheme,
                     nrandominitpivot=nrand, nsearchglobalpivot=nsearch)
    check || return build(initial_positions, nrandominitpivot, nsearchglobalpivot)

    samples = _hopping_samples(f, N)
    mpo = try
        build(initial_positions, nrandominitpivot, nsearchglobalpivot)
    catch err
        (err isa ErrorException && occursin("maxsamplevalue is zero", err.msg)) || rethrow()
        nothing
    end
    mpo !== nothing && _hopping_mismatch(mpo, sites, samples) === nothing && return mpo

    pivots = [collect(initial_positions); _hopping_pivots(samples)]
    isempty(pivots) &&
        error("hopping2MPO: f vanishes at all $(length(samples)) sampled entries and QTCI " *
              "found no nonzero entry either. If f is identically zero, leave the term out; " *
              "otherwise pass `initial_positions` at its nonzero entries.")
    mpo = build(pivots, 0, 0)
    bad = _hopping_mismatch(mpo, sites, samples)
    bad === nothing && return mpo
    i, j, got, want = bad
    error("hopping2MPO: the QTCI-compressed MPO is wrong at entry ($i, $j): $got instead " *
          "of $want, also when rebuilt from the nonzero sampled entries of f as pivots. " *
          "Pass `initial_positions` covering every bond class of f (hopping2MPO), or " *
          "`check=false` to accept the approximate MPO.")
end

# The QTCI build of hopping2MPO (what hopping2MPO runs without `check`).
function _hopping2MPO(f, N, sites; tol, initial_positions, type, unfoldingscheme,
                      nrandominitpivot, nsearchglobalpivot)
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

# The entries (i, j, f(i, j)) that hopping2MPO(...; check=true) compares: every row for
# N ≤ 64, else the rows 2^k and 2^k ± 1 (k = 1 … L-1; a bond out of row 2^k carries
# through k bits of the binary index, so these rows show every carry depth), the last two
# and an odd golden-ratio stride of `nbulk` rows through the bulk; in each row the
# columns j = i + d for the offsets d = 0, ±1, ±2, ±3, ±2^k, ±(2^k ± 1), which hold the
# bonds of chains and of row-major 2^Lx-wide grids (±1, ±Nx, ±(Nx ± 1), ±2Nx, …). f is
# called with Float64 arguments, as QTCI calls it on the grid of hopping2MPO.
function _hopping_samples(f, N; nbulk=16)
    L    = Int(log2(N))
    s    = 2 * round(Int, 0.30901699437494745 * N) + 1
    pow  = [2^k + δ for k in 1:L-1 for δ in (-1, 0, 1)]
    rows = N <= 64 ? collect(1:N) :
           unique([1; pow; N - 1; N; [1 + mod(k * s, N) for k in 0:nbulk-1]])
    offs = unique([0; [σ * d for d in [1; 2; 3; pow] for σ in (1, -1)]])
    return [(i, i + d, f(Float64(i), Float64(i + d))) for i in rows for d in offs
            if 1 <= i + d <= N]
end

# The first sampled entry that the MPO gets wrong, (i, j, got, want), or nothing: off by
# more than half of |f(i, j)| or by more than 1e-3 of the largest sampled |f|.
function _hopping_mismatch(mpo::MPO, sites, samples)
    maxabs = maximum(s -> abs(s[3]), samples; init=0.0)
    arrs   = _mpo_site_arrays(mpo, sites)
    for (i, j, want) in samples
        got = _mpo_array_entry(arrs, i, j)
        abs(got - want) <= max(0.5 * abs(want), 1e-3 * maxabs) || return (i, j, got, want)
    end
    return nothing
end

# Pivots for the rebuild of hopping2MPO(...; check=true): the nonzero sampled entries, at
# most `nmax` of them (evenly spread over the sample).
function _hopping_pivots(samples; nmax=512)
    nz = [(i, j) for (i, j, v) in samples if !iszero(v)]
    length(nz) <= nmax && return nz
    return nz[round.(Int, range(1, length(nz); length=nmax))]
end

# The site tensors of an MPO on the Qubit `sites` as arrays A[k][left, out, in, right]
# (several link indices on a bond merged), for _mpo_array_entry.
function _mpo_site_arrays(mpo::MPO, sites)
    n = length(mpo)
    links = [collect(commoninds(mpo[k], mpo[k + 1])) for k in 1:n - 1]
    return map(1:n) do k
        lk = k > 1 ? links[k - 1] : Index[]
        rk = k < n ? links[k] : Index[]
        s  = sites[k]
        reshape(Array(mpo[k], lk..., prime(s), s, rk...),
                prod(dim, lk; init=1), dim(s), dim(s), prod(dim, rk; init=1))
    end
end

# Entry (i, j) (1-based; row = primed index) of the MPO of `arrs`, site 1 the most
# significant bit.
function _mpo_array_entry(arrs, i::Integer, j::Integer)
    L = length(arrs)
    v = ones(ComplexF64, 1, 1)
    for k in 1:L
        b = L - k
        v = v * arrs[k][:, (((i - 1) >> b) & 1) + 1, (((j - 1) >> b) & 1) + 1, :]
    end
    return only(v)
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
# 3. General NNN 1D kinetic MPO (spatially varying hopping)
# ============================================================

"""
    kineticNNN(L, sites, hopping, nn; apply_kwargs=NamedTuple(), boundary=:open,
               bc=nothing) -> MPO

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
