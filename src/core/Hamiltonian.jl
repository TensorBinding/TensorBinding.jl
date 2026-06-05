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


"""
    kinetic_1d_nn_custom(L, sites, hopping; boundary=:open) -> MPO

Nearest-neighbour 1D kinetic MPO with a site-dependent hopping
encoded as a diagonal MPO `hopping`.  Useful for spatially varying
hopping amplitudes (e.g. SSH model, quasicrystals).
"""
function kinetic_1d_nn_custom(L, sites, hopping; boundary::Symbol=:open, bc=nothing)
    @assert L == length(sites) "L must equal length(sites)"
    bc === nothing || (boundary = Symbol(bc))
    return shift_hopping_mpo(hopping, sites, 1;
                             cyclic=_tb_periodic_boundary(boundary),
                             cutoff=1e-8)
end

# ============================================================
# General QTCI-based hopping MPO
# ============================================================

"""
    hopping2MPO(f, N, sites; tol=1e-8, initial_positions=[], type=Float64,
                unfoldingscheme=:interleaved) -> MPO

Compress an arbitrary NxN hopping matrix `H[i,j] = f(i,j)` into an
MPO using Quantics Tensor Cross Interpolation on a 2D quantics grid
(N must be a power of 2).

`unfoldingscheme` controls the bit ordering of the 2D quantics grid:
  - `:interleaved` (default): row and column bits alternate: r_L c_L ... r_1 c_1
  - `:fused`: all row bits first, then column bits: r_L ... r_1 c_L ... c_1

`initial_positions` seeds the TCI pivots; useful when the matrix has
known structure (e.g. near-diagonal for short-time propagators).
"""
function hopping2MPO(f, N, sites; tol=1e-8, initial_positions=[], type=Float64,
                     unfoldingscheme=:interleaved)
    L     = Int(log2(N))
    qgrid = QuanticsGrids.DiscretizedGrid{2}(
        L, (1, 1), (N, N);
        includeendpoint=true,
        unfoldingscheme=unfoldingscheme,
    )
    if length(initial_positions) >= 1
        initialpivots = [QuanticsGrids.origcoord_to_quantics(qgrid, pos)
                         for pos in initial_positions]
        ci, _, _ = quanticscrossinterpolate(type, f, qgrid;
                                            tolerance=tol,
                                            initialpivots=initialpivots)
    else
        ci, _, _ = quanticscrossinterpolate(type, f, qgrid; tolerance=tol)
    end
    citt = TensorCrossInterpolation.TensorTrain(ci.tci)
    mps  = MPS(citt) # modified from ITensors.MPS to MPS 
    println("MPS COMPUTED!")
    mpo  = unfoldingscheme == :fused ? fused_mpo(mps, sites) : custom_mpo(mps, sites)
    println("Turned into MPO!")
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
    println("got grid!")
    if !isempty(initial_positions)
        initialpivots = [QuanticsGrids.origcoord_to_quantics(qgrid, Float64.(pos))
                         for pos in initial_positions]
        ci, _, _ = quanticscrossinterpolate(type, A_fun, qgrid;
                                            tolerance=tol,
                                            initialpivots=initialpivots)
    else
        ci, _, _ = quanticscrossinterpolate(type, A_fun, qgrid; tolerance=tol)
    end
    println("got qtci!")
    citt = TensorCrossInterpolation.TensorTrain(ci.tci)
    mps  = ITensors.MPS(citt)
    println("got MPS!")
    mpo  = custom_mpo(mps, sites)
    println("got MPO!")
    ITensorMPS.truncate!(mpo; maxdim=20, cutoff=1e-8)
    return mpo
end

# ============================================================
# Specialised modulation functions
# ============================================================

"""
    quasicrystal_modulation_30deg(i, L, L_chain, k, p) -> Float64

On-site modulation for a p-fold quasicrystal pattern at wavevector k,
centred on the middle of the 2D lattice.
"""
function quasicrystal_modulation_30deg(i, L, L_chain, k, p)
    center   = 2^(L - 1) - L_chain / 2
    center_x = mod((center - 1), L_chain) + 0.5
    center_y = div(center - 1, L_chain) + 0.5
    x        = mod((i - 1), L_chain) + 0.5
    y        = div(i - 1, L_chain) + 0.5
    x_rel    = x - center_x
    y_rel    = y - center_y
    modulation = 0.0
    for n in 0:Int(p/2 - 1)
        theta     = 2pi * n / p
        r_proj    = x_rel * cos(theta) + y_rel * sin(theta)
        modulation += cos(k * r_proj)
    end
    return modulation
end


"""
    circular_mod(i, L, L_chain, k) -> Float64

Circularly symmetric on-site modulation `cos(k * r)` where `r` is the
distance from the centre of the 2D lattice.
"""
function circular_mod(i, L, L_chain, k)
    center   = 2^(L - 1) - L_chain / 2
    center_x = mod((center - 1), L_chain) + 0.5
    center_y = div(center - 1, L_chain) + 0.5
    x        = mod((i - 1), L_chain) + 0.5
    y        = div(i - 1, L_chain) + 0.5
    x_rel    = x - center_x
    y_rel    = y - center_y
    return cos(sqrt(x_rel^2 + y_rel^2) * k)
end


# ============================================================
# Fast diagonal MPO builder (via QTCI)
# ============================================================

"""
    qtt_mpo(L, xvals, sites, func; tol_quantics=1e-8, maxbonddim_quantics=50) -> MPO

Compress a scalar function `func(x)` evaluated on the explicit integer grid `xvals`
(typically `0:2^L-1`) into a **diagonal MPO** via Quantics Tensor Cross Interpolation.

The result is `diag(func(0), func(1), ..., func(2^L-1))` stored as an L-site MPO.
Use this to encode spatially varying on-site potentials or hopping amplitudes as
diagonal MPOs for use with `kineticNNN` and the 2D kinetic builders.

`xvals = 0:2^L-1`        for a 1D chain of 2^L sites
`xvals = 0:Nx*Ny-1`      for a row-major flattened 2D grid

See also `get_diagonal_mpo` in utils.jl for a simpler 1-based-index wrapper.
"""
function qtt_mpo(L, xvals, sites, func;
                 tol_quantics::Real    = 1e-8,
                 maxbonddim_quantics::Int = 50)
    qtt = QuanticsTCI.quanticscrossinterpolate(ComplexF64, func, xvals;
              tolerance=tol_quantics, maxbonddim=maxbonddim_quantics)[1]
    tt  = TCI.tensortrain(qtt.tci)
    mps = MPS(tt; sites)
    mpo = outer(mps', mps)
    for s in 1:L
        mpo.data[s] = Quantics._asdiagonal(mps.data[s], sites[s])
    end
    return mpo
end


# ============================================================
# Exponentiation-by-squaring for MPO composition
# ============================================================

"""
    compose_power(base, nn; side=:right, apply_kwargs=NamedTuple()) -> MPO

Compose `base` with itself `nn` times using **exponentiation-by-squaring** (O(log n) applies).
Replaces the old `arbitarty_offline` helper which used O(n) sequential applies.

- `side = :right`: `acc = apply(acc, base)` at each set bit
- `side = :left`: `acc = apply(base, acc)` at each set bit

`apply_kwargs` (e.g. `(; cutoff=1e-8, maxdim=200)`) are forwarded to every `apply` call.
`nn = 0` returns the identity MPO; `nn = 1` returns `base` unchanged.
"""
function compose_power(base::MPO, nn::Integer;
                       side::Symbol    = :right,
                       apply_kwargs    = NamedTuple())
    @assert nn >= 0 "nn must be non-negative"
    nn == 0 && return MPO(siteinds(base), "Id")
    nn == 1 && return base
    acc = nothing
    cur = base
    k   = nn
    while k > 0
        if (k & 1) == 1
            acc = acc === nothing ? cur :
                  side === :right ? apply(acc, cur; apply_kwargs...) :
                                    apply(cur, acc; apply_kwargs...)
        end
        k >>>= 1
        k > 0 && (cur = apply(cur, cur; apply_kwargs...))
    end
    return acc::MPO
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
