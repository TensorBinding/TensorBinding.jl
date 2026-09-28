# hopping2d.jl — kinetic (hopping) MPO builders for 2D lattice geometries, built from
# the quantics binary representation: the binary shift MPOs, the square-lattice NN
# hoppings and the long-range (NNN) kinetic builders for the square, triangular and
# honeycomb lattices, the latter thin wrappers over one kernel (masked_shift_hopping).
#
# Entry points: generate_kin_u, generate_kin_d, intrachain_hopping,
#   interchain_hopping_square, masked_shift_hopping, kineticintra2DNNN,
#   kineticinterNNNSWNE, kineticinterNNNSENW, kineticinterNNNtriSWNE,
#   kineticinterNNNtriSENW, kineticinterNNNtri_bravais_diag, kineticintra2DNNhex.
#
# Depends on: core/Utils.jl (shift_mpo, shift_pair_mpos, shift_adjoint_mpo) and
# lattice/masks2d.jl (the row-break, row-select and checkerboard masks).
#
# Encoding convention (row-major):
#   linear index  n = ix + iy * 2^Lx
#   site ordering: sites 1..Ly hold iy bits (MSB first),
#                  sites Ly+1..L hold ix bits (MSB first).
#
# Split from the former lattice/2Dlattice_tk.jl.

# ============================================================
# 1. Binary shift MPOs
# ============================================================

"""
    generate_kin_u(sites, num_site) -> MPO

Binary-increment MPO: |n⟩ → |n+1⟩ (mod 2^L) on L = log2(num_site) qubits.
Built as `shift_mpo(sites, 1; cyclic=true)`, after checking that `num_site` is
`2^length(sites)`. The lattice builders call `shift_mpo` directly where that check
cannot fail; `add_hopping_2D!` keeps `generate_kin_u/d`, because the check is the
only thing there that rejects a Hamiltonian on a projected (non-binary) position space.
"""
function generate_kin_u(sites, num_site)
    L  = Int(log2(num_site))
    @assert L == length(sites) "num_site must match the number of qubit sites"
    return shift_mpo(sites, 1; cyclic=true)
end


"""
    generate_kin_d(sites, num_site) -> MPO

Binary-decrement MPO: |n⟩ → |n-1⟩ (mod 2^L). Hermitian conjugate of
`generate_kin_u`, built as `shift_mpo(sites, -1; cyclic=true)` after the same check.
"""
function generate_kin_d(sites, num_site)
    L  = Int(log2(num_site))
    @assert L == length(sites) "num_site must match the number of qubit sites"
    return shift_mpo(sites, -1; cyclic=true)
end


# ============================================================
# 2. Square lattice
# ============================================================

"""
    intrachain_hopping(L_chain, num_site, sites; hopping=MPO(sites, "Id"), t=1) -> MPO

NN hopping along rows (x-direction) of a 2D lattice with `L_chain` sites per
row: `t · K·B·W + conj(t) · W†·B·K†`, with `K` the +1 shift, `B` the row break
`_row_break_mpo(:xplus)` (no hop from ix = Nx-1 to the next row) and `W` the diagonal
`hopping` profile (weights at the source of the forward hop). Hermitian.
"""
function intrachain_hopping(L_chain, num_site, sites;
                            hopping=MPO(sites, "Id"), t=1)
    Lx  = Int(log2(L_chain))
    Ly  = Int(log2(num_site)) - Lx
    brk = _row_break_mpo(Lx, Ly, sites; which=:xplus)
    K   = shift_mpo(sites, 1; cyclic=false)
    Kd  = shift_adjoint_mpo(K)
    hop_fwd = apply(apply(K, brk), hopping)
    # (K·B·W)† = W†·B·K†: the break on the destination side of the backward hop.
    hop_bwd = apply(apply(dag(hopping), brk), Kd)
    return +(t * hop_fwd, conj(t) * hop_bwd; cutoff=1e-8)
end


"""
    interchain_hopping_square(L_chain, num_site, sites; hopping=MPO(sites, "Id"), t=1) -> MPO

NN hopping along columns (y-direction) of a square lattice:
`t · W·K + conj(t) · K†·W†`, with `K` the shift by `L_chain` (one column step) and `W`
the diagonal `hopping` profile. Hermitian.
"""
function interchain_hopping_square(L_chain, num_site, sites;
                                   hopping=MPO(sites, "Id"), t=1)
    K       = shift_mpo(sites, L_chain; cyclic=false)
    Kd      = shift_adjoint_mpo(K)
    hop_fwd = apply(hopping, K)
    hop_bwd = apply(Kd, dag(hopping))
    return t * hop_fwd + conj(t) * hop_bwd
end


# ============================================================
# 3. NNN 2D kinetic builders
# ============================================================

# Every builder of this section is one masked_shift_hopping call: a shift by q
# with open boundaries, the hopping weights on the destination side and a source
# mask that removes the bonds crossing a row boundary (plus, on the triangular and
# honeycomb lattices, a row or checkerboard filter). The builders differ only in
# q and the mask, and keep their own argument checks.

"""
    masked_shift_hopping(Lx, Ly, sites, hopping, q; src_mask,
                         apply_kwargs=NamedTuple()) -> MPO

Hermitian pair of shift hoppings on a `2^Lx × 2^Ly` grid (row-major encoding),

    (hopping · K) · M  +  M · (K† · hopping†),    K = shift_mpo(sites, q; cyclic=false),

summed at `cutoff=1e-12`, where the source mask `M` is given by `src_mask`:

- `:xplus` / `:xplain`: `_row_break_mpo(Lx, Ly, sites; which=src_mask)`, which zeroes
  the sources at the end / start of each row;
- `:even` / `:odd`: `_row_select_mpo(Lx, Ly, sites; keep=src_mask)`;
- `:checker`: `_row_checker_mpo(Lx, Ly, sites)`;
- a tuple of these, e.g. `(:xplus, :even)`: their product, taken left to right;
- an `MPO`, used as given.

`apply_kwargs` go to every `apply` (the mask products included). This is the kernel of
`kineticintra2DNNN`, `kineticinterNNNSWNE`, `kineticinterNNNSENW`,
`kineticinterNNNtriSWNE`, `kineticinterNNNtriSENW`, `kineticinterNNNtri_bravais_diag`
and `kineticintra2DNNhex`.
"""
function masked_shift_hopping(Lx, Ly, sites, hopping::MPO, q::Integer;
                              src_mask, apply_kwargs = NamedTuple())
    K, Kdag = shift_pair_mpos(sites, q; cyclic=false)
    src = _source_mask_mpo(Lx, Ly, sites, src_mask, apply_kwargs)
    hop_fwd = apply(apply(hopping, K; apply_kwargs...), src; apply_kwargs...)
    hop_bwd = apply(src, apply(Kdag, dag(hopping); apply_kwargs...); apply_kwargs...)
    return +(hop_fwd, hop_bwd; cutoff=1e-12)
end

# The source mask of masked_shift_hopping from its `src_mask` description.
_source_mask_mpo(Lx, Ly, sites, M::MPO, apply_kwargs) = M

function _source_mask_mpo(Lx, Ly, sites, name::Symbol, apply_kwargs)
    name in (:even, :odd) && return _row_select_mpo(Lx, Ly, sites; keep=name)
    name === :checker     && return _row_checker_mpo(Lx, Ly, sites)
    return _row_break_mpo(Lx, Ly, sites; which=name)
end

function _source_mask_mpo(Lx, Ly, sites, names::Tuple, apply_kwargs)
    masks = [_source_mask_mpo(Lx, Ly, sites, name, apply_kwargs) for name in names]
    return foldl((a, b) -> apply(a, b; apply_kwargs...), masks)
end


"""
    kineticintra2DNNN(Lx, Ly, sites, hopping, nn; apply_kwargs=NamedTuple()) -> MPO

Long-range intra-row hopping on a `2^Lx × 2^Ly` square lattice (nn bonds
along x).  Row wrap-around at ix = Nx-1 is suppressed by `_row_break_mpo(:xplus)`.
"""
function kineticintra2DNNN(Lx, Ly, sites, hopping::MPO, nn::Integer; apply_kwargs = NamedTuple())
    L = Lx + Ly
    @assert L == length(sites) && nn >= 1
    return masked_shift_hopping(Lx, Ly, sites, hopping, nn; src_mask=:xplus, apply_kwargs)
end


"""
    kineticinterNNNSWNE(Lx, Ly, sites, hopping, nn; apply_kwargs=NamedTuple()) -> MPO

Long-range inter-row hopping along the SW↗NE diagonal of a `2^Lx × 2^Ly`
square lattice.  Row end-wrap suppressed by `_row_break_mpo(:xplus)`.
"""
function kineticinterNNNSWNE(Lx, Ly, sites, hopping::MPO, nn::Integer; apply_kwargs = NamedTuple())
    L = Lx + Ly
    @assert L == length(sites) && nn >= 1
    return masked_shift_hopping(Lx, Ly, sites, hopping, nn; src_mask=:xplus, apply_kwargs)
end


"""
    kineticinterNNNSENW(Lx, Ly, sites, hopping, nn; apply_kwargs=NamedTuple()) -> MPO

Long-range inter-row hopping along the SE↖NW diagonal.
Row start-wrap suppressed by `_row_break_mpo(:xplain)`.
"""
function kineticinterNNNSENW(Lx, Ly, sites, hopping::MPO, nn::Integer; apply_kwargs = NamedTuple())
    L = Lx + Ly
    @assert L == length(sites) && nn >= 1
    return masked_shift_hopping(Lx, Ly, sites, hopping, nn; src_mask=:xplain, apply_kwargs)
end


"""
    kineticinterNNNtriSWNE(Lx, Ly, sites, hopping, nn; apply_kwargs=NamedTuple()) -> MPO

SW↗NE diagonal inter-row hopping for a triangular lattice.
Applies `_row_break_mpo(:xplus)` and `_row_select_mpo(:even)` to restrict
hops to the correct sublattice rows.
"""
function kineticinterNNNtriSWNE(Lx, Ly, sites, hopping::MPO, nn::Integer; apply_kwargs = NamedTuple())
    L = Lx + Ly
    @assert L == length(sites) && nn >= 1
    return masked_shift_hopping(Lx, Ly, sites, hopping, nn; src_mask=(:xplus, :even), apply_kwargs)
end


"""
    kineticinterNNNtriSENW(Lx, Ly, sites, hopping, nn; apply_kwargs=NamedTuple()) -> MPO

SE↖NW diagonal inter-row hopping for a triangular lattice.
Applies `_row_break_mpo(:xplain)` and `_row_select_mpo(:odd)`.
"""
function kineticinterNNNtriSENW(Lx, Ly, sites, hopping::MPO, nn::Integer; apply_kwargs = NamedTuple())
    L = Lx + Ly
    @assert L == length(sites) && nn >= 1
    return masked_shift_hopping(Lx, Ly, sites, hopping, nn; src_mask=(:xplain, :odd), apply_kwargs)
end


"""
    kineticinterNNNtri_bravais_diag(Lx, Ly, sites, hopping; apply_kwargs=NamedTuple()) -> MPO

Bravais triangular-lattice third-bond hopping: (Δix=+1, Δiy=−1), linear shift 1−Nx.
Mirrors `kineticinterNNNSWNE` with kd/ku swapped.  Row x-wrap at ix=Nx−1 is
suppressed by `_row_break_mpo(:xplus)`.
"""
function kineticinterNNNtri_bravais_diag(Lx, Ly, sites, hopping::MPO;
                                          apply_kwargs = NamedTuple())
    L  = Lx + Ly
    Nx = 2^Lx
    @assert L == length(sites)
    return masked_shift_hopping(Lx, Ly, sites, hopping, -(Nx - 1); src_mask=:xplus, apply_kwargs)
end


"""
    kineticintra2DNNhex(Lx, Ly, sites, hopping, nn; apply_kwargs=NamedTuple()) -> MPO

Intra-row hopping for a honeycomb lattice.  Applies `_row_break_mpo(:xplus)`
and `_row_checker_mpo` to implement the alternating A/B sublattice pattern.
"""
function kineticintra2DNNhex(Lx, Ly, sites, hopping::MPO, nn::Integer; apply_kwargs = NamedTuple())
    return masked_shift_hopping(Lx, Ly, sites, hopping, nn; src_mask=(:xplus, :checker), apply_kwargs)
end
