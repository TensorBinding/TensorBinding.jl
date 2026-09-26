# hopping2d.jl — kinetic/hopping MPO builders for 2D lattice geometries: binary
# shift MPOs, square-lattice NN hoppings and the NNN kinetic builders.
# Split from the former lattice/2Dlattice_tk.jl; the masks they apply live in masks2d.jl.
#
# Provides hopping MPOs for square, triangular, and honeycomb lattices
# built from the quantics binary representation.
#
# Encoding convention (row-major):
#   linear index  n = ix + iy * 2^Lx
#   site ordering: sites 1..Ly hold iy bits (MSB first),
#                  sites Ly+1..L hold ix bits (MSB first).
#
# compose_power lives in core/MPOTools.jl; low-level utilities
# (to_binary_vector, binary_to_MPS) live in core/Utils.jl.

# ============================================================
# 1. Binary shift MPOs
# ============================================================

"""
    generate_kin_u(sites, num_site) -> MPO

Binary-increment MPO: |n⟩ → |n+1⟩ (mod 2^L) on L = log2(num_site) qubits.
Built as `shift_mpo(sites, 1; cyclic=true)`.
"""
function generate_kin_u(sites, num_site)
    L  = Int(log2(num_site))
    @assert L == length(sites) "num_site must match the number of qubit sites"
    return shift_mpo(sites, 1; cyclic=true)
end


"""
    generate_kin_d(sites, num_site) -> MPO

Binary-decrement MPO: |n⟩ → |n-1⟩ (mod 2^L). Hermitian conjugate of
`generate_kin_u`, built as `shift_mpo(sites, -1; cyclic=true)`.
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
    intrachain_hopping(L_chain, num_site, sites; hopping=Id, t=1) -> MPO

NN hopping along rows (x-direction) of a 2D lattice with `L_chain` sites per
row.  Hops that would wrap ix = Nx-1 → 0 are suppressed by `_row_break_mpo`.
"""
function intrachain_hopping(L_chain, num_site, sites;
                            hopping=MPO(sites, "Id"), t=1)
    Lx  = Int(log2(L_chain))
    Ly  = Int(log2(num_site)) - Lx
    brk = _row_break_mpo(Lx, Ly, sites; which=:xplus)
    K   = shift_mpo(sites, 1; cyclic=false)
    Kd  = shift_adjoint_mpo(K)
    hop_fwd = apply(apply(K, brk), hopping)
    hop_bwd = apply(apply(hopping, Kd), brk)
    return +(t * hop_fwd, conj(t) * hop_bwd; cutoff=1e-8)
end


"""
    interchain_hopping_square(L_chain, num_site, sites; hopping=Id, t=1) -> MPO

NN hopping along columns (y-direction) of a square lattice.
One column step = linear shift by L_chain sites = ku composed L_chain times.
"""
function interchain_hopping_square(L_chain, num_site, sites;
                                   hopping=MPO(sites, "Id"), t=1)
    K       = shift_mpo(sites, L_chain; cyclic=false)
    Kd      = shift_adjoint_mpo(K)
    hop_fwd = apply(hopping, K)
    hop_bwd = apply(Kd, hopping)
    return t * hop_fwd + conj(t) * hop_bwd
end


# ============================================================
# 3. NNN 2D kinetic builders
#    Pattern for every function:
#      1. Build K, Kdag = shift_pair_mpos(sites, nn) (or one shift_mpo)
#      2. Apply hopping weights: hop_fwd = h * K,  hop_bwd = Kdag * dag(h)
#      3. Mask with _row_break_mpo and optionally _row_select/_checker
# ============================================================

"""
    kineticintra2DNNN(Lx, Ly, sites, hopping, nn; apply_kwargs) -> MPO

Long-range intra-row hopping on a `2^Lx × 2^Ly` square lattice (nn bonds
along x).  Row wrap-around at ix = Nx-1 is suppressed by `_row_break_mpo(:xplus)`.
"""
function kineticintra2DNNN(Lx, Ly, sites, hopping::MPO, nn::Integer; apply_kwargs = NamedTuple())
    L = Lx + Ly
    @assert L == length(sites) && nn >= 1
    K, Kdag = shift_pair_mpos(sites, nn; cyclic=false)
    brk = _row_break_mpo(Lx, Ly, sites; which=:xplus)
    hop_fwd = apply(apply(hopping, K; apply_kwargs...), brk; apply_kwargs...)
    hop_bwd = apply(brk, apply(Kdag, dag(hopping); apply_kwargs...); apply_kwargs...)
    return +(hop_fwd, hop_bwd; cutoff=1e-12)
end


"""
    kineticinterNNNSWNE(Lx, Ly, sites, hopping, nn; apply_kwargs) -> MPO

Long-range inter-row hopping along the SW↗NE diagonal of a `2^Lx × 2^Ly`
square lattice.  Row end-wrap suppressed by `_row_break_mpo(:xplus)`.
"""
function kineticinterNNNSWNE(Lx, Ly, sites, hopping::MPO, nn::Integer; apply_kwargs = NamedTuple())
    L = Lx + Ly
    @assert L == length(sites) && nn >= 1
    K, Kdag = shift_pair_mpos(sites, nn; cyclic=false)
    brk = _row_break_mpo(Lx, Ly, sites; which=:xplus)
    hop_fwd = apply(apply(hopping, K; apply_kwargs...), brk; apply_kwargs...)
    hop_bwd = apply(brk, apply(Kdag, dag(hopping); apply_kwargs...); apply_kwargs...)
    return +(hop_fwd, hop_bwd; cutoff=1e-12)
end


"""
    kineticinterNNNSENW(Lx, Ly, sites, hopping, nn; apply_kwargs) -> MPO

Long-range inter-row hopping along the SE↖NW diagonal.
Row start-wrap suppressed by `_row_break_mpo(:xplain)`.
"""
function kineticinterNNNSENW(Lx, Ly, sites, hopping::MPO, nn::Integer; apply_kwargs = NamedTuple())
    L = Lx + Ly
    @assert L == length(sites) && nn >= 1
    K, Kdag = shift_pair_mpos(sites, nn; cyclic=false)
    brk = _row_break_mpo(Lx, Ly, sites; which=:xplain)
    hop_fwd = apply(apply(hopping, K; apply_kwargs...), brk; apply_kwargs...)
    hop_bwd = apply(brk, apply(Kdag, dag(hopping); apply_kwargs...); apply_kwargs...)
    return +(hop_fwd, hop_bwd; cutoff=1e-12)
end


"""
    kineticinterNNNtriSWNE(Lx, Ly, sites, hopping, nn; apply_kwargs) -> MPO

SW↗NE diagonal inter-row hopping for a triangular lattice.
Applies `_row_break_mpo(:xplus)` and `_row_select_mpo(:even)` to restrict
hops to the correct sublattice rows.
"""
function kineticinterNNNtriSWNE(Lx, Ly, sites, hopping::MPO, nn::Integer; apply_kwargs = NamedTuple())
    L = Lx + Ly
    @assert L == length(sites) && nn >= 1
    K, Kdag = shift_pair_mpos(sites, nn; cyclic=false)
    brk = _row_break_mpo(Lx, Ly, sites; which=:xplus)
    sel = _row_select_mpo(Lx, Ly, sites; keep=:even)
    src = apply(brk, sel; apply_kwargs...)
    hop_fwd = apply(apply(hopping, K; apply_kwargs...), src; apply_kwargs...)
    hop_bwd = apply(src, apply(Kdag, dag(hopping); apply_kwargs...); apply_kwargs...)
    return +(hop_fwd, hop_bwd; cutoff=1e-12)
end


"""
    kineticinterNNNtriSENW(Lx, Ly, sites, hopping, nn; apply_kwargs) -> MPO

SE↖NW diagonal inter-row hopping for a triangular lattice.
Applies `_row_break_mpo(:xplain)` and `_row_select_mpo(:odd)`.
"""
function kineticinterNNNtriSENW(Lx, Ly, sites, hopping::MPO, nn::Integer; apply_kwargs = NamedTuple())
    L = Lx + Ly
    @assert L == length(sites) && nn >= 1
    K, Kdag = shift_pair_mpos(sites, nn; cyclic=false)
    brk = _row_break_mpo(Lx, Ly, sites; which=:xplain)
    sel = _row_select_mpo(Lx, Ly, sites; keep=:odd)
    src = apply(brk, sel; apply_kwargs...)
    hop_fwd = apply(apply(hopping, K; apply_kwargs...), src; apply_kwargs...)
    hop_bwd = apply(src, apply(Kdag, dag(hopping); apply_kwargs...); apply_kwargs...)
    return +(hop_fwd, hop_bwd; cutoff=1e-12)
end


"""
    kineticinterNNNtri_bravais_diag(Lx, Ly, sites, hopping; apply_kwargs) -> MPO

Bravais triangular-lattice third-bond hopping: (Δix=+1, Δiy=−1), linear shift 1−Nx.
Mirrors `kineticinterNNNSWNE` with kd/ku swapped.  Row x-wrap at ix=Nx−1 is
suppressed by `_row_break_mpo(:xplus)`.
"""
function kineticinterNNNtri_bravais_diag(Lx, Ly, sites, hopping::MPO;
                                          apply_kwargs = NamedTuple())
    L  = Lx + Ly
    Nx = 2^Lx
    @assert L == length(sites)
    K = shift_mpo(sites, -(Nx - 1); cyclic=false)
    Kdag = shift_adjoint_mpo(K)
    brk = _row_break_mpo(Lx, Ly, sites; which=:xplus)
    hop_fwd = apply(apply(hopping, K; apply_kwargs...), brk; apply_kwargs...)
    hop_bwd = apply(brk, apply(Kdag, dag(hopping); apply_kwargs...); apply_kwargs...)
    return +(hop_fwd, hop_bwd; cutoff=1e-12)
end


"""
    kineticintra2DNNhex(Lx, Ly, sites, hopping, nn; apply_kwargs) -> MPO

Intra-row hopping for a honeycomb lattice.  Applies `_row_break_mpo(:xplus)`
and `_row_checker_mpo` to implement the alternating A/B sublattice pattern.
"""
function kineticintra2DNNhex(Lx, Ly, sites, hopping::MPO, nn::Integer; apply_kwargs = NamedTuple())
    L = Lx + Ly
    K, Kdag = shift_pair_mpos(sites, nn; cyclic=false)
    brk = _row_break_mpo(Lx, Ly, sites; which=:xplus)
    chk = _row_checker_mpo(Lx, Ly, sites)
    src = apply(brk, chk; apply_kwargs...)
    hop_fwd = apply(apply(hopping, K; apply_kwargs...), src; apply_kwargs...)
    hop_bwd = apply(src, apply(Kdag, dag(hopping); apply_kwargs...); apply_kwargs...)
    return +(hop_fwd, hop_bwd; cutoff=1e-12)
end
