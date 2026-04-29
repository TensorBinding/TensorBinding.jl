# 2D_lattice.jl — MPO building blocks for 2D lattice geometries
#
# Provides hopping MPOs for square, triangular, and honeycomb lattices
# built from the quantics binary representation.
#
# Encoding convention (row-major):
#   linear index  n = ix + iy * 2^Lx
#   site ordering: sites 1..Ly hold iy bits (MSB first),
#                  sites Ly+1..L hold ix bits (MSB first).
#
# compose_power lives in Hamiltonian.jl; low-level utilities
# (to_binary_vector, binary_to_MPS) live in utils.jl.

# ============================================================
# 1. Single-qubit projectors and binary shift MPOs
# ============================================================

ITensors.op(::OpName"sigma_d",::SiteType"Qubit") = [0 0; 0 1]   # |1><1|
ITensors.op(::OpName"sigma_u",::SiteType"Qubit") = [1 0; 0 0]   # |0><0|


"""
    generate_kin_u(sites, num_site) -> MPO

Binary-increment MPO: |n⟩ → |n+1⟩ (mod 2^L) on L = log2(num_site) qubits.
Each term i handles one carry level: sigma_plus at bit i, sigma_minus on all
lower bits (the bits that were 1 and get reset by the carry).
"""
function generate_kin_u(sites, num_site)
    L  = Int(log2(num_site))
    os = OpSum()
    for i in 1:L                               # i = 1 is LSB, i = L is MSB
        term  = OpSum()
        term += 1, "sigma_plus",  L - (i-1)   # flip bit i: 0 → 1
        for j in 1:L-i;   term *= ("Id",          j); end
        for j in L+2-i:L; term *= ("sigma_minus",  j); end  # reset lower bits (carry-in)
        os += term
    end
    return MPO(os, sites)
end


"""
    generate_kin_d(sites, num_site) -> MPO

Binary-decrement MPO: |n⟩ → |n-1⟩ (mod 2^L). Hermitian conjugate of
`generate_kin_u`; each term handles one borrow level.
"""
function generate_kin_d(sites, num_site)
    L  = Int(log2(num_site))
    os = OpSum()
    for i in 1:L
        term  = OpSum()
        term += 1, "sigma_minus", L - (i-1)   # flip bit i: 1 → 0
        for j in 1:L-i;   term *= ("Id",         j); end
        for j in L+2-i:L; term *= ("sigma_plus",  j); end  # set lower bits (borrow-in)
        os += term
    end
    return MPO(os, sites)
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
    ku  = generate_kin_u(sites, num_site)
    kd  = generate_kin_d(sites, num_site)
    # brk placed opposite the shift direction to kill the wrap-around bond
    hop_fwd = apply(brk, apply(ku, hopping))
    hop_bwd = apply(apply(hopping, kd), brk)
    return +(t * hop_fwd, conj(t) * hop_bwd; cutoff=1e-8)
end


"""
    interchain_hopping_square(L_chain, num_site, sites; hopping=Id, t=1) -> MPO

NN hopping along columns (y-direction) of a square lattice.
One column step = linear shift by L_chain sites = ku composed L_chain times.
"""
function interchain_hopping_square(L_chain, num_site, sites;
                                   hopping=MPO(sites, "Id"), t=1)
    ku      = generate_kin_u(sites, num_site)
    kd      = generate_kin_d(sites, num_site)
    hop_fwd = apply(apply(hopping, ku), compose_power(ku, L_chain - 1))
    hop_bwd = apply(compose_power(kd, L_chain - 1), apply(kd, hopping))
    return t * hop_fwd + conj(t) * hop_bwd
end


"""
    interchain_hopping_square_2nd_plus(L_chain, num_site, sites; hopping=Id, t2=1) -> MPO

NNN hopping in the (+x,+y) diagonal direction (linear shift +L_chain+1).
The single x-step is masked to prevent row wrap-around.
"""
function interchain_hopping_square_2nd_plus(L_chain, num_site, sites;
                                            hopping=MPO(sites, "Id"), t2=1)
    Lx  = Int(log2(L_chain))
    Ly  = Int(log2(num_site)) - Lx
    brk = _row_break_mpo(Lx, Ly, sites; which=:xplus)
    ku  = generate_kin_u(sites, num_site)
    kd  = generate_kin_d(sites, num_site)
    # one masked x-step then L_chain y-steps
    hop_fwd = apply(compose_power(ku, L_chain), apply(hopping, apply(brk, ku)))
    hop_bwd = apply(compose_power(kd, L_chain), apply(apply(kd, brk), hopping))
    return t2 * hop_fwd + conj(t2) * hop_bwd
end


"""
    interchain_hopping_square_2nd_minus(L_chain, num_site, sites; hopping=Id, t2=1) -> MPO

NNN hopping in the (+x,−y) diagonal direction (linear shift −(L_chain−1)).
The single x-step is masked to prevent row wrap-around.
"""
function interchain_hopping_square_2nd_minus(L_chain, num_site, sites;
                                             hopping=MPO(sites, "Id"), t2=1)
    Lx  = Int(log2(L_chain))
    Ly  = Int(log2(num_site)) - Lx
    brk = _row_break_mpo(Lx, Ly, sites; which=:xplus)
    ku  = generate_kin_u(sites, num_site)
    kd  = generate_kin_d(sites, num_site)
    hop_fwd = apply(apply(hopping, apply(brk, ku)), compose_power(ku, L_chain - 2))
    hop_bwd = apply(compose_power(kd, L_chain - 2), apply(apply(kd, brk), hopping))
    return t2 * hop_fwd + conj(t2) * hop_bwd
end


# ============================================================
# 3. Triangular lattice
# ============================================================

"""
    skeleton(L_chain, num_site, sites) -> MPO

Diagonal mask: 0 where ix == 1 (second column, 0-indexed), 1 elsewhere.
Used in the triangular lattice to exclude the wrap-around bond that enters
at ix = 1 when shifting by L_chain − 1 sites.
"""
function skeleton(L_chain, num_site, sites)
    L     = Int(log2(num_site))
    Ly    = L - Int(log2(L_chain))
    Id_op = MPO(sites, "Id")
    # proj_{ix=1}: LSB of ix is 1, all higher x-bits are 0
    os = OpSum()
    os += 1, "sigma_d", L
    for i in Ly+1:L-1; os *= 1, "sigma_u", i; end
    return Id_op - MPO(os, sites)
end


"""
    interchain_hopping_triangle(L_chain, num_site, sites) -> MPO

Inter-row hopping for a triangular lattice.  Two diagonal bonds:
- SW→NE (shift +L_chain): pure row hop, no masking needed.
- SE→NW (shift +L_chain−1): row hop + one step back in x; `skeleton`
  suppresses the spurious ix=1 wrap-around entry.
"""
function interchain_hopping_triangle(L_chain, num_site, sites)
    ku   = generate_kin_u(sites, num_site)
    kd   = generate_kin_d(sites, num_site)
    skel = skeleton(L_chain, num_site, sites)

    hop_swne_fwd = compose_power(ku, L_chain)
    hop_swne_bwd = compose_power(kd, L_chain)
    hop_senw_fwd = apply(skel, compose_power(ku, L_chain - 1))
    hop_senw_bwd = apply(compose_power(kd, L_chain - 1), skel)

    return hop_swne_fwd + hop_swne_bwd + hop_senw_fwd + hop_senw_bwd
end


# ============================================================
# 4. Honeycomb lattice
# ============================================================

"""
    odd_template(_, num_site, sites) -> MPO

Diagonal mask: 1 where ix is odd (LSB of ix = 1).  Selects the A-sublattice
columns within each row for the honeycomb inter-row bonds.
"""
function odd_template(::Any, num_site, sites)
    L  = Int(log2(num_site))
    os = OpSum()
    os += 1, "sigma_d", L    # site L is LSB of ix
    return MPO(os, sites)
end

"""
    even_template(L_chain, num_site, sites) -> MPO

Diagonal mask: 1 where ix is odd AND ix ≠ 1.  Selects B-sublattice columns
(odd ix, excluding the boundary ix = 1 column which wraps into the next row).
Computed as proj_{odd ix} − proj_{ix=1}.
"""
function even_template(L_chain, num_site, sites)
    L  = Int(log2(num_site))
    Ly = L - Int(log2(L_chain))
    os_odd = OpSum(); os_odd += 1, "sigma_d", L
    os_one = OpSum(); os_one += 1, "sigma_d", L
    for i in Ly+1:L-1; os_one *= 1, "sigma_u", i; end  # proj_{ix=1}: higher x-bits = 0
    return MPO(os_odd, sites) - MPO(os_one, sites)
end

"""
    odd_skeleton(L_chain, num_site, sites) -> MPO

Diagonal mask: 1 where iy is even (0-based, LSB of iy = 0).
Selects the even rows for the upper honeycomb inter-row bond.
"""
function odd_skeleton(L_chain, num_site, sites)
    L  = Int(log2(num_site))
    Ly = L - Int(log2(L_chain))
    os = OpSum()
    os += 1, "sigma_u", Ly   # site Ly is LSB of iy; = 0 → even row
    return MPO(os, sites)
end

"""
    even_skeleton(L_chain, num_site, sites) -> MPO

Diagonal mask: 1 where iy is odd (0-based, LSB of iy = 1).
Selects the odd rows for the lower honeycomb inter-row bond.
"""
function even_skeleton(L_chain, num_site, sites)
    L  = Int(log2(num_site))
    Ly = L - Int(log2(L_chain))
    os = OpSum()
    os += 1, "sigma_d", Ly   # site Ly is LSB of iy; = 1 → odd row
    return MPO(os, sites)
end


"""
    interchain_hopping_honeycomb(L_chain, num_site, sites) -> MPO

Inter-row hopping for a honeycomb lattice with `L_chain` sites per row.
Two inequivalent inter-row bonds, each with a forward and backward term:
- Upper bond (shift +L_chain+1): connectivity mask = odd_template * odd_skeleton
- Lower bond (shift +L_chain−1): connectivity mask = even_template * even_skeleton
"""
function interchain_hopping_honeycomb(L_chain, num_site, sites)
    ku = generate_kin_u(sites, num_site)
    kd = generate_kin_d(sites, num_site)

    mask_up = apply(odd_template( L_chain, num_site, sites),
                    odd_skeleton( L_chain, num_site, sites))
    mask_dn = apply(even_template(L_chain, num_site, sites),
                    even_skeleton(L_chain, num_site, sites))

    hop_up_fwd = apply(mask_up, compose_power(ku, L_chain + 1))
    hop_up_bwd = apply(compose_power(kd, L_chain + 1), mask_up)
    hop_dn_fwd = apply(mask_dn, compose_power(ku, L_chain - 1))
    hop_dn_bwd = apply(compose_power(kd, L_chain - 1), mask_dn)

    return hop_up_fwd + hop_up_bwd + hop_dn_fwd + hop_dn_bwd
end


# ============================================================
# 5. Row/column/checkerboard mask MPOs (diagonal, exact)
#    Bit layout: sites 1..Ly → iy (MSB first), sites Ly+1..L → ix (MSB first)
# ============================================================

"""
    _row_break_mpo(Lx, Ly, sites; which) -> MPO

Diagonal mask that zeroes wrap-around couplings at row boundaries of a
`2^Lx × 2^Ly` grid (row-major encoding).

- `which = :xplus`  → 0 where ix == 2^Lx − 1  (end of each row)
- `which = :xplain` → 0 where ix == 0           (start of each row)

Multiply a kinetic MPO by this mask on the appropriate side to suppress the
bond that crosses a row boundary.
"""
function _row_break_mpo(Lx, Ly, sites; which::Symbol)
    L     = Lx + Ly
    Id_op = MPO(sites, "Id")
    proj  = which === :xplus  ? "sigma_d" :
            which === :xplain ? "sigma_u" :
            error("unknown which=:$(which); use :xplus or :xplain")
    # projector onto ix = Nx-1 (:xplus) or ix = 0 (:xplain):
    # product of proj on all Lx x-bit sites (Ly+1 .. L)
    os = OpSum()
    os += 1, proj, Ly+1
    for i in Ly+2:L; os *= 1, proj, i; end
    return Id_op - MPO(os, sites)
end


"""
    _row_select_mpo(_, Ly, sites; keep=:even) -> MPO

Diagonal mask that retains only even or odd rows of a `2^Lx × 2^Ly` grid.

- `keep = :even` → 1 where iy % 2 == 1  (0-based; LSB of iy = 1)
- `keep = :odd`  → 1 where iy % 2 == 0  (0-based; LSB of iy = 0)
"""
function _row_select_mpo(::Any, Ly, sites; keep::Symbol = :even)
    proj = keep === :even ? "sigma_d" :
           keep === :odd  ? "sigma_u" :
           error("unknown keep=:$(keep); use :even or :odd")
    os = OpSum()
    os += 1, proj, Ly    # site Ly is the LSB of iy
    return MPO(os, sites)
end


"""
    _col_select_mpo(Lx, Ly, sites; keep=:even) -> MPO

Diagonal mask that retains only even or odd columns of a `2^Lx × 2^Ly` grid.

- `keep = :even` → 1 where ix % 2 == 1  (0-based; LSB of ix = 1)
- `keep = :odd`  → 1 where ix % 2 == 0  (0-based; LSB of ix = 0)
"""
function _col_select_mpo(Lx, Ly, sites; keep::Symbol = :even)
    proj = keep === :even ? "sigma_d" :
           keep === :odd  ? "sigma_u" :
           error("unknown keep=:$(keep); use :even or :odd")
    os = OpSum()
    os += 1, proj, Lx + Ly   # site L = Lx+Ly is the LSB of ix
    return MPO(os, sites)
end


"""
    _row_checker_mpo(Lx, Ly, sites) -> MPO

Diagonal checkerboard mask: 1 where (ix + iy) is even, 0 otherwise.
Equivalent to projecting onto LSB(ix) == LSB(iy), i.e. both qubits agree:
  proj_{iy-LSB=0, ix-LSB=0}  +  proj_{iy-LSB=1, ix-LSB=1}
"""
function _row_checker_mpo(Lx, Ly, sites)
    os = OpSum()
    os += 1, "sigma_u", Ly, "sigma_u", Lx + Ly   # both LSBs = 0
    os += 1, "sigma_d", Ly, "sigma_d", Lx + Ly   # both LSBs = 1
    return MPO(os, sites)
end


# ============================================================
# 6. NNN 2D kinetic builders
#    Pattern for every function:
#      1. Build ku = generate_kin_u, kd = generate_kin_d
#      2. Raise to the nn-th power with compose_power
#      3. Apply hopping weights: hop_fwd = h * ku^n,  hop_bwd = kd^n * h†
#      4. Mask with _row_break_mpo and optionally _row_select/_checker
# ============================================================

"""
    kineticintra2DNNN(Lx, Ly, sites, hopping, nn; apply_kwargs) -> MPO

Long-range intra-row hopping on a `2^Lx × 2^Ly` square lattice (nn bonds
along x).  Row wrap-around at ix = Nx-1 is suppressed by `_row_break_mpo(:xplus)`.
"""
function kineticintra2DNNN(Lx, Ly, sites, hopping::MPO, nn::Integer; apply_kwargs = NamedTuple())
    L = Lx + Ly
    @assert L == length(sites) && nn ≥ 1
    ku   = generate_kin_u(sites, 2^L)
    kd   = generate_kin_d(sites, 2^L)
    ku_n = compose_power(ku, nn; side=:right, apply_kwargs)
    kd_n = compose_power(kd, nn; side=:left,  apply_kwargs)
    hop_fwd = apply(hopping, ku_n; apply_kwargs...)
    hop_bwd = apply(kd_n, dag(hopping); apply_kwargs...)
    brk = _row_break_mpo(Lx, Ly, sites; which=:xplus)
    hop_fwd = apply(brk, hop_fwd; apply_kwargs...)
    hop_bwd = apply(hop_bwd, brk; apply_kwargs...)
    return +(hop_fwd, hop_bwd; cutoff=1e-12)
end


"""
    kineticinterNNNSWNE(Lx, Ly, sites, hopping, nn; apply_kwargs) -> MPO

Long-range inter-row hopping along the SW↗NE diagonal of a `2^Lx × 2^Ly`
square lattice.  Row end-wrap suppressed by `_row_break_mpo(:xplus)`.
"""
function kineticinterNNNSWNE(Lx, Ly, sites, hopping::MPO, nn::Integer; apply_kwargs = NamedTuple())
    L = Lx + Ly
    @assert L == length(sites) && nn ≥ 1
    ku   = generate_kin_u(sites, 2^L)
    kd   = generate_kin_d(sites, 2^L)
    ku_n = compose_power(ku, nn; side=:right, apply_kwargs)
    kd_n = compose_power(kd, nn; side=:left,  apply_kwargs)
    hop_fwd = apply(hopping, ku_n; apply_kwargs...)
    hop_bwd = apply(kd_n, dag(hopping); apply_kwargs...)
    brk = _row_break_mpo(Lx, Ly, sites; which=:xplus)
    hop_fwd = apply(brk, hop_fwd; apply_kwargs...)
    hop_bwd = apply(hop_bwd, brk; apply_kwargs...)
    return +(hop_fwd, hop_bwd; cutoff=1e-12)
end


"""
    kineticinterNNNSENW(Lx, Ly, sites, hopping, nn; apply_kwargs) -> MPO

Long-range inter-row hopping along the SE↖NW diagonal.
Row start-wrap suppressed by `_row_break_mpo(:xplain)`.
"""
function kineticinterNNNSENW(Lx, Ly, sites, hopping::MPO, nn::Integer; apply_kwargs = NamedTuple())
    L = Lx + Ly
    @assert L == length(sites) && nn ≥ 1
    ku   = generate_kin_u(sites, 2^L)
    kd   = generate_kin_d(sites, 2^L)
    ku_n = compose_power(ku, nn; side=:right, apply_kwargs)
    kd_n = compose_power(kd, nn; side=:left,  apply_kwargs)
    hop_fwd = apply(hopping, ku_n; apply_kwargs...)
    hop_bwd = apply(kd_n, dag(hopping); apply_kwargs...)
    brk = _row_break_mpo(Lx, Ly, sites; which=:xplain)
    hop_fwd = apply(brk, hop_fwd; apply_kwargs...)
    hop_bwd = apply(hop_bwd, brk; apply_kwargs...)
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
    @assert L == length(sites) && nn ≥ 1
    ku   = generate_kin_u(sites, 2^L)
    kd   = generate_kin_d(sites, 2^L)
    ku_n = compose_power(ku, nn; side=:right, apply_kwargs)
    kd_n = compose_power(kd, nn; side=:left,  apply_kwargs)
    hop_fwd = apply(hopping, ku_n; apply_kwargs...)
    hop_bwd = apply(kd_n, dag(hopping); apply_kwargs...)
    brk = _row_break_mpo(Lx, Ly, sites; which=:xplus)
    sel = _row_select_mpo(Lx, Ly, sites; keep=:even)
    hop_fwd = apply(sel, apply(brk, hop_fwd; apply_kwargs...); apply_kwargs...)
    hop_bwd = apply(apply(hop_bwd, brk; apply_kwargs...), sel; apply_kwargs...)
    return +(hop_fwd, hop_bwd; cutoff=1e-12)
end


"""
    kineticinterNNNtriSENW(Lx, Ly, sites, hopping, nn; apply_kwargs) -> MPO

SE↖NW diagonal inter-row hopping for a triangular lattice.
Applies `_row_break_mpo(:xplain)` and `_row_select_mpo(:odd)`.
"""
function kineticinterNNNtriSENW(Lx, Ly, sites, hopping::MPO, nn::Integer; apply_kwargs = NamedTuple())
    L = Lx + Ly
    @assert L == length(sites) && nn ≥ 1
    ku   = generate_kin_u(sites, 2^L)
    kd   = generate_kin_d(sites, 2^L)
    ku_n = compose_power(ku, nn; side=:right, apply_kwargs)
    kd_n = compose_power(kd, nn; side=:left,  apply_kwargs)
    hop_fwd = apply(hopping, ku_n; apply_kwargs...)
    hop_bwd = apply(kd_n, dag(hopping); apply_kwargs...)
    brk = _row_break_mpo(Lx, Ly, sites; which=:xplain)
    sel = _row_select_mpo(Lx, Ly, sites; keep=:odd)
    hop_fwd = apply(sel, apply(brk, hop_fwd; apply_kwargs...); apply_kwargs...)
    hop_bwd = apply(apply(hop_bwd, brk; apply_kwargs...), sel; apply_kwargs...)
    return +(hop_fwd, hop_bwd; cutoff=1e-12)
end


"""
    kineticintra2DNNhex(Lx, Ly, sites, hopping, nn; apply_kwargs) -> MPO

Intra-row hopping for a honeycomb lattice.  Applies `_row_break_mpo(:xplus)`
and `_row_checker_mpo` to implement the alternating A/B sublattice pattern.
"""
function kineticintra2DNNhex(Lx, Ly, sites, hopping::MPO, nn::Integer; apply_kwargs = NamedTuple())
    L = Lx + Ly
    ku   = generate_kin_u(sites, 2^L)
    kd   = generate_kin_d(sites, 2^L)
    ku_n = compose_power(ku, nn; side=:right, apply_kwargs)
    kd_n = compose_power(kd, nn; side=:left,  apply_kwargs)
    hop_fwd = apply(hopping, ku_n; apply_kwargs...)
    hop_bwd = apply(kd_n, dag(hopping); apply_kwargs...)
    brk = _row_break_mpo(Lx, Ly, sites; which=:xplus)
    chk = _row_checker_mpo(Lx, Ly, sites)
    hop_fwd = apply(chk, apply(brk, hop_fwd; apply_kwargs...); apply_kwargs...)
    hop_bwd = apply(apply(hop_bwd, brk; apply_kwargs...), chk; apply_kwargs...)
    return +(hop_fwd, hop_bwd; cutoff=1e-12)
end


# ============================================================
# 7. Preset model Hamiltonians
#    All return an MPO.  Sites are created internally (Qubit sites).
#    Use build_hamiltonian("model_name", ...) for the registry interface.
# ============================================================

# ---- 1D models ----

"""
    HUniform(L, t; v=1e-6, tol_quantics=1e-8, maxbonddim_quantics=10, nn=1) -> MPO

Uniform-hopping tight-binding chain on 2^L sites with an optional uniform onsite
potential `v`.  A small nonzero `v` is required to avoid TCI failure on constant functions.
"""
function HUniform(L::Integer, t;
                  v::Real                  = 1e-6,
                  tol_quantics::Real       = 1e-8,
                  maxbonddim_quantics::Int = 10,
                  nn::Integer              = 1)
    v == 0.0 && @warn "onsite potential v=0 can cause TCI failure; set v to a small nonzero value"
    N     = 2^L
    sites = siteinds("Qubit", L)
    xvals = 0:N-1
    hops_MPO   = qtt_mpo(L, xvals, sites, _ -> t;  tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)
    onsite_MPO = qtt_mpo(L, xvals, sites, _ -> v;  tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)
    return +(kineticNNN(L, sites, hops_MPO, nn), onsite_MPO; cutoff=1e-8)
end


"""
    HSSH(L, t, d; tol_quantics=1e-8, maxbonddim_quantics=10, nn=1) -> MPO

SSH (Su-Schrieffer-Heeger) Hamiltonian: dimerized hopping `t±d` on alternating bonds.
"""
function HSSH(L::Integer, t, d;
              tol_quantics::Real       = 1e-8,
              maxbonddim_quantics::Int = 10,
              nn::Integer              = 1)
    N     = 2^L
    sites = siteinds("Qubit", L)
    xvals = 0:N-1
    hops_MPO = qtt_mpo(L, xvals, sites, x -> iseven(x) ? (t + d) : (t - d);
                       tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)
    return kineticNNN(L, sites, hops_MPO, nn)
end


"""
    HAAH(L, V, phi, t; b=(1+√5)/2, tol_quantics=1e-8, maxbonddim_quantics=50) -> MPO

Aubry–André–Harper quasicrystal:
    H = t * Σ c†_{i+1}c_i + V * cos(2π b i + φ) * n_i
"""
function HAAH(L::Integer, V, phi, t;
              b::Real                  = (1 + sqrt(5)) / 2,
              tol_quantics::Real       = 1e-8,
              maxbonddim_quantics::Int = 50)
    N     = 2^L
    sites = siteinds("Qubit", L)
    xvals = 0:N-1
    hops_MPO   = qtt_mpo(L, xvals, sites, _ -> t;  tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)
    onsite_MPO = qtt_mpo(L, xvals, sites, x -> V * cos(2π * b * x + phi);
                         tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)
    return +(kineticNNN(L, sites, hops_MPO, 1), onsite_MPO; cutoff=1e-8)
end


# ---- 2D models ----

"""
    HUniform2Dsquare(Lx, Ly, t; tol_quantics=1e-8, maxbonddim_quantics=10, cutoff=1e-10) -> MPO

Uniform tight-binding Hamiltonian on a `2^Lx × 2^Ly` square lattice (row-major encoding).
Intra-row: `kineticintra2DNNN(…, nn=1)`.  Inter-row: `kineticNNN(…, nn=Nx)`.
"""
function HUniform2Dsquare(Lx::Integer, Ly::Integer, t;
                          tol_quantics::Real       = 1e-8,
                          maxbonddim_quantics::Int = 10,
                          cutoff::Real             = 1e-10)
    Nx    = 2^Lx
    L     = Lx + Ly
    N     = Nx * 2^Ly
    sites = siteinds("Qubit", L)
    xvals = 0:N-1
    hops  = qtt_mpo(L, xvals, sites, _ -> t; tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)
    Hintra = kineticintra2DNNN(Lx, Ly, sites, hops, 1)
    Hinter = kineticNNN(L, sites, hops, Nx)
    return +(Hintra, Hinter; cutoff=cutoff)
end


"""
    HUniform2Dhex(Lx, Ly, t; tol_quantics=1e-8, maxbonddim_quantics=10, cutoff=1e-10) -> MPO

Uniform tight-binding Hamiltonian on a `2^Lx × 2^Ly` hexagonal lattice.
Intra-row uses `kineticintra2DNNhex` (checkerboard mask); inter-row uses `kineticNNN(…, Nx)`.
"""
function HUniform2Dhex(Lx::Integer, Ly::Integer, t;
                       tol_quantics::Real       = 1e-8,
                       maxbonddim_quantics::Int = 10,
                       cutoff::Real             = 1e-10)
    Nx    = 2^Lx
    L     = Lx + Ly
    N     = Nx * 2^Ly
    sites = siteinds("Qubit", L)
    xvals = 0:N-1
    hops  = qtt_mpo(L, xvals, sites, _ -> t; tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)
    Hintra = kineticintra2DNNhex(Lx, Ly, sites, hops, 1)
    Hinter = kineticNNN(L, sites, hops, Nx)
    return +(Hintra, Hinter; cutoff=cutoff)
end


"""
    HUniform2Dtri(Lx, Ly, t; tol_quantics=1e-8, maxbonddim_quantics=10, cutoff=1e-10) -> MPO

Uniform tight-binding Hamiltonian on a `2^Lx × 2^Ly` triangular lattice.
Three kinetic terms:
- `kineticintra2DNNN(…, 1)` — intra-row NN
- `kineticinterNNNtriSWNE(…, Nx+1)` — SW↗NE diagonal
- `kineticinterNNNtriSENW(…, Nx-1)` — SE↖NW diagonal
"""
function HUniform2Dtri(Lx::Integer, Ly::Integer, t;
                       tol_quantics::Real       = 1e-8,
                       maxbonddim_quantics::Int = 10,
                       cutoff::Real             = 1e-10)
    Nx    = 2^Lx
    L     = Lx + Ly
    N     = Nx * 2^Ly
    sites = siteinds("Qubit", L)
    xvals = 0:N-1
    hops       = qtt_mpo(L, xvals, sites, _ -> t; tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)
    HintraNN   = kineticintra2DNNN(       Lx, Ly, sites, hops,  1)
    HinterNN   = kineticNNN(              L,       sites, hops, Nx)
    HinterSWNE = kineticinterNNNtriSWNE(  Lx, Ly, sites, hops, Nx + 1)
    HinterSENW = kineticinterNNNtriSENW(  Lx, Ly, sites, hops, Nx - 1)
    Htot = +(HintraNN,   HinterNN;   cutoff=cutoff)
    Htot = +(Htot,       HinterSWNE; cutoff=cutoff)
    return  +(Htot,      HinterSENW; cutoff=cutoff)
end


"""
    HChern8(Lx, Ly, V, t; a=5/64*2^Lx, t2=0.2t, tol_quantics=1e-8, ...) -> MPO

8-fold "Chern mosaic" Hamiltonian: uniform intra/inter-row hoppings modulated by
a spatially varying 8-fold pattern using 4 rotated k-vectors.
"""
function HChern8(Lx::Integer, Ly::Integer, V, t;
                 a::Real                  = 5/64 * 2^Lx,
                 t2::Real                 = 0.2 * t,
                 tol_quantics::Real       = 1e-8,
                 maxbonddim_quantics::Int = 10,
                 cutoff::Real             = 1e-10)
    Nx    = 2^Lx
    L     = Lx + Ly
    N     = Nx * 2^Ly
    sites = siteinds("Qubit", L)
    xvals = 0:N-1

    alt_hop_x(x) = (-1)^mod(x + 1, Nx) * t

    function func8fold(x, y)
        Ka1 = (2π/a) .* [1.0, 0.0];  Kb1 = (2π/a) .* [0.0, 1.0]
        θ = deg2rad(45.0);  Rt = [cos(θ) sin(θ); -sin(θ) cos(θ)]
        K = (Ka1, Kb1, Rt*Ka1, Rt*Kb1)
        return sum(1im * V * t2 * cos(dot(k, [x, y]))^2 for k in K)
    end

    wrap(f) = i -> f(i % Nx, i ÷ Nx)

    w_alt = wrap((x,y) -> alt_hop_x(x))
    w1    = wrap((x,y) -> t)
    w2    = wrap((x,y) -> alt_hop_x(mod(x-1, Nx)) * func8fold(x+0.5, y+0.5))
    w3    = wrap((x,y) -> alt_hop_x(x)             * func8fold(x-0.5, y+0.5))

    hops_MPO  = qtt_mpo(L, xvals, sites, w_alt; tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)
    hops_MPO1 = qtt_mpo(L, xvals, sites, w1;    tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)
    hops_MPO2 = qtt_mpo(L, xvals, sites, w2;    tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)
    hops_MPO3 = qtt_mpo(L, xvals, sites, w3;    tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)

    HinterNN  = kineticNNN(          L,    sites, hops_MPO,  Nx)
    HintraNN  = kineticintra2DNNN(   Lx, Ly, sites, hops_MPO1, 1)
    HinterSWNE = kineticinterNNNSWNE(Lx, Ly, sites, hops_MPO2, Nx+1)
    HinterSENW = kineticinterNNNSENW(Lx, Ly, sites, hops_MPO3, Nx-1)

    Htot = +(HinterNN,  HinterSWNE; cutoff=cutoff)
    Htot = +(Htot,      HinterSENW; cutoff=cutoff)
    return  +(Htot,     HintraNN;   cutoff=cutoff)
end


"""
    H2DChernhex(Lx, Ly, t, t2, ms; uniformhaldane=false, uniformsemenoff=false, ...) -> MPO

Haldane-type Chern insulator on a hexagonal lattice.
- NN hopping `t` (intra-row hex + vertical inter-row)
- Complex NNN hopping `±i·T2(x,y)` (checkerboard alternation) for next-nearest
- Semenoff mass `Ms(x,y)` on-site term
- Domain wall in t2 and Ms along x = Nx/2 by default
  (`uniformhaldane=true` and `uniformsemenoff=true` override to uniform fields)
"""
function H2DChernhex(Lx::Integer, Ly::Integer, t, t2, ms;
                     uniformhaldane::Bool     = false,
                     uniformsemenoff::Bool    = false,
                     tol_quantics::Real       = 1e-8,
                     maxbonddim_quantics::Int = 10,
                     cutoff::Real             = 1e-10)
    Nx    = 2^Lx
    L     = Lx + Ly
    N     = Nx * 2^Ly
    sites = siteinds("Qubit", L)
    xvals = 0:N-1

    T2 = uniformhaldane ? ((_,_) -> t2) :
         ((x,_) -> x < div(Nx, 2) ? t2 : -t2)

    Ms = uniformsemenoff ? ((_,_) -> ms) :
         ((x,_) -> x < div(Nx, 2) ? ms : ms + 3.3*sqrt(3)*t2)

    alt_hop_xy(x,y) = isodd(x+y) ? -1im*T2(x,y) : 1im*T2(x,y)
    semenoff(x,y)   = isodd(x+y) ? Ms(x,y) : -Ms(x,y)

    wrap(f) = i -> f(i % Nx, i ÷ Nx)

    hops_MPO      = qtt_mpo(L, xvals, sites, wrap((x,y) -> t);               tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)
    hops_MPOalter = qtt_mpo(L, xvals, sites, wrap((x,y) -> alt_hop_xy(x,y)); tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)
    on_site_MPO   = qtt_mpo(L, xvals, sites, wrap((x,y) -> semenoff(x,y));   tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)

    Hintra    = kineticintra2DNNhex( Lx, Ly, sites, hops_MPO,      1)
    Hinter    = kineticNNN(          L,       sites, hops_MPO,      Nx)
    HNNinter1 = kineticNNN(          L,       sites, hops_MPOalter, 2*Nx)
    HNNinter2 = kineticinterNNNSWNE( Lx, Ly, sites, hops_MPOalter, Nx+1)
    HNNinter3 = kineticinterNNNSENW( Lx, Ly, sites, hops_MPOalter, Nx-1)

    Htot = +(Hintra, Hinter;    cutoff=cutoff)
    Htot = +(Htot,   HNNinter1; cutoff=cutoff)
    Htot = +(Htot,   HNNinter2; cutoff=cutoff)
    Htot = +(Htot,   HNNinter3; cutoff=cutoff)
    return  +(Htot,  on_site_MPO; cutoff=cutoff)
end


"""
    HQC2Dsquare(Lx, Ly, t; tol_quantics=1e-8, maxbonddim_quantics=100, cutoff=1e-10) -> MPO

Quasicrystal-modulated square lattice.  The hopping amplitude at each bond is evaluated
at the bond midpoint using an 8-fold modulation with two competing wavevectors
`b1 = 5√5 a/2` and `b2 = √3 Nx a/16`.
"""
function HQC2Dsquare(Lx::Integer, Ly::Integer, t::Real = 1.0;
                     tol_quantics::Real       = 1e-8,
                     maxbonddim_quantics::Int = 100,
                     cutoff::Real             = 1e-10)
    Nx    = 2^Lx
    L     = Lx + Ly
    N     = Nx * 2^Ly
    sites = siteinds("Qubit", L)
    xvals = 0:N-1

    function func8fold(x, y)
        a  = 1
        b1 = 5*sqrt(5)*a/2
        b2 = sqrt(3)*(Nx*a/16)
        Ka1 = 2π .* [1.0, 0.0]; Kb1 = 2π .* [0.0, 1.0]
        tht = deg2rad(45.0); Rt = [cos(tht) sin(tht); -sin(tht) cos(tht)]
        K   = (Ka1, Kb1, Rt*Ka1, Rt*Kb1)
        xy  = [x - Nx/2, y - Nx/2]
        return t * (1 + 0.1 * sum(2.5*cos(dot(k,xy)/b1) + cos(dot(k,xy)/b2) for k in K))
    end

    intra = i -> func8fold(i%Nx + 0.5, i÷Nx)
    inter = i -> func8fold(i%Nx,        i÷Nx + 0.5)

    hops_intra = qtt_mpo(L, xvals, sites, intra; tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)
    hops_inter = qtt_mpo(L, xvals, sites, inter; tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)

    Hintra = kineticintra2DNNN(Lx, Ly, sites, hops_intra, 1)
    Hinter = kineticNNN(L,           sites, hops_inter, Nx)
    return +(Hinter, Hintra; cutoff=cutoff)
end


# ============================================================
# 8. Kagomé lattice
# ============================================================

"""
    kagome_positions(Lx, Ly) -> Matrix{Float64}

Return the (3·2^L × 2) real-space atom-position matrix for a kagomé lattice
of 2^Lx × 2^Ly unit cells (L = Lx+Ly), consistent with the MPO site ordering.

For total 1-indexed site i:
  n_cell  = (i-1) ÷ 3          (0-indexed unit cell, row-major)
  s       = (i-1) % 3 + 1      (sublattice: A=1, B=2, C=3)
  ix = n_cell % Nx,  iy = n_cell ÷ Nx

Atom positions (lattice vectors a₁=(1,0), a₂=(½,√3/2)):
  A: (ix + iy/2,        iy·√3/2       )
  B: (ix + iy/2 + ½,    iy·√3/2       )
  C: (ix + iy/2 + ¼,    iy·√3/2 + √3/4)
"""
function kagome_positions(Lx::Int, Ly::Int)
    Nx    = 2^Lx
    N_uc  = 2^(Lx + Ly)
    rs    = Matrix{Float64}(undef, 3 * N_uc, 2)
    sq3_2 = sqrt(3) / 2
    sq3_4 = sqrt(3) / 4
    for n in 0:N_uc-1
        ix   = n % Nx
        iy   = n ÷ Nx
        ax   = ix + iy * 0.5
        ay   = iy * sq3_2
        base = 3n + 1
        rs[base,   :] = [ax,        ay        ]   # A
        rs[base+1, :] = [ax + 0.5,  ay        ]   # B
        rs[base+2, :] = [ax + 0.25, ay + sq3_4]   # C
    end
    return rs
end


"""
    kagome_hamiltonian(Lx, Ly[, t]; cutoff, maxdim) -> TBHamiltonian

Build a uniform kagomé tight-binding Hamiltonian as a `TBHamiltonian`.

**Encoding** (L+1 sites total, L = Lx+Ly):
- Sites 1…L : L position qubits for 2^L unit cells on a triangular Bravais lattice
              (row-major: n = ix + iy·2^Lx)
- Site  L+1 : dim-3 "Kagome" sublattice index (A=1, B=2, C=3), postpended

**Hopping structure** (uniform amplitude `t`):

*Intra-cell* — all three bonds of the unit-cell triangle (same unit cell):
  A-B, B-C, A-C

*Inter-cell* — one bond type per triangular lattice direction:
  x   (shift +1   ): B(n+1)    ↔ A(n)
  y   (shift +Nx  ): A(n+Nx)   ↔ C(n)
  diag(shift +Nx-1): B(n+Nx-1) ↔ C(n)

Boundary wrapping is not enforced.  Use `kagome_positions(Lx, Ly)` for
real-space atom coordinates.  The sublattice index is stored in `H.sublattice_s`;
`H.aux_side = :post`.
"""
function kagome_hamiltonian(Lx::Integer, Ly::Integer, t::Number = 1.0;
                             cutoff::Real = 1e-8,
                             maxdim::Int  = 200)
    Nx = 2^Lx
    L  = Lx + Ly
    N  = 2^L

    pos_sites = siteinds("Qubit", L)
    kag_s     = Index(3, "Kagome")
    all_sites = [pos_sites; kag_s]

    ku   = generate_kin_u(pos_sites, N)
    kd   = generate_kin_d(pos_sites, N)
    Id   = MPO(pos_sites, "Id")
    apkw = (; cutoff = cutoff, maxdim = maxdim)

    brk_xp = _row_break_mpo(Lx, Ly, pos_sites; which=:xplus)   # zeros ix = Nx-1
    brk_xn = _row_break_mpo(Lx, Ly, pos_sites; which=:xplain)  # zeros ix = 0

    # ── Intra-cell: full off-diagonal 3×3 triangle ────────────────────────────
    H_intra = postpend_op(Id, kag_s, t * Float64[0 1 1; 1 0 1; 1 1 0])

    # ── Inter-cell x: A(n) ↔ B(n+1), shift ±1 ───────────────────────────────
    # Break suppresses B(Nx-1) ↔ A(0) wrap-around across row boundary
    H_x = +(t        * postpend_op(apply(brk_xp, ku;  apkw...), kag_s, 1, 2),
             conj(t) * postpend_op(apply(kd, brk_xp; apkw...), kag_s, 2, 1); cutoff=cutoff)

    # ── Inter-cell y: C(n) ↔ A(n+Nx), shift ±Nx ─────────────────────────────
    ku_y = compose_power(ku, Nx; apply_kwargs=apkw)
    kd_y = compose_power(kd, Nx; apply_kwargs=apkw)
    H_y  = +(t        * postpend_op(ku_y, kag_s, 1, 3),
              conj(t) * postpend_op(kd_y, kag_s, 3, 1); cutoff=cutoff)

    # ── Inter-cell diagonal: C(n) ↔ B(n+Nx-1), shift ±(Nx-1) ────────────────
    # Break suppresses C(0) ↔ B(Nx-1) same-row spurious bond (ix=0 source wraps)
    ku_d = compose_power(ku, Nx - 1; apply_kwargs=apkw)
    kd_d = compose_power(kd, Nx - 1; apply_kwargs=apkw)
    H_d  = +(t        * postpend_op(apply(brk_xn, ku_d; apkw...), kag_s, 2, 3),
              conj(t) * postpend_op(apply(kd_d, brk_xn; apkw...), kag_s, 3, 2); cutoff=cutoff)

    # ── Assembly ───────────────────────────────────────────────────────────────
    H_total = +(H_intra, H_x;    cutoff=cutoff)
    H_total = +(H_total, H_y;    cutoff=cutoff)
    H_total = +(H_total, H_d;    cutoff=cutoff)
    ITensorMPS.truncate!(H_total; maxdim=maxdim, cutoff=cutoff)

    # Kagomé spectrum: flat band at −2t, dispersive bands reaching up to +4t
    scale = 4.5 * abs(t)
    return TBHamiltonian(L, N, all_sites, H_total, nothing, scale, 0.0,
                         nothing, nothing, nothing, kag_s, :post, nothing, nothing, 0, nothing)
end



# ============================================================
# 8b. Lieb lattice
# ============================================================

"""
    lieb_positions(Lx, Ly) -> Matrix{Float64}

Return the (3·2^L × 2) real-space atom-position matrix for a Lieb lattice
of 2^Lx × 2^Ly unit cells (L = Lx+Ly), consistent with the MPO site ordering.

For total 1-indexed site i:
  n_cell  = (i-1) ÷ 3          (0-indexed unit cell, row-major)
  s       = (i-1) % 3 + 1      (sublattice: A=1, B=2, C=3)
  ix = n_cell % Nx,  iy = n_cell ÷ Nx

Atom positions (lattice vectors a₁=(1,0), a₂=(0,1)):
  A: (ix,       iy      )   corner
  B: (ix + 0.5, iy      )   x-edge center
  C: (ix,       iy + 0.5)   y-edge center
"""
function lieb_positions(Lx::Int, Ly::Int)
    Nx   = 2^Lx
    N_uc = 2^(Lx + Ly)
    rs   = Matrix{Float64}(undef, 3 * N_uc, 2)
    for n in 0:N_uc-1
        ix   = n % Nx
        iy   = n ÷ Nx
        base = 3n + 1
        rs[base,   :] = [ix,       iy       ]   # A
        rs[base+1, :] = [ix + 0.5, iy       ]   # B
        rs[base+2, :] = [ix,       iy + 0.5 ]   # C
    end
    return rs
end


"""
    lieb_hamiltonian(Lx, Ly[, t]; cutoff, maxdim) -> TBHamiltonian

Build a uniform Lieb tight-binding Hamiltonian as a `TBHamiltonian`.

**Encoding** (L+1 sites total, L = Lx+Ly):
- Sites 1…L : L position qubits for 2^L unit cells on a square Bravais lattice
              (row-major: n = ix + iy·2^Lx)
- Site  L+1 : dim-3 "Lieb" sublattice index (A=1, B=2, C=3), postpended

**Hopping structure** (uniform amplitude `t`):

*Intra-cell* — two bonds within the unit-cell cross:
  A-B, A-C

*Inter-cell*:
  x (shift +1 ): B(n) ↔ A(n+1)  — break at ix=Nx-1
  y (shift +Nx): C(n) ↔ A(n+Nx) — no x-break needed (pure y step)

The spectrum has a flat band at E=0 and two dispersive bands at ±2t.
Use `lieb_positions(Lx, Ly)` for real-space atom coordinates.
The sublattice index is stored in `H.sublattice_s`; `H.aux_side = :post`.
"""
function lieb_hamiltonian(Lx::Integer, Ly::Integer, t::Number = 1.0;
                           cutoff::Real = 1e-8,
                           maxdim::Int  = 200)
    Nx = 2^Lx
    L  = Lx + Ly
    N  = 2^L

    pos_sites = siteinds("Qubit", L)
    lieb_s    = Index(3, "Lieb")
    all_sites = [pos_sites; lieb_s]

    ku   = generate_kin_u(pos_sites, N)
    kd   = generate_kin_d(pos_sites, N)
    Id   = MPO(pos_sites, "Id")
    apkw = (; cutoff = cutoff, maxdim = maxdim)

    brk_xp = _row_break_mpo(Lx, Ly, pos_sites; which=:xplus)

    # ── Intra-cell: A↔B and A↔C within the same unit cell ────────────────────
    H_intra = postpend_op(Id, lieb_s, t * Float64[0 1 1; 1 0 0; 1 0 0])

    # ── Inter-cell x: B(n) ↔ A(n+1), shift ±1 ───────────────────────────────
    # Break suppresses B(Nx-1) ↔ A(0) wrap-around across row boundary
    H_x = +(t        * postpend_op(apply(brk_xp, ku;  apkw...), lieb_s, 1, 2),
             conj(t) * postpend_op(apply(kd, brk_xp; apkw...), lieb_s, 2, 1); cutoff=cutoff)

    # ── Inter-cell y: C(n) ↔ A(n+Nx), shift ±Nx ─────────────────────────────
    ku_y = compose_power(ku, Nx; apply_kwargs=apkw)
    kd_y = compose_power(kd, Nx; apply_kwargs=apkw)
    H_y  = +(t        * postpend_op(ku_y, lieb_s, 1, 3),
              conj(t) * postpend_op(kd_y, lieb_s, 3, 1); cutoff=cutoff)

    # ── Assembly ───────────────────────────────────────────────────────────────
    H_total = +(H_intra, H_x;    cutoff=cutoff)
    H_total = +(H_total, H_y;    cutoff=cutoff)
    ITensorMPS.truncate!(H_total; maxdim=maxdim, cutoff=cutoff)

    # Lieb spectrum: flat band at 0, dispersive bands up to ±2t
    scale = 2.5 * abs(t)
    return TBHamiltonian(L, N, all_sites, H_total, nothing, scale, 0.0,
                         nothing, nothing, nothing, lieb_s, :post, nothing, nothing, 0, nothing)
end


# ============================================================
# 9. Antiferromagnetic / Néel initial-guess density matrices
#    Used as seeds for mean-field SCF on interacting models.
#    Return (density_MPO, density_MPS).
# ============================================================

"""
    initial_guess_trivial_up_1D(L, sites) -> (MPO, MPS)

Diagonal density MPO with occupation `x % 2` on site `x` (spin-up Néel seed for 1D).
"""
function initial_guess_trivial_up_1D(L, sites)
    xvals = range(0, 2^L - 1; length=2^L)
    qtt   = QuanticsTCI.quanticscrossinterpolate(Float64, x -> Float64(Int(x) % 2), xvals;
                maxbonddim=10, tolerance=1e-8)[1]
    mps   = MPS(TCI.tensortrain(qtt.tci); sites)
    mpo   = outer(mps', mps)
    for i in 1:L; mpo.data[i] = Quantics._asdiagonal(mps.data[i], sites[i]); end
    return mpo, mps
end


"""
    initial_guess_trivial_down_1D(L, sites) -> (MPO, MPS)

Diagonal density MPO with occupation `(x+1) % 2` on site `x` (spin-down Néel seed for 1D).
"""
function initial_guess_trivial_down_1D(L, sites)
    xvals = range(0, 2^L - 1; length=2^L)
    qtt   = QuanticsTCI.quanticscrossinterpolate(Float64, x -> Float64((Int(x)+1) % 2), xvals;
                maxbonddim=10, tolerance=1e-8)[1]
    mps   = MPS(TCI.tensortrain(qtt.tci); sites)
    mpo   = outer(mps', mps)
    for i in 1:L; mpo.data[i] = Quantics._asdiagonal(mps.data[i], sites[i]); end
    return mpo, mps
end


"""
    initial_guess_Neel_up(Lx, Ly, sites) -> (MPO, MPS)

Checkerboard spin-up density seed for 2D Hubbard: occupation 1 where `(ix+iy)` is even.
"""
function initial_guess_Neel_up(Lx, Ly, sites)
    Nx    = 2^Lx
    L     = Lx + Ly
    N     = Nx * 2^Ly
    xvals = 0:N-1
    f     = i -> isodd(i%Nx + i÷Nx) ? 0.0 : 1.0
    qtt   = QuanticsTCI.quanticscrossinterpolate(Float64, f, xvals; maxbonddim=10, tolerance=1e-8)[1]
    mps   = MPS(TCI.tensortrain(qtt.tci); sites)
    mpo   = outer(mps', mps)
    for i in 1:L; mpo.data[i] = Quantics._asdiagonal(mps.data[i], sites[i]); end
    return mpo, mps
end


"""
    initial_guess_Neel_dn(Lx, Ly, sites) -> (MPO, MPS)

Checkerboard spin-down density seed for 2D Hubbard: occupation 1 where `(ix+iy)` is odd.
"""
function initial_guess_Neel_dn(Lx, Ly, sites)
    Nx    = 2^Lx
    L     = Lx + Ly
    N     = Nx * 2^Ly
    xvals = 0:N-1
    f     = i -> isodd(i%Nx + i÷Nx) ? 1.0 : 0.0
    qtt   = QuanticsTCI.quanticscrossinterpolate(Float64, f, xvals; maxbonddim=10, tolerance=1e-8)[1]
    mps   = MPS(TCI.tensortrain(qtt.tci); sites)
    mpo   = outer(mps', mps)
    for i in 1:L; mpo.data[i] = Quantics._asdiagonal(mps.data[i], sites[i]); end
    return mpo, mps
end


# ============================================================
# 9. Model registry + build_hamiltonian dispatcher
# ============================================================

"""
    _parse_param_string(s) -> Dict{Symbol,Any}

Parse `"key1=val1, key2=val2, …"` into a Dict. Values are auto-typed
as Bool, Int, Float64, or String.
"""
function _parse_param_string(s::AbstractString)
    d = Dict{Symbol,Any}()
    isempty(strip(s)) && return d
    for tok in split(strip(s), [',',' ','\t'])
        isempty(tok) && continue
        kv = split(tok, '='; limit=2)
        length(kv) == 2 || error("Bad param token '$tok' (expected key=value)")
        k   = Symbol(strip(kv[1]))
        v   = strip(kv[2])
        vl  = lowercase(v)
        val::Any = vl in ("true","false")                               ? (vl=="true") :
                   occursin(r"^[+-]?\d+$", v)                          ? parse(Int, v) :
                   occursin(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$", v) ? parse(Float64, v) :
                   v
        d[k] = val
    end
    return d
end


# Maps model name → (function, dim, required_params, kw_defaults)
const MODEL_REGISTRY = Dict{String, Tuple{Symbol,Int,Vector{Symbol},NamedTuple}}(
    "uniform"         => (:HUniform,         1, [:t],          (; v=1e-6, tol_quantics=1e-8, maxbonddim_quantics=10, nn=1)),
    "ssh"             => (:HSSH,             1, [:t, :d],      (; tol_quantics=1e-8, maxbonddim_quantics=10, nn=1)),
    "aah"             => (:HAAH,             1, [:V, :phi, :t],(; b=(1+sqrt(5))/2, tol_quantics=1e-8, maxbonddim_quantics=50)),
    "square_2d"       => (:HUniform2Dsquare, 2, [:t],          (; tol_quantics=1e-8, maxbonddim_quantics=10, cutoff=1e-10)),
    "hex_2d"          => (:HUniform2Dhex,    2, [:t],          (; tol_quantics=1e-8, maxbonddim_quantics=10, cutoff=1e-10)),
    "triangular_2d"   => (:HUniform2Dtri,    2, [:t],          (; tol_quantics=1e-8, maxbonddim_quantics=10, cutoff=1e-10)),
    "chern8"          => (:HChern8,          2, [:V, :t],      (; t2=0.2, tol_quantics=1e-8, maxbonddim_quantics=10, cutoff=1e-10)),
    "chernhex"        => (:H2DChernhex,      2, [:t, :t2, :ms],(; uniformhaldane=false, uniformsemenoff=false,
                                                                   tol_quantics=1e-8, maxbonddim_quantics=10, cutoff=1e-10)),
    "qc2dsquare"      => (:HQC2Dsquare,      2, [:t],          (; tol_quantics=1e-9, maxbonddim_quantics=250, cutoff=1e-10)),
)


"""
    build_hamiltonian(model, L; mparams="", mparam_dict=Dict()) -> MPO          (1D)
    build_hamiltonian(model, Lx, Ly; mparams="", mparam_dict=Dict()) -> MPO     (2D)

Build a Hamiltonian MPO by model name using the MODEL_REGISTRY.

```julia
H = build_hamiltonian("aah",  8; mparams="V=2.0, phi=0.0, t=1.0")
H = build_hamiltonian("square_2d", 4, 4; mparams="t=1.0")
H = build_hamiltonian("chernhex",  4, 4; mparam_dict=Dict(:t=>1.0, :t2=>0.3, :ms=>0.0))
```

Known models: $(join(sort(collect(keys(MODEL_REGISTRY))), ", "))
"""
function build_hamiltonian(model::AbstractString, L::Integer;
                           mparams::AbstractString = "",
                           mparam_dict             = Dict{Symbol,Any}())
    key = lowercase(model)
    haskey(MODEL_REGISTRY, key) || error("Unknown model '$model'. Known: $(sort(collect(keys(MODEL_REGISTRY))))")
    fn_sym, dim, required, kw_defaults = MODEL_REGISTRY[key]
    dim == 1 || error("Model '$model' is 2D; call build_hamiltonian(model, Lx, Ly; …)")
    fn = getfield(@__MODULE__, fn_sym)
    p  = _parse_param_string(mparams)
    for (k,v) in mparam_dict; p[k] = v; end
    missing_p = [k for k in required if !haskey(p, k)]
    isempty(missing_p) || error("Missing required params for '$model': $missing_p")
    pos   = [p[k] for k in required]
    extra = Dict(k=>v for (k,v) in p if !(k in required))
    return fn(L, pos...; kw_defaults..., extra...)
end

function build_hamiltonian(model::AbstractString, Lx::Integer, Ly::Integer;
                           mparams::AbstractString = "",
                           mparam_dict             = Dict{Symbol,Any}())
    key = lowercase(model)
    haskey(MODEL_REGISTRY, key) || error("Unknown model '$model'. Known: $(sort(collect(keys(MODEL_REGISTRY))))")
    fn_sym, dim, required, kw_defaults = MODEL_REGISTRY[key]
    dim == 2 || error("Model '$model' is 1D; call build_hamiltonian(model, L; …)")
    fn = getfield(@__MODULE__, fn_sym)
    p  = _parse_param_string(mparams)
    for (k,v) in mparam_dict; p[k] = v; end
    missing_p = [k for k in required if !haskey(p, k)]
    isempty(missing_p) || error("Missing required params for '$model': $missing_p")
    pos   = [p[k] for k in required]
    extra = Dict(k=>v for (k,v) in p if !(k in required))
    return fn(Lx, Ly, pos...; kw_defaults..., extra...)
end
