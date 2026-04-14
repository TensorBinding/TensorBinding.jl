# 2D_lattice.jl — MPO building blocks for 2D lattice geometries
#
# Provides hopping MPOs for square, triangular, and honeycomb lattices,
# built from the quantics binary representation.  All functions rely on
# the kinematic shift operators (generate_kin_u / generate_kin_d) and
# the QTCI-based break_chain mask.
#
# Low-level utilities (to_binary_vector, binary_to_MPS, get_diagonal_mpo,
# ITensors.op extensions) live in utils.jl and are available here through
# the module scope.

# ============================================================
# 1. General tools
# ============================================================

"""
    break_chain(x_start, L_chain, num_site, sites) -> MPO

Diagonal MPO that is 0 at sites `x_start, x_start + L_chain, …` and
1 elsewhere.  Used to suppress hopping at row boundaries in 2D lattices.
"""

ITensors.op(::OpName"sigma_d",::SiteType"Qubit") =
 [0 0
  0 1]

function test_break(x_start, L_chain, num_sites, sites) 
    L = Int(log2(num_sites))
    Id_op = MPO(sites, "Id")
    L_row_log = Int(L - Int(log2(L_chain)))
    
    os = OpSum()
    
    for i in 1:L
        os += 1/L, "Id",i 
    end
 
    for i in L_row_log+1 :L
        
        os *=  1,"sigma_d",i
    end
    
    k_mpo_1 = MPO(os,sites)
    break_mpo = Id_op - k_mpo_1
    return break_mpo
end


"""
    generate_kin_u(sites, num_site) -> MPO

'Up-shift' kinematic MPO: moves every basis state |n⟩ → |n+1⟩ in the
quantics binary representation.  Represents hopping to the right / up.
"""
function generate_kin_u(sites, num_site)
    L         = Int(log2(num_site))
    kinetic_1 = OpSum()
    for i in 1:L
        os = OpSum()
        os += 1, "sigma_plus", L - (i - 1)
        for j in 1:L-i
            os *= ("Id", j)
        end
        for j in L+2-i:L
            os *= ("sigma_minus", j)
        end
        kinetic_1 += os
    end
    return MPO(kinetic_1, sites)
end


"""
    generate_kin_d(sites, num_site) -> MPO

'Down-shift' kinematic MPO: moves every basis state |n⟩ → |n-1⟩.
Hermitian conjugate of `generate_kin_u`.
"""
function generate_kin_d(sites, num_site)
    L         = Int(log2(num_site))
    kinetic_2 = OpSum()
    for i in 1:L
        os = OpSum()
        os += 1, "sigma_minus", L - (i - 1)
        for j in 1:L-i
            os *= ("Id", j)
        end
        for j in L+2-i:L
            os *= ("sigma_plus", j)
        end
        kinetic_2 += os
    end
    return MPO(kinetic_2, sites)
end


"""
    arbitarty_offline(k_mpo, demand_order) -> MPO

Return `k_mpo^demand_order` (repeated MPO application), used to shift
hopping by an arbitrary number of sites.
"""
function arbitarty_offline(k_mpo, demand_order)
    k_mpo_o1   = k_mpo
    k_mpo_o2   = apply(k_mpo, k_mpo)
    target_mpo = k_mpo
    for iter_num in 1:demand_order
        if iter_num == 1
            target_mpo = k_mpo_o1
        elseif iter_num == 2
            target_mpo = k_mpo_o2
        else
            target_mpo = apply(k_mpo, k_mpo_o2)
            k_mpo_o2   = target_mpo
        end
    end
    return target_mpo
end


# ============================================================
# 2. Square lattice
# ============================================================

"""
    intrachain_hopping(L_chain, num_site, sites; hopping=Id, t=1) -> MPO

Nearest-neighbour hopping along rows (x-direction) of a 2D lattice
with `L_chain` sites per row.  Boundary links between rows are
suppressed by `break_chain`.
"""
function intrachain_hopping(L_chain, num_site, sites;
                            hopping=MPO(sites, "Id"), t=1)
    break_mpo  = test_break(L_chain, L_chain, num_site, sites)
    k_mpo_2    = generate_kin_d(sites, num_site)
    true_hop_2 = apply(apply(hopping, k_mpo_2), break_mpo)
    k_mpo_1    = generate_kin_u(sites, num_site)
    true_hop_1 = apply(break_mpo, apply(k_mpo_1, hopping))
    return +(t * true_hop_1, conj(t) * true_hop_2; cutoff=1e-8)
end


"""
    interchain_hopping_square(L_chain, num_site, sites; hopping=Id, t=1) -> MPO

Nearest-neighbour hopping along columns (y-direction) of a square lattice.
"""
function interchain_hopping_square(L_chain, num_site, sites;
                                   hopping=MPO(sites, "Id"), t=1)
    k_mpo_1       = generate_kin_u(sites, num_site)
    K_mpo_1_true  = apply(apply(hopping, k_mpo_1),
                          arbitarty_offline(k_mpo_1, L_chain - 1))
    k_mpo_2       = generate_kin_d(sites, num_site)
    K_mpo_2_true  = apply(arbitarty_offline(k_mpo_2, L_chain - 1),
                          apply(k_mpo_2, hopping))
    return t * K_mpo_1_true + conj(t) * K_mpo_2_true
end


"""
    interchain_hopping_square_2nd_plus(L_chain, num_site, sites; hopping=Id, t2=1) -> MPO

Next-nearest-neighbour hopping in the (+x+y) diagonal direction on a
square lattice.
"""
function interchain_hopping_square_2nd_plus(L_chain, num_site, sites;
                                            hopping=MPO(sites, "Id"), t2=1)
    break_mpo    = test_break(L_chain, L_chain, num_site, sites)
    K_mpo_1      = generate_kin_u(sites, num_site)
    K_mpo_1_true = apply(arbitarty_offline(K_mpo_1, L_chain + 1 - 1),
                         apply(hopping, apply(break_mpo, K_mpo_1)))
    K_mpo_2      = generate_kin_d(sites, num_site)
    K_mpo_2_true = apply(arbitarty_offline(K_mpo_2, L_chain + 1 - 1),
                         apply(apply(K_mpo_2, break_mpo), hopping))
    return t2 * K_mpo_1_true + conj(t2) * K_mpo_2_true
end


"""
    interchain_hopping_square_2nd_minus(L_chain, num_site, sites; hopping=Id, t2=1) -> MPO

Next-nearest-neighbour hopping in the (+x−y) diagonal direction on a
square lattice.
"""
function interchain_hopping_square_2nd_minus(L_chain, num_site, sites;
                                             hopping=MPO(sites, "Id"), t2=1)
    break_mpo    = test_break(1, L_chain, num_site, sites)
    K_mpo_1      = generate_kin_u(sites, num_site)
    K_mpo_1_true = apply(apply(hopping, apply(break_mpo, K_mpo_1)),
                         arbitarty_offline(K_mpo_1, L_chain - 1 - 1))
    K_mpo_2      = generate_kin_d(sites, num_site)
    K_mpo_2_true = apply(arbitarty_offline(K_mpo_2, L_chain - 1 - 1),
                         apply(apply(K_mpo_2, break_mpo), hopping))
    return t2 * K_mpo_1_true + conj(t2) * K_mpo_2_true
end


# ============================================================
# 3. Triangular lattice
# ============================================================

"""
    skeleton(L_chain, num_site, sites) -> MPO

Diagonal MPO that is 0 at site indices `1, L_chain+1, 2*L_chain+1, …`
and 1 elsewhere.  Used as a connectivity mask for triangular hoppings.
"""
function skeleton(L_chain, num_site, sites)
    L = Int(log2(num_site))
    f(x) = Float64(x % L_chain != 1)
    return get_diagonal_mpo(L, sites, f)
end


"""
    interchain_hopping_triangle(L_chain, num_site, sites) -> MPO

Hopping MPO for the two diagonal next-row bonds in a triangular lattice
(both (+1, +L_chain) and (+1, +L_chain−1) directions).
"""
function interchain_hopping_triangle(L_chain, num_site, sites)
    k_mpo_1 = generate_kin_u(sites, num_site)
    k_mpo_2 = generate_kin_d(sites, num_site)
    tri_hop = skeleton(L_chain, num_site, sites)

    K1_up = arbitarty_offline(k_mpo_1, L_chain)
    conn1 = apply(tri_hop, arbitarty_offline(k_mpo_1, L_chain - 1))
    K2_up = arbitarty_offline(k_mpo_2, L_chain)
    conn2 = apply(arbitarty_offline(k_mpo_2, L_chain - 1), tri_hop)

    return K1_up + K2_up + conn2 + conn1
end


# ============================================================
# 4. Honeycomb lattice
# ============================================================

function odd_template(::Any, num_site, sites)
    L = Int(log2(num_site))
    return get_diagonal_mpo(L, sites, x -> Float64(isodd(x)))
end

function even_template(L_chain, num_site, sites)
    L = Int(log2(num_site))
    return get_diagonal_mpo(L, sites,
                            x -> Float64(!(x % 2 == 0 || (x - 1) % L_chain == 0)))
end

function odd_skeleton(L_chain, num_site, sites)
    L = Int(log2(num_site))
    return get_diagonal_mpo(L, sites, x -> Float64(iseven(div(x, L_chain))))
end

function even_skeleton(L_chain, num_site, sites)
    L = Int(log2(num_site))
    return get_diagonal_mpo(L, sites, x -> Float64(isodd(div(x, L_chain))))
end


"""
    interchain_hopping_honeycomb(L_chain, num_site, sites) -> MPO

Inter-row hopping MPO for a honeycomb lattice with `L_chain` sites per
row.  Handles the two inequivalent inter-row bonds (A→B going up-left
and up-right).
"""
function interchain_hopping_honeycomb(L_chain, num_site, sites)
    connect_up = apply(odd_template(L_chain, num_site, sites),
                       odd_skeleton(L_chain, num_site, sites))
    connect_dn = apply(even_template(L_chain, num_site, sites),
                       even_skeleton(L_chain, num_site, sites))

    k_mpo_1 = generate_kin_u(sites, num_site)
    k_mpo_2 = generate_kin_d(sites, num_site)

    hop_up_1 = apply(connect_up, arbitarty_offline(k_mpo_1, L_chain + 1))
    hop_dn_1 = apply(connect_dn, arbitarty_offline(k_mpo_1, L_chain - 1))
    hop_up_2 = apply(arbitarty_offline(k_mpo_2, L_chain + 1), connect_up)
    hop_dn_2 = apply(arbitarty_offline(k_mpo_2, L_chain - 1), connect_dn)

    return hop_up_1 + hop_up_2 + hop_dn_1 + hop_dn_2
end


# ============================================================
# 5. Row/column/checkerboard mask MPOs (diagonal, 0/1 entries)
#    Conventions:
#      row-major flattening:  linear index i = ix + iy * 2^Lx
#      bit split:  low Lx bits → ix,  next Ly bits → iy
# ============================================================

"""
    _row_break_mpo(Lx, Ly, sites; which) -> MPO

Diagonal mask that zeroes out wrap-around couplings at row boundaries of a
`2^Lx × 2^Ly` grid flattened row-major.

- `which = :xplus`  → mask 0 where `(ix+1) % 2^Lx == 0`  (end of each row)
- `which = :xplain` → mask 0 where `ix % 2^Lx == 0`       (start of each row)

Multiply a kinetic MPO by this mask (left or right) to prevent hops wrapping
from the last site of one row to the first site of the next.
"""
function _row_break_mpo(Lx, Ly, sites; which::Symbol)
    L     = Lx + Ly
    xvals = 0:(2^L - 1)
    f = which === :xplus  ? (x -> iszero(mod(x + 1, 2^Lx)) ? 0.0 : 1.0) :
        which === :xplain ? (x -> iszero(mod(x,     2^Lx)) ? 0.0 : 1.0) :
        error("unknown which=:$(which); use :xplus or :xplain")
    qttb    = QuanticsTCI.quanticscrossinterpolate(ComplexF64, f, xvals; tolerance=1e-8)[1]
    ttb     = TCI.tensortrain(qttb.tci)
    maskmps = MPS(ttb; sites)
    maskmpo = outer(maskmps', maskmps)
    for i in 1:L
        maskmpo.data[i] = Quantics._asdiagonal(maskmps.data[i], sites[i])
    end
    return maskmpo
end


"""
    _row_select_mpo(Lx, Ly, sites; keep=:even) -> MPO

Diagonal mask that keeps only even (1-based: 2,4,…) or odd (1,3,…) rows
of a `2^Lx × 2^Ly` grid flattened row-major.

- `keep = :even` → retain rows where `iy % 2 == 1` (0-based)
- `keep = :odd`  → retain rows where `iy % 2 == 0` (0-based)
"""
function _row_select_mpo(Lx, Ly, sites; keep::Symbol = :even)
    L     = Lx + Ly
    xvals = 0:(2^L - 1)
    row_keep = keep === :even ? (iy -> iy % 2 == 1) :
               keep === :odd  ? (iy -> iy % 2 == 0) :
               error("unknown keep=:$(keep); use :even or :odd")
    f = x -> begin
        xi = x isa Integer ? x : Int(floor(x))
        row_keep(xi >>> Lx) ? 1.0 : 0.0
    end
    qttb = QuanticsTCI.quanticscrossinterpolate(ComplexF64, f, xvals; tolerance=1e-8)[1]
    ttb  = TCI.tensortrain(qttb.tci)
    mps  = MPS(ttb; sites)
    mpo  = outer(mps', mps)
    for i in 1:L
        mpo.data[i] = Quantics._asdiagonal(mps.data[i], sites[i])
    end
    return mpo
end


"""
    _col_select_mpo(Lx, Ly, sites; keep=:even) -> MPO

Diagonal mask that keeps only even or odd **columns** of a `2^Lx × 2^Ly` grid.
`keep = :even` retains columns where `ix % 2 == 1`; `:odd` where `ix % 2 == 0`.
"""
function _col_select_mpo(Lx, Ly, sites; keep::Symbol = :even)
    L     = Lx + Ly
    xvals = 0:(2^L - 1)
    xmask = (1 << Lx) - 1
    col_keep = keep === :even ? (ix -> ix % 2 == 1) :
               keep === :odd  ? (ix -> ix % 2 == 0) :
               error("unknown keep=:$(keep); use :even or :odd")
    f = x -> begin
        xi = x isa Integer ? x : Int(floor(x))
        col_keep(xi & xmask) ? 1.0 : 0.0
    end
    qttb = QuanticsTCI.quanticscrossinterpolate(ComplexF64, f, xvals; tolerance=1e-8)[1]
    ttb  = TCI.tensortrain(qttb.tci)
    mps  = MPS(ttb; sites)
    mpo  = outer(mps', mps)
    for i in 1:L
        mpo.data[i] = Quantics._asdiagonal(mps.data[i], sites[i])
    end
    return mpo
end


"""
    _row_checker_mpo(Lx, Ly, sites) -> MPO

Diagonal checkerboard mask: entry 1 where `(ix + iy)` is even, 0 otherwise.
Useful for Néel/AFM patterns and alternating sublattice filters.
"""
function _row_checker_mpo(Lx, Ly, sites)
    L     = Lx + Ly
    xvals = 0:(2^L - 1)
    xmask = (1 << Lx) - 1
    f = x -> begin
        xi = x isa Integer ? x : Int(floor(x))
        ix = xi & xmask
        iy = xi >>> Lx
        ((ix + iy) & 1 == 0) ? 1.0 : 0.0
    end
    qttb = QuanticsTCI.quanticscrossinterpolate(ComplexF64, f, xvals; tolerance=1e-8)[1]
    ttb  = TCI.tensortrain(qttb.tci)
    mps  = MPS(ttb; sites)
    mpo  = outer(mps', mps)
    for i in 1:L
        mpo.data[i] = Quantics._asdiagonal(mps.data[i], sites[i])
    end
    return mpo
end


# ============================================================
# 6. NNN 2D kinetic builders
#    All use compose_power (from Hamiltonian.jl) and the masks above.
# ============================================================

"""
    kineticintra2DNNN(Lx, Ly, sites, hopping, nn; apply_kwargs) -> MPO

Long-range **intra-row** hopping on a `2^Lx × 2^Ly` square lattice flattened row-major.
Row-boundary wrap-around is prevented by left/right multiplication with `_row_break_mpo`.
"""
function kineticintra2DNNN(Lx, Ly, sites, hopping::MPO, nn::Integer; apply_kwargs = NamedTuple())
    L = Lx + Ly
    @assert L == length(sites) && nn ≥ 1
    kinetic_1 = OpSum(); kinetic_2 = OpSum()
    for i in 1:L
        os = OpSum(); os += 1, "sigma_plus", L - (i-1)
        for j in 1:L-i;     os *= ("Id",          j); end
        for j in (L+2-i):L; os *= ("sigma_minus",  j); end
        kinetic_1 += os
    end
    for i in 1:L
        os = OpSum(); os += 1, "sigma_minus", L - (i-1)
        for j in 1:L-i;     os *= ("Id",         j); end
        for j in (L+2-i):L; os *= ("sigma_plus",  j); end
        kinetic_2 += os
    end
    k1 = MPO(kinetic_1, sites); k2 = MPO(kinetic_2, sites)
    An = compose_power(k1, nn; side=:right, apply_kwargs)
    Am = compose_power(k2, nn; side=:left,  apply_kwargs)
    th1 = apply(hopping, An; apply_kwargs...)
    th2 = apply(Am, dag(hopping); apply_kwargs...)
    brk = _row_break_mpo(Lx, Ly, sites; which=:xplus)
    th1 = apply(brk, th1; apply_kwargs...)
    th2 = apply(th2, brk; apply_kwargs...)
    return +(th1, th2; cutoff=1e-12)
end


"""
    kineticinterNNNSWNE(Lx, Ly, sites, hopping, nn; apply_kwargs) -> MPO

Long-range **inter-row** hopping along the SW↗NE diagonal of an `Lx × Ly` square lattice.
Prevents horizontal wrap-around with `:xplus` row-break mask.
"""
function kineticinterNNNSWNE(Lx, Ly, sites, hopping::MPO, nn::Integer; apply_kwargs = NamedTuple())
    L = Lx + Ly
    @assert L == length(sites) && nn ≥ 1
    kinetic_1 = OpSum(); kinetic_2 = OpSum()
    for i in 1:L
        os = OpSum(); os += 1, "sigma_plus", L - (i-1)
        for j in 1:L-i;     os *= ("Id",         j); end
        for j in (L+2-i):L; os *= ("sigma_minus", j); end
        kinetic_1 += os
    end
    for i in 1:L
        os = OpSum(); os += 1, "sigma_minus", L - (i-1)
        for j in 1:L-i;     os *= ("Id",        j); end
        for j in (L+2-i):L; os *= ("sigma_plus", j); end
        kinetic_2 += os
    end
    k1 = MPO(kinetic_1, sites); k2 = MPO(kinetic_2, sites)
    An = compose_power(k1, nn; side=:right, apply_kwargs)
    Am = compose_power(k2, nn; side=:left,  apply_kwargs)
    th1 = apply(hopping, An; apply_kwargs...)
    th2 = apply(Am, dag(hopping); apply_kwargs...)
    brk = _row_break_mpo(Lx, Ly, sites; which=:xplus)
    th1 = apply(brk, th1; apply_kwargs...)
    th2 = apply(th2, brk; apply_kwargs...)
    return +(th1, th2; cutoff=1e-12)
end


"""
    kineticinterNNNSENW(Lx, Ly, sites, hopping, nn; apply_kwargs) -> MPO

Long-range **inter-row** hopping along the SE↖NW diagonal.
Uses `:xplain` row-break mask (breaks at row start rather than row end).
"""
function kineticinterNNNSENW(Lx, Ly, sites, hopping::MPO, nn::Integer; apply_kwargs = NamedTuple())
    L = Lx + Ly
    @assert L == length(sites) && nn ≥ 1
    kinetic_1 = OpSum(); kinetic_2 = OpSum()
    for i in 1:L
        os = OpSum(); os += 1, "sigma_plus", L - (i-1)
        for j in 1:L-i;     os *= ("Id",         j); end
        for j in (L+2-i):L; os *= ("sigma_minus", j); end
        kinetic_1 += os
    end
    for i in 1:L
        os = OpSum(); os += 1, "sigma_minus", L - (i-1)
        for j in 1:L-i;     os *= ("Id",        j); end
        for j in (L+2-i):L; os *= ("sigma_plus", j); end
        kinetic_2 += os
    end
    k1 = MPO(kinetic_1, sites); k2 = MPO(kinetic_2, sites)
    An = compose_power(k1, nn; side=:right, apply_kwargs)
    Am = compose_power(k2, nn; side=:left,  apply_kwargs)
    th1 = apply(hopping, An; apply_kwargs...)
    th2 = apply(Am, dag(hopping); apply_kwargs...)
    brk = _row_break_mpo(Lx, Ly, sites; which=:xplain)
    th1 = apply(brk, th1; apply_kwargs...)
    th2 = apply(th2, brk; apply_kwargs...)
    return +(th1, th2; cutoff=1e-12)
end


"""
    kineticinterNNNtriSWNE(Lx, Ly, sites, hopping, nn; apply_kwargs) -> MPO

SW↗NE diagonal inter-row hopping for a **triangular lattice**.
Applies both an `:xplus` row-break mask and an `:even`-row selection mask.
"""
function kineticinterNNNtriSWNE(Lx, Ly, sites, hopping::MPO, nn::Integer; apply_kwargs = NamedTuple())
    L = Lx + Ly
    @assert L == length(sites) && nn ≥ 1
    kinetic_1 = OpSum(); kinetic_2 = OpSum()
    for i in 1:L
        os = OpSum(); os += 1, "sigma_plus", L - (i-1)
        for j in 1:L-i;     os *= ("Id",         j); end
        for j in (L+2-i):L; os *= ("sigma_minus", j); end
        kinetic_1 += os
    end
    for i in 1:L
        os = OpSum(); os += 1, "sigma_minus", L - (i-1)
        for j in 1:L-i;     os *= ("Id",        j); end
        for j in (L+2-i):L; os *= ("sigma_plus", j); end
        kinetic_2 += os
    end
    k1 = MPO(kinetic_1, sites); k2 = MPO(kinetic_2, sites)
    An = compose_power(k1, nn; side=:right, apply_kwargs)
    Am = compose_power(k2, nn; side=:left,  apply_kwargs)
    th1 = apply(hopping, An; apply_kwargs...)
    th2 = apply(Am, dag(hopping); apply_kwargs...)
    brk     = _row_break_mpo(Lx, Ly, sites; which=:xplus)
    rowmask = _row_select_mpo(Lx, Ly, sites; keep=:even)
    th1 = apply(rowmask, apply(brk, th1; apply_kwargs...); apply_kwargs...)
    th2 = apply(apply(th2, brk; apply_kwargs...), rowmask; apply_kwargs...)
    return +(th1, th2; cutoff=1e-12)
end


"""
    kineticinterNNNtriSENW(Lx, Ly, sites, hopping, nn; apply_kwargs) -> MPO

SE↖NW diagonal inter-row hopping for a **triangular lattice**.
Applies both an `:xplain` row-break mask and an `:odd`-row selection mask.
"""
function kineticinterNNNtriSENW(Lx, Ly, sites, hopping::MPO, nn::Integer; apply_kwargs = NamedTuple())
    L = Lx + Ly
    @assert L == length(sites) && nn ≥ 1
    kinetic_1 = OpSum(); kinetic_2 = OpSum()
    for i in 1:L
        os = OpSum(); os += 1, "sigma_plus", L - (i-1)
        for j in 1:L-i;     os *= ("Id",         j); end
        for j in (L+2-i):L; os *= ("sigma_minus", j); end
        kinetic_1 += os
    end
    for i in 1:L
        os = OpSum(); os += 1, "sigma_minus", L - (i-1)
        for j in 1:L-i;     os *= ("Id",        j); end
        for j in (L+2-i):L; os *= ("sigma_plus", j); end
        kinetic_2 += os
    end
    k1 = MPO(kinetic_1, sites); k2 = MPO(kinetic_2, sites)
    An = compose_power(k1, nn; side=:right, apply_kwargs)
    Am = compose_power(k2, nn; side=:left,  apply_kwargs)
    th1 = apply(hopping, An; apply_kwargs...)
    th2 = apply(Am, dag(hopping); apply_kwargs...)
    brk     = _row_break_mpo(Lx, Ly, sites; which=:xplain)
    rowmask = _row_select_mpo(Lx, Ly, sites; keep=:odd)
    th1 = apply(rowmask, apply(brk, th1; apply_kwargs...); apply_kwargs...)
    th2 = apply(apply(th2, brk; apply_kwargs...), rowmask; apply_kwargs...)
    return +(th1, th2; cutoff=1e-12)
end


"""
    kineticintra2DNNhex(Lx, Ly, sites, hopping, nn; apply_kwargs) -> MPO

Intra-row hopping for a **honeycomb lattice** (hexagonal geometry).
Applies both an `:xplus` row-break mask and a checkerboard mask to implement
the alternating A/B sublattice connectivity pattern.
"""
function kineticintra2DNNhex(Lx, Ly, sites, hopping::MPO, nn::Integer; apply_kwargs = NamedTuple())
    L = Lx + Ly
    kinetic_1 = OpSum(); kinetic_2 = OpSum()
    for i in 1:L
        os = OpSum(); os += 1, "sigma_plus", L - (i-1)
        for j in 1:L-i;     os *= ("Id",         j); end
        for j in (L+2-i):L; os *= ("sigma_minus", j); end
        kinetic_1 += os
    end
    for i in 1:L
        os = OpSum(); os += 1, "sigma_minus", L - (i-1)
        for j in 1:L-i;     os *= ("Id",        j); end
        for j in (L+2-i):L; os *= ("sigma_plus", j); end
        kinetic_2 += os
    end
    k1 = MPO(kinetic_1, sites); k2 = MPO(kinetic_2, sites)
    An = compose_power(k1, nn; side=:right, apply_kwargs)
    Am = compose_power(k2, nn; side=:left,  apply_kwargs)
    th1 = apply(hopping, An; apply_kwargs...)
    th2 = apply(Am, dag(hopping); apply_kwargs...)
    brk     = _row_break_mpo(Lx, Ly, sites; which=:xplus)
    checker = _row_checker_mpo(Lx, Ly, sites)
    th1 = apply(checker, apply(brk, th1; apply_kwargs...); apply_kwargs...)
    th2 = apply(apply(th2, brk; apply_kwargs...), checker; apply_kwargs...)
    return +(th1, th2; cutoff=1e-12)
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

Uniform tight-binding Hamiltonian on a `2^Lx × 2^Ly` **square lattice** (row-major encoding).
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
    w     = i -> t   # constant hopping
    hops  = qtt_mpo(L, xvals, sites, w; tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)
    Hintra = kineticintra2DNNN(Lx, Ly, sites, hops, 1)
    Hinter = kineticNNN(L, sites, hops, Nx)
    return +(Hintra, Hinter; cutoff=cutoff)
end


"""
    HUniform2Dhex(Lx, Ly, t; tol_quantics=1e-8, maxbonddim_quantics=10, cutoff=1e-10) -> MPO

Uniform tight-binding Hamiltonian on a `2^Lx × 2^Ly` **hexagonal lattice**.
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

Uniform tight-binding Hamiltonian on a `2^Lx × 2^Ly` **triangular lattice**.
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
    hops  = qtt_mpo(L, xvals, sites, _ -> t; tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)
    HintraNN  = kineticintra2DNNN(           Lx, Ly, sites, hops,  1)
    HinterSWNE = kineticinterNNNtriSWNE(     Lx, Ly, sites, hops, Nx + 1)
    HinterSENW = kineticinterNNNtriSENW(     Lx, Ly, sites, hops, Nx - 1)
    Htot = +(HintraNN, HinterSWNE; cutoff=cutoff)
    return  +(Htot,    HinterSENW; cutoff=cutoff)
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

    alt_hop_x(x)       = (-1)^mod(x + 1, Nx) * t

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

    HinterNN   = kineticNNN(           L,    sites, hops_MPO,  Nx)
    HintraNN   = kineticintra2DNNN(    Lx, Ly, sites, hops_MPO1, 1)
    HinterNNN  = kineticinterNNNSWNE(  Lx, Ly, sites, hops_MPO2, Nx+1)
    HinterNNN2 = kineticinterNNNSENW(  Lx, Ly, sites, hops_MPO3, Nx-1)

    Htot = +(HinterNN,  HinterNNN;  cutoff=cutoff)
    Htot = +(Htot,      HinterNNN2; cutoff=cutoff)
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

    hops_MPO      = qtt_mpo(L, xvals, sites, wrap((x,y) -> t);             tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)
    hops_MPOalter = qtt_mpo(L, xvals, sites, wrap((x,y) -> alt_hop_xy(x,y)); tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)
    on_site_MPO   = qtt_mpo(L, xvals, sites, wrap((x,y) -> semenoff(x,y));  tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)

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
# 8. Antiferromagnetic / Néel initial-guess density matrices
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
