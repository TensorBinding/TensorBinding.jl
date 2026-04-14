# RPA_tk.jl — Random Phase Approximation (polarization bubble + Dyson inversion)
#
# The working pipeline is:
#
#   1. get_Tnlists        — build the three Chebyshev lists needed for the bubble
#   2. get_bublle_expanded_from_Tn — compute Π₀(ω) as a 2L-site MPO
#   3. build_bubble_mpo   — wrap Π₀ into the final L-site polarization bubble MPO
#   4. rpa_from_bubble_diag — solve (I - Π₀V) χ = Π₀ for the RPA susceptibility

# ============================================================
# Tensor product utilities (MPO/MPS Kronecker product)
# ============================================================

"""
    mpo_kron(A, B) -> MPO

Concatenate two MPOs into a single MPO on the combined site space,
joined by a bond-dimension-1 link.
"""
function mpo_kron(A::MPO, B::MPO)
    LA = length(A)
    LB = length(B)
    M  = MPO([ITensor() for _ in 1:(LA+LB)], 1, LA+LB)
    for j in 1:LA;  M[j]    = A[j];  end
    for j in 1:LB;  M[LA+j] = B[j];  end
    link     = Index(1, "Link_AB")
    M[LA]   *= delta(link)
    M[LA+1] *= delta(link)
    return M
end


"""
    mps_kron(A, B) -> MPS

Concatenate two MPS into a single MPS on the combined site space,
joined by a bond-dimension-1 link.
"""
function mps_kron(A::MPS, B::MPS)
    LA = length(A)
    LB = length(B)
    M  = MPS([ITensor() for _ in 1:(LA+LB)], 1, LA+LB)
    for j in 1:LA;  M[j]    = A[j];  end
    for j in 1:LB;  M[LA+j] = B[j];  end
    link     = Index(1, "Link_AB")
    M[LA]   *= delta(link)
    M[LA+1] *= delta(link)
    return M
end

# ============================================================
# Site-index manipulation helpers
# ============================================================

# Identify (bra, ket) among the two site legs of an MPO site tensor.
@inline function _bra_ket(sij)
    @assert length(sij) == 2 "MPO site tensor should have exactly 2 site legs"
    return plev(sij[1]) == 1 ? (sij[1], sij[2]) : (sij[2], sij[1])
end

nsitelegs(T::ITensor) = count(i -> hastags(i, "Site"), inds(T))


"""
    replace_sites(MPOin, newsites) -> MPO

Replace the physical (bra + ket) indices of each site in `MPOin`
with the corresponding index from `newsites`, preserving prime levels.
"""
function replace_sites(MPOin::MPO, newsites)
    L      = length(MPOin)
    indsMPO = siteinds(MPOin)
    T = MPO(L)
    for n in 1:L
        bra_old, ket_old = _bra_ket(indsMPO[n])
        T[n] = MPOin[n] *
               delta(bra_old, prime(newsites[n])) *
               delta(ket_old, newsites[n])
    end
    return T
end


"""
    swap_every_other_legs(MPOin, newsites) -> MPO

Replace site indices and additionally swap bra↔ket on every even-numbered
site.  Used to convert the 2L-site bubble MPO from the interleaved ordering
into the form expected by `collapse_mpo_pairs`.
"""
function swap_every_other_legs(MPOin::MPO, newsites)
    L2      = length(MPOin)
    @assert length(newsites) == L2
    indsMPO = siteinds(MPOin)
    T = MPO(L2)
    for n in 1:L2
        s     = indsMPO[n]
        new_s = newsites[n]
        if iseven(n)
            T[n] = MPOin[n] * delta(s[1], prime(new_s)) * delta(s[2], new_s)
        else
            T[n] = MPOin[n] * delta(s[1], new_s)        * delta(s[2], prime(new_s))
        end
    end
    return T
end


"""
    collapse_mpo_pairs(mpo2L, out_sites) -> MPO

Merge each consecutive pair of sites `(2n-1, 2n)` in a `2L`-site MPO
into a single site of an `L`-site MPO by contracting and tying the
shared bra and ket indices to `out_sites[n]`.
"""
function collapse_mpo_pairs(mpo2L::MPO, out_sites)
    L2 = length(mpo2L)
    @assert iseven(L2) "Input MPO must have even length (2L)."
    L = L2 ÷ 2
    @assert length(out_sites) == L
    sinds = siteinds(mpo2L)
    T = MPO(L)
    for n in 1:L
        bra1, ket1 = _bra_ket(sinds[2n-1])
        bra2, ket2 = _bra_ket(sinds[2n])
        snew = out_sites[n]
        W    = mpo2L[2n-1] * mpo2L[2n]
        W   *= delta(bra1, bra2, prime(snew))
        W   *= delta(ket1, ket2, snew)
        T[n] = W
    end
    return T
end

# ============================================================
# Interleaving (embed an L-site MPO into a 2L-site space)
# ============================================================

"""
    interleave_mpo(target_mpo, phys_sites, n) -> MPO

Embed an `L`-site MPO into a `2L`-site space by interleaving it with
identity operators.  `phys_sites` must have length `2L`.

- `n = 0` : operator sits at odd positions (1, 3, 5, …), identities at even
- `n = 1` : operator sits at even positions (2, 4, 6, …), identities at odd
"""
function interleave_mpo(target_mpo, phys_sites, n)
    N_old = length(target_mpo)
    N_new = 2 * N_old
    @assert length(phys_sites) == N_new

    new_mpo  = MPO(phys_sites)
    link_map = Dict{Index, Vector{Index}}()
    for k in 1:N_old-1
        ol          = linkind(target_mpo, k)
        d           = dim(ol)
        link_map[ol] = [Index(d, "Link,l=$(2k-1)"), Index(d, "Link,l=$(2k)")]
    end

    for i in 1:N_old
        idx_orig  = (n == 1) ? 2i-1 : 2i
        idx_ident = (n == 1) ? 2i   : 2i-1

        W = target_mpo[i]
        W = replaceinds(W, siteinds(target_mpo, i) =>
                           (phys_sites[idx_orig], phys_sites[idx_orig]'))
        if i > 1
            ol_left = linkind(target_mpo, i-1)
            W = replaceind(W, ol_left => link_map[ol_left][2])
        end
        if i < N_old
            ol_right = linkind(target_mpo, i)
            W = replaceind(W, ol_right => link_map[ol_right][1])
        end
        new_mpo[idx_orig] = W

        if idx_ident == 1 || idx_ident == N_new
            new_mpo[idx_ident] = delta(phys_sites[idx_ident], phys_sites[idx_ident]')
        else
            ol     = linkind(target_mpo, idx_ident ÷ 2)
            l_left  = link_map[ol][1]
            l_right = link_map[ol][2]
            new_mpo[idx_ident] = delta(l_left, l_right) *
                                  delta(phys_sites[idx_ident], phys_sites[idx_ident]')
        end
    end
    return new_mpo
end

# ============================================================
# Diagonal extraction
# ============================================================

"""
    extract_diagonal_to_mps(M) -> MPS

Extract the diagonal of an MPO `M` as an MPS by projecting each site
tensor onto the subspace where bra and ket indices are equal.
"""
function extract_diagonal_to_mps(M::MPO)::MPS
    N            = length(M)
    new_tensors  = Vector{ITensor}(undef, N)
    for i in 1:N
        t      = M[i]
        s2, s1 = siteinds(M, i)   # s2 = bra, s1 = ket
        dim_s  = dim(s1)
        v_inds = uniqueinds(t, s1, s2)
        res    = ITensor(v_inds..., s1)
        for v in 1:dim_s
            slice = t * onehot(s1 => v) * onehot(s2 => v)
            res  += slice * onehot(s1 => v)
        end
        new_tensors[i] = res
    end
    return MPS(new_tensors)
end

# ============================================================
# MPO/MPS merging utilities
# ============================================================

"""
    merge_mps_to_mpo(mps) -> MPO

Contract each consecutive pair `(2i-1, 2i)` of an MPS into a single
MPO tensor.  The resulting MPO has `length(mps) ÷ 2` sites.
"""
function merge_mps_to_mpo(mps)
    N     = length(mps)
    new_N = N ÷ 2
    mpo   = MPO(new_N)
    for i in 1:new_N
        mpo[i] = mps[2i-1] * mps[2i]
    end
    return mpo
end


"""
    convert_mpo(old_mps, new_sites) -> MPO

Convert a `2N`-site MPS (typically from QTCI) into an `N`-site MPO
by merging pairs and remapping physical indices to `new_sites`.
"""
function convert_mpo(old_mps, new_sites)
    N       = length(new_sites)
    old_mpo = merge_mps_to_mpo(old_mps)
    new_mpo = MPO(N)
    for i in 1:N
        old_s1 = siteind(old_mps, 2i-1)
        old_s2 = siteind(old_mps, 2i)
        new_mpo[i] = replaceinds(old_mpo[i],
                                 [old_s1, old_s2] => [new_sites[i]', new_sites[i]])
    end
    return new_mpo
end

# ============================================================
# Swap-based interleaving (alternative approach via SWAP gates)
# ============================================================

function _swap_mpo(i::Integer, j::Integer, sites)::MPO
    os  = OpSum()
    os += 0.5, "Id", i, "Id", j
    os += 0.5, "X",  i, "X",  j
    os += 0.5, "Y",  i, "Y",  j
    os += 0.5, "Z",  i, "Z",  j
    return MPO(os, sites)
end


"""
    apply_interleave_swaps(W, sites; cutoff, maxdim, verbose) -> MPO

Re-order the sites of `W` from `[1…N, N+1…2N]` to the interleaved order
`[1, N+1, 2, N+2, …]` by composing a sequence of adjacent SWAP gates.
Truncates after each swap to control bond dimension growth.
"""
function apply_interleave_swaps(W::MPO, sites;
                                cutoff::Real=1e-16, maxdim::Int=200,
                                verbose::Bool=false)
    L = length(sites)
    @assert iseven(L)
    N = L ÷ 2

    swaps = Tuple{Int,Int}[]
    order = collect(1:L)
    for p in 1:L
        desired = isodd(p) ? (p + 1) ÷ 2 : N + p ÷ 2
        q = findfirst(==(desired), order)
        if q != p
            push!(swaps, (p, q))
            order[p], order[q] = order[q], order[p]
        end
    end
    verbose && @info "Number of swaps" length(swaps)

    Wcur = W
    for (a, b) in swaps
        verbose && @info "Applying swap" (a, b)
        S    = _swap_mpo(a, b, sites)
        Wcur = apply(dag(S), Wcur, S)
        ITensorMPS.truncate!(Wcur; cutoff=cutoff, maxdim=maxdim)
        verbose && @info "maxlinkdim(W)" maxlinkdim(Wcur)
    end
    return Wcur
end

# ============================================================
# Polarization bubble
# ============================================================

"""
    get_Tnlists(H, H2, sites, sites2, N; a, maxdim)
        -> (Tn_list1, Tn_list2, Tn_listeff)

Build the three Chebyshev moment lists needed for the bubble:
- `Tn_list1`   : moments for H₁ (system 1)
- `Tn_list2`   : moments for H₂ (system 2)
- `Tn_listeff` : moments for H_eff = I⊗H₂ − H₁⊗I on the combined 2L-site space
"""
function get_Tnlists(H, H2, sites, sites2, N; a=6, maxdim=100)
    id1            = MPO(sites,  "Id")
    id2            = MPO(sites2, "Id")
    sites_combined = vcat(sites, sites2)
    Tn_list1       = KPM_Tn(H,  N, sites,  maxdim=maxdim)
    Tn_list2       = KPM_Tn(H2, N, sites2, maxdim=maxdim)
    H2op  = interleave_mpo(a * H2, sites_combined, 1)
    Iop2  = interleave_mpo(id2,    sites_combined, 1)
    Iop1  = interleave_mpo(id1,    sites_combined, 0)
    H1op  = interleave_mpo(a * H,  sites_combined, 0)
    Heff  = apply(Iop1, H2op) - apply(H1op, Iop2)
    Tn_listeff = KPM_Tn(Heff / a, N, sites_combined, maxdim=maxdim)
    return Tn_list1, Tn_list2, Tn_listeff
end


"""
    get_bublle_expanded_from_Tn(Tn_list1, Tn_list2, Tn_listeff,
                                 sites1, sites2, N, ω, ϵF;
                                 a, maxdim) -> MPO

Compute the non-interacting polarization bubble Π₀(ω) as a 2L-site MPO
using the Lehmann representation in terms of the Chebyshev moments.

The bubble is:
    Π₀(ω) = (P₁⊗I₂ − I₁⊗P₂) · G_eff(ω)
where P₁, P₂ are density matrices (filled-band projectors) and
G_eff is the retarded Green's function of H_eff = I⊗H₂ − H₁⊗I.
"""
function get_bublle_expanded_from_Tn(Tn_list1, Tn_list2, Tn_listeff,
                                      sites1, sites2, N, ω, ϵF;
                                      a=6, maxdim=200)
    P1 = get_density_from_Tn(Tn_list1, N; fermi=ϵF/a, maxdim=maxdim)
    println("Got P1")
    P2 = get_density_from_Tn(Tn_list2, N; fermi=ϵF/a, maxdim=maxdim)
    println("Got P2")
    id1            = MPO(sites1, "Id")
    id2            = MPO(sites2, "Id")
    sites_combined = vcat(sites1, sites2)

    P2op      = interleave_mpo(P2,  sites_combined, 1)
    Iop2      = interleave_mpo(id2, sites_combined, 1)
    Iop1      = interleave_mpo(id1, sites_combined, 0)
    P1op      = interleave_mpo(P1,  sites_combined, 0)
    numerator = ITensorMPS.truncate!(
        apply(Iop1, P2op) - apply(P1op, Iop2); cutoff=1e-8)
    println("Got numerator")

    GF_rescaled = (1/a) * get_Green_retarded_from_Tn(Tn_listeff, N, ω/a;
                                                      η=1e-3, maxdim=maxdim)
    bubble2L = ITensorMPS.truncate!(apply(GF_rescaled, numerator); cutoff=1e-8)
    println("Got GF")
    return bubble2L
end

# ============================================================
# Public RPA pipeline
# ============================================================

# Default parameters — override by passing kwargs to build_bubble_mpo/rpa_from_bubble_diag
const CHI_BUBBLE    = 150
const SIGN_BUBBLE   = -1
const A_BUBBLE      = 6
const MAXDIM_BUBBLE = 200
const NSWEEPS_SOLVE = 20
const MAXDIM_SOLVE  = 400
const CUTOFF_SOLVE  = 1e-8


"""
    build_bubble_mpo(ω; Tn_list1, Tn_list2, Tn_listeff,
                        sites, sites2, finalsites, finalfinalsites,
                        chi, sign, a, maxdim) -> MPO

Compute the L-site polarization bubble Π₀(ω) from pre-computed
Chebyshev lists.  This is the main entry point for the RPA calculation.
"""
function build_bubble_mpo(ω;
                          Tn_list1, Tn_list2, Tn_listeff,
                          sites, sites2, finalsites, finalfinalsites,
                          chi=CHI_BUBBLE, sign=SIGN_BUBBLE,
                          a=A_BUBBLE, maxdim=MAXDIM_BUBBLE)
    bubble             = get_bublle_expanded_from_Tn(
        Tn_list1, Tn_list2, Tn_listeff,
        sites, sites2, chi, ω, sign; a=a, maxdim=maxdim)
    bubble_interleaved = swap_every_other_legs(bubble, finalsites)
    return collapse_mpo_pairs(bubble_interleaved, finalfinalsites)
end


"""
    rpa_from_bubble_diag(Π, MPOV, finalsites, finalfinalsites;
                         nsweeps, maxdim, cutoff) -> MPS

Solve the RPA Dyson equation  (I − Π₀V) χ = Π₀  for the interacting
susceptibility χ using DMRG-style linear solve.

Returns a 2L-site MPS encoding the diagonal χ_{iijj}^RPA.
"""
function rpa_from_bubble_diag(Π, MPOV, finalsites, finalfinalsites;
                               nsweeps=NSWEEPS_SOLVE,
                               maxdim=MAXDIM_SOLVE,
                               cutoff=CUTOFF_SOLVE)
    L   = length(finalfinalsites)
    Id  = MPO(finalfinalsites, "Id")
    ΠV  = apply(Π, MPOV; maxdim=maxdim, cutoff=cutoff)
    A   = Id - ΠV

    Aop = interleave_mpo(A, finalsites, 0)
    Πop = interleave_mpo(Π, finalsites, 0)
    b   = extract_diagonal_to_mps(Πop)

    # Align site indices between Aop and b
    for j in 1:2L
        sA = siteind(Aop, j)
        sb = siteind(b, j)
        if sb != sA
            replaceinds!(b[j], sb => noprime(sb))
        end
    end

    x0 = deepcopy(b)
    return ITensorMPS.linsolve(Aop, b, x0;
                               nsweeps=nsweeps, maxdim=maxdim, cutoff=cutoff)
end
