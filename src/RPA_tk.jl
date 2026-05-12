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
    interleave_mpo_tb(op, sites_A, sites_B, which) -> MPO

Generalization of `interleave_mpo` for heterogeneous site spaces (Layer,
Qubit, Honeycomb, etc.).  Embeds an `N`-site MPO `op` into the `2N`-site
product space whose physical indices are ordered as

    [sites_A[1], sites_B[1], sites_A[2], sites_B[2], …, sites_A[N], sites_B[N]]

so every consecutive pair `(sites_A[i], sites_B[i])` shares the same
dimension regardless of site type.

- `which = :A` : `op` acts on `sites_A` (odd positions), identity on `sites_B`
- `which = :B` : `op` acts on `sites_B` (even positions), identity on `sites_A`
"""
function interleave_mpo_tb(op::MPO,
                            sites_A::Vector{<:Index},
                            sites_B::Vector{<:Index},
                            which::Symbol)
    N = length(op)
    @assert length(sites_A) == length(sites_B) == N "sites_A, sites_B and op must all have length N"
    sites_combined = reduce(vcat, [[sa, sb] for (sa, sb) in zip(sites_A, sites_B)])
    n = (which == :A) ? 1 : 0
    return interleave_mpo(op, sites_combined, n)
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

- `n = 0` : operator sits at even positions (2, 4, 6, …), identities at odd
- `n = 1` : operator sits at odd positions (1, 3, 5, …), identities at even

**Note**: `phys_sites` must be interleaved as `[A[1], B[1], A[2], B[2], …]`
so that each operator site lands on an index with the correct dimension.
For heterogeneous site spaces (layer, sublattice, …), use `interleave_mpo_tb`
which builds the interleaved site list automatically.
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

"""
    build_bubble_mpo(ω; Tn_list1, Tn_list2, Tn_listeff,
                        sites, sites2, finalsites, finalfinalsites,
                        chi, ϵF, a, maxdim) -> MPO

Compute the L-site polarization bubble Π₀(ω) from pre-computed
Chebyshev lists.
"""
function build_bubble_mpo(ω;
                          Tn_list1, Tn_list2, Tn_listeff,
                          sites, sites2, finalsites, finalfinalsites,
                          chi=150, ϵF=0.0, a=6, maxdim=200)
    bubble             = get_bublle_expanded_from_Tn(
        Tn_list1, Tn_list2, Tn_listeff,
        sites, sites2, chi, ω, ϵF; a=a, maxdim=maxdim)
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
                               nsweeps=20, maxdim=400, cutoff=1e-8)
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

# ============================================================
# Internal helpers for TBHamiltonian API
# ============================================================

function _get_density_matrix(H::TBHamiltonian, ϵF::Real,
                              P_method::Symbol, Ncheb::Int,
                              maxdim::Int, cutoff::Real,
                              purify_method::Symbol, purify_maxdim::Int,
                              purify_maxiters::Int, purify_tol::Float64,
                              verbose::Bool)
    if P_method == :purification
        if H._density_cache !== nothing
            verbose && println("  Reusing cached density matrix")
            return H._density_cache
        end
        if purify_method == :mcweeny
            verbose && println("  Running McWeeny purification")
            return mcweeny_purify(H; maxiters=purify_maxiters, maxdim=purify_maxdim,
                                     cutoff=cutoff, tol=purify_tol, verbose=verbose)
        elseif purify_method == :sp2
            Nel = H.N ÷ 2
            verbose && println("  Running SP2 purification (Nel=$Nel)")
            return sp2_purify(H; Nel=Nel, maxiters=purify_maxiters, maxdim=purify_maxdim,
                                 cutoff=cutoff, tol=purify_tol, verbose=verbose)
        else
            error("Unknown purify_method: $purify_method. Choose :mcweeny or :sp2")
        end
    elseif P_method == :kpm
        _ensure_scale!(H)
        Tn_list, _, _ = KPM_Tn(H.mpo, Ncheb, H.sites;
                                 scale=H.scale, center=H.center, maxdim=maxdim, cutoff=cutoff)
        fermi_rescaled = (ϵF - H.center) / H.scale
        return get_density_from_Tn(Tn_list, Ncheb; fermi=fermi_rescaled, maxdim=maxdim,
                                    cutoff=cutoff)
    else
        error("Unknown P_method: $P_method. Choose :purification or :kpm")
    end
end


function _build_heff(H1_mpo::MPO, H2_mpo::MPO,
                     sites1::Vector{<:Index}, sites2::Vector{<:Index})
    id1  = MPO(sites1, "Id")
    id2  = MPO(sites2, "Id")
    H2op = interleave_mpo_tb(H2_mpo, sites1, sites2, :B)
    Iop2 = interleave_mpo_tb(id2,    sites1, sites2, :B)
    Iop1 = interleave_mpo_tb(id1,    sites1, sites2, :A)
    H1op = interleave_mpo_tb(H1_mpo, sites1, sites2, :A)
    return apply(Iop1, H2op) - apply(H1op, Iop2)
end

# ============================================================
# High-level TBHamiltonian API
# ============================================================

"""
    get_bubble_mpo(H1::TBHamiltonian, H2::TBHamiltonian, ω; ...) -> MPO

Compute the non-interacting polarization bubble Π₀(ω) on `H1.sites`.

**Keyword arguments**
- `ϵF`             : Fermi energy (physical units). Default `0.0`.
- `P_method`       : `:purification` (default) or `:kpm` — how to compute density matrices.
  With `:purification`, `H._density_cache` is reused if present.
- `GF_method`      : `:kpm` (default) or `:krylov` — how to compute G_eff(ω).
- `Ncheb`          : Chebyshev order (KPM methods only). Default `150`.
- `maxdim`         : Maximum bond dimension. Default `200`.
- `cutoff`         : SVD truncation cutoff. Default `1e-8`.
- `purify_method`  : `:mcweeny` (default) or `:sp2`.
- `purify_maxdim`  : Max bond dim during purification. Default `40`.
- `purify_maxiters`: Max purification iterations. Default `30`.
- `purify_tol`     : Purification convergence tolerance. Default `1e-5`.
- `η`              : Lorentzian broadening for the GF. Default `1e-3`.
- `krylov_nsweeps` : DMRG sweeps for Krylov solver. Default `12`.
- `krylov_maxdim`  : Max bond dim for Krylov solver. Default `100`.
- `krylov_cutoff`  : SVD cutoff for Krylov solver. Default `1e-8`.
- `verbose`        : Print progress. Default `false`.
"""
function get_bubble_mpo(H1::TBHamiltonian, H2::TBHamiltonian, ω::Real;
                        ϵF::Real              = 0.0,
                        P_method::Symbol      = :purification,
                        GF_method::Symbol     = :kpm,
                        Ncheb::Int            = 150,
                        maxdim::Int           = 200,
                        cutoff::Real          = 1e-8,
                        purify_method::Symbol = :mcweeny,
                        purify_maxdim::Int    = 40,
                        purify_maxiters::Int  = 30,
                        purify_tol::Float64   = 1e-5,
                        η::Real               = 1e-3,
                        krylov_nsweeps::Int   = 12,
                        krylov_maxdim::Int    = 100,
                        krylov_cutoff::Real   = 1e-8,
                        verbose::Bool         = false)

    L1 = H1.L; L2 = H2.L
    @assert L1 == L2 "H1 and H2 must have the same number of sites (got $L1 vs $L2)"
    L      = L1
    sites1 = H1.sites
    sites2 = H2.sites

    # Interleaved combined sites: [s1[1], s2[1], s1[2], s2[2], …]
    # This ensures each (A, B) pair has matching dimensions regardless of site type
    # (Layer dim=5, Qubit dim=2, Honeycomb dim=2, etc.), making interleave_mpo_tb safe.
    sites_combined = reduce(vcat, [[s1, s2] for (s1, s2) in zip(sites1, sites2)])

    # ---- Density matrices ----
    verbose && println("Polarization bubble: computing P1 (P_method=$P_method)...")
    P1 = _get_density_matrix(H1, ϵF, P_method, Ncheb, maxdim, cutoff,
                              purify_method, purify_maxdim, purify_maxiters,
                              purify_tol, verbose)
    verbose && println("Polarization bubble: computing P2...")
    P2 = _get_density_matrix(H2, ϵF, P_method, Ncheb, maxdim, cutoff,
                              purify_method, purify_maxdim, purify_maxiters,
                              purify_tol, verbose)

    # ---- Numerator: I₁⊗P₂ − P₁⊗I₂ ----
    id1  = MPO(sites1, "Id")
    id2  = MPO(sites2, "Id")
    P2op = interleave_mpo_tb(P2,  sites1, sites2, :B)
    Iop2 = interleave_mpo_tb(id2, sites1, sites2, :B)
    Iop1 = interleave_mpo_tb(id1, sites1, sites2, :A)
    P1op = interleave_mpo_tb(P1,  sites1, sites2, :A)
    numerator = ITensorMPS.truncate!(
        apply(Iop1, P2op; maxdim, cutoff) - apply(P1op, Iop2; maxdim, cutoff);
        cutoff=cutoff)
    verbose && println("Polarization bubble: computed numerator")

    # ---- GF of Heff = I⊗H₂ − H₁⊗I ----
    Heff = _build_heff(H1.mpo, H2.mpo, sites1, sites2)
    verbose && println("Polarization bubble: Heff maxlinkdim = ", maxlinkdim(Heff))
    if GF_method == :kpm
        # Auto-estimate Heff spectral bounds via DMRG (scale=0 triggers estimator)
        Tn_listeff, scaleeff, centereff = KPM_Tn(Heff, Ncheb, sites_combined;
                                                   maxdim=maxdim, cutoff=cutoff)
        GF_mpo = (1/scaleeff) * get_Green_retarded_from_Tn(
            Tn_listeff, Ncheb, (ω - centereff)/scaleeff;
            η = η/scaleeff, maxdim=maxdim, cutoff=cutoff)
    elseif GF_method == :krylov
        GF_mpo = get_green_krylov(Heff, sites_combined, ω;
                                   η=η, nsweeps=krylov_nsweeps,
                                   maxdim=krylov_maxdim, cutoff=krylov_cutoff,
                                   verbose=verbose)
    else
        error("Unknown GF_method: $GF_method. Choose :kpm or :krylov")
    end
    verbose && println("Polarization bubble: computed Heff GF (GF_method=$GF_method)")

    # ---- Bubble: GF_eff · numerator ----
    bubble2L = ITensorMPS.truncate!(apply(GF_mpo, numerator; maxdim, cutoff); cutoff=cutoff)
    verbose && println("Polarization bubble: assembled bubble")

    # ---- Collapse 2L-site MPO → L-site Π₀ on H1.sites ----
    # finalsites mirrors sites_combined dims so swap_every_other_legs never hits a
    # dimension mismatch, even when sites1 contains heterogeneous indices (Layer, Honeycomb…).
    finalsites = [Index(dim(s), "Bubble,n=$i") for (i, s) in enumerate(sites_combined)]
    bubble_iv  = swap_every_other_legs(bubble2L, finalsites)
    return collapse_mpo_pairs(bubble_iv, H1.sites)
end


"""
    get_rpa_susceptibility(H::TBHamiltonian, MPOV, ω; ...) -> MPS

Compute the RPA susceptibility χ(ω) for a system described by `H` with
interaction MPO `MPOV`.  Returns a 2L-site MPS encoding the diagonal
χ_{iijj}^RPA.

Internally calls `get_bubble_mpo(H, H, ω; ...)` then solves the Dyson
equation (I − Π₀V) χ = Π₀.  All `get_bubble_mpo` keyword arguments are
accepted and forwarded.

**Additional keywords (Dyson solve)**
- `rpa_nsweeps` : sweeps for the RPA linsolve. Default `20`.
- `rpa_maxdim`  : max bond dim for the RPA linsolve. Default `400`.
- `rpa_cutoff`  : cutoff for the RPA linsolve. Default `1e-8`.
"""
function get_rpa_susceptibility(H::TBHamiltonian, MPOV::MPO, ω::Real;
                                 rpa_nsweeps::Int  = 20,
                                 rpa_maxdim::Int   = 400,
                                 rpa_cutoff::Real  = 1e-8,
                                 ϵF::Real              = 0.0,
                                 P_method::Symbol      = :purification,
                                 GF_method::Symbol     = :kpm,
                                 Ncheb::Int            = 150,
                                 maxdim::Int           = 200,
                                 cutoff::Real          = 1e-8,
                                 purify_method::Symbol = :mcweeny,
                                 purify_maxdim::Int    = 40,
                                 purify_maxiters::Int  = 30,
                                 purify_tol::Float64   = 1e-5,
                                 η::Real               = 1e-3,
                                 krylov_nsweeps::Int   = 12,
                                 krylov_maxdim::Int    = 100,
                                 krylov_cutoff::Real   = 1e-8,
                                 verbose::Bool         = false)

    Π = get_bubble_mpo(H, H, ω;
                        ϵF, P_method, GF_method, Ncheb, maxdim, cutoff,
                        purify_method, purify_maxdim, purify_maxiters, purify_tol,
                        η, krylov_nsweeps, krylov_maxdim, krylov_cutoff, verbose)

    finalsites = siteinds("Qubit", 2 * length(H.sites))
    return rpa_from_bubble_diag(Π, MPOV, finalsites, H.sites;
                                 nsweeps=rpa_nsweeps, maxdim=rpa_maxdim, cutoff=rpa_cutoff)
end

# ============================================================
# Wynn ε-algorithm accelerated RPA
# ============================================================

"""
    wynn_epsilon(s) -> Vector{ComplexF64}

Wynn ε-algorithm applied to a scalar sequence `s = [s₀, s₁, ..., sK]`.
Returns the even-column first-row Padé estimates `[ε₂(0), ε₄(0), ...]`.
Uses only additions and reciprocals — no matrix operations.
"""
function wynn_epsilon(s::AbstractVector{<:Number})
    n   = length(s)
    eps = zeros(ComplexF64, n+1, n+1)
    for k in 1:n
        eps[2, k] = s[k]
    end
    for j in 1:n-1
        for k in 1:n-j
            d = eps[j+1, k+1] - eps[j+1, k]
            eps[j+2, k] = abs(d) < 1e-30 ? complex(1e30) : eps[j, k+1] + 1/d
        end
    end
    return [eps[j+2, 1] for j in 2:2:n-1]
end


"""
    get_rpa_susceptibility_wynn(H, MPOV, ωlist; K_max, maxdim_apply, cutoff_apply,
                                 verbose, <bubble kwargs>) -> (chi_partial, chi_wynn)

Compute the RPA susceptibility χ_RPA(q,ω) for all frequencies in `ωlist` using
the Wynn ε-algorithm for Padé acceleration of the geometric (bubble) series.

**Key idea**: instead of inverting (I − Π₀V), build the Neumann series
  T₀ = Π₀,  Tₙ = Tₙ₋₁·V·Π₀  (so Σ Tₙ → χ_RPA as K→∞),
extract scalars `sₙ(q,ω) = −Im⟨q|Tₙ(ω)|q⟩/π` via `get_spect_k`, and apply
Wynn ε to the partial-sum sequence per (q,ω) for fast convergence.

**Returns**
- `chi_partial[k+1, i_ω, q]` : partial sum Σₙ₌₀ᵏ sₙ(q,ω)
- `chi_wynn[m, i_ω, q]`      : Wynn ε_{2m}(0) estimate (uses 2m+1 terms)

**Keyword arguments**
- `K_max`         : highest order in the series (total K_max+1 terms). Default `6`.
- `maxdim_apply`  : bond dim for the Tₙ·V·Π₀ products. Default `200`.
- `cutoff_apply`  : truncation cutoff for those products. Default `1e-8`.
- `verbose`       : print per-ω progress. Default `false`.
- All `get_bubble_mpo` keywords (`ϵF`, `P_method`, `GF_method`, `Ncheb`, `a`,
  `Ncheb`, `maxdim`, `cutoff`, `purify_*`, `η`, `krylov_*`) are accepted and forwarded.
"""
function get_rpa_susceptibility_wynn(H::TBHamiltonian, MPOV::MPO,
                                      ωlist::AbstractVector{<:Real};
                                      K_max::Int         = 6,
                                      maxdim_apply::Int  = 200,
                                      cutoff_apply::Real = 1e-8,
                                      ϵF::Real              = 0.0,
                                      P_method::Symbol      = :purification,
                                      GF_method::Symbol     = :kpm,
                                      Ncheb::Int            = 150,
                                      maxdim::Int           = 200,
                                      cutoff::Real          = 1e-8,
                                      purify_method::Symbol = :mcweeny,
                                      purify_maxdim::Int    = 40,
                                      purify_maxiters::Int  = 30,
                                      purify_tol::Float64   = 1e-5,
                                      η::Real               = 1e-3,
                                      krylov_nsweeps::Int   = 12,
                                      krylov_maxdim::Int    = 100,
                                      krylov_cutoff::Real   = 1e-8,
                                      verbose::Bool         = false)

    nω     = length(ωlist)
    n_wynn = K_max ÷ 2

    chi_partial = nothing   # allocated on first ω once nq is known
    chi_wynn    = nothing

    for (i, ω) in enumerate(ωlist)
        verbose && println("Wynn RPA: ω $i/$nω  (ω = $ω)")

        Π0 = get_bubble_mpo(H, H, ω;
                             ϵF, P_method, GF_method, Ncheb, maxdim, cutoff,
                             purify_method, purify_maxdim, purify_maxiters, purify_tol,
                             η, krylov_nsweeps, krylov_maxdim, krylov_cutoff, verbose)

        term = deepcopy(Π0)
        s0   = -imag.(get_spect_k(term))

        if chi_partial === nothing
            nq          = length(s0)
            chi_partial = zeros(Float64, K_max+1, nω, nq)
            chi_wynn    = zeros(Float64, n_wynn,  nω, nq)
        end

        # Individual term contributions: spect_terms[n+1, i_ω, q]
        spect_terms          = zeros(Float64, K_max+1, nq)
        spect_terms[1, :]    = s0

        for n in 1:K_max
            term             = apply(term, MPOV; maxdim=maxdim_apply, cutoff=cutoff_apply)
            term             = apply(term, Π0;   maxdim=maxdim_apply, cutoff=cutoff_apply)
            spect_terms[n+1, :] = -imag.(get_spect_k(term))
        end

        # Partial sums (Wynn input)
        partial_sums = cumsum(spect_terms; dims=1)
        chi_partial[:, i, :] = partial_sums

        # Apply Wynn ε per q-point
        for q in 1:nq
            ests = wynn_epsilon(complex.(partial_sums[:, q]))
            for m in 1:min(n_wynn, length(ests))
                chi_wynn[m, i, q] = real(ests[m])
            end
        end

        verbose && println("  done ($(K_max+1) terms, $(n_wynn) Wynn estimates)")
    end

    return chi_partial, chi_wynn
end
