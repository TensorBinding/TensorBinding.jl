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

nsitelegs(T::ITensor) = count(i -> hastags(i, "Site"), inds(T))


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
    if H1 === H2
        P2 = P1
    else
        verbose && println("Polarization bubble: computing P2...")
        P2 = _get_density_matrix(H2, ϵF, P_method, Ncheb, maxdim, cutoff,
                                  purify_method, purify_maxdim, purify_maxiters,
                                  purify_tol, verbose)
    end

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
    get_rpa_susceptibility(H::TBHamiltonian, MPOV, ω; mode, ...) -> MPS

Compute the RPA susceptibility χ(ω) for a system described by `H` with
interaction MPO `MPOV`.  Returns a 2L-site MPS encoding the diagonal
χ_{iijj}^RPA.

**`mode` keyword**
- `:charge` (default) — density–density bubble χ^{ρρ}: calls
  `get_bubble_mpo(H, H, ω)`.
- `:magnetic` — transverse spin bubble χ^{+−}: projects H onto its
  spin-↑ and spin-↓ blocks and calls `get_magnon_bubble`.
  Requires `H.spin_s !== nothing`.

Internally solves the Dyson equation (I − Π₀V) χ = Π₀.
All `get_bubble_mpo` keyword arguments are accepted and forwarded.

**Additional keywords (Dyson solve)**
- `rpa_nsweeps` : sweeps for the RPA linsolve. Default `20`.
- `rpa_maxdim`  : max bond dim for the RPA linsolve. Default `400`.
- `rpa_cutoff`  : cutoff for the RPA linsolve. Default `1e-8`.
"""
function get_rpa_susceptibility(H::TBHamiltonian, MPOV::MPO, ω::Real;
                                 mode::Symbol          = :charge,
                                 rpa_nsweeps::Int      = 20,
                                 rpa_maxdim::Int       = 400,
                                 rpa_cutoff::Real      = 1e-8,
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

    bubble_kw = (; ϵF, P_method, GF_method, Ncheb, maxdim, cutoff,
                   purify_method, purify_maxdim, purify_maxiters, purify_tol,
                   η, krylov_nsweeps, krylov_maxdim, krylov_cutoff, verbose)

    if mode == :charge
        Π         = get_bubble_mpo(H, H, ω; bubble_kw...)
        out_sites = H.sites
    elseif mode == :magnetic
        H.spin_s === nothing &&
            error("get_rpa_susceptibility: mode=:magnetic requires a spinful H (call add_spin!(H) first)")
        H_up      = _project_spin_sector(H, 1)
        H_dn      = _project_spin_sector(H, 2)
        Π         = get_bubble_mpo(H_up, H_dn, ω; bubble_kw...)
        out_sites = H_up.sites
    else
        error("get_rpa_susceptibility: unknown mode=$mode. Choose :charge or :magnetic")
    end

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
    rpa_wynn_from_bubbles(Π0_list, MPOV; K_max, maxdim_apply, cutoff_apply, verbose)
    -> (chi_partial, chi_wynn)

Wynn ε-accelerated RPA susceptibility from a pre-computed list of bubble MPOs.

Accepts the output of `get_bubble_mpo_cheb2d_tucker` (or any `Vector{MPO}`) directly,
skipping the internal bubble calculation.  All other logic is identical to
`get_rpa_susceptibility_wynn`: Neumann series T₀ = Π₀, Tₙ = Tₙ₋₁·V·Π₀, followed
by per-k-point Wynn ε-acceleration of the partial-sum sequence.

**Returns**
- `chi_partial[k+1, iω, q]` : partial sum Σₙ₌₀ᵏ (−Im⟨q|Tₙ(ω)|q⟩)
- `chi_wynn[m, iω, q]`      : Wynn ε_{2m}(0) estimate

**Keyword arguments**
- `K_max`         : series order (K_max+1 terms). Default `6`.
- `maxdim_apply`  : bond dim for Tₙ·V·Π₀ products. Default `200`.
- `cutoff_apply`  : truncation cutoff for those products. Default `1e-8`.
- `verbose`       : print per-ω progress. Default `false`.
"""
function rpa_wynn_from_bubbles(Π0_list::Vector{<:MPO}, MPOV::MPO;
                                K_max::Int         = 6,
                                maxdim_apply::Int  = 200,
                                cutoff_apply::Real = 1e-8,
                                verbose::Bool      = false)
    nω     = length(Π0_list)
    n_wynn = K_max ÷ 2

    chi_partial = nothing
    chi_wynn    = nothing

    for (i, Π0) in enumerate(Π0_list)
        verbose && println("rpa_wynn_from_bubbles: bubble $i/$nω")
        term = deepcopy(Π0)
        s0   = -imag.(get_spect_k(term))

        if chi_partial === nothing
            nq          = length(s0)
            chi_partial = zeros(Float64, K_max+1, nω, nq)
            chi_wynn    = zeros(Float64, n_wynn,  nω, nq)
        end

        spect_terms       = zeros(Float64, K_max+1, length(s0))
        spect_terms[1, :] = s0

        for n in 1:K_max
            term                = apply(term, MPOV; maxdim=maxdim_apply, cutoff=cutoff_apply)
            term                = apply(term, Π0;   maxdim=maxdim_apply, cutoff=cutoff_apply)
            spect_terms[n+1, :] = -imag.(get_spect_k(term))
        end

        partial_sums         = cumsum(spect_terms; dims=1)
        chi_partial[:, i, :] = partial_sums

        for q in 1:length(s0)
            ests = wynn_epsilon(complex.(partial_sums[:, q]))
            for m in 1:min(n_wynn, length(ests))
                chi_wynn[m, i, q] = real(ests[m])
            end
        end

        verbose && println("  done ($(K_max+1) terms, $n_wynn Wynn estimates)")
    end

    return chi_partial, chi_wynn
end


# ============================================================
# Haydock recursion (operator-level Krylov)
# ============================================================

"""
    haydock_cf(H_mpo, seed, N_steps; maxdim, cutoff, verbose)
        -> (a, b, basis, norm0)

Haydock (Lanczos) recursion with H_mpo acting on MPO vectors from the left.
Starting from `seed`, builds an orthogonal Krylov basis under H_mpo using
the Frobenius (Hilbert-Schmidt) inner product (A, B) = Tr[A† B].

Three-term recurrence (Φ₀ = seed / β₀, β₀ = ||seed||_F):

    Φₙ₊₁ = H·Φₙ − aₙ·Φₙ − bₙ·Φₙ₋₁    (b₁ = 0)

Returns:
- `a`    : diagonal coefficients a[1..N]
- `b`    : b[1] = norm0 = ||seed||_F; b[2..N] = off-diagonal βₙ
- `basis`: normalized Krylov MPOs {Φ₀, …, Φₙ₋₁}
- `norm0`: sqrt(inner(seed, seed))

The scalar projected GF ⟨seed|(z−H)⁻¹|seed⟩ is recovered via
`eval_haydock_cf(a, b, z)`.  The full resolvent MPO (z−H)⁻¹|seed⟩ is
recovered via `haydock_resolve_mpo(a, b, basis, z)`.
"""
function haydock_cf(H_mpo::MPO, seed::MPO, N_steps::Int;
                    maxdim::Int   = 200,
                    cutoff::Real  = 1e-8,
                    verbose::Bool = false)

    a     = zeros(Float64, N_steps)
    b     = zeros(Float64, N_steps)
    basis = Vector{MPO}(undef, N_steps)

    norm0    = sqrt(real(tr(apply(dag(seed), seed; cutoff=cutoff, maxdim=maxdim))))
    b[1]     = norm0
    Phi_prev = nothing
    Phi_curr = (1.0 / norm0) * seed

    actual_N = N_steps
    for n in 1:N_steps
        basis[n] = Phi_curr

        HPhi = apply(H_mpo, Phi_curr; maxdim=maxdim, cutoff=cutoff)
        a[n] = real(tr(apply(dag(Phi_curr), HPhi; cutoff=cutoff, maxdim=maxdim)))

        r = +(HPhi, (-a[n]) * Phi_curr; maxdim=maxdim)
        ITensorMPS.truncate!(r; cutoff=cutoff)
        if n > 1
            r = +(r, (-b[n]) * Phi_prev; maxdim=maxdim)
            ITensorMPS.truncate!(r; cutoff=cutoff)
        end

        b_next = sqrt(max(0.0, real(tr(apply(dag(r), r; cutoff=cutoff, maxdim=maxdim)))))
        verbose && println("  step $n: a=$(round(a[n];digits=5))  b_next=$(round(b_next;digits=5))  chi=$(maxlinkdim(Phi_curr))")

        if b_next < 1e-12
            verbose && println("  haydock_cf: invariant subspace at step $n")
            actual_N = n
            break
        end

        Phi_prev = Phi_curr
        Phi_curr = (1.0 / b_next) * r
        n < N_steps && (b[n + 1] = b_next)
    end

    return a[1:actual_N], b[1:actual_N], basis[1:actual_N], norm0
end


"""
    eval_haydock_cf(a, b, z) -> ComplexF64

Evaluate the Haydock continued fraction ⟨seed|(z−H)⁻¹|seed⟩ via backward
recursion. `b[1]` must be norm0 = ||seed||_F (as returned by `haydock_cf`).

    G(z) = b[1]² / (z − a[1] − b[2]²/(z − a[2] − b[3]²/…))

Calling with truncated arrays a[1:N], b[1:N] gives the N-th CF convergent,
whose sequence over N is suitable for Wynn ε-acceleration.
"""
function eval_haydock_cf(a::AbstractVector, b::AbstractVector, z::Number)
    N = length(a)
    f = ComplexF64(z) - a[N]
    for n in N-1:-1:1
        f = ComplexF64(z) - a[n] - b[n + 1]^2 / f
    end
    return b[1]^2 / f
end


"""
    haydock_resolve_mpo(a, b, basis, z; maxdim, cutoff) -> MPO

Reconstruct (z−H)⁻¹|seed⟩ as an MPO by solving the N×N Lanczos tridiagonal
system and forming a linear combination of the Krylov basis MPOs:

    (z·I − T) c = b[1]·e₁,   Π₀(z) = Σₙ c[n]·basis[n]

where T has diagonal `a` and off-diagonal `b[2:]`, and b[1] = norm0.
"""
function haydock_resolve_mpo(a::AbstractVector, b::AbstractVector,
                              basis::Vector{<:MPO}, z::Number;
                              maxdim::Int  = 200,
                              cutoff::Real = 1e-8)
    N  = length(a)
    zc = ComplexF64(z)
    d  = [zc - a[n] for n in 1:N]
    ev = N > 1 ? ComplexF64[-b[n] for n in 2:N] : ComplexF64[]
    T  = Tridiagonal(ev, d, ev)
    rhs       = zeros(ComplexF64, N)
    rhs[1]    = b[1]
    c         = T \ rhs

    result = c[1] * basis[1]
    for n in 2:N
        result = +(result, c[n] * basis[n]; maxdim=maxdim)
        ITensorMPS.truncate!(result; cutoff=cutoff)
    end
    return result
end


"""
    get_bubble_mpo_haydock(H1, H2, ωlist; N_steps, η, maxdim, cutoff,
                            ϵF, P_method, purify_method, purify_maxdim,
                            purify_maxiters, purify_tol, Ncheb, verbose)
        -> Vector{MPO}

Compute the bare polarization bubble Π₀(ω) as an L-site MPO for each
frequency in `ωlist` using Haydock recursion on H_eff = I⊗H₂ − H₁⊗I.

The Krylov basis is built once from the seed Φ₀ = I⊗P₂ − P₁⊗I.  For each
ω, the resolvent (ω+iη−H_eff)⁻¹Φ₀ is recovered by solving the N×N Lanczos
tridiagonal system and forming a linear combination of the stored basis MPOs.

Returns a `Vector{MPO}` compatible with `rpa_wynn_from_bubbles`.

**Keyword arguments**
- `N_steps`        : Haydock recursion depth. Default `30`.
- `η`              : Lorentzian broadening. Default `1e-2`.
- `maxdim`         : Maximum bond dimension. Default `200`.
- `cutoff`         : SVD truncation cutoff. Default `1e-8`.
- `ϵF`             : Fermi energy. Default `0.0`.
- `P_method`       : `:purification` (default) or `:kpm`.
- `purify_method`  : `:mcweeny` (default) or `:sp2`.
- `purify_maxdim`  : Max bond dim during purification. Default `40`.
- `purify_maxiters`: Max purification iterations. Default `30`.
- `purify_tol`     : Purification convergence tolerance. Default `1e-5`.
- `Ncheb`          : Chebyshev order for `:kpm` P_method. Default `150`.
- `verbose`        : Print progress. Default `false`.
"""
function get_bubble_mpo_haydock(H1::TBHamiltonian, H2::TBHamiltonian,
                                  ωlist::AbstractVector{<:Real};
                                  N_steps::Int          = 30,
                                  η::Real               = 1e-2,
                                  maxdim::Int           = 200,
                                  cutoff::Real          = 1e-8,
                                  ϵF::Real              = 0.0,
                                  P_method::Symbol      = :purification,
                                  purify_method::Symbol = :mcweeny,
                                  purify_maxdim::Int    = 40,
                                  purify_maxiters::Int  = 30,
                                  purify_tol::Float64   = 1e-5,
                                  Ncheb::Int            = 150,
                                  verbose::Bool         = false)

    L1 = H1.L; L2 = H2.L
    @assert L1 == L2 "H1 and H2 must have the same number of sites (got $L1 vs $L2)"
    L              = L1
    sites_combined = vcat(H1.sites, H2.sites)

    # ---- Density matrices ----
    verbose && println("Haydock bubble: computing P1 (P_method=$P_method)...")
    P1 = _get_density_matrix(H1, ϵF, P_method, Ncheb, maxdim, cutoff,
                              purify_method, purify_maxdim, purify_maxiters, purify_tol, verbose)
    verbose && println("Haydock bubble: computing P2...")
    P2 = _get_density_matrix(H2, ϵF, P_method, Ncheb, maxdim, cutoff,
                              purify_method, purify_maxdim, purify_maxiters, purify_tol, verbose)

    # ---- Seed: I⊗P₂ − P₁⊗I on 2L-site combined space ----
    id1  = MPO(H1.sites, "Id"); id2 = MPO(H2.sites, "Id")
    P1op = interleave_mpo(P1,  sites_combined, 0)
    Iop2 = interleave_mpo(id2, sites_combined, 1)
    Iop1 = interleave_mpo(id1, sites_combined, 0)
    P2op = interleave_mpo(P2,  sites_combined, 1)
    seed = ITensorMPS.truncate!(
        apply(Iop1, P2op; maxdim=maxdim, cutoff=cutoff) -
        apply(P1op, Iop2; maxdim=maxdim, cutoff=cutoff); cutoff=cutoff)
    verbose && println("Haydock bubble: seed built, chi=$(maxlinkdim(seed))")

    # ---- H_eff = I⊗H₂ − H₁⊗I ----
    Heff = _build_heff(H1.mpo, H2.mpo, sites_combined)
    verbose && println("Haydock bubble: H_eff built, chi=$(maxlinkdim(Heff))")

    # ---- Haydock recursion (once, independent of ω) ----
    verbose && println("Haydock bubble: running $N_steps steps...")
    a, b, basis, norm0 = haydock_cf(Heff, seed, N_steps;
                                     maxdim=maxdim, cutoff=cutoff, verbose=verbose)
    verbose && println("Haydock bubble: $(length(a)) steps completed, norm0=$(round(norm0;digits=4))")

    # ---- Assemble Π₀(ω) for each frequency ----
    finalsites = siteinds("Qubit", 2L)
    bubbles    = Vector{MPO}(undef, length(ωlist))
    for (i, ω) in enumerate(ωlist)
        verbose && println("Haydock bubble: assembling Pi0 at omega=$ω ($i/$(length(ωlist)))...")
        z          = ComplexF64(ω + im * η)
        b2L        = haydock_resolve_mpo(a, b, basis, z; maxdim=maxdim, cutoff=cutoff)
        biv        = swap_every_other_legs(b2L, finalsites)
        bubbles[i] = collapse_mpo_pairs(biv, H1.sites)
    end

    return bubbles
end


"""
    get_spect_k(W; tol, maxdim) -> Vector{ComplexF64}

Extract the k-space diagonal of MPO `W` as a dense vector of 2^L values.

Conjugates `W` by the QFT (giving W̃ = QFT·W·QFT†), extracts the diagonal
as an MPS, then evaluates each ⟨k|W̃|k⟩ for the 2^L quantics k-indices.
Uses LSB-first quantics convention (site 1 = least significant bit).
"""
function get_spect_k(W::MPO; tol::Real=1e-9, maxdim::Int=100)
    Wk   = conjugate_by_qft(W; tol=tol, maxdim=maxdim)
    diag = extract_diagonal_to_mps(Wk)
    L    = length(diag)
    N    = 2^L
    sd   = siteinds(diag)
    return ComplexF64[inner(MPS(sd, [string((k >> (i-1)) & 1) for i in 1:L]), diag)
                      for k in 0:N-1]
end


"""
    get_rpa_susceptibility_wynn(H, MPOV, ωlist; mode, K_max, maxdim_apply,
                                 cutoff_apply, verbose, <bubble kwargs>)
                                 -> (chi_partial, chi_wynn)

Compute the RPA susceptibility χ_RPA(q,ω) for all frequencies in `ωlist` using
the Wynn ε-algorithm for Padé acceleration of the geometric (bubble) series.

**`mode` keyword**
- `:charge` (default) — density–density channel; uses `get_bubble_mpo(H, H, ω)`.
- `:magnetic` — transverse spin channel S⁺S⁻; uses `get_magnon_bubble(H, ω)`.
  The spin-↑/↓ projections are performed once before the ω loop.
  Requires `H.spin_s !== nothing`.

**Key idea**: instead of inverting (I − Π₀V), build the Neumann series
  T₀ = Π₀,  Tₙ = Tₙ₋₁·V·Π₀  (so Σ Tₙ → χ_RPA as K→∞),
extract scalars `sₙ(q,ω) = −Im⟨q|Tₙ(ω)|q⟩/π` via `get_spect_k`, and apply
Wynn ε to the partial-sum sequence per (q,ω) for fast convergence.

**Returns**
- `chi_partial[k+1, i_ω, q]` : partial sum Σₙ₌₀ᵏ sₙ(q,ω)
- `chi_wynn[m, i_ω, q]`      : Wynn ε_{2m}(0) estimate (uses 2m+1 terms)

**Keyword arguments**
- `mode`          : `:charge` (default) or `:magnetic`.
- `K_max`         : highest order in the series (total K_max+1 terms). Default `6`.
- `maxdim_apply`  : bond dim for the Tₙ·V·Π₀ products. Default `200`.
- `cutoff_apply`  : truncation cutoff for those products. Default `1e-8`.
- `verbose`       : print per-ω progress. Default `false`.
- All `get_bubble_mpo` keywords (`ϵF`, `P_method`, `GF_method`, `Ncheb`,
  `maxdim`, `cutoff`, `purify_*`, `η`, `krylov_*`) are accepted and forwarded.
"""
function get_rpa_susceptibility_wynn(H::TBHamiltonian, MPOV::MPO,
                                      ωlist::AbstractVector{<:Real};
                                      mode::Symbol       = :charge,
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

    mode ∈ (:charge, :magnetic) ||
        error("get_rpa_susceptibility_wynn: unknown mode=$mode. Choose :charge or :magnetic")
    mode == :magnetic && H.spin_s === nothing &&
        error("get_rpa_susceptibility_wynn: mode=:magnetic requires a spinful H (call add_spin!(H) first)")

    bubble_kw = (; ϵF, P_method, GF_method, Ncheb, maxdim, cutoff,
                   purify_method, purify_maxdim, purify_maxiters, purify_tol,
                   η, krylov_nsweeps, krylov_maxdim, krylov_cutoff, verbose)

    # For :magnetic, project spin sectors once before the ω loop
    H_up = mode == :magnetic ? _project_spin_sector(H, 1) : nothing
    H_dn = mode == :magnetic ? _project_spin_sector(H, 2) : nothing

    nω     = length(ωlist)
    n_wynn = K_max ÷ 2

    chi_partial = nothing
    chi_wynn    = nothing

    for (i, ω) in enumerate(ωlist)
        verbose && println("Wynn RPA (mode=$mode): ω $i/$nω  (ω = $ω)")

        Π0 = if mode == :charge
            get_bubble_mpo(H, H, ω; bubble_kw...)
        else
            get_bubble_mpo(H_up, H_dn, ω; bubble_kw...)
        end

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

# ============================================================
# Magnon susceptibility (transverse S⁺S⁻ spin channel)
# ============================================================

"""
    _project_spin_sector(H, sector) -> TBHamiltonian

Project a spinful `TBHamiltonian` onto spin sector `sector` (1 = ↑, 2 = ↓)
by contracting the spin site tensor with the projector |sector⟩⟨sector|.

The spin index is identified by its "Spin" tag, so the function is robust
to whether spin is prepended or postpended.  The contracted tensor is
absorbed into its neighbour, leaving a valid L-qubit MPO.

Returns a new `TBHamiltonian` with `spin_s = nothing` and fresh (empty)
caches; `scale` and `center` are reset to 0.0 so `_ensure_scale!` will
re-estimate them on the first KPM call.
"""
function _project_spin_sector(H::TBHamiltonian, sector::Int)
    H.spin_s === nothing &&
        error("_project_spin_sector: H is not spinful (spin_s is nothing)")
    s = H.spin_s

    spin_pos = findfirst(n -> any(i -> hastags(i, "Spin"), siteinds(H.mpo, n)),
                         1:length(H.mpo))
    spin_pos === nothing && error("_project_spin_sector: spin Index not found in MPO")

    proj = ITensor(ComplexF64, s', s)
    proj[s' => sector, s => sector] = 1.0

    tensors    = ITensor[H.mpo[i] for i in 1:length(H.mpo)]
    contracted = tensors[spin_pos] * proj   # only link indices remain

    if spin_pos == 1
        tensors[2]  = contracted * tensors[2]
        new_tensors = tensors[2:end]
    else
        tensors[spin_pos - 1] = tensors[spin_pos - 1] * contracted
        new_tensors = tensors[1:spin_pos - 1]
    end

    new_sites = filter(i -> !hastags(i, "Spin"), H.sites)

    return TBHamiltonian(
        H.L, H.N, new_sites, MPO(new_tensors),
        H.geometry, H.geometry_uc,
        0.0, 0.0,
        nothing, H.nambu_s, H.layer_s, H.sublattice_s,
        H.aux_side,
        nothing, nothing, 0, nothing
    )
end


"""
    get_magnon_bubble(H, ω; ...) -> MPO

Non-interacting transverse spin polarization bubble Π₀^{+−}(ω) for a
spinful `TBHamiltonian`.

The spin degree of freedom is projected out analytically: the spin-↑ and
spin-↓ blocks of `H` become two independent L-qubit Hamiltonians `H_↑`
and `H_↓`, which are passed to `get_bubble_mpo(H_↑, H_↓, ω)`.
This corresponds to the Kubo S⁺S⁻ bubble

    Π₀^{+−}(ω) = ∑_k (f_{k↓} − f_{k↑}) / (ω − (ε_{k↑} − ε_{k↓}) + iη)

Errors if `H.spin_s === nothing`.  All `get_bubble_mpo` keyword arguments
are accepted and forwarded.
"""
function get_magnon_bubble(H::TBHamiltonian, ω::Real; kwargs...)
    H.spin_s === nothing &&
        error("get_magnon_bubble: H is not spinful — call add_spin!(H) first")
    H_up = _project_spin_sector(H, 1)
    H_dn = _project_spin_sector(H, 2)
    return get_bubble_mpo(H_up, H_dn, ω; kwargs...)
end


"""
    get_magnon_susceptibility(H, MPOV, ω; ...) -> MPS

RPA transverse spin susceptibility χ^{+−}_RPA(ω) for a spinful
`TBHamiltonian`.

Builds Π₀^{+−}(ω) via `get_magnon_bubble`, then solves the Dyson
equation (I − Π₀ V) χ = Π₀.  For a Hubbard-like interaction the
interaction MPO is `MPOV = U · Id` on the orbital sites.

**Keyword arguments**
- `rpa_nsweeps`, `rpa_maxdim`, `rpa_cutoff` : Dyson linsolve parameters.
- All `get_bubble_mpo` keywords forwarded via `kwargs...`.
"""
function get_magnon_susceptibility(H::TBHamiltonian, MPOV::MPO, ω::Real;
                                   rpa_nsweeps::Int = 20,
                                   rpa_maxdim::Int  = 400,
                                   rpa_cutoff::Real = 1e-8,
                                   kwargs...)
    H.spin_s === nothing &&
        error("get_magnon_susceptibility: H is not spinful — call add_spin!(H) first")
    H_up = _project_spin_sector(H, 1)
    H_dn = _project_spin_sector(H, 2)
    Π = get_bubble_mpo(H_up, H_dn, ω; kwargs...)
    finalsites = siteinds("Qubit", 2 * H.L)
    return rpa_from_bubble_diag(Π, MPOV, finalsites, H_up.sites;
                                nsweeps=rpa_nsweeps, maxdim=rpa_maxdim, cutoff=rpa_cutoff)
end


"""
    get_magnon_susceptibility_wynn(H, MPOV, ωlist; K_max, ...) -> (chi_partial, chi_wynn)

Wynn ε-accelerated transverse spin RPA susceptibility over a frequency list.

The spin-↑/↓ sector projections are performed once outside the ω loop;
the Neumann-series / Wynn logic is identical to `get_rpa_susceptibility_wynn`.

Returns `(chi_partial, chi_wynn)` with the same layout as
`get_rpa_susceptibility_wynn`.

**Keyword arguments**
- `K_max`, `maxdim_apply`, `cutoff_apply`, `verbose` : series / Wynn control.
- All `get_bubble_mpo` keywords forwarded via `kwargs...`.
"""
function get_magnon_susceptibility_wynn(H::TBHamiltonian, MPOV::MPO,
                                         ωlist::AbstractVector{<:Real};
                                         K_max::Int         = 6,
                                         maxdim_apply::Int  = 200,
                                         cutoff_apply::Real = 1e-8,
                                         verbose::Bool      = false,
                                         kwargs...)
    H.spin_s === nothing &&
        error("get_magnon_susceptibility_wynn: H is not spinful — call add_spin!(H) first")

    H_up = _project_spin_sector(H, 1)
    H_dn = _project_spin_sector(H, 2)

    nω     = length(ωlist)
    n_wynn = K_max ÷ 2

    chi_partial = nothing
    chi_wynn    = nothing

    for (i, ω) in enumerate(ωlist)
        verbose && println("Magnon Wynn RPA: ω $i/$nω  (ω = $ω)")

        Π0   = get_bubble_mpo(H_up, H_dn, ω; verbose, kwargs...)
        term = deepcopy(Π0)
        s0   = -imag.(get_spect_k(term))

        if chi_partial === nothing
            nq          = length(s0)
            chi_partial = zeros(Float64, K_max + 1, nω, nq)
            chi_wynn    = zeros(Float64, n_wynn,    nω, nq)
        end

        spect_terms       = zeros(Float64, K_max + 1, nq)
        spect_terms[1, :] = s0

        for n in 1:K_max
            term             = apply(term, MPOV; maxdim=maxdim_apply, cutoff=cutoff_apply)
            term             = apply(term, Π0;   maxdim=maxdim_apply, cutoff=cutoff_apply)
            spect_terms[n+1, :] = -imag.(get_spect_k(term))
        end

        partial_sums         = cumsum(spect_terms; dims=1)
        chi_partial[:, i, :] = partial_sums

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

# ============================================================
# Double Chebyshev decomposition for the polarization bubble
# ============================================================

"""
    chebyshev2d_gf_coeffs(ω, scale1, center1, scale2, center2, η, N) -> Matrix{ComplexF64}

Compute 2D Chebyshev expansion coefficients c_{mn} for the scalar Green's function

    f(x, y) = 1 / (ω + iη − (scale2·y + center2 − scale1·x − center1))

on the domain [-1,1]×[-1,1], using an N×N Chebyshev-Gauss grid and 2D DCT-II.

`C[m+1, n+1]` = c_{mn} in the expansion  f(x,y) ≈ Σ_{m,n} c_{mn} T_m(x) T_n(y).

`N` should equal `length(Tn_list)` from `KPM_Tn` (i.e. `Ncheb + 1`).
"""
function chebyshev2d_gf_coeffs(ω::Real, scale1::Real, center1::Real,
                                 scale2::Real, center2::Real,
                                 η::Real, N::Int)
    j     = 0:N-1
    nodes = cos.(π .* (j .+ 0.5) ./ N)
    ω_eff = ω - center2 + center1 + im * η
    F = [1.0 / (ω_eff + scale1 * nodes[j1+1] - scale2 * nodes[k1+1])
         for j1 in 0:N-1, k1 in 0:N-1]
    Cr = FFTW.r2r(real.(F), FFTW.REDFT10, [1, 2])
    Ci = FFTW.r2r(imag.(F), FFTW.REDFT10, [1, 2])
    C  = (Cr .+ im .* Ci) ./ (2N)^2
    C[1, :] ./= 2   # m = 0 row
    C[:, 1] ./= 2   # n = 0 column
    return C
end




"""
    get_bubble_mpo_cheb2d(H1, H2, ωlist; Ncheb, maxdim, cutoff,
                           ϵF, P_method, purify_*, η, verbose) -> Vector{MPO}

Compute the non-interacting polarization bubble Π₀(ω) for each ω in `ωlist`
using the **double Chebyshev decomposition**.

Instead of building the 2L-site effective Hamiltonian Heff = I⊗H₂ − H₁⊗I and
running KPM on it (where bond dimension grows at each Chebyshev step due to
entanglement between subsystems), this routine decomposes G_eff as

    G_eff(ω) ≈ Σ_{mn} c_{mn}(ω) · T_m(H̃₁) ⊗ T_n(H̃₂)

where T_m, T_n are Chebyshev polynomials of the *L-site* rescaled Hamiltonians
and c_{mn}(ω) are scalar 2D Chebyshev coefficients (cheap, via DCT-II).

The bubble on L-site MPOs is assembled as

    Π₀(ω) = Σ_{mn} c_{mn}(ω) · D_{mn}

where D_{mn} = (T_m(H̃₁)·P₁) ⊙ T_n(H̃₂) − T_m(H̃₁) ⊙ (T_n(H̃₂)·P₂)
and ⊙ is the site-wise Hadamard product (`hadamard_mpo`).

**Online multi-ω sweep**: All coefficient matrices C[m,n](ω) are precomputed
at once (cheap DCT scalars). The (m,n) double loop runs once; each D_{mn} is
computed once and accumulated into every Π(ω) simultaneously using the scalar
c_{mn}(ω). This matches the KPM "online" paradigm: the expensive MPO work
(Hadamard products) is done once and shared across all frequencies.

**Keyword arguments**
- `Ncheb`         : Chebyshev expansion order. Default `50`.
- `maxdim`        : Max bond dimension throughout. Default `200`.
- `cutoff`        : SVD truncation cutoff. Default `1e-8`.
- `ϵF`            : Fermi energy. Default `0.0`.
- `P_method`      : `:purification` (default) or `:kpm`.
- `purify_method` : `:mcweeny` (default) or `:sp2`.
- `purify_maxdim`, `purify_maxiters`, `purify_tol` : purification controls.
- `η`             : Lorentzian broadening. Default `1e-3`.
- `coeff_tol`     : Skip (m,n) pairs where |C[m,n]| < coeff_tol, and entire m rows
                    where the row maximum is below coeff_tol. For smooth integrands
                    (large η) this prunes most of the N² terms at negligible accuracy cost.
                    Default `1e-12`.
- `verbose`       : Print progress. Default `false`.
"""
function get_bubble_mpo_cheb2d(H1::TBHamiltonian, H2::TBHamiltonian,
                                ωlist::AbstractVector{<:Real};
                                Ncheb::Int            = 50,
                                maxdim::Int           = 200,
                                cutoff::Real          = 1e-8,
                                ϵF::Real              = 0.0,
                                P_method::Symbol      = :purification,
                                purify_method::Symbol = :mcweeny,
                                purify_maxdim::Int    = 40,
                                purify_maxiters::Int  = 30,
                                purify_tol::Float64   = 1e-5,
                                η::Real               = 1e-3,
                                coeff_tol::Real       = 1e-12,
                                verbose::Bool         = false)
    L1 = H1.L; L2 = H2.L
    @assert L1 == L2 "get_bubble_mpo_cheb2d: H1 and H2 must have the same number of sites (got $L1 vs $L2)"
    L = L1

    _ensure_scale!(H1)
    _ensure_scale!(H2)
    scale1  = H1.scale;  center1 = H1.center
    scale2  = H2.scale;  center2 = H2.center

    verbose && println("cheb2d: building T_n(H1) moments (Ncheb=$Ncheb)...")
    Tn1, _, _ = KPM_Tn(H1.mpo, Ncheb, H1.sites;
                         scale=scale1, center=center1,
                         maxdim=maxdim, cutoff=cutoff, verbose=false)
    if H1 === H2
        Tn2 = Tn1
    else
        verbose && println("cheb2d: building T_n(H2) moments...")
        Tn2, _, _ = KPM_Tn(H2.mpo, Ncheb, H2.sites;
                             scale=scale2, center=center2,
                             maxdim=maxdim, cutoff=cutoff, verbose=false)
    end
    N = length(Tn1)   # = Ncheb + 1  (T_0 … T_Ncheb)

    verbose && println("cheb2d: computing P1...")
    P1 = _get_density_matrix(H1, ϵF, P_method, Ncheb, maxdim, cutoff,
                              purify_method, purify_maxdim, purify_maxiters,
                              purify_tol, verbose)
    if H1 === H2
        P2 = P1
    else
        verbose && println("cheb2d: computing P2...")
        P2 = _get_density_matrix(H2, ϵF, P_method, Ncheb, maxdim, cutoff,
                                  purify_method, purify_maxdim, purify_maxiters,
                                  purify_tol, verbose)
    end

    verbose && println("cheb2d: precomputing T_m(H1)·P1 and T_n(H2)·P2...")
    TP1 = [ITensorMPS.truncate!(
               apply(Tn1[m], P1; maxdim=maxdim, cutoff=cutoff); cutoff=cutoff)
           for m in 1:N]
    TP2 = [ITensorMPS.truncate!(
               apply(Tn2[n], P2; maxdim=maxdim, cutoff=cutoff); cutoff=cutoff)
           for n in 1:N]

    # Fresh physical indices shared by all Hadamard product calls
    out_sites = siteinds("Qubit", L)
    nω        = length(ωlist)

    # --- Online multi-ω: precompute all coefficient matrices at once, ---
    # --- then sweep (m,n) once and accumulate into every Π(ω).       ---
    verbose && println("cheb2d: precomputing C[m,n](ω) for all $nω frequencies...")
    C_all = [chebyshev2d_gf_coeffs(ω, scale1, center1, scale2, center2, η, N)
             for ω in ωlist]

    Π = Vector{Union{Nothing, MPO}}(nothing, nω)
    n_computed = 0
    n_skipped  = 0

    for m in 1:N
        # Row-level skip: if |C[m,n]| < coeff_tol for ALL n and ALL ω, skip
        max_row = maximum(maximum(abs, @view C[m, :]) for C in C_all)
        if max_row < coeff_tol
            n_skipped += N
            continue
        end

        for n in 1:N
            # Pair-level skip: negligible for every ω → no MPO work needed
            max_c = maximum(abs(C[m, n]) for C in C_all)
            if max_c < coeff_tol
                n_skipped += 1
                continue
            end
            n_computed += 1

            # D_mn = TP1[m] ⊙ Tn2[n] − Tn1[m] ⊙ TP2[n]  (ω-independent)
            had_A = hadamard_mpo(TP1[m], Tn2[n], out_sites; maxdim=maxdim, cutoff=cutoff)
            had_B = hadamard_mpo(Tn1[m], TP2[n], out_sites; maxdim=maxdim, cutoff=cutoff)
            D_mn  = ITensorMPS.truncate!(+(had_A, -1 * had_B; maxdim=maxdim); cutoff=cutoff)

            # Accumulate c_mn(ω) · D_mn into each Π(ω) simultaneously
            for (iω, C) in enumerate(C_all)
                c = C[m, n]
                abs(c) < coeff_tol && continue
                if Π[iω] === nothing
                    Π[iω] = c * D_mn
                else
                    Π[iω] = +(Π[iω], c * D_mn; maxdim=maxdim)
                    ITensorMPS.truncate!(Π[iω]; cutoff=cutoff)
                end
            end
        end

        verbose && println("  m=$m/$N  (computed $n_computed, skipped $n_skipped so far)")
    end

    verbose && println("cheb2d: done — $(n_computed)/$(N*N) (m,n) pairs computed, $n_skipped skipped")

    # Map output physical indices (out_sites) back to H1.sites
    return [replace_sites(Π[iω], H1.sites) for iω in 1:nω]
end


"""
    get_bubble_mpo_cheb2d_tucker(H1, H2, ωlist; Ncheb, maxdim, cutoff,
                                  ϵF, P_method, purify_*, η, coeff_tol,
                                  tucker_tol, tucker_maxrank, kernel,
                                  hooi_iters, verbose) -> Vector{MPO}

Tucker-accelerated variant of `get_bubble_mpo_cheb2d`.

Returns the full non-interacting polarization bubble Π₀(ω) as an MPO at each
frequency in `ωlist`, suitable for RPA resummation and Wynn acceleration on the
Dyson geometric series.

The Tucker-2 decomposition of the stacked coefficient tensor finds global bases
U_m (N×r_m) and V_n (N×r_n) satisfying

    C[m,n](ω) ≈ Σ_{s₁,s₂} G[s₁,s₂,ω] · (U_m)_{ms₁} · conj((V_n)_{ns₂})

with C ≈ U_m G(ω) V_n†.  The r_m + r_n frequency-independent weighted MPO sums
and the r_m × r_n Hadamard products are computed once; per-ω cost is only
r_m × r_n cheap scalar-weighted MPO additions.

Speedup over `get_bubble_mpo_cheb2d`: N²→r_m·r_n Hadamard products.

**Additional keyword arguments** (beyond `get_bubble_mpo_cheb2d`):
- `tucker_tol`    : relative singular-value cutoff for both mode SVDs. Default `1e-3`.
- `tucker_maxrank`: hard cap on r_m and r_n. Default `20`.
- `kernel`        : `:jackson` (default) or `:none`. Jackson damping reduces Tucker rank.
- `hooi_iters`    : HOOI refinement iterations after HOSVD initialisation. Default `3`.
"""
function get_bubble_mpo_cheb2d_tucker(H1::TBHamiltonian, H2::TBHamiltonian,
                                       ωlist::AbstractVector{<:Real};
                                       Ncheb::Int            = 50,
                                       maxdim::Int           = 200,
                                       cutoff::Real          = 1e-8,
                                       ϵF::Real              = 0.0,
                                       P_method::Symbol      = :purification,
                                       purify_method::Symbol = :mcweeny,
                                       purify_maxdim::Int    = 40,
                                       purify_maxiters::Int  = 30,
                                       purify_tol::Float64   = 1e-5,
                                       η::Real               = 1e-3,
                                       coeff_tol::Real       = 1e-12,
                                       tucker_tol::Real      = 1e-3,
                                       tucker_maxrank::Int   = 20,
                                       kernel::Symbol        = :jackson,
                                       hooi_iters::Int       = 3,
                                       verbose::Bool         = false)
    L1 = H1.L; L2 = H2.L
    @assert L1 == L2 "get_bubble_mpo_cheb2d_tucker: H1 and H2 must have the same number of sites (got $L1 vs $L2)"
    L  = L1
    nω = length(ωlist)

    _ensure_scale!(H1); _ensure_scale!(H2)
    scale1 = H1.scale; center1 = H1.center
    scale2 = H2.scale; center2 = H2.center

    verbose && println("cheb2d_mpo_tucker: building Chebyshev moments (Ncheb=$Ncheb)...")
    Tn1, _, _ = KPM_Tn(H1.mpo, Ncheb, H1.sites;
                        scale=scale1, center=center1,
                        maxdim=maxdim, cutoff=cutoff, verbose=false)
    if H1 === H2
        Tn2 = Tn1
    else
        Tn2, _, _ = KPM_Tn(H2.mpo, Ncheb, H2.sites;
                            scale=scale2, center=center2,
                            maxdim=maxdim, cutoff=cutoff, verbose=false)
    end
    N = length(Tn1)

    verbose && println("cheb2d_mpo_tucker: computing density matrices...")
    P1 = _get_density_matrix(H1, ϵF, P_method, Ncheb, maxdim, cutoff,
                             purify_method, purify_maxdim, purify_maxiters,
                             purify_tol, verbose)
    P2 = H1 === H2 ? P1 : _get_density_matrix(H2, ϵF, P_method, Ncheb, maxdim, cutoff,
                                               purify_method, purify_maxdim, purify_maxiters,
                                               purify_tol, verbose)

    out_sites = siteinds("Qubit", L)

    verbose && println("cheb2d_mpo_tucker: computing coefficient matrices for $nω frequencies...")
    C_all = [chebyshev2d_gf_coeffs(ω, scale1, center1, scale2, center2, η, N)
             for ω in ωlist]

    if kernel == :jackson
        g_jk  = _jackson_kernel(N)
        G_jk  = g_jk * g_jk'
        C_all = [G_jk .* C for C in C_all]
        verbose && println("cheb2d_mpo_tucker: Jackson kernel applied")
    elseif kernel != :none
        error("get_bubble_mpo_cheb2d_tucker: unknown kernel=$kernel (use :jackson or :none)")
    end

    # ── Tucker bases: HOSVD initialisation + HOOI refinement ────────────────
    T1 = hcat(C_all...)
    T2 = hcat([transpose(C) for C in C_all]...)
    F1 = svd(T1); F2 = svd(T2)
    r_m = min(tucker_maxrank, sum(F1.S .> tucker_tol * F1.S[1]))
    r_n = min(tucker_maxrank, sum(F2.S .> tucker_tol * F2.S[1]))
    U_m = F1.U[:, 1:r_m]
    V_n = F2.U[:, 1:r_n]

    for _ in 1:hooi_iters
        Y   = hcat([C * V_n  for C in C_all]...)
        U_m = svd(Y).U[:, 1:r_m]
        Z   = hcat([C' * U_m for C in C_all]...)
        V_n = svd(Z).U[:, 1:r_n]
    end
    verbose && println("cheb2d_mpo_tucker: Tucker ranks r_m=$r_m, r_n=$r_n (HOSVD + $hooi_iters HOOI iters) → $(r_m*r_n) Hadamard operations")

    # ── Core tensor G[s₁,s₂,ω] = (U_m† C(ω) V_n)[s₁,s₂] ──────────────────
    A_core = zeros(ComplexF64, r_m, r_n, nω)
    for iω in 1:nω
        A_core[:, :, iω] = U_m' * C_all[iω] * V_n
    end

    # ── ω-independent weighted MPO sums ──────────────────────────────────────
    # Sum bare Chebyshev moments first, then apply P once per component.
    # This costs r_m + r_n MPO-MPO multiplications total, vs 2N for the plain
    # variant that precomputes TP1[m] = Tn1[m]·P1 for all N moments.
    verbose && println("cheb2d_mpo_tucker: computing Tucker MPO components (r_m=$r_m, r_n=$r_n)...")
    C_tuck = [_weighted_mpo_sum(U_m[:, s1],        Tn1; maxdim=maxdim, cutoff=cutoff) for s1 in 1:r_m]
    B_tuck = [_weighted_mpo_sum(conj.(V_n[:, s2]), Tn2; maxdim=maxdim, cutoff=cutoff) for s2 in 1:r_n]
    A_tuck = [isnothing(C_tuck[s1]) ? nothing :
              ITensorMPS.truncate!(apply(C_tuck[s1], P1; maxdim=maxdim, cutoff=cutoff); cutoff=cutoff)
              for s1 in 1:r_m]
    E_tuck = [isnothing(B_tuck[s2]) ? nothing :
              ITensorMPS.truncate!(apply(B_tuck[s2], P2; maxdim=maxdim, cutoff=cutoff); cutoff=cutoff)
              for s2 in 1:r_n]

    # ── ω-independent Hadamard products: r_m × r_n total ────────────────────
    # D[s₁,s₂] = (A_tuck[s₁] ⊙ B_tuck[s₂]) − (C_tuck[s₁] ⊙ E_tuck[s₂])
    #           = Σ_{m,n} U[m,s₁] conj(V[n,s₂]) · D_mn   (ω-independent MPO)
    verbose && println("cheb2d_mpo_tucker: computing $(r_m*r_n) Hadamard products...")
    D_tuck = Matrix{Union{Nothing, MPO}}(nothing, r_m, r_n)
    for s1 in 1:r_m, s2 in 1:r_n
        (isnothing(A_tuck[s1]) || isnothing(B_tuck[s2]) ||
         isnothing(C_tuck[s1]) || isnothing(E_tuck[s2])) && continue

        had_A = hadamard_mpo(A_tuck[s1], B_tuck[s2], out_sites; maxdim=maxdim, cutoff=cutoff)
        had_B = hadamard_mpo(C_tuck[s1], E_tuck[s2], out_sites; maxdim=maxdim, cutoff=cutoff)
        D_tuck[s1, s2] = ITensorMPS.truncate!(+(had_A, -1 * had_B; maxdim=maxdim); cutoff=cutoff)
        if verbose
            idx = (s1 - 1) * r_n + s2
            (idx % 10 == 0 || idx == r_m * r_n) &&
                println("  ($s1,$s2)/($r_m,$r_n) done  [$idx/$(r_m*r_n)]")
        end
    end

    # ── Per-ω accumulation: scalar × MPO additions only ──────────────────────
    # Π(ω) = Σ_{s₁,s₂} G[s₁,s₂,ω] · D[s₁,s₂]
    Π = Vector{Union{Nothing, MPO}}(nothing, nω)
    for iω in 1:nω
        for s1 in 1:r_m, s2 in 1:r_n
            g = A_core[s1, s2, iω]
            (abs(g) < coeff_tol || isnothing(D_tuck[s1, s2])) && continue
            if Π[iω] === nothing
                Π[iω] = g * D_tuck[s1, s2]
            else
                Π[iω] = +(Π[iω], g * D_tuck[s1, s2]; maxdim=maxdim)
                ITensorMPS.truncate!(Π[iω]; cutoff=cutoff)
            end
        end
    end

    verbose && println("cheb2d_mpo_tucker: done — r_m=$r_m, r_n=$r_n, $(count(!isnothing, Π))/$nω non-zero")
    return [replace_sites(Π[iω]::MPO, H1.sites) for iω in 1:nω]
end


"""
    get_bubble_diag_cheb2d(H1, H2, ωlist; Ncheb, maxdim, cutoff,
                            ϵF, P_method, purify_*, η, coeff_tol,
                            qft_tol, qft_maxdim, verbose) -> Vector{MPS}

Diagonal-only variant of `get_bubble_mpo_cheb2d`.

Returns the k-space diagonal of the non-interacting polarization bubble,
    diag_Π₀(k, ω) = ⟨k| Π₀(ω) |k⟩,
as a `Vector{MPS}` (one MPS per ω in `ωlist`) ready for direct plotting.

Compared to `get_bubble_mpo_cheb2d`, this function:

  - QFTs each `D_mn` once (inside the (m,n) loop) and extracts its k-space
    diagonal as an MPS.
  - Accumulates `c_mn(ω) · diag(D_mn)` as **MPS** sums instead of MPO sums.
  - Skips the per-ω `conjugate_by_qft + extract_diagonal_to_mps` steps.

This follows the same paradigm as `get_bands`: the expensive MPO-level work
(Hadamard products, QFT conjugation) is done once per (m,n) pair and shared
across all frequencies; per-ω cost is a cheap scalar-weighted MPS addition.

Use `get_bubble_mpo_cheb2d` when you need the full off-diagonal MPO (e.g. for
RPA resummation).  Use this function when only χ₀(k,ω) is needed.

**Keyword arguments** — identical to `get_bubble_mpo_cheb2d`, plus:
- `qft_tol`     : truncation tolerance inside `conjugate_by_qft`. Default `1e-9`.
- `qft_maxdim`  : max bond dimension inside `conjugate_by_qft`. Default `100`.
"""
function get_bubble_diag_cheb2d(H1::TBHamiltonian, H2::TBHamiltonian,
                                 ωlist::AbstractVector{<:Real};
                                 Ncheb::Int            = 50,
                                 maxdim::Int           = 200,
                                 cutoff::Real          = 1e-8,
                                 ϵF::Real              = 0.0,
                                 P_method::Symbol      = :purification,
                                 purify_method::Symbol = :mcweeny,
                                 purify_maxdim::Int    = 40,
                                 purify_maxiters::Int  = 30,
                                 purify_tol::Float64   = 1e-5,
                                 η::Real               = 1e-3,
                                 coeff_tol::Real       = 1e-12,
                                 qft_tol::Real         = 1e-9,
                                 qft_maxdim::Int       = 100,
                                 verbose::Bool         = false)
    L1 = H1.L; L2 = H2.L
    @assert L1 == L2 "get_bubble_diag_cheb2d: H1 and H2 must have the same number of sites (got $L1 vs $L2)"
    L = L1
    nω = length(ωlist)

    _ensure_scale!(H1)
    _ensure_scale!(H2)
    scale1  = H1.scale;  center1 = H1.center
    scale2  = H2.scale;  center2 = H2.center

    verbose && println("cheb2d_diag: building T_n(H1) moments (Ncheb=$Ncheb)...")
    Tn1, _, _ = KPM_Tn(H1.mpo, Ncheb, H1.sites;
                         scale=scale1, center=center1,
                         maxdim=maxdim, cutoff=cutoff, verbose=false)
    if H1 === H2
        Tn2 = Tn1
    else
        verbose && println("cheb2d_diag: building T_n(H2) moments...")
        Tn2, _, _ = KPM_Tn(H2.mpo, Ncheb, H2.sites;
                             scale=scale2, center=center2,
                             maxdim=maxdim, cutoff=cutoff, verbose=false)
    end
    N = length(Tn1)

    verbose && println("cheb2d_diag: computing P1...")
    P1 = _get_density_matrix(H1, ϵF, P_method, Ncheb, maxdim, cutoff,
                              purify_method, purify_maxdim, purify_maxiters,
                              purify_tol, verbose)
    if H1 === H2
        P2 = P1
    else
        verbose && println("cheb2d_diag: computing P2...")
        P2 = _get_density_matrix(H2, ϵF, P_method, Ncheb, maxdim, cutoff,
                                  purify_method, purify_maxdim, purify_maxiters,
                                  purify_tol, verbose)
    end

    verbose && println("cheb2d_diag: precomputing T_m(H1)·P1 and T_n(H2)·P2...")
    TP1 = [ITensorMPS.truncate!(
               apply(Tn1[m], P1; maxdim=maxdim, cutoff=cutoff); cutoff=cutoff)
           for m in 1:N]
    TP2 = [ITensorMPS.truncate!(
               apply(Tn2[n], P2; maxdim=maxdim, cutoff=cutoff); cutoff=cutoff)
           for n in 1:N]

    out_sites = siteinds("Qubit", L)

    verbose && println("cheb2d_diag: precomputing C[m,n](ω) for all $nω frequencies...")
    C_all = [chebyshev2d_gf_coeffs(ω, scale1, center1, scale2, center2, η, N)
             for ω in ωlist]

    # Accumulate diagonal MPS (not full MPO) for each ω
    diag_Π = Vector{Union{Nothing, MPS}}(nothing, nω)
    n_computed = 0
    n_skipped  = 0

    for m in 1:N
        max_row = maximum(maximum(abs, @view C[m, :]) for C in C_all)
        if max_row < coeff_tol
            n_skipped += N
            continue
        end

        for n in 1:N
            max_c = maximum(abs(C[m, n]) for C in C_all)
            if max_c < coeff_tol
                n_skipped += 1
                continue
            end
            n_computed += 1

            # D_mn = TP1[m] ⊙ Tn2[n] − Tn1[m] ⊙ TP2[n]  (ω-independent)
            had_A = hadamard_mpo(TP1[m], Tn2[n], out_sites; maxdim=maxdim, cutoff=cutoff)
            had_B = hadamard_mpo(Tn1[m], TP2[n], out_sites; maxdim=maxdim, cutoff=cutoff)
            D_mn  = ITensorMPS.truncate!(+(had_A, -1 * had_B; maxdim=maxdim); cutoff=cutoff)

            # QFT + diagonal extraction — done ONCE per (m,n), shared across all ω.
            # replace_sites maps out_sites → H1.sites so conjugate_by_qft can find
            # the correct Qubit site structure.
            D_mn_phys = replace_sites(D_mn, H1.sites)
            D_k       = conjugate_by_qft(D_mn_phys; tol=qft_tol, maxdim=qft_maxdim)
            diag_D    = ITensorMPS.truncate!(extract_diagonal_to_mps(D_k); cutoff=cutoff)

            # Accumulate c_mn(ω) · diag_D into each diag_Π[iω] as MPS sums.
            # MPS additions are much cheaper than MPO additions (bond dim ∝ D vs D²).
            for (iω, C) in enumerate(C_all)
                c = C[m, n]
                abs(c) < coeff_tol && continue
                if diag_Π[iω] === nothing
                    diag_Π[iω] = c * diag_D
                else
                    diag_Π[iω] = +(diag_Π[iω], c * diag_D; maxdim=maxdim)
                    ITensorMPS.truncate!(diag_Π[iω]; cutoff=cutoff)
                end
            end
        end

        verbose && println("  m=$m/$N  (computed $n_computed, skipped $n_skipped so far)")
    end

    verbose && println("cheb2d_diag: done — $(n_computed)/$(N*N) pairs computed, $n_skipped skipped")

    return [diag_Π[iω] for iω in 1:nω]
end



# ── Jackson kernel weights for Chebyshev order N ──────────────────────────────
# g[m+1] = ((N-m)cos(πm/(N+1)) + sin(πm/(N+1))/tan(π/(N+1))) / (N+1)
# Suppresses Gibbs oscillations from truncation; broadening ≈ π·scale/N.
function _jackson_kernel(N::Int)
    m = 0:N-1
    return @. ((N - m) * cos(π * m / (N+1)) +
               sin(π * m / (N+1)) / tan(π / (N+1))) / (N+1)
end

# ── Helper: weighted MPO sum  Σ_i w_i · mpos[i]  with online truncation ──────
# Accepts real or complex weights; complex weights produce complex-tensor MPOs.
function _weighted_mpo_sum(weights::AbstractVector{<:Number}, mpos::Vector{MPO};
                           maxdim::Int, cutoff::Real, weight_tol::Real = 1e-14)
    result = nothing
    for (w, mpo) in zip(weights, mpos)
        abs(w) < weight_tol && continue
        if result === nothing
            result = w * mpo
        else
            result = ITensorMPS.truncate!(+(result, w * mpo; maxdim=maxdim); cutoff=cutoff)
        end
    end
    return result
end


"""
    get_bubble_diag_cheb2d_svd(H1, H2, ωlist; ..., svd_tol, svd_maxrank) -> Vector{MPS}

Per-ω SVD-accelerated variant of `get_bubble_diag_cheb2d`.

For each frequency ω the coefficient matrix C[m,n](ω) is rank-truncated via its own SVD:

    C[m,n](ω) = Σ_s  S_s(ω) · U[m,s](ω) · conj(V[n,s](ω))   (exact up to truncation)

The per-ω rank r(ω) (typically 2–5 for smooth Lorentzian kernels) is usually much
smaller than the Tucker/joint-SVD rank, which must span all frequencies simultaneously.
For each (ω, s) one Hadamard product and one QFT are performed, giving

    diag_Π[ω] = Σ_s S_s(ω) · diag(QFT( A_s ⊙ B_s − C_s ⊙ E_s ))

where A_s = Σ_m U[m,s]·TP1[m], B_s = Σ_n conj(V[n,s])·Tn2[n], etc.

Total Hadamard+QFT operations: Σ_ω r(ω) — compared to N² for the plain variant or
r_m·r_n for Tucker. When r(ω) ≪ r_Tucker the per-ω SVD is both faster and more
accurate, because it uses the optimal basis for each individual C(ω).

**Additional keyword arguments** (beyond `get_bubble_diag_cheb2d`):
- `svd_tol`    : relative singular-value cutoff per ω (fraction of σ_max(ω)). Default `1e-6`.
- `svd_maxrank`: hard cap on the per-ω rank. Default `20`.
- `kernel`     : Chebyshev damping kernel applied before SVD. `:jackson` (default) suppresses
                 Gibbs oscillations and dramatically reduces per-ω rank (typically 2–5 instead
                 of ~26). Use `:none` for exact Chebyshev coefficients.
"""
function get_bubble_diag_cheb2d_svd(H1::TBHamiltonian, H2::TBHamiltonian,
                                     ωlist::AbstractVector{<:Real};
                                     Ncheb::Int            = 50,
                                     maxdim::Int           = 200,
                                     cutoff::Real          = 1e-8,
                                     ϵF::Real              = 0.0,
                                     P_method::Symbol      = :purification,
                                     purify_method::Symbol = :mcweeny,
                                     purify_maxdim::Int    = 40,
                                     purify_maxiters::Int  = 30,
                                     purify_tol::Float64   = 1e-5,
                                     η::Real               = 1e-3,
                                     qft_tol::Real         = 1e-9,
                                     qft_maxdim::Int       = 100,
                                     svd_tol::Real         = 1e-6,
                                     svd_maxrank::Int      = 20,
                                     kernel::Symbol        = :jackson,
                                     verbose::Bool         = false)
    L1 = H1.L; L2 = H2.L
    @assert L1 == L2 "get_bubble_diag_cheb2d_svd: H1 and H2 must have the same number of sites (got $L1 vs $L2)"
    L  = L1
    nω = length(ωlist)

    _ensure_scale!(H1); _ensure_scale!(H2)
    scale1 = H1.scale; center1 = H1.center
    scale2 = H2.scale; center2 = H2.center

    verbose && println("cheb2d_diag_svd: building Chebyshev moments (Ncheb=$Ncheb)...")
    Tn1, _, _ = KPM_Tn(H1.mpo, Ncheb, H1.sites;
                        scale=scale1, center=center1,
                        maxdim=maxdim, cutoff=cutoff, verbose=false)
    if H1 === H2
        Tn2 = Tn1
    else
        Tn2, _, _ = KPM_Tn(H2.mpo, Ncheb, H2.sites;
                            scale=scale2, center=center2,
                            maxdim=maxdim, cutoff=cutoff, verbose=false)
    end
    N = length(Tn1)

    verbose && println("cheb2d_diag_svd: computing density matrices...")
    P1 = _get_density_matrix(H1, ϵF, P_method, Ncheb, maxdim, cutoff,
                             purify_method, purify_maxdim, purify_maxiters,
                             purify_tol, verbose)
    P2 = H1 === H2 ? P1 : _get_density_matrix(H2, ϵF, P_method, Ncheb, maxdim, cutoff,
                                               purify_method, purify_maxdim, purify_maxiters,
                                               purify_tol, verbose)

    out_sites = siteinds("Qubit", L)

    verbose && println("cheb2d_diag_svd: computing coefficient matrices for $nω frequencies...")
    C_all = [chebyshev2d_gf_coeffs(ω, scale1, center1, scale2, center2, η, N)
             for ω in ωlist]

    if kernel == :jackson
        g_jk  = _jackson_kernel(N)
        G_jk  = g_jk * g_jk'         # N×N outer product, applied element-wise
        C_all = [G_jk .* C for C in C_all]
        verbose && println("cheb2d_diag_svd: Jackson kernel applied")
    elseif kernel != :none
        error("get_bubble_diag_cheb2d_svd: unknown kernel=$kernel (use :jackson or :none)")
    end

    # ── Per-ω SVD of the coefficient matrix C[m,n](ω) ───────────────────────
    # For each ω, the exact SVD gives the optimal low-rank factorisation:
    #   C(ω) = U(ω) · Diagonal(S(ω)) · V(ω)ᴴ
    # The per-ω rank r(ω) is typically much smaller than the Tucker/joint rank,
    # because each individual C(ω) is structured by a single Lorentzian kernel
    # and doesn't need to share a common basis with other frequencies.
    diag_Π = Vector{Union{Nothing, MPS}}(nothing, nω)
    ranks   = Int[]

    for (iω, C) in enumerate(C_all)
        F_ω   = svd(C)
        σ_cut = svd_tol * F_ω.S[1]
        r_ω   = min(svd_maxrank, sum(F_ω.S .> σ_cut))
        push!(ranks, r_ω)

        for s in 1:r_ω
            σ_s = F_ω.S[s]
            u_s = F_ω.U[:, s]           # complex left singular vector
            v_s = conj.(F_ω.V[:, s])    # SVD: C = U S Vᴴ, so right factor is conj(V[:,s])

            # Sum bare moments first, then apply P once — saves one MPO-MPO
            # multiplication per component vs pre-multiplying each Tn by P.
            C_s = _weighted_mpo_sum(u_s, Tn1; maxdim=maxdim, cutoff=cutoff)
            B_s = _weighted_mpo_sum(v_s, Tn2; maxdim=maxdim, cutoff=cutoff)
            (isnothing(C_s) || isnothing(B_s)) && continue
            A_s = ITensorMPS.truncate!(apply(C_s, P1; maxdim=maxdim, cutoff=cutoff); cutoff=cutoff)
            E_s = ITensorMPS.truncate!(apply(B_s, P2; maxdim=maxdim, cutoff=cutoff); cutoff=cutoff)

            (isnothing(A_s) || isnothing(E_s)) && continue

            had_A = hadamard_mpo(A_s, B_s, out_sites; maxdim=maxdim, cutoff=cutoff)
            had_B = hadamard_mpo(C_s, E_s, out_sites; maxdim=maxdim, cutoff=cutoff)
            D     = ITensorMPS.truncate!(+(had_A, -1 * had_B; maxdim=maxdim); cutoff=cutoff)

            D_phys = replace_sites(D, H1.sites)
            D_k    = conjugate_by_qft(D_phys; tol=qft_tol, maxdim=qft_maxdim)
            diag_s = ITensorMPS.truncate!(extract_diagonal_to_mps(D_k); cutoff=cutoff)

            if diag_Π[iω] === nothing
                diag_Π[iω] = σ_s * diag_s
            else
                diag_Π[iω] = +(diag_Π[iω], σ_s * diag_s; maxdim=maxdim)
                ITensorMPS.truncate!(diag_Π[iω]; cutoff=cutoff)
            end
        end

        verbose && println("  ω=$(round(ωlist[iω];digits=3))  rank=$r_ω")
    end

    r_min, r_max = extrema(ranks)
    r_mean = round(sum(ranks) / length(ranks); digits=1)
    verbose && println("cheb2d_diag_svd: done — per-ω ranks min=$r_min max=$r_max mean=$r_mean, $(count(!isnothing, diag_Π))/$nω non-zero")
    return [diag_Π[iω] for iω in 1:nω]
end


"""
    get_bubble_diag_cheb2d_tucker(H1, H2, ωlist; ..., tucker_tol, tucker_maxrank, kernel) -> Vector{MPS}

Tucker (HOSVD) variant of `get_bubble_diag_cheb2d`.

Finds a global low-rank basis in the (m, n) indices shared across all frequencies by
stacking the coefficient matrices and performing two mode-SVDs:

  - Mode-1 SVD of  [C(ω₁) | C(ω₂) | … | C(ωₙ)]  (shape N × N·nω) → basis U_m (N × r_m)
  - Mode-2 SVD of  [C(ω₁)ᵀ | … | C(ωₙ)ᵀ]        (shape N × N·nω) → basis V_n (N × r_n)

Then for each (s1, s2) ∈ {1…r_m} × {1…r_n}, one Hadamard product and one QFT are computed
(both ω-independent), giving `r_m × r_n` such operations in total.  Per-ω cost reduces to
cheap scalar-weighted MPS additions over the core tensor Ã[s1, s2, ω] = U_mᵀ C(ω) V_n.

**Scaling comparison** (assuming rank saturation with Jackson kernel):
- Plain diagonal: N² Hadamard+QFT
- Per-ω SVD:      nω × r_per_ω Hadamard+QFT
- Tucker:         r_m × r_n Hadamard+QFT  ← frequency-independent

Tucker wins when r_m × r_n < nω × r_per_ω, which holds for large nω or when the global
(m,n) structure is very low-dimensional (as it is after Jackson damping).

**Additional keyword arguments** (beyond `get_bubble_diag_cheb2d`):
- `tucker_tol`    : relative singular-value cutoff for both mode SVDs. Default `1e-3`.
- `tucker_maxrank`: hard cap on r_m and r_n. Default `20`.
- `kernel`        : `:jackson` (default) or `:none`. Jackson damping is essential here —
                    without it the Tucker rank is high and results are inaccurate.
- `coeff_tol`     : skip core-tensor entries |Ã[s1,s2,ω]| below this threshold. Default `1e-12`.
- `hooi_iters`    : number of Higher-Order Orthogonal Iteration refinement steps after the
                    initial HOSVD.  Each iteration re-optimises U_m and V_n jointly, giving the
                    globally optimal Tucker bases for the given rank (vs. the independent
                    mode-SVDs of plain HOSVD).  Default `3`; set to `0` for HOSVD only.
"""
function get_bubble_diag_cheb2d_tucker(H1::TBHamiltonian, H2::TBHamiltonian,
                                        ωlist::AbstractVector{<:Real};
                                        Ncheb::Int            = 50,
                                        maxdim::Int           = 200,
                                        cutoff::Real          = 1e-8,
                                        ϵF::Real              = 0.0,
                                        P_method::Symbol      = :purification,
                                        purify_method::Symbol = :mcweeny,
                                        purify_maxdim::Int    = 40,
                                        purify_maxiters::Int  = 30,
                                        purify_tol::Float64   = 1e-5,
                                        η::Real               = 1e-3,
                                        coeff_tol::Real       = 1e-12,
                                        qft_tol::Real         = 1e-9,
                                        qft_maxdim::Int       = 100,
                                        tucker_tol::Real      = 1e-3,
                                        tucker_maxrank::Int   = 20,
                                        kernel::Symbol        = :jackson,
                                        hooi_iters::Int       = 3,
                                        verbose::Bool         = false)
    L1 = H1.L; L2 = H2.L
    @assert L1 == L2 "get_bubble_diag_cheb2d_tucker: H1 and H2 must have the same number of sites (got $L1 vs $L2)"
    L  = L1
    nω = length(ωlist)

    _ensure_scale!(H1); _ensure_scale!(H2)
    scale1 = H1.scale; center1 = H1.center
    scale2 = H2.scale; center2 = H2.center

    verbose && println("cheb2d_tucker: building Chebyshev moments (Ncheb=$Ncheb)...")
    Tn1, _, _ = KPM_Tn(H1.mpo, Ncheb, H1.sites;
                        scale=scale1, center=center1,
                        maxdim=maxdim, cutoff=cutoff, verbose=false)
    if H1 === H2
        Tn2 = Tn1
    else
        Tn2, _, _ = KPM_Tn(H2.mpo, Ncheb, H2.sites;
                            scale=scale2, center=center2,
                            maxdim=maxdim, cutoff=cutoff, verbose=false)
    end
    N = length(Tn1)

    verbose && println("cheb2d_tucker: computing density matrices...")
    P1 = _get_density_matrix(H1, ϵF, P_method, Ncheb, maxdim, cutoff,
                             purify_method, purify_maxdim, purify_maxiters,
                             purify_tol, verbose)
    P2 = H1 === H2 ? P1 : _get_density_matrix(H2, ϵF, P_method, Ncheb, maxdim, cutoff,
                                               purify_method, purify_maxdim, purify_maxiters,
                                               purify_tol, verbose)

    out_sites = siteinds("Qubit", L)

    verbose && println("cheb2d_tucker: computing coefficient matrices for $nω frequencies...")
    C_all = [chebyshev2d_gf_coeffs(ω, scale1, center1, scale2, center2, η, N)
             for ω in ωlist]

    if kernel == :jackson
        g_jk  = _jackson_kernel(N)
        G_jk  = g_jk * g_jk'
        C_all = [G_jk .* C for C in C_all]
        verbose && println("cheb2d_tucker: Jackson kernel applied")
    elseif kernel != :none
        error("get_bubble_diag_cheb2d_tucker: unknown kernel=$kernel (use :jackson or :none)")
    end

    # ── Tucker bases: HOSVD initialisation + HOOI refinement ────────────────
    # HOSVD: independent mode SVDs give a fast but sub-optimal starting point.
    T1 = hcat(C_all...)                            # N × (N·nω) — mode-1 unfolding
    T2 = hcat([transpose(C) for C in C_all]...)   # N × (N·nω) — mode-2 unfolding
    F1 = svd(T1); F2 = svd(T2)
    r_m = min(tucker_maxrank, sum(F1.S .> tucker_tol * F1.S[1]))
    r_n = min(tucker_maxrank, sum(F2.S .> tucker_tol * F2.S[1]))
    U_m = F1.U[:, 1:r_m]
    V_n = F2.U[:, 1:r_n]

    # HOOI: alternating projection onto the optimal subspaces for the given rank.
    # Each step re-contracts the full tensor against the current other-mode basis
    # and extracts the leading singular vectors — converges in a few iterations.
    for _ in 1:hooi_iters
        Y   = hcat([C * V_n  for C in C_all]...)   # N × (r_n·nω): contract n with V_n
        U_m = svd(Y).U[:, 1:r_m]
        Z   = hcat([C' * U_m for C in C_all]...)   # N × (r_m·nω): contract m with U_m
        V_n = svd(Z).U[:, 1:r_n]
    end
    verbose && println("cheb2d_tucker: Tucker ranks r_m=$r_m, r_n=$r_n (HOSVD + $hooi_iters HOOI iters) → $(r_m*r_n) Hadamard+QFT operations")

    # ── Core tensor: project each C(ω) onto Tucker bases ─────────────────────
    A_core = zeros(ComplexF64, r_m, r_n, nω)
    for iω in 1:nω
        A_core[:, :, iω] = U_m' * C_all[iω] * V_n
    end

    # ── ω-independent weighted MPO sums + deferred P application ────────────
    # Sum bare moments first (r_m + r_n weighted sums of N terms each), then
    # apply P1/P2 once per component — saves 2N MPO-MPO multiplications vs
    # precomputing TP1[m]/TP2[n] globally and reduces to r_m + r_n applies.
    verbose && println("cheb2d_tucker: computing Tucker MPO components...")
    C_tuck = [_weighted_mpo_sum(U_m[:, s1],        Tn1; maxdim=maxdim, cutoff=cutoff) for s1 in 1:r_m]
    B_tuck = [_weighted_mpo_sum(conj.(V_n[:, s2]), Tn2; maxdim=maxdim, cutoff=cutoff) for s2 in 1:r_n]
    A_tuck = [isnothing(C_tuck[s1]) ? nothing :
              ITensorMPS.truncate!(apply(C_tuck[s1], P1; maxdim=maxdim, cutoff=cutoff); cutoff=cutoff)
              for s1 in 1:r_m]
    E_tuck = [isnothing(B_tuck[s2]) ? nothing :
              ITensorMPS.truncate!(apply(B_tuck[s2], P2; maxdim=maxdim, cutoff=cutoff); cutoff=cutoff)
              for s2 in 1:r_n]

    # ── ω-independent Hadamard + QFT  (r_m × r_n total) ─────────────────────
    verbose && println("cheb2d_tucker: computing $(r_m*r_n) Hadamard+QFT components...")
    diag_D = Matrix{Union{Nothing, MPS}}(nothing, r_m, r_n)
    for s1 in 1:r_m, s2 in 1:r_n
        (isnothing(A_tuck[s1]) || isnothing(B_tuck[s2]) ||
         isnothing(C_tuck[s1]) || isnothing(E_tuck[s2])) && continue

        had_A = hadamard_mpo(A_tuck[s1], B_tuck[s2], out_sites; maxdim=maxdim, cutoff=cutoff)
        had_B = hadamard_mpo(C_tuck[s1], E_tuck[s2], out_sites; maxdim=maxdim, cutoff=cutoff)
        D     = ITensorMPS.truncate!(+(had_A, -1 * had_B; maxdim=maxdim); cutoff=cutoff)

        D_phys          = replace_sites(D, H1.sites)
        D_k             = conjugate_by_qft(D_phys; tol=qft_tol, maxdim=qft_maxdim)
        diag_D[s1, s2]  = ITensorMPS.truncate!(extract_diagonal_to_mps(D_k); cutoff=cutoff)
        if verbose
            idx = (s1 - 1) * r_n + s2
            (idx % 10 == 0 || idx == r_m * r_n) &&
                println("  ($s1,$s2)/($r_m,$r_n) done  [$idx/$(r_m*r_n)]")
        end
    end

    # ── Accumulate per ω: scalar × MPS additions only ────────────────────────
    diag_Π = Vector{Union{Nothing, MPS}}(nothing, nω)
    for iω in 1:nω
        for s1 in 1:r_m, s2 in 1:r_n
            a = A_core[s1, s2, iω]
            (abs(a) < coeff_tol || isnothing(diag_D[s1, s2])) && continue
            if diag_Π[iω] === nothing
                diag_Π[iω] = a * diag_D[s1, s2]
            else
                diag_Π[iω] = +(diag_Π[iω], a * diag_D[s1, s2]; maxdim=maxdim)
                ITensorMPS.truncate!(diag_Π[iω]; cutoff=cutoff)
            end
        end
    end

    verbose && println("cheb2d_tucker: done — r_m=$r_m, r_n=$r_n, $(count(!isnothing, diag_Π))/$nω non-zero")
    return [diag_Π[iω] for iω in 1:nω]
end
