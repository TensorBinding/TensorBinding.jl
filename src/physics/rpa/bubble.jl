# physics/rpa/bubble.jl — non-interacting polarization bubble Π₀(ω) from the Green's
# function of H_eff = I⊗H₂ − H₁⊗I on the interleaved 2L-site space: the density-matrix
# and H_eff helpers (_get_density_matrix, _build_heff), get_bubble_mpo (KPM or Krylov
# Green's function) and the Haydock-recursion variant get_bubble_mpo_haydock (the
# recursion haydock_cf, eval_haydock_cf, haydock_resolve_mpo is in solvers/Krylov.jl).
# Split verbatim from the former physics/RPA_tk.jl.
#
# Entry points: get_bubble_mpo, get_bubble_mpo_haydock.
# Depends on: core/Utils.jl, core/MPOTools.jl, core/TBSystem.jl, solvers/DMRG.jl,
#   solvers/kpm/recursion.jl, solvers/kpm/cached.jl, solvers/Krylov.jl,
#   physics/Purification.jl* (* = included later; see the source map in
#   src/TensorBinding.jl).

# ============================================================
# 1. Internal helpers for the TBHamiltonian API
# ============================================================

# The density matrix of a bubble: get_density's dispatcher (_density_matrix,
# physics/Purification.jl) with the bubbles' own method symbols and rules.
#
#   P_method=:purification  purify_method (:mcweeny or :sp2), reusing the density
#                           cache whatever ϵF (as get_density does). McWeeny is
#                           purified at ϵF, with mcweeny_purify's convention: the
#                           level of its initial guess is H.center + ϵF. SP2 fixes
#                           the filling through Nel = H.N ÷ 2 instead, so it refuses
#                           ϵF ≠ 0. (Until the fix ϵF was not passed and every
#                           purification ran at ϵF = 0.)
#   P_method=:kpm           a fresh Chebyshev list of order Ncheb from the raw
#                           KPM_Tn (not cached on H; its progress lines follow
#                           `verbose`), no density cache read or written; the
#                           level is ϵF.
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
        purify_method in (:mcweeny, :sp2) ||
            error("Unknown purify_method: $purify_method. Choose :mcweeny or :sp2")
        purify_method == :sp2 && !iszero(ϵF) &&
            throw(ArgumentError("purify_method=:sp2 fixes the filling at Nel = H.N ÷ 2 " *
                                "and cannot take a Fermi level (got ϵF = $ϵF); use " *
                                "purify_method=:mcweeny or P_method=:kpm"))
        Nel = H.N ÷ 2
        verbose && println(purify_method == :mcweeny ? "  Running McWeeny purification" :
                                                       "  Running SP2 purification (Nel=$Nel)")
        return _density_matrix(H, purify_method; ϵF=ϵF, Nel=Nel, maxiters=purify_maxiters,
                               maxdim=purify_maxdim, cutoff=cutoff, tol=purify_tol,
                               verbose=verbose)
    elseif P_method == :kpm
        _ensure_scale!(H)
        Tn_list, _, _ = KPM_Tn(H.mpo, Ncheb, H.sites;
                                 scale=H.scale, center=H.center,
                                 identity_mpo=physical_projector(H),
                                 maxdim=maxdim, cutoff=cutoff, verbose=verbose)
        return _density_matrix(H, :kpm; ϵF=ϵF, maxdim=maxdim, cutoff=cutoff,
                               Tn=(Tn_list, Ncheb), store=false)
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
# 2. High-level TBHamiltonian API
# ============================================================

"""
    get_bubble_mpo(H1::TBHamiltonian, H2::TBHamiltonian, ω; ...) -> MPO

Compute the non-interacting polarization bubble Π₀(ω) on `H1.sites`.

**Keyword arguments**
- `ϵF`             : Fermi energy (physical units). Default `0.0`. The `:kpm` density is
  θ(ϵF − H); McWeeny purification starts from the level `H.center + ϵF` (the
  convention of `mcweeny_purify`, the same level when `H.center = 0`);
  `purify_method=:sp2` fixes the filling instead and requires `ϵF = 0`.
- `P_method`       : `:purification` (default) or `:kpm` — how to compute density matrices.
  With `:purification`, `H._density_cache` is reused if present, whatever `ϵF`
  (set `H._density_cache = nothing` after changing it).
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
    # H1 and H2 usually share site indices (H1 === H2 for the charge bubble, the two
    # spin sectors of one H for the magnon bubble). Interleaved as they are, the same
    # Index would sit on two neighbouring tensors, so H2's operators are moved onto
    # fresh copies of its sites.
    sites2 = sim.(H2.sites)

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
    P2 = replace_sites(P2, sites2)

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
    Heff = _build_heff(H1.mpo, replace_sites(H2.mpo, sites2), sites1, sites2)
    verbose && println("Polarization bubble: Heff maxlinkdim = ", maxlinkdim(Heff))
    if GF_method == :kpm
        # Auto-estimate Heff spectral bounds via DMRG (scale=0 triggers estimator)
        Tn_listeff, scaleeff, centereff = KPM_Tn(Heff, Ncheb, sites_combined;
                                                   maxdim=maxdim, cutoff=cutoff,
                                                   verbose=verbose)
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

# ============================================================
# 3. Haydock-recursion bubble (haydock_cf & co.: solvers/Krylov.jl)
# ============================================================

"""
    get_bubble_mpo_haydock(H1, H2, ωlist; N_steps=30, η=1e-2, maxdim=200, cutoff=1e-8,
                            ϵF=0.0, P_method=:purification, purify_method=:mcweeny,
                            purify_maxdim=40, purify_maxiters=30, purify_tol=1e-5,
                            Ncheb=150, verbose=false) -> Vector{MPO}

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
- `ϵF`             : Fermi energy. Default `0.0`. It reaches the density matrices
  as in `get_bubble_mpo`.
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
    # Same 2L-site layout as get_bubble_mpo (and _build_heff): interleaved
    # [s1[1], s2[1], …] with a fresh copy of H2's sites as the second register.
    sites1         = H1.sites
    sites2         = sim.(H2.sites)
    sites_combined = reduce(vcat, [[s1, s2] for (s1, s2) in zip(sites1, sites2)])

    # ---- Density matrices ----
    verbose && println("Haydock bubble: computing P1 (P_method=$P_method)...")
    P1 = _get_density_matrix(H1, ϵF, P_method, Ncheb, maxdim, cutoff,
                              purify_method, purify_maxdim, purify_maxiters, purify_tol, verbose)
    verbose && println("Haydock bubble: computing P2...")
    P2 = _get_density_matrix(H2, ϵF, P_method, Ncheb, maxdim, cutoff,
                              purify_method, purify_maxdim, purify_maxiters, purify_tol, verbose)
    P2 = replace_sites(P2, sites2)

    # ---- Seed: I⊗P₂ − P₁⊗I on 2L-site combined space ----
    id1  = MPO(sites1, "Id"); id2 = MPO(sites2, "Id")
    P1op = interleave_mpo_tb(P1,  sites1, sites2, :A)
    Iop2 = interleave_mpo_tb(id2, sites1, sites2, :B)
    Iop1 = interleave_mpo_tb(id1, sites1, sites2, :A)
    P2op = interleave_mpo_tb(P2,  sites1, sites2, :B)
    seed = ITensorMPS.truncate!(
        apply(Iop1, P2op; maxdim=maxdim, cutoff=cutoff) -
        apply(P1op, Iop2; maxdim=maxdim, cutoff=cutoff); cutoff=cutoff)
    verbose && println("Haydock bubble: seed built, chi=$(maxlinkdim(seed))")

    # ---- H_eff = I⊗H₂ − H₁⊗I ----
    Heff = _build_heff(H1.mpo, replace_sites(H2.mpo, sites2), sites1, sites2)
    verbose && println("Haydock bubble: H_eff built, chi=$(maxlinkdim(Heff))")

    # ---- Haydock recursion (once, independent of ω) ----
    verbose && println("Haydock bubble: running $N_steps steps...")
    a, b, basis, norm0 = haydock_cf(Heff, seed, N_steps;
                                     maxdim=maxdim, cutoff=cutoff, verbose=verbose)
    verbose && println("Haydock bubble: $(length(a)) steps completed, norm0=$(round(norm0;digits=4))")

    # ---- Assemble Π₀(ω) for each frequency ----
    finalsites = [Index(dim(s), "Bubble,n=$i") for (i, s) in enumerate(sites_combined)]
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
