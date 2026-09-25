# physics/rpa/cheb2d.jl — double Chebyshev decomposition of the polarization bubble:
# chebyshev2d_gf_coeffs, the full-MPO bubbles get_bubble_mpo_cheb2d(_tucker), the
# k-space diagonal bubbles get_bubble_diag_cheb2d(_svd, _tucker), and their helpers
# (_cheb2d_out_sites, _cheb2d_require_position_sites, _jackson_kernel, _weighted_mpo_sum).
# Split verbatim from physics/RPA_tk.jl.

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


# Output indices for the Hadamard products of the cheb2d MPO bubbles: one fresh index per
# site of H1.sites, of the same dimension, so Π₀ lives on all of H1.sites (spin, Nambu,
# layer and sublattice indices included), like the result of get_bubble_mpo.
function _cheb2d_out_sites(H1::TBHamiltonian, H2::TBHamiltonian, fname::AbstractString)
    dim.(H1.sites) == dim.(H2.sites) ||
        throw(ArgumentError("$fname: H1 and H2 must have the same site structure " *
                            "(site dimensions $(dim.(H1.sites)) vs $(dim.(H2.sites)))"))
    return [sim(s) for s in H1.sites]
end


# The cheb2d diagonal bubbles Fourier-transform every site of D_mn as a position qubit
# (conjugate_by_qft(W)), so H.sites must be exactly the H.L position qubits.
function _cheb2d_require_position_sites(H1::TBHamiltonian, H2::TBHamiltonian,
                                        fname::AbstractString)
    for (name, H) in (("H1", H1), ("H2", H2))
        length(H.sites) == H.L && continue
        throw(ArgumentError(
            "$fname: $name has $(length(H.sites)) site indices for $name.L = $(H.L) " *
            "position qubits. This k-space diagonal Fourier-transforms every site, so " *
            "spin, Nambu, layer and sublattice indices are not supported. For a " *
            "spin-conserving spinful H, pass each spin sector " *
            "TensorBinding._project_spin_sector(H, σ), σ = 1, 2, and add the two " *
            "results to get the charge bubble. Otherwise use get_bubble_mpo_cheb2d, " *
            "which keeps all of H.sites, with conjugate_by_qft(H, Π)."))
    end
    return nothing
end




"""
    get_bubble_mpo_cheb2d(H1, H2, ωlist; Ncheb, maxdim, cutoff,
                           ϵF, P_method, purify_*, η, verbose) -> Vector{MPO}

Compute the non-interacting polarization bubble Π₀(ω) for each ω in `ωlist`
using the **double Chebyshev decomposition**.

Π₀ lives on `H1.sites`, including any spin, Nambu, layer or sublattice index, and
is resolved in those indices, as the result of `get_bubble_mpo` is. `H1` and `H2`
must have the same site structure.

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

**Online multi-ω sweep**: All coefficient matrices `C[m,n](ω)` are precomputed
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
- `coeff_tol`     : Skip (m,n) pairs where `|C[m,n]| < coeff_tol`, and entire m rows
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
    # Fresh physical indices shared by all Hadamard product calls
    out_sites = _cheb2d_out_sites(H1, H2, "get_bubble_mpo_cheb2d")

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

    `C[m,n](ω) ≈ Σ_{s₁,s₂} G[s₁,s₂,ω] · (U_m)_{ms₁} · conj((V_n)_{ns₂})`

with C ≈ U_m G(ω) V_n†.  The r_m + r_n frequency-independent weighted MPO sums
and the r_m × r_n Hadamard products are computed once; per-ω cost is only
r_m × r_n cheap scalar-weighted MPO additions.

Speedup over `get_bubble_mpo_cheb2d`: N²→r_m·r_n Hadamard products.

As in `get_bubble_mpo_cheb2d`, Π₀ lives on `H1.sites`, including any spin, Nambu,
layer or sublattice index.

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
    out_sites = _cheb2d_out_sites(H1, H2, "get_bubble_mpo_cheb2d_tucker")

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

`H1.sites` and `H2.sites` must be exactly the `H.L` position qubits: the QFT
here treats every site as one, so a spin, Nambu, layer or sublattice index
raises an `ArgumentError`. For a spin-conserving spinful `H`, the charge bubble
is the sum of the results for the two spin sectors
`TensorBinding._project_spin_sector(H, σ)`, `σ = 1, 2`.

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
    _cheb2d_require_position_sites(H1, H2, "get_bubble_diag_cheb2d")
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

Per-ω SVD-accelerated variant of `get_bubble_diag_cheb2d`, with the same
requirement that `H.sites` be the `H.L` position qubits.

For each frequency ω the coefficient matrix `C[m,n](ω)` is rank-truncated via its own SVD:

    `C[m,n](ω) = Σ_s  S_s(ω) · U[m,s](ω) · conj(V[n,s](ω))`   (exact up to truncation)

The per-ω rank r(ω) (typically 2–5 for smooth Lorentzian kernels) is usually much
smaller than the Tucker/joint-SVD rank, which must span all frequencies simultaneously.
For each (ω, s) one Hadamard product and one QFT are performed, giving

    `diag_Π[ω] = Σ_s S_s(ω) · diag(QFT( A_s ⊙ B_s − C_s ⊙ E_s ))`

where `A_s = Σ_m U[m,s]·TP1[m]`, `B_s = Σ_n conj(V[n,s])·Tn2[n]`, etc.

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
    _cheb2d_require_position_sites(H1, H2, "get_bubble_diag_cheb2d_svd")
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

Tucker (HOSVD) variant of `get_bubble_diag_cheb2d`, with the same requirement
that `H.sites` be the `H.L` position qubits.

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
    _cheb2d_require_position_sites(H1, H2, "get_bubble_diag_cheb2d_tucker")
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
