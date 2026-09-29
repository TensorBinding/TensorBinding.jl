# physics/rpa/cheb2d.jl — double Chebyshev decomposition of the polarization bubble:
# chebyshev2d_gf_coeffs, the full-MPO bubbles get_bubble_mpo_cheb2d(_tucker), the
# k-space diagonal bubbles get_bubble_diag_cheb2d(_svd, _tucker), and their helpers
# (_cheb2d_out_sites, _cheb2d_require_position_sites, _weighted_mpo_sum).
# Split from the former physics/RPA_tk.jl. The five bubbles share the kernels of
# section 2: the prologue _cheb2d_setup, the plain (m,n) sweep _cheb2d_pair_sweep!, the
# Tucker steps _tucker_bases, _tucker_components, _tucker_hadamard and _tucker_accumulate,
# and the per-term _hadamard_difference (with the H₁-side transposes _transpose_mpo,
# _real_mpo), _cheb2d_kdiag and _accumulate_scaled!.
#
# Entry points: get_bubble_mpo_cheb2d, get_bubble_mpo_cheb2d_tucker,
#   get_bubble_diag_cheb2d, get_bubble_diag_cheb2d_svd, get_bubble_diag_cheb2d_tucker,
#   chebyshev2d_gf_coeffs.
# Depends on: core/Utils.jl, core/TBSystem.jl, solvers/DMRG.jl, solvers/kpm/recursion.jl,
#   solvers/kpm/kernels.jl (_kpm_kernel), solvers/kpm/cached.jl (_chebyshev_sum),
#   physics/rpa/bubble.jl,
#   physics/qft/conjugation.jl (see the source map in src/TensorBinding.jl).

# ============================================================
# 1. Chebyshev coefficients and site helpers
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
    # REDFT10 along each dimension is Y_k = 2 Σ_j F_j cos(πk(j + ½)/N), and the
    # Chebyshev interpolant has c_k = (2/N) Σ_j F_j cos(πk(j + ½)/N) with c_0 halved:
    # c = Y/N per dimension, Y/N² in 2D. (It was Y/(2N)², a quarter of f.)
    Cr = FFTW.r2r(real.(F), FFTW.REDFT10, [1, 2])
    Ci = FFTW.r2r(imag.(F), FFTW.REDFT10, [1, 2])
    C  = (Cr .+ im .* Ci) ./ N^2
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


# ============================================================
# 2. Shared kernels: prologue, (m,n) sweep, Tucker steps
# ============================================================

# The prologue of every cheb2d bubble, in the order each of them ran it: check the
# site structure (_cheb2d_out_sites for the MPO bubbles, which also makes their output
# indices; _cheb2d_require_position_sites for the k-space diagonals), fix the spectral
# bounds, build T_n(H̃₁) and T_n(H̃₂) (shared when H1 === H2), the density matrices P₁
# and P₂, and the coefficient matrices C[m,n](ω) of every ω in `ωlist`.
#
#   lowrank = false  the plain (m,n) sweep: also precomputes TP1[m] = T_m(H̃₁)·P₁ and
#                    TP2[n] = T_n(H̃₂)·P₂, and reports each Hamiltonian's steps.
#   lowrank = true   the SVD and Tucker variants: damps every C(ω) with `kernel`
#                    (:jackson or :none) and reports the steps jointly.
#   diagonal = true  the output indices are siteinds("Qubit", H.L), made after P₂.
#
# `fname` names the caller in errors, `tag` starts its progress lines.
# Returns (; N, Tn1, Tn2, P1, P2, TP1, TP2, C_all, out_sites, transpose1) with N = Ncheb + 1,
# TP1 = TP2 = nothing when lowrank, and transpose1 the flag of _hadamard_difference.
function _cheb2d_setup(H1::TBHamiltonian, H2::TBHamiltonian, ωlist::AbstractVector{<:Real},
                       fname::AbstractString, tag::AbstractString;
                       diagonal::Bool, lowrank::Bool, kernel::Symbol = :none,
                       Ncheb::Int, maxdim::Int, cutoff::Real, ϵF::Real, P_method::Symbol,
                       purify_method::Symbol, purify_maxdim::Int, purify_maxiters::Int,
                       purify_tol::Float64, η::Real, verbose::Bool)
    L1 = H1.L; L2 = H2.L
    @assert L1 == L2 "$fname: H1 and H2 must have the same number of sites (got $L1 vs $L2)"
    L  = L1
    nω = length(ωlist)
    if diagonal
        _cheb2d_require_position_sites(H1, H2, fname)
        out_sites = nothing
    else
        # Fresh physical indices shared by all Hadamard product calls
        out_sites = _cheb2d_out_sites(H1, H2, fname)
    end

    _ensure_scale!(H1)
    _ensure_scale!(H2)
    scale1 = H1.scale; center1 = H1.center
    scale2 = H2.scale; center2 = H2.center

    verbose && println(lowrank ? "$tag: building Chebyshev moments (Ncheb=$Ncheb)..." :
                                 "$tag: building T_n(H1) moments (Ncheb=$Ncheb)...")
    Tn1, _, _ = KPM_Tn(H1.mpo, Ncheb, H1.sites;
                       scale=scale1, center=center1,
                       identity_mpo=physical_projector(H1),
                       maxdim=maxdim, cutoff=cutoff, verbose=false)
    if H1 === H2
        Tn2 = Tn1
    else
        verbose && !lowrank && println("$tag: building T_n(H2) moments...")
        Tn2, _, _ = KPM_Tn(H2.mpo, Ncheb, H2.sites;
                           scale=scale2, center=center2,
                           identity_mpo=physical_projector(H2),
                           maxdim=maxdim, cutoff=cutoff, verbose=false)
    end
    N = length(Tn1)   # = Ncheb + 1  (T_0 … T_Ncheb)

    verbose && println(lowrank ? "$tag: computing density matrices..." :
                                 "$tag: computing P1...")
    P1 = _get_density_matrix(H1, ϵF, P_method, Ncheb, maxdim, cutoff,
                             purify_method, purify_maxdim, purify_maxiters,
                             purify_tol, verbose)
    if H1 === H2
        P2 = P1
    else
        verbose && !lowrank && println("$tag: computing P2...")
        P2 = _get_density_matrix(H2, ϵF, P_method, Ncheb, maxdim, cutoff,
                                 purify_method, purify_maxdim, purify_maxiters,
                                 purify_tol, verbose)
    end

    TP1 = TP2 = nothing
    if !lowrank
        verbose && println("$tag: precomputing T_m(H1)·P1 and T_n(H2)·P2...")
        TP1 = [ITensorMPS.truncate!(
                   apply(Tn1[m], P1; maxdim=maxdim, cutoff=cutoff); cutoff=cutoff)
               for m in 1:N]
        TP2 = [ITensorMPS.truncate!(
                   apply(Tn2[n], P2; maxdim=maxdim, cutoff=cutoff); cutoff=cutoff)
               for n in 1:N]
    end

    diagonal && (out_sites = siteinds("Qubit", L))

    # All coefficient matrices at once: cheap DCT scalars, one N×N matrix per ω.
    verbose && println(lowrank ? "$tag: computing coefficient matrices for $nω frequencies..." :
                                 "$tag: precomputing C[m,n](ω) for all $nω frequencies...")
    C_all = [chebyshev2d_gf_coeffs(ω, scale1, center1, scale2, center2, η, N)
             for ω in ωlist]

    if lowrank
        if kernel == :jackson
            # The textbook Jackson kernel g_n (g_0 = 1) for the N moments T_0 … T_Ncheb:
            # _kpm_kernel(N + 1, :jackson) is (N + 1)·g_n for n = 0…N (see its comment).
            # Suppresses Gibbs oscillations; broadening ≈ π·scale/N.
            g_jk  = _kpm_kernel(N + 1, :jackson)[1:N] ./ (N + 1)
            G_jk  = g_jk * g_jk'         # N×N outer product, applied element-wise
            C_all = [G_jk .* C for C in C_all]
            verbose && println("$tag: Jackson kernel applied")
        elseif kernel != :none
            error("$fname: unknown kernel=$kernel (use :jackson or :none)")
        end
    end

    # Transpose the H₁-side Hadamard factors unless H₁ and P₁ are real (see
    # _hadamard_difference).
    transpose1 = !(_real_mpo(H1.mpo) && _real_mpo(P1))

    return (; N, Tn1, Tn2, P1, P2, TP1, TP2, C_all, out_sites, transpose1)
end


# The transpose of an MPO: the ket and bra legs of every tensor exchanged.
function _transpose_mpo(M::MPO)
    T = copy(M)
    for i in eachindex(M)
        ket, bra = _mpo_site_pair(M, i)
        T[i] = replaceinds(M[i], (ket, bra), (bra, ket))
    end
    return T
end

# Whether every tensor of an MPO stores real numbers.
_real_mpo(M::MPO) = all(T -> eltype(T) <: Real, M)


# D = (Aᵀ ⊙ B) − (Cᵀ ⊙ E) on `out_sites`, the ω-independent term of every cheb2d bubble:
# D_mn = TP1[m]ᵀ ⊙ Tn2[n] − Tn1[m]ᵀ ⊙ TP2[n] in the plain sweep, the same built from
# weighted sums of the moments in the SVD and Tucker variants. The H₁-side factors A
# and C are transposed so that Σ c_mn D_mn = Σ_ab f(ε_a, ε_b)(f_a − f_b) P_aᵀ ⊙ P_b,
# the Lindhard structure of get_bubble_mpo (Π_ij ∝ ⟨j|a⟩⟨a|i⟩⟨i|b⟩⟨b|j⟩). With
# `transpose1 = false` (_cheb2d_setup's choice for a real H₁ and P₁, whose spectral
# projectors are symmetric, so that the transpose would change only rounding) they are
# used as they are. (Until the fix they were never transposed: for a complex H₁ the
# bubble was P_a ⊙ P_b, which does not conserve particles.)
function _hadamard_difference(A::MPO, B::MPO, C::MPO, E::MPO, out_sites;
                              transpose1::Bool, maxdim::Int, cutoff::Real)
    if transpose1
        A = _transpose_mpo(A)
        C = _transpose_mpo(C)
    end
    had_A = hadamard_mpo(A, B, out_sites; maxdim=maxdim, cutoff=cutoff)
    had_B = hadamard_mpo(C, E, out_sites; maxdim=maxdim, cutoff=cutoff)
    return ITensorMPS.truncate!(+(had_A, -1 * had_B; maxdim=maxdim); cutoff=cutoff)
end


# k-space diagonal of an ω-independent term D, as an MPS: replace_sites maps D's output
# indices onto `sites` (H1.sites, the Qubit structure conjugate_by_qft expects), then
# QFT conjugation and diagonal extraction.
function _cheb2d_kdiag(D::MPO, sites; qft_tol::Real, qft_maxdim::Int, cutoff::Real)
    D_phys = replace_sites(D, sites)
    D_k    = conjugate_by_qft(D_phys; tol=qft_tol, maxdim=qft_maxdim)
    return ITensorMPS.truncate!(extract_diagonal_to_mps(D_k); cutoff=cutoff)
end


# acc[i] += c·X, truncated after each addition (maxdim in the sum, then cutoff); the
# first term initialises an empty (`nothing`) slot. The per-ω accumulation of every
# cheb2d bubble, for MPO and MPS terms alike.
function _accumulate_scaled!(acc::AbstractVector, i::Int, c::Number, X;
                             maxdim::Int, cutoff::Real)
    if acc[i] === nothing
        acc[i] = c * X
    else
        acc[i] = +(acc[i], c * X; maxdim=maxdim)
        ITensorMPS.truncate!(acc[i]; cutoff=cutoff)
    end
    return acc
end


# The online multi-ω (m,n) sweep of the plain cheb2d bubbles: the double loop runs once,
# each pair's ω-independent term X = term(m, n) is built once and c_mn(ω)·X is added
# into acc[iω] for every ω with |c_mn(ω)| ≥ coeff_tol. Rows m and pairs (m,n) whose
# coefficient is below coeff_tol for every ω are skipped without any MPO work.
# Returns (n_computed, n_skipped).
function _cheb2d_pair_sweep!(term, acc::AbstractVector, C_all::AbstractVector, N::Int;
                             coeff_tol::Real, maxdim::Int, cutoff::Real, verbose::Bool)
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

            X = term(m, n)

            # Accumulate c_mn(ω) · X into each acc[iω] simultaneously
            for (iω, C) in enumerate(C_all)
                c = C[m, n]
                abs(c) < coeff_tol && continue
                _accumulate_scaled!(acc, iω, c, X; maxdim=maxdim, cutoff=cutoff)
            end
        end

        verbose && println("  m=$m/$N  (computed $n_computed, skipped $n_skipped so far)")
    end

    return n_computed, n_skipped
end


# Tucker-2 bases of the stacked coefficient matrices, shared by all frequencies:
# HOSVD (independent mode SVDs of the mode-1 unfolding [C(ω₁) | C(ω₂) | …] and the
# mode-2 unfolding [C(ω₁)ᵀ | …], each N × (N·nω)) gives U_m (N×r_m) and V_n (N×r_n),
# r = min(tucker_maxrank, #σ > tucker_tol·σ_max); `hooi_iters` HOOI steps then
# re-optimise them jointly (alternating projection onto the optimal subspaces for the
# given rank). Returns (U_m, V_n, A_core) with the core tensor
# A_core[s₁,s₂,ω] = (U_m† C(ω) V_n)[s₁,s₂], so that C(ω) ≈ U_m A_core[:,:,ω] V_n†.
function _tucker_bases(C_all::AbstractVector{<:AbstractMatrix};
                       tucker_tol::Real, tucker_maxrank::Int, hooi_iters::Int)
    T1 = hcat(C_all...)                            # N × (N·nω) — mode-1 unfolding
    T2 = hcat([transpose(C) for C in C_all]...)   # N × (N·nω) — mode-2 unfolding
    F1 = svd(T1); F2 = svd(T2)
    r_m = min(tucker_maxrank, sum(F1.S .> tucker_tol * F1.S[1]))
    r_n = min(tucker_maxrank, sum(F2.S .> tucker_tol * F2.S[1]))
    U_m = F1.U[:, 1:r_m]
    V_n = F2.U[:, 1:r_n]

    for _ in 1:hooi_iters
        Y   = hcat([C * V_n  for C in C_all]...)   # N × (r_n·nω): contract n with V_n
        U_m = svd(Y).U[:, 1:r_m]
        Z   = hcat([C' * U_m for C in C_all]...)   # N × (r_m·nω): contract m with U_m
        V_n = svd(Z).U[:, 1:r_n]
    end

    nω     = length(C_all)
    A_core = zeros(ComplexF64, r_m, r_n, nω)
    for iω in 1:nω
        A_core[:, :, iω] = U_m' * C_all[iω] * V_n
    end
    return U_m, V_n, A_core
end


# The ω-independent Tucker components: the bare moments are summed first and P applied
# once per component, r_m + r_n MPO-MPO multiplications in total instead of the 2N of
# the plain variants' TP1/TP2:
#   C_tuck[s₁] = Σ_m U_m[m,s₁]·Tn1[m],        A_tuck[s₁] = C_tuck[s₁]·P1,
#   B_tuck[s₂] = Σ_n conj(V_n[n,s₂])·Tn2[n],  E_tuck[s₂] = B_tuck[s₂]·P2.
# A component whose weights are all negligible is `nothing`.
# Returns (A_tuck, B_tuck, C_tuck, E_tuck).
function _tucker_components(U_m::AbstractMatrix, V_n::AbstractMatrix,
                            Tn1::Vector{MPO}, Tn2::Vector{MPO}, P1, P2;
                            maxdim::Int, cutoff::Real)
    r_m = size(U_m, 2); r_n = size(V_n, 2)
    C_tuck = [_weighted_mpo_sum(U_m[:, s1],        Tn1; maxdim=maxdim, cutoff=cutoff) for s1 in 1:r_m]
    B_tuck = [_weighted_mpo_sum(conj.(V_n[:, s2]), Tn2; maxdim=maxdim, cutoff=cutoff) for s2 in 1:r_n]
    A_tuck = [isnothing(C_tuck[s1]) ? nothing :
              ITensorMPS.truncate!(apply(C_tuck[s1], P1; maxdim=maxdim, cutoff=cutoff); cutoff=cutoff)
              for s1 in 1:r_m]
    E_tuck = [isnothing(B_tuck[s2]) ? nothing :
              ITensorMPS.truncate!(apply(B_tuck[s2], P2; maxdim=maxdim, cutoff=cutoff); cutoff=cutoff)
              for s2 in 1:r_n]
    return A_tuck, B_tuck, C_tuck, E_tuck
end


# The r_m × r_n ω-independent Tucker terms
#   D[s₁,s₂] = post((A_tuck[s₁]ᵀ ⊙ B_tuck[s₂]) − (C_tuck[s₁]ᵀ ⊙ E_tuck[s₂]))
#            = post(Σ_{m,n} U[m,s₁] conj(V[n,s₂]) · D_mn),
# with post = identity for the MPO bubble and the k-space diagonal (_cheb2d_kdiag) for
# the diagonal one (the transposes as `transpose1` says, see _hadamard_difference).
# Returns a Matrix{Union{Nothing, T}}, `nothing` where a component is.
function _tucker_hadamard(post, A_tuck::AbstractVector, B_tuck::AbstractVector,
                          C_tuck::AbstractVector, E_tuck::AbstractVector, out_sites,
                          ::Type{T}; transpose1::Bool, maxdim::Int, cutoff::Real,
                          verbose::Bool) where {T}
    r_m = length(A_tuck); r_n = length(B_tuck)
    D   = Matrix{Union{Nothing, T}}(nothing, r_m, r_n)
    for s1 in 1:r_m, s2 in 1:r_n
        (isnothing(A_tuck[s1]) || isnothing(B_tuck[s2]) ||
         isnothing(C_tuck[s1]) || isnothing(E_tuck[s2])) && continue

        D[s1, s2] = post(_hadamard_difference(A_tuck[s1], B_tuck[s2], C_tuck[s1], E_tuck[s2],
                                              out_sites; transpose1=transpose1, maxdim=maxdim,
                                              cutoff=cutoff))
        if verbose
            idx = (s1 - 1) * r_n + s2
            (idx % 10 == 0 || idx == r_m * r_n) &&
                println("  ($s1,$s2)/($r_m,$r_n) done  [$idx/$(r_m*r_n)]")
        end
    end
    return D
end


# Per-ω Tucker accumulation, scalar × term additions only:
#   acc[ω] = Σ_{s₁,s₂} A_core[s₁,s₂,ω] · D[s₁,s₂],
# skipping core entries below coeff_tol and missing terms. Returns a
# Vector{Union{Nothing, T}}, `nothing` for an ω that received no term.
function _tucker_accumulate(A_core::AbstractArray{<:Number,3}, D::AbstractMatrix,
                            ::Type{T}; coeff_tol::Real, maxdim::Int,
                            cutoff::Real) where {T}
    r_m, r_n, nω = size(A_core)
    acc = Vector{Union{Nothing, T}}(nothing, nω)
    for iω in 1:nω
        for s1 in 1:r_m, s2 in 1:r_n
            a = A_core[s1, s2, iω]
            (abs(a) < coeff_tol || isnothing(D[s1, s2])) && continue
            _accumulate_scaled!(acc, iω, a, D[s1, s2]; maxdim=maxdim, cutoff=cutoff)
        end
    end
    return acc
end


# ============================================================
# 3. Full-MPO bubbles
# ============================================================

"""
    get_bubble_mpo_cheb2d(H1, H2, ωlist; Ncheb=50, maxdim=200, cutoff=1e-8,
                           ϵF=0.0, P_method=:purification, purify_method=:mcweeny,
                           purify_maxdim=40, purify_maxiters=30, purify_tol=1e-5,
                           η=1e-3, coeff_tol=1e-12, verbose=false) -> Vector{MPO}

Compute the non-interacting polarization bubble Π₀(ω) for each ω in `ωlist`
using the **double Chebyshev decomposition**.

Π₀ lives on `H1.sites`, including any spin, Nambu, layer or sublattice index, and
is resolved in those indices, as the result of `get_bubble_mpo` is. `H1` and `H2`
must have the same site structure.

Sign: the cheb2d bubbles carry the opposite sign to `get_bubble_mpo` (Π₀ here is
`−get_bubble_mpo(H1, H2, ω)` up to the expansion error).

Instead of building the 2L-site effective Hamiltonian Heff = I⊗H₂ − H₁⊗I and
running KPM on it (where bond dimension grows at each Chebyshev step due to
entanglement between subsystems), this routine decomposes G_eff as

    G_eff(ω) ≈ Σ_{mn} c_{mn}(ω) · T_m(H̃₁) ⊗ T_n(H̃₂)

where T_m, T_n are Chebyshev polynomials of the *L-site* rescaled Hamiltonians
and c_{mn}(ω) are scalar 2D Chebyshev coefficients (cheap, via DCT-II).

The bubble on L-site MPOs is assembled as

    Π₀(ω) = Σ_{mn} c_{mn}(ω) · D_{mn},

where D_{mn} = (T_m(H̃₁)·P₁)ᵀ ⊙ T_n(H̃₂) − T_m(H̃₁)ᵀ ⊙ (T_n(H̃₂)·P₂)
and ⊙ is the site-wise Hadamard product (`hadamard_mpo`). The transposes give the
Lindhard structure Π₀ = Σ_ab (f_a − f_b)/(ω + iη − (ε_b − ε_a)) · (P_a)ᵀ ⊙ P_b of the
eigenprojectors P_a of H₁ and P_b of H₂ (f the occupations of P₁, P₂); for a real H₁
(and P₁) they change nothing and are skipped.

**Online multi-ω sweep**: All coefficient matrices `C[m,n](ω)` are precomputed
at once (cheap DCT scalars). The (m,n) double loop runs once; each D_{mn} is
computed once and accumulated into every Π(ω) simultaneously using the scalar
c_{mn}(ω). This matches the KPM "online" paradigm: the expensive MPO work
(Hadamard products) is done once and shared across all frequencies.

**Keyword arguments**
- `Ncheb`         : Chebyshev expansion order. Default `50`.
- `maxdim`        : Max bond dimension throughout. Default `200`.
- `cutoff`        : SVD truncation cutoff. Default `1e-8`.
- `ϵF`            : Fermi energy. Default `0.0`. It reaches the density matrices as
                    in `get_bubble_mpo`.
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
    S = _cheb2d_setup(H1, H2, ωlist, "get_bubble_mpo_cheb2d", "cheb2d";
                      diagonal=false, lowrank=false, Ncheb, maxdim, cutoff, ϵF, P_method,
                      purify_method, purify_maxdim, purify_maxiters, purify_tol, η, verbose)
    N  = S.N   # = Ncheb + 1  (T_0 … T_Ncheb)
    nω = length(ωlist)

    # --- Online multi-ω: sweep (m,n) once and accumulate into every Π(ω). ---
    Π = Vector{Union{Nothing, MPO}}(nothing, nω)
    n_computed, n_skipped = _cheb2d_pair_sweep!(Π, S.C_all, N; coeff_tol, maxdim, cutoff,
                                                verbose) do m, n
        # D_mn = TP1[m]ᵀ ⊙ Tn2[n] − Tn1[m]ᵀ ⊙ TP2[n]  (ω-independent)
        _hadamard_difference(S.TP1[m], S.Tn2[n], S.Tn1[m], S.TP2[n], S.out_sites;
                             transpose1=S.transpose1, maxdim, cutoff)
    end

    verbose && println("cheb2d: done — $(n_computed)/$(N*N) (m,n) pairs computed, $n_skipped skipped")

    # Map output physical indices (out_sites) back to H1.sites
    return [replace_sites(Π[iω], H1.sites) for iω in 1:nω]
end


"""
    get_bubble_mpo_cheb2d_tucker(H1, H2, ωlist; Ncheb=50, maxdim=200, cutoff=1e-8,
                                  ϵF=0.0, P_method=:purification, purify_method=:mcweeny,
                                  purify_maxdim=40, purify_maxiters=30, purify_tol=1e-5,
                                  η=1e-3, coeff_tol=1e-12, tucker_tol=1e-3,
                                  tucker_maxrank=20, kernel=:jackson, hooi_iters=3,
                                  verbose=false) -> Vector{MPO}

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
layer or sublattice index, and carries the opposite sign to `get_bubble_mpo`.

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
    S = _cheb2d_setup(H1, H2, ωlist, "get_bubble_mpo_cheb2d_tucker", "cheb2d_mpo_tucker";
                      diagonal=false, lowrank=true, kernel, Ncheb, maxdim, cutoff, ϵF,
                      P_method, purify_method, purify_maxdim, purify_maxiters, purify_tol, η,
                      verbose)
    nω = length(ωlist)

    # ── Tucker bases (HOSVD + HOOI) and core tensor G[s₁,s₂,ω] = (U_m† C(ω) V_n)[s₁,s₂] ──
    U_m, V_n, A_core = _tucker_bases(S.C_all; tucker_tol, tucker_maxrank, hooi_iters)
    r_m, r_n = size(U_m, 2), size(V_n, 2)
    verbose && println("cheb2d_mpo_tucker: Tucker ranks r_m=$r_m, r_n=$r_n (HOSVD + $hooi_iters HOOI iters) → $(r_m*r_n) Hadamard operations")

    # ── ω-independent weighted MPO sums ──────────────────────────────────────
    verbose && println("cheb2d_mpo_tucker: computing Tucker MPO components (r_m=$r_m, r_n=$r_n)...")
    A_tuck, B_tuck, C_tuck, E_tuck = _tucker_components(U_m, V_n, S.Tn1, S.Tn2, S.P1, S.P2;
                                                        maxdim, cutoff)

    # ── ω-independent Hadamard products: r_m × r_n total ────────────────────
    verbose && println("cheb2d_mpo_tucker: computing $(r_m*r_n) Hadamard products...")
    D_tuck = _tucker_hadamard(identity, A_tuck, B_tuck, C_tuck, E_tuck, S.out_sites, MPO;
                              transpose1=S.transpose1, maxdim, cutoff, verbose)

    # ── Per-ω accumulation: Π(ω) = Σ_{s₁,s₂} G[s₁,s₂,ω] · D[s₁,s₂] ──────────
    Π = _tucker_accumulate(A_core, D_tuck, MPO; coeff_tol, maxdim, cutoff)

    verbose && println("cheb2d_mpo_tucker: done — r_m=$r_m, r_n=$r_n, $(count(!isnothing, Π))/$nω non-zero")
    return [replace_sites(Π[iω]::MPO, H1.sites) for iω in 1:nω]
end


# ============================================================
# 4. k-space diagonal bubble
# ============================================================

"""
    get_bubble_diag_cheb2d(H1, H2, ωlist; Ncheb=50, maxdim=200, cutoff=1e-8,
                            ϵF=0.0, P_method=:purification, purify_method=:mcweeny,
                            purify_maxdim=40, purify_maxiters=30, purify_tol=1e-5,
                            η=1e-3, coeff_tol=1e-12, qft_tol=1e-9, qft_maxdim=100,
                            verbose=false) -> Vector{MPS}

Diagonal-only variant of `get_bubble_mpo_cheb2d`.

Returns the k-space diagonal of the non-interacting polarization bubble,
    diag_Π₀(k, ω) = ⟨k| Π₀(ω) |k⟩,
as a `Vector{MPS}` (one MPS per ω in `ωlist`) ready for direct plotting.
Π₀ is that of `get_bubble_mpo_cheb2d`, with the opposite sign to `get_bubble_mpo`.

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
    S = _cheb2d_setup(H1, H2, ωlist, "get_bubble_diag_cheb2d", "cheb2d_diag";
                      diagonal=true, lowrank=false, Ncheb, maxdim, cutoff, ϵF, P_method,
                      purify_method, purify_maxdim, purify_maxiters, purify_tol, η, verbose)
    N  = S.N
    nω = length(ωlist)

    # Accumulate diagonal MPS (not full MPO) for each ω. The QFT and the diagonal
    # extraction are done ONCE per (m,n), shared across all ω, and the MPS additions
    # are much cheaper than MPO additions (bond dim ∝ D vs D²).
    diag_Π = Vector{Union{Nothing, MPS}}(nothing, nω)
    n_computed, n_skipped = _cheb2d_pair_sweep!(diag_Π, S.C_all, N; coeff_tol, maxdim,
                                                cutoff, verbose) do m, n
        # D_mn = TP1[m]ᵀ ⊙ Tn2[n] − Tn1[m]ᵀ ⊙ TP2[n]  (ω-independent)
        D_mn = _hadamard_difference(S.TP1[m], S.Tn2[n], S.Tn1[m], S.TP2[n], S.out_sites;
                                    transpose1=S.transpose1, maxdim, cutoff)
        _cheb2d_kdiag(D_mn, H1.sites; qft_tol, qft_maxdim, cutoff)
    end

    verbose && println("cheb2d_diag: done — $(n_computed)/$(N*N) pairs computed, $n_skipped skipped")

    return [diag_Π[iω] for iω in 1:nω]
end


# ============================================================
# 5. Shared helper: weighted MPO sum
# ============================================================

# (The Jackson kernel of the low-rank variants is the shared _kpm_kernel, see
# _cheb2d_setup. The RPA-only _jackson_kernel(N) it replaced had (N − m) where the
# kernel for N moments has (N − m + 1), so its g_0 was N/(N + 1).)

# Weighted MPO sum  Σ_i w_i · mpos[i]  with online truncation (_chebyshev_sum,
# solvers/kpm/cached.jl), over the pairs with |w_i| ≥ weight_tol; `nothing` if none.
# Accepts real or complex weights; complex weights produce complex-tensor MPOs.
function _weighted_mpo_sum(weights::AbstractVector{<:Number}, mpos::Vector{MPO};
                           maxdim::Int, cutoff::Real, weight_tol::Real = 1e-14)
    terms = [(mpo, w) for (w, mpo) in zip(weights, mpos) if !(abs(w) < weight_tol)]
    isempty(terms) && return nothing
    return _chebyshev_sum(first.(terms), last.(terms); maxdim=maxdim, cutoff=cutoff)
end


# ============================================================
# 6. Low-rank k-space diagonal bubbles (per-ω SVD, Tucker)
# ============================================================

"""
    get_bubble_diag_cheb2d_svd(H1, H2, ωlist; Ncheb=50, maxdim=200, cutoff=1e-8,
                                ϵF=0.0, P_method=:purification, purify_method=:mcweeny,
                                purify_maxdim=40, purify_maxiters=30, purify_tol=1e-5,
                                η=1e-3, qft_tol=1e-9, qft_maxdim=100, svd_tol=1e-6,
                                svd_maxrank=20, kernel=:jackson,
                                verbose=false) -> Vector{MPS}

Per-ω SVD-accelerated variant of `get_bubble_diag_cheb2d`, with the same
requirement that `H.sites` be the `H.L` position qubits.  It takes the keywords of
`get_bubble_diag_cheb2d` except `coeff_tol`.

For each frequency ω the coefficient matrix `C[m,n](ω)` is rank-truncated via its own SVD:

    `C[m,n](ω) = Σ_s  S_s(ω) · U[m,s](ω) · conj(V[n,s](ω))`   (exact up to truncation)

The per-ω rank r(ω) (typically 2–5 for smooth Lorentzian kernels) is usually much
smaller than the Tucker/joint-SVD rank, which must span all frequencies simultaneously.
For each (ω, s) one Hadamard product and one QFT are performed, giving

    `diag_Π[ω] = Σ_s S_s(ω) · diag(QFT( A_sᵀ ⊙ B_s − C_sᵀ ⊙ E_s ))`

where `A_s = Σ_m U[m,s]·TP1[m]`, `B_s = Σ_n conj(V[n,s])·Tn2[n]`, etc. (the transposes
as in `get_bubble_mpo_cheb2d`). The sign is opposite to that of `get_bubble_mpo`.

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
    S = _cheb2d_setup(H1, H2, ωlist, "get_bubble_diag_cheb2d_svd", "cheb2d_diag_svd";
                      diagonal=true, lowrank=true, kernel, Ncheb, maxdim, cutoff, ϵF,
                      P_method, purify_method, purify_maxdim, purify_maxiters, purify_tol, η,
                      verbose)
    nω = length(ωlist)

    # ── Per-ω SVD of the coefficient matrix C[m,n](ω) ───────────────────────
    # For each ω, the exact SVD gives the optimal low-rank factorisation:
    #   C(ω) = U(ω) · Diagonal(S(ω)) · V(ω)ᴴ
    # The per-ω rank r(ω) is typically much smaller than the Tucker/joint rank,
    # because each individual C(ω) is structured by a single Lorentzian kernel
    # and doesn't need to share a common basis with other frequencies.
    diag_Π = Vector{Union{Nothing, MPS}}(nothing, nω)
    ranks   = Int[]

    for (iω, C) in enumerate(S.C_all)
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
            C_s = _weighted_mpo_sum(u_s, S.Tn1; maxdim=maxdim, cutoff=cutoff)
            B_s = _weighted_mpo_sum(v_s, S.Tn2; maxdim=maxdim, cutoff=cutoff)
            (isnothing(C_s) || isnothing(B_s)) && continue
            A_s = ITensorMPS.truncate!(apply(C_s, S.P1; maxdim=maxdim, cutoff=cutoff); cutoff=cutoff)
            E_s = ITensorMPS.truncate!(apply(B_s, S.P2; maxdim=maxdim, cutoff=cutoff); cutoff=cutoff)

            (isnothing(A_s) || isnothing(E_s)) && continue

            D      = _hadamard_difference(A_s, B_s, C_s, E_s, S.out_sites;
                                          transpose1=S.transpose1, maxdim, cutoff)
            diag_s = _cheb2d_kdiag(D, H1.sites; qft_tol, qft_maxdim, cutoff)
            _accumulate_scaled!(diag_Π, iω, σ_s, diag_s; maxdim, cutoff)
        end

        verbose && println("  ω=$(round(ωlist[iω];digits=3))  rank=$r_ω")
    end

    r_min, r_max = extrema(ranks)
    r_mean = round(sum(ranks) / length(ranks); digits=1)
    verbose && println("cheb2d_diag_svd: done — per-ω ranks min=$r_min max=$r_max mean=$r_mean, $(count(!isnothing, diag_Π))/$nω non-zero")
    return [diag_Π[iω] for iω in 1:nω]
end


"""
    get_bubble_diag_cheb2d_tucker(H1, H2, ωlist; Ncheb=50, maxdim=200, cutoff=1e-8,
                                   ϵF=0.0, P_method=:purification, purify_method=:mcweeny,
                                   purify_maxdim=40, purify_maxiters=30, purify_tol=1e-5,
                                   η=1e-3, coeff_tol=1e-12, qft_tol=1e-9, qft_maxdim=100,
                                   tucker_tol=1e-3, tucker_maxrank=20, kernel=:jackson,
                                   hooi_iters=3, verbose=false) -> Vector{MPS}

Tucker (HOSVD) variant of `get_bubble_diag_cheb2d`, with the same requirement
that `H.sites` be the `H.L` position qubits, and the same sign (opposite to that of
`get_bubble_mpo`).

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
    S = _cheb2d_setup(H1, H2, ωlist, "get_bubble_diag_cheb2d_tucker", "cheb2d_tucker";
                      diagonal=true, lowrank=true, kernel, Ncheb, maxdim, cutoff, ϵF,
                      P_method, purify_method, purify_maxdim, purify_maxiters, purify_tol, η,
                      verbose)
    nω = length(ωlist)

    # ── Tucker bases (HOSVD + HOOI) and core tensor Ã[s1,s2,ω] = (U_m† C(ω) V_n)[s1,s2] ──
    U_m, V_n, A_core = _tucker_bases(S.C_all; tucker_tol, tucker_maxrank, hooi_iters)
    r_m, r_n = size(U_m, 2), size(V_n, 2)
    verbose && println("cheb2d_tucker: Tucker ranks r_m=$r_m, r_n=$r_n (HOSVD + $hooi_iters HOOI iters) → $(r_m*r_n) Hadamard+QFT operations")

    # ── ω-independent weighted MPO sums + deferred P application ────────────
    verbose && println("cheb2d_tucker: computing Tucker MPO components...")
    A_tuck, B_tuck, C_tuck, E_tuck = _tucker_components(U_m, V_n, S.Tn1, S.Tn2, S.P1, S.P2;
                                                        maxdim, cutoff)

    # ── ω-independent Hadamard + QFT  (r_m × r_n total) ─────────────────────
    verbose && println("cheb2d_tucker: computing $(r_m*r_n) Hadamard+QFT components...")
    kdiag  = D -> _cheb2d_kdiag(D, H1.sites; qft_tol, qft_maxdim, cutoff)
    diag_D = _tucker_hadamard(kdiag, A_tuck, B_tuck, C_tuck, E_tuck, S.out_sites, MPS;
                              transpose1=S.transpose1, maxdim, cutoff, verbose)

    # ── Accumulate per ω: scalar × MPS additions only ────────────────────────
    diag_Π = _tucker_accumulate(A_core, diag_D, MPS; coeff_tol, maxdim, cutoff)

    verbose && println("cheb2d_tucker: done — r_m=$r_m, r_n=$r_n, $(count(!isnothing, diag_Π))/$nω non-zero")
    return [diag_Π[iω] for iω in 1:nω]
end
