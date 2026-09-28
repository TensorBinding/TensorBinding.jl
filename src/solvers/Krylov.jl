# solvers/Krylov.jl — Krylov-space Green's functions
#
# Contents: the retarded single-particle Green's function G(ω) = (ω + iη − H)⁻¹ as
# an MPO, from the vectorized linear system
#
#     [(ω + iη − H) ⊗ I] |G⟩⟩ = |I⟩⟩
#
# solved by ITensorMPS.linsolve (get_green_krylov), where |M⟩⟩ is the vectorized
# (MPS) representation of the matrix M on a 2L-site interleaved quantics chain (odd
# sites = row bits, even sites = column bits; built by _vec_mps_from_mpo); and the
# operator-level Haydock (Lanczos) recursion with its continued fraction and
# resolvent MPO (haydock_cf, eval_haydock_cf, haydock_resolve_mpo).
#
# Entry points: get_green_krylov, haydock_cf, eval_haydock_cf, haydock_resolve_mpo
# Depends on: core/Utils.jl (custom_mpo), core/MPOTools.jl (interleave_mpo),
#   core/TBSystem.jl (TBHamiltonian).
#
# Section 2 moved here from physics/rpa/bubble.jl in Tier 1.


# ============================================================
# 1. Green's function by vectorized linear solve
# ============================================================

"""
    _vec_mps_from_mpo(G_mpo::MPO, sites2; cutoff=1e-12, maxdim=typemax(Int)) -> MPS

Convert an L-site MPO into the 2L-site interleaved vectorized MPS that
`get_green_krylov` uses for the right-hand side `|I⟩⟩` and for the optional
initial guess `x0_mpo`.

Each MPO tensor at site k is split by SVD into two MPS tensors at positions
(2k-1, 2k): the bra (primed) physical index maps to the odd row site and the
ket (unprimed) index maps to the even column site, the encoding that `custom_mpo`
reads back into an MPO.
"""
function _vec_mps_from_mpo(G_mpo::MPO, sites2::Vector{<:Index};
                            cutoff::Real = 1e-12,
                            maxdim::Int  = typemax(Int))
    L = length(G_mpo)
    @assert length(sites2) == 2L "_vec_mps_from_mpo: sites2 length $(length(sites2)) ≠ 2L=$(2L)"
    tensors = Vector{ITensor}(undef, 2L)

    for k in 1:L
        T     = G_mpo[k]
        s_ket = noprime(siteind(G_mpo, k))
        s_bra = prime(s_ket)

        # Relabel physical indices: bra→odd row qubit, ket→even col qubit
        T2 = replaceinds(T, [s_bra, s_ket] => [sites2[2k-1], sites2[2k]])

        left_inds = Index[sites2[2k-1]]
        k > 1 && push!(left_inds, commonind(G_mpo[k], G_mpo[k-1]))

        U, S, V = svd(T2, left_inds...;
                      cutoff    = cutoff,
                      maxdim    = maxdim,
                      lefttags  = "Link,l=$(2k-1)",
                      righttags = "Link,l=$(2k-1)r")
        tensors[2k-1] = U * S
        tensors[2k]   = V
    end
    return MPS(tensors)
end


"""
    get_green_krylov(H_mpo::MPO, sites, ω_phys; η=1e-2, nsweeps=12, maxdim=100,
                     cutoff=1e-8, x0_mpo=nothing, ishermitian=false, tol=1e-10,
                     maxiter=600, krylovdim=30, verbose=false) -> MPO

Low-level: compute the retarded Green's function G(ω) = (ω + iη − H)⁻¹ for a
raw MPO `H_mpo` defined on `sites`, via the vectorized linear system

    [(ω + iη − H) ⊗ I] |G⟩⟩ = |I⟩⟩

See the `TBHamiltonian` overload for the full keyword-argument reference.
"""
function get_green_krylov(H_mpo::MPO, sites::Vector{<:Index}, ω_phys::Real;
                          η::Real                    = 1e-2,
                          nsweeps::Int               = 12,
                          maxdim::Int                = 100,
                          cutoff::Real               = 1e-8,
                          x0_mpo::Union{MPO,Nothing} = nothing,
                          ishermitian::Bool          = false,
                          tol::Real                  = 1e-10,
                          maxiter::Int               = 600,
                          krylovdim::Int             = 30,
                          verbose::Bool              = false)
    N      = length(sites)
    sites2 = siteinds("Qubit", 2N)

    z     = ComplexF64(ω_phys + im * η)
    ω_mpo = z * MPO(sites, "Id") - H_mpo
    Lop   = interleave_mpo(ω_mpo, sites2, 0)
    rhs   = _vec_mps_from_mpo(MPO(sites, "Id"), sites2)
    x0    = isnothing(x0_mpo) ? deepcopy(rhs) :
                _vec_mps_from_mpo(x0_mpo, sites2; cutoff=cutoff, maxdim=maxdim)

    verbose && println("Krylov GF: ω = $ω_phys + $(η)i  (N=$N, maxdim=$maxdim, nsweeps=$nsweeps)",
                       isnothing(x0_mpo) ? "" : "  [KPM warm start]")

    sol = ITensorMPS.linsolve(Lop, rhs, x0;
                              nsweeps        = nsweeps,
                              maxdim         = maxdim,
                              cutoff         = cutoff,
                              updater_kwargs = (; ishermitian, tol, maxiter, krylovdim))
    return custom_mpo(sol, sites)
end


"""
    get_green_krylov(H::TBHamiltonian, ω_phys; η=1e-2, nsweeps=12, maxdim=100,
                     cutoff=1e-8, x0_mpo=nothing, ishermitian=false, tol=1e-10,
                     maxiter=600, krylovdim=30, verbose=false) -> MPO

Compute the retarded Green's function

    G(ω) = (ω + iη − H)⁻¹

as an MPO by solving the vectorized linear system

    [(ω + iη − H) ⊗ I] |G⟩⟩ = |I⟩⟩

using `ITensorMPS.linsolve` (DMRG-like Krylov solver).  The Hamiltonian is used
unscaled — no KPM Chebyshev expansion required.

**Keyword arguments**
- `η`           : Lorentzian broadening. Default `1e-2`.
- `nsweeps`     : Number of DMRG sweeps for the linear solver. Default `12`.
- `maxdim`      : Maximum bond dimension of the solution MPS. Default `100`.
- `cutoff`      : SVD truncation cutoff. Default `1e-8`.
- `x0_mpo`      : Optional MPO initial guess for G(ω).  When provided it is
                  vectorized via `_vec_mps_from_mpo` and passed as `x0` to
                  `linsolve`, replacing the default identity-matrix guess.
                  Typical use: pass a low-accuracy KPM Green's function to
                  warm-start the Krylov iteration.  Default `nothing`.
- `ishermitian` : Set `true` only when the shifted operator is Hermitian
                  (requires purely imaginary η = 0, not physical for GF). Default `false`.
- `tol`         : Krylov solver convergence tolerance. Default `1e-10`.
- `maxiter`     : Maximum Krylov iterations per site. Default `600`.
- `krylovdim`   : Krylov subspace dimension. Default `30`.
- `verbose`     : Print progress messages. Default `false`.

**Usage**
```julia
G = get_green_krylov(H, ω; η=0.05, nsweeps=20, maxdim=200)
dos  = -imag(tr(G)) / π                              # total DoS
ldos = real(inner(psi_i, apply(G, psi_i)))           # LDoS at site i
gij  = inner(psi_i, apply(G, psi_j))                 # off-diagonal element

# Warm-start from a cheap KPM estimate
TensorBinding.KPM_Tn(H, 15; maxdim=50)
G_kpm = TensorBinding.get_Green_retarded_from_Tn(H._tn_cache, 15, ω; η=η, maxdim=50)
G_ws  = get_green_krylov(H, ω; x0_mpo=G_kpm, nsweeps=6, maxdim=200)
```
"""
function get_green_krylov(H::TBHamiltonian, ω_phys::Real;
                          η::Real                    = 1e-2,
                          nsweeps::Int               = 12,
                          maxdim::Int                = 100,
                          cutoff::Real               = 1e-8,
                          x0_mpo::Union{MPO,Nothing} = nothing,
                          ishermitian::Bool          = false,
                          tol::Real                  = 1e-10,
                          maxiter::Int               = 600,
                          krylovdim::Int             = 30,
                          verbose::Bool              = false)
    return get_green_krylov(H.mpo, H.sites, ω_phys;
                            η, nsweeps, maxdim, cutoff, x0_mpo, ishermitian,
                            tol, maxiter, krylovdim, verbose)
end


# ============================================================
# 2. Haydock recursion (operator-level Krylov)
# ============================================================

# The Hilbert-Schmidt (Frobenius) inner product Tr[A† B] of two MPOs on the same
# sites, contracted exactly: ITensorMPS's `inner` pairs dag(A[j]) with B[j] over both
# site legs. Until the fix haydock_cf took tr(apply(dag(A), B)) = Tr[conj(A)·B], which
# is Tr[A† B] only for symmetric A (and is negative for A = B imaginary Hermitian, so
# the seed norm threw a DomainError), with the product truncated before the trace.
_hs_inner(A::MPO, B::MPO) = inner(A, B)

"""
    haydock_cf(H_mpo::MPO, seed::MPO, N_steps::Int; maxdim=200, cutoff=1e-8,
               verbose=false)
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

`H_mpo` must be Hermitian (then H· is Hermitian for Tr[A† B] and every aₙ is real);
the seed may be any MPO on the same sites, real or complex.
"""
function haydock_cf(H_mpo::MPO, seed::MPO, N_steps::Int;
                    maxdim::Int   = 200,
                    cutoff::Real  = 1e-8,
                    verbose::Bool = false)

    a     = zeros(Float64, N_steps)
    b     = zeros(Float64, N_steps)
    basis = Vector{MPO}(undef, N_steps)

    norm0    = sqrt(real(_hs_inner(seed, seed)))
    b[1]     = norm0
    Phi_prev = nothing
    Phi_curr = (1.0 / norm0) * seed

    actual_N = N_steps
    for n in 1:N_steps
        basis[n] = Phi_curr

        HPhi = apply(H_mpo, Phi_curr; maxdim=maxdim, cutoff=cutoff)
        a[n] = real(_hs_inner(Phi_curr, HPhi))

        r = +(HPhi, (-a[n]) * Phi_curr; maxdim=maxdim)
        ITensorMPS.truncate!(r; cutoff=cutoff)
        if n > 1
            r = +(r, (-b[n]) * Phi_prev; maxdim=maxdim)
            ITensorMPS.truncate!(r; cutoff=cutoff)
        end

        b_next = sqrt(max(0.0, real(_hs_inner(r, r))))
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
    haydock_resolve_mpo(a, b, basis, z; maxdim=200, cutoff=1e-8) -> MPO

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
