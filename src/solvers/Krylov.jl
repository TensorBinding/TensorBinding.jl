# solvers/Krylov.jl — Krylov-space Green's functions
#
# Contents: the retarded single-particle Green's function G(ω) = (ω + iη − H)⁻¹ as
# an MPO, from the vectorized linear system
#
#     [I ⊗ (ω + iη − H)ᵀ] |G⟩⟩ = |I⟩⟩     (G (ω + iη − H) = I, operator on the column sites)
#
# solved by ITensorMPS.linsolve (get_green_krylov), where |M⟩⟩ is the vectorized
# (MPS) representation of the matrix M on a 2L-site interleaved quantics chain (odd
# sites = row bits, even sites = column bits; built by _vec_mps_from_mpo), with
# KrylovKit's GMRES or MINRES (_minres_shifted, _minres_updater) as the local solver
# of the sweeps; and the operator-level Haydock (Lanczos) recursion with its
# continued fraction and resolvent MPO (haydock_cf, eval_haydock_cf,
# haydock_resolve_mpo) with its Hilbert-Schmidt inner product _hs_inner.
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
    _minres_shifted(A, b, x0, a0, a1; tol=1e-10, maxiter=2000) -> (x, info)

Solve `(a0 + a1·A) x = b` for a Hermitian linear map `A` (a function `v -> A v`) and
complex `a0`, `a1 ≠ 0` by MINRES: the Lanczos three-term recurrence of `A` from the
residual of `x0`, with complex Givens rotations on the shifted tridiagonal
`T − σ`, `σ = −a0/a1`. In exact arithmetic the iterates are those of GMRES without
restarts, kept in three vectors.

Stops when the residual estimate `|a1|·|t|` is at most `tol` (absolute, as
KrylovKit's `tol`), on a Lanczos breakdown, or after `maxiter` products with `A`.
`info = (; converged, numops, residual)`, with `numops` the products with `A`
(the residual of `x0` included) and `residual` the final estimate.

Works on any vector type with `norm`, `dot`, `copy` and linear combinations
(ITensors, `Vector`s). A non-Hermitian `A` gives a wrong `x` without an error.
"""
function _minres_shifted(A, b, x0, a0::Number, a1::Number;
                         tol::Real = 1e-10, maxiter::Int = 2000)
    σ   = -a0 / a1
    r   = b - (a0 * x0 + a1 * A(x0))
    nops = 1
    rhs = r / a1
    β1  = norm(rhs)
    x   = copy(x0)
    abs(a1) * β1 <= tol && return x, (; converged = true, numops = nops, residual = abs(a1) * β1)
    v     = rhs / β1
    vprev = nothing
    βk    = 0.0
    m1 = nothing; m2 = nothing                        # columns k−1, k−2 of V R⁻¹
    p1, q1 = one(ComplexF64), 0.0                     # rotation G_{k−1} = [p̄ q; −q p]
    p2, q2 = one(ComplexF64), 0.0                     # rotation G_{k−2}
    t   = complex(β1)                                 # current component of the rotated rhs
    res = abs(a1) * β1
    for _ in 1:maxiter
        w = A(v); nops += 1
        vprev === nothing || (w = w - βk * vprev)
        α = real(dot(v, w))
        w = w - α * v
        βn = norm(w)
        # column k of T − σ: rows k−2, k−1, k, k+1 hold (0, βk, α − σ, βn)
        ε  = q2 * βk                                  # G_{k−2} on (0, βk)
        δ  = p2 * βk
        δ2 = conj(p1) * δ + q1 * (α - σ)              # G_{k−1} on (δ, α − σ)
        γ  = -q1 * δ + p1 * (α - σ)
        ρ  = sqrt(abs2(γ) + βn^2)                     # G_k annihilates βn
        p, q = γ / ρ, βn / ρ
        τ = conj(p) * t
        t = -q * t
        m = v
        m1 === nothing || (m = m - δ2 * m1)
        m2 === nothing || (m = m - ε * m2)
        m = m / ρ
        x = x + τ * m
        res = abs(a1) * abs(t)
        m2, m1 = m1, m
        p2, q2, p1, q1 = p1, q1, p, q
        (res <= tol || βn <= 1e-14 * β1) &&
            return x, (; converged = res <= tol, numops = nops, residual = res)
        vprev = v
        v     = w / βn
        βk    = βn
    end
    return x, (; converged = false, numops = nops, residual = res)
end


# Local updater of ITensorMPS.linsolve: MINRES on (a0 + a1·A_eff) x = b, where A_eff is
# the operator projected on the two-site space; Hermitian when the MPO is (orthonormal
# environments). `stats` (local solves, products, unconverged solves) is updated in place.
function _minres_updater(problem, init; internal_kwargs, coefficients,
                         tol = 1e-10, maxiter = 2000, stats = nothing, kwargs...)
    A = ITensorMPS.operator(problem)
    b = ITensorMPS.constant_term(problem)
    x, info = _minres_shifted(v -> A(v), b, init, coefficients[1], coefficients[2];
                              tol, maxiter)
    if stats !== nothing
        stats[1] += 1
        stats[2] += info.numops
        info.converged || (stats[3] += 1)
    end
    return x, (; info)
end


# ‖H − H†‖_F / ‖H‖_F from Hilbert-Schmidt products of the MPOs (exact contractions, no
# MPO sum): ‖H − H†‖² = 2‖H‖² − 2 Re Tr[H H]. The difference cancels to ~√eps, so a
# Hermitian MPO gives ≲ 1e-7.
function _hermiticity_defect(H::MPO)
    n2 = real(inner(H, H))
    n2 > 0 || return 0.0
    Hd = swapprime(dag(H), 0 => 1)                    # links primed too: harmless for inner
    d2 = 2n2 - 2real(inner(H, Hd))
    return sqrt(max(d2, 0.0) / n2)
end


"""
    get_green_krylov(H_mpo::MPO, sites, ω_phys; η=1e-2, nsweeps=12, maxdim=100,
                     cutoff=1e-8, x0_mpo=nothing, solver=:gmres, ishermitian=false,
                     tol=1e-10, maxiter=600, krylovdim=30, verbose=false) -> MPO

Low-level: compute the retarded Green's function G(ω) = (ω + iη − H)⁻¹ for a
raw MPO `H_mpo` defined on `sites`, via the vectorized linear system

    [I ⊗ (ω + iη − H)ᵀ] |G⟩⟩ = |I⟩⟩

(`G (ω + iη − H) = I`, with the operator on the column sites of |G⟩⟩).

See the `TBHamiltonian` overload for the full keyword-argument reference.
"""
function get_green_krylov(H_mpo::MPO, sites::Vector{<:Index}, ω_phys::Real;
                          η::Real                    = 1e-2,
                          nsweeps::Int               = 12,
                          maxdim::Int                = 100,
                          cutoff::Real               = 1e-8,
                          x0_mpo::Union{MPO,Nothing} = nothing,
                          solver::Symbol             = :gmres,
                          ishermitian::Bool          = false,
                          tol::Real                  = 1e-10,
                          maxiter::Int               = 600,
                          krylovdim::Int             = 30,
                          verbose::Bool              = false)
    solver in (:gmres, :minres) ||
        throw(ArgumentError("get_green_krylov: solver must be :gmres or :minres, got :$solver"))
    solver === :minres &&
        return _green_krylov_minres(H_mpo, sites, ω_phys; η, nsweeps, maxdim, cutoff,
                                    x0_mpo, tol, maxiter, krylovdim, verbose)
    N      = length(sites)
    sites2 = siteinds("Qubit", 2N)

    z     = ComplexF64(ω_phys + im * η)
    ω_mpo = z * MPO(sites, "Id") - H_mpo
    # |G⟩⟩ holds G[r, c] with r on the odd and c on the even qubits (_vec_mps_from_mpo),
    # and Lop acts on c: G(z − H) = I, i.e. (z − H)ᵀ on the column register, so the
    # operator is embedded transposed (interleave_mpo embeds it as given).
    Lop   = interleave_mpo(swapprime(ω_mpo, 0 => 1), sites2, 0)
    rhs   = _vec_mps_from_mpo(MPO(sites, "Id"), sites2)
    # Default start: |I⟩⟩ with its bond bases enlarged by a global Krylov expansion
    # (ITensorMPS.expand, krylovdim = 2: directions Lop^k|I⟩⟩ folded in). From |I⟩⟩
    # itself (bond dimension 1 between the sites of H) the two-site sweeps stalled on
    # real two-register operators such as the bubble's H_eff = I⊗H₂ − H₁⊗I: residual
    # ~0.9 and bonds capped below the solution's whatever nsweeps, a ~50 % error in G.
    # Converged results elsewhere move by ~1e-14.
    x0    = isnothing(x0_mpo) ?
                ITensorMPS.expand(rhs, Lop; alg="global_krylov", krylovdim=2, cutoff=cutoff) :
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


# get_green_krylov with MINRES as the local solver. The operator is I ⊗ Hᵀ, Hermitian
# when H is, and z goes into linsolve's coefficients: (z − I ⊗ Hᵀ)|G⟩⟩ = |I⟩⟩, so no
# z·I − H is formed. The start is |I⟩⟩ expanded as in the GMRES path, here along the
# Krylov space of I ⊗ Hᵀ, which is that of the shifted operator.
function _green_krylov_minres(H_mpo::MPO, sites::Vector{<:Index}, ω_phys::Real;
                              η, nsweeps, maxdim, cutoff, x0_mpo, tol, maxiter,
                              krylovdim, verbose)
    defect = _hermiticity_defect(H_mpo)
    defect <= 1e-6 || throw(ArgumentError(
        "get_green_krylov(solver=:minres) needs a Hermitian H, but ‖H − H†‖/‖H‖ = " *
        "$(round(defect; sigdigits=2)); use solver=:gmres"))
    N      = length(sites)
    sites2 = siteinds("Qubit", 2N)

    z   = ComplexF64(ω_phys + im * η)
    Aop = interleave_mpo(swapprime(H_mpo, 0 => 1), sites2, 0)
    rhs = _vec_mps_from_mpo(MPO(sites, "Id"), sites2)
    x0  = isnothing(x0_mpo) ?
              ITensorMPS.expand(rhs, Aop; alg="global_krylov", krylovdim=2, cutoff=cutoff) :
              _vec_mps_from_mpo(x0_mpo, sites2; cutoff=cutoff, maxdim=maxdim)

    verbose && println("Krylov GF: ω = $ω_phys + $(η)i  (N=$N, maxdim=$maxdim, nsweeps=$nsweeps, MINRES)",
                       isnothing(x0_mpo) ? "" : "  [KPM warm start]")

    stats = [0, 0, 0]
    sol = ITensorMPS.linsolve(Aop, rhs, x0, z, -1.0;
                              nsweeps        = nsweeps,
                              maxdim         = maxdim,
                              cutoff         = cutoff,
                              updater        = _minres_updater,
                              updater_kwargs = (; tol, maxiter = maxiter * krylovdim, stats))
    verbose && println("  MINRES: $(stats[1]) local solves, $(stats[2]) products, ",
                       "$(stats[3]) stopped above tol")
    return custom_mpo(sol, sites)
end


"""
    get_green_krylov(H::TBHamiltonian, ω_phys; η=1e-2, nsweeps=12, maxdim=100,
                     cutoff=1e-8, x0_mpo=nothing, solver=:gmres, ishermitian=false,
                     tol=1e-10, maxiter=600, krylovdim=30, verbose=false) -> MPO

Compute the retarded Green's function

    G(ω) = (ω + iη − H)⁻¹

as an MPO by solving the vectorized linear system

    [I ⊗ (ω + iη − H)ᵀ] |G⟩⟩ = |I⟩⟩

(`G (ω + iη − H) = I`, with the operator on the column sites of |G⟩⟩)

using `ITensorMPS.linsolve` (DMRG-like Krylov solver).  The Hamiltonian is used
unscaled — no KPM Chebyshev expansion required.

**Keyword arguments**
- `η`           : Lorentzian broadening. Default `1e-2`.
- `nsweeps`     : Number of DMRG sweeps for the linear solver. Default `12`.
- `maxdim`      : Maximum bond dimension of the solution MPS. Default `100`.
- `cutoff`      : SVD truncation cutoff. Default `1e-8`.
- `x0_mpo`      : Optional MPO initial guess for G(ω).  When provided it is
                  vectorized via `_vec_mps_from_mpo` and passed as `x0` to
                  `linsolve`, replacing the default guess (the vectorized identity
                  with its bonds enlarged by a global Krylov expansion,
                  `ITensorMPS.expand`).
                  Typical use: pass a low-accuracy KPM Green's function to
                  warm-start the Krylov iteration.  Default `nothing`.
- `solver`      : Local solver of the sweeps. `:gmres` (default): KrylovKit's
                  GMRES, restarted every `krylovdim` products. `:minres`: MINRES,
                  for Hermitian H only (an `ArgumentError` otherwise, from
                  ‖H − H†‖/‖H‖). Each local problem (z − H_eff)x = b is a complex
                  shift of a Hermitian matrix, which MINRES solves by a three-term
                  recurrence without restarts: the same G in fewer products, most
                  at small η, where restarted GMRES slows down. MINRES uses
                  `tol`, and `maxiter · krylovdim` as its product budget per local
                  solve (the GMRES one); `ishermitian` does not apply.
- `ishermitian` : GMRES only. Set `true` only when the shifted operator is Hermitian
                  (requires purely imaginary η = 0, not physical for GF). Default `false`.
- `tol`         : Krylov solver convergence tolerance (absolute, on the local
                  residual). Default `1e-10`.
- `maxiter`     : Maximum Krylov iterations per site (GMRES restarts). Default `600`.
- `krylovdim`   : Krylov subspace dimension. Default `30`.
- `verbose`     : Print progress messages (with MINRES, also the number of local
                  solves and products). Default `false`.

**Usage**
```julia
G = get_green_krylov(H, ω; η=0.05, nsweeps=20, maxdim=200)
dos  = -imag(tr(G)) / π                              # total DoS
ldos = real(inner(psi_i, apply(G, psi_i)))           # LDoS at site i
gij  = inner(psi_i, apply(G, psi_j))                 # off-diagonal element

# Warm-start from a cheap KPM estimate (get_Green_retarded_from_Tn works in the
# rescaled window: ω and η are rescaled, and the result is divided by the scale)
TensorBinding.KPM_Tn(H, 15; maxdim=50)
G_kpm = (1 / H.scale) * TensorBinding.get_Green_retarded_from_Tn(H._tn_cache, 15,
            (ω - H.center) / H.scale; η=η / H.scale, maxdim=50)
G_ws  = get_green_krylov(H, ω; x0_mpo=G_kpm, nsweeps=6, maxdim=200)

# Small η with a Hermitian H: MINRES as the local solver
G_mr  = get_green_krylov(H, ω; η=1e-3, nsweeps=20, maxdim=200, solver=:minres)
```
"""
function get_green_krylov(H::TBHamiltonian, ω_phys::Real;
                          η::Real                    = 1e-2,
                          nsweeps::Int               = 12,
                          maxdim::Int                = 100,
                          cutoff::Real               = 1e-8,
                          x0_mpo::Union{MPO,Nothing} = nothing,
                          solver::Symbol             = :gmres,
                          ishermitian::Bool          = false,
                          tol::Real                  = 1e-10,
                          maxiter::Int               = 600,
                          krylovdim::Int             = 30,
                          verbose::Bool              = false)
    return get_green_krylov(H.mpo, H.sites, ω_phys;
                            η, nsweeps, maxdim, cutoff, x0_mpo, solver, ishermitian,
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
