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


# The residual check holds the environment of ⟨Ax|Ax⟩ times one site: χ²w²d numbers
# (χ, w the bond dimensions of x and A). Above this many it is skipped: 2^27 ComplexF64
# are 2 GiB, and the contraction holds two or three such arrays at a time. (A Ref, so
# that the tests can reach the skipped branch on small systems.)
const _RESIDUAL_MAX_ELEMENTS = Ref(2.0^27)
# Largest bond dimension, (w + 1)χ + χ_b, of the direct sum the exact evaluation
# orthogonalizes.
const _RESIDUAL_EXACT_MAXDIM = 2000
# Rounding of the two evaluations, in units of eps and of the scale
# S = (|a0| + |a1|σ)²‖x‖² + ‖b‖², σ = ‖A‖_F/√D the rms singular value of A: ‖r‖² from the
# inner products is known to ± FLOOR_FACTOR·eps·S (measured: up to 5.8·eps·S), ‖r‖ from
# the direct sum to ± EXACT_FLOOR_FACTOR·eps·√S (measured: up to 37·eps·√S), over
# correction vectors at η = 1e-2…1e-8, Green's functions and Dyson systems of condition
# number up to 1e5 (examples/nontracked/fixes_checks/review_022/residual_floor.jl).
const _RESIDUAL_FLOOR_FACTOR       = 32.0
const _RESIDUAL_EXACT_FLOOR_FACTOR = 64.0

# ⟨Ax|Ax⟩ contracted one tensor at a time (environment · x · A · A† · x†): χ³w²d + χ²w³d²
# per site, where ITensorMPS's inner(A, x, A, x) joins A[j]·x[j] and its conjugate first,
# χ³w³d. The environment holds χ²w² numbers either way. The bra copies get their own
# link indices and their own copy of the site index A acts on.
function _norm2_Ax(A::MPO, x::MPS)
    xb = sim(linkinds, dag(x))
    Ab = sim(linkinds, dag(A))
    E  = ITensor(1.0)
    for j in eachindex(x)
        s  = commonind(A[j], x[j])
        sb = sim(s)
        E  = ((E * x[j]) * A[j]) * replaceind(Ab[j], s, sb)
        E  = E * replaceind(xb[j], s, sb)
    end
    return real(scalar(E))
end

# ‖(a0 + a1·A)x − b‖ read off the orthogonality centre of the direct sum
# a0·x ⊕ a1·Ax ⊕ (−b), with Ax the untruncated product: accurate to ~eps of the largest
# term, where the inner products cancel to ~√eps of it. `nothing` when the sum cannot be
# formed (a bond of x or b without a link index; A may lack some, as interleave_mpo's
# output does) or would exceed _RESIDUAL_EXACT_MAXDIM.
function _linsolve_residual_exact(A::MPO, b::MPS, x::MPS, a0::Number, a1::Number)
    n = length(x)
    n >= 2 || return nothing
    haslinks(ψ) = all(j -> linkind(ψ, j) !== nothing, 1:n-1)
    (haslinks(x) && haslinks(b)) || return nothing
    χ, w = maxlinkdim(x), maxlinkdim(A)
    (iszero(a0) ? w * χ : (w + 1) * χ) + maxlinkdim(b) <= _RESIDUAL_EXACT_MAXDIM ||
        return nothing
    Ax = apply(A, x; alg = "naive", truncate = false)
    r  = iszero(a0) ? +(a1 * Ax, -1 * b; alg = "directsum") :
                      +(a0 * x, a1 * Ax, -1 * b; alg = "directsum")
    orthogonalize!(r, 1)
    return norm(r[1])
end

# The relative residual ‖(a0 + a1·A)x − b‖/‖b‖ of a linsolve result, as
# (; residual, floor, method).
#
# method = :inner: from exact inner products, with no MPO–MPS product,
#     ‖r‖² = |a0|²⟨x|x⟩ + |a1|²⟨Ax|Ax⟩ + ⟨b|b⟩ + 2Re(ā0 a1⟨x|A|x⟩) − 2Re(ā0⟨x|b⟩) − 2Re(a1⟨b|A|x⟩).
#   The terms cancel, and so do the products inside ⟨Ax|Ax⟩ where Ax is small next to
#   ‖A‖‖x‖ (an ill-conditioned A): ‖r‖² is known to ± f² = _RESIDUAL_FLOOR_FACTOR·eps·S
#   only, S = (|a0| + |a1|σ)²‖x‖² + ‖b‖² with σ = ‖A‖_F/√D standing for ‖A‖. That is a
#   rounding floor f/‖b‖ ≈ 1e-7·(|a0| + |a1|σ)‖x‖/‖b‖, which grows with ‖x‖: as 1/η for a
#   resolvent, with a constant energy offset in A, with the condition number of a Dyson
#   solve. `residual` is the upper bound √(max(‖r‖², 0) + f²)/‖b‖, never below the floor:
#   a converged solve reads as its floor, not as 0, and noise cannot pass for a small
#   residual.
# method = :exact: when the floor leaves the comparison with `threshold` open (the
#   threshold lies between the lower and the upper bound), from the direct sum
#   (_linsolve_residual_exact), where it is affordable; `floor` is then ~eps-sized.
# method = :skipped: the contraction would hold more than _RESIDUAL_MAX_ELEMENTS numbers;
#   `residual` and `floor` are NaN.
function _linsolve_residual(A::MPO, b::MPS, x::MPS, a0::Number, a1::Number;
                            threshold::Real = Inf)
    Float64(maxlinkdim(x))^2 * Float64(maxlinkdim(A))^2 * maximum(dim, siteinds(x)) >
        _RESIDUAL_MAX_ELEMENTS[] && return (; residual = NaN, floor = NaN, method = :skipped)
    nb2 = real(inner(b, b))
    nx2 = real(inner(x, x))
    bAx = inner(b', A, x)
    r2  = abs2(a1) * _norm2_Ax(A, x) + nb2 - 2real(a1 * bAx)
    iszero(a0) || (r2 += abs2(a0) * nx2 + 2real(conj(a0) * a1 * inner(x', A, x)) -
                         2real(conj(a0) * inner(x, b)))
    r2  = max(r2, 0.0)
    σ   = sqrt(real(inner(A, A)) / _site_space_dim(siteinds(x)))
    S   = (abs(a0) + abs(a1) * σ)^2 * nx2 + nb2
    f2  = _RESIDUAL_FLOOR_FACTOR * eps(Float64) * S
    if isfinite(threshold) && r2 - f2 <= threshold^2 * nb2 < r2 + f2
        rx = _linsolve_residual_exact(A, b, x, a0, a1)
        rx === nothing ||
            return (; residual = rx / sqrt(nb2), method = :exact,
                      floor = _RESIDUAL_EXACT_FLOOR_FACTOR * eps(Float64) * sqrt(S / nb2))
    end
    return (; residual = sqrt((r2 + f2) / nb2), floor = sqrt(f2 / nb2), method = :inner)
end

# One line, once per session and function, when the residual check was skipped; `then`
# says what the caller does about it.
function _residual_skipped_info(fname::AbstractString, x::MPS, A::MPO, then::AbstractString)
    @info "$fname: the residual check was skipped: at bond dimensions " *
          "$(maxlinkdim(x)) (solution) and $(maxlinkdim(A)) (operator) its contraction " *
          "would hold more than 2^27 numbers (2 GiB) at a time. $then" maxlog=1 _id=Symbol(
          fname, :_residual_skipped)
    return nothing
end


# Start of the linsolve sweeps of get_green_krylov. :random (from v0.2.2): |I⟩⟩ plus
# 0.1·(a random MPS of bond dimension 8 from `rng`), with its bond bases enlarged by an
# 8-vector global Krylov expansion along A. :identity (up to v0.2.1): |I⟩⟩ with a
# 2-vector expansion. The two-site sweeps keep the rank their start spans: from the
# :identity start they stalled on a short-ranged G (the bulk G without the end
# corrections of an open chain; ω outside the band in 2D), wrong by 1e-3–0.2 with no
# sign of it. The random part fills every bond, the expansion adds the directions of A.
#
# The weight of the random part is absolute: the random MPS has norm 1, a row of G a norm
# of order 1, and |I⟩⟩ the norm √D (D the dimension of the site space). At 0.1 it is a
# tenth of one row at every size, below what sets a row apart from the others. Scaled up
# with √D (0.1·√D/64 was tried) it outweighs those rows: the ends of an open chain came
# back as bulk rows from 2^20 sites. It must survive the truncations, though: next to
# ‖|I⟩⟩‖² = D its weight is 0.01/D, above the default cutoff of the sweeps (1e-5/D beyond
# 2^10 sites), but the sum |I⟩⟩ + 0.1·r at cutoff 1e-14 drops it from about 2^33 sites
# (pure |I⟩⟩ at 2^36). Beyond 2^30 sites the sum is therefore a direct sum, which keeps
# it; up to 2^30 it is the truncated sum, as before. From about 2^39 sites 0.01/D is
# below the floor 1e-14 of the cutoff and the expansion drops it anyway: the start is
# then |I⟩⟩, the sweeps stall, and the residual check says so (see _green_krylov_cutoff).
function _green_krylov_start(rhs::MPS, A::MPO, sites2::Vector{<:Index}, start::Symbol,
                             rng::AbstractRNG, cutoff::Real, D::Real)
    start === :identity &&
        return ITensorMPS.expand(rhs, A; alg="global_krylov", krylovdim=2, cutoff=cutoff)
    r  = random_mps(rng, ComplexF64, sites2; linkdims=8)
    x0 = D > 2.0^30 ? +(rhs, 0.1 * r; alg="directsum") : +(rhs, 0.1 * r; cutoff=1e-14)
    return ITensorMPS.expand(x0, A; alg="global_krylov", krylovdim=8, cutoff=cutoff)
end


# The default cutoff of get_green_krylov on a site space of dimension D: 1e-8 up to 2^10
# sites, as before, 1e-8·2^10/D beyond, and never below 1e-14. A cutoff is relative to
# ‖|G⟩⟩‖² ∝ D, so a fixed one weighs D times more next to a single row of G as the system
# grows, and what sets that row apart from the others (a chain end, an impurity) is
# truncated away: at 1e-8 the end rows of a clean open chain came back as bulk rows from
# 2^20 sites (G_11 50 % off) and an impurity site was 2.6 % off from 2^14. Scaled with
# 1/D, the cutoff keeps the weight it has next to one row at 2^10 sites.
#
# The floor 1e-14 is that of double precision: below it the truncations of the start
# (ITensorMPS.expand) cut nothing, and its bond dimensions and memory grew without bound
# at 2^36 sites. It is reached at 2^30 sites; beyond, the cutoff weighs more and more
# next to one row again (a row is 1/D of ‖|G⟩⟩‖²: 1.5e-11 at 2^36, 9e-13 at 2^40), which
# the residual check reports.
#
# Measured at the default settings on open chains, clean and with one impurity (V = 2),
# at ω = 2.3, 0.5 (η = 0.1), −2.05 (η = 0.05) and 1.0 (η = 0.03), against analytic
# elements (examples/nontracked/fixes_checks/green_krylov_sizeaware_cutoff*.jl):
#   2^14–2^30 sites: chain ends, middle bond (both directions) and impurity site to 4e-6
#     or better; ‖G(z − H) − 1‖_F ≤ 4e-2; no warning.
#   2^34, 2^36: the same elements to 1e-4, except the impurity chain at the band edge
#     (0.3–2 %, ‖G(z − H) − 1‖_F = 0.10 and 0.19, with the warning).
#   2^40: the chain ends are bulk rows (‖G(z − H) − 1‖_F = 0.9, with the warning).
# Against the dense inverse at 2^10 and 2^12 sites, ‖G(z − H) − 1‖_F was 1–8 times the
# relative error of the worst row. A looser 1e-8·2^13/D left rows 3 % off and diagonal
# elements 1 % off at the band edge; tighter ones (down to 1e-8·2^3/D) did not lower
# the worst rows reliably.
_green_krylov_cutoff(D::Real) = max(1e-8 * min(1.0, 1024 / Float64(D)), 1e-14)


"""
    get_green_krylov(H_mpo::MPO, sites, ω_phys; η=1e-2, nsweeps=12, maxdim=100,
                     cutoff=nothing, x0_mpo=nothing, start=:random, rng=Xoshiro(1),
                     solver=:gmres, ishermitian=false, tol=1e-10, maxiter=600,
                     krylovdim=30, return_info=false, residual_warn=0.1,
                     verbose=false) -> MPO

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
                          cutoff::Union{Real,Nothing} = nothing,     # _green_krylov_cutoff
                          x0_mpo::Union{MPO,Nothing} = nothing,
                          start::Symbol              = :random,
                          rng::AbstractRNG           = Random.Xoshiro(1),
                          solver::Symbol             = :gmres,
                          ishermitian::Bool          = false,
                          tol::Real                  = 1e-10,
                          maxiter::Int               = 600,
                          krylovdim::Int             = 30,
                          return_info::Bool          = false,
                          residual_warn::Real        = 0.1,
                          verbose::Bool              = false)
    solver in (:gmres, :minres) ||
        throw(ArgumentError("get_green_krylov: solver must be :gmres or :minres, got :$solver"))
    start in (:random, :identity) ||
        throw(ArgumentError("get_green_krylov: start must be :random or :identity, got :$start"))
    D      = _site_space_dim(sites)
    cutoff = something(cutoff, _green_krylov_cutoff(D))
    N      = length(sites)
    sites2 = siteinds("Qubit", 2N)
    z      = ComplexF64(ω_phys + im * η)
    rhs    = _vec_mps_from_mpo(MPO(sites, "Id"), sites2)

    # |G⟩⟩ holds G[r, c] with r on the odd and c on the even qubits (_vec_mps_from_mpo),
    # and the operator acts on c: G(z − H) = I, i.e. (z − H)ᵀ on the column register, so
    # it is embedded transposed (interleave_mpo embeds it as given). linsolve solves
    # (a0 + a1·A)|G⟩⟩ = |I⟩⟩.
    if solver === :minres
        # A = I ⊗ Hᵀ, Hermitian when H is, with z in the coefficients: no z·I − H is formed.
        defect = _hermiticity_defect(H_mpo)
        defect <= 1e-6 || throw(ArgumentError(
            "get_green_krylov(solver=:minres) needs a Hermitian H, but ‖H − H†‖/‖H‖ = " *
            "$(round(defect; sigdigits=2)); use solver=:gmres"))
        A, a0, a1 = interleave_mpo(swapprime(H_mpo, 0 => 1), sites2, 0), z, -1.0
    else
        ω_mpo = z * MPO(sites, "Id") - H_mpo
        A, a0, a1 = interleave_mpo(swapprime(ω_mpo, 0 => 1), sites2, 0), 0.0, 1.0
    end
    x0 = isnothing(x0_mpo) ? _green_krylov_start(rhs, A, sites2, start, rng, cutoff, D) :
                             _vec_mps_from_mpo(x0_mpo, sites2; cutoff=cutoff, maxdim=maxdim)

    verbose && println("Krylov GF: ω = $ω_phys + $(η)i  (N=$N, maxdim=$maxdim, nsweeps=$nsweeps",
                       solver === :minres ? ", MINRES)" : ")",
                       isnothing(x0_mpo) ? "" : "  [KPM warm start]")

    if solver === :minres
        stats = [0, 0, 0]
        sol = ITensorMPS.linsolve(A, rhs, x0, a0, a1;
                                  nsweeps        = nsweeps,
                                  maxdim         = maxdim,
                                  cutoff         = cutoff,
                                  updater        = _minres_updater,
                                  updater_kwargs = (; tol, maxiter = maxiter * krylovdim, stats))
        verbose && println("  MINRES: $(stats[1]) local solves, $(stats[2]) products, ",
                           "$(stats[3]) stopped above tol")
    else
        sol = ITensorMPS.linsolve(A, rhs, x0;
                                  nsweeps        = nsweeps,
                                  maxdim         = maxdim,
                                  cutoff         = cutoff,
                                  updater_kwargs = (; ishermitian, tol, maxiter, krylovdim))
    end
    G = custom_mpo(sol, sites)
    (return_info || isfinite(residual_warn)) || return G

    # ‖|I⟩⟩‖ = √D: the relative residual of the solve is the root mean square of the D row
    # residuals ‖G_r(z − H) − e_r‖, so a wrong row costs it 1/√D and it cannot show one on
    # a large system (a chain end returned as a bulk row: 8e-4 at 2^20 sites). The warning
    # is on the absolute residual ‖G(z − H) − 1‖_F = √D × the relative one, which bounds
    # the residual of every row.
    nrhs = sqrt(D)
    info = _linsolve_residual(A, rhs, sol, a0, a1; threshold = residual_warn / nrhs)
    res, absres = info.residual, info.residual * nrhs
    if info.method === :skipped
        isfinite(residual_warn) &&
            _residual_skipped_info("get_green_krylov", sol, A,
                "G is returned unchecked; residual_warn = Inf silences this.")
    else
        verbose && println("  residual ‖G(z − H) − 1‖_F = $absres (relative to ‖1‖_F: $res)")
        absres <= residual_warn ||
            @warn "get_green_krylov: G is not converged: ‖G(z − H) − 1‖_F = " *
                  "$(round(absres; sigdigits=2)) > residual_warn = $residual_warn" *
                  (info.floor * nrhs > residual_warn / 2 ?
                   " (rounding floor of this residual: $(round(info.floor * nrhs; sigdigits=2)))" : "") *
                  ". Some rows of G are off by about that much, relative to themselves: " *
                  "lower cutoff, or raise maxdim or nsweeps (cutoff = $cutoff, maxlinkdim " *
                  "reached: $(maxlinkdim(sol)))." ω_phys η
    end
    return return_info ? (; G, residual = res, residual_abs = absres,
                            residual_floor = info.floor, maxlinkdim = maxlinkdim(sol)) : G
end


"""
    get_green_krylov(H::TBHamiltonian, ω_phys; η=1e-2, nsweeps=12, maxdim=100,
                     cutoff=nothing, x0_mpo=nothing, start=:random, rng=Xoshiro(1),
                     solver=:gmres, ishermitian=false, tol=1e-10, maxiter=600,
                     krylovdim=30, return_info=false, residual_warn=0.1,
                     verbose=false) -> MPO

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
- `cutoff`      : SVD truncation cutoff of the sweeps, relative to ‖|G⟩⟩‖², which
                  grows as the dimension D of the site space (the number of sites).
                  Default `nothing`: `1e-8` up to 2^10 sites and `1e-8·2^10/D`
                  beyond (never below `1e-14`), so that it weighs the same next to
                  one row of G at every size. A fixed `1e-8` (the default up to
                  v0.2.1, still available by passing it) truncates away what sets
                  a row apart from the others on a large system: the end rows of
                  a clean open chain came back as bulk rows from 2^20 sites.
                  The cutoff sets the accuracy of G row by row. With the default,
                  on open chains of 2^10–2^30 sites with one impurity, the chain
                  ends, the middle bond and the impurity site were right to 4e-6
                  or better and the rows with the largest error were off by a few
                  1e-3 of their norm (‖G(z − H) − 1‖_F up to 4e-2). Beyond 2^30
                  sites the floor `1e-14` binds and rows lose accuracy again:
                  at 2^34–2^36 the same elements were right to 1e-4, or to
                  0.3–2 % at a band edge, with the warning; from about 2^38
                  sites a single row is below double precision next to
                  ‖|G⟩⟩‖², and the ends of a chain come back as bulk rows, with
                  the warning. For local quantities on such systems use the
                  correction vectors (`get_ldos_cv`), which solve for one row.
- `x0_mpo`      : Optional MPO initial guess for G(ω).  When provided it is
                  vectorized via `_vec_mps_from_mpo` and passed as `x0` to
                  `linsolve`, replacing the start chosen by `start`.
                  Typical use: pass a low-accuracy KPM Green's function to
                  warm-start the Krylov iteration.  Default `nothing`. It is used
                  as given, without the basis expansion of the two starts below:
                  from a low-rank guess (a multiple of the identity) the sweeps
                  stall as they do from `:identity`.
- `start`       : The start of the sweeps when `x0_mpo` is not given.
                  `:random` (default since v0.2.2): |I⟩⟩ + 0.1·(a random MPS of
                  bond dimension 8 and norm 1, drawn from `rng`: a tenth of one
                  row of G at every size), with its bonds enlarged by an
                  8-vector global Krylov expansion (`ITensorMPS.expand`).
                  `:identity`: |I⟩⟩ with a 2-vector expansion, the start up to
                  v0.2.1, which stalls on a short-ranged G (an open chain's end
                  corrections, ω outside the band) and returns a G wrong by
                  1e-3–0.2. The sweeps keep the rank their start spans.
- `rng`         : Random number generator of the `:random` start. Default
                  `Random.Xoshiro(1)`, created anew at each call, so the result
                  does not depend on the global RNG state.
- `solver`      : Local solver of the sweeps. `:gmres` (default): KrylovKit's
                  GMRES, restarted every `krylovdim` products. `:minres`: MINRES,
                  for Hermitian H only (an `ArgumentError` otherwise, from
                  ‖H − H†‖/‖H‖ > 1e-6 in the Frobenius norm, which a non-Hermitian
                  term on a single site passes beyond ~2^40 sites; the residual
                  still shows a wrong G). Each local problem (z − H_eff)x = b is a complex
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
- `return_info` : Return `(; G, residual, residual_abs, residual_floor, maxlinkdim)`
                  instead of the MPO `G`: `residual_abs` = ‖G(z − H) − 1‖_F, and
                  `residual` = `residual_abs`/√D, the residual of the solve
                  relative to ‖|I⟩⟩‖ = √D. Default `false`.
- `residual_warn`: Warn when the residual ‖G(z − H) − 1‖_F exceeds this value: G is
                  then not converged (`cutoff`, `maxdim` or `nsweeps` too coarse).
                  This residual bounds that of every row, ‖G_r(z − H) − e_r‖, and
                  on open chains with an impurity it was 1–8 times the relative
                  error of the worst row of G: the default `0.1` flags rows that
                  are off by a few % or more (the bulk rows returned for the ends
                  of a long chain give 0.9), where the default settings leave up
                  to 4e-2. The relative residual cannot do that: it is the root
                  mean square over the D rows, 8e-4 for those wrong end rows at
                  2^20 sites. The residual comes from inner products whose terms
                  cancel: it is an upper bound that includes the rounding floor
                  of that evaluation (`residual_floor`, a few 1e-7 relative, i.e.
                  times √D for the absolute one), so a converged solve reads as
                  its floor, not 0; beyond ~2^32 sites, where that floor nears
                  the threshold, the residual is evaluated exactly instead. The
                  check costs about one sweep or less, and is skipped with a
                  message when its contraction would hold more than 2^27 numbers
                  (the residuals are then `NaN`). `Inf` skips its computation
                  unless `return_info=true`. Default `0.1`.
- `verbose`     : Print progress messages (with MINRES, also the number of local
                  solves and products) and the residual. Default `false`.

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
                          cutoff::Union{Real,Nothing} = nothing,
                          x0_mpo::Union{MPO,Nothing} = nothing,
                          start::Symbol              = :random,
                          rng::AbstractRNG           = Random.Xoshiro(1),
                          solver::Symbol             = :gmres,
                          ishermitian::Bool          = false,
                          tol::Real                  = 1e-10,
                          maxiter::Int               = 600,
                          krylovdim::Int             = 30,
                          return_info::Bool          = false,
                          residual_warn::Real        = 0.1,
                          verbose::Bool              = false)
    return get_green_krylov(H.mpo, H.sites, ω_phys;
                            η, nsweeps, maxdim, cutoff, x0_mpo, start, rng, solver,
                            ishermitian, tol, maxiter, krylovdim, return_info,
                            residual_warn, verbose)
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
