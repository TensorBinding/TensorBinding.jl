# solvers/CorrectionVector.jl — correction vectors of a probe state
#
# Contents: the correction vector x = (ω + iη − H)⁻¹|b⟩ of one probe |b⟩ and its
# projection G = ⟨b|x⟩ (get_correction_vector), by ITensorMPS.linsolve with three
# safeguards: a Krylov seed, sweeps until G stops moving, and a check of the residual
# with residual-expansion rounds; scans over ω with warm starts and the LDOS built on
# them (get_ldos_cv for a site, get_exciton_ldos_cv for the exciton contact |X, X⟩); and
# the refinement of an isolated peak into its energy and weight by Rayleigh-quotient
# iteration on the correction vector (refine_bound_state).
#
# Entry points: get_correction_vector, get_ldos_cv, get_exciton_ldos_cv,
#   refine_bound_state
# Depends on: solvers/Krylov.jl (_minres_updater, _linsolve_residual,
#   _hermiticity_defect), core/AuxDOF.jl (probe_state, _aux_projection,
#   _probe_sectors), physics/TwoParticle.jl (mpsexciton), core/TBSystem.jl.


# ============================================================
# 1. One frequency
# ============================================================

_cv_mpo(H::TBHamiltonian) = H.mpo
_cv_mpo(H::MPO) = H

# b + Σₖ Hᵏb/‖Hᵏb‖ (k = 1…K), compressed to `maxdim`. From the probe alone, a product
# state, the two-site sweeps stall on interleaved two-particle registers: a hop flips
# carry bits two sites apart, outside the tangent space of the probe (residual ~1–30,
# χ ≈ 7, G wrong). The seed spans those directions.
function _cv_krylov_seed(Hm::MPO, b::MPS, K::Int; maxdim::Int, cutoff::Real)
    acc = (1.0 + 0.0im) * b
    v   = b
    for _ in 1:K
        v  = apply(Hm, v; maxdim, cutoff)
        nv = norm(v)
        nv > 0 || break                               # Hᵏb = 0 (a decoupled site): nothing to add
        v   = v / nv
        acc = +(acc, v; maxdim, cutoff)
    end
    return acc
end

# `nsweeps` sweeps of linsolve on (z − H)x = b, from x.
function _cv_sweeps(Hm::MPO, b::MPS, x::MPS, z::Complex; nsweeps::Int, solver::Symbol,
                    maxdim::Int, cutoff::Real, tol::Real, krylovdim::Int, maxiter::Int)
    if solver === :minres
        return ITensorMPS.linsolve(Hm, b, x, z, -1.0; nsweeps, maxdim, cutoff,
                                   outputlevel = 0, updater = _minres_updater,
                                   updater_kwargs = (; tol, maxiter = maxiter * krylovdim))
    end
    return ITensorMPS.linsolve(Hm, b, x, z, -1.0; nsweeps, maxdim, cutoff, outputlevel = 0,
                               updater_kwargs = (; ishermitian = false, tol, krylovdim,
                                                   maxiter, verbosity = 0))
end

# One frequency from the start x: sweeps one at a time until |ΔG| ≤ gtol·|G| (at least
# 2, at most nsweeps), then the residual; above rtol, up to `rounds` residual expansions
# x ← x + 0.3‖x‖/‖r‖·r, r = (z − H)x − b (as an MPS), with 3 sweeps each, keeping the
# best x. A two-site sweep can stop on a wrong vector that looks converged in G; the
# residual shows it, and the expansion moves x out of the tangent space it is stuck in.
# The residual is the upper bound of _linsolve_residual (never below its rounding floor,
# exact where the floor would leave the comparison with rtol open); NaN, with
# converged = false, when its contraction is too large to run.
function _cv_point(Hm::MPO, b::MPS, x::MPS, z::Complex; nsweeps::Int, gtol::Real,
                   rtol::Real, rounds::Int, kw...)
    G = complex(NaN)
    n = 0
    while n < nsweeps
        x  = _cv_sweeps(Hm, b, x, z; nsweeps = 1, kw...)
        n += 1
        Gn = inner(b, x)
        (n >= 2 && abs(Gn - G) <= gtol * abs(Gn)) && (G = Gn; break)
        G = Gn
    end
    info = _linsolve_residual(Hm, b, x, z, -1.0; threshold = rtol)
    res  = info.residual
    info.method === :skipped && _residual_skipped_info("get_correction_vector", x, Hm,
        "It is reported as residual = NaN, converged = false.")
    k = 0
    while k < rounds && res > rtol
        k += 1
        md = kw[:maxdim] > typemax(Int) ÷ 2 ? typemax(Int) : 2 * kw[:maxdim]
        r  = +(z * x, -1.0 * apply(Hm, x; cutoff = kw[:cutoff], maxdim = md), -1.0 * b;
               cutoff = kw[:cutoff], maxdim = md)
        x1 = +(x, (0.3 * norm(x) / norm(r)) * r; cutoff = kw[:cutoff], maxdim = md)
        x1 = _cv_sweeps(Hm, b, x1, z; nsweeps = 3, kw...)
        n += 3
        i1 = _linsolve_residual(Hm, b, x1, z, -1.0; threshold = rtol)
        i1.residual < res && ((x, info, res) = (x1, i1, i1.residual))
    end
    return (; G = inner(b, x), x, residual = res, residual_floor = info.floor,
              converged = res <= rtol, maxlinkdim = maxlinkdim(x), sweeps = n, rounds = k)
end

function _cv_check_solver(Hm::MPO, solver::Symbol, fname::String)
    solver in (:gmres, :minres) ||
        throw(ArgumentError("$fname: solver must be :gmres or :minres, got :$solver"))
    if solver === :minres
        defect = _hermiticity_defect(Hm)
        defect <= 1e-6 || throw(ArgumentError(
            "$fname(solver=:minres) needs a Hermitian H, but ‖H − H†‖/‖H‖ = " *
            "$(round(defect; sigdigits=2)); use solver=:gmres"))
    end
    return nothing
end

"""
    get_correction_vector(H, b::MPS, ω; eta=1e-2, x0=nothing, eta_path=(),
                          solver=:gmres, maxdim=400, cutoff=1e-12, nsweeps=12,
                          gtol=1e-6, rtol=0.02, rounds=3, seed_krylov=40,
                          seed_maxdim=64, tol=1e-10, krylovdim=30, maxiter=30)
        -> (; G, x, residual, residual_floor, converged, maxlinkdim, sweeps, rounds)

The correction vector `x = (ω + iη − H)⁻¹|b⟩` of the probe `|b⟩` and
`G = ⟨b|x⟩ = ⟨b|(ω + iη − H)⁻¹|b⟩`, the resolvent projected on the probe, with η =
`eta`. `H` is a `TBHamiltonian` or an MPO on the sites of `b`; the energies are
physical (no KPM rescaling). The local density of the probe is `−imag(G)/π`, per
unit physical energy for a normalised `b`.

For one probe a correction vector costs far less than the full Green's function
(`get_green_krylov`), and it resolves what KPM needs many moments for: an isolated
peak at small η, a gap in a band. Its bond dimension stays small where `x` is
dominated by a bound state, and grows where `ω` lies in a continuum: there a KPM run
that serves every ω is usually cheaper.

**Algorithm.** `ITensorMPS.linsolve` sweeps on `(ω + iη − H)x = b`, one at a time,
until `|ΔG| ≤ gtol·|G|` (at least 2, at most `nsweeps`), from
- `x0` when given (a warm start, e.g. the `x` of a neighbouring ω);
- otherwise the Krylov seed `b + Σₖ Hᵏb/‖Hᵏb‖`, `k = 1…seed_krylov`, compressed to
  `seed_maxdim`. From the probe alone, a product state, the two-site sweeps stall on
  two-particle (interleaved) registers and return a wrong `x` that looks converged.

Then the relative residual `‖(ω + iη − H)x − b‖/‖b‖`. Above `rtol`, up to `rounds`
residual expansions `x ← x + 0.3‖x‖/‖r‖·r` with 3 sweeps each, keeping the best `x`.
`converged = residual ≤ rtol`: when `false`, `G` is not to be trusted (raise `maxdim`,
or warm-start).

The residual comes from inner products whose terms cancel to about
`1e-7·(|ω| + ‖H‖)·‖x‖/‖b‖`, a rounding floor that grows as 1/η (and with a constant
energy offset in `H`). `residual` is an upper bound that includes this floor
(`residual_floor`); where the floor would leave the comparison with `rtol` open
(η ≲ 1e-5 on a level, or an offset of 10³ at η = 1e-4) the residual is evaluated
exactly instead, from the direct sum of its terms. The truncation sets a second
floor, about `√cutoff·‖x‖/‖b‖`: at small η, `rtol` is reachable only with
`cutoff ≲ (rtol·η)²`. Where the contraction of the check would hold more than 2^27
numbers it is skipped with a message: `residual = NaN`, `converged = false`.

`eta_path`: broadenings solved first, in order, each warm-starting the next and the
last one `eta` (continuation in η at fixed ω). At small η a cold start can fail where
a path from a larger η converges, provided its first step converges.

**Keywords**
- `solver`: local solver of the sweeps, `:gmres` (KrylovKit) or `:minres` (Hermitian
  H only, see `get_green_krylov`).
- `maxdim`, `cutoff`: truncation of `x`. The default cutoff `1e-12` is tighter than
  elsewhere: the bound-state part of `x` grows as 1/η and the rest is small next to it.
- `tol`, `krylovdim`, `maxiter`: of the local solver (GMRES restarts `maxiter`; MINRES
  takes `maxiter·krylovdim` products).

Returns `x` and `G`, the residual and its rounding floor, `converged`, the bond
dimension of `x`, and the numbers of sweeps and expansion rounds.

```julia
b = TensorBinding.probe_state(H, 17)                # site 17
r = get_correction_vector(H, b, 0.3; eta=1e-3)
r.converged || @warn "not converged" r.residual
ρ = -imag(r.G) / π
```
"""
function get_correction_vector(H::Union{TBHamiltonian,MPO}, b::MPS, ω::Real;
                               eta::Real           = 1e-2,
                               x0::Union{MPS,Nothing} = nothing,
                               eta_path            = (),
                               solver::Symbol      = :gmres,
                               maxdim::Int         = 400,
                               cutoff::Real        = 1e-12,
                               nsweeps::Int        = 12,
                               gtol::Real          = 1e-6,
                               rtol::Real          = 0.02,
                               rounds::Int         = 3,
                               seed_krylov::Int    = 40,
                               seed_maxdim::Int    = 64,
                               tol::Real           = 1e-10,
                               krylovdim::Int      = 30,
                               maxiter::Int        = 30)
    Hm = _cv_mpo(H)
    _cv_check_solver(Hm, solver, "get_correction_vector")
    eta > 0 || throw(ArgumentError("get_correction_vector: eta must be positive, got $eta"))
    x = x0 === nothing ? _cv_krylov_seed(Hm, b, seed_krylov; maxdim = seed_maxdim, cutoff) : x0
    kw = (; nsweeps, gtol, rtol, rounds, solver, maxdim, cutoff, tol, krylovdim, maxiter)
    sweeps = 0
    for e in eta_path
        r = _cv_point(Hm, b, x, ComplexF64(ω + im * e); kw...)
        x = r.x
        sweeps += r.sweeps
    end
    r = _cv_point(Hm, b, x, ComplexF64(ω + im * eta); kw...)
    return merge(r, (; sweeps = r.sweeps + sweeps))
end


# ============================================================
# 2. Scans: site and exciton LDOS
# ============================================================

# Correction vectors of one probe along ω_vals, each warm-started from the previous x.
function _cv_scan(Hm::MPO, b::MPS, ω_vals, eta::Real; kwargs...)
    n   = length(ω_vals)
    G   = zeros(ComplexF64, n)
    res = zeros(Float64, n)
    χ   = zeros(Int, n)
    x   = nothing
    for (i, ω) in enumerate(ω_vals)
        r = get_correction_vector(Hm, b, ω; eta, x0 = x, kwargs...)
        x = r.x
        G[i], res[i], χ[i] = r.G, r.residual, r.maxlinkdim
    end
    return G, res, χ
end

function _cv_ldos_result(Gs, ress, χs, rtol, fname)
    G   = sum(Gs)
    res = reduce((a, b) -> max.(a, b), ress)
    χ   = reduce((a, b) -> max.(a, b), χs)
    bad = count(r -> !(r <= rtol), res)               # a NaN residual counts
    bad > 0 && @warn "$fname: $bad of $(length(res)) frequencies not converged " *
                     "(residual > rtol = $rtol): raise maxdim or nsweeps, or use eta_path"
    return (; ldos = -imag.(G) ./ π, G, residual = res, converged = res .<= rtol,
              maxlinkdim = χ)
end

"""
    get_ldos_cv(H::TBHamiltonian, X::Int, ω_vals; eta=1e-2, <projection keywords>,
                <get_correction_vector keywords>)
        -> (; ldos, G, residual, converged, maxlinkdim)

The local density of states at position `X` (1-indexed in `1:H.N`) from correction
vectors: `ldos[i] = −Im⟨X|(ω_i + iη − H)⁻¹|X⟩/π`, a Lorentzian of width `eta` around
each level, per unit **physical** energy (one state integrates to 1). The KPM
functions return the density per unit rescaled energy times `1/π` (Solvers page,
Conventions): `get_ldos_online(…) · π/H.scale` is in the units of `ldos` here.

The frequencies are solved in the given order, each correction vector warm-starting
the next (only the first is seeded), so a fine grid is cheaper per point than a coarse
one. Each point keeps its residual; a warning counts the points above `rtol`.

On an exciton Hamiltonian (two-particle register) the probe is the contact state
|X, X⟩, as in `get_exciton_ldos_cv`. Auxiliary degrees of freedom are summed or
selected with the keywords of `get_ldos_online` (`spin_proj`, `proj_s`, `nambu_proj`,
`proj_nambu`, `layer_proj`, `proj_layer`, `sublat_proj`, `proj_sl`), one scan per
sector. Every other keyword goes to `get_correction_vector`.
"""
function get_ldos_cv(H::TBHamiltonian, X::Int, ω_vals;
                     eta::Real         = 1e-2,
                     rtol::Real        = 0.02,
                     nambu_proj::Bool  = false,
                     proj_nambu        = nothing,
                     spin_proj::Bool   = false,
                     proj_s            = nothing,
                     layer_proj::Bool  = false,
                     proj_layer        = nothing,
                     sublat_proj::Bool = false,
                     proj_sl           = nothing,
                     kwargs...)
    1 <= X <= H.N || throw(BoundsError(1:H.N, X))
    aux = _aux_projection(H; nambu_proj, proj_nambu, spin_proj, proj_s,
                             layer_proj, proj_layer, sublat_proj, proj_sl)
    Gs = Vector{ComplexF64}[]; ress = Vector{Float64}[]; χs = Vector{Int}[]
    for σ in _probe_sectors(aux)
        G, res, χ = _cv_scan(H.mpo, probe_state(H, X, σ), ω_vals, eta; rtol, kwargs...)
        push!(Gs, G); push!(ress, res); push!(χs, χ)
    end
    return _cv_ldos_result(Gs, ress, χs, rtol, "get_ldos_cv")
end

"""
    get_exciton_ldos_cv(H::TBHamiltonian, X::Int, ω_vals; eta=1e-2,
                        <get_correction_vector keywords>)
        -> (; ldos, G, residual, converged, maxlinkdim)

The exciton LDOS at the contact probe |X, X⟩ (electron and hole on position `X`,
1-indexed in `1:H.N`; `mpsexciton`, the probe of `get_exciton_ldos`) from correction
vectors, in the units and with the scan of `get_ldos_cv`: per unit physical energy,
`ldos = −Im G/π`; `get_exciton_ldos(…) · π/H.scale` is in the same units.

The bound-exciton peaks of a contact probe are where correction vectors pay most: `x`
stays at a small bond dimension down to small η, where KPM-MPS needs moments beyond
its truncation floor. In the two-particle continuum `x` spreads and its bond
dimension grows; check `converged`.
"""
function get_exciton_ldos_cv(H::TBHamiltonian, X::Int, ω_vals;
                             eta::Real  = 1e-2,
                             rtol::Real = 0.02,
                             kwargs...)
    1 <= X <= H.N || throw(BoundsError(1:H.N, X))
    length(H.sites) == 2 * H.L ||
        throw(ArgumentError("get_exciton_ldos_cv: H is not an exciton Hamiltonian " *
                            "(2L interleaved electron-hole sites); use get_ldos_cv"))
    G, res, χ = _cv_scan(H.mpo, mpsexciton(X, H.sites), ω_vals, eta; rtol, kwargs...)
    return _cv_ldos_result([G], [res], [χ], rtol, "get_exciton_ldos_cv")
end


# ============================================================
# 3. Isolated peaks: energy and weight
# ============================================================

"""
    refine_bound_state(H, b::MPS, ω0; eta_steps=(0.05, 0.01, 1e-3, 1e-4), x0=nothing,
                       <get_correction_vector keywords>)
        -> (; energy, weight, x, residual, converged, history)

The energy `E` and the probe weight `w = |⟨ψ|b⟩|²/⟨b|b⟩` of an isolated eigenstate
`ψ` near `ω0`, by Rayleigh-quotient iteration on the correction vector: for each η of
`eta_steps`, `x = (ω + iη − H)⁻¹|b⟩` (warm-started), then
`ω ← ⟨x|H|x⟩/⟨x|x⟩` and `w = |⟨b|x⟩|²/(⟨x|x⟩⟨b|b⟩)`. As η → 0 at ω = E, x is
dominated by ψ, so the quotients converge to E and w: the content of an isolated
peak, which a spectrum at finite η only broadens. The other levels enter at order
η²/Δ (Δ the distance to the next level the probe sees), so the last η sets the
accuracy: with the default 1e-4 and Δ of order one, E to ~1e-8.

Start `ω0` within the peak's width at the first η (a coarse scan with `get_ldos_cv`
at η ≈ 0.1–0.2 finds it). A level that is not isolated at the last η (another state
within it) gives a mixed quotient; the history shows it as an energy that keeps
moving. `history` holds `(; eta, omega, energy, weight, residual)` per step.
"""
function refine_bound_state(H::Union{TBHamiltonian,MPO}, b::MPS, ω0::Real;
                            eta_steps = (0.05, 0.01, 1e-3, 1e-4),
                            x0::Union{MPS,Nothing} = nothing,
                            kwargs...)
    Hm = _cv_mpo(H)
    nb = real(inner(b, b))
    ω  = Float64(ω0)
    x  = x0
    history = NamedTuple[]
    r = nothing
    for η in eta_steps
        r  = get_correction_vector(Hm, b, ω; eta = η, x0 = x, kwargs...)
        x  = r.x
        nx = real(inner(x, x))
        E  = real(inner(x', Hm, x)) / nx
        w  = abs2(inner(b, x)) / (nx * nb)
        push!(history, (; eta = η, omega = ω, energy = E, weight = w, residual = r.residual))
        ω = E
    end
    last = history[end]
    return (; energy = last.energy, weight = last.weight, x, residual = last.residual,
              converged = r.converged, history)
end
