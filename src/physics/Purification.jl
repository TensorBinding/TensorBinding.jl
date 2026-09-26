# Purification.jl — density matrix purification methods
#
# These methods iteratively drive the eigenvalues of an approximate
# density matrix toward exactly 0 or 1, approximating the zero-temperature
# step function θ(μ - H) without Chebyshev expansion.
#
# Two algorithms are provided:
#
#   McWeeny purification:  ρ_{n+1} = 3ρ² - 2ρ³
#       Quadratic convergence: ε_{n+1} ≈ 3ε_n² near each fixed point,
#       so accurate digits double each step.  Each step costs two
#       MPO-MPO products.  Requires a reasonable initial guess.
#
#   SP2 (second-order spectral projection):  ρ_{n+1} = ρ²  or  2ρ - ρ²
#       Quadratic convergence.  Each step costs one MPO-MPO product.
#       Direction chosen by comparing Tr(ρ²) to the target electron
#       count N_el, which drives the chemical potential implicitly.
#
# Typical usage:
#   ρ0 = get_density(H; Ncheb=30, method=:kpm)   # cheap rough guess
#   ρ  = mcweeny_purify(ρ0; maxdim=40)
#
# Both functions accept `cutoff` and `maxdim` to control truncation
# at each MPO-MPO multiplication step.
#
# Entry points: get_density, mcweeny_purify, sp2_purify, purification_initial_guess,
#   sign_mpo, get_ldos_drho, get_dos_drho. The dispatcher behind get_density,
#   _density_matrix, also computes the density matrices of the RPA bubbles
#   (_get_density_matrix) and the topological markers (_get_projector).
# Depends on: core/Utils.jl, core/TBSystem.jl, solvers/DMRG.jl, solvers/kpm/recursion.jl,
#   solvers/kpm/cached.jl.

# ============================================================
# 1. Shared helpers
# ============================================================

"""
    _mpo_sq(ρ; maxdim, cutoff, trunc=(:maxdim, :cutoff)) -> MPO

Compute `ρ²` via `apply` and truncate immediately.  The intermediate
bond dimension of `apply` is controlled by `maxdim`; the truncation receives the
parameters named in `trunc` (`(:cutoff,)` in the GPU purifications).
"""
function _mpo_sq(ρ::MPO; maxdim::Int, cutoff::Real, trunc::Tuple = (:maxdim, :cutoff))
    ρ2 = apply(ρ, ρ; maxdim, cutoff)
    ITensorMPS.truncate!(ρ2; _trunc_kwargs(trunc, maxdim, cutoff)...)
    return ρ2
end


"""
    _idempotency_error(ρ, ρ2) -> Float64

Compute ‖ρ² - ρ‖ / ‖ρ‖ as a measure of how far `ρ` is from a
projection.  Returns 0 for an exact density matrix.
"""
function _idempotency_error(ρ::MPO, ρ2::MPO)
    diff = +(ρ2, -1.0 * ρ; cutoff=1e-12)
    n_diff = norm(diff)
    n_rho  = norm(ρ)
    return n_rho > 0 ? n_diff / n_rho : n_diff
end


"""
    _purified_pair(guess, a₊, a₋; maxiters, maxdim, cutoff, tol, verbose) -> (ρ₊, ρ₋)

McWeeny-purify the initial guess `guess(a₊)`, then `guess(a₋)`, each built just
before its purification: the two density matrices whose difference is
`sign_mpo` (a = ±1/scale) and the finite-difference `get_ldos_drho` /
`get_dos_drho` (a = ω ± δμ, the Fermi level of the initial guess).
"""
function _purified_pair(guess, a_p, a_m; maxiters, maxdim, cutoff, tol, verbose)
    ρ_p = mcweeny_purify(guess(a_p); maxiters=maxiters, maxdim=maxdim,
                                     cutoff=cutoff, tol=tol, verbose=verbose)
    ρ_m = mcweeny_purify(guess(a_m); maxiters=maxiters, maxdim=maxdim,
                                     cutoff=cutoff, tol=tol, verbose=verbose)
    return ρ_p, ρ_m
end


# ============================================================
# 2. McWeeny purification
# ============================================================

"""
    mcweeny_purify(ρ0; maxiters=30, maxdim=40, cutoff=1e-8,
                  tol=1e-5, verbose=false) -> MPO

Iterate the McWeeny map  ρ_{n+1} = 3ρ_n² - 2ρ_n³  until the
idempotency residual ‖ρ² - ρ‖/‖ρ‖ < `tol` or `maxiters` is reached.


# Arguments
- `ρ0`       : initial density matrix MPO (need not be idempotent)
- `maxiters` : maximum number of iterations
- `maxdim`   : maximum MPO bond dimension during multiplication
- `cutoff`   : cutoff during truncation
- `tol`      : convergence threshold on ‖ρ²−ρ‖/‖ρ‖
- `verbose`  : print the residual every 15 iterations

# Returns
Purified density matrix MPO.
"""
function mcweeny_purify(ρ0::MPO;
                        maxiters::Int   = 30,
                        maxdim::Int     = 40,
                        cutoff::Float64 = 1e-8,
                        tol::Float64    = 1e-5,
                        verbose::Bool   = false)
    progress = verbose ? function (iter, err, ρ)
        iter % 15 == 0 &&
            println("McWeeny iter $iter: ‖ρ²-ρ‖/‖ρ‖ = $err, maxlinkdim = $(ITensorMPS.maxlinkdim(ρ))")
    end : nothing
    return _mcweeny_iterate(deepcopy(ρ0); maxiters, maxdim, cutoff, tol, progress)
end

"""
    _mcweeny_iterate(ρ; maxiters, maxdim, cutoff, tol, trunc=(:maxdim, :cutoff),
                     progress=nothing, after_step=nothing) -> MPO

The McWeeny loop of `mcweeny_purify`, also run on GPU MPOs by `get_C_gpu` and the
GPU purifications of gpu/purification.jl. Each iteration: `ρ² = apply(ρ, ρ; maxdim,
cutoff)`, `truncate!(ρ²; <trunc>)`, the residual ‖ρ² − ρ‖/‖ρ‖, `progress(iter, err, ρ)`,
a stop when it is below `tol`, then `ρ ← apply(ρ, +(3ρ, −2ρ²; cutoff); maxdim, cutoff)`,
`truncate!(ρ; <trunc>)` and `after_step()`. `<trunc>` passes the parameters named in
`trunc` (see `_trunc_kwargs`): both on the CPU, `(:cutoff,)` on the GPU. `ρ` is not
copied.
"""
function _mcweeny_iterate(ρ::MPO; maxiters::Int, maxdim::Int, cutoff::Real, tol::Real,
                          trunc::Tuple = (:maxdim, :cutoff),
                          progress = nothing, after_step = nothing)
    trunc_kwargs = _trunc_kwargs(trunc, maxdim, cutoff)
    for iter in 1:maxiters
        ρ2  = _mpo_sq(ρ; maxdim, cutoff, trunc)
        err = _idempotency_error(ρ, ρ2)
        progress === nothing || progress(iter, err, ρ)
        err < tol && break
        # 3ρ² - 2ρ³ = ρ·(3ρ - 2ρ²)
        ρ_inte = +(3.0 * ρ , -2.0 * ρ2; cutoff)
        ρ  = apply(ρ, ρ_inte; maxdim, cutoff)
        ITensorMPS.truncate!(ρ; trunc_kwargs...)
        after_step === nothing || after_step()
    end
    return ρ
end


# ============================================================
# 3. SP2 purification
# ============================================================

"""
    sp2_purify(ρ0, Nel; maxiters=40, maxdim=40, cutoff=1e-8,
               tol=1e-5, verbose=false) -> MPO

Iterate the SP2 map until convergence:

    if Tr(ρ_n²) ≥ N_el:   ρ_{n+1} = ρ_n²          (contract)
    else:                  ρ_{n+1} = 2ρ_n - ρ_n²   (expand)

Each step costs one MPO-MPO product.  The direction rule drives
Tr(ρ) toward `Nel` and simultaneously pushes eigenvalues to 0 or 1.
Convergence is quadratic.

The spectrum of `ρ0` must lie in `[0, 1]`; normalise with
`ρ0 = (Id - H/scale) / 2` if starting from scratch.

# Arguments
- `ρ0`    : initial density matrix MPO with spectrum ⊆ [0,1]
- `Nel`   : target electron number (Tr of the converged projector)
- remaining kwargs: same as `mcweeny_purify`
"""
function sp2_purify(ρ0::MPO, Nel::Real;
                    maxiters::Int   = 40,
                    maxdim::Int     = 40,
                    cutoff::Float64 = 1e-8,
                    tol::Float64    = 1e-5,
                    verbose::Bool   = false)
    progress = verbose ? function (iter, err, ρ)
        println("SP2 iter $iter: ‖ρ²-ρ‖/‖ρ‖ = $err, maxlinkdim = $(ITensorMPS.maxlinkdim(ρ))")
    end : nothing
    return _sp2_iterate(deepcopy(ρ0), Nel; maxiters, maxdim, cutoff, tol, progress)
end

"""
    _sp2_iterate(ρ, Nel; maxiters, maxdim, cutoff, tol, trunc=(:maxdim, :cutoff),
                 add_trunc=(:cutoff,), progress=nothing, after_step=nothing) -> MPO

The SP2 loop of `sp2_purify`, also run on GPU MPOs by `get_C_gpu`. Each iteration:
`ρ² = apply(ρ, ρ; maxdim, cutoff)`, `truncate!(ρ²; <trunc>)`, the residual,
`progress(iter, err, ρ)`, a stop below `tol`, then `ρ ← ρ²` when Tr ρ² ≥ `Nel`, else
`ρ ← +(2ρ, −ρ²; <add_trunc>)` and `truncate!(ρ; <trunc>)`; `after_step()` ends every
iteration that did not stop. The CPU truncates with both parameters and sums with
`cutoff`; the GPU truncates with `cutoff` and sums with both. `ρ` is not copied.
"""
function _sp2_iterate(ρ::MPO, Nel::Real; maxiters::Int, maxdim::Int, cutoff::Real,
                      tol::Real, trunc::Tuple = (:maxdim, :cutoff),
                      add_trunc::Tuple = (:cutoff,), progress = nothing,
                      after_step = nothing)
    trunc_kwargs = _trunc_kwargs(trunc, maxdim, cutoff)
    add_kwargs   = _trunc_kwargs(add_trunc, maxdim, cutoff)
    for iter in 1:maxiters
        ρ2  = _mpo_sq(ρ; maxdim, cutoff, trunc)
        err = _idempotency_error(ρ, ρ2)
        progress === nothing || progress(iter, err, ρ)
        err < tol && break
        tr_ρ2 = real(tr(ρ2))
        if tr_ρ2 >= Nel
            # contract toward 0: keep ρ²
            ρ = ρ2
        else
            # expand toward 1: 2ρ - ρ²
            ρ = +(2.0 * ρ, -1.0 * ρ2; add_kwargs...)
            ITensorMPS.truncate!(ρ; trunc_kwargs...)
        end
        after_step === nothing || after_step()
    end
    return ρ
end


# ============================================================
# 4. Initial guess (Id - H/scale)/2 and the TBHamiltonian overloads
# ============================================================

"""
    purification_initial_guess(H_mpo, scale, sites; maxdim=40, cutoff=1e-8) -> MPO
    purification_initial_guess(H::TBHamiltonian; ϵF=0.0, maxdim=40, cutoff=1e-8) -> MPO

Construct the simplest valid initial guess for purification:

    ρ₀ = (I - (H − center·I)/scale) / 2

This maps the rescaled spectrum ∈ [-1, 1] to ρ₀ eigenvalues ∈ [0, 1],
the required input range for both `mcweeny_purify` and `sp2_purify`.

The `TBHamiltonian` overload calls `_ensure_scale!` automatically and
accounts for a non-zero spectral center; `ϵF` shifts the level of the guess,
ρ₀ = (I − (H − (center + ϵF)·I)/scale) / 2.
"""
function purification_initial_guess(H_mpo::MPO, scale::Float64, sites; maxdim::Int= 40,
                                    cutoff::Float64 = 1e-8)
    Id  = MPO(sites, "Id")
    ρ0  = +(0.5 * Id, (-0.5 / scale) * H_mpo; cutoff)
    ITensorMPS.truncate!(ρ0; maxdim=maxdim, cutoff)
    return ρ0
end

function purification_initial_guess(H::TBHamiltonian; ϵF::Real=0.0,
                                    maxdim::Int=40, cutoff::Float64=1e-8)
    _require_binary_position_space(H, "purification_initial_guess")
    _ensure_scale!(H)
    return _linear_density_guess(H.mpo, MPO(H.sites, "Id"); ϵF=ϵF, center=H.center,
                                 scale=H.scale, maxdim=maxdim, cutoff=cutoff)
end

# ρ₀ = (1/2 + (ϵF + center)/(2 scale))·Id − H/(2 scale), summed with `cutoff` and
# truncated with `maxdim` and `cutoff`: the guess of the TBHamiltonian method above,
# also formed from GPU MPOs by _purification_initial_guess_gpu (gpu/purification.jl).
function _linear_density_guess(H_mpo::MPO, Id::MPO; ϵF::Real, center::Real, scale::Real,
                               maxdim::Int, cutoff::Real)
    coeff_I  = 0.5 + (ϵF + center) / (2 * scale)
    coeff_H  = -0.5 / scale
    ρ0       = +(coeff_I * Id, coeff_H * H_mpo; cutoff)
    ITensorMPS.truncate!(ρ0; maxdim=maxdim, cutoff)
    return ρ0
end


"""
    mcweeny_purify(H::TBHamiltonian; ϵF=0.0, maxiters=30, maxdim=40, cutoff=1e-8,
                   tol=1e-5, verbose=false) -> MPO

High-level overload: builds the initial guess from `H`, runs McWeeny purification,
caches the result in `H._density_cache`, and returns the purified density matrix.

`ϵF` shifts the Fermi level of the initial guess ρ₀ = (I − (H − (center + ϵF)·I)/scale) / 2
(see `purification_initial_guess`), allowing purification to target a band other
than half-filling. Default `0.0`.
"""
function mcweeny_purify(H::TBHamiltonian;
                        ϵF::Real        = 0.0,
                        maxiters::Int   = 30,
                        maxdim::Int     = 40,
                        cutoff::Float64 = 1e-8,
                        tol::Float64    = 1e-5,
                        verbose::Bool   = false)
    _require_binary_position_space(H, "mcweeny_purify")
    ρ0 = purification_initial_guess(H; ϵF=ϵF, maxdim=maxdim, cutoff=cutoff)
    ρ  = mcweeny_purify(ρ0; maxiters=maxiters, maxdim=maxdim, cutoff=cutoff,
                            tol=tol, verbose=verbose)
    H._density_cache = ρ
    return ρ
end


"""
    sp2_purify(H::TBHamiltonian; Nel=H.N ÷ 2, maxiters=40, maxdim=40, cutoff=1e-8,
               tol=1e-5, verbose=false) -> MPO

High-level overload: builds the initial guess from `H`, runs SP2 purification,
caches the result in `H._density_cache`, and returns the purified density matrix.
`Nel` defaults to half-filling (`H.N ÷ 2`).
"""
function sp2_purify(H::TBHamiltonian;
                    Nel::Int        = H.N ÷ 2,
                    maxiters::Int   = 40,
                    maxdim::Int     = 40,
                    cutoff::Float64 = 1e-8,
                    tol::Float64    = 1e-5,
                    verbose::Bool   = false)
    _require_binary_position_space(H, "sp2_purify")
    ρ0 = purification_initial_guess(H; maxdim=maxdim, cutoff=cutoff)
    ρ  = sp2_purify(ρ0, Nel; maxiters=maxiters, maxdim=maxdim, cutoff=cutoff,
                              tol=tol, verbose=verbose)
    H._density_cache = ρ
    return ρ
end


# ============================================================
# 5. Unified high-level density-matrix wrapper
# ============================================================

"""
    get_density(H::TBHamiltonian; method=:mcweeny, ϵF=0.0, Ncheb=150, kernel=:jackson,
                lambda=4.0, maxdim=40, cutoff=1e-8, Nel=H.N ÷ 2, maxiters=30,
                tol=1e-5, verbose=false) -> MPO

Compute and cache the zero-temperature density matrix P = θ(ϵF − H).

If `H._density_cache` is already populated it is returned immediately.
Set `H._density_cache = nothing` to force a fresh computation.

**method**
- `:mcweeny` (default) — McWeeny purification P_{n+1} = 3P_n² − 2P_n³
- `:sp2`               — SP2 purification (1 MPO product/step), requires `Nel`
- `:kpm`               — KPM Chebyshev expansion of the Fermi step function

**Keyword arguments**
- `ϵF`      : Fermi energy in physical units (`:kpm` and `:mcweeny`; `:sp2` fixes
  the filling through `Nel` instead). Default `0.0`.
- `Ncheb`   : Chebyshev order (`:kpm` only). Default `150`.
  Reuses `H._tn_cache` if already built at order ≥ `Ncheb`; otherwise calls
  `KPM_Tn` to build and cache it.
- `kernel`  : KPM kernel — `:jackson` (default). HODC is not meaningful for the
  step function so only convolution kernels are supported.
- `lambda`  : Jackson kernel damping parameter. Default `4.0`.
- `maxdim`  : Maximum bond dimension. Default `40`.
- `cutoff`  : SVD truncation cutoff. Default `1e-8`.
- `Nel`     : Target electron count (`:sp2` only). Default `H.N ÷ 2`.
- `maxiters`: Maximum purification iterations. Default `30`.
- `tol`     : Idempotency convergence tolerance (purification). Default `1e-5`.
- `verbose` : Print iteration progress. Default `false`.
"""
function get_density(H::TBHamiltonian;
                     method::Symbol   = :mcweeny,
                     ϵF::Real         = 0.0,
                     Ncheb::Int       = 150,
                     kernel::Symbol   = :jackson,
                     lambda::Real     = 4.0,
                     maxdim::Int      = 40,
                     cutoff::Float64  = 1e-8,
                     Nel::Int         = H.N ÷ 2,
                     maxiters::Int    = 30,
                     tol::Float64     = 1e-5,
                     verbose::Bool    = false)

    method === :kpm || _require_binary_position_space(H, "get_density(method=:$method)")

    if H._density_cache !== nothing
        verbose && println("get_density: returning cached density matrix")
        return H._density_cache
    end

    return _density_matrix(H, method; ϵF=ϵF, Ncheb=Ncheb, kernel=kernel, lambda=lambda,
                           maxdim=maxdim, cutoff=cutoff, Nel=Nel, maxiters=maxiters,
                           tol=tol, verbose=verbose)
end


"""
    _density_matrix(H, method; ϵF=0.0, Ncheb=150, kernel=:jackson, lambda=4.0,
                    maxdim=40, cutoff=1e-8, Nel=H.N ÷ 2, maxiters=30, tol=1e-5,
                    verbose=false, Tn=nothing, store=true) -> MPO

The density-matrix dispatcher behind `get_density`, which checks the position
space and the density cache first; the defaults are `get_density`'s. RPA's
`_get_density_matrix` (physics/rpa/bubble.jl) and Topology's `_get_projector`
(physics/Topology.jl) call it too, after translating their own method symbols,
cache rules and defaults (see each).

- `:mcweeny` / `:sp2`: `mcweeny_purify(H; ϵF, …)` / `sp2_purify(H; Nel, …)`, which
  store the result in `H._density_cache`.
- `:kpm`: `get_density_from_Tn` on the Chebyshev list `Tn = (Tn_list, N)`. With
  `Tn = nothing` that is `H._tn_cache`, built by `KPM_Tn(H, Ncheb; …)` when it is
  absent or shorter than `Ncheb`. The result is stored in `H._density_cache`
  unless `store=false`. The expansion is that of `get_density_from_Tn`, the
  occupied-state projector θ(μ − x) (it was θ(x − μ) until v0.1.1).
"""
function _density_matrix(H::TBHamiltonian, method::Symbol;
                         ϵF       = 0.0,
                         Ncheb    = 150,
                         kernel   = :jackson,
                         lambda   = 4.0,
                         maxdim   = 40,
                         cutoff   = 1e-8,
                         Nel      = H.N ÷ 2,
                         maxiters = 30,
                         tol      = 1e-5,
                         verbose  = false,
                         Tn       = nothing,
                         store    = true)
    if method == :mcweeny
        return mcweeny_purify(H; ϵF=ϵF, maxiters=maxiters, maxdim=maxdim, cutoff=cutoff,
                                 tol=tol, verbose=verbose)
    elseif method == :sp2
        return sp2_purify(H; Nel=Nel, maxiters=maxiters, maxdim=maxdim, cutoff=cutoff,
                             tol=tol, verbose=verbose)
    elseif method == :kpm
        if Tn === nothing
            if H._tn_cache === nothing || H._tn_Ncheb < Ncheb
                KPM_Tn(H, Ncheb; maxdim=maxdim, cutoff=cutoff, verbose=verbose)
            end
            Tn = (H._tn_cache, H._tn_Ncheb)
        end
        Tn_list, N = Tn
        fermi_r = (ϵF - H.center) / H.scale
        ρ = get_density_from_Tn(Tn_list, N;
                                  fermi=fermi_r, maxdim=maxdim, cutoff=cutoff,
                                  kernel=kernel, lambda=lambda)
        store && (H._density_cache = ρ)
        return ρ
    else
        error("Unknown method: $method. Choose :mcweeny, :sp2, or :kpm")
    end
end


# ============================================================
# 6. Sign of an MPO via purification
# ============================================================

"""
    sign_mpo(A::MPO, sites; scale=1.0, maxdim=500, cutoff=1e-8,
             maxiters=30, tol=1e-5, verbose=false) -> MPO

Compute `sign(A)` for a Hermitian MPO `A` via two McWeeny purifications:

    sign(A) = θ(A) − θ(−A)

where `θ(A)` is the projector onto the positive-eigenvalue subspace of `A`.
The two initial guesses

    ρ₀₊ = (I + A/scale) / 2   →  McWeeny  →  θ(A)
    ρ₀₋ = (I − A/scale) / 2   →  McWeeny  →  θ(−A)

map eigenvalues of ±A from [−scale, +scale] to [0, 1] before purification.
`scale` must be ≥ the spectral radius of `A`; it defaults to 1.0, which is
correct when A has already been constructed with normalised eigenvalues (e.g.
from `get_valley_operator` whose spectrum lies in (−1, +1)).

The returned MPO has eigenvalues in {−1, +1}.
"""
function sign_mpo(A::MPO, sites;
                  scale::Real     = 1.0,
                  maxdim::Int     = 500,
                  cutoff::Float64 = 1e-8,
                  maxiters::Int   = 30,
                  tol::Float64    = 1e-5,
                  verbose::Bool   = false)
    Id = MPO(sites, "Id")

    # ρ₀± = (I ± A/scale) / 2
    function guess(c)
        ρ0 = 0.5 * +(Id, c * A; maxdim=maxdim, cutoff=cutoff)
        ITensorMPS.truncate!(ρ0; maxdim=maxdim, cutoff=cutoff)
        return ρ0
    end
    ρ_p, ρ_m = _purified_pair(guess, 1.0 / scale, -1.0 / scale;
                              maxiters=maxiters, maxdim=maxdim, cutoff=cutoff,
                              tol=tol, verbose=verbose)

    sA = +(ρ_p, -1.0 * ρ_m; maxdim=maxdim, cutoff=cutoff)
    ITensorMPS.truncate!(sA; maxdim=maxdim, cutoff=cutoff)
    return sA
end


# ============================================================
# 7. LDoS and DOS via the finite-difference density-matrix derivative
# ============================================================

"""
    get_ldos_drho(H::TBHamiltonian, ω; mode=:mpo, dmu=0.05, maxdim=40,
                  cutoff=1e-8, maxiters=30, tol=1e-5, verbose=false) -> MPO or MPS

    get_ldos_drho(H::TBHamiltonian, ωs::AbstractVector; ...) -> Vector{MPO} or Vector{MPS}

Compute the local density-of-states at energy `ω` as the finite-difference
derivative of the McWeeny density matrix with respect to μ:

    A(ω) ≈ [ρ(ω + δμ) − ρ(ω − δμ)] / (2δμ)

where ρ(μ) = θ(μ − H) is the purified density matrix at Fermi level μ.

# Modes

- `mode=:mpo` (default): returns the full MPO `A(ω)`.  Its diagonal elements
  give the LDoS: `LDoS(r, ω) = ⟨r|A(ω)|r⟩`.  Subtraction and truncation are
  performed on the full MPO bond structure.

- `mode=:mps`: extracts the diagonal of each ρ as an MPS via
  `extract_diagonal_to_mps` before taking the difference.  The returned MPS
  encodes `diag(ρ(ω+δμ)) − diag(ρ(ω−δμ))` scaled by `1/(2δμ)`.  This is
  cheaper than `:mpo` since the subtraction and all subsequent operations
  stay in the MPS bond space (no MPO–MPO off-diagonal contributions).

# Arguments
- `ω`       : energy (or `ωs`: vector) at which to evaluate LDoS
- `mode`    : `:mpo` (full operator) or `:mps` (diagonal only, cheaper)
- `dmu`     : finite-difference step / broadening
- `maxdim`  : bond dimension for purification and the final combination
- `cutoff`  : SVD truncation threshold throughout
- `maxiters`: maximum McWeeny iterations per purification call
- `tol`     : idempotency convergence threshold ‖ρ²−ρ‖/‖ρ‖
- `verbose` : print McWeeny residuals
"""
function get_ldos_drho(H::TBHamiltonian, ω::Real;
                       mode::Symbol    = :mpo,
                       dmu::Real       = 0.05,
                       maxdim::Int     = 40,
                       cutoff::Float64 = 1e-8,
                       maxiters::Int   = 30,
                       tol::Float64    = 1e-5,
                       verbose::Bool   = false)
    _require_binary_position_space(H, "get_ldos_drho")
    mode in (:mpo, :mps) ||
        error("get_ldos_drho: mode must be :mpo or :mps, got :$mode")
    _ensure_scale!(H)

    guess(ϵ) = purification_initial_guess(H; ϵF=ϵ, maxdim=maxdim, cutoff=cutoff)
    ρ_p, ρ_m = _purified_pair(guess, ω + dmu, ω - dmu;
                              maxiters=maxiters, maxdim=maxdim, cutoff=cutoff,
                              tol=tol, verbose=verbose)

    if mode == :mpo
        dρ = (1.0 / (2dmu)) * +(ρ_p, -1.0 * ρ_m; maxdim=maxdim, cutoff=cutoff)
        ITensorMPS.truncate!(dρ; maxdim=maxdim, cutoff=cutoff)
        return dρ
    else  # :mps
        d_p = extract_diagonal_to_mps(ρ_p)
        d_m = extract_diagonal_to_mps(ρ_m)
        dρ_mps = (1.0 / (2dmu)) * +(d_p, -1.0 * d_m; maxdim=maxdim, cutoff=cutoff)
        ITensorMPS.truncate!(dρ_mps; maxdim=maxdim, cutoff=cutoff)
        return dρ_mps
    end
end

"""
    get_dos_drho(H::TBHamiltonian, ω; dmu=0.05, maxdim=40, cutoff=1e-8,
                 maxiters=30, tol=1e-5, verbose=false) -> Float64

    get_dos_drho(H::TBHamiltonian, ωs::AbstractVector; ...) -> Vector{Float64}

Compute the total density of states at energy `ω` as the finite-difference
derivative of Tr[ρ(μ)] with respect to μ:

    DOS(ω) ≈ (Tr[ρ(ω + δμ)] − Tr[ρ(ω − δμ)]) / (2δμ)

This is the cheapest variant: only the scalar trace of each purified density
matrix is needed, so no MPO or MPS subtraction is performed.  All other
arguments are identical to `get_ldos_drho`.
"""
function get_dos_drho(H::TBHamiltonian, ω::Real;
                      dmu::Real       = 0.05,
                      maxdim::Int     = 40,
                      cutoff::Float64 = 1e-8,
                      maxiters::Int   = 30,
                      tol::Float64    = 1e-5,
                      verbose::Bool   = false)
    _require_binary_position_space(H, "get_dos_drho")
    _ensure_scale!(H)

    guess(ϵ) = purification_initial_guess(H; ϵF=ϵ, maxdim=maxdim, cutoff=cutoff)
    ρ_p, ρ_m = _purified_pair(guess, ω + dmu, ω - dmu;
                              maxiters=maxiters, maxdim=maxdim, cutoff=cutoff,
                              tol=tol, verbose=verbose)

    return real(tr(ρ_p) - tr(ρ_m)) / (2dmu)
end

function get_dos_drho(H::TBHamiltonian, ωs::AbstractVector{<:Real};
                      dmu::Real       = 0.05,
                      maxdim::Int     = 40,
                      cutoff::Float64 = 1e-8,
                      maxiters::Int   = 30,
                      tol::Float64    = 1e-5,
                      verbose::Bool   = false)
    return [get_dos_drho(H, ω; dmu=dmu, maxdim=maxdim, cutoff=cutoff,
                         maxiters=maxiters, tol=tol, verbose=verbose)
            for ω in ωs]
end


function get_ldos_drho(H::TBHamiltonian, ωs::AbstractVector{<:Real};
                       mode::Symbol    = :mpo,
                       dmu::Real       = 0.05,
                       maxdim::Int     = 40,
                       cutoff::Float64 = 1e-8,
                       maxiters::Int   = 30,
                       tol::Float64    = 1e-5,
                       verbose::Bool   = false)
    return [get_ldos_drho(H, ω; mode=mode, dmu=dmu, maxdim=maxdim, cutoff=cutoff,
                          maxiters=maxiters, tol=tol, verbose=verbose)
            for ω in ωs]
end
