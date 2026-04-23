# Purification_tk.jl — density matrix purification methods
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
#   ρ0 = get_density(H; Ncheb=30, method=:KPM)   # cheap rough guess
#   ρ  = mcweeny_purify(ρ0; maxdim=40)
#
# Both functions accept `cutoff` and `maxdim` to control truncation
# at each MPO-MPO multiplication step.

# ============================================================
# Shared helpers
# ============================================================

"""
    _mpo_sq(ρ, sites; maxdim, cutoff) -> MPO

Compute `ρ²` via `apply` and truncate immediately.  The intermediate
bond dimension of `apply` is controlled by `maxdim`.
"""
function _mpo_sq(ρ::MPO; maxdim::Int, cutoff::Float64)
    ρ2 = apply(ρ, ρ; maxdim, cutoff)
    ITensorMPS.truncate!(ρ2; maxdim, cutoff)
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


# ============================================================
# McWeeny purification
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
- `verbose`  : print residual at each iteration

# Returns
Purified density matrix MPO.
"""
function mcweeny_purify(ρ0::MPO;
                        maxiters::Int   = 30,
                        maxdim::Int     = 40,
                        cutoff::Float64 = 1e-8,
                        tol::Float64    = 1e-5,
                        verbose::Bool   = false)
    ρ = deepcopy(ρ0)
    for iter in 1:maxiters
        ρ2  = _mpo_sq(ρ; maxdim, cutoff)
        err = _idempotency_error(ρ, ρ2)
        verbose && println("McWeeny iter $iter: ‖ρ²-ρ‖/‖ρ‖ = $err, maxlinkdim = $(ITensorMPS.maxlinkdim(ρ))")
        err < tol && break
        # ρ³ = ρ² · ρ
        ρ3 = apply(ρ2, ρ; maxdim, cutoff)
        ITensorMPS.truncate!(ρ3; maxdim, cutoff)
        # 3ρ² - 2ρ³
        ρ = +(3.0 * ρ2, -2.0 * ρ3; cutoff)
        ITensorMPS.truncate!(ρ; maxdim, cutoff)
    end
    return ρ
end


# ============================================================
# SP2 purification
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
    ρ = deepcopy(ρ0)
    for iter in 1:maxiters
        ρ2  = _mpo_sq(ρ; maxdim, cutoff)
        err = _idempotency_error(ρ, ρ2)
        verbose && println("SP2 iter $iter: ‖ρ²-ρ‖/‖ρ‖ = $err, maxlinkdim = $(ITensorMPS.maxlinkdim(ρ))")
        err < tol && break
        tr_ρ2 = real(tr(ρ2))
        if tr_ρ2 >= Nel
            # contract toward 0: keep ρ²
            ρ = ρ2
        else
            # expand toward 1: 2ρ - ρ²
            ρ = +(2.0 * ρ, -1.0 * ρ2; cutoff)
            ITensorMPS.truncate!(ρ; maxdim, cutoff)
        end
    end
    return ρ
end


# ============================================================
# Convenience: build initial guess from the resolvent (Id - H/scale)/2
# ============================================================

"""
    purification_initial_guess(H_mpo, scale, sites; maxdim=40, cutoff=1e-8) -> MPO
    purification_initial_guess(H::TBHamiltonian; maxdim=40, cutoff=1e-8) -> MPO

Construct the simplest valid initial guess for purification:

    ρ₀ = (I - (H − center·I)/scale) / 2

This maps the rescaled spectrum ∈ [-1, 1] to ρ₀ eigenvalues ∈ [0, 1],
the required input range for both `mcweeny_purify` and `sp2_purify`.

The `TBHamiltonian` overload calls `_ensure_scale!` automatically and
accounts for a non-zero spectral center.
"""
function purification_initial_guess(H_mpo::MPO, scale::Float64, sites; maxdim::Int= 40,
                                    cutoff::Float64 = 1e-8)
    Id  = MPO(sites, "Id")
    ρ0  = +(0.5 * Id, (-0.5 / scale) * H_mpo; cutoff)
    ITensorMPS.truncate!(ρ0; maxdim=maxdim, cutoff)
    return ρ0
end

function purification_initial_guess(H::TBHamiltonian; maxdim::Int=40, cutoff::Float64=1e-8)
    _ensure_scale!(H)
    Id       = MPO(H.sites, "Id")
    coeff_I  = 0.5 + H.center / (2 * H.scale)
    coeff_H  = -0.5 / H.scale
    ρ0       = +(coeff_I * Id, coeff_H * H.mpo; cutoff)
    ITensorMPS.truncate!(ρ0; maxdim=maxdim, cutoff)
    return ρ0
end


"""
    mcweeny_purify(H::TBHamiltonian; Nel, maxiters, maxdim, cutoff, tol, verbose) -> MPO

High-level overload: builds the initial guess from `H`, runs McWeeny purification,
caches the result in `H._density_cache`, and returns the purified density matrix.
"""
function mcweeny_purify(H::TBHamiltonian;
                        maxiters::Int   = 30,
                        maxdim::Int     = 40,
                        cutoff::Float64 = 1e-8,
                        tol::Float64    = 1e-5,
                        verbose::Bool   = false)
    ρ0 = purification_initial_guess(H; maxdim=maxdim, cutoff=cutoff)
    ρ  = mcweeny_purify(ρ0; maxiters=maxiters, maxdim=maxdim, cutoff=cutoff,
                            tol=tol, verbose=verbose)
    H._density_cache = ρ
    return ρ
end


"""
    sp2_purify(H::TBHamiltonian; Nel, maxiters, maxdim, cutoff, tol, verbose) -> MPO

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
    ρ0 = purification_initial_guess(H; maxdim=maxdim, cutoff=cutoff)
    ρ  = sp2_purify(ρ0, Nel; maxiters=maxiters, maxdim=maxdim, cutoff=cutoff,
                              tol=tol, verbose=verbose)
    H._density_cache = ρ
    return ρ
end
