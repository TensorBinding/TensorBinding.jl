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
        if iter%15 == 0 && verbose
            println("McWeeny iter $iter: ‖ρ²-ρ‖/‖ρ‖ = $err, maxlinkdim = $(ITensorMPS.maxlinkdim(ρ))")
        end
        err < tol && break
        # ρ³ = ρ² · ρ
        #ρ3 = apply(ρ2, ρ; maxdim, cutoff)
        #ITensorMPS.truncate!(ρ3; maxdim, cutoff)
        # 3ρ² - 2ρ³
        ρ_inte = +(3.0 * ρ , -2.0 * ρ2; cutoff)
        ρ  = apply(ρ, ρ_inte; maxdim, cutoff) 
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

function purification_initial_guess(H::TBHamiltonian; ϵF::Real=0.0,
                                    maxdim::Int=40, cutoff::Float64=1e-8)
    _ensure_scale!(H)
    Id       = MPO(H.sites, "Id")
    coeff_I  = 0.5 + (ϵF + H.center) / (2 * H.scale)
    coeff_H  = -0.5 / H.scale
    ρ0       = +(coeff_I * Id, coeff_H * H.mpo; cutoff)
    ITensorMPS.truncate!(ρ0; maxdim=maxdim, cutoff)
    return ρ0
end


"""
    mcweeny_purify(H::TBHamiltonian; ϵF, maxiters, maxdim, cutoff, tol, verbose) -> MPO

High-level overload: builds the initial guess from `H`, runs McWeeny purification,
caches the result in `H._density_cache`, and returns the purified density matrix.

`ϵF` shifts the Fermi level of the initial guess ρ₀ = (I − (H − ϵF·I)/scale) / 2,
allowing purification to target a band other than half-filling. Default `0.0`.
"""
function mcweeny_purify(H::TBHamiltonian;
                        ϵF::Real        = 0.0,
                        maxiters::Int   = 30,
                        maxdim::Int     = 40,
                        cutoff::Float64 = 1e-8,
                        tol::Float64    = 1e-5,
                        verbose::Bool   = false)
    ρ0 = purification_initial_guess(H; ϵF=ϵF, maxdim=maxdim, cutoff=cutoff)
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


# ============================================================
# Unified high-level density-matrix wrapper
# ============================================================

"""
    get_density(H::TBHamiltonian; method, ϵF, Ncheb, kernel, lambda,
                maxdim, cutoff, Nel, maxiters, tol, verbose) -> MPO

Compute and cache the zero-temperature density matrix P = θ(ϵF − H).

If `H._density_cache` is already populated it is returned immediately.
Set `H._density_cache = nothing` to force a fresh computation.

**method**
- `:mcweeny` (default) — McWeeny purification P_{n+1} = 3P_n² − 2P_n³
- `:sp2`               — SP2 purification (1 MPO product/step), requires `Nel`
- `:kpm`               — KPM Chebyshev expansion of the Fermi step function

**Keyword arguments**
- `ϵF`      : Fermi energy in physical units (`:kpm` only). Default `0.0`.
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

    if H._density_cache !== nothing
        verbose && println("get_density: returning cached density matrix")
        return H._density_cache
    end

    if method == :mcweeny
        return mcweeny_purify(H; ϵF=ϵF, maxiters=maxiters, maxdim=maxdim, cutoff=cutoff,
                                 tol=tol, verbose=verbose)
    elseif method == :sp2
        return sp2_purify(H; Nel=Nel, maxiters=maxiters, maxdim=maxdim, cutoff=cutoff,
                             tol=tol, verbose=verbose)
    elseif method == :kpm
        if H._tn_cache === nothing || H._tn_Ncheb < Ncheb
            KPM_Tn(H, Ncheb; maxdim=maxdim, cutoff=cutoff, verbose=verbose)
        end
        fermi_r = (ϵF - H.center) / H.scale
        ρ = get_density_from_Tn(H._tn_cache, H._tn_Ncheb;
                                  fermi=fermi_r, maxdim=maxdim, cutoff=cutoff,
                                  kernel=kernel, lambda=lambda)
        H._density_cache = ρ
        return ρ
    else
        error("Unknown method: $method. Choose :mcweeny, :sp2, or :kpm")
    end
end
