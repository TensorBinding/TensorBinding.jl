# solvers/DMRG.jl — variational DMRG: ground state, spectral DMRG, KPM spectral bounds
#
# Contents: ground-state DMRG (dmrg_gs); the resolvent-squared MPO
# K(ω,η) = (H−ωI)² + η²I (build_K) and its DMRG ground state (dmrg_spectral), whose
# position weight |⟨i|ψ(ω)⟩|² (local_weight) is an LDoS proxy; and the DMRG estimate
# of the Chebyshev rescaling (scale, center), used when H carries no analytic
# estimate yet (_estimate_spectral_bounds, _ensure_scale!).
#
# Entry points: dmrg_gs, dmrg_spectral, build_K, local_weight; _ensure_scale!
#   (internal: the Chebyshev solvers call it before rescaling H).
# Depends on: core/Utils.jl (binary_to_MPS), core/TBSystem.jl (TBHamiltonian).
#
# LDoS connection: minimising ⟨ψ|K|ψ⟩ forces |ψ(ω)⟩ onto the eigenstate of H
# closest to ω, so |⟨i|ψ(ω)⟩|² approximates the LDoS at position i and energy ω
# (broadened by η); sweeping ω reconstructs the site-resolved spectral function.
# Section 5 moved here from solvers/kpm/recursion.jl in Tier 1.


# ============================================================
# 1. Ground-state DMRG
# ============================================================

"""
    dmrg_gs(H_mpo::MPO, sites; linkdim_init=10, nsweeps=10,
            maxdim=[10, 20, 50, 100, 200], cutoff=1e-8,
            noise=[1e-6, 1e-7, 1e-8, 0.0], kwargs...) -> (E, ψ)

Find the ground state and energy of `H_mpo` using DMRG.

# Keyword arguments
- `linkdim_init` : bond dimension of the random initial MPS. Default `10`.
- `nsweeps`      : total number of DMRG sweeps. Default `10`.
- `maxdim`       : max bond dimension per sweep (scalar or vector). Default
                   `[10, 20, 50, 100, 200]`.
- `cutoff`       : SVD truncation cutoff. Default `1e-8`.
- `noise`        : perturbative noise per sweep (scalar or vector; aids convergence).
                   Default `[1e-6, 1e-7, 1e-8, 0.0]`.
- `kwargs...`    : passed on to `ITensorMPS.dmrg` (e.g. `outputlevel=0`).

Returns `(E, ψ)`.
"""
function dmrg_gs(H_mpo::MPO, sites;
                 linkdim_init::Int = 10,
                 nsweeps::Int      = 10,
                 maxdim            = [10, 20, 50, 100, 200],
                 cutoff::Real      = 1e-8,
                 noise             = [1e-6, 1e-7, 1e-8, 0.0],
                 kwargs...)
    ψ0 = random_mps(sites; linkdims = linkdim_init)
    E, ψ = dmrg(H_mpo, ψ0;
                nsweeps = nsweeps,
                maxdim  = maxdim,
                cutoff  = cutoff,
                noise   = noise,
                kwargs...)
    return E, ψ
end


# ============================================================
# 2. Resolvent-squared MPO K(ω,η) = (H−ωI)² + η²I
# ============================================================

"""
    build_K(H_mpo::MPO, sites, ω, η; maxdim_K=200, cutoff_K=1e-8) -> MPO

Build the resolvent-squared MPO

    K(ω,η) = (H − ω I)² + η² I

The ground state energy of K satisfies `E_K ≥ η²`, with `E_K → η²` when ω
coincides with an eigenvalue of H.  The ground state concentrates on the
eigenstate of H nearest to ω.

`maxdim_K` and `cutoff_K` control truncation of the intermediate MPO product;
larger `maxdim_K` gives a more accurate K at the cost of DMRG wall time.
"""
function build_K(H_mpo::MPO, sites, ω::Real, η::Real;
                 maxdim_K::Int  = 200,
                 cutoff_K::Real = 1e-8)
    I_mpo   = MPO(sites, "Id")
    H_shift = +(H_mpo, (-ω) * I_mpo; cutoff = cutoff_K)
    H_sq    = apply(H_shift, H_shift; cutoff = cutoff_K, maxdim = maxdim_K)
    return +(H_sq, (η^2) * I_mpo; cutoff = cutoff_K)
end


# ============================================================
# 3. Spectral DMRG (minimise K)
# ============================================================

"""
    dmrg_spectral(H_mpo::MPO, sites, ω, η; ψ0=nothing, maxdim_K=200, cutoff_K=1e-8,
                  linkdim_init=10, nsweeps=10, maxdim=[10, 20, 50, 100, 200],
                  cutoff=1e-8, noise=[1e-6, 1e-7, 1e-8, 0.0], kwargs...) -> (E_K, ψ)

Find the ground state of `K(ω,η) = (H−ωI)² + η²I` using DMRG.

The ground state `|ψ(ω)⟩` concentrates on the eigenstate of H closest to ω.
Use `local_weight(ψ, i, L, sites)` to read off the LDoS proxy `|⟨i|ψ(ω)⟩|²`.

# Keyword arguments
- `ψ0`                 : warm-start MPS (random if `nothing`, the default)
- `maxdim_K`, `cutoff_K` : truncation for building K (see `build_K`). Defaults
                         `200`, `1e-8`.
- `linkdim_init`       : bond dimension of the random initial MPS (if ψ0=nothing).
                         Default `10`.
- `nsweeps`, `maxdim`, `cutoff`, `noise` : DMRG sweep parameters, with the same
                         defaults as `dmrg_gs`.
- `kwargs...`          : passed on to `ITensorMPS.dmrg` (e.g. `outputlevel=0`).

Returns `(E_K, ψ)` where `E_K ≈ η²` at spectral peaks.
"""
function dmrg_spectral(H_mpo::MPO, sites, ω::Real, η::Real;
                       ψ0                = nothing,
                       maxdim_K::Int     = 200,
                       cutoff_K::Real    = 1e-8,
                       linkdim_init::Int = 10,
                       nsweeps::Int      = 10,
                       maxdim            = [10, 20, 50, 100, 200],
                       cutoff::Real      = 1e-8,
                       noise             = [1e-6, 1e-7, 1e-8, 0.0],
                       kwargs...)
    K    = build_K(H_mpo, sites, ω, η; maxdim_K = maxdim_K, cutoff_K = cutoff_K)
    init = isnothing(ψ0) ? random_mps(sites; linkdims = linkdim_init) : ψ0
    E_K, ψ = dmrg(K, init;
                  nsweeps = nsweeps,
                  maxdim  = maxdim,
                  cutoff  = cutoff,
                  noise   = noise,
                  kwargs...)
    return E_K, ψ
end


# ============================================================
# 4. LDoS proxy |⟨i|ψ⟩|²
# ============================================================

"""
    local_weight(ψ, i, L, sites) -> Float64

Return `|⟨i|ψ⟩|²` where `|i⟩` is the position-basis state for 0-based
integer `i` (big-endian quantics encoding over `L` qubit `sites`).

When `ψ = ψ(ω)` is the DMRG ground state of `K(ω,η)`, this gives the
**LDoS proxy** at site `i` and energy `ω`:

    ρ(ω, i) ≈ |⟨i|ψ(ω)⟩|²

which peaks at eigenvalues of H that have support on site i.

For a BdG or spin-extended system pass the full `ext_sites` and set
`L = length(ext_sites)`.
"""
function local_weight(ψ::MPS, i::Integer, L::Integer, sites)
    ket = binary_to_MPS(i, L, sites)
    return abs2(inner(ket, ψ))
end


# ============================================================
# 5. KPM spectral bounds (DMRG estimate of the Chebyshev rescaling)
# ============================================================

"""
    _estimate_spectral_bounds(H_mpo::MPO, sites; dmrg_nsweeps=5,
                              dmrg_maxdim=[10, 20, 40], dmrg_linkdim=4)
        -> (scale, center)

Run two short DMRG sweeps (minimising H and −H) to find the spectral edges
E_min and E_max, then return:
    center = (E_max + E_min) / 2
    scale  = (E_max − E_min) / 2 × 1.1   (10 % buffer)
"""
function _estimate_spectral_bounds(H_mpo::MPO, sites;
                                    dmrg_nsweeps::Int = 5,
                                    dmrg_maxdim       = [10, 20, 40],
                                    dmrg_linkdim::Int = 4)
    E_min, _ = dmrg_gs(H_mpo, sites;
                        nsweeps      = dmrg_nsweeps,
                        maxdim       = dmrg_maxdim,
                        linkdim_init = dmrg_linkdim,
                        noise        = [1e-6, 1e-7, 0.0],
                        outputlevel  = 0)
    E_max_neg, _ = dmrg_gs((-1.0) * H_mpo, sites;
                             nsweeps      = dmrg_nsweeps,
                             maxdim       = dmrg_maxdim,
                             linkdim_init = dmrg_linkdim,
                             noise        = [1e-6, 1e-7, 0.0],
                             outputlevel  = 0)
    E_max  = -E_max_neg
    center = (E_max + E_min) / 2
    scale  = (E_max - E_min) / 2 * 1.1
    # Visible by default: an automatic scale that misses the spectrum breaks KPM silently.
    @info "KPM_Tn: spectral bounds estimated by DMRG" E_min E_max center scale
    return scale, center
end


"""
    _ensure_scale!(H::TBHamiltonian; dmrg_nsweeps=5, dmrg_maxdim=[10, 20, 40],
                   dmrg_linkdim=4)

If `H.scale == 0` (sentinel meaning "not yet determined"), run
`_estimate_spectral_bounds` and store the results in `H.scale` and `H.center`.
No-op if `H.scale > 0` (analytic estimate already set at construction or
a previous KPM call already ran DMRG).
"""
function _ensure_scale!(H::TBHamiltonian;
                         dmrg_nsweeps::Int = 5,
                         dmrg_maxdim       = [10, 20, 40],
                         dmrg_linkdim::Int = 4)
    H.scale > 0.0 && return H
    sc, c = _estimate_spectral_bounds(H.mpo, H.sites;
                                      dmrg_nsweeps = dmrg_nsweeps,
                                      dmrg_maxdim  = dmrg_maxdim,
                                      dmrg_linkdim = dmrg_linkdim)
    # setfield!, not assignment: filling an undetermined window empties no cache (none is
    # computed before the window is set, and a density set by hand stays).
    setfield!(H, :scale, Float64(sc))
    setfield!(H, :center, Float64(c))
    return H
end
