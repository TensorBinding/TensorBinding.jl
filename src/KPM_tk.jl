"""
    _estimate_spectral_bounds(H_mpo, sites; dmrg_nsweeps, dmrg_maxdim, dmrg_linkdim)
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
    println("KPM_Tn: estimating spectral bounds via DMRG…")
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
    println("  E_min = $(round(E_min; digits=4)),  E_max = $(round(E_max; digits=4))")
    println("  center = $(round(center; digits=4)),  scale = $(round(scale; digits=4))")
    return scale, center
end


"""
    _ensure_scale!(H::TBHamiltonian; dmrg_nsweeps, dmrg_maxdim, dmrg_linkdim)

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
    H.scale, H.center = _estimate_spectral_bounds(H.mpo, H.sites;
                             dmrg_nsweeps = dmrg_nsweeps,
                             dmrg_maxdim  = dmrg_maxdim,
                             dmrg_linkdim = dmrg_linkdim)
    return H
end


"""
    KPM_Tn(H_mpo, N, sites; scale=nothing, center=0.0, maxdim=40,
           dmrg_nsweeps, dmrg_maxdim, dmrg_linkdim) -> (Tn_list, scale, center)

Build the list of Chebyshev MPOs `T_n((H−center·I)/scale)` for `n = 0…N`.

## Scale / center arguments
- If `scale=nothing` (default): spectral bounds estimated automatically via
  `_estimate_spectral_bounds` (two short DMRG runs).
- If `scale` is provided: used directly; `center` defaults to `0.0` but can be
  set explicitly for non-symmetric spectra.

## High-level overload
Pass a `TBHamiltonian` as the first argument to skip manual rescaling entirely:
    Tn, scale, center = KPM_Tn(H, Ncheb; maxdim=100)
`H.scale` and `H.center` are computed lazily on the first call and cached.

## Return value
Returns `(Tn_list, scale, center)`.  To convert a physical energy ω:
    ω_r = (ω − center) / scale  ∈ (−1, 1)
"""
function KPM_Tn(H_mpo::MPO, N::Int, sites;
                scale::Union{Real, Nothing} = nothing,
                center::Real       = 0.0,
                maxdim::Int        = 40,
                dmrg_nsweeps::Int  = 5,
                dmrg_maxdim        = [10, 20, 40],
                dmrg_linkdim::Int  = 4,
                cutoff::Real       = 1e-8,
                verbose::Bool    = true)

    # ── Spectral bounds ───────────────────────────────────────────────────
    if isnothing(scale)
        scale, center = _estimate_spectral_bounds(H_mpo, sites;
                             dmrg_nsweeps = dmrg_nsweeps,
                             dmrg_maxdim  = dmrg_maxdim,
                             dmrg_linkdim = dmrg_linkdim)
    end

    # ── Scaled Hamiltonian: (H − center·I) / scale ────────────────────────
    I_mpo   = MPO(sites, "Id")
    Ham_n   = (1 / scale) * +(H_mpo, (-center) * I_mpo; cutoff = cutoff)

    # ── Chebyshev recursion T_0 = I,  T_1 = H_scaled,  T_k = 2H·T_{k-1} − T_{k-2}
    T_k_minus_2 = I_mpo
    T_k_minus_1 = Ham_n
    Tn_list = [T_k_minus_2, T_k_minus_1]

    for k in 3:N+1
        T_k = +(2 * apply(Ham_n, T_k_minus_1; cutoff = cutoff),
                -T_k_minus_2; maxdim = maxdim)
        T_k = ITensorMPS.truncate!(T_k; cutoff = cutoff)
        T_k_minus_2 = T_k_minus_1
        T_k_minus_1 = T_k
        push!(Tn_list, T_k)
        if verbose
            if k%10 == 0 || k == N+1 # print info every 10 iterations and at the end
                println("Computed T_$((k-1)) with maxlinkdim = ", ITensorMPS.maxlinkdim(T_k))
            end 
        end
    end

    return Tn_list, scale, center
end


"""
    KPM_Tn(H::TBHamiltonian, Ncheb; maxdim=40, cutoff=1e-8,
           dmrg_nsweeps, dmrg_maxdim, dmrg_linkdim) -> (Tn_list, scale, center)

High-level overload: accepts a `TBHamiltonian` directly.

- Calls `_ensure_scale!` to lazily determine `H.scale` and `H.center` via DMRG
  if not already set (i.e. when `H.scale == 0`).
- Builds the rescaled Chebyshev list `T_n((H−center·I)/scale)`.
- Caches the result in `H._tn_cache` / `H._tn_Ncheb`.
- Returns `(Tn_list, H.scale, H.center)` — same tuple as the low-level method.

Usage:
    Tn, scale, center = KPM_Tn(H, 200; maxdim=100)
    ω_r = (ω_phys - center) / scale   # rescaled energy in (-1, 1)
"""
function KPM_Tn(H::TBHamiltonian, Ncheb::Int;
                mode::Symbol                  = :mpo,
                psi0::Union{MPS, Nothing}     = nothing,
                maxdim::Int                   = 40,
                cutoff::Real                  = 1e-8,
                dmrg_nsweeps::Int             = 5,
                dmrg_maxdim                   = [10, 20, 40],
                dmrg_linkdim::Int             = 4,
                verbose::Bool                 = false)
    _ensure_scale!(H; dmrg_nsweeps=dmrg_nsweeps,
                      dmrg_maxdim=dmrg_maxdim,
                      dmrg_linkdim=dmrg_linkdim)
    if mode == :mpo
        Tn, _, _ = KPM_Tn(H.mpo, Ncheb, H.sites;
                           scale    = H.scale,
                           center   = H.center,
                           maxdim   = maxdim,
                           cutoff   = cutoff,
                           verbose  = verbose)
        H._tn_cache = Tn
    elseif mode == :mps
        psi0 === nothing && error("KPM_Tn with mode=:mps requires the psi0 keyword argument")
        Tn, _, _ = KPM_Tn_mps(H.mpo, Ncheb, psi0, H.sites;
                               scale    = H.scale,
                               center   = H.center,
                               maxdim   = maxdim,
                               cutoff   = cutoff,
                               verbose  = verbose)
        H._tn_mps_cache = Tn
    else
        error("Unknown KPM mode: $mode. Choose :mpo or :mps")
    end
    H._tn_Ncheb = Ncheb
    return Tn, H.scale, H.center
end


"""
    KPM_Tn_mps(H_mpo, N, psi0, sites; scale=nothing, center=0.0, maxdim=40,
               dmrg_nsweeps, dmrg_maxdim, dmrg_linkdim, cutoff, verbose)
    -> (Tn_mps_list, scale, center)

MPS-based Chebyshev expansion. Instead of storing Chebyshev MPOs T_n(H) (as
`KPM_Tn` does), this builds the projected MPS states

    |φ_n⟩ = T_n((H − center·I)/scale) |ψ₀⟩,   n = 0 … N

via the three-term recurrence

    |φ₀⟩ = |ψ₀⟩,   |φ₁⟩ = H̃|ψ₀⟩,   |φ_k⟩ = 2H̃|φ_{k-1}⟩ − |φ_{k-2}⟩.

This is more memory-efficient than the full MPO version when only a single
reference state is needed (e.g. site-resolved LDoS). Moments and spectral
weights are then obtained as `inner(ref_mps, Tn_mps_list[n+1])`.

`psi0` is normalised internally. `scale`/`center` follow the same convention as
`KPM_Tn`: if `scale=nothing` the spectral bounds are estimated via DMRG.
Returns `(Tn_mps_list, scale, center)` where `Tn_mps_list[n+1]` = |φ_n⟩.
"""
function KPM_Tn_mps(H_mpo::MPO, N::Int, psi0::MPS, sites;
                    scale::Union{Real, Nothing} = nothing,
                    center::Real       = 0.0,
                    maxdim::Int        = 40,
                    dmrg_nsweeps::Int  = 5,
                    dmrg_maxdim        = [10, 20, 40],
                    dmrg_linkdim::Int  = 4,
                    cutoff::Real       = 1e-8,
                    verbose::Bool    = true)

    # ── Spectral bounds ───────────────────────────────────────────────────
    if isnothing(scale)
        scale, center = _estimate_spectral_bounds(H_mpo, sites;
                             dmrg_nsweeps = dmrg_nsweeps,
                             dmrg_maxdim  = dmrg_maxdim,
                             dmrg_linkdim = dmrg_linkdim)
    end

    # ── Scaled Hamiltonian: (H − center·I) / scale ────────────────────────
    I_mpo = MPO(sites, "Id")
    Ham_n = (1 / scale) * +(H_mpo, (-center) * I_mpo; cutoff = cutoff)

    # ── Chebyshev recursion T_0 = |ψ₀⟩,  |T_1⟩ = H_scaled|ψ₀⟩,  |T_k⟩ = 2H_scaled|ψ_{k-1}⟩ − |ψ_{k-2}⟩
    psi0_n      = psi0 / norm(psi0)  # ensure normalisation
    T_k_minus_2 = psi0_n
    T_k_minus_1 = apply(Ham_n, psi0_n; cutoff = cutoff, maxdim = maxdim)
    Tn_mps_list = [T_k_minus_2, T_k_minus_1]

    for k in 3:N+1
        T_k = +(2 * apply(Ham_n, T_k_minus_1; cutoff = cutoff, maxdim = maxdim),
                -T_k_minus_2; cutoff = cutoff, maxdim = maxdim)
        T_k_minus_2 = T_k_minus_1
        T_k_minus_1 = T_k
        push!(Tn_mps_list, T_k)
        if verbose
            if k % 10 == 0 || k == N + 1
                println("Computed MPS T_$(k-1) with maxlinkdim = ", maxlinkdim(T_k))
            end
        end
    end

    return Tn_mps_list, scale, center
end

function KPM_Tn_mps(H::TBHamiltonian, N::Int, psi0::MPS;
                    maxdim::Int        = 40,
                    cutoff::Real       = 1e-8,
                    dmrg_nsweeps::Int  = 5,
                    dmrg_maxdim        = [10, 20, 40],
                    dmrg_linkdim::Int  = 4,
                    verbose::Bool    = false)
    _ensure_scale!(H; dmrg_nsweeps=dmrg_nsweeps,
                      dmrg_maxdim=dmrg_maxdim,
                      dmrg_linkdim=dmrg_linkdim)
    Tn_mps, _, _ = KPM_Tn_mps(H.mpo, N, psi0, H.sites;
                                scale     = H.scale,
                                center    = H.center,
                                maxdim    = maxdim,
                                cutoff    = cutoff,
                                verbose   = verbose)
    H._tn_mps_cache = Tn_mps
    H._tn_Ncheb     = N
    return Tn_mps, H.scale, H.center
end


"""
    get_ldos(H::TBHamiltonian, ω_phys; psi0, kernel, lambda, eta, m_order, maxdim, zl, wl)

Compute the local density of states at physical energy `ω_phys` using the
Chebyshev expansion cached in `H` by a prior `KPM_Tn` or `KPM_Tn_mps` call.

Dispatches on `H._tn_mode`:

- `:mpo` — calls `get_ldos_w_from_Tn(H._tn_cache, N, E; ...)` and returns an `MPO`
  representing the spectral weight operator at energy `ω_phys`.

- `:mps` — computes moments `μₙ = ⟨ψ₀|φₙ⟩` from the cached MPS states and
  calls `get_ldos_from_mun(μ, N, E; ...)`, returning a `Real` (the site-resolved
  LDoS for the reference state `psi0`).  `psi0` must be the same state that was
  passed to `KPM_Tn_mps`.

The physical energy is converted to the rescaled energy `E = (ω_phys − H.center) / H.scale`
internally, so `ω_phys` should be in the same units as the Hamiltonian.
"""
function get_ldos(H::TBHamiltonian, ω_phys::Real;
                  mode::Symbol              = :mpo,
                  psi0::Union{MPS, Nothing} = nothing,
                  kernel::Symbol  = :jackson,
                  lambda::Real    = 4.0,
                  eta::Real       = 0.0,       # 0.0 → use 1/(N+1) default
                  m_order::Int    = 4,
                  maxdim::Int     = 40,
                  zl              = nothing,
                  wl              = nothing)
    N    = H._tn_Ncheb
    E    = (ω_phys - H.center) / H.scale
    eta_ = eta == 0.0 ? 1 / (N + 1) : eta

    if mode == :mpo
        H._tn_cache === nothing && error("No MPO Chebyshev cache. Call KPM_Tn(H, N; mode=:mpo) first.")
        return get_ldos_w_from_Tn(H._tn_cache, N, E;
                                  maxdim = maxdim,
                                  kernel = kernel,
                                  lambda = lambda,
                                  zl     = zl,
                                  wl     = wl,
                                  eta    = eta_)
    elseif mode == :mps
        H._tn_mps_cache === nothing && error("No MPS Chebyshev cache. Call KPM_Tn(H, N; mode=:mps, psi0=...) first.")
        psi0 === nothing && error("get_ldos with mode=:mps requires the psi0 keyword argument")
        mun = [inner(psi0, H._tn_mps_cache[n]) for n in 1:N]
        return get_ldos_from_mun(mun, N, E;
                                 kernel  = kernel,
                                 lambda  = lambda,
                                 eta     = eta_,
                                 m_order = m_order)
    else
        error("Unknown mode: $mode. Choose :mpo or :mps")
    end
end


"""
    get_ldos_from_mun(mun_list, N, E; kernel=:jackson, lambda=4.0) -> Real

Reconstruct the local spectral weight at rescaled energy `E ∈ (−1, 1)` from a
list of Chebyshev moments `μ_n = ⟨ψ₀|T_n(H̃)|ψ₀⟩` produced by `KPM_Tn_mps`.

Equivalent to computing `⟨ψ₀|δ(E − H̃)|ψ₀⟩` via the KPM expansion:

    A(E) ≈ [g₀μ₀ + 2 Σ_{n≥1} gₙ Tₙ(E) μₙ] / (π √(1−E²))

where `gₙ` are the kernel damping weights (Jackson by default). Supported
`kernel` values: `:jackson`, `:lorentz` (requires `lambda`), `:fejer`,
`:dirichlet`. Returns `0` for `|E| ≥ 1`.

To convert a physical energy ω: `E = (ω − center) / scale`.
To obtain the density of states per site, sum over all sites and divide by N.
"""
function get_ldos_from_mun(mun_list, N::Int, E::Real;
                           kernel::Symbol = :jackson,
                           lambda::Real   = 4.0,
                           eta::Real      = 1/(N+1),
                           m_order::Int   = 4)
    abs(E) >= 1.0 && return 0.0

    if kernel == :hodc
        return get_ldos_hodc_from_mun(mun_list, N, E; eta = eta, m_order = m_order)
    end

    kweights = _kpm_kernel(N, kernel; lambda = lambda)
    G_n(n)   = cos((n - 1) * acos(E))

    val = real(mun_list[1]) * G_n(1) * kweights[1]
    for n in 2:N
        val += 2.0 * real(mun_list[n]) * G_n(n) * kweights[n]
    end

    return val / (π^2 * N * sqrt(1 - E^2))
end


"""
    get_ldos_hodc_from_mun(mun_list, N, E; eta=0.02, m_order=6) -> Real

HODC (High-Order Damping Correction) variant of `get_ldos_from_mun`. Uses a
contour-based kernel that gives sharper spectral features than the Jackson
kernel, at the cost of `m_order` extra parameters.

The HODC weights `νₖ` (from `compute_hodc_params` / `get_hodc_weights`) already
carry the full KPM normalisation, so no extra denominator is needed:

    A_hodc(E) ≈ ν₁μ₁ + 2 Σ_{n≥2} νₙ μₙ

Returns `0` for `|E| ≥ 1`.
"""
function get_ldos_hodc_from_mun(mun_list, N::Int, E::Real;
                                eta::Real    = 0.02,
                                m_order::Int = 4)
    abs(E) >= 1.0 && return 0.0

    zl, wl = compute_hodc_params(m_order)
    nu_k   = get_hodc_weights(E, N, eta, zl, wl)

    val = real(mun_list[1]) * nu_k[1]
    for n in 2:N
        val += real(mun_list[n]) * nu_k[n]
    end

    return real(val)
end


# All kernels are unnormalized (max ≈ N at n=0) so caller's existing /N stays correct.
# Supported: :jackson (default), :lorentz (param lambda), :fejer, :dirichlet
function _kpm_kernel(N::Int, kernel::Symbol; lambda::Real = 4.0)
    if kernel == :jackson
        return [(N - n) * cos(π * n / N) + sin(π * n / N) / tan(π / N) for n in 0:N-1]
    elseif kernel == :lorentz
        return [N * sinh(lambda * (1 - n / N)) / sinh(lambda) for n in 0:N-1]
    elseif kernel == :fejer
        return Float64[N - n for n in 0:N-1]
    elseif kernel == :dirichlet
        return fill(Float64(N), N)
    else
        error("Unknown KPM kernel: $kernel. Choose :jackson, :lorentz, :fejer, or :dirichlet")
    end
end

# === HODC kernel helpers ===

function compute_hodc_params(m=6)
    xl = range(-2.5, 2.5, length=m)
    zl = xl .+ 1im
    A = [z^k for k in 0:m-1, z in zl]
    b = zeros(ComplexF64, m)
    b[1] = 1.0
    wl = A \ b
    return zl, wl
end

function get_hodc_weights(y_target, N, eta, zl, wl)
    j = 0:N-1
    nodes = cos.(π .* (j .+ 0.5) ./ N)
    kernel_vals = map(nodes) do x
        term = sum(wl ./ (y_target - x .+ eta .* zl))
        return -1.0/π * imag(term)
    end
    nu = FFTW.r2r(kernel_vals, FFTW.REDFT10) ./ N
    nu[1] /= 2.0
    return nu
end

# Returns complex weights π*(ν_HT - i*ν_δ) for the retarded Green's function.
# ν_δ comes from -Im[...]/π  (same as get_hodc_weights),
# ν_HT comes from  Re[...]/π (real part of the same rational sum — no extra cost).
function get_hodc_gf_weights(y_target, N, eta, zl, wl)
    j = 0:N-1
    nodes = cos.(π .* (j .+ 0.5) ./ N)

    sums = map(nodes) do x
        sum(wl ./ (y_target - x .+ eta .* zl))
    end

    nu_delta = FFTW.r2r(-imag.(sums) ./ π, FFTW.REDFT10) ./ N
    nu_delta[1] /= 2.0

    nu_HT = FFTW.r2r(real.(sums) ./ π, FFTW.REDFT10) ./ N
    nu_HT[1] /= 2.0

    return π .* (nu_HT .- im .* nu_delta)
end

function get_density_from_Tn(Tn_list,N;fermi=0,maxdim=40,kernel=:jackson,lambda=4.0)

    jackson_kernel = _kpm_kernel(N, kernel; lambda=lambda)

    function G_n(n)
        if n == 1
            return acos(fermi)
        else
            return sin((n-1) * acos(fermi)) / (n-1)
        end
    end

    # Compute electronic density
    A = Tn_list[1] * G_n(1) * jackson_kernel[1] 
    for n in 2:N
        A = +(A,  2 *  Tn_list[n] * G_n(n) * jackson_kernel[n]; maxdim=maxdim)
        A = ITensorMPS.truncate!(A;cutoff=1e-8)
    end
    A /= (π* N)
    
    return  A
end

function get_Green_retarded_from_Tn(Tn_list, N, ω; η=1e-2, maxdim=40,
                                     kernel=:jackson, lambda=4.0,
                                     zl=nothing, wl=nothing)
    if kernel == :hodc
        zl === nothing && error("kernel=:hodc requires zl and wl from compute_hodc_params()")
        return get_Green_retarded_from_Tn_hodc(Tn_list, N, ω, zl, wl; eta=η, maxdim=maxdim)
    end

    kweights = _kpm_kernel(N, kernel; lambda=lambda)

    function G_n(n, ω, η)
        z = ω + 1im*η
        θ = acos(z)
        return -2im/(1 + ==(n-1,0)) * exp(-1im * (n-1) * θ) / sqrt(1 - z^2)
    end

    G = Tn_list[1] * G_n(1, ω, η) * kweights[1]
    for n in 2:N
        G = +(G, Tn_list[n] * G_n(n, ω, η) * kweights[n]; maxdim=maxdim)
        G = ITensorMPS.truncate!(G; cutoff=1e-8)
    end
    G /= N

    return G
end

function get_Green_retarded_from_Tn_hodc(Tn_list, N, ω, zl, wl; eta=1e-2, maxdim=40)
    c = get_hodc_gf_weights(ω, N, eta, zl, wl)

    G = Tn_list[1] * c[1]
    for n in 2:N
        G = +(G, Tn_list[n] * c[n]; maxdim=maxdim)
        G = ITensorMPS.truncate!(G; cutoff=1e-8)
    end
    return G
end


function get_ldos_w_from_Tn(Tn_list, N, ω; maxdim=40, kernel=:jackson, lambda=4.0,
                             zl=nothing, wl=nothing, eta=1e-2)
    if kernel == :hodc
        zl === nothing && error("kernel=:hodc requires zl and wl from compute_hodc_params()")
        return get_ldos_w_from_Tn_hodc(Tn_list, N, ω, zl, wl; eta=eta, maxdim=maxdim)
    end

    kweights = _kpm_kernel(N, kernel; lambda=lambda)
    G_n(n) = cos((n - 1) * acos(ω)) / (π * sqrt(1 - ω^2))

    A = Tn_list[1] * G_n(1) * kweights[1]
    for n in 2:N
        A = +(A, 2 * Tn_list[n] * G_n(n) * kweights[n]; maxdim=maxdim)
        A = ITensorMPS.truncate!(A; cutoff=1e-8)
    end
    A /= (π * N)
    return A
end

# HODC variant: nu coefficients encode both kernel and spectral target directly.
# Call compute_hodc_params once per expansion order, then pass zl, wl here.
function get_ldos_w_from_Tn_hodc(Tn_list, N, ω, zl, wl; eta=1e-2, maxdim=40)
    nu = get_hodc_weights(ω, N, eta, zl, wl)

    A = Tn_list[1] * nu[1]
    for n in 2:N
        A = +(A, Tn_list[n] * nu[n]; maxdim=maxdim)
        A = ITensorMPS.truncate!(A; cutoff=1e-8)
    end
    return A
end

function get_PH_from_Tn(Tn_list, N, ω; maxdim=40, kernel=:jackson, lambda=4.0)
    kweights = _kpm_kernel(N, kernel; lambda=lambda)
    G_n(n) = cos((n - 1) * acos(ω)) / (π * sqrt(1 - ω^2))

    A = Tn_list[1] * G_n(1) * kweights[1]
    for n in 2:N
        A = +(A, 2 * Tn_list[n] * G_n(n) * kweights[n]; maxdim=maxdim)
        A = ITensorMPS.truncate!(A; cutoff=1e-8)
    end
    A /= (π * N)
    return A
end


#for getting electron densities
function get_density_quantics(A,L)
    
    xvals = range(0, (2^L - 1); length=2^L)
    f(x) =  1 -  inner(random_mps(sites,to_binary_vector(Int(x),L))',A, random_mps(sites,to_binary_vector(Int(x),L)))
    qtt, ranks, errors = quanticscrossinterpolate(Float64, f,  xvals ; tolerance=1e-8)

    tt = TCI.tensortrain(qtt.tci)
    density_mps = ITensors.MPS(tt;sites)
  
    density_mpo = outer(density_mps',density_mps)
    for i in 1:L
        density_mpo.data[i] =  Quantics._asdiagonal(density_mps.data[i],sites[i])
    end
    
    return qtt,density_mpo,density_mps
end