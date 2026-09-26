# solvers/kpm/cached.jl — spectral quantities from stored Chebyshev data
#
# Contents: the LDOS from the Chebyshev cache that KPM_Tn leaves on H (get_ldos,
# get_ldos_spectrum); the scalar reconstructions from a list of moments μ_n
# (get_ldos_from_mun, get_ldos_hodc_from_mun); and the operator-valued
# reconstructions from a stored T_n MPO list: the KPM step-function expansion at
# `fermi` (get_density_from_Tn), the retarded Green's function
# (get_Green_retarded_from_Tn, _hodc variant), the spectral-weight MPO
# (get_ldos_w_from_Tn, _hodc variant) and the diagonal LDOS MPS at many energies
# (get_ldos_diag_from_Tn).
#
# Entry points: get_ldos, get_ldos_spectrum, get_ldos_from_mun, get_density_from_Tn,
#   get_Green_retarded_from_Tn, get_ldos_w_from_Tn, get_ldos_diag_from_Tn
# Depends on: core/Utils.jl (extract_diagonal_to_mps), core/TBSystem.jl
#   (TBHamiltonian), solvers/kpm/kernels.jl (_kpm_kernel, _kpm_weight_matrix,
#   compute_hodc_params, get_hodc_weights, get_hodc_gf_weights).
#
# Split from the former solvers/KPM_tk.jl in Tier 1.

# ============================================================
# 1. Cached-Chebyshev LDOS: get_ldos, get_ldos_spectrum
# ============================================================

"""
    get_ldos(H::TBHamiltonian, ω_phys; mode=:diag, psi0=nothing, kernel=:jackson,
             lambda=4.0, eta=0.0, m_order=4, maxdim=40, cutoff=1e-8,
             zl=nothing, wl=nothing)

Compute the local density of states at physical energy `ω_phys` using the
Chebyshev expansion cached in `H` by a prior `KPM_Tn` or `KPM_Tn_mps` call.

**Modes**

- `:mpo`  (legacy) — calls `get_ldos_w_from_Tn` and returns a full spectral-weight
  **MPO** at `ω_phys`.  Requires `KPM_Tn(H, N; mode=:mpo)`.  Retains off-diagonal
  information; use when spatial correlations are needed.

- `:diag` (default) — calls `get_ldos_diag_from_Tn` and returns an **MPS** encoding
  only the diagonal `A(r, ω_phys)` (`nothing` outside the spectral support).
  Requires `KPM_Tn(H, N; mode=:mpo)`.  Much cheaper than
  `:mpo` for typical LDOS use-cases; mirrors the `get_bands` momentum-space pattern.
  Use `get_ldos_spectrum` to compute all energies in a single pass.
  Use `get_ldos_online` to avoid storing the Tn cache entirely.

- `:mps`  — computes moments `μₙ = ⟨ψ₀|φₙ⟩` from the MPS Chebyshev cache and
  calls `get_ldos_from_mun`, returning a **Real**.  Requires
  `KPM_Tn(H, N; mode=:mps, psi0=...)` and the same `psi0` here.

**Keywords**

- `kernel`, `lambda` — KPM kernel (`:jackson`, `:lorentz` with `lambda`, `:fejer`,
  `:dirichlet`; `:hodc` in modes `:mpo` and `:mps` only).
- `eta`, `m_order` — HODC broadening and order; `eta=0` means `1/(N+1)`.
  `m_order` is used in mode `:mps`; mode `:mpo` takes the HODC nodes `zl`, `wl`
  from `compute_hodc_params` instead.
- `maxdim`, `cutoff` — truncation of the MPO/MPS sums (modes `:diag`, `:mpo`).

Physical energies are converted via `E = (ω_phys − H.center) / H.scale`.
"""
function get_ldos(H::TBHamiltonian, ω_phys::Real;
                  mode::Symbol              = :diag,
                  psi0::Union{MPS, Nothing} = nothing,
                  kernel::Symbol  = :jackson,
                  lambda::Real    = 4.0,
                  eta::Real       = 0.0,
                  m_order::Int    = 4,
                  maxdim::Int     = 40,
                  cutoff::Real    = 1e-8,
                  zl              = nothing,
                  wl              = nothing)
    N    = H._tn_Ncheb
    E    = (ω_phys - H.center) / H.scale
    eta_ = eta == 0.0 ? 1 / (N + 1) : eta

    if mode == :diag
        H._tn_cache === nothing && error("No MPO Chebyshev cache. Call KPM_Tn(H, N; mode=:mpo) first.")
        result = get_ldos_diag_from_Tn(H._tn_cache, N, [E];
                                        kernel=kernel, lambda=lambda,
                                        maxdim=maxdim, cutoff=cutoff)
        result[1] === nothing && return nothing
        return result[1]

    elseif mode == :mpo
        H._tn_cache === nothing && error("No MPO Chebyshev cache. Call KPM_Tn(H, N; mode=:mpo) first.")
        return get_ldos_w_from_Tn(H._tn_cache, N, E;
                                  maxdim = maxdim,
                                  cutoff = cutoff,
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
        error("Unknown mode: $mode. Choose :diag, :mpo, or :mps")
    end
end


"""
    get_ldos_spectrum(H::TBHamiltonian, ω_phys_vals; kernel=:jackson, lambda=4.0,
                      maxdim=40, cutoff=1e-8)
        -> Vector{Union{Nothing, MPS}}

Compute the site-resolved LDOS at **all** physical energies in `ω_phys_vals` in a
single pass over the cached Chebyshev MPO list — the real-space equivalent of
`get_bands`.

At each Chebyshev step the diagonal of `T_n` is extracted once and its weighted
contribution is accumulated into every energy slot simultaneously, so the cost
scales as `O(Ncheb)` MPO operations regardless of how many energy points are
requested.

Returns `Vector{Union{Nothing, MPS}}` of length `length(ω_phys_vals)`.  Each MPS
encodes the site-resolved LDOS `A(r, ω)` at the corresponding physical energy;
entries are `nothing` for energies outside the spectral support.

Requires `KPM_Tn(H, Ncheb; mode=:mpo)` to have been called first.

Example
-------
```julia
KPM_Tn(H, 200; mode=:mpo, maxdim=100)

ωlist    = range(-4.0, 4.0; length=200)
ldos_vec = get_ldos_spectrum(H, ωlist)

# Evaluate LDOS at site x=16 (0-indexed position 15) for each energy:
ldos_at_16 = [l === nothing ? 0.0 : eval_mps(l, 15) for l in ldos_vec]
```
"""
function get_ldos_spectrum(H::TBHamiltonian, ω_phys_vals;
                            kernel::Symbol = :jackson,
                            lambda::Real   = 4.0,
                            maxdim::Int    = 40,
                            cutoff::Real   = 1e-8)
    H._tn_cache === nothing &&
        error("No MPO Chebyshev cache. Call KPM_Tn(H, Ncheb; mode=:mpo) first.")
    N      = H._tn_Ncheb
    ω_vals = (collect(ω_phys_vals) .- H.center) ./ H.scale
    return get_ldos_diag_from_Tn(H._tn_cache, N, ω_vals;
                                  kernel=kernel, lambda=lambda,
                                  maxdim=maxdim, cutoff=cutoff)
end


# ============================================================
# 2. Spectral reconstruction from Chebyshev moments μ_n
# ============================================================

"""
    get_ldos_from_mun(mun_list, N::Int, E::Real; kernel=:jackson, lambda=4.0,
                      eta=1/(N+1), m_order=4) -> Real

Reconstruct the local spectral weight at rescaled energy `E ∈ (−1, 1)` from a
list of Chebyshev moments `μ_n = ⟨ψ₀|T_n(H̃)|ψ₀⟩` produced by `KPM_Tn_mps`.

Computes the KPM expansion of `⟨ψ₀|δ(E − H̃)|ψ₀⟩` with an extra factor `1/π`:

    A(E) ≈ [g₀μ₀ + 2 Σ_{n≥1} gₙ Tₙ(E) μₙ] / (π² √(1−E²))

where `gₙ` are the kernel damping weights normalised to `g₀ = 1` (Jackson by
default), so `∫ A dE = 1/π` per state for these kernels. The `:hodc` weights
carry the δ-function normalisation itself (`∫ A dE = 1`). Supported
`kernel` values: `:jackson`, `:lorentz` (requires `lambda`), `:fejer`,
`:dirichlet`, and `:hodc`, which hands over to `get_ldos_hodc_from_mun` with the
contour broadening `eta` (default `1/(N+1)`) and order `m_order` (default `4`);
`eta` and `m_order` are ignored by the other kernels. Returns `0` for `|E| ≥ 1`.

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
    get_ldos_hodc_from_mun(mun_list, N::Int, E::Real; eta=0.02, m_order=4) -> Real

HODC (Higher-Order Delta Chebyshev) variant of `get_ldos_from_mun`. Uses a
contour-based kernel that gives sharper spectral features than the Jackson
kernel, at the cost of `m_order` extra parameters.

The HODC weights `νₖ` (from `compute_hodc_params` / `get_hodc_weights`) already
carry the full KPM normalisation, including the factor 2 of the `n ≥ 2` terms, so
no extra denominator is needed:

    A_hodc(E) ≈ Σ_{n≥1} νₙ μₙ

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


# ============================================================
# 3. Density, Green's functions and LDOS from cached T_n MPOs
# ============================================================

function get_density_from_Tn(Tn_list, N; fermi=0, maxdim=40, cutoff=1e-8,
                              kernel=:jackson, lambda=4.0)
    jackson_kernel = _kpm_kernel(N, kernel; lambda=lambda)

    function G_n(n)
        n == 1 ? acos(fermi) : sin((n-1) * acos(fermi)) / (n-1)
    end

    A = Tn_list[1] * G_n(1) * jackson_kernel[1]
    for n in 2:N
        A = +(A, 2 * Tn_list[n] * G_n(n) * jackson_kernel[n]; maxdim=maxdim)
        A = ITensorMPS.truncate!(A; cutoff=cutoff)
    end
    A /= (π * N)
    return A
end

function get_Green_retarded_from_Tn(Tn_list, N, ω; η=1e-2, maxdim=40, cutoff=1e-8,
                                     kernel=:jackson, lambda=4.0,
                                     zl=nothing, wl=nothing)
    if kernel == :hodc
        zl === nothing && error("kernel=:hodc requires zl and wl from compute_hodc_params()")
        return get_Green_retarded_from_Tn_hodc(Tn_list, N, ω, zl, wl;
                                                eta=η, maxdim=maxdim, cutoff=cutoff)
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
        G = ITensorMPS.truncate!(G; cutoff=cutoff)
    end
    G /= N
    return G
end

function get_Green_retarded_from_Tn_hodc(Tn_list, N, ω, zl, wl; eta=1e-2, maxdim=40,
                                          cutoff=1e-8)
    c = get_hodc_gf_weights(ω, N, eta, zl, wl)

    G = Tn_list[1] * c[1]
    for n in 2:N
        G = +(G, Tn_list[n] * c[n]; maxdim=maxdim)
        G = ITensorMPS.truncate!(G; cutoff=cutoff)
    end
    return G
end


function get_ldos_w_from_Tn(Tn_list, N, ω; maxdim=40, cutoff=1e-8, kernel=:jackson,
                             lambda=4.0, zl=nothing, wl=nothing, eta=1e-2)
    if kernel == :hodc
        zl === nothing && error("kernel=:hodc requires zl and wl from compute_hodc_params()")
        return get_ldos_w_from_Tn_hodc(Tn_list, N, ω, zl, wl; eta=eta, maxdim=maxdim, cutoff=cutoff)
    end

    kweights = _kpm_kernel(N, kernel; lambda=lambda)
    G_n(n) = cos((n - 1) * acos(ω)) / (π * sqrt(1 - ω^2))

    A = Tn_list[1] * G_n(1) * kweights[1]
    for n in 2:N
        A = +(A, 2 * Tn_list[n] * G_n(n) * kweights[n]; maxdim=maxdim)
        A = ITensorMPS.truncate!(A; cutoff=cutoff)
    end
    A /= (π * N)
    return A
end

# HODC variant: nu coefficients encode both kernel and spectral target directly.
# Call compute_hodc_params once per expansion order, then pass zl, wl here.
function get_ldos_w_from_Tn_hodc(Tn_list, N, ω, zl, wl; eta=1e-2, maxdim=40, cutoff=1e-8)
    nu = get_hodc_weights(ω, N, eta, zl, wl)

    A = Tn_list[1] * nu[1]
    for n in 2:N
        A = +(A, Tn_list[n] * nu[n]; maxdim=maxdim)
        A = ITensorMPS.truncate!(A; cutoff=cutoff)
    end
    return A
end


# ============================================================
# 4. Diagonal-MPS accumulation (mirrors the get_bands / momentum-space pattern)
# ============================================================

"""
    get_ldos_diag_from_Tn(Tn_list, N::Int, ω_vals; kernel=:jackson, lambda=4.0,
                          maxdim=40, cutoff=1e-8)
        -> Vector{Union{Nothing, MPS}}

Compute site-resolved LDOS at every energy in `ω_vals` from a stored Chebyshev
MPO list `Tn_list`, using the same online-accumulation + diagonal-extraction
pattern as `get_bands` in momentum space.

At each Chebyshev step `n`, the diagonal of `Tn_list[n]` is extracted as an MPS
via `extract_diagonal_to_mps` and its KPM-weighted contribution is accumulated
into the LDOS for every energy point simultaneously — avoiding construction and
storage of full weighted MPOs.

Returns a `Vector` of length `length(ω_vals)`.  Each entry is either:
- an `MPS` encoding the site-resolved LDOS `A(r, ω)` at that (rescaled) energy, or
- `nothing` for energies with `|ω| ≥ 1` (outside the rescaled spectral support).

`ω_vals` must be rescaled energies in `(−1, 1)` — convert from physical units with
`E = (ω_phys − center) / scale`.  The legacy per-energy full-MPO path is still
available via `get_ldos_w_from_Tn`.
"""
function get_ldos_diag_from_Tn(Tn_list, N::Int, ω_vals;
                                 kernel::Symbol = :jackson,
                                 lambda::Real   = 4.0,
                                 maxdim::Int    = 40,
                                 cutoff::Real   = 1e-8)
    Nω    = length(ω_vals)
    W     = _kpm_weight_matrix(N, ω_vals; kernel=kernel, lambda=lambda)
    valid = [abs(ω) < 1.0 for ω in ω_vals]

    ldos_accum = Vector{Union{Nothing, MPS}}(nothing, Nω)

    for n in 1:N
        diag_n = ITensorMPS.truncate!(extract_diagonal_to_mps(Tn_list[n]); cutoff=cutoff)
        for iω in 1:Nω
            valid[iω] || continue
            w = W[n, iω]
            iszero(w) && continue
            if ldos_accum[iω] === nothing
                ldos_accum[iω] = w * diag_n
            else
                ldos_accum[iω] = ITensorMPS.truncate!(
                    +(ldos_accum[iω], w * diag_n; maxdim=maxdim); cutoff=cutoff)
            end
        end
    end

    # Normalize: A(r, ω) = [accumulated] / (π² · N · √(1 − ω²))
    for iω in 1:Nω
        valid[iω] && ldos_accum[iω] !== nothing || continue
        ldos_accum[iω] = ITensorMPS.truncate!(
            ldos_accum[iω] / (π^2 * N * sqrt(1 - ω_vals[iω]^2)); cutoff=cutoff)
    end

    return ldos_accum
end
