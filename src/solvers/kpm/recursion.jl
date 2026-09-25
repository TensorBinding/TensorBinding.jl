# solvers/kpm/recursion.jl — Chebyshev recursions of the kernel polynomial method:
# DMRG spectral bounds (_estimate_spectral_bounds, _ensure_scale!), the cached MPO
# and MPS recursions KPM_Tn / KPM_Tn_mps, and the online MPS recursion
# _run_kpm_mps!. Moved verbatim from solvers/KPM_tk.jl (Tier 1 split).

# ============================================================
# Spectral bounds (DMRG estimate of the Chebyshev rescaling)
# ============================================================

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


# ============================================================
# Cached Chebyshev MPO recursion: KPM_Tn
# ============================================================

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
                identity_mpo::Union{MPO,Nothing} = nothing,
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
    I_mpo   = isnothing(identity_mpo) ? MPO(sites, "Id") : copy(identity_mpo)
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
            if k%5 == 0 || k == N+1 # print info every 5 iterations and at the end
                println("Computed T_$((k-1)) with maxlinkdim = ", ITensorMPS.maxlinkdim(T_k))
            end
        end
    end

    return Tn_list, scale, center
end


"""
    KPM_Tn(H::TBHamiltonian, Ncheb; mode=:mpo, psi0=nothing,
           maxdim=40, cutoff=1e-8, dmrg_nsweeps, dmrg_maxdim, dmrg_linkdim)
        -> (Tn_list, scale, center)

High-level Chebyshev expansion for a `TBHamiltonian`.

Lazily determines `H.scale` and `H.center` via DMRG if not already set, builds
the rescaled Chebyshev list, caches the result on `H`, and returns
`(Tn_list, H.scale, H.center)`.

**`mode` keyword**

| `mode` | What is cached | Used by |
|--------|---------------|---------|
| `:mpo` (default) | MPO list `{T_n(H̃)}` in `H._tn_cache` | `get_ldos`, `get_ldos_spectrum`, `get_ldos_spatial` |
| `:mps` | MPS list `{T_n(H̃)|ψ₀⟩}` in `H._tn_mps_cache` | `get_ldos(…; mode=:mps, psi0=…)` |

`mode=:mps` requires `psi0` (a reference MPS).  The MPS pathway is more
memory-efficient when a single reference state is sufficient.

**Overview of the three KPM pathways**

```
Pathway 1 — MPO × MPO cache  [legacy / rarely used]
  KPM_Tn(H, Ncheb; mode=:mpo)           # build and cache {T_n(H̃)} MPOs
  → get_ldos_spectrum(H, ωlist)          # all ω in one pass → Vector{MPS}
  → get_ldos(H, ω; mode=:diag)           # single ω → diagonal site-LDOS MPS
  → get_ldos(H, ω; mode=:mpo)            # single ω → full off-diagonal MPO

  ⚠ Bond dimension χ_T grows at each MPO × MPO step; memory scales as
  O(Ncheb × χ_T²).  Prefer Pathways 3 or 4 unless the cache is reused for
  multiple downstream calls.  Kept mainly for legacy compatibility.

Pathway 2 — MPS cache  [legacy / fixed reference state]
  KPM_Tn(H, Ncheb; mode=:mps, psi0=ψ₀)  # cache {T_n(H̃)|ψ₀⟩} MPS on H
  → get_ldos(H, ω; mode=:mps, psi0=ψ₀)  # μₙ = ⟨ψ₀|T_n(H̃)|ψ₀⟩ → scalar

  Propagates a single reference MPS and stores the full trajectory
  {|φ_n⟩ = T_n(H̃)|ψ₀⟩} for repeated re-use across many energy queries on the
  same state.  Kept for advanced workflows; the public LDOS helpers below avoid
  storing this cache.

Pathway 3 — Online MPO × MPO  [k-space and spatial spectral functions]
  get_bands(H, Ncheb, D, ωlist; …)       # k-resolved A(k,ω), QFT-conjugated
  get_ldos_spatial(H, Ncheb, ωlist; …)   # real-space LDOS heatmap (default mode)

  Runs the MPO × MPO Chebyshev recursion online with no prior KPM_Tn call;
  only 3 MPOs alive at a time (truncated after each step).  Preferred for
  computing band structures and spatial LDOS over many positions simultaneously.

Pathway 4 — Online MPO × MPS  [single-particle default, most memory-efficient]
  get_ldos_online(H, Ncheb, X, ωlist; …) # LDOS at one site, all ω
  get_ldos_spatial(H, Ncheb, ωlist; mode=:mps; …)  # per-position MPS recursion
  get_exciton_ldos_spatial(H, Ncheb, ωlist; …)      # bound-pair exciton LDOS
  get_exciton_ldos(H, X, ωlist; …)                  # one-position wrapper
  get_dos_stochastic(H, Ncheb, ωlist; …) # stochastic trace DOS

  Propagates MPS states rather than full MPOs: only 3 MPS alive per sample/site.
  For single-particle problems this is almost always the best choice —
  memory cost is O(χ_H × χ_ψ) instead of O(χ_T²).  get_dos_stochastic
  applies this with random initial states for a stochastic trace estimate.
```

All pathways share the same KPM kernel and normalization.
Aux-DOF projections (`spin_proj`, `nambu_proj`, `sublat_proj`, …) are supported
in **Pathways 1, 3, and 4**.  Pathway 2 operates on a fixed reference state
and does not expose per-DOF projection keywords.
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
                           identity_mpo = physical_projector(H),
                           maxdim   = maxdim,
                           cutoff   = cutoff,
                           verbose  = verbose)
        H._tn_cache = Tn
    elseif mode == :mps
        psi0 === nothing && error("KPM_Tn with mode=:mps requires the psi0 keyword argument")
        Tn, _, _ = KPM_Tn_mps(H.mpo, Ncheb, psi0, H.sites;
                               scale    = H.scale,
                               center   = H.center,
                               identity_mpo = physical_projector(H),
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


# ============================================================
# Cached Chebyshev MPS recursion: KPM_Tn_mps
# ============================================================

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
                    identity_mpo::Union{MPO,Nothing} = nothing,
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
    I_mpo = isnothing(identity_mpo) ? MPO(sites, "Id") : copy(identity_mpo)
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
                                identity_mpo = physical_projector(H),
                                maxdim    = maxdim,
                                cutoff    = cutoff,
                                verbose   = verbose)
    H._tn_mps_cache = Tn_mps
    H._tn_Ncheb     = N
    return Tn_mps, H.scale, H.center
end


# ============================================================
# Online MPS Chebyshev recursion (shared by the LDOS, DOS, exciton and band solvers)
# ============================================================

"""
    _run_kpm_mps!(Ham_n, psi0, Ncheb, W, valid, accum;
                  weight=1.0, cutoff=1e-8, maxdim=100,
                  verbose=false, label="") -> Int

Online MPS Chebyshev KPM recursion.  Computes `μ_n = ⟨psi0|T_n(Ham_n)|psi0⟩`
for n = 1…Ncheb and accumulates `W[n,iω] × μ_n × weight` into `accum[iω]`
for each valid energy index.  Returns the `maxlinkdim` of the final state.
"""
function _run_kpm_mps!(Ham_n::MPO, psi0::MPS, Ncheb::Int,
                        W::Matrix{Float64}, valid::Vector{Bool},
                        accum::Vector{Float64};
                        weight::Float64 = 1.0,
                        cutoff::Real    = 1e-8,
                        maxdim::Int     = 100,
                        verbose::Bool   = false,
                        label::String   = "")
    Nω = length(valid)
    function kpm_step!(phi, n)
        mu = real(inner(psi0, phi))
        for iω in 1:Nω
            valid[iω] || continue
            accum[iω] += W[n, iω] * mu * weight
        end
    end
    phi_km2 = psi0
    phi_km1 = apply(Ham_n, psi0; cutoff=cutoff, maxdim=maxdim)
    kpm_step!(phi_km2, 1)
    kpm_step!(phi_km1, 2)
    for k in 3:Ncheb
        phi_k = +(2 * apply(Ham_n, phi_km1; cutoff=cutoff, maxdim=maxdim),
                  -phi_km2; cutoff=cutoff, maxdim=maxdim)
        kpm_step!(phi_k, k)
        phi_km2 = phi_km1
        phi_km1 = phi_k
        verbose && (k % 10 == 0 || k == Ncheb) &&
            println(label, " step $k/$Ncheb  maxlinkdim=$(maxlinkdim(phi_km1))")
    end
    return maxlinkdim(phi_km1)
end
