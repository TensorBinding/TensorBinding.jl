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
    get_ldos(H::TBHamiltonian, ω_phys; mode, psi0, kernel, lambda, eta, m_order,
             maxdim, cutoff, zl, wl)

Compute the local density of states at physical energy `ω_phys` using the
Chebyshev expansion cached in `H` by a prior `KPM_Tn` or `KPM_Tn_mps` call.

**Modes**

- `:mpo`  (legacy) — calls `get_ldos_w_from_Tn` and returns a full spectral-weight
  **MPO** at `ω_phys`.  Requires `KPM_Tn(H, N; mode=:mpo)`.  Retains off-diagonal
  information; use when spatial correlations are needed.

- `:diag` — calls `get_ldos_diag_from_Tn` and returns an **MPS** encoding only the
  diagonal `A(r, ω_phys)`.  Requires `KPM_Tn(H, N; mode=:mpo)`.  Much cheaper than
  `:mpo` for typical LDOS use-cases; mirrors the `get_bands` momentum-space pattern.
  Use `get_ldos_spectrum` to compute all energies in a single pass.
  Use `get_ldos_online` to avoid storing the Tn cache entirely.

- `:mps`  — computes moments `μₙ = ⟨ψ₀|φₙ⟩` from the MPS Chebyshev cache and
  calls `get_ldos_from_mun`, returning a **Real**.  Requires
  `KPM_Tn(H, N; mode=:mps, psi0=...)`.

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
    get_ldos_spectrum(H::TBHamiltonian, ω_phys_vals; kernel, lambda, maxdim, cutoff)
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

# Evaluate LDOS at site x=16 for each energy:
ldos_at_16 = [l === nothing ? 0.0 : _eval_diag_mps(l, 15) for l in ldos_vec]
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


# ─────────────────────────────────────────────────────────────────────────────
# Shared KPM helpers  (used by get_ldos_online, get_ldos_spatial, get_dos_stochastic)
# ─────────────────────────────────────────────────────────────────────────────

"""
    _aux_setup(H, nambu_proj, proj_nambu, spin_proj, proj_s,
               layer_proj, proj_layer, sublat_proj, proj_sl) -> NamedTuple

Detect all auxiliary DOF indices from `H` and compute sector iteration ranges.
Returns a NamedTuple with fields:
  `nambu_s_det`, `nambu_side_det`, `spin_s_det`,
  `layer_s_det`, `layer_side_det`, `sublat_s_det`, `sublat_side_det`,
  `nambu_range`, `spin_range`, `layer_range`, `sl_range`, `any_aux_proj`.
"""
function _aux_setup(H::TBHamiltonian,
                    nambu_proj::Bool, proj_nambu,
                    spin_proj::Bool,  proj_s,
                    layer_proj::Bool, proj_layer,
                    sublat_proj::Bool, proj_sl)
    nambu_s_det,  nambu_side_det  = !isnothing(H.nambu_s)      ? aux_site(H, :nambu)      : (nothing, :pre)
    spin_s_det                    = H.spin_s
    layer_s_det,  layer_side_det  = !isnothing(H.layer_s)      ? aux_site(H, :layer)      : (nothing, :pre)
    sublat_s_det, sublat_side_det = !isnothing(H.sublattice_s) ? aux_site(H, :sublattice) : (nothing, :post)

    nambu_range = (nambu_proj && !isnothing(nambu_s_det)) ?
        (isnothing(proj_nambu) ? (1:dim(nambu_s_det::Index)) : (proj_nambu:proj_nambu)) : (1:1)
    spin_range  = (spin_proj  && !isnothing(spin_s_det)) ?
        (isnothing(proj_s)     ? (1:2)                        : (proj_s:proj_s))         : (1:1)
    layer_range = (layer_proj && !isnothing(layer_s_det)) ?
        (isnothing(proj_layer) ? (1:dim(layer_s_det::Index))  : (proj_layer:proj_layer))  : (1:1)
    sl_range    = (sublat_proj && !isnothing(sublat_s_det)) ?
        (isnothing(proj_sl)    ? (1:dim(sublat_s_det::Index)) : (proj_sl:proj_sl))        : (1:1)
    any_aux_proj = nambu_proj || spin_proj || layer_proj || sublat_proj

    return (; nambu_s_det, nambu_side_det, spin_s_det,
              layer_s_det, layer_side_det, sublat_s_det, sublat_side_det,
              nambu_range, spin_range, layer_range, sl_range, any_aux_proj)
end


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


"""
    get_ldos_online(H::TBHamiltonian, Ncheb::Int, X::Int, ω_phys_vals;
                    kernel, lambda, maxdim, cutoff, verbose,
                    nambu_proj, proj_nambu, spin_proj, proj_s,
                    layer_proj, proj_layer, sublat_proj, proj_sl)
        -> Vector{Float64}

Online real-space LDOS at unit-cell position `X` for all physical energies in
`ω_phys_vals`.  Never stores more than **3 MPS** simultaneously (no Chebyshev cache).

**Algorithm**: MPS Chebyshev recursion
`|φ_k⟩ = T_k(H̃)|X⟩` with moment accumulation `μ_k = ⟨X|φ_k⟩`.
Auxiliary DOF sectors are summed by running the recursion once per requested sector.

`X ∈ {1, …, H.N}` is the 1-indexed unit-cell position.

**Auxiliary DOF projections** (same interface as `get_bands` and `get_ldos_spatial`):

- `spin_proj`, `nambu_proj`, `layer_proj`, `sublat_proj` — enable projection of the
  corresponding auxiliary DOF auto-detected from `H`.
- `proj_s`, `proj_nambu`, `proj_layer`, `proj_sl` — sector selector: `nothing` sums
  all sectors of that DOF; an integer selects a single sector (1-based).
- Contributions from all requested sectors are accumulated into a single result vector.

Returns `Vector{Float64}` of length `Nω` with `0.0` outside the spectral support.

Examples
--------
```julia
ωlist = range(-3.0, 3.0; length=300)
ldos  = get_ldos_online(H, 200, 2^(H.L-1), ωlist)          # no aux DOF

# Spin-summed LDOS at site 16
ldos_tot = get_ldos_online(H_spin, 200, 16, ωlist; spin_proj=true)

# Spin-↑ LDOS only
ldos_up  = get_ldos_online(H_spin, 200, 16, ωlist; spin_proj=true, proj_s=1)
```
"""
function get_ldos_online(H::TBHamiltonian, Ncheb::Int, X::Int, ω_phys_vals;
                          kernel::Symbol = :jackson,
                          lambda::Real   = 4.0,
                          maxdim::Int    = 100,
                          cutoff::Real   = 1e-8,
                          verbose::Bool  = false,
                          # Auxiliary DOF projections — same interface as get_bands:
                          nambu_proj::Bool  = false,
                          proj_nambu        = nothing,
                          spin_proj::Bool   = false,
                          proj_s            = nothing,
                          layer_proj::Bool  = false,
                          proj_layer        = nothing,
                          sublat_proj::Bool = false,
                          proj_sl           = nothing)
    _ensure_scale!(H)
    nambu_proj, spin_proj, layer_proj, sublat_proj =
        _autoenable_proj(H, nambu_proj, spin_proj, layer_proj, sublat_proj)

    I_mpo = MPO(H.sites, "Id")
    Ham_n = (1 / H.scale) * +(H.mpo, (-H.center) * I_mpo; cutoff=cutoff)

    ω_vals = (collect(ω_phys_vals) .- H.center) ./ H.scale
    Nω     = length(ω_vals)
    W      = _kpm_weight_matrix(Ncheb, ω_vals; kernel=kernel, lambda=lambda)
    valid  = [abs(ω) < 1.0 for ω in ω_vals]
    accum  = zeros(Float64, Nω)

    (; nambu_range, spin_range, layer_range, sl_range, any_aux_proj) =
        _aux_setup(H, nambu_proj, proj_nambu, spin_proj, proj_s,
                      layer_proj, proj_layer, sublat_proj, proj_sl)
    L_tot = length(H.sites)

    # ── MPS-based Chebyshev recursion, summed over requested aux sectors ──────
    for σ_n in nambu_range, σ_s in spin_range, σ_l in layer_range, σ_sl in sl_range
        psi0 = any_aux_proj ?
               _ldos_make_psi0(H, X, σ_n, σ_s, σ_l, σ_sl) :
               (L_tot == H.L ? binary_to_MPS(X - 1, H.L, H.sites) :
                               mpsexciton(X, H.sites))
        _run_kpm_mps!(Ham_n, psi0, Ncheb, W, valid, accum;
                      cutoff=cutoff, maxdim=maxdim,
                      verbose=verbose, label="get_ldos_online")
    end  # sector loop

    result = zeros(Float64, Nω)
    for iω in 1:Nω
        valid[iω] || continue
        result[iω] = accum[iω] / (π^2 * Ncheb * sqrt(1 - ω_vals[iω]^2))
    end
    return result
end


# ─────────────────────────────────────────────────────────────────────────────
# Auxiliary DOF helpers for LDOS
# ─────────────────────────────────────────────────────────────────────────────

"""
    _ldos_make_psi0(H, x, σ_n, σ_s, σ_l, σ_sl) -> MPS

Product-state MPS over all `H.sites` for KPM evaluation in LDOS `:mps` mode.

- Position sites encode `x-1` in big-endian binary (first position site = MSB).
- Auxiliary sites are set to 1-based sector indices:
  `σ_n` (nambu), `σ_s` (spin), `σ_l` (layer), `σ_sl` (sublattice).
  Indices for absent aux dofs are ignored.
"""
function _ldos_make_psi0(H::TBHamiltonian, x::Int,
                          σ_n::Int, σ_s::Int, σ_l::Int, σ_sl::Int)
    k       = 0
    pos_bit = H.L - 1   # bit index: MSB of (x-1) goes to the first position site
    for s in H.sites
        k *= dim(s)
        if     !isnothing(H.nambu_s)      && s == H.nambu_s;      k += σ_n  - 1
        elseif !isnothing(H.spin_s)       && s == H.spin_s;       k += σ_s  - 1
        elseif !isnothing(H.layer_s)      && s == H.layer_s;      k += σ_l  - 1
        elseif !isnothing(H.sublattice_s) && s == H.sublattice_s; k += σ_sl - 1
        else
            k += (x - 1) >> pos_bit & 1
            pos_bit -= 1
        end
    end
    return _basis_state_mps(k, H.sites)
end


# ─────────────────────────────────────────────────────────────────────────────
# Spatial LDOS at multiple x-positions (real-space analogue of get_bands)
# ─────────────────────────────────────────────────────────────────────────────

"""
    get_ldos_spatial(H, Ncheb, ω_phys_vals;
                     num_x, num_avg, mode, x_start, x_end, x_groups,
                     kernel, lambda, maxdim, cutoff, verbose,
                     nambu_proj, proj_nambu, spin_proj, proj_s,
                     layer_proj, proj_layer, sublat_proj, proj_sl)
        -> Matrix{Float64}

Spatially-resolved LDOS, real-space analogue of `get_bands`.

**Sampling procedures (`reduce`)** — full detail in [`spatial_sampling_plan`](@ref).

- `:point` (default) — read the LDOS *at* `num_x[×num_y]` sample cells; with
  `box_half > 0` each pixel is the mean over a `(2·box_half+1)²` box. Cheap, but
  a grid coarser than a feature's width **aliases** it (thin in-gap edge /
  domain-wall channels can fall between pixels and be missed).
- `:block` — partition the system into `num_x × num_y` blocks (powers of two) and
  report the **integral** over each block, computed by tracing out the
  within-block position bits (a partial contraction, cost independent of block
  size). Gap-free: every cell belongs to one block, so a thin feature on a gapped
  background **cannot** be missed. Use it for large-scale maps of edge networks.
  Output columns are row-major over coarse pixels (`col = ixp + iyp·num_x + 1`);
  block centres come from the plan. `:mpo` mode only.

**Return shape (sublattice geometry-awareness)**

For a multi-atom unit cell (`H.sublattice_s` set) the layout depends on the
*sampling scale*, decided by [`spatial_sampling_plan`](@ref) from the local
stride (see `sublattice` below):

- **resolved** (atomic scale — every unit cell probed, or `proj_sl=k`):
  `(Nω × ng×n_sub)`, columns interleaved in atom order matching the
  `*_positions` functions — `[A₀, B₀, A₁, B₁, …]` (2-sublattice),
  `[A₀, B₀, C₀, …]` (3-sublattice). `proj_sl=k` fills only sublattice `k`.
- **averaged** (large scale — the grid skips unit cells): `(Nω × ng)`, the
  sublattice is traced out into one value per unit cell (mean over the `n_sub`
  atoms).

With no sublattice DOF the shape is always `(Nω × ng)`, `ng = num_x`.

**`sublattice` (resolve vs average)**

- `:auto` (default) — resolve when sampling at full unit-cell resolution
  (`stride == 1`, e.g. zooming a small window and probing every cell); average
  whenever the grid is coarser or `box_half > 0`.
- `:resolve` — always emit per-atom columns. `:average` — always trace the
  sublattice to one value per cell. `proj_sl=k` always resolves that one atom.

**Sampling parameters**

- `num_x`/`num_y` : sample counts (per axis for `grid=true`; `num_x` is the total
  for the default 1D linear sweep). Default `H.N` = full resolution.
- `num_avg`  : sub-samples per coarse block for local averaging (1D, default 1).
- `x_start`, `x_end` : 1-indexed linear position range (1D layout).
- `grid`     : `true` lays centers on a 2D `num_x × num_y` unit-cell grid.
- `xwin`, `ywin` : 0-indexed unit-cell `(lo, hi)` windows for `grid=true` (e.g.
  zoom into a patch of a large system).
- `x_groups` : explicit `Vector{Vector{Int}}` override (treated as atomic).
- `box_half` : 2D neighbourhood half-width (averages, forces sublattice averaging).
- `reduce`   : `:point` (sample/box) or `:block` (block-integrate; see above).

**Modes**

- `:mpo` (default) — single Chebyshev pass; evaluates all positions simultaneously.
  Cost `∝ Ncheb × (MPO×MPO)`, independent of `num_x` or `n_sub`.
- `:mps` — independent MPS recursion per (position, sector) combination.

**Other auxiliary DOF projections** (same interface as `get_bands`):
`nambu_proj`/`proj_nambu`, `spin_proj`/`proj_s`, `layer_proj`/`proj_layer`.

Examples
--------
```julia
# Standard 1D chain — shape (Nω × 8)
ldos = get_ldos_spatial(H, 200, ωlist; num_x=8)

# Honeycomb, large-scale map — sublattice averaged, shape (Nω × 64)
ldos_uc  = get_ldos_spatial(H_hc, 200, ωlist; num_x=64)

# Honeycomb, atomic zoom into a 50×50 patch — sublattice resolved (Nω × 50*50*2)
ldos_zoom = get_ldos_spatial(H_hc, 200, ωlist; grid=true,
                             xwin=(1000, 1049), ywin=(1000, 1049))

# Large 2^14×2^14 system: 128×128 block-integrated map — catches thin in-gap
# edge channels that point sampling would alias away. Shape (Nω × 128*128).
ldos_blk = get_ldos_spatial(H_big, 200, ωlist; reduce=:block, num_x=128, num_y=128)

# Kagome: sublattice A only — only A columns filled, B/C columns zero
ldos_A   = get_ldos_spatial(H_kg, 200, ωlist; proj_sl=1, num_x=H_kg.N)
```
"""
function get_ldos_spatial(H::TBHamiltonian, Ncheb::Int, ω_phys_vals;
                           num_x::Int     = H.N,
                           num_y          = nothing,
                           num_avg::Int   = 1,
                           mode::Symbol   = :mpo,
                           x_start::Int   = 1,
                           x_end::Int     = H.N,
                           x_groups       = nothing,
                           grid::Bool     = false,
                           xwin           = nothing,
                           ywin           = nothing,
                           box_half::Int  = 0,
                           reduce::Symbol = :point,
                           sublattice::Symbol = :auto,
                           kernel::Symbol = :jackson,
                           lambda::Real   = 4.0,
                           maxdim::Int    = 100,
                           cutoff::Real   = 1e-8,
                           verbose::Bool  = false,
                           nambu_proj::Bool  = false,
                           proj_nambu        = nothing,
                           spin_proj::Bool   = false,
                           proj_s            = nothing,
                           layer_proj::Bool  = false,
                           proj_layer        = nothing,
                           sublat_proj::Bool = false,   # kept for backward compat; auto-on when H.sublattice_s is set
                           proj_sl           = nothing)

    # ── Geometry-aware sampling plan (unit-cell groups + sublattice decision) ──
    if box_half > 0 || grid || xwin !== nothing || ywin !== nothing || reduce === :block
        isnothing(H.geometry) &&
            error("get_ldos_spatial: box_half/grid/window/block sampling requires H.geometry to be set.")
        length(H.geometry(1)) == 2 ||
            error("get_ldos_spatial: box_half/grid/window/block sampling is only supported for 2D systems.")
    end
    reduce === :block && mode === :mps &&
        error("get_ldos_spatial: reduce=:block is only supported in mode=:mpo.")
    Lx_uc   = something(H.Lx, H.L ÷ 2)
    Ly_uc   = H.L - Lx_uc
    n_sub_H = isnothing(H.sublattice_s) ? 1 : dim(H.sublattice_s)
    plan = spatial_sampling_plan(H.L;
        Lx       = Lx_uc,
        grid     = grid,
        reduce   = reduce,
        n_sub    = n_sub_H,
        num_x    = num_x, num_y = num_y, num_avg = num_avg,
        x_start  = x_start, x_end = x_end,
        xwin     = xwin, ywin = ywin,
        x_groups = x_groups, box_half = box_half,
        sublattice = sublattice)
    groups   = plan.groups
    is_block = plan.reduce === :block
    block_a  = plan.a
    block_b  = plan.b
    nbx      = 2^block_a

    _ensure_scale!(H)
    nambu_proj, spin_proj, layer_proj, sublat_proj =
        _autoenable_proj(H, nambu_proj, spin_proj, layer_proj, sublat_proj)

    (; nambu_s_det, nambu_side_det, spin_s_det,
       layer_s_det, layer_side_det, sublat_s_det, sublat_side_det,
       nambu_range, spin_range, layer_range, any_aux_proj) =
        _aux_setup(H, nambu_proj, proj_nambu, spin_proj, proj_s,
                      layer_proj, proj_layer, sublat_proj, proj_sl)

    # ── Bernal top-view guard ─────────────────────────────────────────────────
    if layer_proj && isnothing(proj_layer) && !isnothing(sublat_s_det)
        n_lay = dim(layer_s_det::Index)
        @warn """get_ldos_spatial: proj_layer=nothing on a layered+sublattice Hamiltonian.
  Result accumulates sublattice columns by label across all $n_lay layers.
  For Bernal stacking this is NOT the physical top-view: even layers have their
  sublattice-A/B registries physically swapped in 2D, so the sum is misleading.
  This call is also $(n_lay)× slower than a single-layer call.
  For a correct Bernal top-view use:
    ldos_layers = [get_ldos_spatial(H, Nc, ωlist; proj_layer=k, ...) for k in 1:$n_lay]
    plot_ldos_multilayer(ldos_layers, ωlist, ω; stacking=:Bernal, ...)
  For AA stacking the label-wise sum is physically correct (this warning fires
  regardless of stacking type; shown only once per session).""" maxlog=1
    end

    # ── Sublattice layout ─────────────────────────────────────────────────────
    # When H.sublattice_s is set the geometry-aware plan decides the layout:
    #   • resolve (atomic scale, or proj_sl=k): one column per atom,
    #       shape (Nω, ng × n_sub),  col = (ig-1)*n_sub + s   (matches the
    #       honeycomb/kagome/lieb positions-function atom ordering; proj_sl=k
    #       fills only sublattice k, others stay 0).
    #   • average (large scale): the sublattice is traced out — one value per
    #       unit cell, shape (Nω, ng),  col = ig  (mean over the n_sub atoms).
    has_sublat   = !isnothing(sublat_s_det)
    n_sub        = has_sublat ? dim(sublat_s_det::Index) : 1
    # proj_sl=k pins a single sublattice → always resolved (that one column).
    resolve_sl   = has_sublat && (plan.resolve_sublattice || !isnothing(proj_sl))
    average_sl   = has_sublat && !resolve_sl
    sl_fill      = has_sublat ?
        (isnothing(proj_sl) ? (1:n_sub) : (proj_sl:proj_sl)) :
        (1:1)

    I_mpo = MPO(H.sites, "Id")
    Ham_n = (1 / H.scale) * +(H.mpo, (-H.center) * I_mpo; cutoff=cutoff)

    ω_vals = (collect(ω_phys_vals) .- H.center) ./ H.scale
    Nω     = length(ω_vals)
    W      = _kpm_weight_matrix(Ncheb, ω_vals; kernel=kernel, lambda=lambda)
    valid  = [abs(ω) < 1.0 for ω in ω_vals]

    ng     = length(groups)
    # Averaging collapses the n_sub atoms into one column per group.
    n_cols = average_sl ? ng : ng * n_sub
    result = zeros(Float64, Nω, n_cols)
    L_tot  = length(H.sites)

    if mode == :mps
        # ── MPS mode ──────────────────────────────────────────────────────────
        n_total = sum(length(g) for g in groups)
        n_done  = 0

        for (ig, grp) in enumerate(groups)
            grp_accum = zeros(Float64, Nω, n_sub)  # per-sublattice accumulator

            for x in grp
                for σ_n in nambu_range, σ_s in spin_range, σ_l in layer_range,
                        σ_sl in sl_fill
                    psi0 = any_aux_proj ?
                           _ldos_make_psi0(H, x, σ_n, σ_s, σ_l, σ_sl) :
                           (L_tot == H.L ? binary_to_MPS(x - 1, H.L, H.sites) :
                                           mpsexciton(x, H.sites))
                    accum_loc = zeros(Float64, Nω)
                    _run_kpm_mps!(Ham_n, psi0, Ncheb, W, valid, accum_loc;
                                  cutoff=cutoff, maxdim=maxdim)

                    for iω in 1:Nω
                        valid[iω] || continue
                        grp_accum[iω, σ_sl] += accum_loc[iω] / (π^2 * Ncheb * sqrt(1 - ω_vals[iω]^2))
                    end
                end  # sector loop
            end  # x

            if average_sl
                # Trace out the sublattice: one column per unit cell (mean atom).
                result[:, ig] = vec(sum(grp_accum; dims=2)) ./ (n_sub * length(grp))
            else
                for s in sl_fill
                    result[:, (ig-1)*n_sub + s] = grp_accum[:, s] ./ length(grp)
                end
            end

            n_done += length(grp)
            verbose && n_done % 15 == 0 &&
                println("get_ldos_spatial [:mps]  group $ig/$ng  ($n_done/$n_total)")
        end

    elseif mode == :mpo
        # ── MPO mode: single online Chebyshev pass ────────────────────────────
        all_xs = unique(vcat(groups...))

        # Build position-only eval states (drop sublat + any other projected aux)
        aux_to_drop = Set{Index}()
        (nambu_proj && !isnothing(nambu_s_det)) && push!(aux_to_drop, nambu_s_det::Index)
        (spin_proj  && !isnothing(spin_s_det))  && push!(aux_to_drop, spin_s_det::Index)
        (layer_proj && !isnothing(layer_s_det)) && push!(aux_to_drop, layer_s_det::Index)
        has_sublat                              && push!(aux_to_drop, sublat_s_det::Index)
        pos_sites = filter(s -> s ∉ aux_to_drop, H.sites)

        psi_dict = if isempty(aux_to_drop)
            Dict(x => (L_tot == H.L ? binary_to_MPS(x - 1, H.L, H.sites) :
                                      mpsexciton(x, H.sites)) for x in all_xs)
        else
            @assert length(pos_sites) == H.L "get_ldos_spatial: $(length(pos_sites)) position sites after dropping aux but expected H.L=$(H.L)."
            Dict(x => binary_to_MPS(x - 1, H.L, pos_sites) for x in all_xs)
        end

        accum = zeros(Float64, Nω, n_cols)

        local _nambu_side  = nambu_side_det
        local _layer_side  = layer_side_det
        local _sublat_side = sublat_side_det

        # Reduce a diagonal profile MPS to per-pixel scalars, returning (u, value)
        # pairs where u is the 1-indexed output pixel (column unit):
        #   reduce=:point → mean of inner products over each group's cells,
        #   reduce=:block → integral over each coarse block (_eval_block_mps).
        function spatial_vals_cpu(diag_n)
            if is_block
                return [(ixp + iyp * nbx + 1,
                         _eval_block_mps(diag_n, ixp, iyp, block_a, block_b, Lx_uc, Ly_uc))
                        for iyp in 0:(2^block_b - 1) for ixp in 0:(nbx - 1)]
            else
                return [(ig, sum(real(inner(psi_dict[x], diag_n)) for x in grp) / length(grp))
                        for (ig, grp) in enumerate(groups)]
            end
        end

        function accumulate_Tn!(Tk, n)
            # Non-sublattice projections (nambu → spin → layer)
            after_nambu = nambu_proj ?
                [project_aux(Tk, nambu_s_det::Index, sec; side=_nambu_side)
                 for sec in (isnothing(proj_nambu) ? (1:2) : (proj_nambu:proj_nambu))] :
                MPO[Tk]

            after_spin = spin_proj ?
                [project_aux(T, spin_s_det::Index, sec; side=:pre)
                 for T in after_nambu, sec in (isnothing(proj_s) ? (1:2) : (proj_s:proj_s))] :
                after_nambu

            after_layer = if layer_proj
                n_lay     = dim(layer_s_det::Index)
                lay_range = isnothing(proj_layer) ? (1:n_lay) : (proj_layer:proj_layer)
                [project_aux(T, layer_s_det::Index, sec; side=_layer_side)
                 for T in after_spin for sec in lay_range]
            else
                after_spin
            end

            if has_sublat
                # Project per sublattice sector. Resolved → each sector gets its
                # own column; averaged (large scale) → all sectors fold into the
                # single per-pixel column u (mean over the n_sub atoms).
                for Tl in after_layer, s in sl_fill
                    Tp     = project_aux(Tl, sublat_s_det::Index, s; side=_sublat_side)
                    diag_n = ITensorMPS.truncate!(extract_diagonal_to_mps(Tp); cutoff=cutoff)
                    scale  = average_sl ? 1.0 / n_sub : 1.0
                    for (u, val) in spatial_vals_cpu(diag_n)
                        col = average_sl ? u : (u - 1) * n_sub + s
                        for iω in 1:Nω
                            valid[iω] || continue
                            accum[iω, col] += W[n, iω] * val * scale
                        end
                    end
                end
            else
                # No sublattice: one column per pixel (original behavior)
                for Tp in after_layer
                    diag_n = ITensorMPS.truncate!(extract_diagonal_to_mps(Tp); cutoff=cutoff)
                    for (u, val) in spatial_vals_cpu(diag_n)
                        for iω in 1:Nω
                            valid[iω] || continue
                            accum[iω, u] += W[n, iω] * val
                        end
                    end
                end
            end
        end

        Tkm2 = I_mpo;  Tkm1 = Ham_n
        accumulate_Tn!(Tkm2, 1);  accumulate_Tn!(Tkm1, 2)

        for k in 3:Ncheb
            Tk   = +(2 * apply(Ham_n, Tkm1; cutoff=cutoff), -Tkm2; maxdim=maxdim)
            Tk   = ITensorMPS.truncate!(Tk; cutoff=cutoff)
            accumulate_Tn!(Tk, k)
            Tkm2 = Tkm1;  Tkm1 = Tk
            verbose && (k % 15 == 0 || k == Ncheb) &&
                println("get_ldos_spatial [:mpo]  step $k/$Ncheb  " *
                        "maxlinkdim=$(maxlinkdim(Tkm1))")
        end

        for iω in 1:Nω
            valid[iω] || continue
            result[iω, :] = accum[iω, :] ./ (π^2 * Ncheb * sqrt(1 - ω_vals[iω]^2))
        end

    else
        error("Unknown mode :$mode. Choose :mps or :mpo")
    end

    return result
end


# ─────────────────────────────────────────────────────────────────────────────
# Stochastic full DOS (trace estimation via random diagonal sampling)
# ─────────────────────────────────────────────────────────────────────────────

"""
    get_dos_stochastic(H::TBHamiltonian, Ncheb::Int, ω_phys_vals;
                       N_sample, N_bound, seed, normalize, dos_weighting,
                       kernel, lambda, eta, m_order, maxdim, cutoff, verbose,
                       nambu_proj, proj_nambu, spin_proj, proj_s,
                       layer_proj, proj_layer, sublat_proj, proj_sl)
        -> Vector{Float64}

Stochastic full DOS via random trace estimation (MPS Chebyshev, 3 MPS per sample).

**Normalization**

- `dos_weighting=:trace` (default): returns the trace DOS. With
  `normalize=false` this is the total spectral weight `Tr[δ(ω-H)]`; with
  `normalize=true` it is divided by the traced Hilbert-space dimension.
- `dos_weighting=:sample`: returns the unweighted sample signal. For exciton
  stratified runs this is `avg_full + avg_bound` (when `N_bound > 0`), with no
  phase-space factor multiplying the continuum. This is intended for
  visualising the bound peak; `normalize` is ignored in this mode.

**Auxiliary DOF projections**

Unlike `get_ldos_spatial`, projections are **not** auto-enabled here.  The
default is full Hilbert-space sampling over all D states — always correct and
cheapest.  Projections must be requested explicitly:

- `layer_proj=true, proj_layer=k` — DOS on layer k only.
- `sublat_proj=true, proj_sl=k` — DOS on sublattice k only.
- Combining multiple `*_proj=true` flags is supported.

When any flag is set the function samples from position-basis states with fixed
aux sectors (`N_phys` effective states), which is `n_sectors×` slower than the
default.  Only use projections when you actually need a sector-resolved DOS.

**Exciton stratification** (no aux projections)

For exciton Hamiltonians (`length(H.sites) == 2*H.L`), stratified sampling
dedicates `N_bound` samples to the bound sector `|x,x⟩` and `N_sample` to the
full Hilbert space, combining with proper weights:
  `DOS = N_phys × avg_bound + (D − N_phys) × avg_scatter`.
`N_bound = 0` (default) = uniform sampling over all D states.
Set `dos_weighting=:sample` to inspect the sampled spectral signal before these
sector-size weights are applied.

**Reconstruction kernel**

`kernel=:hodc` uses the Higher-Order Delta Chebyshev contour reconstruction
(`eta`, `m_order` control it; `eta=0` → `1/(Ncheb+1)`), whose weights already
carry the full KPM normalisation.  Other values are convolution kernels
(`:jackson` default, `:lorentz` with `lambda`, `:fejer`, `:dirichlet`).

Examples
--------
```julia
# Per-state DOS, size-independent
dos = get_dos_stochastic(H, 100, ωlist; N_sample=50, normalize=true)

# Sublattice-resolved per-state DOS (kagome)
dos_A = get_dos_stochastic(H_kg, 100, ωlist; sublat_proj=true, proj_sl=1,
                            N_sample=50, normalize=true)

# BdG: particle + hole combined per-state DOS
dos   = get_dos_stochastic(H_bdg, 100, ωlist; nambu_proj=true,
                            N_sample=50, normalize=true)
```
"""
function get_dos_stochastic(H::TBHamiltonian, Ncheb::Int, ω_phys_vals;
                             N_sample::Int            = 50,
                             N_bound::Int             = 0,
                             seed::Union{Int,Nothing} = 42,
                             normalize::Bool          = false,
                             dos_weighting::Symbol    = :trace,
                             kernel::Symbol           = :jackson,
                             lambda::Real             = 4.0,
                             eta::Real                = 0.0,
                             m_order::Int             = 4,
                             maxdim::Int              = 100,
                             cutoff::Real             = 1e-8,
                             verbose::Bool            = false,
                             # Auxiliary DOF projections — same interface as get_bands:
                             nambu_proj::Bool  = false,
                             proj_nambu        = nothing,
                             spin_proj::Bool   = false,
                             proj_s            = nothing,
                             layer_proj::Bool  = false,
                             proj_layer        = nothing,
                             sublat_proj::Bool = false,
                             proj_sl           = nothing)
    _ensure_scale!(H)
    dos_weighting in (:trace, :sample) ||
        error("get_dos_stochastic: dos_weighting must be :trace or :sample.")

    I_mpo = MPO(H.sites, "Id")
    Ham_n = (1 / H.scale) * +(H.mpo, (-H.center) * I_mpo; cutoff=cutoff)

    D      = prod(ITensors.dim(s) for s in H.sites)
    N_phys = H.N
    is_exc = length(H.sites) == 2 * H.L

    (; nambu_range, spin_range, layer_range, sl_range, any_aux_proj) =
        _aux_setup(H, nambu_proj, proj_nambu, spin_proj, proj_s,
                      layer_proj, proj_layer, sublat_proj, proj_sl)

    ω_vals = (collect(ω_phys_vals) .- H.center) ./ H.scale
    Nω     = length(ω_vals)
    W, denom = _dos_weight_matrix(Ncheb, ω_vals;
                                  kernel=kernel, lambda=lambda, eta=eta, m_order=m_order)
    valid  = [abs(ω) < 1.0 for ω in ω_vals]

    rng         = seed === nothing ? Random.default_rng() : Random.MersenneTwister(seed)
    accum_full  = zeros(Float64, Nω)
    accum_bound = zeros(Float64, Nω)

    if any_aux_proj
        # ── Projected DOS: sample position states with fixed aux sectors ─────
        # Trace over position basis only, with aux dofs projected to selected
        # sectors.  Effective dimension = N_phys × n_sectors.
        # D_eff = N_phys: the sector loop already sums all sectors into accum_full,
        # so only the position average (× N_phys) is needed for normalisation.
        D_eff = N_phys

        xs = rand(rng, 1:N_phys, N_sample)
        for (i, x) in enumerate(xs)
            for σ_n in nambu_range, σ_s in spin_range, σ_l in layer_range, σ_sl in sl_range
                psi0 = _ldos_make_psi0(H, x, σ_n, σ_s, σ_l, σ_sl)
                χ = _run_kpm_mps!(Ham_n, psi0, Ncheb, W, valid, accum_full;
                                   weight=1.0/N_sample, cutoff=cutoff, maxdim=maxdim)
                verbose && i % 15 == 0 && σ_n == first(nambu_range) &&
                    σ_s == first(spin_range) && σ_l == first(layer_range) &&
                    σ_sl == first(sl_range) &&
                    println("Projected DOS sample $i/$N_sample  maxlinkdim=$χ")
            end
        end

        result = zeros(Float64, Nω)
        for iω in 1:Nω
            valid[iω] || continue
            if dos_weighting == :sample
                result[iω] = accum_full[iω] / denom[iω]
            else
                result[iω] = D_eff * accum_full[iω] / denom[iω]
            end
        end
        normalize && dos_weighting == :trace && (result ./= D_eff)
        return result
    end

    # ── Full Hilbert space samples (weight = D / N_sample per sample) ─────────
    samples = rand(rng, 0:(D - 1), N_sample)
    for (i, k) in enumerate(samples)
        psi0 = _basis_state_mps(k, H.sites)
        χ = _run_kpm_mps!(Ham_n, psi0, Ncheb, W, valid, accum_full;
                           weight=1.0/N_sample, cutoff=cutoff, maxdim=maxdim)
        verbose && i % 15 == 0 && println("Full sample $i/$N_sample  maxlinkdim=$χ")
    end

    # ── Bound-sector samples (exciton: random |x,x⟩, weight = N_phys/N_bound) ─
    if N_bound > 0 && is_exc
        xs = rand(rng, 1:N_phys, N_bound)
        for (i, x) in enumerate(xs)
            psi0 = mpsexciton(x, H.sites)
            χ = _run_kpm_mps!(Ham_n, psi0, Ncheb, W, valid, accum_bound;
                               weight=1.0/N_bound, cutoff=cutoff, maxdim=maxdim)
            verbose && i % 15 == 0 && println("Bound sample $i/$N_bound  (x=$x)  maxlinkdim=$χ")
        end
    end

    # ── Combine and normalise ─────────────────────────────────────────────────
    # DOS ≈ (D - N_phys) × avg_full  +  N_phys × avg_bound
    result = zeros(Float64, Nω)
    for iω in 1:Nω
        valid[iω] || continue
        if dos_weighting == :sample
            result[iω] = (accum_full[iω] +
                          ((N_bound > 0 && is_exc) ? accum_bound[iω] : 0.0)) / denom[iω]
        elseif N_bound > 0 && is_exc
            result[iω] = ((D - N_phys) * accum_full[iω] +
                          N_phys       * accum_bound[iω]) / denom[iω]
        else
            result[iω] = D * accum_full[iω] / denom[iω]
        end
    end
    normalize && dos_weighting == :trace && (result ./= D)
    return result
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

HODC (Higher-Order Delta Chebyshev) variant of `get_ldos_from_mun`. Uses a
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

"""
    _dos_weight_matrix(Ncheb, ω_vals; kernel, lambda, eta, m_order)
        -> (W::Matrix, denom::Vector)

Stochastic-DOS reconstruction weights `W[n, iω]` and per-ω normalisation
`denom[iω]` for a given KPM `kernel`.  The DOS is recovered from the (sample-
averaged) Chebyshev moments `μ_n` as `Σ_n W[n,iω] μ_n / denom[iω]`.

- Convolution kernels (`:jackson`, `:lorentz`, `:fejer`, `:dirichlet`):
  `W` follows `_kpm_weight_matrix` and `denom = π²·Ncheb·√(1−ω²)`, matching
  `get_ldos_from_mun`.
- `:hodc`: the contour weights `νₙ(ω)` from `get_hodc_weights` already carry the
  full normalisation (`denom = 1`), matching `get_ldos_hodc_from_mun`.  `eta=0`
  falls back to `1/(Ncheb+1)`.

Entries with `|ω| ≥ 1` are zeroed in `W` (outside the rescaled spectral support).
"""
function _dos_weight_matrix(Ncheb::Int, ω_vals;
                            kernel::Symbol = :jackson,
                            lambda::Real   = 4.0,
                            eta::Real      = 0.0,
                            m_order::Int   = 4)
    Nω = length(ω_vals)
    if kernel == :hodc
        eta_   = eta == 0.0 ? 1 / (Ncheb + 1) : eta
        zl, wl = compute_hodc_params(m_order)
        W = zeros(Float64, Ncheb, Nω)
        for iω in 1:Nω
            abs(ω_vals[iω]) >= 1.0 && continue
            W[:, iω] .= get_hodc_weights(ω_vals[iω], Ncheb, eta_, zl, wl)
        end
        return W, ones(Float64, Nω)
    else
        W     = _kpm_weight_matrix(Ncheb, ω_vals; kernel=kernel, lambda=lambda)
        denom = [π^2 * Ncheb * sqrt(max(1 - ω^2, 0.0)) for ω in ω_vals]
        return W, denom
    end
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


# ─────────────────────────────────────────────────────────────────────────────
# Diagonal-MPS accumulation (mirrors the get_bands / momentum-space pattern)
# ─────────────────────────────────────────────────────────────────────────────

"""
    get_ldos_diag_from_Tn(Tn_list, N, ω_vals; kernel, lambda, maxdim, cutoff)
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


#for getting electron densities
function get_density_quantics(A,L)
    
    xvals = range(0, (2^L - 1); length=2^L)
    f(x) =  1 -  inner(random_mps(sites,to_binary_vector(Int(x),L))',A, random_mps(sites,to_binary_vector(Int(x),L)))
    qtt, ranks, errors = quanticscrossinterpolate(Float64, f,  xvals ; tolerance=1e-8)

    tt = TCI.tensortrain(qtt.tci)
    density_mps = MPS(tt;sites)
  
    density_mpo = outer(density_mps',density_mps)
    for i in 1:L
        density_mpo.data[i] =  Quantics._asdiagonal(density_mps.data[i],sites[i])
    end
    
    return qtt,density_mpo,density_mps
end




# ─────────────────────────────────────────────────────────────────────────────
# Exciton LDOS  (MPS-based only — no MPO Chebyshev for the 2L-site chain)
# ─────────────────────────────────────────────────────────────────────────────

"""
    KPM_Tn(H::TBHamiltonian, Ncheb::Int, X::Int; maxdim, cutoff, verbose, ...)
        -> (Tn_mps_list, scale, center)

MPS-based Chebyshev expansion for the exciton state |X,X⟩ on the exciton
`TBHamiltonian`.  `X ∈ {1, …, 2^L}` is the 1-indexed exciton site.

Handles rescaling automatically from `H.scale`/`H.center`, caches the result
in `H._tn_mps_cache` and `H._tn_Ncheb`.

No MPO-mode is provided for exciton systems: the 2L-site MPO chain makes
the MPO Chebyshev variant prohibitively expensive.
"""
function KPM_Tn(H::TBHamiltonian, Ncheb::Int, X::Int;
                maxdim::Int       = 40,
                cutoff::Real      = 1e-8,
                dmrg_nsweeps::Int = 5,
                dmrg_maxdim       = [10, 20, 40],
                dmrg_linkdim::Int = 4,
                verbose::Bool     = false)
    _ensure_scale!(H; dmrg_nsweeps=dmrg_nsweeps, dmrg_maxdim=dmrg_maxdim,
                      dmrg_linkdim=dmrg_linkdim)
    psi0        = mpsexciton(X, H.sites)
    Tn, _, _    = KPM_Tn_mps(H.mpo, Ncheb, psi0, H.sites;
                               scale   = H.scale,
                               center  = H.center,
                               maxdim  = maxdim,
                               cutoff  = cutoff,
                               verbose = verbose)
    H._tn_mps_cache = Tn
    H._tn_Ncheb     = Ncheb
    return Tn, H.scale, H.center
end


"""
    _get_exciton_ldos_cached(H::TBHamiltonian, X::Int, omega; kwargs...) -> Real

Internal legacy cache-backed scalar exciton LDOS helper. Public exciton LDOS now
routes through `get_exciton_ldos_spatial`, which performs online MPS recursion
without storing a Chebyshev cache on `H`.
"""
function _get_exciton_ldos_cached(H::TBHamiltonian, X::Int, ω_phys::Real;
                           Ncheb::Int     = 200,
                           kernel::Symbol = :jackson,
                           lambda::Real   = 4.0,
                           eta::Real      = 0.0,
                           m_order::Int   = 4,
                           maxdim::Int    = 40,
                           cutoff::Real   = 1e-8,
                           verbose::Bool  = false)
    _ensure_scale!(H)
    E    = (ω_phys - H.center) / H.scale
    abs(E) >= 1.0 && return 0.0
    eta_ = eta == 0.0 ? 1 / (Ncheb + 1) : eta

    if H._tn_mps_cache !== nothing && H._tn_Ncheb == Ncheb
        psi_X = mpsexciton(X, H.sites)
        mun   = [inner(psi_X, H._tn_mps_cache[n]) for n in 1:Ncheb]
    else
        Tn, _, _ = KPM_Tn(H, Ncheb, X; maxdim=maxdim, cutoff=cutoff, verbose=verbose)
        psi_X    = mpsexciton(X, H.sites)
        mun      = [inner(psi_X, Tn[n]) for n in 1:Ncheb]
    end

    return get_ldos_from_mun(mun, Ncheb, E;
                              kernel=kernel, lambda=lambda,
                              eta=eta_, m_order=m_order)
end


"""
    get_exciton_ldos_spatial(H, Ncheb, omega_phys_vals; X_list, X_groups,
                             num_x, num_avg, x_start, x_end, kernel,
                             lambda, eta, m_order, maxdim, cutoff,
                             verbose, printinfo) -> Matrix{Float64}

CPU spatial exciton LDOS. For each bound exciton position `X` (electron = hole =
`X`, 1-indexed in `1:H.N`) this runs an online MPS Chebyshev recursion from
`|X,X>` and accumulates all requested energies in one pass. No Chebyshev cache is
stored on `H`.

Rows are energies, columns are positions/groups. `X_list` selects positions
directly. `X_groups` (or alias `x_groups`) averages several bound-pair probes into
one output column. If no explicit positions are provided, `num_x` coarse groups
are generated over `x_start:x_end`, with `num_avg` subpositions per group.

`kernel=:hodc` uses the HODC reconstruction (`eta`, `m_order`); otherwise the
standard KPM kernels are available (`:jackson`, `:lorentz`, `:fejer`,
`:dirichlet`).

`return_maxlinkdim=true` returns `(result, linkdims)` instead of just `result`,
where `linkdims::Vector{Int}` is the reached MPS bond dimension per output column
(the χ the Chebyshev recursion hit under the given `maxdim`/`cutoff`). Mirrors the
GPU entry point; useful for cutoff/tolerance studies where χ is the observable.
"""
function get_exciton_ldos_spatial(H::TBHamiltonian, Ncheb::Int, omega_phys_vals;
                                  X_list           = nothing,
                                  X_groups         = nothing,
                                  x_groups         = nothing,
                                  num_x::Int       = H.N,
                                  num_avg::Int     = 1,
                                  x_start::Int     = 1,
                                  x_end::Int       = H.N,
                                  kernel::Symbol   = :jackson,
                                  lambda::Real     = 4.0,
                                  eta::Real        = 0.0,
                                  m_order::Int     = 4,
                                  maxdim::Int      = 100,
                                  cutoff::Real     = 1e-8,
                                  verbose::Bool    = false,
                                  printinfo::Bool  = false,
                                  return_maxlinkdim::Bool = false)
    _ensure_scale!(H)
    length(H.sites) == 2 * H.L ||
        error("get_exciton_ldos_spatial: H is not an exciton Hamiltonian (expected length(H.sites) == 2*H.L).")

    X_groups !== nothing && x_groups !== nothing &&
        error("get_exciton_ldos_spatial: pass only one of X_groups or x_groups.")
    X_list !== nothing && (X_groups !== nothing || x_groups !== nothing) &&
        error("get_exciton_ldos_spatial: pass either X_list or grouped positions, not both.")

    group_arg = X_groups !== nothing ? X_groups : x_groups
    groups = if group_arg !== nothing
        group_arg isa AbstractVector{<:AbstractVector} ?
            [collect(Int, grp) for grp in group_arg] :
            [[Int(x)] for x in group_arg]
    elseif X_list !== nothing
        [[Int(x)] for x in X_list]
    else
        num_x > 0 || error("get_exciton_ldos_spatial: num_x must be positive.")
        num_avg > 0 || error("get_exciton_ldos_spatial: num_avg must be positive.")
        1 <= x_start <= x_end <= H.N ||
            error("get_exciton_ldos_spatial: expected 1 <= x_start <= x_end <= H.N.")
        window = x_end - x_start + 1
        num_x <= window ||
            error("get_exciton_ldos_spatial: num_x=$num_x exceeds sampling window length $window.")
        dx     = div(window, num_x)
        dx_sub = max(1, div(dx, num_avg))
        [[x_start + (i - 1) * dx + k * dx_sub
          for k in 0:num_avg-1
          if x_start + (i - 1) * dx + k * dx_sub <= x_end]
         for i in 1:num_x]
    end

    isempty(groups) && error("get_exciton_ldos_spatial: no spatial groups were selected.")
    for grp in groups
        isempty(grp) && error("get_exciton_ldos_spatial: empty spatial group.")
        all(x -> 1 <= x <= H.N, grp) ||
            error("get_exciton_ldos_spatial: all positions must lie in 1:H.N.")
    end

    I_mpo = MPO(H.sites, "Id")
    Ham_n = (1 / H.scale) * +(H.mpo, (-H.center) * I_mpo; cutoff=cutoff)

    omega_vals = (collect(omega_phys_vals) .- H.center) ./ H.scale
    Nomega     = length(omega_vals)
    W, denom   = _dos_weight_matrix(Ncheb, omega_vals;
                                    kernel=kernel, lambda=lambda,
                                    eta=eta, m_order=m_order)
    valid      = [abs(omega) < 1.0 for omega in omega_vals]

    nX     = length(groups)
    Xs     = first.(groups)
    result = zeros(Float64, Nomega, nX)
    linkdims = zeros(Int, nX)   # reached MPS bond dim per output column (see return_maxlinkdim)

    for (j, group) in enumerate(groups)
        last_linkdim = 0
        accum_group  = zeros(Float64, Nomega)

        for X in group
            psi0 = mpsexciton(X, H.sites)
            last_linkdim = _run_kpm_mps!(Ham_n, psi0, Ncheb, W, valid, accum_group;
                                         weight=1.0 / length(group),
                                         cutoff=cutoff, maxdim=maxdim)
        end

        for iomega in 1:Nomega
            valid[iomega] || continue
            result[iomega, j] = accum_group[iomega] / denom[iomega]
        end
        linkdims[j] = last_linkdim

        (verbose || printinfo) && (j % 5 == 0 || j == nX) &&
            println("  exciton ldos $j/$nX (X=$(Xs[j]), n_avg=$(length(group)))  maxlinkdim=$last_linkdim")
    end

    return return_maxlinkdim ? (result, linkdims) : result
end

function get_exciton_ldos(H::TBHamiltonian, X::Int, omega_phys::Real;
                          Ncheb::Int     = 200,
                          kernel::Symbol = :jackson,
                          lambda::Real   = 4.0,
                          eta::Real      = 0.0,
                          m_order::Int   = 4,
                          maxdim::Int    = 40,
                          cutoff::Real   = 1e-8,
                          verbose::Bool  = false)
    ldos = get_exciton_ldos_spatial(H, Ncheb, [omega_phys];
                                    X_list=[X], kernel=kernel,
                                    lambda=lambda, eta=eta,
                                    m_order=m_order, maxdim=maxdim,
                                    cutoff=cutoff, verbose=verbose)
    return ldos[1, 1]
end

function get_exciton_ldos(H::TBHamiltonian, X::Int, omega_phys_vals;
                          Ncheb::Int     = 200,
                          kernel::Symbol = :jackson,
                          lambda::Real   = 4.0,
                          eta::Real      = 0.0,
                          m_order::Int   = 4,
                          maxdim::Int    = 40,
                          cutoff::Real   = 1e-8,
                          verbose::Bool  = false)
    ldos = get_exciton_ldos_spatial(H, Ncheb, omega_phys_vals;
                                    X_list=[X], kernel=kernel,
                                    lambda=lambda, eta=eta,
                                    m_order=m_order, maxdim=maxdim,
                                    cutoff=cutoff, verbose=verbose)
    return vec(ldos[:, 1])
end


"""
    get_exciton_ldos_separation(H, Ncheb, omega_phys_vals; d_list, R_list=1:H.N,
                                boundary=:open, kernel=:jackson, lambda=4.0, eta=0.0,
                                m_order=4, maxdim=100, cutoff=1e-8, verbose=false,
                                printinfo=false) -> Array{Float64,3}

Relative-separation-resolved exciton LDOS

    rho(d, R, E) = <R+d, R| delta(E - H) |R+d, R>

generalising `get_exciton_ldos_spatial` (the `d = 0` slice) to electron-hole probes
with electron at `R+d` and hole at `R` (both 1-indexed in `1:H.N`).

For each `(d, R)` pair the probe `|R+d, R>` is a single product state
(`mpsexciton(R+d, R, H.sites)`) — no superposition / carry construction is needed,
unlike the momentum-space probes `mpsexcitonQ`/`mpsexcitonQTrace`. An online MPS
Chebyshev recursion accumulates all energies in one pass, exactly as in
`get_exciton_ldos_spatial`.

`boundary=:open` (default): pairs with `R+d` outside `1:H.N` are left as `NaN`.
`boundary=:periodic`: `R+d` is wrapped modulo `H.N`.

Returns an `(Nomega, length(d_list), length(R_list))` array. Cost is
`length(d_list) * length(R_list)` Chebyshev recursions, so keep these (and `Ncheb`)
small for a first pass.
"""
function get_exciton_ldos_separation(H::TBHamiltonian, Ncheb::Int, omega_phys_vals;
                                     d_list,
                                     R_list           = 1:H.N,
                                     boundary::Symbol = :open,
                                     kernel::Symbol   = :jackson,
                                     lambda::Real     = 4.0,
                                     eta::Real        = 0.0,
                                     m_order::Int     = 4,
                                     maxdim::Int      = 100,
                                     cutoff::Real     = 1e-8,
                                     verbose::Bool    = false,
                                     printinfo::Bool  = false)
    _ensure_scale!(H)
    length(H.sites) == 2 * H.L ||
        error("get_exciton_ldos_separation: H is not an exciton Hamiltonian (expected length(H.sites) == 2*H.L).")
    boundary in (:open, :periodic) ||
        error("get_exciton_ldos_separation: boundary must be :open or :periodic, got $boundary.")

    I_mpo = MPO(H.sites, "Id")
    Ham_n = (1 / H.scale) * +(H.mpo, (-H.center) * I_mpo; cutoff=cutoff)

    omega_vals = (collect(omega_phys_vals) .- H.center) ./ H.scale
    Nomega     = length(omega_vals)
    W, denom   = _dos_weight_matrix(Ncheb, omega_vals;
                                    kernel=kernel, lambda=lambda,
                                    eta=eta, m_order=m_order)
    valid      = [abs(omega) < 1.0 for omega in omega_vals]

    ds = collect(Int, d_list)
    Rs = collect(Int, R_list)
    result = fill(NaN, Nomega, length(ds), length(Rs))

    for (jR, R) in enumerate(Rs), (jd, d) in enumerate(ds)
        xe = R + d
        if boundary == :periodic
            xe = mod(xe - 1, H.N) + 1
        elseif !(1 <= xe <= H.N)
            continue   # leave as NaN: electron position falls outside the chain
        end

        psi0  = mpsexciton(xe, R, H.sites)
        accum = zeros(Float64, Nomega)
        last_linkdim = _run_kpm_mps!(Ham_n, psi0, Ncheb, W, valid, accum;
                                     cutoff=cutoff, maxdim=maxdim)

        for iomega in 1:Nomega
            valid[iomega] || continue
            result[iomega, jd, jR] = accum[iomega] / denom[iomega]
        end

        (verbose || printinfo) &&
            println("  exciton separation d=$d, R=$R (x_e=$xe)  maxlinkdim=$last_linkdim")
    end

    return result
end


"""
    exciton_radius2(rho, d_list) -> Matrix{Float64}

Energy-resolved exciton radius

    xi^2(R, E) = sum_d d^2 * P(d | R, E),   P(d | R, E) = rho(d,R,E) / sum_d' rho(d',R,E)

from a `rho` array as returned by `get_exciton_ldos_separation`
(`size(rho) == (Nomega, length(d_list), n_R)`). Returns an `(Nomega, n_R)` matrix.
Entries where any `rho[:, :, R]` along `d` is `NaN` (e.g. `R+d` outside the chain
under `boundary=:open`), or where the normalisation `sum_d rho(d,R,E)` is zero, are
returned as `NaN`.
"""
function exciton_radius2(rho::AbstractArray{<:Real,3}, d_list)
    Nomega, nd, nR = size(rho)
    nd == length(d_list) || error("exciton_radius2: size(rho,2) must match length(d_list).")
    d2 = Float64.(collect(d_list)) .^ 2

    xi2 = fill(NaN, Nomega, nR)
    for iR in 1:nR, iomega in 1:Nomega
        slice = @view rho[iomega, :, iR]
        any(isnan, slice) && continue
        norm = sum(slice)
        norm == 0 && continue
        xi2[iomega, iR] = sum(d2 .* slice) / norm
    end
    return xi2
end


"""
    ldos_exc_KPM_Tn(H, N, X; cutoff, maxdim) -> Vector

Low-level: Chebyshev moment list ⟨X|T_n(H)|X⟩ for the exciton state |X,X⟩.
`X ∈ {1, …, 2^LPhys}` (1-indexed). `H` must already be rescaled so its
spectrum lies in (−1, 1).  Prefer `KPM_Tn(H::TBHamiltonian, Ncheb, X)` for
the high-level interface which handles rescaling and caching automatically.
"""
function ldos_exc_KPM_Tn(H::MPO, N::Int64, X; cutoff=1e-9, maxdim=200)
    apply_kwargs = (cutoff=cutoff, maxdim=maxdim)
    sites        = getindex.(siteinds(H), 2)

    T_k_minus_2 = mpsexciton(X, sites)
    mu_1        = inner(mpsexciton(X, sites)', T_k_minus_2)
    T_k_minus_1 = apply(H, T_k_minus_2; apply_kwargs...)
    mu_2        = inner(mpsexciton(X, sites)', T_k_minus_1)
    mun_list    = [mu_1, mu_2]

    for k in 3:N
        T_k = +(2 * apply(H, T_k_minus_1; apply_kwargs...),
                -T_k_minus_2; apply_kwargs...)
        push!(mun_list, inner(mpsexciton(X, sites)', T_k))
        T_k_minus_2 = T_k_minus_1
        T_k_minus_1 = T_k
    end

    return mun_list
end


# ---------------------------------------------------------------------
# HODC-based DOS / LDOS for exciton KPM moments
# (compute_hodc_params and get_hodc_weights live in KPM_tk.jl)
# ---------------------------------------------------------------------

function get_mus_raw(tn_lis)
    return [real(tr(tns)) for tns in tn_lis]
end


"""
    compute_dos_ldos_hodc(N, tn_lis, en_num; eta, m_order, maxdim)
        -> (dos_vec, ldos_mpo_list)

Compute DOS and energy-resolved LDOS MPOs from a list of Chebyshev moment
MPOs `tn_lis` using the HODC kernel.
"""
function compute_dos_ldos_hodc(N, tn_lis, en_num;
                                eta=0.02, m_order=6, maxdim=200)
    @assert length(tn_lis) == N

    mus_raw = get_mus_raw(tn_lis)
    yvals   = range(-0.99, 0.99; length=en_num)
    zl, wl  = compute_hodc_params(m_order)

    dos_vec       = zeros(Float64, en_num)
    ldos_mpo_list = Vector{MPO}(undef, en_num)

    for (i, y) in enumerate(yvals)
        nu_k = get_hodc_weights(y, N, eta, zl, wl)

        val = mus_raw[1] * nu_k[1]
        for k in 2:N
            val += 2.0 * mus_raw[k] * nu_k[k]
        end
        dos_vec[i] = val

        mpo = nu_k[1] * tn_lis[1]
        for k in 2:N
            mpo = +(mpo, (2.0 * nu_k[k]) * tn_lis[k]; maxdim=maxdim)
        end
        ldos_mpo_list[i] = mpo
    end

    dos_vec ./= maximum(dos_vec)
    dos_vec   = max.(0, dos_vec)

    return dos_vec, ldos_mpo_list
end


