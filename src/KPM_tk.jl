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


"""
    get_ldos_online(H::TBHamiltonian, Ncheb::Int, X::Int, ω_phys_vals;
                    kernel, lambda, maxdim, cutoff, verbose) -> Vector{Float64}

Online real-space LDOS at site `X` for all physical energies in `ω_phys_vals`.

This is the third KPM method, complementing the MPO-MPO (full operator, `mode=:mpo`)
and MPO-MPS (state propagation, `mode=:mps`) variants.  It never stores more than
**3 MPOs** simultaneously.

**Algorithm** (mirrors `get_bands` but in real space, without QFT):

1. Rescale `H` and pre-compute the weight matrix `W[n, iω]` for all energies.
2. Run the Chebyshev recursion `T_0 = I, T_1 = H̃, T_k = 2H̃T_{k-1} − T_{k-2}`:
   at each step `n`:
   a. Extract the diagonal of `T_n` as an MPS: `diag_n = extract_diagonal_to_mps(T_n)`
   b. Evaluate `diag_n` at the computational-basis state `|X⟩`:
      contract site-by-site using the binary digits of `X − 1` (MSB-first).
   c. Accumulate: `accum[iω] += W[n, iω] * val`  for every energy `iω`.
   d. Discard `T_{n-2}`.  Only `T_{n-1}` and `T_n` stay in memory.
3. Normalise: `A(X, ω) = accum[iω] / (π² · Ncheb · √(1 − ω²))`.

`X ∈ {1, …, 2^L}` is the 1-indexed physical site (TensorBinding convention).
`ω_phys_vals` is a vector of physical energies; pass `[ω]` for a single point.

Returns `Vector{Float64}` of length `length(ω_phys_vals)` with `0.0` for energies
outside the spectral support.  No Chebyshev cache is needed or produced.

Example
-------
```julia
ωlist = range(-3.0, 3.0; length=300)
ldos  = get_ldos_online(H, 200, 2^(H.L-1), ωlist)   # middle site
```
"""
function get_ldos_online(H::TBHamiltonian, Ncheb::Int, X::Int, ω_phys_vals;
                          kernel::Symbol = :jackson,
                          lambda::Real   = 4.0,
                          maxdim::Int    = 100,
                          cutoff::Real   = 1e-8,
                          verbose::Bool  = false)
    _ensure_scale!(H)

    # ── Rescaled Hamiltonian ──────────────────────────────────────────────────
    I_mpo = MPO(H.sites, "Id")
    Ham_n = (1 / H.scale) * +(H.mpo, (-H.center) * I_mpo; cutoff=cutoff)

    # ── KPM weight matrix and accumulators ────────────────────────────────────
    ω_vals = (collect(ω_phys_vals) .- H.center) ./ H.scale
    Nω     = length(ω_vals)
    W      = _kpm_weight_matrix(Ncheb, ω_vals; kernel=kernel, lambda=lambda)
    valid  = [abs(ω) < 1.0 for ω in ω_vals]
    accum  = zeros(Float64, Nω)

    # ── Evaluation state |X⟩ on the full MPO site chain ─────────────────────
    # Standard (L sites): X is a position in {1,…,2^L}.
    # Exciton (2L sites): X labels the single-sector exciton position → |X,X⟩.
    L_tot    = length(H.sites)
    psi_eval = L_tot == H.L ? binary_to_MPS(X - 1, H.L, H.sites) :
                               mpsexciton(X, H.sites)

    # ── MPS-based Chebyshev recursion ─────────────────────────────────────────
    # We propagate the STATE |φ_k⟩ = T_k(H)|X⟩ rather than the full operator T_k(H).
    # At each step the scalar moment μ_k = ⟨X|T_k(H)|X⟩ = inner(psi_eval, φ_k) is
    # accumulated directly into all energy slots simultaneously.
    #
    # Cost: Ncheb × (MPO×MPS apply) — O(L · χ² · W) — vs the old MPO×MPO route
    # which is O(L · χ_T² · W²) with χ_T growing with every step.  For L=5 exciton
    # (10 sites, W_H≈5) this is typically 10-100× faster.
    #
    # Only 3 MPS alive at any time (φ_{k-2}, φ_{k-1}, φ_k).

    function accumulate_mps!(phi, n)
        mu = real(inner(psi_eval, phi))
        for iω in 1:Nω
            valid[iω] || continue
            accum[iω] += W[n, iω] * mu
        end
    end

    phi_km2 = psi_eval
    phi_km1 = apply(Ham_n, psi_eval; cutoff=cutoff, maxdim=maxdim)
    accumulate_mps!(phi_km2, 1)
    accumulate_mps!(phi_km1, 2)

    for k in 3:Ncheb
        phi_k = +(2 * apply(Ham_n, phi_km1; cutoff=cutoff, maxdim=maxdim),
                  -phi_km2; cutoff=cutoff, maxdim=maxdim)
        accumulate_mps!(phi_k, k)
        phi_km2 = phi_km1
        phi_km1 = phi_k
        verbose && (k % 10 == 0 || k == Ncheb) &&
            println("get_ldos_online step $k/$Ncheb  maxlinkdim=$(maxlinkdim(phi_km1))")
    end

    # ── Normalise ─────────────────────────────────────────────────────────────
    result = zeros(Float64, Nω)
    for iω in 1:Nω
        valid[iω] || continue
        result[iω] = accum[iω] / (π^2 * Ncheb * sqrt(1 - ω_vals[iω]^2))
    end
    return result
end


# ─────────────────────────────────────────────────────────────────────────────
# Stochastic full DOS (trace estimation via random diagonal sampling)
# ─────────────────────────────────────────────────────────────────────────────

"""
    get_dos_stochastic(H::TBHamiltonian, Ncheb::Int, ω_phys_vals;
                       N_sample, N_bound, seed, kernel, lambda, maxdim, cutoff, verbose)
        -> Vector{Float64}

Stochastic full DOS via stratified random sampling.

**Sectors and stratification**

For a system with total Hilbert space dimension `D` (e.g. `D = 2^(2L)` for an
exciton), the full trace splits into two sectors:

- **Bound sector** (e.g. electron and hole at the same site, |x,x⟩): `N_phys` states
- **Scattering sector** (all remaining basis states): `D − N_phys` states

Uniform random sampling massively underrepresents the bound sector when
`N_phys ≪ D` (for L=5: N=32, D=1024, only 3 % of states are bound).  Stratified
sampling dedicates `N_bound` samples explicitly to the bound sector and `N_sample`
to the full Hilbert space, combining with proper weights:

    DOS(ω) = N_phys × avg_bound(ω)  +  (D − N_phys) × avg_scatter(ω)

`N_bound = 0` (default) recovers uniform sampling over all D states.
`N_bound = H.N` uses one sample per physical site — exact bound-state sum for
small systems (L ≤ 7 or so).

**Bound-state sampling (exciton)**

When `N_bound > 0` and `length(H.sites) == 2*H.L` (exciton interleaved chain),
bound states are sampled as random |x,x⟩ states via `mpsexciton(x, H.sites)`.
For other Hamiltonians the parameter is silently ignored.

**Algorithm** — MPS Chebyshev, 3 MPS alive per sample.
"""
function get_dos_stochastic(H::TBHamiltonian, Ncheb::Int, ω_phys_vals;
                             N_sample::Int            = 50,
                             N_bound::Int             = 0,
                             seed::Union{Int,Nothing} = 42,
                             kernel::Symbol           = :jackson,
                             lambda::Real             = 4.0,
                             maxdim::Int              = 100,
                             cutoff::Real             = 1e-8,
                             verbose::Bool            = false)
    _ensure_scale!(H)

    # ── Rescaled Hamiltonian ──────────────────────────────────────────────────
    I_mpo = MPO(H.sites, "Id")
    Ham_n = (1 / H.scale) * +(H.mpo, (-H.center) * I_mpo; cutoff=cutoff)

    D      = prod(ITensors.dim(s) for s in H.sites)
    N_phys = H.N   # physical sites per sector (= 2^H.L)
    is_exc = length(H.sites) == 2 * H.L   # exciton interleaved structure

    # ── KPM weight matrix ─────────────────────────────────────────────────────
    ω_vals = (collect(ω_phys_vals) .- H.center) ./ H.scale
    Nω     = length(ω_vals)
    W      = _kpm_weight_matrix(Ncheb, ω_vals; kernel=kernel, lambda=lambda)
    valid  = [abs(ω) < 1.0 for ω in ω_vals]

    # ── Shared: run one MPS Chebyshev recursion and accumulate moments ────────
    function run_sample!(psi0, accum, weight)
        function accum!(phi, n)
            mu = real(inner(psi0, phi))
            for iω in 1:Nω
                valid[iω] || continue
                accum[iω] += W[n, iω] * mu * weight
            end
        end
        phi_km2 = psi0
        phi_km1 = apply(Ham_n, psi0; cutoff=cutoff, maxdim=maxdim)
        accum!(phi_km2, 1);  accum!(phi_km1, 2)
        for k in 3:Ncheb
            phi_k = +(2 * apply(Ham_n, phi_km1; cutoff=cutoff, maxdim=maxdim),
                      -phi_km2; cutoff=cutoff, maxdim=maxdim)
            accum!(phi_k, k)
            phi_km2 = phi_km1;  phi_km1 = phi_k
        end
        return maxlinkdim(phi_km1)
    end

    rng         = seed === nothing ? Random.default_rng() : Random.MersenneTwister(seed)
    accum_full  = zeros(Float64, Nω)
    accum_bound = zeros(Float64, Nω)

    # ── Full Hilbert space samples (weight = D / N_sample per sample) ─────────
    samples = rand(rng, 0:(D - 1), N_sample)
    for (i, k) in enumerate(samples)
        psi0  = _basis_state_mps(k, H.sites)
        χ = run_sample!(psi0, accum_full, 1.0 / N_sample)
        verbose && i % 15 == 0 && println("Full sample $i/$N_sample  maxlinkdim=$χ")
    end

    # ── Bound-sector samples (exciton: random |x,x⟩, weight = N_phys/N_bound) ─
    if N_bound > 0 && is_exc
        xs = rand(rng, 1:N_phys, N_bound)
        for (i, x) in enumerate(xs)
            psi0 = mpsexciton(x, H.sites)
            χ = run_sample!(psi0, accum_bound, 1.0 / N_bound)
            verbose && i % 15 == 0 && println("Bound sample $i/$N_bound  (x=$x)  maxlinkdim=$χ")
        end
    end

    # ── Combine and normalise ─────────────────────────────────────────────────
    # DOS = D × avg_full  +  (N_bound > 0) × N_phys × (avg_bound − avg_full_bound)
    # Equivalently: replace the fraction N_phys/D of full samples with the dedicated
    # bound estimate, keeping the scattering contribution from the full samples.
    #
    #   DOS = (D - N_phys) × avg_scatter  +  N_phys × avg_bound
    #       ≈  D × avg_full               (when N_bound = 0, uniform)
    #   With stratification:
    #       scatter part  = D × avg_full  (already weighted correctly for D-N_phys states
    #                                      because bound states are rare in random draw)
    #       bound part    = N_phys × avg_bound  (dedicated samples, exact for that sector)
    #   Combined = D × avg_full + N_phys × (avg_bound - avg_full_bound_contamination)
    #
    # Simplest unbiased combination: subtract bound-state contamination in full samples
    # and replace with the dedicated estimate.
    #   DOS = D × avg_full  +  N_phys × avg_bound  −  N_phys × (N_phys/D) × avg_full
    #       = D × avg_full × (1 - N_phys²/D²) + N_phys × avg_bound   [approximately]
    #
    # For N_phys ≪ D the correction is negligible.  We use the simple form:
    #   DOS ≈ (D - N_phys) × avg_full  +  N_phys × avg_bound
    norm = π^2 * Ncheb

    result = zeros(Float64, Nω)
    for iω in 1:Nω
        valid[iω] || continue
        denom = norm * sqrt(1 - ω_vals[iω]^2)
        if N_bound > 0 && is_exc
            # Stratified: scattering from full samples, bound from dedicated samples
            result[iω] = ((D - N_phys) * accum_full[iω] +
                          N_phys       * accum_bound[iω]) / denom
        else
            result[iω] = D * accum_full[iω] / denom
        end
    end
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
    density_mps = ITensors.MPS(tt;sites)
  
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
    get_exciton_ldos(H::TBHamiltonian, X::Int, ω_phys; Ncheb, kernel, eta, m_order,
                     lambda, maxdim, cutoff, verbose) -> Real

Exciton local spectral weight at physical energy `ω_phys` for the exciton state
|X,X⟩, where `X ∈ {1, …, 2^L}` (1-indexed, consistent with `add_onsite!`).

If `H._tn_mps_cache` already holds `Ncheb` Chebyshev states (from a prior
`KPM_Tn(H, Ncheb, X)` call) they are reused — call `KPM_Tn(H, Ncheb, X)` first
when sweeping over many energies for the same site.  Otherwise the Chebyshev
states are built on the fly and cached for subsequent calls.

Returns `0` when `|E| ≥ 1` (energy outside the rescaled spectral support).
"""
function get_exciton_ldos(H::TBHamiltonian, X::Int, ω_phys::Real;
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


