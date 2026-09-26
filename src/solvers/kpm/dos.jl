# solvers/kpm/dos.jl — total density of states
#
# Contents: the stochastic trace estimate over random basis states, with optional
# aux-DOF projections and exciton bound-sector stratification (get_dos_stochastic,
# whose sampling kernel _dos_stochastic is also the body of get_dos_stochastic_gpu),
# and the deterministic trace of each online Chebyshev MPO (get_dos_trace).
#
# Entry points: get_dos_stochastic, get_dos_trace
# Depends on: core/Utils.jl (_basis_state_mps, extract_diagonal_to_mps,
#   mpsexciton, to_binary_vector, _on_host), core/TBSystem.jl (TBHamiltonian, physical_projector,
#   physical_site_state, _is_binary_position_space), core/AuxDOF.jl
#   (_aux_projection, _probe_sectors, probe_state), solvers/DMRG.jl
#   (_ensure_scale!), solvers/kpm/kernels.jl
#   (_kpm_energy_grid), solvers/kpm/recursion.jl (_scaled_hamiltonian,
#   chebyshev_foreach, _run_kpm_mps!).
#
# Split from the former solvers/KPM_tk.jl in Tier 1.

# ============================================================
# 1. Stochastic full DOS (trace estimation via random diagonal sampling)
# ============================================================

"""
    get_dos_stochastic(H::TBHamiltonian, Ncheb::Int, ω_phys_vals;
                       N_sample=50, N_bound=0, seed=42, normalize=false,
                       dos_weighting=:trace, kernel=:jackson, lambda=4.0, eta=0.0,
                       m_order=4, maxdim=100, cutoff=1e-8, verbose=false,
                       nambu_proj=false, proj_nambu=nothing, spin_proj=false,
                       proj_s=nothing, layer_proj=false, proj_layer=nothing,
                       sublat_proj=false, proj_sl=nothing)
        -> Vector{Float64}

Stochastic full DOS via random trace estimation (MPS Chebyshev, 3 MPS per sample).

**Sampling**: `N_sample` random basis states (default `50`), drawn from a
`MersenneTwister(seed)`; `seed=nothing` draws from the global RNG. `maxdim` and
`cutoff` truncate each MPS recursion step; `verbose` prints the bond dimension
every 15 samples.

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

    Ham_n = _scaled_hamiltonian(H; cutoff=cutoff)

    # Projections are not switched on automatically here (see the docstring).
    aux = _aux_projection(H; nambu_proj, proj_nambu, spin_proj, proj_s,
                             layer_proj, proj_layer, sublat_proj, proj_sl,
                             autoenable=false)

    progress = verbose ? function (kind, i, n, χ, info)
        i % 15 == 0 || return nothing
        kind === :projected ? println("Projected DOS sample $i/$n  maxlinkdim=$χ") :
        kind === :full      ? println("Full sample $i/$n  maxlinkdim=$χ") :
                              println("Bound sample $i/$n  (x=$info)  maxlinkdim=$χ")
        return nothing
    end : nothing
    return _dos_stochastic(H, Ham_n, Ncheb, ω_phys_vals, aux;
                           N_sample, N_bound, seed, normalize, dos_weighting, kernel,
                           lambda, eta, m_order, maxdim, cutoff, progress)
end

"""
    _dos_stochastic(H, H̃, Ncheb, ω_phys_vals, aux; N_sample, N_bound, seed, normalize,
                    dos_weighting, kernel, lambda, eta, m_order, maxdim, cutoff,
                    continuum_only=false, caller="get_dos_stochastic",
                    to_device=_on_host, device_type=ComplexF64, after_run=nothing,
                    progress=nothing) -> Vector{Float64}

The sampling and normalisation of `get_dos_stochastic` (documented there), also the
body of `get_dos_stochastic_gpu`, from the rescaled Hamiltonian `H̃` (on the device)
and the aux projection `aux` (`autoenable=false`). Each probe state is built on the
CPU, moved by `to_device(·, device_type)` (see `_on_host`) and run through
`_run_kpm_mps!`; `after_run()` follows each run and `progress(kind, i, n, χ, info)`
reports it, with `kind` one of `:projected` (first sector of sample `i` only),
`:full`, `:continuum` (`info = (x_e, x_h)`) and `:bound` (`info = x`).

`continuum_only=true` (the GPU option; exciton Hamiltonians, checked by the caller)
draws the `N_sample` probes as ordered electron-hole pairs `x_e ≠ x_h`, weighted by
`D − N_phys`; together with aux projections it is an error, reported as `caller`'s.
"""
function _dos_stochastic(H::TBHamiltonian, Ham_n::MPO, Ncheb::Int, ω_phys_vals,
                         aux::AuxProjection;
                         N_sample::Int, N_bound::Int, seed, normalize::Bool,
                         dos_weighting::Symbol, kernel::Symbol, lambda::Real, eta::Real,
                         m_order::Int, maxdim::Int, cutoff::Real,
                         continuum_only::Bool = false,
                         caller::String       = "get_dos_stochastic",
                         to_device            = _on_host,
                         device_type::Type    = ComplexF64,
                         after_run            = nothing,
                         progress             = nothing)
    projected_position_space = !_is_binary_position_space(H)
    D      = projected_position_space ? H.N : prod(ITensors.dim(s) for s in H.sites)
    N_phys = H.N
    is_exc = length(H.sites) == 2 * H.L

    ω_vals, W, denom, valid = _kpm_energy_grid(H, Ncheb, ω_phys_vals;
                                               kernel=kernel, lambda=lambda, eta=eta,
                                               m_order=m_order, allow_hodc=true)
    Nω     = length(ω_vals)

    rng         = seed === nothing ? Random.default_rng() : Random.MersenneTwister(seed)
    accum_full  = zeros(Float64, Nω)
    accum_bound = zeros(Float64, Nω)

    # One probe: moved to the device, run, followed by after_run(); returns the
    # largest bond dimension of its recursion.
    function run!(psi0, accum, weight)
        χ = _run_kpm_mps!(Ham_n, to_device(psi0, device_type), Ncheb, W, valid, accum;
                           weight=weight, cutoff=cutoff, maxdim=maxdim)
        after_run === nothing || after_run()
        return χ
    end

    if _any_projected(aux)
        continuum_only &&
            error("$caller: continuum_only is not supported together with auxiliary projections.")
        # ── Projected DOS: sample position states with fixed aux sectors ─────
        # Trace over position basis only, with aux dofs projected to selected
        # sectors.  Effective dimension = N_phys × n_sectors.
        # D_eff = N_phys: the sector loop already sums all sectors into accum_full,
        # so only the position average (× N_phys) is needed for normalisation.
        D_eff = N_phys

        sectors = _probe_sectors(aux)
        xs = rand(rng, 1:N_phys, N_sample)
        for (i, x) in enumerate(xs)
            for σ in sectors
                χ = run!(probe_state(H, x, σ), accum_full, 1.0/N_sample)
                progress !== nothing && σ == first(sectors) &&
                    progress(:projected, i, N_sample, χ, x)
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

    if continuum_only
        # ── Continuum samples: ordered electron-hole pairs x_e ≠ x_h ──────────
        xs_e = rand(rng, 1:N_phys, N_sample)
        ys_h = rand(rng, 1:(N_phys - 1), N_sample)
        for i in 1:N_sample
            xe = xs_e[i]
            xh = ys_h[i] < xe ? ys_h[i] : ys_h[i] + 1
            χ = run!(_exciton_pair_state(H, xe, xh), accum_full, 1.0/N_sample)
            progress === nothing || progress(:continuum, i, N_sample, χ, (xe, xh))
        end
    else
        # ── Full Hilbert space samples (weight = D / N_sample per sample) ─────
        samples = projected_position_space ?
            rand(rng, 1:H.N, N_sample) : rand(rng, 0:(D - 1), N_sample)
        for (i, sample) in enumerate(samples)
            psi0 = projected_position_space ?
                physical_site_state(H, sample) : _basis_state_mps(sample, H.sites)
            χ = run!(psi0, accum_full, 1.0/N_sample)
            progress === nothing || progress(:full, i, N_sample, χ, sample)
        end
    end

    # ── Bound-sector samples (exciton: random |x,x⟩, weight = N_phys/N_bound) ─
    if N_bound > 0 && is_exc
        xs = rand(rng, 1:N_phys, N_bound)
        for (i, x) in enumerate(xs)
            χ = run!(mpsexciton(x, H.sites), accum_bound, 1.0/N_bound)
            progress === nothing || progress(:bound, i, N_bound, χ, x)
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
        elseif continuum_only && is_exc
            result[iω] = (D - N_phys) * accum_full[iω] / denom[iω]
        else
            result[iω] = D * accum_full[iω] / denom[iω]
        end
    end
    if normalize && dos_weighting == :trace
        norm_dim = (continuum_only && is_exc && N_bound == 0) ? (D - N_phys) : D
        result ./= norm_dim
    end
    return result
end

# The electron-hole product state |x_e, x_h⟩ (1-based positions) on the interleaved
# register [e₁, h₁, e₂, h₂, …] of an exciton Hamiltonian, big-endian bits.
function _exciton_pair_state(H::TBHamiltonian, xe::Int, xh::Int)
    Lphys = div(length(H.sites), 2)
    bits_e = to_binary_vector(xe - 1, Lphys)
    bits_h = to_binary_vector(xh - 1, Lphys)
    state = Vector{String}(undef, 2 * Lphys)
    for b in 1:Lphys
        state[2b - 1] = bits_e[b]
        state[2b]     = bits_h[b]
    end
    return MPS(H.sites, state)
end


# ============================================================
# 2. Deterministic full DOS from the exact tensor-network trace
# ============================================================

"""
    get_dos_trace(H, Ncheb, ω_phys_vals; normalize=false, kernel=:jackson,
                  lambda=4.0, eta=0.0, m_order=4,
                  maxdim=100, cutoff=1e-8, verbose=false) -> Vector{Float64}

Deterministic total DOS from the exact tensor-network trace of each online
Chebyshev MPO. Only three MPOs are retained. At every order the diagonal MPO is
converted to an MPS and contracted with the product MPS `|1,1,...>`, giving
`Tr[T_n(H_tilde)]` without summing LDOS curves or integrating a spectrum.

For projected position spaces, `T_0` is `physical_projector(H)` and the trace is
therefore over physical states only. `normalize=true` divides by `Tr(T_0)`;
otherwise the spectral weight corresponds to the total traced state count.
"""
function get_dos_trace(H::TBHamiltonian, Ncheb::Int, ω_phys_vals;
                       normalize::Bool=false,
                       kernel::Symbol=:jackson,
                       lambda::Real=4.0,
                       eta::Real=0.0,
                       m_order::Int=4,
                       maxdim::Int=100,
                       cutoff::Real=1e-8,
                       verbose::Bool=false)
    Ncheb >= 2 || throw(ArgumentError("Ncheb must be at least 2"))
    _ensure_scale!(H)
    P = physical_projector(H)
    Ham_n = _scaled_hamiltonian(H; cutoff=cutoff, identity=P)

    function trace_diagonal(Tn::MPO)
        diagonal = extract_diagonal_to_mps(Tn)
        ITensorMPS.truncate!(diagonal; cutoff=cutoff, maxdim=maxdim)
        ones_state = MPS([ITensor(ones(Float64, dim(s)), s)
                          for s in siteinds(diagonal)])
        return real(inner(ones_state, diagonal))
    end

    moments = zeros(Float64, Ncheb)
    chebyshev_foreach(Ham_n, P, Ncheb; T1=Ham_n, maxdim=maxdim, cutoff=cutoff,
                      apply_trunc=(:cutoff,), post_trunc=(:cutoff, :maxdim)) do n, Tn
        k = n + 1
        moments[k] = trace_diagonal(Tn)
        verbose && k >= 3 && (k % 10 == 0 || k == Ncheb) &&
            println("get_dos_trace step $k/$Ncheb  maxlinkdim=$(maxlinkdim(Tn))")
    end

    ω_vals, W, denom, valid = _kpm_energy_grid(H, Ncheb, ω_phys_vals;
                                               kernel, lambda, eta, m_order,
                                               allow_hodc=true)
    result = zeros(Float64, length(ω_vals))
    for iω in eachindex(ω_vals)
        valid[iω] || continue
        result[iω] = dot(moments, view(W, :, iω)) / denom[iω]
    end
    normalize && (result ./= moments[1])
    return result
end
