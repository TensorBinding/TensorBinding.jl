# solvers/kpm/dos.jl — total density of states
#
# Contents: the stochastic trace estimate over random basis states, with optional
# aux-DOF projections and exciton bound-sector stratification (get_dos_stochastic),
# and the deterministic trace of each online Chebyshev MPO (get_dos_trace).
#
# Entry points: get_dos_stochastic, get_dos_trace
# Depends on: core/Utils.jl (_basis_state_mps, extract_diagonal_to_mps,
#   mpsexciton), core/TBSystem.jl (TBHamiltonian, physical_projector,
#   physical_site_state, _is_binary_position_space), core/AuxDOF.jl
#   (_aux_projection, _probe_sectors, probe_state), solvers/DMRG.jl
#   (_ensure_scale!), solvers/kpm/kernels.jl
#   (_kpm_energy_grid), solvers/kpm/recursion.jl (_scaled_hamiltonian,
#   _run_kpm_mps!).
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

    projected_position_space = !_is_binary_position_space(H)
    D      = projected_position_space ? H.N : prod(ITensors.dim(s) for s in H.sites)
    N_phys = H.N
    is_exc = length(H.sites) == 2 * H.L

    # Projections are not switched on automatically here (see the docstring).
    aux = _aux_projection(H; nambu_proj, proj_nambu, spin_proj, proj_s,
                             layer_proj, proj_layer, sublat_proj, proj_sl,
                             autoenable=false)

    ω_vals, W, denom, valid = _kpm_energy_grid(H, Ncheb, ω_phys_vals;
                                               kernel=kernel, lambda=lambda, eta=eta,
                                               m_order=m_order, allow_hodc=true)
    Nω     = length(ω_vals)

    rng         = seed === nothing ? Random.default_rng() : Random.MersenneTwister(seed)
    accum_full  = zeros(Float64, Nω)
    accum_bound = zeros(Float64, Nω)

    if _any_projected(aux)
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
                χ = _run_kpm_mps!(Ham_n, probe_state(H, x, σ), Ncheb, W, valid, accum_full;
                                   weight=1.0/N_sample, cutoff=cutoff, maxdim=maxdim)
                verbose && i % 15 == 0 && σ == first(sectors) &&
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
    samples = projected_position_space ?
        rand(rng, 1:H.N, N_sample) : rand(rng, 0:(D - 1), N_sample)
    for (i, sample) in enumerate(samples)
        psi0 = projected_position_space ?
            physical_site_state(H, sample) : _basis_state_mps(sample, H.sites)
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
    Tkm2 = P
    Tkm1 = Ham_n
    moments[1] = trace_diagonal(Tkm2)
    moments[2] = trace_diagonal(Tkm1)
    for k in 3:Ncheb
        Tk = +(2 * apply(Ham_n, Tkm1; cutoff=cutoff), -Tkm2;
               cutoff=cutoff, maxdim=maxdim)
        ITensorMPS.truncate!(Tk; cutoff=cutoff, maxdim=maxdim)
        moments[k] = trace_diagonal(Tk)
        Tkm2, Tkm1 = Tkm1, Tk
        verbose && (k % 10 == 0 || k == Ncheb) &&
            println("get_dos_trace step $k/$Ncheb  maxlinkdim=$(maxlinkdim(Tkm1))")
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
