# solvers/kpm/kernels.jl — KPM reconstruction kernels and weight matrices
#
# Contents: the damping kernels g_n (_kpm_kernel); the Chebyshev weight matrix
# W[n, ω] = c_n g_n T_{n-1}(ω) (_kpm_weight_matrix); the HODC contour weights for
# δ(ω − H) and for the retarded Green's function (compute_hodc_params,
# get_hodc_weights, get_hodc_gf_weights); the DOS weights with their per-ω
# normalisation for any kernel, HODC included (_dos_weight_matrix); the
# moment-column LDOS reconstruction shared with gpu/kpm.jl
# (_reconstruct_ldos_moment_columns); and the energy grid of every online KPM
# solver: rescaled energies, weights, normalisation and support mask
# (_rescaled_energies, _kpm_energy_grid). Pure numerics on moments: no MPS or MPO.
#
# Entry points: _kpm_kernel, _kpm_weight_matrix, _dos_weight_matrix,
#   compute_hodc_params, get_hodc_weights, get_hodc_gf_weights,
#   _reconstruct_ldos_moment_columns, _rescaled_energies, _kpm_energy_grid
# Depends on: core/TBSystem.jl (TBHamiltonian: the scale and centre of the energy
#   grid); FFTW for the HODC transforms.
#
# Split from the former solvers/KPM_tk.jl in Tier 1; _kpm_weight_matrix came from
# physics/qft/bands.jl and _reconstruct_ldos_moment_columns from gpu/kpm.jl.
# _kpm_energy_grid (Tier 2) replaced the rescale / weights / mask block and the
# π²·N·√(1−ω²) normalisation written out in each solver.

# ============================================================
# 1. KPM damping kernels
# ============================================================

# All kernels are unnormalized (max ≈ N at n=0) so caller's existing /N stays correct.
# Supported: :jackson (default), :lorentz (param lambda), :fejer, :dirichlet
#
# :jackson is N·g_n, where g_n = [(M−n+1)cos(πn/(M+1)) + sin(πn/(M+1))cot(π/(M+1))]/(M+1)
# is the textbook Jackson kernel for M = N−1 moments (its weight at n = N−1 vanishes
# up to rounding). The NH KPM (physics/nh/kpm.jl, gpu/nh.jl) uses the textbook kernel
# for all N moments, unnormalised, (N+1)·g_n with M = N: _kpm_kernel(N + 1, :jackson)[1:N].
function _kpm_kernel(N::Int, kernel::Symbol; lambda::Real = 4.0)
    if kernel == :jackson
        return [(N - n) * cos(π * n / N) + sin(π * n / N) / tan(π / N) for n in 0:N-1]
    elseif kernel == :lorentz
        return [N * sinh(lambda * (1 - n / N)) / sinh(lambda) for n in 0:N-1]
    elseif kernel == :fejer
        return Float64[N - n for n in 0:N-1]
    elseif kernel == :dirichlet
        return fill(Float64(N), N)
    elseif kernel == :hodc
        # HODC is a choice of weights, not a damping kernel: _dos_weight_matrix builds them.
        error("Unknown KPM kernel: hodc. Choose :jackson, :lorentz, :fejer, or :dirichlet " *
              "(:hodc is taken only by the functions with eta and m_order keywords).")
    else
        error("Unknown KPM kernel: $kernel. Choose :jackson, :lorentz, :fejer, or :dirichlet")
    end
end


# ============================================================
# 2. Chebyshev-KPM weight matrix
# ============================================================

"""
    _kpm_weight_matrix(Ncheb, ω_vals; kernel=:jackson, lambda=4.0) -> Matrix{Float64}

Precompute the full KPM weight matrix `W[n, iω]` for fast in-loop accumulation.

```
W[n, iω] = c_n · g_n · cos((n-1) · arccos(ω_iω))
```

- `c_n = 1` for n=1, `c_n = 2` otherwise (Chebyshev expansion factor)
- `g_n` = kernel damping: Jackson (default, finite-size ringing suppressed)
  or Lorentz (controlled width `lambda`, smoother tails)
- Entries for `|ω| ≥ 1` are set to zero (outside the spectral support)

Pre-computing W avoids recomputing cos((n-1)·arccos(ω)) inside the inner loop,
which is called Ncheb × Nω times.
"""
function _kpm_weight_matrix(Ncheb::Int, ω_vals; kernel::Symbol=:jackson, lambda::Real=4.0)
    kweights = _kpm_kernel(Ncheb, kernel; lambda=lambda)
    Nω = length(ω_vals)
    W = zeros(Float64, Ncheb, Nω)
    for iω in 1:Nω
        abs(ω_vals[iω]) >= 1.0 && continue
        for n in 1:Ncheb
            W[n, iω] = (n == 1 ? 1.0 : 2.0) * kweights[n] * cos((n-1) * acos(ω_vals[iω]))
        end
    end
    return W
end


# ============================================================
# 3. HODC kernel helpers
# ============================================================

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


# ============================================================
# 4. Stochastic-DOS reconstruction weights
# ============================================================

"""
    _dos_weight_matrix(Ncheb, ω_vals; kernel=:jackson, lambda=4.0, eta=0.0, m_order=4)
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
        denom = _kpm_denominator(Ncheb, ω_vals)
        return W, denom
    end
end

# Per-ω normalisation π²·Ncheb·√(1−ω²) of the convolution kernels: A(ω) is
# Σ_n W[n, ω] μ_n divided by it (|ω| ≥ 1 gives 0, never used: W is zero there).
_kpm_denominator(Ncheb::Int, ω_vals) = [π^2 * Ncheb * sqrt(max(1 - ω^2, 0.0)) for ω in ω_vals]


# ============================================================
# 5. Moment-column LDOS reconstruction
# ============================================================

"""
    _reconstruct_ldos_moment_columns(moments, W, denom, valid)
        -> Matrix{Float64}

Reconstruct one LDOS column per column of raw Chebyshev `moments`. The weight
matrix follows `_dos_weight_matrix`: `W[n, iω]` multiplies moment order `n-1`,
and `denom[iω]` supplies the kernel-specific normalization. Invalid energies
are returned as zero columns in energy space.
"""
function _reconstruct_ldos_moment_columns(
    moments::AbstractMatrix{<:Real},
    W::AbstractMatrix{<:Real},
    denom::AbstractVector{<:Real},
    valid::AbstractVector{Bool},
)
    Ncheb, ncols = size(moments)
    size(W, 1) == Ncheb || throw(DimensionMismatch(
        "moment rows ($(size(moments, 1))) must match weight rows ($(size(W, 1))).",
    ))
    Nω = size(W, 2)
    length(denom) == Nω || throw(DimensionMismatch(
        "denominator length ($(length(denom))) must match energy count ($Nω).",
    ))
    length(valid) == Nω || throw(DimensionMismatch(
        "valid-mask length ($(length(valid))) must match energy count ($Nω).",
    ))

    result = zeros(Float64, Nω, ncols)
    mul!(result, transpose(W), moments)
    for iω in 1:Nω
        if valid[iω]
            view(result, iω, :) ./= denom[iω]
        else
            fill!(view(result, iω, :), 0.0)
        end
    end
    return result
end


# ============================================================
# 6. Energy grid of an online KPM reconstruction
# ============================================================

"""
    _rescaled_energies(H, ω_phys_vals) -> Vector

Physical energies mapped onto the Chebyshev interval, `(ω − H.center) / H.scale`
element by element.
"""
_rescaled_energies(H::TBHamiltonian, ω_phys_vals) =
    (collect(ω_phys_vals) .- H.center) ./ H.scale

"""
    _kpm_energy_grid(H, Ncheb, ω_phys_vals; kernel=:jackson, lambda=4.0, eta=0.0,
                     m_order=4, allow_hodc=false) -> (ω_r, W, denom, valid)
    _kpm_energy_grid(Ncheb, ω_r; <same keywords>)  -> (ω_r, W, denom, valid)

Everything an online KPM solver needs about its energies, before the recursion:

- `ω_r` — the rescaled energies (`_rescaled_energies(H, ω_phys_vals)`; the second
  method takes them already rescaled and returns them as given);
- `W[n, iω]` — the weight of moment `n-1` at `ω_r[iω]`, zero for `|ω_r| ≥ 1`;
- `denom[iω]` — the per-ω normalisation: the result is `Σ_n W[n, iω] μ_n / denom[iω]`;
- `valid[iω]` — `abs(ω_r[iω]) < 1.0`, the energies inside the spectral support.

For the convolution kernels (`:jackson`, `:lorentz` with `lambda`, `:fejer`,
`:dirichlet`) `W` is `_kpm_weight_matrix` and `denom = π²·Ncheb·√(1−ω²)`.
`allow_hodc=true` also accepts `kernel=:hodc` (`eta`, `m_order`; `eta=0` means
`1/(Ncheb+1)`), whose contour weights carry the normalisation (`denom = 1`); this is
`_dos_weight_matrix`. With `allow_hodc=false` (the solvers without an HODC option)
`kernel=:hodc` throws the unknown-kernel error of `_kpm_kernel`.
"""
function _kpm_energy_grid(Ncheb::Int, ω_r;
                          kernel::Symbol    = :jackson,
                          lambda::Real      = 4.0,
                          eta::Real         = 0.0,
                          m_order::Int      = 4,
                          allow_hodc::Bool  = false)
    W, denom = if allow_hodc
        _dos_weight_matrix(Ncheb, ω_r; kernel, lambda, eta, m_order)
    else
        (_kpm_weight_matrix(Ncheb, ω_r; kernel, lambda), _kpm_denominator(Ncheb, ω_r))
    end
    valid = [abs(ω) < 1.0 for ω in ω_r]
    return ω_r, W, denom, valid
end

_kpm_energy_grid(H::TBHamiltonian, Ncheb::Int, ω_phys_vals; kwargs...) =
    _kpm_energy_grid(Ncheb, _rescaled_energies(H, ω_phys_vals); kwargs...)
