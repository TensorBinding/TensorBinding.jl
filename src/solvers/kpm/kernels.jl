# solvers/kpm/kernels.jl — KPM damping kernels (_kpm_kernel), the Chebyshev-KPM
# weight matrix _kpm_weight_matrix, the HODC contour kernel helpers
# (compute_hodc_params, get_hodc_weights, get_hodc_gf_weights), the stochastic-DOS
# weight matrix _dos_weight_matrix and the moment-column LDOS reconstruction
# _reconstruct_ldos_moment_columns. Moved verbatim from the former solvers/KPM_tk.jl
# (Tier 1 split), physics/qft/bands.jl (_kpm_weight_matrix) and gpu/kpm.jl
# (_reconstruct_ldos_moment_columns).

# ============================================================
# KPM damping kernels
# ============================================================

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


# ============================================================
# Chebyshev-KPM weight matrix
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
# HODC kernel helpers
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
# Stochastic-DOS reconstruction weights
# ============================================================

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


# ============================================================
# Moment-column LDOS reconstruction
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
