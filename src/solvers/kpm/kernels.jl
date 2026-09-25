# solvers/kpm/kernels.jl — KPM damping kernels (_kpm_kernel), the HODC contour
# kernel helpers (compute_hodc_params, get_hodc_weights, get_hodc_gf_weights) and
# the stochastic-DOS weight matrix _dos_weight_matrix. Moved verbatim from
# solvers/KPM_tk.jl (Tier 1 split); _kpm_weight_matrix still lives in QFT_tk.jl.

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
