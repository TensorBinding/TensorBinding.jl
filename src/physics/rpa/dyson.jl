# physics/rpa/dyson.jl — Random Phase Approximation: Dyson solve, Wynn-accelerated
# series and the magnon channel
#
# The working pipeline is:
#
#   1. get_bubble_mpo       — polarization bubble Π₀(ω) as an L-site MPO
#   2. rpa_from_bubble_diag — solve (I - Π₀V) χ = Π₀ for the RPA susceptibility
#
# Step 1 lives in bubble.jl and cheb2d.jl. This file holds step 2 and the drivers
# built on it: the Dyson linear solve (rpa_from_bubble_diag, get_rpa_susceptibility),
# the Wynn ε-accelerated Neumann series (wynn_epsilon, rpa_wynn_from_bubbles,
# get_rpa_susceptibility_wynn; its k-space readout get_spect_k is in
# physics/qft/conjugation.jl) and the transverse-spin channel (get_magnon_*; its
# spin-sector projector _project_spin_sector is in core/AuxDOF.jl). The three Wynn
# drivers are thin wrappers over one series kernel, _rpa_wynn_series.
# Split from the former physics/RPA_tk.jl.
#
# Entry points: get_rpa_susceptibility, get_rpa_susceptibility_wynn,
#   rpa_wynn_from_bubbles, rpa_from_bubble_diag, get_magnon_bubble,
#   get_magnon_susceptibility, get_magnon_susceptibility_wynn.
# Depends on: core/MPOTools.jl, core/TBSystem.jl, core/AuxDOF.jl,
#   solvers/Krylov.jl (_vec_mps_from_mpo), physics/rpa/bubble.jl,
#   physics/qft/conjugation.jl (see the source map in src/TensorBinding.jl).

# ============================================================
# 1. Dyson solve
# ============================================================

"""
    rpa_from_bubble_diag(Π, MPOV, finalsites, finalfinalsites;
                         nsweeps=20, maxdim=400, cutoff=1e-8, return_info=false,
                         residual_warn=1e-2) -> MPS

Solve the RPA Dyson equation  (I − Π₀V) χ = Π₀  for the interacting
susceptibility χ using DMRG-style linear solve.

`Π` (Π₀) and `MPOV` (V) are MPOs on the L indices `finalfinalsites`; `finalsites` are
2L indices, sites 2n−1 and 2n of the dimension of `finalfinalsites[n]`
(`_rpa_pair_sites`). Returns vec(χ), χ = (I − Π₀V)⁻¹Π₀ the full L-site response
matrix χ_ij, as a 2L-site MPS on `finalsites` in the layout of `_vec_mps_from_mpo`:
the row index i on the odd sites, the column index j on the even ones (site n of
`finalfinalsites` on sites 2n−1 and 2n), so that `custom_mpo(χ, finalfinalsites)` is
χ as an MPO. The solve is [(I − Π₀V) ⊗ I] vec(χ) = vec(Π₀), with I − Π₀V on the odd
sites, started from vec(Π₀).

Near the RPA instability I − Π₀V is nearly singular and the truncation at `cutoff`
limits the accuracy: on a 32-site half-filled chain with V = U·1 at U = 0.999 U_c
(condition number 10³) the default `cutoff = 1e-8` gives χ to 5e-3, `1e-12` to 1e-7.
The relative residual ‖(I − Π₀V)χ − Π₀‖/‖Π₀‖ of the solve is checked, for Π₀V as
truncated at `cutoff`: above `residual_warn` (default `1e-2`) a warning suggests a
smaller `cutoff`. `return_info=true` returns `(; chi, residual, residual_floor,
maxlinkdim)` instead of vec(χ); `residual` is an upper bound that includes the
rounding floor of its evaluation (`residual_floor`, about 1e-7 times the condition
number). The check contracts ⟨Aχ|Aχ⟩ once, at χ³w² + χ²w³ operations per site for bond
dimensions χ of vec(χ) and w of I − Π₀V: less than a sweep while w is small, more for a
high-rank Π₀V. It is skipped, with a message and `residual = NaN`, when it would hold
more than 2^27 numbers (χ²w² > 6.7e7); `residual_warn = Inf` skips it altogether.
"""
function rpa_from_bubble_diag(Π, MPOV, finalsites, finalfinalsites;
                               nsweeps=20, maxdim=400, cutoff=1e-8,
                               # The default cutoff stays 1e-8 on purpose (0.2.2): its error
                               # grows like κ·1e-5 near the instability (κ the condition number
                               # of I − Π₀V) and the residual check warns there. A tighter default
                               # would cost ~5× on every call and move every RPA result.
                               return_info::Bool=false, residual_warn::Real=1e-2)
    Id  = MPO(finalfinalsites, "Id")
    ΠV  = apply(Π, MPOV; maxdim=maxdim, cutoff=cutoff)
    A   = Id - ΠV

    # vec(χ) holds χ[i, j] with i on the odd and j on the even sites (_vec_mps_from_mpo,
    # the layout custom_mpo reads back), and (I − Π₀V)χ = Π₀ acts on i: A on the odd
    # sites. (Until the fix the right-hand side was diag(Π₀) on the even sites, the same
    # for every state of the odd ones, and the result the rank-1 array
    # [(I − Π₀V)⁻¹ diag Π₀]_j in every row i, not χ.)
    Aop = interleave_mpo(A, finalsites, 1)
    b   = _vec_mps_from_mpo(Π, finalsites)

    # Π₀ = 0: χ = 0, and a residual relative to ‖Π₀‖ is undefined.
    iszero(norm(b)) &&
        return return_info ? (; chi = b, residual = 0.0, residual_floor = 0.0,
                                maxlinkdim = maxlinkdim(b)) : b

    x0 = deepcopy(b)
    χ  = ITensorMPS.linsolve(Aop, b, x0;
                             nsweeps=nsweeps, maxdim=maxdim, cutoff=cutoff)
    (return_info || isfinite(residual_warn)) || return χ

    info = _linsolve_residual(Aop, b, χ, 0.0, 1.0; threshold = residual_warn)
    res  = info.residual
    if info.method === :skipped
        isfinite(residual_warn) && _residual_skipped_info("rpa_from_bubble_diag", χ, Aop,
            "χ is returned unchecked; residual_warn = Inf (rpa_residual_warn) silences this.")
    else
        res <= residual_warn ||
            @warn "rpa_from_bubble_diag: the Dyson solve is not converged: relative residual " *
                  "$(round(res; sigdigits=2)) > residual_warn = $residual_warn. Close to the RPA " *
                  "instability lower `cutoff` (`rpa_cutoff`), e.g. to 1e-12; else raise " *
                  "`nsweeps` or `maxdim` (maxlinkdim reached: $(maxlinkdim(χ)))."
    end
    return return_info ? (; chi = χ, residual = res, residual_floor = info.floor,
                            maxlinkdim = maxlinkdim(χ)) : χ
end


"""
    _rpa_pair_sites(out_sites) -> Vector{Index}

The `2L`-site `finalsites` for `rpa_from_bubble_diag` when Π₀ lives on the `L`
indices `out_sites`: sites `2n-1` and `2n` both take `dim(out_sites[n])`, so
`interleave_mpo` meets no dimension mismatch on a Kagome/Lieb sublattice or a
Layer index. Dim-2 sites are the `"Qubit"` sites `siteinds("Qubit", 2L)` gives.
"""
function _rpa_pair_sites(out_sites)
    return [dim(out_sites[cld(n, 2)]) == 2 ? siteind("Qubit", n) :
                Index(dim(out_sites[cld(n, 2)]), "Site,n=$n")
            for n in 1:2 * length(out_sites)]
end

# ============================================================
# 2. High-level TBHamiltonian API
# ============================================================

"""
    get_rpa_susceptibility(H::TBHamiltonian, MPOV, ω; mode=:charge, rpa_nsweeps=20,
                           rpa_maxdim=400, rpa_cutoff=1e-8, rpa_residual_warn=1e-2,
                           return_info=false, <get_bubble_mpo keywords>) -> MPS

Compute the RPA susceptibility χ(ω) for a system described by `H` with
interaction MPO `MPOV`.  Returns χ = (I − Π₀V)⁻¹Π₀, the full response matrix
χ_ij on the sites of Π₀, vectorized as the 2L-site MPS of `rpa_from_bubble_diag`
(row index i on the odd sites, column index j on the even ones; `custom_mpo(χ, sites)`
turns it into an MPO on those `sites`).

**`mode` keyword**
- `:charge` (default) — density–density bubble χ^{ρρ}: calls
  `get_bubble_mpo(H, H, ω)`.
- `:magnetic` — transverse spin bubble χ^{+−}: projects H onto its
  spin-↑ and spin-↓ blocks and calls `get_magnon_bubble`.
  Requires `H.spin_s !== nothing`. `MPOV` and the result then live on the
  spin-projected sites (`H.sites` without the spin index), as in
  `get_magnon_susceptibility`.

Internally solves the Dyson equation (I − Π₀V) χ = Π₀.
All `get_bubble_mpo` keyword arguments are accepted and forwarded.

**Additional keywords (Dyson solve)**
- `rpa_nsweeps` : sweeps for the RPA linsolve. Default `20`.
- `rpa_maxdim`  : max bond dim for the RPA linsolve. Default `400`.
- `rpa_cutoff`  : cutoff for the RPA linsolve. Default `1e-8`. Close to the RPA
  instability use a smaller one (see `rpa_from_bubble_diag`).
- `rpa_residual_warn` : warn when the relative residual of the Dyson solve exceeds
  this value (`Inf`: no check). Default `1e-2`.
- `return_info` : return `(; chi, residual, maxlinkdim)` of the Dyson solve instead of
  vec(χ). Default `false`.
"""
function get_rpa_susceptibility(H::TBHamiltonian, MPOV::MPO, ω::Real;
                                 mode::Symbol          = :charge,
                                 rpa_nsweeps::Int      = 20,
                                 rpa_maxdim::Int       = 400,
                                 # 1e-8 on purpose, an accuracy floor: see rpa_from_bubble_diag.
                                 rpa_cutoff::Real      = 1e-8,
                                 rpa_residual_warn::Real = 1e-2,
                                 return_info::Bool     = false,
                                 ϵF::Real              = 0.0,
                                 P_method::Symbol      = :purification,
                                 GF_method::Symbol     = :kpm,
                                 Ncheb::Int            = 150,
                                 maxdim::Int           = 200,
                                 cutoff::Real          = 1e-8,
                                 purify_method::Symbol = :mcweeny,
                                 purify_maxdim::Int    = 40,
                                 purify_maxiters::Int  = 30,
                                 purify_tol::Float64   = 1e-5,
                                 η::Real               = 1e-3,
                                 krylov_nsweeps::Int   = 12,
                                 krylov_maxdim::Int    = 100,
                                 krylov_cutoff::Real   = 1e-8,
                                 verbose::Bool         = false)

    bubble_kw = (; ϵF, P_method, GF_method, Ncheb, maxdim, cutoff,
                   purify_method, purify_maxdim, purify_maxiters, purify_tol,
                   η, krylov_nsweeps, krylov_maxdim, krylov_cutoff, verbose)

    if mode == :charge
        Π         = get_bubble_mpo(H, H, ω; bubble_kw...)
        out_sites = H.sites
    elseif mode == :magnetic
        H.spin_s === nothing &&
            error("get_rpa_susceptibility: mode=:magnetic requires a spinful H (call add_spin!(H) first)")
        H_up      = _project_spin_sector(H, 1)
        H_dn      = _project_spin_sector(H, 2)
        Π         = get_bubble_mpo(H_up, H_dn, ω; bubble_kw...)
        out_sites = H_up.sites
    else
        error("get_rpa_susceptibility: unknown mode=$mode. Choose :charge or :magnetic")
    end

    # Π lives on out_sites: for :magnetic these are the spin-projected H_up.sites,
    # one fewer than the spinful H.sites.
    finalsites = _rpa_pair_sites(out_sites)
    return rpa_from_bubble_diag(Π, MPOV, finalsites, out_sites;
                                 nsweeps=rpa_nsweeps, maxdim=rpa_maxdim, cutoff=rpa_cutoff,
                                 residual_warn=rpa_residual_warn, return_info)
end


# The get_bubble_mpo keywords of get_rpa_susceptibility and their defaults there, for
# get_rpa_susceptibility_column (which must give a column of the same χ).
const _RPA_BUBBLE_DEFAULTS = (; ϵF = 0.0, P_method = :purification, GF_method = :kpm,
    Ncheb = 150, maxdim = 200, cutoff = 1e-8, purify_method = :mcweeny, purify_maxdim = 40,
    purify_maxiters = 30, purify_tol = 1e-5, η = 1e-3, krylov_nsweeps = 12,
    krylov_maxdim = 100, krylov_cutoff = 1e-8, verbose = false)

# One column of the Dyson solution: (I − Π₀V)x = Π₀|b⟩ on the L sites of Π₀, from x = Π₀|b⟩.
function _rpa_column_solve(Π::MPO, MPOV::MPO, sites, b::MPS; nsweeps::Int, maxdim::Int,
                           cutoff::Real, residual_warn::Real, return_info::Bool)
    A   = MPO(sites, "Id") - apply(Π, MPOV; maxdim=maxdim, cutoff=cutoff)
    rhs = apply(Π, b; maxdim=maxdim, cutoff=min(cutoff, 1e-14))
    # Π₀|b⟩ = 0: x = 0, and a residual relative to ‖Π₀|b⟩‖ is undefined.
    iszero(norm(rhs)) &&
        return return_info ? (; chi = rhs, residual = 0.0, residual_floor = 0.0,
                                maxlinkdim = maxlinkdim(rhs)) : rhs
    x   = ITensorMPS.linsolve(A, rhs, deepcopy(rhs); nsweeps, maxdim, cutoff)
    (return_info || isfinite(residual_warn)) || return x
    info = _linsolve_residual(A, rhs, x, 0.0, 1.0; threshold = residual_warn)
    res  = info.residual
    if info.method === :skipped
        isfinite(residual_warn) && _residual_skipped_info("get_rpa_susceptibility_column", x, A,
            "The column is returned unchecked; rpa_residual_warn = Inf silences this.")
    else
        res <= residual_warn ||
            @warn "get_rpa_susceptibility_column: the Dyson solve is not converged: relative " *
                  "residual $(round(res; sigdigits=2)) > rpa_residual_warn = $residual_warn. " *
                  "Lower `rpa_cutoff` or raise `rpa_nsweeps` / `rpa_maxdim` (maxlinkdim " *
                  "reached: $(maxlinkdim(x)))."
    end
    return return_info ? (; chi = x, residual = res, residual_floor = info.floor,
                            maxlinkdim = maxlinkdim(x)) : x
end

"""
    get_rpa_susceptibility_column(H::TBHamiltonian, MPOV, ω, j; mode=:charge,
                                  rpa_nsweeps=40, rpa_maxdim=400, rpa_cutoff=1e-12,
                                  rpa_residual_warn=1e-2, return_info=false,
                                  <get_bubble_mpo keywords>) -> MPS

One column of the RPA susceptibility χ = (I − Π₀V)⁻¹Π₀: the L-site MPS x = χ|j⟩, with
χ_ij = ⟨i|x⟩, from the Dyson equation (I − Π₀V)x = Π₀|j⟩ on the sites of Π₀. `j` is
the 1-indexed basis state of those sites (`_basis_state_mps(j − 1, sites)`: the
position `j` when they are position qubits only), or any MPS on them (then x = χ|j⟩).

Where one site's response is enough (χ_jj, a local probe, one column of a real-space
response) this is much cheaper than `get_rpa_susceptibility`, which solves for the
whole χ on the 2L-site vectorized register, and it holds its accuracy close to the
instability: on a 256-site half-filled chain up to U = 0.999 U_c (condition number
10³), χ_jj to 1e-7–2e-6 and the column to 3e-6 in about a second, against minutes for
the whole χ. The tighter defaults (`rpa_nsweeps = 40`, `rpa_cutoff = 1e-12`) cost
little on L sites.

The bubble is that of `get_rpa_susceptibility` with the same keywords and defaults
(`mode`, `ϵF`, `P_method`, `GF_method`, `Ncheb`, `maxdim`, `cutoff`, `η`, …), so the
result is column `j` of its χ. The residual check and `return_info` (returning
`(; chi, residual, maxlinkdim)`) are those of `rpa_from_bubble_diag`.
"""
function get_rpa_susceptibility_column(H::TBHamiltonian, MPOV::MPO, ω::Real, j;
                                       mode::Symbol            = :charge,
                                       rpa_nsweeps::Int        = 40,
                                       rpa_maxdim::Int         = 400,
                                       rpa_cutoff::Real        = 1e-12,
                                       rpa_residual_warn::Real = 1e-2,
                                       return_info::Bool       = false,
                                       kwargs...)
    bubble_kw = merge(_RPA_BUBBLE_DEFAULTS, kwargs)
    if mode == :charge
        Π         = get_bubble_mpo(H, H, ω; bubble_kw...)
        out_sites = H.sites
    elseif mode == :magnetic
        H.spin_s === nothing &&
            error("get_rpa_susceptibility_column: mode=:magnetic requires a spinful H (call add_spin!(H) first)")
        H_up      = _project_spin_sector(H, 1)
        H_dn      = _project_spin_sector(H, 2)
        Π         = get_bubble_mpo(H_up, H_dn, ω; bubble_kw...)
        out_sites = H_up.sites
    else
        error("get_rpa_susceptibility_column: unknown mode=$mode. Choose :charge or :magnetic")
    end
    b = j isa MPS ? j : begin
        D = prod(dim.(out_sites))
        1 <= j <= D || throw(BoundsError(1:D, j))
        _basis_state_mps(j - 1, out_sites)
    end
    return _rpa_column_solve(Π, MPOV, out_sites, b; nsweeps=rpa_nsweeps, maxdim=rpa_maxdim,
                             cutoff=rpa_cutoff, residual_warn=rpa_residual_warn,
                             return_info)
end

# ============================================================
# 3. Wynn ε-algorithm accelerated RPA
# ============================================================

"""
    wynn_epsilon(s) -> Vector{ComplexF64}

Wynn ε-algorithm applied to a scalar sequence `s = [s₀, s₁, ..., sK]`.
Returns the even-column first-row Padé estimates `[ε₂(0), ε₄(0), ...]`.
Uses only additions and reciprocals — no matrix operations.

Two equal entries (|difference| < 1e-30) make the next entry infinite, stored as the
sentinel `1e30`. Two infinite entries in a column make the correction 1/(∞ − ∞)
vanish, so the entry after them is the one two columns back: an exactly converged
sequence returns its limit, not the sentinel. A sentinel is still returned where
the table is genuinely singular (e.g. an arithmetic sequence).
"""
function wynn_epsilon(s::AbstractVector{<:Number})
    n        = length(s)
    sentinel = complex(1e30)
    eps      = zeros(ComplexF64, n+1, n+1)
    for k in 1:n
        eps[2, k] = s[k]
    end
    for j in 1:n-1
        for k in 1:n-j
            d = eps[j+1, k+1] - eps[j+1, k]
            eps[j+2, k] = abs(d) < 1e-30 ?
                          (eps[j+1, k] == sentinel ? eps[j, k+1] : sentinel) :
                          eps[j, k+1] + 1/d
        end
    end
    return [eps[j+2, 1] for j in 2:2:n-1]
end


# The Wynn-accelerated Neumann series behind rpa_wynn_from_bubbles,
# get_rpa_susceptibility_wynn and get_magnon_susceptibility_wynn. For the i-th entry x
# of `items` (a bubble MPO or a frequency) it prints `label(i, x)` when verbose, takes
# Π₀ = bubble(x) and builds T₀ = Π₀, Tₙ = Tₙ₋₁·V·Π₀ (n = 1…K_max, each product
# truncated to maxdim_apply/cutoff_apply), reads out sₙ(q) = −Im⟨q|Tₙ|q⟩ with
# get_spect_k, and stores the partial sums Σₙ₌₀ᵏ sₙ and their Wynn ε estimates.
# Returns (chi_partial, chi_wynn) with the layout the public functions document;
# (nothing, nothing) when `items` is empty.
function _rpa_wynn_series(bubble, items, MPOV::MPO; K_max::Int, maxdim_apply::Int,
                          cutoff_apply::Real, verbose::Bool, label)
    nω     = length(items)
    n_wynn = K_max ÷ 2

    chi_partial = nothing
    chi_wynn    = nothing
    slow        = Tuple{Int,Float64}[]          # (item, ρ estimate) of slow series

    for (i, x) in enumerate(items)
        verbose && println(label(i, x))

        Π0   = bubble(x)
        term = deepcopy(Π0)
        s0   = -imag.(get_spect_k(term))
        nq   = length(s0)

        if chi_partial === nothing
            chi_partial = zeros(Float64, K_max+1, nω, nq)
            chi_wynn    = zeros(Float64, n_wynn,  nω, nq)
        end

        # Individual term contributions: spect_terms[n+1, q]
        spect_terms       = zeros(Float64, K_max+1, nq)
        spect_terms[1, :] = s0
        tnorm             = zeros(Float64, K_max+1)      # ‖Tₙ‖_F
        tnorm[1]          = norm(term)

        for n in 1:K_max
            term                = apply(term, MPOV; maxdim=maxdim_apply, cutoff=cutoff_apply)
            term                = apply(term, Π0;   maxdim=maxdim_apply, cutoff=cutoff_apply)
            spect_terms[n+1, :] = -imag.(get_spect_k(term))
            tnorm[n+1]          = norm(term)
        end
        # ‖Tₙ‖/‖Tₙ₋₁‖ tends to the spectral radius ρ of VΠ₀; over the last two steps:
        if K_max >= 2 && tnorm[K_max-1] > 0
            ρ = sqrt(tnorm[K_max+1] / tnorm[K_max-1])
            ρ > 0.9 && push!(slow, (i, ρ))
        end

        # Partial sums (Wynn input)
        partial_sums         = cumsum(spect_terms; dims=1)
        chi_partial[:, i, :] = partial_sums

        # Apply Wynn ε per q-point
        for q in 1:nq
            ests = wynn_epsilon(complex.(partial_sums[:, q]))
            for m in 1:min(n_wynn, length(ests))
                chi_wynn[m, i, q] = real(ests[m])
            end
        end

        verbose && println("  done ($(K_max+1) terms, $n_wynn Wynn estimates)")
    end

    # Near ρ(VΠ₀) = 1 (the RPA instability, and at finite ω, η already below it) the
    # Neumann series converges slowly or diverges, and a few Wynn steps do not fix it:
    # with K_max = 6 the estimate was 1–90 % off at ρ ≈ 1 on a half-filled chain.
    if !isempty(slow)
        ρmax = maximum(last, slow)
        @warn "RPA Wynn series: ‖Tₙ‖/‖Tₙ₋₁‖ ≈ $(round(ρmax; sigdigits=3)) > 0.9 at " *
              "$(length(slow)) of $nω frequencies (the spectral radius of VΠ₀): the Neumann " *
              "series converges slowly or diverges there, and the Wynn estimates with " *
              "K_max = $K_max can be off by tens of %. Use a larger K_max or the Dyson " *
              "solve (get_rpa_susceptibility, get_rpa_susceptibility_column)."
    end
    return chi_partial, chi_wynn
end


"""
    rpa_wynn_from_bubbles(Π0_list, MPOV; K_max=6, maxdim_apply=200, cutoff_apply=1e-8,
                          verbose=false) -> (chi_partial, chi_wynn)

Wynn ε-accelerated RPA susceptibility from a pre-computed list of bubble MPOs.

Accepts the output of `get_bubble_mpo_cheb2d_tucker` (or any `Vector{MPO}`) directly,
skipping the internal bubble calculation.  All other logic is identical to
`get_rpa_susceptibility_wynn`: Neumann series T₀ = Π₀, Tₙ = Tₙ₋₁·V·Π₀, followed
by per-k-point Wynn ε-acceleration of the partial-sum sequence.

**Returns**
- `chi_partial[k+1, iω, q]` : partial sum Σₙ₌₀ᵏ (−Im⟨q|Tₙ(ω)|q⟩)
- `chi_wynn[m, iω, q]`      : Wynn ε_{2m}(0) estimate

**Keyword arguments**
- `K_max`         : series order (K_max+1 terms). Default `6`.
- `maxdim_apply`  : bond dim for Tₙ·V·Π₀ products. Default `200`.
- `cutoff_apply`  : truncation cutoff for those products. Default `1e-8`.
- `verbose`       : print per-ω progress. Default `false`.
"""
function rpa_wynn_from_bubbles(Π0_list::Vector{<:MPO}, MPOV::MPO;
                                K_max::Int         = 6,
                                maxdim_apply::Int  = 200,
                                cutoff_apply::Real = 1e-8,
                                verbose::Bool      = false)
    nω = length(Π0_list)
    return _rpa_wynn_series(identity, Π0_list, MPOV; K_max, maxdim_apply, cutoff_apply,
                            verbose, label=(i, _) -> "rpa_wynn_from_bubbles: bubble $i/$nω")
end


"""
    get_rpa_susceptibility_wynn(H, MPOV, ωlist; mode=:charge, K_max=6,
                                 maxdim_apply=200, cutoff_apply=1e-8, verbose=false,
                                 <get_bubble_mpo keywords>) -> (chi_partial, chi_wynn)

Compute the RPA susceptibility χ_RPA(q,ω) for all frequencies in `ωlist` using
the Wynn ε-algorithm for Padé acceleration of the geometric (bubble) series.

**`mode` keyword**
- `:charge` (default) — density–density channel; uses `get_bubble_mpo(H, H, ω)`.
- `:magnetic` — transverse spin channel S⁺S⁻; uses `get_magnon_bubble(H, ω)`.
  The spin-↑/↓ projections are performed once before the ω loop.
  Requires `H.spin_s !== nothing`.

**Key idea**: instead of inverting (I − Π₀V), build the Neumann series
  T₀ = Π₀,  Tₙ = Tₙ₋₁·V·Π₀  (so Σ Tₙ → χ_RPA as K→∞),
extract scalars `sₙ(q,ω) = −Im⟨q|Tₙ(ω)|q⟩` (no 1/π factor) via `get_spect_k`, and apply
Wynn ε to the partial-sum sequence per (q,ω) for fast convergence.

**Returns**
- `chi_partial[k+1, i_ω, q]` : partial sum Σₙ₌₀ᵏ sₙ(q,ω)
- `chi_wynn[m, i_ω, q]`      : Wynn ε_{2m}(0) estimate (uses 2m+1 terms)

**Keyword arguments**
- `mode`          : `:charge` (default) or `:magnetic`.
- `K_max`         : highest order in the series (total K_max+1 terms). Default `6`.
- `maxdim_apply`  : bond dim for the Tₙ·V·Π₀ products. Default `200`.
- `cutoff_apply`  : truncation cutoff for those products. Default `1e-8`.
- `verbose`       : print per-ω progress. Default `false`.
- All `get_bubble_mpo` keywords (`ϵF`, `P_method`, `GF_method`, `Ncheb`,
  `maxdim`, `cutoff`, `purify_*`, `η`, `krylov_*`) are accepted and forwarded.
"""
function get_rpa_susceptibility_wynn(H::TBHamiltonian, MPOV::MPO,
                                      ωlist::AbstractVector{<:Real};
                                      mode::Symbol       = :charge,
                                      K_max::Int         = 6,
                                      maxdim_apply::Int  = 200,
                                      cutoff_apply::Real = 1e-8,
                                      ϵF::Real              = 0.0,
                                      P_method::Symbol      = :purification,
                                      GF_method::Symbol     = :kpm,
                                      Ncheb::Int            = 150,
                                      maxdim::Int           = 200,
                                      cutoff::Real          = 1e-8,
                                      purify_method::Symbol = :mcweeny,
                                      purify_maxdim::Int    = 40,
                                      purify_maxiters::Int  = 30,
                                      purify_tol::Float64   = 1e-5,
                                      η::Real               = 1e-3,
                                      krylov_nsweeps::Int   = 12,
                                      krylov_maxdim::Int    = 100,
                                      krylov_cutoff::Real   = 1e-8,
                                      verbose::Bool         = false)

    mode ∈ (:charge, :magnetic) ||
        error("get_rpa_susceptibility_wynn: unknown mode=$mode. Choose :charge or :magnetic")
    mode == :magnetic && H.spin_s === nothing &&
        error("get_rpa_susceptibility_wynn: mode=:magnetic requires a spinful H (call add_spin!(H) first)")

    bubble_kw = (; ϵF, P_method, GF_method, Ncheb, maxdim, cutoff,
                   purify_method, purify_maxdim, purify_maxiters, purify_tol,
                   η, krylov_nsweeps, krylov_maxdim, krylov_cutoff, verbose)

    # For :magnetic, project spin sectors once before the ω loop
    H_up = mode == :magnetic ? _project_spin_sector(H, 1) : nothing
    H_dn = mode == :magnetic ? _project_spin_sector(H, 2) : nothing

    nω     = length(ωlist)
    bubble = mode == :charge ? (ω -> get_bubble_mpo(H, H, ω; bubble_kw...)) :
                               (ω -> get_bubble_mpo(H_up, H_dn, ω; bubble_kw...))
    return _rpa_wynn_series(bubble, ωlist, MPOV; K_max, maxdim_apply, cutoff_apply, verbose,
                            label=(i, ω) -> "Wynn RPA (mode=$mode): ω $i/$nω  (ω = $ω)")
end

# ============================================================
# 4. Magnon susceptibility (transverse S⁺S⁻ spin channel)
# ============================================================

"""
    get_magnon_bubble(H, ω; kwargs...) -> MPO

Non-interacting transverse spin polarization bubble Π₀^{+−}(ω) for a
spinful `TBHamiltonian`.

The spin degree of freedom is projected out analytically: the spin-↑ and
spin-↓ blocks of `H` become two independent L-qubit Hamiltonians `H_↑`
and `H_↓`, which are passed to `get_bubble_mpo(H_↑, H_↓, ω)`.
This corresponds to the Kubo S⁺S⁻ bubble

    Π₀^{+−}(ω) = ∑_k (f_{k↓} − f_{k↑}) / (ω − (ε_{k↓} − ε_{k↑}) + iη)

Errors if `H.spin_s === nothing`.  All `get_bubble_mpo` keyword arguments
are accepted and forwarded.
"""
function get_magnon_bubble(H::TBHamiltonian, ω::Real; kwargs...)
    H.spin_s === nothing &&
        error("get_magnon_bubble: H is not spinful — call add_spin!(H) first")
    H_up = _project_spin_sector(H, 1)
    H_dn = _project_spin_sector(H, 2)
    return get_bubble_mpo(H_up, H_dn, ω; kwargs...)
end


"""
    get_magnon_susceptibility(H, MPOV, ω; rpa_nsweeps=20, rpa_maxdim=400,
                              rpa_cutoff=1e-8, rpa_residual_warn=1e-2,
                              return_info=false, kwargs...) -> MPS

RPA transverse spin susceptibility χ^{+−}_RPA(ω) for a spinful
`TBHamiltonian`.

Builds Π₀^{+−}(ω) via `get_magnon_bubble`, then solves the Dyson
equation (I − Π₀ V) χ = Π₀.  For a Hubbard-like interaction the
interaction MPO is `MPOV = U · Id` on the orbital sites. Returns the full matrix χ
on the spin-projected sites (`H.sites` without the spin index), vectorized as in
`rpa_from_bubble_diag` (row index on the odd sites, column index on the even ones).

**Keyword arguments**
- `rpa_nsweeps`, `rpa_maxdim`, `rpa_cutoff` : Dyson linsolve parameters.
- `rpa_residual_warn`, `return_info` : the residual check of the Dyson solve, as
  `residual_warn` and `return_info` of `rpa_from_bubble_diag`.
- All `get_bubble_mpo` keywords forwarded via `kwargs...`.
"""
function get_magnon_susceptibility(H::TBHamiltonian, MPOV::MPO, ω::Real;
                                   rpa_nsweeps::Int         = 20,
                                   rpa_maxdim::Int          = 400,
                                   # 1e-8 on purpose, an accuracy floor: see rpa_from_bubble_diag.
                                   rpa_cutoff::Real         = 1e-8,
                                   rpa_residual_warn::Real  = 1e-2,
                                   return_info::Bool        = false,
                                   kwargs...)
    H.spin_s === nothing &&
        error("get_magnon_susceptibility: H is not spinful — call add_spin!(H) first")
    H_up = _project_spin_sector(H, 1)
    H_dn = _project_spin_sector(H, 2)
    Π = get_bubble_mpo(H_up, H_dn, ω; kwargs...)
    # H.L counts position qubits only; H_up.sites also keeps any sublattice/layer index.
    finalsites = _rpa_pair_sites(H_up.sites)
    return rpa_from_bubble_diag(Π, MPOV, finalsites, H_up.sites;
                                nsweeps=rpa_nsweeps, maxdim=rpa_maxdim, cutoff=rpa_cutoff,
                                residual_warn=rpa_residual_warn, return_info)
end


"""
    get_magnon_susceptibility_wynn(H, MPOV, ωlist; K_max=6, maxdim_apply=200,
                                   cutoff_apply=1e-8, verbose=false,
                                   kwargs...) -> (chi_partial, chi_wynn)

Wynn ε-accelerated transverse spin RPA susceptibility over a frequency list.

The spin-↑/↓ sector projections are performed once outside the ω loop;
the Neumann-series / Wynn logic is identical to `get_rpa_susceptibility_wynn`.

Returns `(chi_partial, chi_wynn)` with the same layout as
`get_rpa_susceptibility_wynn`.

**Keyword arguments**
- `K_max`, `maxdim_apply`, `cutoff_apply`, `verbose` : series / Wynn control.
- All `get_bubble_mpo` keywords forwarded via `kwargs...`.
"""
function get_magnon_susceptibility_wynn(H::TBHamiltonian, MPOV::MPO,
                                         ωlist::AbstractVector{<:Real};
                                         K_max::Int         = 6,
                                         maxdim_apply::Int  = 200,
                                         cutoff_apply::Real = 1e-8,
                                         verbose::Bool      = false,
                                         kwargs...)
    H.spin_s === nothing &&
        error("get_magnon_susceptibility_wynn: H is not spinful — call add_spin!(H) first")

    H_up = _project_spin_sector(H, 1)
    H_dn = _project_spin_sector(H, 2)

    nω = length(ωlist)
    return _rpa_wynn_series(ω -> get_bubble_mpo(H_up, H_dn, ω; verbose, kwargs...),
                            ωlist, MPOV; K_max, maxdim_apply, cutoff_apply, verbose,
                            label=(i, ω) -> "Magnon Wynn RPA: ω $i/$nω  (ω = $ω)")
end
