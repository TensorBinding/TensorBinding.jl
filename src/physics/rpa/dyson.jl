# physics/rpa/dyson.jl — Random Phase Approximation (polarization bubble + Dyson inversion)
#
# The working pipeline is:
#
#   1. get_bubble_mpo       — polarization bubble Π₀(ω) as an L-site MPO
#   2. rpa_from_bubble_diag — solve (I - Π₀V) χ = Π₀ for the RPA susceptibility
#
# Step 1 lives in bubble.jl and cheb2d.jl. This file holds step 2 and the drivers
# built on it: the Dyson linear solve (rpa_from_bubble_diag, get_rpa_susceptibility),
# the Wynn ε-accelerated Neumann series (wynn_epsilon, rpa_wynn_from_bubbles,
# get_spect_k, get_rpa_susceptibility_wynn) and the transverse-spin channel
# (_project_spin_sector, get_magnon_*).
# Split verbatim from physics/RPA_tk.jl.

# ============================================================
# Public RPA pipeline
# ============================================================

"""
    rpa_from_bubble_diag(Π, MPOV, finalsites, finalfinalsites;
                         nsweeps, maxdim, cutoff) -> MPS

Solve the RPA Dyson equation  (I − Π₀V) χ = Π₀  for the interacting
susceptibility χ using DMRG-style linear solve.

Returns a 2L-site MPS encoding the diagonal χ_{iijj}^RPA.
"""
function rpa_from_bubble_diag(Π, MPOV, finalsites, finalfinalsites;
                               nsweeps=20, maxdim=400, cutoff=1e-8)
    L   = length(finalfinalsites)
    Id  = MPO(finalfinalsites, "Id")
    ΠV  = apply(Π, MPOV; maxdim=maxdim, cutoff=cutoff)
    A   = Id - ΠV

    Aop = interleave_mpo(A, finalsites, 0)
    Πop = interleave_mpo(Π, finalsites, 0)
    b   = extract_diagonal_to_mps(Πop)

    # Align site indices between Aop and b
    for j in 1:2L
        sA = siteind(Aop, j)
        sb = siteind(b, j)
        if sb != sA
            replaceinds!(b[j], sb => noprime(sb))
        end
    end

    x0 = deepcopy(b)
    return ITensorMPS.linsolve(Aop, b, x0;
                               nsweeps=nsweeps, maxdim=maxdim, cutoff=cutoff)
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
# High-level TBHamiltonian API
# ============================================================

"""
    get_rpa_susceptibility(H::TBHamiltonian, MPOV, ω; mode, ...) -> MPS

Compute the RPA susceptibility χ(ω) for a system described by `H` with
interaction MPO `MPOV`.  Returns a 2L-site MPS encoding the diagonal
χ_{iijj}^RPA.

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
- `rpa_cutoff`  : cutoff for the RPA linsolve. Default `1e-8`.
"""
function get_rpa_susceptibility(H::TBHamiltonian, MPOV::MPO, ω::Real;
                                 mode::Symbol          = :charge,
                                 rpa_nsweeps::Int      = 20,
                                 rpa_maxdim::Int       = 400,
                                 rpa_cutoff::Real      = 1e-8,
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
                                 nsweeps=rpa_nsweeps, maxdim=rpa_maxdim, cutoff=rpa_cutoff)
end

# ============================================================
# Wynn ε-algorithm accelerated RPA
# ============================================================

"""
    wynn_epsilon(s) -> Vector{ComplexF64}

Wynn ε-algorithm applied to a scalar sequence `s = [s₀, s₁, ..., sK]`.
Returns the even-column first-row Padé estimates `[ε₂(0), ε₄(0), ...]`.
Uses only additions and reciprocals — no matrix operations.
"""
function wynn_epsilon(s::AbstractVector{<:Number})
    n   = length(s)
    eps = zeros(ComplexF64, n+1, n+1)
    for k in 1:n
        eps[2, k] = s[k]
    end
    for j in 1:n-1
        for k in 1:n-j
            d = eps[j+1, k+1] - eps[j+1, k]
            eps[j+2, k] = abs(d) < 1e-30 ? complex(1e30) : eps[j, k+1] + 1/d
        end
    end
    return [eps[j+2, 1] for j in 2:2:n-1]
end


"""
    rpa_wynn_from_bubbles(Π0_list, MPOV; K_max, maxdim_apply, cutoff_apply, verbose)
    -> (chi_partial, chi_wynn)

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
    nω     = length(Π0_list)
    n_wynn = K_max ÷ 2

    chi_partial = nothing
    chi_wynn    = nothing

    for (i, Π0) in enumerate(Π0_list)
        verbose && println("rpa_wynn_from_bubbles: bubble $i/$nω")
        term = deepcopy(Π0)
        s0   = -imag.(get_spect_k(term))

        if chi_partial === nothing
            nq          = length(s0)
            chi_partial = zeros(Float64, K_max+1, nω, nq)
            chi_wynn    = zeros(Float64, n_wynn,  nω, nq)
        end

        spect_terms       = zeros(Float64, K_max+1, length(s0))
        spect_terms[1, :] = s0

        for n in 1:K_max
            term                = apply(term, MPOV; maxdim=maxdim_apply, cutoff=cutoff_apply)
            term                = apply(term, Π0;   maxdim=maxdim_apply, cutoff=cutoff_apply)
            spect_terms[n+1, :] = -imag.(get_spect_k(term))
        end

        partial_sums         = cumsum(spect_terms; dims=1)
        chi_partial[:, i, :] = partial_sums

        for q in 1:length(s0)
            ests = wynn_epsilon(complex.(partial_sums[:, q]))
            for m in 1:min(n_wynn, length(ests))
                chi_wynn[m, i, q] = real(ests[m])
            end
        end

        verbose && println("  done ($(K_max+1) terms, $n_wynn Wynn estimates)")
    end

    return chi_partial, chi_wynn
end


"""
    get_spect_k(W; tol, maxdim) -> Vector{ComplexF64}

Extract the k-space diagonal of MPO `W` as a dense vector of 2^L values.

Conjugates `W` by the QFT (giving W̃ = QFT·W·QFT†), extracts the diagonal
as an MPS, then evaluates each ⟨k|W̃|k⟩ for the 2^L quantics k-indices.
Uses LSB-first quantics convention (site 1 = least significant bit).
"""
function get_spect_k(W::MPO; tol::Real=1e-9, maxdim::Int=100)
    Wk   = conjugate_by_qft(W; tol=tol, maxdim=maxdim)
    diag = extract_diagonal_to_mps(Wk)
    L    = length(diag)
    N    = 2^L
    sd   = siteinds(diag)
    return ComplexF64[inner(MPS(sd, [string((k >> (i-1)) & 1) for i in 1:L]), diag)
                      for k in 0:N-1]
end


"""
    get_rpa_susceptibility_wynn(H, MPOV, ωlist; mode, K_max, maxdim_apply,
                                 cutoff_apply, verbose, <bubble kwargs>)
                                 -> (chi_partial, chi_wynn)

Compute the RPA susceptibility χ_RPA(q,ω) for all frequencies in `ωlist` using
the Wynn ε-algorithm for Padé acceleration of the geometric (bubble) series.

**`mode` keyword**
- `:charge` (default) — density–density channel; uses `get_bubble_mpo(H, H, ω)`.
- `:magnetic` — transverse spin channel S⁺S⁻; uses `get_magnon_bubble(H, ω)`.
  The spin-↑/↓ projections are performed once before the ω loop.
  Requires `H.spin_s !== nothing`.

**Key idea**: instead of inverting (I − Π₀V), build the Neumann series
  T₀ = Π₀,  Tₙ = Tₙ₋₁·V·Π₀  (so Σ Tₙ → χ_RPA as K→∞),
extract scalars `sₙ(q,ω) = −Im⟨q|Tₙ(ω)|q⟩/π` via `get_spect_k`, and apply
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
    n_wynn = K_max ÷ 2

    chi_partial = nothing
    chi_wynn    = nothing

    for (i, ω) in enumerate(ωlist)
        verbose && println("Wynn RPA (mode=$mode): ω $i/$nω  (ω = $ω)")

        Π0 = if mode == :charge
            get_bubble_mpo(H, H, ω; bubble_kw...)
        else
            get_bubble_mpo(H_up, H_dn, ω; bubble_kw...)
        end

        term = deepcopy(Π0)
        s0   = -imag.(get_spect_k(term))
        nq   = length(s0)

        if chi_partial === nothing
            chi_partial = zeros(Float64, K_max+1, nω, nq)
            chi_wynn    = zeros(Float64, n_wynn,  nω, nq)
        end

        # Individual term contributions: spect_terms[n+1, i_ω, q]
        spect_terms          = zeros(Float64, K_max+1, nq)
        spect_terms[1, :]    = s0

        for n in 1:K_max
            term             = apply(term, MPOV; maxdim=maxdim_apply, cutoff=cutoff_apply)
            term             = apply(term, Π0;   maxdim=maxdim_apply, cutoff=cutoff_apply)
            spect_terms[n+1, :] = -imag.(get_spect_k(term))
        end

        # Partial sums (Wynn input)
        partial_sums = cumsum(spect_terms; dims=1)
        chi_partial[:, i, :] = partial_sums

        # Apply Wynn ε per q-point
        for q in 1:nq
            ests = wynn_epsilon(complex.(partial_sums[:, q]))
            for m in 1:min(n_wynn, length(ests))
                chi_wynn[m, i, q] = real(ests[m])
            end
        end

        verbose && println("  done ($(K_max+1) terms, $(n_wynn) Wynn estimates)")
    end

    return chi_partial, chi_wynn
end

# ============================================================
# Magnon susceptibility (transverse S⁺S⁻ spin channel)
# ============================================================

"""
    _project_spin_sector(H, sector) -> TBHamiltonian

Project a spinful `TBHamiltonian` onto spin sector `sector` (1 = ↑, 2 = ↓)
by contracting the spin site tensor with the projector |sector⟩⟨sector|.

The spin index is identified by its "Spin" tag, so the function is robust
to whether spin is prepended or postpended.  The contracted tensor is
absorbed into its neighbour, leaving a valid L-qubit MPO.

Returns a new `TBHamiltonian` with `spin_s = nothing` and fresh (empty)
caches; `scale` and `center` are reset to 0.0 so `_ensure_scale!` will
re-estimate them on the first KPM call. All other fields (`Lx`,
`interaction_mpo`, `fock_mpo`, `position_space`, …) are copied from `H`.
"""
function _project_spin_sector(H::TBHamiltonian, sector::Int)
    H.spin_s === nothing &&
        error("_project_spin_sector: H is not spinful (spin_s is nothing)")
    s = H.spin_s

    spin_pos = findfirst(n -> any(i -> hastags(i, "Spin"), siteinds(H.mpo, n)),
                         1:length(H.mpo))
    spin_pos === nothing && error("_project_spin_sector: spin Index not found in MPO")

    proj = ITensor(ComplexF64, s', s)
    proj[s' => sector, s => sector] = 1.0

    tensors    = ITensor[H.mpo[i] for i in 1:length(H.mpo)]
    contracted = tensors[spin_pos] * proj   # only link indices remain

    if spin_pos == 1
        tensors[2]  = contracted * tensors[2]
        new_tensors = tensors[2:end]
    else
        tensors[spin_pos - 1] = tensors[spin_pos - 1] * contracted
        new_tensors = tensors[1:spin_pos - 1]
    end

    new_sites = filter(i -> !hastags(i, "Spin"), H.sites)

    # interaction_mpo / fock_mpo live on the position sites (add_interaction!), which
    # the projection leaves untouched, so they are kept along with Lx and position_space.
    return TBHamiltonian(H; sites=new_sites, mpo=MPO(new_tensors),
                         scale=0.0, center=0.0, spin_s=nothing)
end


"""
    get_magnon_bubble(H, ω; ...) -> MPO

Non-interacting transverse spin polarization bubble Π₀^{+−}(ω) for a
spinful `TBHamiltonian`.

The spin degree of freedom is projected out analytically: the spin-↑ and
spin-↓ blocks of `H` become two independent L-qubit Hamiltonians `H_↑`
and `H_↓`, which are passed to `get_bubble_mpo(H_↑, H_↓, ω)`.
This corresponds to the Kubo S⁺S⁻ bubble

    Π₀^{+−}(ω) = ∑_k (f_{k↓} − f_{k↑}) / (ω − (ε_{k↑} − ε_{k↓}) + iη)

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
    get_magnon_susceptibility(H, MPOV, ω; ...) -> MPS

RPA transverse spin susceptibility χ^{+−}_RPA(ω) for a spinful
`TBHamiltonian`.

Builds Π₀^{+−}(ω) via `get_magnon_bubble`, then solves the Dyson
equation (I − Π₀ V) χ = Π₀.  For a Hubbard-like interaction the
interaction MPO is `MPOV = U · Id` on the orbital sites.

**Keyword arguments**
- `rpa_nsweeps`, `rpa_maxdim`, `rpa_cutoff` : Dyson linsolve parameters.
- All `get_bubble_mpo` keywords forwarded via `kwargs...`.
"""
function get_magnon_susceptibility(H::TBHamiltonian, MPOV::MPO, ω::Real;
                                   rpa_nsweeps::Int = 20,
                                   rpa_maxdim::Int  = 400,
                                   rpa_cutoff::Real = 1e-8,
                                   kwargs...)
    H.spin_s === nothing &&
        error("get_magnon_susceptibility: H is not spinful — call add_spin!(H) first")
    H_up = _project_spin_sector(H, 1)
    H_dn = _project_spin_sector(H, 2)
    Π = get_bubble_mpo(H_up, H_dn, ω; kwargs...)
    # H.L counts position qubits only; H_up.sites also keeps any sublattice/layer index.
    finalsites = _rpa_pair_sites(H_up.sites)
    return rpa_from_bubble_diag(Π, MPOV, finalsites, H_up.sites;
                                nsweeps=rpa_nsweeps, maxdim=rpa_maxdim, cutoff=rpa_cutoff)
end


"""
    get_magnon_susceptibility_wynn(H, MPOV, ωlist; K_max, ...) -> (chi_partial, chi_wynn)

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

    nω     = length(ωlist)
    n_wynn = K_max ÷ 2

    chi_partial = nothing
    chi_wynn    = nothing

    for (i, ω) in enumerate(ωlist)
        verbose && println("Magnon Wynn RPA: ω $i/$nω  (ω = $ω)")

        Π0   = get_bubble_mpo(H_up, H_dn, ω; verbose, kwargs...)
        term = deepcopy(Π0)
        s0   = -imag.(get_spect_k(term))
        nq   = length(s0)

        if chi_partial === nothing
            chi_partial = zeros(Float64, K_max + 1, nω, nq)
            chi_wynn    = zeros(Float64, n_wynn,    nω, nq)
        end

        spect_terms       = zeros(Float64, K_max + 1, nq)
        spect_terms[1, :] = s0

        for n in 1:K_max
            term             = apply(term, MPOV; maxdim=maxdim_apply, cutoff=cutoff_apply)
            term             = apply(term, Π0;   maxdim=maxdim_apply, cutoff=cutoff_apply)
            spect_terms[n+1, :] = -imag.(get_spect_k(term))
        end

        partial_sums         = cumsum(spect_terms; dims=1)
        chi_partial[:, i, :] = partial_sums

        for q in 1:nq
            ests = wynn_epsilon(complex.(partial_sums[:, q]))
            for m in 1:min(n_wynn, length(ests))
                chi_wynn[m, i, q] = real(ests[m])
            end
        end

        verbose && println("  done ($(K_max+1) terms, $(n_wynn) Wynn estimates)")
    end

    return chi_partial, chi_wynn
end
