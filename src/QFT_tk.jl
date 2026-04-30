# QFT_tk.jl — Momentum-space band structure via online Chebyshev KPM
#
# Overview
# --------
# The quantics representation encodes a 1D or 2D real-space index as a
# binary string across L qubit sites.  Conjugating any real-space MPO W
# by the Quantum Fourier Transform (QFT) gives its momentum-space image:
#
#   Ã(k,ω) = U · δ(ω − H) · U†
#
# The diagonal ⟨k|Ã(ω)|k⟩ is the k-resolved spectral function A(k,ω).
#
# Design: online Chebyshev accumulation
# --------------------------------------
# Instead of storing all Ncheb Chebyshev MPOs, get_bands runs a single
# Chebyshev recurrence T_0, T_1, … T_Ncheb.  At each step n the current
# Chebyshev MPO passes through up to four composable projection stages
# before the QFT is applied (see get_bands for the full pipeline):
#
#   Step 0  nambu_proj  — project Nambu (BdG particle/hole) index
#   Step 1  spin_proj   — project spin index
#   Step 1b sublat_proj — project sublattice auxiliary index (kagome, Lieb…)
#   Step 2  sublattice  — legacy mask sandwich for 2-sublattice models
#   Step 3  QFT + diagonal extraction + KPM accumulation
#
# Each step is independent and optional; any combination is supported.
# Peak memory: O(3 MPOs) regardless of Ncheb.
#
# Auxiliary index projection (section 4b)
# ----------------------------------------
# Hamiltonians with auxiliary DOFs (spin, Nambu, sublattice…) carry an
# extra site at the front or back of the MPO.  project_aux removes it by
# contracting |σ⟩⟨σ| onto the auxiliary tensor, returning an (L−1)-site
# position-only MPO ready for conjugate_by_qft.  aux_site(H, which)
# extracts the correct Index and side (:pre/:post) from a TBHamiltonian.
#
# Encoding conventions
# --------------------
# 1D  — sites 1…L hold x bits, LSB at site 1 (quantics QFT convention).
# 2D  — sites 1…Ly hold iy bits (MSB first), sites Ly+1…L hold ix bits
#        (MSB first); linear index n = ix + iy * 2^Lx (row-major).
#        Sublattice masks from 2D_lattice.jl respect this convention.
#        ## Note: bit ordering in 2D needs further verification.
#
# Dependencies outside this file
# --------------------------------
# fix_sites, _kpm_kernel               → utils.jl
# extract_diagonal_to_mps              → utils.jl
# _row_checker_mpo, _col_select_mpo    → 2D_lattice.jl
# TBHamiltonian, _ensure_scale!        → TBSystem.jl


# ============================================================
# 1. QFT conjugation
# ============================================================

"""
    conjugate_by_qft(W; tol=1e-9, maxdim=100) -> MPO

Return `U · W · U†` where `U` is the Quantum Fourier Transform MPO
built from `QuanticsTCI.quanticsfouriermpo`.

The `TCI.reverse` call places the LSB at site 1 to match the quantics
encoding used throughout this codebase. Note that the ordering in k-space is the reverse of the site ordering in real space
"""
function conjugate_by_qft(W; tol=1e-9, maxdim::Int=100)
    sites  = getindex.(siteinds(W), 2)
    R      = length(sites)
    FTirev = fix_sites(MPO(TCI.reverse(QuanticsTCI.quanticsfouriermpo(R; sign=-1.0, normalize=true))), sites)
    FTrev  = fix_sites(MPO(TCI.reverse(QuanticsTCI.quanticsfouriermpo(R; sign=+1.0, normalize=true))), sites)
    Op1    = apply(W,                        FTirev; cutoff=tol, maxdim=maxdim)
    Op2    = apply(swapprime(FTrev, 0 => 1), Op1;   cutoff=tol, maxdim=maxdim)
    return TCI.truncate(Op2; cutoff=tol, maxdim=maxdim)
end


# ============================================================
# 2. Sublattice projection
#
# For now only works in 1D and 2D lattices using a checkerboard or alternating-site sublattice decomposition
#
# These wrappers project an MPO O onto one of two sublattices by
# sandwiching it with the corresponding diagonal mask MPO:
#   O_SL = mask · O · mask
#
# The mask MPOs (`_row_checker_mpo`, `_col_select_mpo`) are defined in
# 2D_lattice.jl and operate on a single qubit site each, so they are
# bond-dimension 1 and cheap to apply.
#
# 2D — checkerboard sublattices of a 2^Lx × 2^Ly lattice:
#   SL=1 → sites where (ix+iy) is even  (_row_checker_mpo)
#   SL=2 → sites where (ix+iy) is odd   (Id − _row_checker_mpo)
#
# 1D — alternating sublattices of a 2^L chain:
#   SL=1 → even sites (ix % 2 == 0)    (_col_select_mpo, keep=:odd)
#   SL=2 → odd  sites (ix % 2 == 1)    (_col_select_mpo, keep=:even)
#   Note: the `:odd`/`:even` labels in _col_select_mpo refer to the
#   parity of the LSB qubit state (0 or 1), not of the site index.
# ============================================================

"""
    projop_2DSL(O, sites, Lx, Ly, SL) -> MPO

Project MPO `O` (on a `2^Lx × 2^Ly` lattice) onto sublattice `SL`
(1 = even checkerboard, 2 = odd checkerboard) by sandwiching with the
corresponding diagonal mask: `mask · O · mask`.
"""
function projop_2DSL(O::MPO, sites, Lx, Ly, SL::Integer)
    mask = SL == 1 ? _row_checker_mpo(Lx, Ly, sites) :
                     MPO(sites, "Id") - _row_checker_mpo(Lx, Ly, sites)
    Oproj = apply(mask, O; cutoff=1e-8, maxdim=100)
    Oproj = apply(Oproj, mask; cutoff=1e-8, maxdim=100)
    return Oproj
end

"""
    projop_1DSL(O, sites, Lx, SL) -> MPO

Project MPO `O` (on a `2^Lx` chain) onto sublattice `SL`
(1 = even sites, 2 = odd sites) by sandwiching with the corresponding
diagonal mask: `mask · O · mask`.
"""
function projop_1DSL(O::MPO, sites, Lx, SL::Integer)
    mask = SL == 1 ? _col_select_mpo(Lx, 0, sites; keep=:odd) :
                     _col_select_mpo(Lx, 0, sites; keep=:even)
    Oproj = apply(mask, O; cutoff=1e-8, maxdim=100)
    Oproj = apply(Oproj, mask; cutoff=1e-8, maxdim=100)
    return Oproj
end


# ============================================================
# 3. Internal utilities
# ============================================================

"""
    ilinspace(xmin, xmax, num_x) -> Vector{Int}

Return `num_x` as almost evenly spaced integers in `[xmin, xmax]`, inclusive,
with a preference for the endpoints.  Used to build the k-point center
grid for band-structure sampling.
"""
function ilinspace(xmin, xmax, num_x::Int)
    xvals = xmin:xmax
    _N = length(xvals)
    @assert 1 ≤ num_x ≤ _N
    num_x == 1 && return [0]
    step = (_N - 1) ÷ (num_x - 1)
    return collect(xmin:step:(xmin+step*(num_x-1)))
end


"""
    _eval_diag_mps(A, x) -> Float64

Evaluate the diagonal MPS `A` at the 0-indexed position `x` using a
LSB-first bit encoding (site 1 = bit 0 of x).  Equivalent to
`inner(binary_MPS(x), A)` but avoids constructing the full basis MPS.
"""
function _eval_diag_mps(A::MPS, x::Int)
    L     = length(A)
    sites = siteinds(A)
    acc   = ITensor(1.0)
    for i in 1:L
        b    = (x >> (i - 1)) & 1     # bit i-1 of x, LSB first
        acc *= A[i] * setelt(sites[i] => b + 1)
    end
    return real(scalar(acc))
end


"""
    sample_diag(Tn_k, ikstart, ikend) -> Vector{Float32}

Extract the diagonal of MPO `Tn_k` as an MPS and evaluate it at every
integer index in `ikstart:ikend`.  Convenience wrapper around
`_eval_diag_mps`; used when all k-points in a contiguous range are needed.
"""
function sample_diag(Tn_k::MPO, ikstart::Int, ikend::Int)
    A_mps = extract_diagonal_to_mps(Tn_k)
    A_mps = ITensorMPS.truncate!(A_mps; cutoff=1e-10)
    vals  = zeros(Float32, ikend - ikstart + 1)
    for (iloc, idx) in enumerate(ikstart:ikend)
        vals[iloc] = _eval_diag_mps(A_mps, idx)
    end
    return vals
end


"""
    _kpm_weight_matrix(Ncheb, ω_vals; kernel, lambda) -> Matrix{Float64}

Precompute the KPM weight matrix `W[n, iω]` for fast accumulation.

`W[n, iω] = c_n · g_n · cos((n-1) · arccos(ω_iω))`

where `c_n = 1` for n=1 and 2 otherwise, and `g_n` is the kernel
damping factor (Jackson or Lorentz).  Entries for `|ω| ≥ 1` are left
at zero.

Note: later could include HODC kernel
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
# 4. Online band structure  —  get_bands
#
# Chebyshev recurrence (runs on the FULL MPO space, all aux sites included):
#   T_0 = I,  T_1 = H̃,  T_n = 2 H̃ T_{n-1} − T_{n-2}    (H̃ = (H−center)/scale)
#
# At each step n, T_n passes through four composable projection stages that
# build a list of position-only MPOs; each is QFT'd and accumulated.
#
#   Step 0  nambu_proj  (optional)
#       For each selected Nambu sector σ ∈ {1=particle, 2=hole}:
#           project_aux(T_n, nambu_s, σ; side=nambu_side)  → L_pos+Lspin-site MPO
#       Nambu is the outermost aux (prepended last), so it is projected first.
#
#   Step 1  spin_proj  (optional)
#       For each selected spin channel σ ∈ {1=↑, 2=↓}:
#           project_aux(T, spin_s, σ; side=:pre)  → L_pos-site MPO
#       After Nambu removal, spin is at site 1 of the reduced MPO.
#       spin_s_aux carries the explicit spin Index for disambiguation.
#
#   Step 1b  sublat_proj  (optional — for kagome/Lieb/honeycomb aux index)
#       For each selected sublattice σ ∈ {1…dim(sublat_s)}:
#           project_aux(T, sublat_s, σ; side=sublat_side)  → L_pos-site MPO
#
#   Step 2  sublattice  (optional — legacy mask sandwich for 2-sublattice models)
#       For each selected mask ∈ {mask_A, mask_B}:
#           T_proj = mask · T · mask
#
#   Step 3  QFT + diagonal extraction + KPM accumulation  (always)
#       T_k  = conjugate_by_qft(T_proj)
#       A_mps = extract_diagonal_to_mps(T_k)
#       for each k-group: s = mean(_eval_diag_mps(A_mps, x))
#       ak_accum[iω, ik] += W[n, iω] * s
#
# k-point sampling (mirrors projdos_from_Tn_mpsk):
#   num_x centers placed with ilinspace in [xmin, xmax]; each center
#   is averaged over num_avg nearby offset points (±half_step).
#   2D: diagonal zip of x/y-offsets, combined as (iy << Lx) | ix.
#
# Projection combinations:
#   nambu_proj  ×  spin_proj  ×  sublat_proj  ×  sublattice
#   yields up to 2 × 2 × dim(sublat_s) × 2 MPOs per Chebyshev step.
#   All contributions are summed unless specific sectors are selected
#   via proj_nambu / proj_s / proj_sl.
# ============================================================

"""
    get_bands(H_mpo, scale, center, sites, Ncheb, D, ω_vals; kwargs...) -> Matrix{Float64}

Memory-efficient band structure via online Chebyshev KPM accumulation.
See the section 4 block comment above for the full four-step projection pipeline.

# Arguments
- `H_mpo`        : unscaled Hamiltonian MPO on all sites (position + any aux).
- `scale, center`: energy rescaling so that H̃ = (H−center)/scale ∈ (−1, 1).
- `sites`        : the full site list of `H_mpo` including any aux indices.
                   Position-only site count is inferred as `L_pos = L − n_aux`.
- `Ncheb`        : number of Chebyshev moments.
- `D`            : spatial dimension (1 or 2).
- `ω_vals`       : rescaled energies ∈ (−1, 1) at which to evaluate A(k,ω).

# Projection keyword arguments
Each projection flag is independent; any combination is valid.

**Nambu (BdG particle/hole) projection — Step 0:**
- `nambu_proj`   : project each T_n onto Nambu sectors (default `false`).
- `proj_nambu`   : `1` = particle only, `2` = hole only, `nothing` = sum both.
- `nambu_s`      : the Nambu `Index` (auto-detected from `H.nambu_s` via the
                   `TBHamiltonian` overload).
- `nambu_side`   : `:pre` (default) or `:post` — position of the Nambu site.

**Spin projection — Step 1:**
- `spin_proj`    : project each T_n onto spin channels (default `false`).
- `proj_s`       : `1` = ↑ only, `2` = ↓ only, `nothing` = sum both.
- `spin_s_aux`   : explicit spin `Index`; when `nothing`, falls back to
                   `sites[1]`.  Set automatically by the `TBHamiltonian` overload
                   so that spin is correctly identified even when Nambu is also
                   prepended at site 1.

**Sublattice auxiliary projection — Step 1b (kagome, Lieb, honeycomb):**
- `sublat_proj`  : project each T_n onto sublattice aux sectors (default `false`).
- `proj_sl`      : `k` = sublattice k only, `nothing` = sum all.  Shared with
                   the legacy `sublattice` flag (Step 2).
- `sublat_s`     : the sublattice `Index` (auto-detected from `H.sublattice_s`).
- `sublat_side`  : `:post` (default) or `:pre` — position of the sublattice site.

**Legacy sublattice mask projection — Step 2 (2-sublattice models without aux index):**
- `sublattice`   : apply a mask sandwich `mask · T_n · mask` (default `false`).
- `proj_sl`      : `1` = mask A only, `2` = mask B only, `nothing` = both.

# k-point sampling keyword arguments
- `xmin, xmax, num_x` : grid in x (1D) or kx (2D).  Default: full range, 10 pts.
- `ymin, ymax, num_y` : grid in ky (2D only).
- `num_avg`      : number of offset points averaged around each center (default 1).

# Truncation and performance
- `kernel`       : KPM broadening kernel (`:jackson` or `:lorentz`).
- `lambda`       : Lorentz kernel width (ignored for Jackson).
- `tol, maxdim, cutoff` : MPO truncation parameters passed to `apply` and `truncate!`.
- `printinfo`    : print `maxlinkdim` every 10 Chebyshev steps (default `false`).

# Returns
`Matrix{Float64}` of shape `(Nω, num_x)`.
"""
function get_bands(H_mpo::MPO, scale::Real, center::Real, sites,
                          Ncheb::Int, D::Int, ω_vals;
                          spin_proj::Bool   = false,
                          proj_s            = nothing,
                          spin_s_aux        = nothing,
                          nambu_proj::Bool  = false,
                          proj_nambu        = nothing,
                          nambu_s           = nothing,
                          nambu_side::Symbol  = :pre,
                          sublattice::Bool  = false,
                          proj_sl           = nothing,
                          sublat_proj::Bool = false,
                          sublat_s          = nothing,
                          sublat_side::Symbol = :post,
                          xmin::Int       = 0,
                          xmax            = nothing,
                          num_x::Int      = 10,
                          num_avg::Int    = 1,
                          ymin::Int       = 0,
                          ymax            = nothing,
                          num_y::Int      = 10,
                          kernel::Symbol  = :jackson,
                          lambda::Real    = 4.0,
                          tol::Real       = 1e-9,
                          maxdim::Int     = 100,
                          cutoff::Real    = 1e-10,
                          printinfo::Bool = false)

    L = length(sites)
    # Nambu and sublattice are always internal aux DOFs, never position qubits.
    # Spin is subtracted only when spin_proj=true (it may be part of the physical encoding).
    L_pos = L - (spin_proj ? 1 : 0) - (!isnothing(nambu_s) ? 1 : 0) - (!isnothing(sublat_s) ? 1 : 0)
    N     = 2^L_pos

    # ── Scaled Hamiltonian ────────────────────────────────────────────────────
    # sites already includes all aux sites; MPO(sites, "Id") is correctly sized.
    I_mpo = MPO(sites, "Id")
    Ham_n = (1 / scale) * +(H_mpo, (-center) * I_mpo; cutoff = cutoff)

    # ── KPM weight matrix  W[n, iω] ──────────────────────────────────────────
    Nω    = length(ω_vals)
    valid = [abs(ω) < 1.0 for ω in ω_vals]
    W     = _kpm_weight_matrix(Ncheb, ω_vals; kernel = kernel, lambda = lambda)

    # ── Build k-point groups (mirrors projdos_from_Tn_mpsk sampling) ─────────
    if D == 1
        _xmax     = xmax === nothing ? N - 1 : Int(xmax)
        xcenters  = ilinspace(xmin, _xmax, num_x)
        half_step = num_x > 1 ? (_xmax - xmin) / (2 * num_x) : 0
        offsets   = num_avg > 1 ? round.(Int, range(-half_step, half_step; length=num_avg)) : Int[0]
        k_groups  = [clamp.(xcenters[i] .+ offsets, 0, N - 1) for i in 1:num_x]
    elseif D == 2
        Lx     = div(L_pos, 2)
        Nx_loc = 2^Lx
        Ny_loc = 2^(L_pos - Lx)
        _xmax  = xmax === nothing ? Nx_loc - 1 : Int(xmax)
        _ymax  = ymax === nothing ? Ny_loc - 1 : Int(ymax)
        xcenters    = ilinspace(xmin, _xmax, Nx_loc)
        ycenters    = ilinspace(ymin, _ymax, Ny_loc)
        half_step_x = num_x > 1 ? (_xmax - xmin) / (2 * num_x) : 0
        half_step_y = num_y > 1 ? (_ymax - ymin) / (2 * num_y) : 0
        x_offs = num_avg > 1 ? round.(Int, range(-half_step_x, half_step_x; length=num_avg)) : Int[0]
        y_offs = num_avg > 1 ? round.(Int, range(-half_step_y, half_step_y; length=num_avg)) : Int[0]
        k_groups = [
            begin
                xs = clamp.(xcenters[i] .+ x_offs, 0, Nx_loc - 1)
                ys = clamp.(ycenters[i] .+ y_offs, 0, Ny_loc - 1)
                [(y << Lx) | x for (x, y) in zip(xs, ys)]  # diagonal zip in 2D k-space
            end
            for i in 1:num_x
        ]
    else
        error("D must be 1 or 2")
    end

    Ak_w = zeros(Float64, Nω, num_x)

    # ── Precompute sublattice masks once (reused every Chebyshev step) ────────
    # Masks are applied to the position-only MPO (after all aux projections),
    # so they must be built from the position sites only.
    # pos_sites = position qubits only, for legacy sublattice mask building.
    # All known aux indices are excluded regardless of whether their projection
    # is active — a "Kagome" or "Spin" tagged index must never reach OpSum.
    aux_to_drop = Set{Index}()
    spin_proj             && push!(aux_to_drop, sites[1])
    !isnothing(nambu_s)   && push!(aux_to_drop, nambu_s::Index)
    !isnothing(sublat_s)  && push!(aux_to_drop, sublat_s::Index)
    pos_sites = filter(s -> s ∉ aux_to_drop, sites)
    if sublattice
        if D == 1
            mask_A = _col_select_mpo(L_pos, 0, pos_sites; keep=:odd)   # even sites (ix % 2 == 0)
            mask_B = _col_select_mpo(L_pos, 0, pos_sites; keep=:even)  # odd  sites (ix % 2 == 1)
        else
            Ly = L_pos - Lx
            mask_A = _row_checker_mpo(Lx, Ly, pos_sites)                           # (ix+iy) even
            mask_B = MPO(pos_sites, "Id") - _row_checker_mpo(Lx, Ly, pos_sites)    # (ix+iy) odd
        end
    end

    # ── Online accumulation: project → QFT → sample → accumulate ─────────────
    # Four independent, composable projection steps build a list of position
    # MPOs; every MPO in the list is QFT'd, sampled, and its contribution summed.
    #
    #  Step 0  nambu_proj         → project aux Nambu (BdG) index  (×1 or ×2)
    #  Step 1  spin_proj          → project aux spin index          (×1 or ×2)
    #  Step 1b sublat_proj        → project aux sublattice index    (×1 … ×dim)
    #  Step 2  sublattice (legacy)→ apply mask sandwich             (×1 or ×2)
    #
    # Nambu is projected before spin because it is the outermost aux site.
    # After nambu removal the spin index moves to position 1, so project_aux(:pre)
    # on the reduced MPO lands on spin automatically.  spin_s_aux carries the
    # explicit spin Index so the correct site is targeted even when nambu is present.
    local _nambu_side = nambu_side
    local _sublat_side = sublat_side
    local _spin_idx    = isnothing(spin_s_aux) ? sites[1] : spin_s_aux
    function accumulate_Tn!(ak_accum, Tn, n)
        # Step 0: Nambu (BdG particle/hole) projection — outermost aux, project first.
        # proj_nambu=nothing → sum particle+hole; proj_nambu=1/2 → select one sector.
        after_nambu = nambu_proj ? [project_aux(Tn, nambu_s::Index, σ; side=_nambu_side)
                                    for σ in (isnothing(proj_nambu) ? (1:2) : (proj_nambu:proj_nambu))] : MPO[Tn]

        # Step 1: spin aux projection.
        # Uses spin_s_aux (explicit Index) when provided, falls back to sites[1].
        # proj_s=nothing → sum both channels; proj_s=1/2 → select one.
        after_spin = spin_proj ? [project_aux(T, _spin_idx, σ; side=:pre)
                                  for T in after_nambu, σ in (isnothing(proj_s) ? (1:2) : (proj_s:proj_s))] : after_nambu

        # Step 1b: sublattice aux projection (kagome/Lieb/honeycomb with H.sublattice_s).
        # proj_sl=nothing → sum all sublattices; proj_sl=k → select sublattice k.
        after_sl_aux = if sublat_proj
            σ_sl = isnothing(proj_sl) ? (1:dim(sublat_s::Index)) : (proj_sl:proj_sl)
            [project_aux(T, sublat_s::Index, σ; side=_sublat_side)
             for T in after_spin for σ in σ_sl]
        else
            after_spin
        end

        # Step 2: legacy sublattice mask projection (for 2-sublattice models without aux index)
        # proj_sl=nothing applies both masks; proj_sl=1/2 selects one.
        if sublattice
            masks = isnothing(proj_sl) ? [mask_A, mask_B] :
                    proj_sl == 1       ? [mask_A]          : [mask_B]
            sl_mpas = MPO[]
            for T in after_sl_aux, mask in masks
                push!(sl_mpas, apply(apply(mask, T; cutoff=cutoff, maxdim=maxdim), mask; cutoff=cutoff, maxdim=maxdim))
            end
        else
            sl_mpas = after_sl_aux
        end

        # Step 3: QFT + diagonal sample + accumulate for every MPO in the list
        for T in sl_mpas
            Tn_k  = conjugate_by_qft(T; tol=tol, maxdim=maxdim)
            A_mps = ITensorMPS.truncate!(extract_diagonal_to_mps(Tn_k); cutoff=cutoff)
            for (ik, xs) in enumerate(k_groups)
                s = sum(_eval_diag_mps(A_mps, x) for x in xs) / length(xs)
                for ie in 1:Nω
                    ak_accum[ie, ik] += W[n, ie] * s
                end
            end
        end
    end

    # ── Chebyshev recurrence  T_0 = I,  T_1 = H̃,  T_n = 2H̃T_{n-1} − T_{n-2}
    # The recurrence runs on the full MPO space (L+1 sites when spin_proj=true).
    # Projection happens inside accumulate_Tn! so T_n itself is never modified.
    Tkm2 = I_mpo   # T_0
    Tkm1 = Ham_n   # T_1

    accumulate_Tn!(Ak_w, Tkm2, 1)
    accumulate_Tn!(Ak_w, Tkm1, 2)

    for k in 3:Ncheb
        Tk = +(2 * apply(Ham_n, Tkm1; cutoff=cutoff), -Tkm2; maxdim=maxdim)
        Tk = ITensorMPS.truncate!(Tk; cutoff=cutoff)
        accumulate_Tn!(Ak_w, Tk, k)
        Tkm2 = Tkm1
        Tkm1 = Tk
        printinfo && (k % 10 == 0 || k == Ncheb) &&
            println("Online KPM step $k/$Ncheb  maxlinkdim=$(maxlinkdim(Tkm1))")
    end

    # ── Normalization: divide by the KPM DOS weight ───────────────────────────
    for iω in 1:Nω
        valid[iω] || continue
        Ak_w[iω, :] ./= (π^2 * Ncheb * sqrt(1 - ω_vals[iω]^2))
    end

    return Ak_w
end


"""
    get_bands(H, Ncheb, D, ω_phys_vals; kwargs...) -> Matrix{Float64}

High-level overload of `get_bands` for a `TBHamiltonian`.

Physical energies `ω_phys_vals` are rescaled automatically via `H.scale` and
`H.center`.  All projection keyword arguments (`spin_proj`, `nambu_proj`,
`sublat_proj`, `sublattice`, `proj_s`, `proj_nambu`, `proj_sl`) are
accepted exactly as in the low-level MPO method.

The auxiliary indices (`nambu_s`/`nambu_side`, `spin_s_aux`,
`sublat_s`/`sublat_side`) are **auto-detected** from the `TBHamiltonian`
fields (`H.nambu_s`, `H.spin_s`, `H.sublattice_s`) and never need to be
passed manually.  See the low-level method docstring for full documentation.
"""
function get_bands(H::TBHamiltonian, Ncheb::Int, D::Int, ω_phys_vals;
                          spin_proj::Bool   = false,
                          proj_s            = nothing,
                          nambu_proj::Bool  = false,
                          proj_nambu        = nothing,
                          sublattice::Bool  = false,
                          proj_sl           = nothing,
                          sublat_proj::Bool = false,
                          xmin::Int       = 0,
                          xmax            = nothing,
                          num_x::Int      = 10,
                          num_avg::Int    = 1,
                          ymin::Int       = 0,
                          ymax            = nothing,
                          num_y::Int      = 10,
                          kernel::Symbol  = :jackson,
                          lambda::Real    = 4.0,
                          tol::Real       = 1e-9,
                          maxdim::Int     = 100,
                          cutoff::Real    = 1e-10,
                          printinfo::Bool = false)

    _ensure_scale!(H)
    ω_resc = (collect(ω_phys_vals) .- H.center) ./ H.scale

    # Auto-detect all aux indices so the low-level function can exclude them
    # from L_pos and pos_sites regardless of which projections are active.
    nambu_s_det, nambu_side_det = !isnothing(H.nambu_s) ?
        aux_site(H, :nambu) : (nothing, :pre)

    spin_s_det = H.spin_s   # may be nothing; low-level falls back to sites[1] when nothing

    sublat_s_det, sublat_side_det = !isnothing(H.sublattice_s) ?
        aux_site(H, :sublattice) : (nothing, :post)

    return get_bands(H.mpo, H.scale, H.center, H.sites, Ncheb, D, ω_resc;
                            spin_proj  = spin_proj,  proj_s     = proj_s,
                            spin_s_aux = spin_s_det,
                            nambu_proj = nambu_proj, proj_nambu = proj_nambu,
                            nambu_s    = nambu_s_det, nambu_side = nambu_side_det,
                            sublattice = sublattice, proj_sl    = proj_sl,
                            sublat_proj = sublat_proj,
                            sublat_s    = sublat_s_det,
                            sublat_side = sublat_side_det,
                            xmin = xmin, xmax = xmax,
                            num_x = num_x, num_avg = num_avg,
                            ymin = ymin, ymax = ymax, num_y = num_y,
                            kernel = kernel, lambda = lambda,
                            tol = tol, maxdim = maxdim, cutoff = cutoff,
                            printinfo = printinfo)
end


# ============================================================
# 4b. Auxiliary index projection utilities
#
# Any auxiliary DOF (spin, sublattice, Nambu …) added with prepend_op /
# postpend_op lives at the first or last site of the MPO as a dim-1-bonded
# tensor.
#
# project_aux(W, aux_s, σ; side)
#   Contracts the projector |σ⟩⟨σ| onto the bra and ket physical indices of
#   the aux tensor at the front (:pre) or back (:post) of the MPO, then
#   absorbs the remaining dim-1 link into the adjacent position site.
#   Returns an (L−1)-site MPO ready for conjugate_by_qft.
#   project_spin is a convenience alias for the common :pre case.
#
# aux_site(H, which) -> (Index, Symbol)
#   Extracts the auxiliary Index and its side (:pre/:post) from a
#   TBHamiltonian by looking up the appropriate field (H.spin_s, H.nambu_s,
#   H.sublattice_s, H.layer_s) and finding its position in H.sites.
#   Used internally by the TBHamiltonian overload of get_bands to pass the
#   correct aux_s / side to the low-level function without user intervention.
# ============================================================

"""
    project_aux(W, aux_s, σ; side=:pre) -> MPO

Remove an auxiliary site from MPO `W` by projecting onto state `σ`.

- `side=:pre`  — aux site is at position 1 (prepended, e.g. spin).
- `side=:post` — aux site is at the last position (postpended, e.g. sublattice).

Contracts the projector |σ⟩⟨σ| on both bra and ket physical indices of the
aux tensor; the resulting dim-1 link is absorbed into the adjacent position
site.  Returns an (L−1)-site MPO suitable for `conjugate_by_qft`.
"""
function project_aux(W::MPO, aux_s::Index, σ::Integer; side::Symbol = :pre)
    L        = length(W)
    pos      = side === :pre ? 1 : L
    aux_proj = W[pos] * setelt(aux_s' => σ) * setelt(aux_s => σ)
    new_tensors = Vector{ITensor}(undef, L - 1)
    if side === :pre
        new_tensors[1] = W[2] * aux_proj
        for i in 2:L-1; new_tensors[i] = W[i+1]; end
    else  # :post
        for i in 1:L-2; new_tensors[i] = W[i]; end
        new_tensors[L-1] = W[L-1] * aux_proj
    end
    return MPO(new_tensors)
end

# Nothing-overloads: give Julia a compilable method when the Index is nothing,
# so branches in get_bands can be type-checked without a MethodError.
project_aux(::MPO, ::Nothing, ::Integer; side::Symbol=:pre) =
    error("sublat_proj=true requires sublat_s to be set (detected from H.sublattice_s)")

# Convenience alias — spin is always prepended (:pre)
"""
    project_spin(W, spin_s, σ) -> MPO

Convenience wrapper for `project_aux` when the auxiliary site is prepended
(spin at site 1).  Equivalent to `project_aux(W, spin_s, σ; side=:pre)`.
"""
project_spin(W::MPO, spin_s::Index,   σ::Integer) = project_aux(W, spin_s, σ; side=:pre)
project_spin(W::MPO, ::Nothing, ::Integer) =
    error("spin_proj=true requires spin_s — detected via sites[1] when spin_proj=true")


"""
    aux_site(H, which) -> (Index, Symbol)

Return the auxiliary `Index` and its position side (`:pre` or `:post`) for
the named auxiliary degree of freedom in `H`.

`which` ∈ `:spin`, `:sublattice`, `:nambu`, `:layer`.

Useful for passing the correct arguments to `project_aux` without manually
inspecting `H.sites`.

```julia
s, side = aux_site(H, :sublattice)
W_A = project_aux(W, s, 1; side=side)   # sublattice-A channel
```
"""
function aux_site(H::TBHamiltonian, which::Symbol)
    s = which === :spin       ? H.spin_s        :
        which === :sublattice ? H.sublattice_s  :
        which === :nambu      ? H.nambu_s       :
        which === :layer      ? H.layer_s       :
        error("Unknown auxiliary type :$which.  Use :spin, :sublattice, :nambu, or :layer.")
    isnothing(s) && error("H has no $which auxiliary index.")
    pos  = findfirst(==(s), H.sites)
    isnothing(pos) && error("Auxiliary index not found in H.sites — this is a bug.")
    side = pos == 1             ? :pre  :
           pos == length(H.sites) ? :post :
           error("Auxiliary $which index found at interior position $pos (unsupported).")
    return s, side
end





# ============================================================
# 5. Legacy — kept for reference, not part of the public API
# ============================================================

# ── get_spect_k / get_spect_k_doubled (inner-product approach, superseded) ──
# These evaluated the band structure by constructing basis MPS |k⟩ and
# computing ⟨k|Ã(ω)|k⟩ directly.  Correct but O(N) inner products per ω.
# Replaced by the diagonal-extraction approach in get_bands.

# function get_spect_k(W; tol=1e-9, maxdim::Int=100)
#     Akop  = conjugate_by_qft(W; tol=tol, maxdim=maxdim)
#     sites = getindex.(siteinds(W), 2)
#     L     = length(sites)
#     N     = 2^L
#     # LSB at site 1 — matches the quantics QFT convention (see KPM_LDOS_1D notebook)
#     lsb_state(k) = [string((k >> (i-1)) & 1) for i in 1:L]
#     mpsk(k) = MPS(sites, lsb_state(Int(k)))
#     kvals   = range(0, N - 1; length=N)
#     return [inner(mpsk(k)', Akop, mpsk(k)) for k in kvals]
# end

# function get_spect_k_doubled(W; tol=1e-9, maxdim::Int=100)
#     Akop  = conjugate_by_qft(W; tol=tol, maxdim=maxdim)
#     sites = getindex.(siteinds(W), 2)
#     L     = div(length(sites), 2)
#     N     = 2^L
#     lsb_state(k) = [string((k >> (i-1)) & 1) for i in 1:L]
#     mpsk(k)  = MPS(sites[1:L],    lsb_state(Int(k)))
#     mpsk1(k) = MPS(sites[L+1:2L], lsb_state(Int(k)))
#     mpsk2(k) = mps_kron(mpsk(k), mpsk1(k))
#     kvals    = range(0, N - 1; length=N)
#     return [inner(mpsk2(k)', Akop, mpsk2(k)) for k in kvals]
# end


#= ── OLD get_bands (offline inner-product approach) — kept for reference ─────
#
# Key difference from the current get_bands:
#   OLD: loop over ω first → for each ω build δ(ω−H) via get_ldos_w_from_Tn,
#        conjugate with QFT, then loop over all k to compute inner products.
#        Memory: O(1 MPO) per ω, but O(N) inner products × Nω evaluations.
#   NEW: loop over Chebyshev steps → project, QFT, sample diagonal, accumulate.
#        Memory: O(3 MPOs), no inner products, scales to large N.
#
# The TBHamiltonian overload below also handled auxiliary (spin/orbital) sites
# via prepend_op / postpend_op — that generality is not yet ported to get_bands.

function get_bands(Tn_list, Ncheb::Int, sites, ω_vals;
                   tol=1e-9, maxdim::Int=100)
    L   = length(sites)
    N   = 2^L
    Nω  = length(ω_vals)
    FTirev = fix_sites(MPO(TCI.reverse(
        QuanticsTCI.quanticsfouriermpo(L; sign=-1.0, normalize=true))), sites)
    FTrev  = fix_sites(MPO(TCI.reverse(
        QuanticsTCI.quanticsfouriermpo(L; sign=+1.0, normalize=true))), sites)
    lsb_state(k) = [string((k >> (i-1)) & 1) for i in 1:L]
    mpsk = [MPS(sites, lsb_state(k)) for k in 0:N-1]
    Ak_w = zeros(Float64, N, Nω)
    for (iω, ω) in enumerate(ω_vals)
        abs(ω) >= 1.0 && continue
        δH = get_ldos_w_from_Tn(Tn_list, Ncheb, ω; maxdim=maxdim)
        Op1  = apply(δH,                        FTirev; cutoff=tol, maxdim=maxdim)
        Akop = apply(swapprime(FTrev, 0 => 1),  Op1;   cutoff=tol, maxdim=maxdim)
        for k in 0:N-1
            Ak_w[k+1, iω] = real(inner(mpsk[k+1]', Akop, mpsk[k+1]))
        end
    end
    return Ak_w
end

function get_bands(H::TBHamiltonian, ω_phys_vals;
                   aux_proj = nothing, tol=1e-9, maxdim::Int=100)
    H._tn_cache === nothing &&
        error("No Chebyshev cache found.  Call KPM_Tn(H, Ncheb; ...) first.")
    pos_sites = _pos_sites(H)
    Lpos      = length(pos_sites)
    Npos      = 2^Lpos
    pos_set   = Set(pos_sites)
    aux_sites = filter(s -> s ∉ pos_set, H.sites)
    FTirev = fix_sites(MPO(TCI.reverse(
        QuanticsTCI.quanticsfouriermpo(Lpos; sign=-1.0, normalize=true))), pos_sites)
    FTrev  = fix_sites(MPO(TCI.reverse(
        QuanticsTCI.quanticsfouriermpo(Lpos; sign=+1.0, normalize=true))), pos_sites)
    if H.aux_side === :pre
        for s in reverse(aux_sites)
            Id = Matrix{Float64}(LinearAlgebra.I, dim(s), dim(s))
            FTirev = prepend_op(FTirev, s, Id)
            FTrev  = prepend_op(FTrev,  s, Id)
        end
    else
        for s in aux_sites
            Id = Matrix{Float64}(LinearAlgebra.I, dim(s), dim(s))
            FTirev = postpend_op(FTirev, s, Id)
            FTrev  = postpend_op(FTrev,  s, Id)
        end
    end
    aux_combos = if isnothing(aux_proj) || isempty(aux_sites)
        collect(Iterators.product((1:dim(s) for s in aux_sites)...))
    else
        proj = aux_proj isa Integer ? fill(Int(aux_proj), length(aux_sites)) :
                                      collect(Int, aux_proj)
        [Tuple(proj)]
    end
    pos_kvals(k) = [((k >> (i-1)) & 1) + 1 for i in 1:Lpos]
    all_sites_ord = H.aux_side === :pre ? [aux_sites; pos_sites] : [pos_sites; aux_sites]
    kmps = Dict{Any, Vector{MPS}}()
    for σ_combo in aux_combos
        σ_vals = collect(Int, σ_combo)
        states = Vector{MPS}(undef, Npos)
        for k in 0:Npos-1
            all_vals = H.aux_side === :pre ? [σ_vals; pos_kvals(k)] : [pos_kvals(k); σ_vals]
            states[k+1] = _product_state_mps(all_sites_ord, all_vals)
        end
        kmps[σ_combo] = states
    end
    ω_resc = (collect(ω_phys_vals) .- H.center) ./ H.scale
    Nω     = length(ω_resc)
    Ak_w   = zeros(Float64, Npos, Nω)
    for (iω, ω) in enumerate(ω_resc)
        abs(ω) >= 1.0 && continue
        δH   = get_ldos_w_from_Tn(H._tn_cache, H._tn_Ncheb, ω; maxdim=maxdim)
        Op1  = apply(δH,                       FTirev; cutoff=tol, maxdim=maxdim)
        Akop = apply(swapprime(FTrev, 0 => 1), Op1;   cutoff=tol, maxdim=maxdim)
        for σ_combo in aux_combos, k in 0:Npos-1
            psi = kmps[σ_combo][k+1]
            Ak_w[k+1, iω] += real(inner(psi', Akop, psi))
        end
    end
    return Ak_w
end
=# # ── END OLD get_bands ────────────────────────────────────────────────────────
