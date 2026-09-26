# bands.jl — Momentum-space band structure via online Chebyshev KPM
#
# Contains get_bands (low-level MPO method and the TBHamiltonian overloads)
# (its helpers _eval_diag_mps and _kpm_weight_matrix now live in core/Utils.jl and
# solvers/kpm/kernels.jl).  Moved verbatim from sections 3, 4 and 5 of
# physics/QFT_tk.jl, together with that file's overview, which now describes the
# whole physics/qft/ folder.
#
# ─────────────────────────────────────────────────────────────────────────────
# Overview
# ─────────────────────────────────────────────────────────────────────────────
# The quantics representation encodes a 1D or 2D real-space position index as
# a binary string across L qubit sites. Conjugating any real-space MPO W by the
# Quantum Fourier Transform gives its momentum-space counterpart:
#
#   A(k,omega) = <k| U delta(omega - H) U^dag |k>
#
# For exciton Hamiltonians, `conjugate_by_qft_exciton` applies independent QFTs
# to the interleaved electron and hole registers. The resulting two-particle
# momentum basis can be sampled by total momentum `Q`: `get_exciton_bands`
# probes the coherent pair state sum_k |k,Q-k>, while `get_exciton_continuum`
# estimates the incoherent electron-hole continuum trace over |k,Q-k> with
# random-phase MPS probes. These exciton routines use online MPS-KPM on an
# already-QFT-conjugated MPO; they do not use the single-particle MPO-KPM
# `get_bands` pipeline.
#
# -----------------------------------------------------------------------------
# Online Chebyshev accumulation
# ─────────────────────────────────────────────────────────────────────────────
# Single-particle `get_bands` runs a Chebyshev recurrence over the full MPO
# space. At each step n the current T_n passes through five composable
# projection stages before the QFT is applied:
#
#   Step 0  nambu_proj   — project Nambu (BdG particle/hole) auxiliary index
#   Step 1  spin_proj    — project spin auxiliary index
#   Step 1c layer_proj   — project layer auxiliary index (bilayer/multilayer)
#   Step 1b sublat_proj  — project sublattice auxiliary index (kagome, Lieb, …)
#   Step 2  sublattice   — legacy mask sandwich (preset models without aux index)
#   Step 3  QFT + diagonal extraction + KPM weight accumulation
#
# All steps are independent and optional; any combination is valid.
# Peak memory: O(3 MPOs) regardless of Ncheb.
#
# ─────────────────────────────────────────────────────────────────────────────
# Auxiliary DOF projection  (section 5b, core/AuxDOF.jl)
# ─────────────────────────────────────────────────────────────────────────────
# Models with auxiliary DOFs (spin, Nambu, layer, sublattice) have an extra
# site at the front (`:pre`) or back (`:post`) of the MPO.  `project_aux`
# removes it by contracting |σ⟩⟨σ| onto the auxiliary tensor, returning an
# (L−1)-site position-only MPO ready for `conjugate_by_qft`.
#
# When `H::TBHamiltonian` is passed to `get_bands`, all auxiliary indices are
# auto-detected from the struct fields (H.spin_s, H.nambu_s, H.layer_s,
# H.sublattice_s) and never need to be passed manually.
#
# ─────────────────────────────────────────────────────────────────────────────
# High-symmetry k-path shortcut  (section 3b, kpath.jl)
# ─────────────────────────────────────────────────────────────────────────────
# The `kpath` kwarg in the `TBHamiltonian` overload of `get_bands` eliminates
# the manual kpath setup:
#
#   res = get_bands(H, Ncheb, 2, omega;
#                   kpath=[:G, :M, :Kp, :G], kpath_lattice=:honeycomb, num_x=30)
#   # res.Ak, res.ticks, res.labels  ← all path metadata included
#
# ─────────────────────────────────────────────────────────────────────────────
# Encoding conventions
# ─────────────────────────────────────────────────────────────────────────────
# 1D  — sites 1…L hold x bits, LSB at site 1 (quantics QFT convention).
# 2D  — sites 1…Ly hold iy bits (MSB first), sites Ly+1…L hold ix bits
#        (MSB first); linear index n = ix + iy·2^Lx (row-major).
#
# ─────────────────────────────────────────────────────────────────────────────
# Dependencies outside physics/qft/
# ─────────────────────────────────────────────────────────────────────────────
# fix_sites, _kpm_kernel               → utils.jl
# extract_diagonal_to_mps              → utils.jl
# _eval_diag_mps                       → core/Utils.jl
# interleave_mpo                       → core/MPOTools.jl
# _row_checker_mpo, _col_select_mpo    → lattice/masks2d.jl
# TBHamiltonian                        → TBSystem.jl
# _ensure_scale!                       → solvers/DMRG_tk.jl
# _kpm_weight_matrix                   → solvers/kpm/kernels.jl
# project_aux, aux_site, _autoenable_proj → core/AuxDOF.jl
# _run_kpm_mps!, _dos_weight_matrix    → KPM_tk.jl
# _kpm_weight_matrix                   → KPM_tk.jl
#
# ─────────────────────────────────────────────────────────────────────────────
# File structure  (src/physics/qft/, in include order)
# ─────────────────────────────────────────────────────────────────────────────
# conjugation.jl
#   1.  QFT conjugation
#       1a. Single-particle QFT    conjugate_by_qft, _embed_in_full_sites,
#                                  _embed_displacement_in_full_sites
#       1b. Exciton QFT            conjugate_by_qft_exciton
#       1c. k-space diagonal       get_spect_k
# bands.jl  (this file)
#   3.  Internal utilities         (none left here: _eval_diag_mps, ilinspace and
#                                  kspace_sampling_plan are in core/Utils.jl,
#                                  _kpm_weight_matrix in solvers/kpm/kernels.jl)
#   4.  Online band structure      get_bands (low-level MPO method)
#   5.  High-level overloads       get_bands (TBHamiltonian, single-particle)
# kpath.jl
#   3b. High-symmetry k-path       kpath_2d, hsk_honeycomb/square/triangular,
#                                  kpath_setup, _hs_label, _hsk
# exciton_spectra.jl
#       Exciton spectra (MPS-KPM)  get_exciton_bands, get_exciton_continuum
#       (exciton MPS probes mpsexcitonQ/QTrace/KQ now live in TwoParticle_tk.jl,
#        mpsexciton in core/Utils.jl)
# (5b. Aux index projection — project_aux, _autoenable_proj, aux_site — is in
#  core/AuxDOF.jl.)



# ============================================================
# 3. Internal utilities
#
# ilinspace       — evenly-spaced integer grid for k-center placement
# _kpm_weight_matrix — Chebyshev-KPM weights W[n, iω] (in solvers/kpm/kernels.jl)
# ============================================================

# `ilinspace` and `kspace_sampling_plan` (k-point centre placement and grouping
# shared with get_bands_gpu) live in core/Utils.jl with the other sampling plans;
# `_eval_diag_mps` (LSB-first diagonal readout) lives there beside `eval_mps`.


# ============================================================
# 4. Online band structure  —  get_bands
#
# ── Chebyshev recurrence ────────────────────────────────────────────────────
# Runs on the FULL MPO space (all auxiliary sites included):
#   T_0 = I,  T_1 = H̃,  T_n = 2 H̃ T_{n-1} − T_{n-2}    (H̃ = (H−center)/scale)
#
# ── Projection pipeline (five composable steps) ─────────────────────────────
# At each step n, T_n is passed through the following stages.  Each stage
# builds a list of position-only MPOs; every MPO in the final list is QFT'd,
# sampled, and its contribution added to Ak_w.
#
#   Step 0  nambu_proj  — Nambu (BdG particle/hole) aux index
#       Outermost aux (prepended last), projected first.
#       Sectors: 1=particle, 2=hole.  kwarg: proj_nambu.
#
#   Step 1  spin_proj   — spin aux index
#       After Nambu removal, spin is at site 1 of the reduced MPO.
#       spin_s_aux carries the explicit spin Index to avoid ambiguity when
#       both Nambu and spin are prepended.
#       Channels: 1=↑, 2=↓.  kwarg: proj_s.
#
#   Step 1c layer_proj  — layer aux index (bilayer / multilayer)
#       For bilayer_hamiltonian / twisted_bilayer_hamiltonian models (H.layer_s).
#       Sectors: 1…n_layers.  kwarg: proj_layer.
#
#   Step 1b sublat_proj — sublattice aux index (explicit aux models only)
#       For honeycomb_sublattice_hamiltonian, kagome_hamiltonian,
#       lieb_hamiltonian, dice_hamiltonian (H.sublattice_s set).
#       Sectors: 1…dim(sublat_s).  kwarg: proj_sl.
#
#   Step 2  sublattice  — LEGACY mask sandwich (preset models, no aux index)
#       For HUniform2Dhex, H2DChernhex, HUniform2Dtri, HQC2Dsquare, …
#       The sublattice structure is implicit in the hopping MPO.
#       Use this when H.sublattice_s is nothing.  kwarg: proj_sl (shared).
#
#   Step 3  QFT + diagonal extraction + KPM weight accumulation  (always)
#       T_k   = conjugate_by_qft(T_proj)
#       A_mps = extract_diagonal_to_mps(T_k)
#       for each k-group:  s = mean(_eval_diag_mps(A_mps, x) for x in group)
#       ak_accum[iω, ik] += W[n, iω] * s
#
# ── k-point groups ───────────────────────────────────────────────────────────
# Default (grid) mode: num_x centres placed with ilinspace in [xmin,xmax];
#   each centre is averaged over num_avg offset points (±half_step).
#   2D: offsets zipped diagonally, combined as (iy << Lx) | ix.
#
# Path mode: pass k_groups_override (from kpath_2d) or use the kpath kwarg
#   in the TBHamiltonian overload; this bypasses all grid parameters.
#
# ── Projection count per Chebyshev step ──────────────────────────────────────
#   nambu(×2) × spin(×2) × layer(×n) × sublat_aux(×dim) × sublat_mask(×2)
#   All contributions are summed unless a specific sector is selected via the
#   corresponding proj_* kwarg.
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

**Layer projection — Step 1c (bilayer / multilayer):**
- `layer_proj`   : project each T_n onto individual layers (default `false`).
- `proj_layer`   : `k` = layer k only, `nothing` = sum all layers.
- `layer_s`      : the layer `Index` (auto-detected from `H.layer_s`).
- `layer_side`   : `:pre` (default) — layer is always prepended.

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
                          layer_proj::Bool  = false,
                          proj_layer        = nothing,
                          layer_s           = nothing,
                          layer_side::Symbol  = :pre,
                          sublattice::Bool  = false,
                          proj_sl           = nothing,
                          sublat_proj::Bool = false,
                          sublat_s          = nothing,
                          sublat_side::Symbol = :post,
                          k_groups_override   = nothing,
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
    L_pos = L - (spin_proj ? 1 : 0) - (!isnothing(nambu_s) ? 1 : 0) -
                (!isnothing(layer_s) ? 1 : 0) - (!isnothing(sublat_s) ? 1 : 0)
    N     = 2^L_pos

    # ── Scaled Hamiltonian ────────────────────────────────────────────────────
    # sites already includes all aux sites; MPO(sites, "Id") is correctly sized.
    I_mpo = MPO(sites, "Id")
    Ham_n = (1 / scale) * +(H_mpo, (-center) * I_mpo; cutoff = cutoff)

    # ── KPM weight matrix  W[n, iω] ──────────────────────────────────────────
    Nω    = length(ω_vals)
    valid = [abs(ω) < 1.0 for ω in ω_vals]
    W     = _kpm_weight_matrix(Ncheb, ω_vals; kernel = kernel, lambda = lambda)

    # Lx is needed for both the 2D k-group builder and the sublattice mask builder;
    # compute it unconditionally so it is always in scope when D==2.
    Lx = D == 2 ? div(L_pos, 2) : 0

    # ── Build k-point groups (shared planner in core/Utils.jl) ────────────────
    # k_groups_override (from kpath_2d) bypasses the grid sampling entirely.
    kplan    = kspace_sampling_plan(L_pos, D; num_x, num_y, num_avg,
                                    xmin, xmax, ymin, ymax, k_groups_override)
    k_groups = kplan.k_groups
    num_x    = kplan.num_x

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
    !isnothing(layer_s)   && push!(aux_to_drop, layer_s::Index)
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
    #  Step 1c layer_proj         → project layer index             (×1 … ×n_layers)
    #  Step 1b sublat_proj        → project aux sublattice index    (×1 … ×dim)
    #  Step 2  sublattice (legacy)→ apply mask sandwich             (×1 or ×2)
    #
    # Aux sites are projected in outermost-first order (nambu → spin → layer →
    # sublat).  After each removal the next aux moves to position 1 of the
    # reduced MPO, so project_aux(:pre) always lands on the right site.
    local _nambu_side = nambu_side
    local _layer_side = layer_side
    local _sublat_side = sublat_side
    local _spin_idx    = isnothing(spin_s_aux) ? sites[1] : spin_s_aux
    function accumulate_Tn!(ak_accum, Tn, n)
        # Step 0: Nambu (BdG particle/hole) projection — outermost aux, project first.
        # proj_nambu=nothing → sum particle+hole; proj_nambu=1/2 → select one sector.
        after_nambu = nambu_proj ? [project_aux(Tn, nambu_s::Index, sec; side=_nambu_side)
                                    for sec in (isnothing(proj_nambu) ? (1:2) : (proj_nambu:proj_nambu))] : MPO[Tn]

        # Step 1: spin aux projection.
        # Uses spin_s_aux (explicit Index) when provided, falls back to sites[1].
        # proj_s=nothing → sum both channels; proj_s=1/2 → select one.
        after_spin = spin_proj ? [project_aux(T, _spin_idx, sec; side=:pre)
                                  for T in after_nambu, sec in (isnothing(proj_s) ? (1:2) : (proj_s:proj_s))] : after_nambu

        # Step 1c: layer projection (bilayer / multilayer with H.layer_s).
        # proj_layer=nothing → sum all layers; proj_layer=k → select layer k.
        after_layer = if layer_proj
            n_lay = dim(layer_s::Index)
            lay_range = isnothing(proj_layer) ? (1:n_lay) : (proj_layer:proj_layer)
            [project_aux(T, layer_s::Index, sec; side=_layer_side)
             for T in after_spin for sec in lay_range]
        else
            after_spin
        end

        # Step 1b: sublattice aux projection (kagome/Lieb/honeycomb with H.sublattice_s).
        # proj_sl=nothing → sum all sublattices; proj_sl=k → select sublattice k.
        after_sl_aux = if sublat_proj
            sl_range = isnothing(proj_sl) ? (1:dim(sublat_s::Index)) : (proj_sl:proj_sl)
            [project_aux(T, sublat_s::Index, sec; side=_sublat_side)
             for T in after_layer for sec in sl_range]
        else
            after_layer
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


# ============================================================
# 5. High-level overloads  —  get_bands
# ============================================================

"""
    get_bands(H, Ncheb, D, ω_phys_vals; kwargs...)
        -> Matrix{Float64}  or  NamedTuple(Ak, ticks, labels)

High-level overload of `get_bands` for a `TBHamiltonian`.

Physical energies `ω_phys_vals` are rescaled via `H.scale` and `H.center`.

**Auto-detection:** All auxiliary site Indices (Nambu, spin, layer, sublattice)
and their positions (:pre/:post) are read from the struct fields and excluded
from position k-space automatically — no manual index passing required.

**Projection kwargs** (forwarded verbatim to the low-level MPO method):
`spin_proj`, `proj_s`, `nambu_proj`, `proj_nambu`, `layer_proj`, `proj_layer`,
`sublat_proj`, `proj_sl`, `sublattice`.

**High-symmetry k-path shortcut** — replaces the manual `hsk_*` + `kpath_2d`
+ `k_groups_override` boilerplate with a single call:

```julia
res = get_bands(H, Ncheb, 2, omega;
                kpath=[:G, :M, :Kp, :G], kpath_lattice=:honeycomb, num_x=30)
```

- `kpath`         : symbol vector defining the path.  Use Latin aliases:
                    `G`=Γ, `M`, `K`, `Kp`=K', `X` — no special characters needed.
- `kpath_lattice` : `:honeycomb`, `:square`, or `:triangular`.
- `kpath_Lx`      : Lx for the 2D grid; defaults to `H.L ÷ 2`.
- `num_x`         : **reused as `npts_per_segment`** when `kpath` is given.

When `kpath` is set the return value is a `NamedTuple`:
  `(Ak = Matrix{Float64}(Nω×Nk),  ticks = Vector{Int},  labels = Vector{String})`

  ```julia
  heatmap(1:size(res.Ak,2), omega, res.Ak; xticks=(res.ticks, res.labels))
  vline!(p, res.ticks; ls=:dash, color=:white)
  ```

Otherwise returns `Matrix{Float64}` as usual (backward-compatible).
"""
function get_bands(H::TBHamiltonian, Ncheb::Int, D::Int, ω_phys_vals;
                          kpath             = nothing,
                          kpath_lattice     = nothing,
                          kpath_Lx          = nothing,
                          spin_proj::Bool   = false,
                          proj_s            = nothing,
                          nambu_proj::Bool  = false,
                          proj_nambu        = nothing,
                          layer_proj::Bool  = false,
                          proj_layer        = nothing,
                          sublattice::Bool  = false,
                          proj_sl           = nothing,
                          sublat_proj::Bool = false,
                          k_groups_override   = nothing,
                          xmin::Int       = 0,
                          xmax            = nothing,
                          num_x::Int      = 60,
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

    _require_binary_position_space(H, "get_bands")
    _ensure_scale!(H)
    nambu_proj, spin_proj, layer_proj, sublat_proj =
        _autoenable_proj(H, nambu_proj, spin_proj, layer_proj, sublat_proj)

    ω_resc = (collect(ω_phys_vals) .- H.center) ./ H.scale

    # ── High-symmetry path shortcut ──────────────────────────────────────────
    # When `kpath` is provided, build k_groups_override from symbols and use
    # num_x as npts_per_segment.  Returns a NamedTuple with tick info.
    kpath_ticks = nothing; kpath_labels = nothing
    if !isnothing(kpath)
        isnothing(kpath_lattice) && error(
            "kpath requires kpath_lattice (:honeycomb, :square, or :triangular).")
        Lx_kp = isnothing(kpath_Lx) ? H.L ÷ 2 : Int(kpath_Lx)
        Ly_kp = H.L - Lx_kp
        k_groups_override, kpath_ticks, kpath_labels =
            kpath_setup(kpath_lattice, Lx_kp, Ly_kp, kpath; npts_per_segment = num_x)
    end

    # Auto-detect all aux indices so the low-level function can exclude them
    # from L_pos and pos_sites regardless of which projections are active.
    nambu_s_det, nambu_side_det = !isnothing(H.nambu_s) ?
        aux_site(H, :nambu) : (nothing, :pre)

    spin_s_det = H.spin_s   # may be nothing; low-level falls back to sites[1] when nothing

    layer_s_det, layer_side_det = !isnothing(H.layer_s) ?
        aux_site(H, :layer) : (nothing, :pre)

    sublat_s_det, sublat_side_det = !isnothing(H.sublattice_s) ?
        aux_site(H, :sublattice) : (nothing, :post)

    Ak_w = get_bands(H.mpo, H.scale, H.center, H.sites, Ncheb, D, ω_resc;
                            spin_proj  = spin_proj,  proj_s     = proj_s,
                            spin_s_aux = spin_s_det,
                            nambu_proj = nambu_proj, proj_nambu = proj_nambu,
                            nambu_s    = nambu_s_det, nambu_side = nambu_side_det,
                            layer_proj = layer_proj, proj_layer  = proj_layer,
                            layer_s    = layer_s_det, layer_side = layer_side_det,
                            sublattice = sublattice, proj_sl    = proj_sl,
                            sublat_proj = sublat_proj,
                            sublat_s    = sublat_s_det,
                            sublat_side = sublat_side_det,
                            k_groups_override = k_groups_override,
                            xmin = xmin, xmax = xmax,
                            num_x = num_x, num_avg = num_avg,
                            ymin = ymin, ymax = ymax, num_y = num_y,
                            kernel = kernel, lambda = lambda,
                            tol = tol, maxdim = maxdim, cutoff = cutoff,
                            printinfo = printinfo)

    # When kpath was used, return a NamedTuple carrying the tick info so the
    # caller can use result.Ak, result.ticks, result.labels directly in plots.
    return isnothing(kpath_ticks) ? Ak_w :
           (Ak = Ak_w, ticks = kpath_ticks, labels = kpath_labels)
end


"""
    get_bands(H, Ncheb, ω_phys_vals; kwargs...)

Convenience overload that infers the spatial dimension `D` from `H.geometry`
(via `length(H.geometry(1))`), so callers do not need to pass `D` explicitly.
All keyword arguments are forwarded unchanged to the 4-argument form.

Errors if `H.geometry` is `nothing` (custom or geometry-free Hamiltonians must
still pass `D` explicitly via the 4-argument form).
"""
function get_bands(H::TBHamiltonian, Ncheb::Int, ω_phys_vals; kwargs...)
    isnothing(H.geometry) &&
        error("get_bands without explicit D requires H.geometry to be set. " *
              "Pass D (1 or 2) as the third argument, or set H.geometry.")
    D = length(H.geometry(1))
    return get_bands(H, Ncheb, D, ω_phys_vals; kwargs...)
end
