# solvers/kpm/ldos.jl — online real-space LDOS (nothing cached on H)
#
# Contents: the LDOS at one position for all energies, from an MPS Chebyshev
# recursion started at |X⟩ (get_ldos_online), and the spatially resolved LDOS map
# over a sampling plan of positions (get_ldos_spatial), computed by one online MPO
# recursion (mode=:mpo) or one MPS recursion per position and sector (mode=:mps).
# Both sum or select auxiliary-DOF sectors (Nambu, spin, layer, sublattice).
#
# Entry points: get_ldos_online, get_ldos_spatial
# Depends on: core/Utils.jl (spatial_sampling_plan, extract_diagonal_to_mps,
#   _eval_block_mps, binary_to_MPS), core/TBSystem.jl (TBHamiltonian,
#   physical_projector, site_permutation), core/AuxDOF.jl (_aux_projection,
#   _probe_sectors, probe_state, _project_aux_sectors), solvers/DMRG.jl
#   (_ensure_scale!), solvers/kpm/kernels.jl (_kpm_energy_grid),
#   solvers/kpm/recursion.jl (_scaled_hamiltonian, _run_kpm_mps!).
#
# Split from the former solvers/KPM_tk.jl in Tier 1.

# ============================================================
# 1. Online LDOS at a single position: get_ldos_online
# ============================================================

"""
    get_ldos_online(H::TBHamiltonian, Ncheb::Int, X::Int, ω_phys_vals;
                    kernel=:jackson, lambda=4.0, maxdim=100, cutoff=1e-8,
                    verbose=false,
                    nambu_proj=false, proj_nambu=nothing, spin_proj=false,
                    proj_s=nothing, layer_proj=false, proj_layer=nothing,
                    sublat_proj=false, proj_sl=nothing)
        -> Vector{Float64}

Online real-space LDOS at unit-cell position `X` for all physical energies in
`ω_phys_vals`.  Never stores more than **3 MPS** simultaneously (no Chebyshev cache).

**Algorithm**: MPS Chebyshev recursion
`|φ_k⟩ = T_k(H̃)|X⟩` with moment accumulation `μ_k = ⟨X|φ_k⟩`.
Auxiliary DOF sectors are summed by running the recursion once per requested sector.

`X ∈ {1, …, H.N}` is the 1-indexed unit-cell position.

**Keywords**

- `kernel`, `lambda` — KPM kernel (`:jackson`, `:lorentz` with `lambda`, `:fejer`,
  `:dirichlet`).
- `maxdim`, `cutoff` — truncation of each MPS recursion step.
- `verbose` — print the bond dimension every 10 Chebyshev steps and at the last.

**Auxiliary DOF projections** (same interface as `get_bands` and `get_ldos_spatial`):

- `spin_proj`, `nambu_proj`, `layer_proj`, `sublat_proj` — enable projection of the
  corresponding auxiliary DOF auto-detected from `H`. A DOF present on `H` is always
  projected: its flag is switched on automatically, with an info line.
- `proj_s`, `proj_nambu`, `proj_layer`, `proj_sl` — sector selector: `nothing` sums
  all sectors of that DOF; an integer selects a single sector (1-based).
- Contributions from all requested sectors are accumulated into a single result vector.

Returns `Vector{Float64}` of length `Nω` with `0.0` outside the spectral support.

Examples
--------
```julia
ωlist = range(-3.0, 3.0; length=300)
ldos  = get_ldos_online(H, 200, 2^(H.L-1), ωlist)          # no aux DOF

# Spin-summed LDOS at site 16
ldos_tot = get_ldos_online(H_spin, 200, 16, ωlist; spin_proj=true)

# Spin-↑ LDOS only
ldos_up  = get_ldos_online(H_spin, 200, 16, ωlist; spin_proj=true, proj_s=1)
```
"""
function get_ldos_online(H::TBHamiltonian, Ncheb::Int, X::Int, ω_phys_vals;
                          kernel::Symbol = :jackson,
                          lambda::Real   = 4.0,
                          maxdim::Int    = 100,
                          cutoff::Real   = 1e-8,
                          verbose::Bool  = false,
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

    Ham_n = _scaled_hamiltonian(H; cutoff=cutoff)

    ω_vals, W, denom, valid = _kpm_energy_grid(H, Ncheb, ω_phys_vals;
                                               kernel=kernel, lambda=lambda)
    Nω     = length(ω_vals)
    accum  = zeros(Float64, Nω)

    aux = _aux_projection(H; nambu_proj, proj_nambu, spin_proj, proj_s,
                             layer_proj, proj_layer, sublat_proj, proj_sl)

    # ── MPS-based Chebyshev recursion, summed over requested aux sectors ──────
    for σ in _probe_sectors(aux)
        _run_kpm_mps!(Ham_n, probe_state(H, X, σ), Ncheb, W, valid, accum;
                      cutoff=cutoff, maxdim=maxdim,
                      verbose=verbose, label="get_ldos_online")
    end  # sector loop

    result = zeros(Float64, Nω)
    for iω in 1:Nω
        valid[iω] || continue
        result[iω] = accum[iω] / denom[iω]
    end
    return result
end


# ============================================================
# 2. Spatial LDOS at multiple positions (real-space analogue of get_bands)
# ============================================================

"""
    get_ldos_spatial(H::TBHamiltonian, Ncheb::Int, ω_phys_vals;
                     num_x=H.N, num_y=nothing, num_avg=1, mode=:mpo,
                     x_start=1, x_end=H.N, x_groups=nothing,
                     grid=false, xwin=nothing, ywin=nothing, box_half=0,
                     reduce=:point, sublattice=:auto,
                     kernel=:jackson, lambda=4.0, maxdim=100, cutoff=1e-8,
                     verbose=false,
                     ordering=:physical, conumber_orientation=:standard,
                     conumber_centered=true, conumber_origin=0,
                     conumber_alignment=:atomic,
                     nambu_proj=false, proj_nambu=nothing, spin_proj=false,
                     proj_s=nothing, layer_proj=false, proj_layer=nothing,
                     sublat_proj=false, proj_sl=nothing)
        -> Matrix{Float64}

Spatially-resolved LDOS, real-space analogue of `get_bands`.

**Sampling procedures (`reduce`)** — full detail in [`spatial_sampling_plan`](@ref).

- `:point` (default) — read the LDOS *at* `num_x[×num_y]` sample cells; with
  `box_half > 0` each pixel is the mean over a `(2·box_half+1)²` box. Cheap, but
  a grid coarser than a feature's width **aliases** it (thin in-gap edge /
  domain-wall channels can fall between pixels and be missed).
- `:block` — partition the system into `num_x × num_y` blocks (powers of two) and
  report the **integral** over each block, computed by tracing out the
  within-block position bits (a partial contraction, cost independent of block
  size). Gap-free: every cell belongs to one block, so a thin feature on a gapped
  background **cannot** be missed. Use it for large-scale maps of edge networks.
  Output columns are row-major over coarse pixels (`col = ixp + iyp·num_x + 1`);
  block centres come from the plan. `:mpo` mode only.

**Return shape (sublattice geometry-awareness)**

For a multi-atom unit cell (`H.sublattice_s` set) the layout depends on the
*sampling scale*, decided by [`spatial_sampling_plan`](@ref) from the local
stride (see `sublattice` below):

- **resolved** (atomic scale — every unit cell probed, or `proj_sl=k`):
  `(Nω × ng×n_sub)`, columns interleaved in atom order matching the
  `*_positions` functions — `[A₀, B₀, A₁, B₁, …]` (2-sublattice),
  `[A₀, B₀, C₀, …]` (3-sublattice). `proj_sl=k` fills only sublattice `k`.
- **averaged** (large scale — the grid skips unit cells): `(Nω × ng)`, the
  sublattice is traced out into one value per unit cell (mean over the `n_sub`
  atoms).

With no sublattice DOF the shape is always `(Nω × ng)`. `ng` is the number of
sample groups of the plan: `num_x` for the default 1D sweep, `num_x × num_y` for
`grid=true` or `reduce=:block`, `length(x_groups)` for explicit groups.

**`sublattice` (resolve vs average)**

- `:auto` (default) — resolve when sampling at full unit-cell resolution
  (`stride == 1`, e.g. zooming a small window and probing every cell); average
  whenever the grid is coarser or `box_half > 0`.
- `:resolve` — always emit per-atom columns. `:average` — always trace the
  sublattice to one value per cell. `proj_sl=k` always resolves that one atom.

**Sampling parameters** (passed to [`spatial_sampling_plan`](@ref))

- `num_x`/`num_y` : sample counts (per axis for `grid=true`; `num_x` is the total
  for the default 1D linear sweep). `num_x` defaults to `H.N` = full resolution;
  `num_y=nothing` (default) uses the same count as `num_x` (capped at the window).
- `num_avg`  : sub-samples per coarse block for local averaging (1D, default `1`).
- `x_start`, `x_end` : 1-indexed linear position range (1D layout; defaults `1`,
  `H.N`).
- `grid`     : `true` lays centers on a 2D `num_x × num_y` unit-cell grid.
  Default `false`.
- `xwin`, `ywin` : 0-indexed unit-cell `(lo, hi)` windows for `grid=true` (e.g.
  zoom into a patch of a large system). Default `nothing` = the full system.
- `x_groups` : explicit `Vector{Vector{Int}}` override (treated as atomic).
  Default `nothing`.
- `box_half` : 2D neighbourhood half-width (averages, forces sublattice averaging).
  Default `0`.
- `reduce`   : `:point` (default; sample/box) or `:block` (block-integrate; see
  above).
- `sublattice` : `:auto` (default), `:resolve` or `:average` (see above).

`grid`, `xwin`, `ywin`, `box_half > 0` and `reduce=:block` need a 2D `H.geometry`.

**Modes**

- `:mpo` (default) — single Chebyshev pass; evaluates all positions simultaneously.
  Cost `∝ Ncheb × (MPO×MPO)`, independent of `num_x` or `n_sub`.
- `:mps` — independent MPS recursion per (position, sector) combination.

**Chebyshev keywords**

- `kernel`, `lambda` : KPM kernel (`:jackson` default, `:lorentz` with `lambda`
  (default `4.0`), `:fejer`, `:dirichlet`).
- `maxdim`, `cutoff` : truncation of each recursion step (defaults `100`, `1e-8`).
- `verbose` : print progress (every 15 steps in `:mpo` mode, every 15 positions in
  `:mps` mode). Default `false`.

**Auxiliary DOF projections** (same interface as `get_bands`):
`nambu_proj`/`proj_nambu`, `spin_proj`/`proj_s`, `layer_proj`/`proj_layer`,
`sublat_proj`/`proj_sl`. The flags default to `false` and the selectors to
`nothing` (sum all sectors); an integer selector picks one 1-based sector. A DOF
present on `H` is always projected: its flag is switched on automatically, with
an info line (`sublat_proj` is kept only for backward compatibility).

**Site ordering** (`ordering`, `conumber_*`)

- `ordering=:physical` (default) samples positions in physical order.
- `ordering=:conumber` (Fibonacci position spaces only) orders the columns by
  conumber: the groups become `[[x] for x in site_permutation(H; ordering=:conumber,
  …)]`, with `conumber_orientation` (`:standard` default, or `:reversed`),
  `conumber_centered` (default `true`; it only relabels the conumbers around zero,
  so the column order does not depend on it), `conumber_origin` (default `0`,
  shifts the site index before conumbering) and `conumber_alignment` passed on as
  `orientation`, `centered`, `origin` and `alignment`. It requires full-resolution
  1D point sampling (`num_x`, `num_y`, `num_avg`, `x_start`, `x_end`, `x_groups`,
  `grid`, `xwin`, `ywin`, `box_half`, `reduce` at their defaults) and no
  auxiliary DOF.

`conumber_alignment=:atomic` (default) places the `AA` sites in one central
block; `:raw` exposes the unshifted modular residues. A recursive atomic zoom
should slice the interval returned by `fibonacci_rg_partition` rather than
re-conumbering its sites with a reduced `L`.

Examples
--------
```julia
# Standard 1D chain — shape (Nω × 8)
ldos = get_ldos_spatial(H, 200, ωlist; num_x=8)

# Honeycomb, large-scale map — sublattice averaged, shape (Nω × 64)
ldos_uc  = get_ldos_spatial(H_hc, 200, ωlist; num_x=64)

# Honeycomb, atomic zoom into a 50×50 patch — sublattice resolved (Nω × 50*50*2)
ldos_zoom = get_ldos_spatial(H_hc, 200, ωlist; grid=true,
                             xwin=(1000, 1049), ywin=(1000, 1049))

# Large 2^14×2^14 system: 128×128 block-integrated map — catches thin in-gap
# edge channels that point sampling would alias away. Shape (Nω × 128*128).
ldos_blk = get_ldos_spatial(H_big, 200, ωlist; reduce=:block, num_x=128, num_y=128)

# Kagome: sublattice A only — only A columns filled, B/C columns zero
ldos_A   = get_ldos_spatial(H_kg, 200, ωlist; proj_sl=1, num_x=H_kg.N)
```
"""
function get_ldos_spatial(H::TBHamiltonian, Ncheb::Int, ω_phys_vals;
                           num_x::Int     = H.N,
                           num_y          = nothing,
                           num_avg::Int   = 1,
                           mode::Symbol   = :mpo,
                           x_start::Int   = 1,
                           x_end::Int     = H.N,
                           x_groups       = nothing,
                           grid::Bool     = false,
                           xwin           = nothing,
                           ywin           = nothing,
                           box_half::Int  = 0,
                           reduce::Symbol = :point,
                           sublattice::Symbol = :auto,
                           kernel::Symbol = :jackson,
                           lambda::Real   = 4.0,
                           maxdim::Int    = 100,
                           cutoff::Real   = 1e-8,
                           verbose::Bool  = false,
                           ordering::Symbol = :physical,
                           conumber_orientation::Symbol = :standard,
                           conumber_centered::Bool = true,
                           conumber_origin::Integer = 0,
                           conumber_alignment::Symbol = :atomic,
                           nambu_proj::Bool  = false,
                           proj_nambu        = nothing,
                           spin_proj::Bool   = false,
                           proj_s            = nothing,
                           layer_proj::Bool  = false,
                           proj_layer        = nothing,
                           sublat_proj::Bool = false,   # kept for backward compat; auto-on when H.sublattice_s is set
                           proj_sl           = nothing)

    ordering in (:physical, :conumber) ||
        throw(ArgumentError("ordering must be :physical or :conumber"))
    x_groups_effective = x_groups
    if ordering === :conumber
        full_resolution = num_x == H.N && num_y === nothing && num_avg == 1 &&
            x_start == 1 && x_end == H.N && x_groups === nothing && !grid &&
            xwin === nothing && ywin === nothing && box_half == 0 && reduce === :point &&
            H.spin_s === nothing && H.nambu_s === nothing && H.layer_s === nothing &&
            H.sublattice_s === nothing
        full_resolution || throw(ArgumentError(
            "ordering=:conumber currently requires full-resolution 1D point sampling " *
            "without averaging, grids, blocks, custom groups, or auxiliary degrees of freedom"
        ))
        permutation = site_permutation(
            H; ordering=:conumber,
            orientation=conumber_orientation,
            centered=conumber_centered,
            origin=conumber_origin,
            alignment=conumber_alignment,
        )
        x_groups_effective = [[x] for x in permutation]
    end

    # ── Geometry-aware sampling plan (unit-cell groups + sublattice decision) ──
    if box_half > 0 || grid || xwin !== nothing || ywin !== nothing || reduce === :block
        isnothing(H.geometry) &&
            error("get_ldos_spatial: box_half/grid/window/block sampling requires H.geometry to be set.")
        length(H.geometry(1)) == 2 ||
            error("get_ldos_spatial: box_half/grid/window/block sampling is only supported for 2D systems.")
    end
    reduce === :block && mode === :mps &&
        error("get_ldos_spatial: reduce=:block is only supported in mode=:mpo.")
    Lx_uc   = something(H.Lx, H.L ÷ 2)
    Ly_uc   = H.L - Lx_uc
    n_sub_H = isnothing(H.sublattice_s) ? 1 : dim(H.sublattice_s)
    plan = spatial_sampling_plan(H.L;
        Lx       = Lx_uc,
        grid     = grid,
        reduce   = reduce,
        n_sub    = n_sub_H,
        num_x    = num_x, num_y = num_y, num_avg = num_avg,
        x_start  = x_start, x_end = x_end,
        xwin     = xwin, ywin = ywin,
        x_groups = x_groups_effective, box_half = box_half,
        sublattice = sublattice)
    groups   = plan.groups
    is_block = plan.reduce === :block
    block_a  = plan.a
    block_b  = plan.b
    nbx      = 2^block_a

    _ensure_scale!(H)
    # Every aux DOF present on H is projected (the flags are switched on here).
    aux = _aux_projection(H; nambu_proj, proj_nambu, spin_proj, proj_s,
                             layer_proj, proj_layer, sublat_proj, proj_sl)

    # ── Bernal top-view guard ─────────────────────────────────────────────────
    if aux.layer.on && isnothing(proj_layer) && !isnothing(aux.sublat.index)
        n_lay = dim(aux.layer.index::Index)
        @warn """get_ldos_spatial: proj_layer=nothing on a layered+sublattice Hamiltonian.
  Result accumulates sublattice columns by label across all $n_lay layers.
  For Bernal stacking this is NOT the physical top-view: even layers have their
  sublattice-A/B registries physically swapped in 2D, so the sum is misleading.
  This call is also $(n_lay)× slower than a single-layer call.
  For a correct Bernal top-view use:
    ldos_layers = [get_ldos_spatial(H, Nc, ωlist; proj_layer=k, ...) for k in 1:$n_lay]
    plot_ldos_multilayer(ldos_layers, ωlist, ω; stacking=:Bernal, ...)
  For AA stacking the label-wise sum is physically correct (this warning fires
  regardless of stacking type; shown only once per session).""" maxlog=1
    end

    # ── Sublattice layout ─────────────────────────────────────────────────────
    # When H.sublattice_s is set the geometry-aware plan decides the layout:
    #   • resolve (atomic scale, or proj_sl=k): one column per atom,
    #       shape (Nω, ng × n_sub),  col = (ig-1)*n_sub + s   (matches the
    #       honeycomb/kagome/lieb positions-function atom ordering; proj_sl=k
    #       fills only sublattice k, others stay 0).
    #   • average (large scale): the sublattice is traced out — one value per
    #       unit cell, shape (Nω, ng),  col = ig  (mean over the n_sub atoms).
    has_sublat   = !isnothing(aux.sublat.index)
    n_sub        = has_sublat ? dim(aux.sublat.index::Index) : 1
    # proj_sl=k pins a single sublattice → always resolved (that one column).
    resolve_sl   = has_sublat && (plan.resolve_sublattice || !isnothing(proj_sl))
    average_sl   = has_sublat && !resolve_sl
    sl_fill      = has_sublat ?
        (isnothing(proj_sl) ? (1:n_sub) : (proj_sl:proj_sl)) :
        (1:1)

    I_mpo = physical_projector(H)
    Ham_n = _scaled_hamiltonian(H; cutoff=cutoff, identity=I_mpo)

    ω_vals, W, denom, valid = _kpm_energy_grid(H, Ncheb, ω_phys_vals;
                                               kernel=kernel, lambda=lambda)
    Nω     = length(ω_vals)

    ng     = length(groups)
    # Averaging collapses the n_sub atoms into one column per group.
    n_cols = average_sl ? ng : ng * n_sub
    result = zeros(Float64, Nω, n_cols)

    if mode == :mps
        # ── MPS mode ──────────────────────────────────────────────────────────
        n_total = sum(length(g) for g in groups)
        n_done  = 0
        sectors = _probe_sectors(aux)

        for (ig, grp) in enumerate(groups)
            grp_accum = zeros(Float64, Nω, n_sub)  # per-sublattice accumulator

            for x in grp
                for σ in sectors
                    σ_sl = σ === nothing ? 1 : σ[4]   # the sublattice column
                    accum_loc = zeros(Float64, Nω)
                    _run_kpm_mps!(Ham_n, probe_state(H, x, σ), Ncheb, W, valid, accum_loc;
                                  cutoff=cutoff, maxdim=maxdim)

                    for iω in 1:Nω
                        valid[iω] || continue
                        grp_accum[iω, σ_sl] += accum_loc[iω] / denom[iω]
                    end
                end  # sector loop
            end  # x

            if average_sl
                # Trace out the sublattice: one column per unit cell (mean atom).
                result[:, ig] = vec(sum(grp_accum; dims=2)) ./ (n_sub * length(grp))
            else
                for s in sl_fill
                    result[:, (ig-1)*n_sub + s] = grp_accum[:, s] ./ length(grp)
                end
            end

            n_done += length(grp)
            verbose && n_done % 15 == 0 &&
                println("get_ldos_spatial [:mps]  group $ig/$ng  ($n_done/$n_total)")
        end

    elseif mode == :mpo
        # ── MPO mode: single online Chebyshev pass ────────────────────────────
        all_xs = unique(vcat(groups...))

        # Build position-only eval states: drop every projected aux site present
        # on H (the flags are on for all of them, the sublattice included).
        aux_to_drop = Set{Index}(d.index for d in (aux.nambu, aux.spin, aux.layer, aux.sublat)
                                 if d.on && !isnothing(d.index))
        pos_sites = filter(s -> s ∉ aux_to_drop, H.sites)

        psi_dict = if isempty(aux_to_drop)
            Dict(x => probe_state(H, x) for x in all_xs)
        else
            @assert length(pos_sites) == H.L "get_ldos_spatial: $(length(pos_sites)) position sites after dropping aux but expected H.L=$(H.L)."
            Dict(x => binary_to_MPS(x - 1, H.L, pos_sites) for x in all_xs)
        end

        accum = zeros(Float64, Nω, n_cols)

        # Reduce a diagonal profile MPS to per-pixel scalars, returning (u, value)
        # pairs where u is the 1-indexed output pixel (column unit):
        #   reduce=:point → mean of inner products over each group's cells,
        #   reduce=:block → integral over each coarse block (_eval_block_mps).
        function spatial_vals_cpu(diag_n)
            if is_block
                return [(ixp + iyp * nbx + 1,
                         _eval_block_mps(diag_n, ixp, iyp, block_a, block_b, Lx_uc, Ly_uc))
                        for iyp in 0:(2^block_b - 1) for ixp in 0:(nbx - 1)]
            else
                return [(ig, sum(real(inner(psi_dict[x], diag_n)) for x in grp) / length(grp))
                        for (ig, grp) in enumerate(groups)]
            end
        end

        # Project T_n onto every aux sector (nambu → spin → layer → sublattice,
        # core/AuxDOF.jl) and accumulate each projection's diagonal. Resolved
        # sublattice → each sector s gets its own column; averaged (large scale)
        # → all sectors fold into the single per-pixel column u (mean over the
        # n_sub atoms). Without a sublattice (n_sub = s = 1, scale 1.0) the
        # column is the pixel u.
        function accumulate_Tn!(Tk, n)
            for (Tp, s) in _project_aux_sectors(Tk, aux; sublattice=has_sublat)
                diag_n = ITensorMPS.truncate!(extract_diagonal_to_mps(Tp); cutoff=cutoff)
                scale  = average_sl ? 1.0 / n_sub : 1.0
                for (u, val) in spatial_vals_cpu(diag_n)
                    col = average_sl ? u : (u - 1) * n_sub + s
                    for iω in 1:Nω
                        valid[iω] || continue
                        accum[iω, col] += W[n, iω] * val * scale
                    end
                end
            end
        end

        Tkm2 = I_mpo;  Tkm1 = Ham_n
        accumulate_Tn!(Tkm2, 1);  accumulate_Tn!(Tkm1, 2)

        for k in 3:Ncheb
            Tk   = +(2 * apply(Ham_n, Tkm1; cutoff=cutoff), -Tkm2; maxdim=maxdim)
            Tk   = ITensorMPS.truncate!(Tk; cutoff=cutoff)
            accumulate_Tn!(Tk, k)
            Tkm2 = Tkm1;  Tkm1 = Tk
            verbose && (k % 15 == 0 || k == Ncheb) &&
                println("get_ldos_spatial [:mpo]  step $k/$Ncheb  " *
                        "maxlinkdim=$(maxlinkdim(Tkm1))")
        end

        for iω in 1:Nω
            valid[iω] || continue
            result[iω, :] = accum[iω, :] ./ denom[iω]
        end

    else
        error("Unknown mode :$mode. Choose :mps or :mpo")
    end

    return result
end
