# gpu/kpm.jl — GPU Chebyshev KPM: the MPO recurrence KPM_Tn_gpu, the spatial LDOS
# from the MPO recurrence (get_ldos_spatial_gpu) or from independent MPS recursions
# per probe (get_ldos_spatial_mps_gpu), and the stochastic DOS
# (get_dos_stochastic_gpu). Moved from the former gpu/GPU_tk.jl; the CPU helper
# _reconstruct_ldos_moment_columns it uses lives in solvers/kpm/kernels.jl.
#
# Main entry points: KPM_Tn_gpu, get_ldos_spatial_gpu, get_ldos_spatial_mps_gpu,
# get_dos_stochastic_gpu.
# Depends on: core/Utils.jl (spatial_sampling_plan, interval_sampling_plan,
# basis-state and exciton MPS builders), core/TBSystem.jl (position-space interface), core/AuxDOF.jl (the
# aux projection _aux_projection/_project_aux_sectors, projected probes
# _probe_sectors/probe_state), solvers/DMRG.jl (spectral bounds, _ensure_scale!),
# solvers/kpm/kernels.jl (energy grid, moment-column reconstruction),
# solvers/kpm/recursion.jl (_scaled_hamiltonian), gpu/device.jl, gpu/primitives.jl.


# ============================================================
# 1. Chebyshev recurrence
# ============================================================

"""
    KPM_Tn_gpu(H_mpo, N, sites; scale=nothing, center=0.0, maxdim=40,
               dmrg_nsweeps=5, dmrg_maxdim=[10, 20, 40], dmrg_linkdim=4,
               cutoff=1e-8, keep_indices=nothing, type=ComplexF32, dtype=nothing,
               verbose=true)
        -> (Tn_list, scale, center)

GPU version of `KPM_Tn`.  Moves the identity and scaled Hamiltonian MPOs to
GPU before the recurrence so all Tn tensors stay on GPU; `Tn_list` holds
T_0 … T_N. Requires `using CUDA`.

`type` (alias `dtype`) is the GPU element type: `ComplexF32` (default),
`ComplexF64`, or `Float32`/`Float64` for a real `H_mpo`. With `scale=nothing`,
`scale` and `center` are estimated by DMRG (`dmrg_nsweeps`, `dmrg_maxdim`,
`dmrg_linkdim`) and the estimated `center` replaces the one passed.

If `keep_indices` is provided (a `Set{Int}`, 1-based into the returned vector
where index 1 = T_0, 2 = T_1, …), only those Tns are retained in memory.
All other slots are set to `nothing`.  The recurrence itself always runs to
completion — `keep_indices` only controls which results are stored.
"""
function KPM_Tn_gpu(H_mpo::MPO, N::Int, sites;
                    scale::Union{Real,Nothing}          = nothing,
                    center::Real                        = 0.0,
                    maxdim::Int                         = 40,
                    dmrg_nsweeps::Int                   = 5,
                    dmrg_maxdim                         = [10, 20, 40],
                    dmrg_linkdim::Int                   = 4,
                    cutoff::Real                        = 1e-8,
                    keep_indices::Union{Nothing,AbstractSet{Int}} = nothing,
                    type::Type{<:Number}                = ComplexF32,
                    dtype::Union{Nothing,Type{<:Number}} = nothing,
                    verbose::Bool                       = true)

    _check_gpu("KPM_Tn_gpu")
    gpu_type = _resolve_gpu_type("KPM_Tn_gpu", type, dtype, cutoff)

    if isnothing(scale)
        scale, center = _estimate_spectral_bounds(H_mpo, sites;
                             dmrg_nsweeps = dmrg_nsweeps,
                             dmrg_maxdim  = dmrg_maxdim,
                             dmrg_linkdim = dmrg_linkdim)
    end

    I_mpo = MPO(sites, "Id")
    Ham_n = _scaled_hamiltonian(H_mpo, scale, center, I_mpo; cutoff = cutoff)

    I_mpo = _to_gpu_mpo(I_mpo, gpu_type)
    Ham_n = _to_gpu_mpo(Ham_n, gpu_type)

    keep = keep_indices
    T_k_minus_2 = I_mpo
    T_k_minus_1 = Ham_n
    Tn_list = Vector{Union{MPO,Nothing}}(undef, N + 1)
    Tn_list[1] = (keep === nothing || 1 ∈ keep) ? T_k_minus_2 : nothing
    Tn_list[2] = (keep === nothing || 2 ∈ keep) ? T_k_minus_1 : nothing

    for k in 3:N+1
        T_k = +(2 * apply(Ham_n, T_k_minus_1; cutoff = cutoff),
                -T_k_minus_2; maxdim = maxdim)
        T_k = ITensorMPS.truncate!(T_k; cutoff = cutoff)
        Tn_list[k] = (keep === nothing || k ∈ keep) ? T_k : nothing
        T_k_minus_2 = T_k_minus_1
        T_k_minus_1 = T_k
        _gpu_gc!()
        if verbose && (k % 5 == 0 || k == N+1)
            println("  [gpu] T_$((k-1)) maxlinkdim=$(ITensorMPS.maxlinkdim(T_k))")
        end
    end

    return Tn_list, scale, center
end


# ============================================================
# 2. Spatial LDOS from the MPO recurrence
# ============================================================

"""
    get_ldos_spatial_gpu(H, Ncheb, ω_phys_vals;
                         x_groups=nothing, num_x=H.N, num_y=nothing, num_avg=1,
                         x_start=1, x_end=H.N, grid=false, xwin=nothing, ywin=nothing,
                         box_half=0, reduce=:point, sublattice=:auto,
                         kernel=:jackson, lambda=4.0, maxdim=100, cutoff=1e-8,
                         verbose=false, printinfo=false,
                         nambu_proj=false, proj_nambu=nothing,
                         spin_proj=false, proj_s=nothing,
                         layer_proj=false, proj_layer=nothing,
                         sublat_proj=false, proj_sl=nothing,
                         type=ComplexF32, dtype=nothing)
        -> Matrix{Float64}   shape (Nω × n_spatial_cols)

GPU-accelerated version of `get_ldos_spatial` (MPO mode only).

**Sampling procedures (`reduce`)** — see [`spatial_sampling_plan`](@ref).

- `:point` (default) — read the LDOS at `num_x[×num_y]` cells / `x_groups`
  (optionally box-averaged). Coarse grids alias thin features.
- `:block` — integrate over `num_x × num_y` blocks (powers of two) by tracing out
  the within-block bits; gap-free, so thin in-gap edge channels on a large system
  cannot be missed. The scalable tool for large-scale edge-state maps.

**Column layout**

- No sublattice DOF: `(Nω × ng)`, one column per pixel (group or block).
- Sublattice resolved: `(Nω × ng×n_sub)`, interleaved `[A₀, B₀, A₁, B₁, …]`.
- Sublattice averaged (large scale / `:block`): `(Nω × ng)`, one value per pixel.

For `:block`, columns are row-major over coarse pixels (`col = ixp + iyp·num_x + 1`).

**GPU/CPU split**

GPU: entire Chebyshev MPO recurrence, aux projections, diagonal extraction,
     real-space scalar sampling (point eval or block integration).
CPU: KPM weight matrix, output accumulation (scalars only).

The keywords shared with `get_ldos_spatial` mean the same. There is no `mode`
(only the single-pass MPO mode is implemented here; the MPS path is
[`get_ldos_spatial_mps_gpu`](@ref)) and no `ordering`/`conumber_*` keywords
(binary position spaces only); `printinfo` prints progress like `verbose`.
`type` (alias `dtype`) is the GPU tensor datatype used consistently throughout
the MPO recurrence, projections, and diagonal extraction: `ComplexF32`
(default), `ComplexF64`, or `Float32`/`Float64` for a real `H`. A warning is
emitted for `ComplexF32` with `cutoff < 1e-6`.

Usage
-----
```julia
using CUDA
# point map
ldos = TensorBinding.get_ldos_spatial_gpu(H, 200, ωlist;
    x_groups = [[uc] for uc in 1:H.N], maxdim=200, printinfo=true)
# block-integrated large-scale edge-state map (num_x, num_y powers of two)
ldos = TensorBinding.get_ldos_spatial_gpu(H, 200, ωlist;
    reduce=:block, num_x=128, num_y=128, sublattice=:average, maxdim=200)
```
"""
function get_ldos_spatial_gpu(H::TBHamiltonian, Ncheb::Int, ω_phys_vals;
                               x_groups         = nothing,
                               num_x::Int        = H.N,
                               num_y             = nothing,
                               num_avg::Int      = 1,
                               x_start::Int      = 1,
                               x_end::Int        = H.N,
                               grid::Bool        = false,
                               xwin              = nothing,
                               ywin              = nothing,
                               box_half::Int     = 0,
                               reduce::Symbol    = :point,
                               sublattice::Symbol = :auto,
                               kernel::Symbol    = :jackson,
                               lambda::Real      = 4.0,
                               maxdim::Int       = 100,
                               cutoff::Real      = 1e-8,
                               verbose::Bool     = false,
                               printinfo::Bool   = false,
                               nambu_proj::Bool  = false,
                               proj_nambu        = nothing,
                               spin_proj::Bool   = false,
                               proj_s            = nothing,
                               layer_proj::Bool  = false,
                               proj_layer        = nothing,
                               sublat_proj::Bool = false,
                               proj_sl           = nothing,
                               type::Type{<:Number} = ComplexF32,
                               dtype::Union{Nothing,Type{<:Number}} = nothing)

    _require_binary_position_space(H, "get_ldos_spatial_gpu")

    _check_gpu("get_ldos_spatial_gpu")
    gpu_type = dtype === nothing ? type : dtype
    dtype !== nothing && dtype != type && type != ComplexF32 &&
        error("get_ldos_spatial_gpu: received both type=$type and dtype=$dtype; pass only one datatype keyword.")

    # ── Geometry-aware sampling plan (same convention as get_ldos_spatial) ────
    if box_half > 0 || grid || xwin !== nothing || ywin !== nothing || reduce === :block
        isnothing(H.geometry) &&
            error("get_ldos_spatial_gpu: box_half/grid/window/block sampling requires H.geometry to be set.")
        length(H.geometry(1)) == 2 ||
            error("get_ldos_spatial_gpu: box_half/grid/window/block sampling is only supported for 2D systems.")
    end
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
        x_groups = x_groups, box_half = box_half,
        sublattice = sublattice)
    groups   = plan.groups
    is_block = plan.reduce === :block
    block_a  = plan.a
    block_b  = plan.b
    nbx      = 2^block_a    # coarse pixels along x (block mode)

    _ensure_scale!(H)
    # ── Aux projection: flags auto-enabled, indices detected ─────────────────
    aux = _aux_projection(H; nambu_proj, proj_nambu, spin_proj, proj_s,
                             layer_proj, proj_layer, sublat_proj, proj_sl)

    has_sublat = !isnothing(aux.sublat.index)
    n_sub      = has_sublat ? dim(aux.sublat.index::Index) : 1
    # Large-scale sampling traces out the sublattice (one value per unit cell);
    # atomic-scale / proj_sl=k resolves it into per-atom columns. See plan above.
    resolve_sl = has_sublat && (plan.resolve_sublattice || !isnothing(proj_sl))
    average_sl = has_sublat && !resolve_sl

    # ── KPM setup ────────────────────────────────────────────────────────────
    ω_vals, W, denom, valid = _kpm_energy_grid(H, Ncheb, ω_phys_vals;
                                               kernel=kernel, lambda=lambda)
    Nω     = length(ω_vals)

    ng     = length(groups)
    n_cols = average_sl ? ng : ng * n_sub
    accum  = zeros(Float64, Nω, n_cols)

    # ── GPU operators ────────────────────────────────────────────────────────
    I_mpo_cpu = physical_projector(H)
    Ham_n_cpu = _scaled_hamiltonian(H; cutoff=cutoff, identity=I_mpo_cpu)
    I_mpo_gpu = _to_gpu_mpo(I_mpo_cpu, gpu_type)
    Ham_n_gpu = _to_gpu_mpo(Ham_n_cpu, gpu_type)

    # The spin Index to project; sites[1] for a Hamiltonian without spin.
    spin_idx = isnothing(aux.spin.index) ? H.sites[1] : aux.spin.index

    # ── Online accumulation (GPU) ────────────────────────────────────────────
    # No QFT sandwich: positions are real-space, so after projections we extract
    # the diagonal MPS and reduce it to per-pixel scalars:
    #   reduce=:point  → evaluate at each group's cells (big-endian) and average,
    #   reduce=:block  → integrate over each coarse block by tracing the within-
    #                    block bits (_eval_block_mps_gpu). Both return (u, value)
    #                    where u is the 1-indexed output pixel (column unit).
    function spatial_vals_gpu(diag_mps)
        if is_block
            return [(ixp + iyp * nbx + 1,
                     _eval_block_mps_gpu(diag_mps, ixp, iyp, block_a, block_b, Lx_uc, Ly_uc))
                    for iyp in 0:(2^block_b - 1) for ixp in 0:(nbx - 1)]
        else
            return [(ig, sum(_eval_mps_bigendian_gpu(diag_mps, x - 1) for x in grp) / length(grp))
                    for (ig, grp) in enumerate(groups)]
        end
    end

    # Project T_n onto every aux sector (nambu → spin → layer → sublattice) and
    # accumulate each projection's diagonal. Resolved sublattice → per-atom
    # column; averaged → fold all atoms into the single per-pixel column u (mean
    # over the n_sub atoms); no sublattice (n_sub = s = 1, scale 1.0) → pixel u.
    function accumulate_Tn_ldos_gpu!(ak_accum, Tn_gpu, n)
        for (Tp, s) in _project_aux_sectors(Tn_gpu, aux; project=_project_aux_gpu,
                                            spin_index=spin_idx, sublattice=has_sublat)
            diag_mps = ITensorMPS.truncate!(extract_diagonal_to_mps_gpu(Tp); cutoff=cutoff)
            scale    = average_sl ? 1.0 / n_sub : 1.0
            for (u, val) in spatial_vals_gpu(diag_mps)
                c = average_sl ? u : (u - 1) * n_sub + s
                for iω in 1:Nω
                    valid[iω] || continue
                    ak_accum[iω, c] += W[n, iω] * val * scale
                end
            end
        end

        _gpu_gc!()
    end

    # ── Chebyshev recurrence (GPU) ───────────────────────────────────────────
    gpu_type == ComplexF32 && cutoff < 1e-6 &&
        @warn "get_ldos_spatial_gpu: cutoff=$cutoff with ComplexF32 may produce NaN on large systems; use type=ComplexF64 or cutoff ≥ 1e-4."
    gpu_cutoff = Float64(cutoff)
    two = gpu_type(2)
    negone = gpu_type(-1)
    Tkm2 = I_mpo_gpu
    Tkm1 = Ham_n_gpu

    (verbose || printinfo) &&
        println("  [gpu] ldos dtype=$gpu_type  eltype(H)=$(eltype(Ham_n_gpu[1]))")

    accumulate_Tn_ldos_gpu!(accum, Tkm2, 1)
    accumulate_Tn_ldos_gpu!(accum, Tkm1, 2)

    for k in 3:Ncheb
        Tk = +(two * apply(Ham_n_gpu, Tkm1; cutoff=gpu_cutoff, maxdim=maxdim),
               negone * Tkm2; cutoff=gpu_cutoff, maxdim=maxdim)
        ITensorMPS.truncate!(Tk; cutoff=gpu_cutoff)
        accumulate_Tn_ldos_gpu!(accum, Tk, k)
        Tkm2 = Tkm1
        Tkm1 = Tk
        _gpu_gc!()
        (verbose || printinfo) && (k % 10 == 0 || k == Ncheb) &&
            println("  [gpu] ldos step $k/$Ncheb  maxlinkdim=$(maxlinkdim(Tkm1))")
    end

    # ── KPM normalization ────────────────────────────────────────────────────
    result = zeros(Float64, Nω, n_cols)
    for iω in 1:Nω
        valid[iω] || continue
        result[iω, :] = accum[iω, :] ./ denom[iω]
    end

    return result
end


# ============================================================
# 3. Spatial LDOS from independent MPS recursions
# ============================================================

"""
    get_ldos_spatial_mps_gpu(H, Ncheb, ω_phys_vals;
                             x_groups=nothing,
                             num_x=min(H.N, 100), num_avg=1,
                             x_start=1, x_end=H.N,
                             kernel=:jackson, lambda=4.0, eta=0.0, m_order=4,
                             maxdim=100, cutoff=1e-8,
                             type=ComplexF32, dtype=nothing,
                             verbose=false, printinfo=false,
                             return_maxlinkdim=false,
                             return_moments=false,
                             # accepted only to reject a non-default value:
                             num_y=nothing, grid=false, xwin=nothing, ywin=nothing,
                             box_half=0, reduce=:point, ordering=:physical,
                             sublattice=:auto,
                             nambu_proj=false, proj_nambu=nothing,
                             spin_proj=false, proj_s=nothing,
                             layer_proj=false, proj_layer=nothing,
                             sublat_proj=false, proj_sl=nothing)
        -> Matrix{Float64}

GPU spatial LDOS from one independent MPS Chebyshev recursion per physical-site
probe. Unlike [`get_ldos_spatial_gpu`](@ref), this path does not construct an MPO
Chebyshev series and supports projected position spaces such as
`FibonacciPositionSpace`.

The rescaled operator is `H̃ = (H - H.center * P) / H.scale`, where
`P = physical_projector(H)`. Probe `x` is constructed with
`physical_site_state(H, x)`, so `x` is always a 1-based *physical* site rather
than an ambient tensor-register index.

`x_groups` can be a vector of positions (one output column per position) or a
vector of position vectors. In the latter case, all probe LDOS values in a group
are averaged into one output column. Without explicit groups, `num_x` intervals
over `x_start:x_end` are sampled with `num_avg` approximately equidistant probes
per interval. Automatic planning allocates only `O(num_x * num_avg)` probe
indices, so callers can sample a huge projected space without enumerating it by
choosing a modest `num_x` (or by supplying `x_groups`).
The default is at most 100 output columns. The groups come from
[`interval_sampling_plan`](@ref).

`kernel=:hodc` uses HODC reconstruction (`eta`, `m_order`; `eta=0` uses
`1/(Ncheb+1)`). Other supported kernels are `:jackson`, `:lorentz` (`lambda`),
`:fejer`, and `:dirichlet`.

`type` (alias `dtype`) is the GPU tensor type: `ComplexF32` (default),
`ComplexF64`, or `Float32`/`Float64` for a real `H`. With
`return_moments=true`, the group-averaged raw Chebyshev moments are also
returned as a `Matrix{Float64}` of size `(Ncheb, length(x_groups))` (or
`(Ncheb, num_x)` for automatic groups):

`moments[n, j] = mean(x -> real(<x|T_(n-1)(Htilde)|x>), group[j])`,

where `Htilde = (H - H.center * P) / H.scale`. These moments contain no kernel
weights or energy-dependent normalization, and can therefore be reconstructed
later on a different energy grid or with a different KPM kernel.

With `return_maxlinkdim=true`, `linkdims[j]` is the largest MPS bond dimension
reached by any probe in group `j`. Return values are unambiguous for all keyword
combinations:

- neither keyword: `ldos`
- `return_maxlinkdim=true`: `(ldos, linkdims)` (the existing API)
- `return_moments=true`: `(ldos, moments)`
- both keywords: `(ldos, moments, linkdims)`

This entry point intentionally supports position-only, one-dimensional point or
explicit-group sampling. Grid/window/box/block sampling, non-physical ordering,
and auxiliary degrees of freedom (on `H` or requested through the projection
keywords) are rejected with targeted errors. For those features use the MPO GPU
path or the CPU `get_ldos_spatial` implementation.
"""
function get_ldos_spatial_mps_gpu(H::TBHamiltonian, Ncheb::Int, ω_phys_vals;
                                   x_groups         = nothing,
                                   num_x::Int        = min(H.N, 100),
                                   num_avg::Int      = 1,
                                   x_start::Int      = 1,
                                   x_end::Int        = H.N,
                                   kernel::Symbol    = :jackson,
                                   lambda::Real      = 4.0,
                                   eta::Real         = 0.0,
                                   m_order::Int      = 4,
                                   maxdim::Int       = 100,
                                   cutoff::Real      = 1e-8,
                                   type::Type{<:Number} = ComplexF32,
                                   dtype::Union{Nothing,Type{<:Number}} = nothing,
                                   verbose::Bool     = false,
                                   printinfo::Bool   = false,
                                   return_maxlinkdim::Bool = false,
                                   return_moments::Bool = false,
                                   # Accepted only to provide clear compatibility errors.
                                   num_y             = nothing,
                                   grid::Bool        = false,
                                   xwin              = nothing,
                                   ywin              = nothing,
                                   box_half::Int     = 0,
                                   reduce::Symbol    = :point,
                                   ordering::Symbol  = :physical,
                                   sublattice::Symbol = :auto,
                                   nambu_proj::Bool  = false,
                                   proj_nambu        = nothing,
                                   spin_proj::Bool   = false,
                                   proj_s            = nothing,
                                   layer_proj::Bool  = false,
                                   proj_layer        = nothing,
                                   sublat_proj::Bool = false,
                                   proj_sl           = nothing)

    Ncheb >= 2 || throw(ArgumentError(
        "get_ldos_spatial_mps_gpu: Ncheb must be at least 2."
    ))
    reduce === :point || throw(ArgumentError(
        "get_ldos_spatial_mps_gpu: only reduce=:point is supported; " *
        "block reduction belongs to the MPO GPU path."
    ))
    if grid || num_y !== nothing || xwin !== nothing || ywin !== nothing || box_half != 0
        throw(ArgumentError(
            "get_ldos_spatial_mps_gpu: grid, num_y, windows, and box averaging " *
            "are unsupported. Supply 1-based physical positions through x_groups."
        ))
    end
    ordering === :physical || throw(ArgumentError(
        "get_ldos_spatial_mps_gpu: only ordering=:physical is supported. " *
        "Map alternate coordinates to physical sites before passing x_groups."
    ))
    sublattice === :auto || throw(ArgumentError(
        "get_ldos_spatial_mps_gpu: sublattice resolution/averaging is unsupported."
    ))

    aux_requested = nambu_proj || spin_proj || layer_proj || sublat_proj ||
                    proj_nambu !== nothing || proj_s !== nothing ||
                    proj_layer !== nothing || proj_sl !== nothing
    has_aux = !isnothing(H.nambu_s) || !isnothing(H.spin_s) ||
              !isnothing(H.layer_s) || !isnothing(H.sublattice_s) ||
              length(H.sites) != H.L
    (aux_requested || has_aux) && throw(ArgumentError(
        "get_ldos_spatial_mps_gpu: only position-only Hamiltonians are supported; " *
        "auxiliary degrees of freedom and auxiliary projections are not available " *
        "on this MPS GPU path."
    ))

    groups = interval_sampling_plan(H.N; x_groups, num_x, num_avg, x_start, x_end,
                                    caller="get_ldos_spatial_mps_gpu")

    _check_gpu("get_ldos_spatial_mps_gpu")
    gpu_type = _resolve_gpu_type(
        "get_ldos_spatial_mps_gpu", type, dtype, cutoff,
    )
    _ensure_scale!(H)

    # P (the default identity of _scaled_hamiltonian), rather than the ambient
    # identity, is essential for projected position spaces: invalid register
    # states must remain zero under the spectral shift.
    Ham_n_cpu = _scaled_hamiltonian(H; cutoff=Float64(cutoff))
    Ham_n_gpu = _to_gpu_mpo(Ham_n_cpu, gpu_type)

    ω_vals, W, denom, valid = _kpm_energy_grid(
        H, Ncheb, ω_phys_vals; kernel=kernel, lambda=lambda, eta=eta, m_order=m_order,
        allow_hodc=true,
    )
    Nω = length(ω_vals)
    # Store kernel-independent, group-averaged moments. Besides making them
    # available for offline reconstruction, this avoids applying all Nω
    # energy weights separately for every probe in an averaged group.
    moments = zeros(Float64, Ncheb, length(groups))
    linkdims = zeros(Int, length(groups))

    apply_kwargs = (cutoff=Float64(cutoff), maxdim=maxdim)
    two = gpu_type(2)
    negone = gpu_type(-1)
    printinfo && println(
        "  [gpu] spatial MPS LDOS dtype=$gpu_type, groups=$(length(groups)), " *
        "projected=$( !(H.position_space isa BinaryPositionSpace) )",
    )

    for (j, group) in enumerate(groups)
        group_moments = view(moments, :, j)
        group_weight = inv(Float64(length(group)))
        group_maxlinkdim = 0

        for x in group
            psi0_gpu = _to_gpu_mps(physical_site_state(H, x), gpu_type)

            function kpm_step!(phi, n)
                mu = Float64(real(inner(psi0_gpu, phi)))
                group_moments[n] += group_weight * mu
            end

            phi_km2 = psi0_gpu
            kpm_step!(phi_km2, 1)
            group_maxlinkdim = max(group_maxlinkdim, maxlinkdim(phi_km2))

            phi_km1 = apply(Ham_n_gpu, phi_km2; apply_kwargs...)
            kpm_step!(phi_km1, 2)
            group_maxlinkdim = max(group_maxlinkdim, maxlinkdim(phi_km1))

            for k in 3:Ncheb
                phi_k = +(
                    two * apply(Ham_n_gpu, phi_km1; apply_kwargs...),
                    negone * phi_km2;
                    apply_kwargs...,
                )
                kpm_step!(phi_k, k)
                group_maxlinkdim = max(group_maxlinkdim, maxlinkdim(phi_k))
                phi_km2 = phi_km1
                phi_km1 = phi_k
            end

            _gpu_gc!()
        end

        linkdims[j] = group_maxlinkdim
        (verbose || printinfo) && (j % 5 == 0 || j == length(groups)) &&
            println(
                "  [gpu] spatial MPS LDOS $j/$(length(groups)) " *
                "(x=$(first(group)), n_avg=$(length(group))) " *
                "maxlinkdim=$group_maxlinkdim",
            )
    end

    result = _reconstruct_ldos_moment_columns(moments, W, denom, valid)

    if return_moments
        return return_maxlinkdim ? (result, moments, linkdims) : (result, moments)
    end
    return return_maxlinkdim ? (result, linkdims) : result
end


# ============================================================
# 4. Stochastic DOS
# ============================================================

"""
    get_dos_stochastic_gpu(H, Ncheb, ω_phys_vals;
                           N_sample=50, N_bound=0, seed=42, normalize=false,
                           dos_weighting=:trace, kernel=:jackson, lambda=4.0,
                           eta=0.0, m_order=4, maxdim=100, cutoff=1e-8,
                           verbose=false, printinfo=false, continuum_only=false,
                           nambu_proj=false, proj_nambu=nothing,
                           spin_proj=false, proj_s=nothing,
                           layer_proj=false, proj_layer=nothing,
                           sublat_proj=false, proj_sl=nothing,
                           type=ComplexF32, dtype=nothing)
        -> Vector{Float64}   length Nω

GPU-accelerated stochastic density of states via MPS Chebyshev KPM.

For each random sample the scaled Hamiltonian MPO lives on GPU and the product-
state MPS is transferred to GPU once before the recursion starts.
The Chebyshev moments ⟨ψ₀|T_n(H̃)|ψ₀⟩ are scalars pulled to CPU at each step.

The keywords shared with `get_dos_stochastic` (CPU) mean the same;
`continuum_only`, `printinfo` and `type`/`dtype` are GPU-only, and only binary
position spaces are supported. `type` (alias `dtype`) is the GPU tensor type:
`ComplexF32` (default), `ComplexF64`, or `Float32`/`Float64` for a real `H`.
`N_bound` (exciton bound-sector enrichment) is supported. Use
`dos_weighting=:sample` to return the unweighted sampled signal
`avg_full + avg_bound`, which is useful when visualising exciton peaks that are
otherwise hidden by continuum phase-space factors in the trace DOS.
For exciton Hamiltonians, `continuum_only=true` samples ordered electron-hole
product states with `x_e != x_h` for the `N_sample` branch.

`kernel=:hodc` selects the Higher-Order Delta Chebyshev reconstruction
(`eta`, `m_order` control the contour); its weights already carry the full KPM
normalisation, so no `√(1−ω²)` denominator is applied.  `eta=0` falls back to
`1/(Ncheb+1)`.  Otherwise `kernel` is a convolution kernel (`:jackson` default,
`:lorentz` with `lambda`, `:fejer`, `:dirichlet`).
"""
function get_dos_stochastic_gpu(H::TBHamiltonian, Ncheb::Int, ω_phys_vals;
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
                                  printinfo::Bool          = false,
                                  continuum_only::Bool     = false,
                                  nambu_proj::Bool         = false,
                                  proj_nambu               = nothing,
                                  spin_proj::Bool          = false,
                                  proj_s                   = nothing,
                                  layer_proj::Bool         = false,
                                  proj_layer               = nothing,
                                  sublat_proj::Bool        = false,
                                  proj_sl                  = nothing,
                                  type::Type{<:Number}     = ComplexF32,
                                  dtype::Union{Nothing,Type{<:Number}} = nothing)

    _require_binary_position_space(H, "get_dos_stochastic_gpu")
    _check_gpu("get_dos_stochastic_gpu")
    gpu_type = _resolve_gpu_type("get_dos_stochastic_gpu", type, dtype, cutoff)
    _ensure_scale!(H)
    dos_weighting in (:trace, :sample) ||
        error("get_dos_stochastic_gpu: dos_weighting must be :trace or :sample.")
    N_sample >= 0 || error("get_dos_stochastic_gpu: N_sample must be non-negative.")
    N_bound >= 0 || error("get_dos_stochastic_gpu: N_bound must be non-negative.")

    Ham_n_cpu = _scaled_hamiltonian(H; cutoff=cutoff)
    Ham_n_gpu = _to_gpu_mpo(Ham_n_cpu, gpu_type)

    D      = prod(ITensors.dim(s) for s in H.sites)
    N_phys = H.N
    is_exc = length(H.sites) == 2 * H.L
    continuum_only && !is_exc &&
        error("get_dos_stochastic_gpu: continuum_only=true requires an exciton Hamiltonian.")
    continuum_only && N_phys < 2 &&
        error("get_dos_stochastic_gpu: continuum_only=true requires H.N >= 2.")

    # Projections are not switched on automatically (as in get_dos_stochastic).
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

    function _run_kpm_mps_gpu!(psi0_gpu, accum, weight)
        apply_kwargs = (cutoff=Float64(cutoff), maxdim=maxdim)
        function kpm_step!(phi, n)
            mu = Float64(real(inner(psi0_gpu, phi)))
            for iω in 1:Nω
                valid[iω] || continue
                accum[iω] += W[n, iω] * mu * weight
            end
        end
        phi_km2 = psi0_gpu
        phi_km1 = apply(Ham_n_gpu, phi_km2; apply_kwargs...)
        kpm_step!(phi_km2, 1)
        kpm_step!(phi_km1, 2)
        for k in 3:Ncheb
            phi_k = +(2 * apply(Ham_n_gpu, phi_km1; apply_kwargs...),
                      -phi_km2; apply_kwargs...)
            kpm_step!(phi_k, k)
            phi_km2 = phi_km1
            phi_km1 = phi_k
        end
        _gpu_gc!()
        return maxlinkdim(phi_km1)
    end

    function _exciton_pair_mps_gpu_seed(xe::Int, xh::Int)
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

    if _any_projected(aux)
        continuum_only &&
            error("get_dos_stochastic_gpu: continuum_only is not supported together with auxiliary projections.")
        D_eff = N_phys
        if N_sample > 0
            sectors = _probe_sectors(aux)
            xs = rand(rng, 1:N_phys, N_sample)
            for (i, x) in enumerate(xs)
                for σ in sectors
                    psi0_gpu = _to_gpu_mps(probe_state(H, x, σ), gpu_type)
                    χ = _run_kpm_mps_gpu!(psi0_gpu, accum_full, 1.0/N_sample)
                    (verbose || printinfo) && i % 10 == 0 && σ == first(sectors) &&
                        println("  [gpu] dos sample $i/$N_sample (projected)  maxlinkdim=$χ")
                end
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

    # ── Full / continuum Hilbert-space sampling ───────────────────────────────
    if N_sample > 0
        if continuum_only
            xs_e = rand(rng, 1:N_phys, N_sample)
            ys_h = rand(rng, 1:(N_phys - 1), N_sample)
            for i in 1:N_sample
                xe = xs_e[i]
                xh = ys_h[i] < xe ? ys_h[i] : ys_h[i] + 1
                psi0_gpu = _to_gpu_mps(_exciton_pair_mps_gpu_seed(xe, xh), gpu_type)
                χ = _run_kpm_mps_gpu!(psi0_gpu, accum_full, 1.0/N_sample)
                (verbose || printinfo) && i % 10 == 0 &&
                    println("  [gpu] dos continuum sample $i/$N_sample (xe=$xe, xh=$xh)  maxlinkdim=$χ")
            end
        else
            samples = rand(rng, 0:(D - 1), N_sample)
            for (i, k) in enumerate(samples)
                psi0_gpu = _to_gpu_mps(_basis_state_mps(k, H.sites), gpu_type)
                χ = _run_kpm_mps_gpu!(psi0_gpu, accum_full, 1.0/N_sample)
                (verbose || printinfo) && i % 10 == 0 &&
                    println("  [gpu] dos sample $i/$N_sample  maxlinkdim=$χ")
            end
        end
    end

    # ── Bound-sector enrichment (exciton) ─────────────────────────────────────
    if N_bound > 0 && is_exc
        xs = rand(rng, 1:N_phys, N_bound)
        for (i, x) in enumerate(xs)
            psi0_gpu = _to_gpu_mps(mpsexciton(x, H.sites), gpu_type)
            χ = _run_kpm_mps_gpu!(psi0_gpu, accum_bound, 1.0/N_bound)
            (verbose || printinfo) && i % 10 == 0 &&
                println("  [gpu] dos bound sample $i/$N_bound (x=$x)  maxlinkdim=$χ")
        end
    end

    # ── Normalise ─────────────────────────────────────────────────────────────
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
