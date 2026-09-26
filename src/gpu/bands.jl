# gpu/bands.jl — the GPU band structure A(k, ω): get_bands_gpu, the GPU mirror of
# the TBHamiltonian overload of get_bands (physics/qft/bands.jl). One entry point,
# so the file has no sections. Moved from the former gpu/GPU_tk.jl.
#
# Main entry point: get_bands_gpu.
# Depends on: core/Utils.jl (fix_sites, kspace_sampling_plan), core/TBSystem.jl,
# core/AuxDOF.jl (_aux_projection, _project_aux_sectors), lattice/masks2d.jl (legacy sublattice
# masks), solvers/DMRG.jl (_ensure_scale!), solvers/kpm/kernels.jl
# (_kpm_energy_grid), solvers/kpm/recursion.jl (_scaled_hamiltonian,
# chebyshev_foreach), physics/qft/kpath.jl (kpath_setup), gpu/device.jl,
# gpu/primitives.jl.

"""
    get_bands_gpu(H, Ncheb, ω_phys_vals;
                  kpath=nothing, kpath_lattice=nothing, kpath_Lx=nothing,
                  spin_proj=false, proj_s=nothing, nambu_proj=false, proj_nambu=nothing,
                  layer_proj=false, proj_layer=nothing, sublattice=false, proj_sl=nothing,
                  sublat_proj=false, k_groups_override=nothing,
                  xmin=0, xmax=nothing, num_x=60, num_avg=1,
                  ymin=0, ymax=nothing, num_y=10,
                  kernel=:jackson, lambda=4.0, tol=1e-9, maxdim=100, cutoff=1e-10,
                  printinfo=false, type=ComplexF32, dtype=nothing)
        -> Matrix{Float64}  or  NamedTuple(Ak, ticks, labels)

GPU-accelerated version of `get_bands`.

GPU handles: the full Chebyshev MPO recurrence (the dominant cost) and the
             QFT sandwich applied to each Chebyshev moment.
CPU handles: k-group setup, KPM weight matrix, final scalar accumulation.

Use `type=ComplexF32` (default) or `type=ComplexF64` to choose the GPU tensor
datatype. Real types are rejected, because the quantics Fourier transform is complex.
`dtype=...` is accepted as an alias for consistency with the non-Hermitian GPU
entry points. ComplexF32 is faster, while ComplexF64 is safer at tight cutoffs
on large systems (a warning is emitted for ComplexF32 with `cutoff < 1e-6`).

The other keyword arguments are identical to the TBHamiltonian overload of
`get_bands`; as in `get_bands(H, Ncheb, ω_phys_vals)`, the dimension D is taken
from `H.geometry`, which must be set.  The return value is also identical: a
plain `Matrix{Float64}` when no `kpath` is given, or a
`NamedTuple(Ak, ticks, labels)` when a high-symmetry path is requested.

Usage:
```julia
using CUDA
res = TensorBinding.get_bands_gpu(H, 500, omega;
        kpath=[:G, :M, :Kp, :G], kpath_lattice=:honeycomb,
        num_x=50, maxdim=200, type=ComplexF64, printinfo=true)
heatmap(1:size(res.Ak,2), omega, res.Ak; xticks=(res.ticks, res.labels))
```
"""
function get_bands_gpu(H::TBHamiltonian, Ncheb::Int, ω_phys_vals;
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
                       k_groups_override = nothing,
                       xmin::Int         = 0,
                       xmax              = nothing,
                       num_x::Int        = 60,
                       num_avg::Int      = 1,
                       ymin::Int         = 0,
                       ymax              = nothing,
                       num_y::Int        = 10,
                       kernel::Symbol    = :jackson,
                       lambda::Real      = 4.0,
                       tol::Real         = 1e-9,
                       maxdim::Int       = 100,
                       cutoff::Real      = 1e-10,
                       printinfo::Bool   = false,
                       type::Type{<:Number} = ComplexF32,
                       dtype::Union{Nothing,Type{<:Number}} = nothing)

    _require_binary_position_space(H, "get_bands_gpu")
    _check_gpu("get_bands_gpu")
    gpu_type = dtype === nothing ? type : dtype
    dtype !== nothing && dtype != type && type != ComplexF32 &&
        error("get_bands_gpu: received both type=$type and dtype=$dtype; pass only one datatype keyword.")
    gpu_type <: Complex || throw(ArgumentError(
        "get_bands_gpu: the quantics Fourier transform is complex; " *
        "use type=ComplexF32 or type=ComplexF64 (got $gpu_type)."))
    gpu_type == ComplexF32 && cutoff < 1e-6 &&
        @warn "get_bands_gpu: cutoff=$cutoff with ComplexF32 may produce NaN on large systems; use type=ComplexF64 or cutoff ≥ 1e-4."

    _ensure_scale!(H)

    ω_resc, W_kpm, denom, valid = _kpm_energy_grid(H, Ncheb, ω_phys_vals;
                                                   kernel=kernel, lambda=lambda)
    Nω     = length(ω_resc)

    # ── Aux projection: flags auto-enabled, indices detected (as on the CPU) ─
    aux = _aux_projection(H; nambu_proj, proj_nambu, spin_proj, proj_s,
                             layer_proj, proj_layer, sublat_proj, proj_sl)
    # The spin Index to project; sites[1] for a Hamiltonian without spin.
    spin_idx = isnothing(aux.spin.index) ? H.sites[1] : aux.spin.index

    # ── L_pos: position qubits only (excluding aux sites) ───────────────────
    # Count from the full site list, as the CPU get_bands does: H.L already
    # excludes the aux indices, so subtracting them from it would drop one too many.
    isnothing(H.geometry) && error("get_bands_gpu: H.geometry must be set (needed to infer D).")
    D     = length(H.geometry(1))
    L     = length(H.sites)
    L_pos = L - (aux.spin.on ? 1 : 0) - (!isnothing(aux.nambu.index) ? 1 : 0) -
                (!isnothing(aux.layer.index) ? 1 : 0) - (!isnothing(aux.sublat.index) ? 1 : 0)

    # ── k-path shortcut ──────────────────────────────────────────────────────
    kpath_ticks = nothing; kpath_labels = nothing
    if !isnothing(kpath)
        isnothing(kpath_lattice) && error("get_bands_gpu: kpath requires kpath_lattice.")
        Lx_kp = isnothing(kpath_Lx) ? H.L ÷ 2 : Int(kpath_Lx)
        Ly_kp = H.L - Lx_kp
        k_groups_override, kpath_ticks, kpath_labels =
            kpath_setup(kpath_lattice, Lx_kp, Ly_kp, kpath; npts_per_segment=num_x)
    end

    # ── k-groups (shared planner in core/Utils.jl, same as CPU get_bands) ────
    Lx_pos   = D == 2 ? div(L_pos, 2) : 0
    kplan    = kspace_sampling_plan(L_pos, D; num_x, num_y, num_avg,
                                    xmin, xmax, ymin, ymax, k_groups_override)
    k_groups = kplan.k_groups
    num_x    = kplan.num_x

    Ak_w = zeros(Float64, Nω, num_x)

    # ── Position sites (used for QFT ops and optional sublattice masks) ──────
    aux_to_drop = Set{Index}()
    aux.spin.on && push!(aux_to_drop, spin_idx::Index)
    for d in (aux.nambu, aux.layer, aux.sublat)
        !isnothing(d.index) && push!(aux_to_drop, d.index::Index)
    end
    pos_sites_cpu = filter(s -> s ∉ aux_to_drop, H.sites)

    # ── Legacy sublattice masks — pre-built on CPU, moved to GPU once ────────
    # Built only when `sublattice=true` (legacy models without a sublat aux index).
    # For models that use H.sublattice_s (honeycomb, kagome…), sublat_proj=true
    # and sublattice=false, so this block is skipped entirely.
    if sublattice
        if D == 1
            mask_A_gpu = _to_gpu_mpo(_col_select_mpo(L_pos, 0, pos_sites_cpu; keep=:odd), gpu_type)
            mask_B_gpu = _to_gpu_mpo(_col_select_mpo(L_pos, 0, pos_sites_cpu; keep=:even), gpu_type)
        else
            Ly_pos = L_pos - Lx_pos
            mask_A_gpu = _to_gpu_mpo(_row_checker_mpo(Lx_pos, Ly_pos, pos_sites_cpu), gpu_type)
            mask_B_gpu = _to_gpu_mpo(MPO(pos_sites_cpu, "Id") -
                                     _row_checker_mpo(Lx_pos, Ly_pos, pos_sites_cpu), gpu_type)
        end
    end

    # ── GPU Ham and QFT operators ────────────────────────────────────────────
    I_mpo_cpu = physical_projector(H)
    Ham_n_cpu = _scaled_hamiltonian(H; cutoff=cutoff, identity=I_mpo_cpu)
    I_mpo_gpu = _to_gpu_mpo(I_mpo_cpu, gpu_type)
    Ham_n_gpu = _to_gpu_mpo(Ham_n_cpu, gpu_type)

    # QFT operators sized for pos_sites_cpu (the post-projection site list).
    # Calling fix_sites maps the abstract QFT indices onto the actual pos_sites.
    R_pos      = length(pos_sites_cpu)
    FTirev_gpu = _to_gpu_mpo(fix_sites(
        MPO(TCI.reverse(QuanticsTCI.quanticsfouriermpo(R_pos; sign=-1.0, normalize=true))),
        pos_sites_cpu), gpu_type)
    FTrev_gpu  = _to_gpu_mpo(fix_sites(
        MPO(TCI.reverse(QuanticsTCI.quanticsfouriermpo(R_pos; sign=+1.0, normalize=true))),
        pos_sites_cpu), gpu_type)

    printinfo && println("  [gpu] bands dtype=$gpu_type  eltype(H)=$(eltype(Ham_n_gpu[1]))")

    # ── Online accumulation — fully on GPU ──────────────────────────────────
    # Workflow: prebuild everything on CPU (done above), then T_n stays on GPU
    # for the entire accumulate step.  Only the final scalar() calls transfer
    # numbers out of the GPU — no explicit MPO/MPS moves back to CPU.
    #
    # Per step:
    #   projection  → _project_aux_sectors with _project_aux_gpu (dense typed
    #                 projector, GPU throughout)
    #   QFT         → _apply_qft_conj_gpu  (pre-built GPU QFT operators)
    #   diagonal    → extract_diagonal_to_mps_gpu (same GPU dtype as input)
    #   sampling    → _eval_diag_mps_gpu  (scalars pulled out of GPU directly)
    function accumulate_Tn_gpu!(ak_accum, Tn_gpu, n)
        # Steps 0, 1, 1c, 1b: Nambu → spin → layer → sublattice projections
        after_sl_aux = _project_aux_sectors(Tn_gpu, aux; project=_project_aux_gpu,
                                            spin_index=spin_idx)

        # Step 2: legacy sublattice mask sandwich (all GPU — masks pre-built above)
        if sublattice
            masks   = isnothing(proj_sl) ? [mask_A_gpu, mask_B_gpu] :
                      proj_sl == 1       ? [mask_A_gpu]              : [mask_B_gpu]
            sl_mpas = MPO[]
            for (T, _) in after_sl_aux, mask in masks
                push!(sl_mpas, apply(apply(mask, T; cutoff=cutoff, maxdim=maxdim), mask;
                                     cutoff=cutoff, maxdim=maxdim))
            end
        else
            sl_mpas = (T for (T, _) in after_sl_aux)
        end

        # Step 3: QFT (GPU) → diagonal MPS (GPU) → scalar sampling (GPU)
        for T_gpu in sl_mpas
            Tn_k_gpu = _apply_qft_conj_gpu(T_gpu, FTirev_gpu, FTrev_gpu;
                                            tol=tol, maxdim=maxdim)
            A_mps_gpu = ITensorMPS.truncate!(extract_diagonal_to_mps_gpu(Tn_k_gpu); cutoff=cutoff)
            for (ik, xs) in enumerate(k_groups)
                s = sum(_eval_diag_mps_gpu(A_mps_gpu, x) for x in xs) / length(xs)
                for ie in 1:Nω
                    ak_accum[ie, ik] += W_kpm[n, ie] * s
                end
            end
        end

        _gpu_gc!()
    end

    # ── Chebyshev recurrence (GPU) ───────────────────────────────────────────
    two = gpu_type(2)
    negone = gpu_type(-1)

    chebyshev_foreach(Ham_n_gpu, I_mpo_gpu, Ncheb; T1=Ham_n_gpu, maxdim=maxdim,
                      cutoff=cutoff, post_trunc=(:cutoff,),
                      two=two, negone=negone) do n, Tn
        k = n + 1
        accumulate_Tn_gpu!(Ak_w, Tn, k)
        if k >= 3
            _gpu_gc!()
            printinfo && (k % 10 == 0 || k == Ncheb) &&
                println("  [gpu] bands step $k/$Ncheb  maxlinkdim=$(maxlinkdim(Tn))")
        end
    end

    # ── KPM normalization: 1 / (π² Ncheb √(1 − ε²)) ────────────────────────
    for iω in 1:Nω
        valid[iω] || continue
        Ak_w[iω, :] ./= denom[iω]
    end

    return isnothing(kpath_ticks) ? Ak_w :
           (Ak = Ak_w, ticks = kpath_ticks, labels = kpath_labels)
end
