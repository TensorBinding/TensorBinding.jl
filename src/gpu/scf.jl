# gpu/scf.jl — GPU collinear magnetic Hubbard SCF: the RMS and Hartree helpers, the
# loop scf_magnetic_hubbard_gpu and its post-convergence observables
# (get_scf_magnetization_gpu with its planner _tb_spatial_plan_gpu, and
# get_scf_bands_gpu). Moved from the former gpu/GPU_tk.jl.
#
# Main entry points: scf_magnetic_hubbard_gpu, get_scf_magnetization_gpu,
# get_scf_bands_gpu.
# Depends on: core/Utils.jl (spatial_sampling_plan, constant_mps), core/TBSystem.jl,
# physics/SCF.jl (spin-channel split, staggered initial densities, _copy_with_mpo),
# gpu/device.jl, gpu/primitives.jl, gpu/bands.jl (get_bands_gpu),
# gpu/purification.jl (_mcweeny_purify_mpo_gpu).


# ============================================================
# 1. RMS and Hartree helpers
# ============================================================

function _rms_error_gpu(a::MPS, b::MPS; cutoff::Real = 1e-12)
    diff = +(a, -1.0 * b; cutoff=Float64(cutoff))
    n = prod(dim(s) for s in siteinds(a))
    return sqrt(abs(real(inner(diff', diff))) / n)
end

function _local_hartree_from_density_gpu(rho::MPS, sites, U::Number, bg::MPS;
                                         maxdim::Int, cutoff::Real)
    coeff = +(rho, -1.0 * bg; maxdim=maxdim, cutoff=Float64(cutoff))
    return _mps_to_diagonal_mpo_gpu(U * coeff, sites)
end

function _hartree_mpo_from_density_gpu(rho::MPS, interaction_op::MPO, sites, bg::MPS;
                                       maxdim::Int, cutoff::Real)
    coeff = +(rho, -1.0 * bg; maxdim=maxdim, cutoff=Float64(cutoff))
    coeff_mps = apply(interaction_op, coeff; maxdim=maxdim, cutoff=Float64(cutoff))
    return _mps_to_diagonal_mpo_gpu(coeff_mps, sites)
end


# ============================================================
# 2. SCF loop
# ============================================================

"""
    scf_magnetic_hubbard_gpu(H0, U; initial_up=nothing, initial_dn=nothing,
                             background=0.5, Nel_up=H0.N ÷ 2, Nel_dn=H0.N ÷ 2,
                             fermi=0.0, scale=(H0.scale == 0.0 ? nothing : H0.scale),
                             purification_scale_padding=1.05, max_scf_iter=30,
                             scf_tol=1e-6, mix=0.4, maxdim=100, cutoff=1e-8,
                             purif_maxiter=40, purif_tol=1e-6,
                             type=ComplexF32, dtype=nothing, verbose=true)
        -> NamedTuple

GPU-accelerated two-channel collinear magnetic mean-field loop for the on-site
Hubbard model. The SCF iteration keeps the density profiles, Hartree MPOs,
Hamiltonian MPOs, density matrices, RMS checks, and mixing on GPU; CPU objects
are built only for initialization and for the compatibility fields returned at
the end.

```text
H_up = H0_up + U·diag(n_dn − background)
H_dn = H0_dn + U·diag(n_up   − background)
```

`U` is a number (on-site) or an interaction MPO, which is then applied to
`n − background` before the diagonal is formed.

There is no `density_method` keyword: the densities always come from McWeeny
purification (grand-canonical at `fermi`); for particle-number-fixed SP2 use the
CPU `scf_magnetic_hubbard`. `Nel_up`/`Nel_dn` only enter the `particle_error`
recorded in `history`. A concrete purification `scale` is required so the GPU
initial guess can be formed without estimating spectral bounds on CPU during
the loop; it is multiplied by `purification_scale_padding`.

`type` (alias `dtype`; default `ComplexF32`) is the element type the
Hamiltonians, densities and background profile are uploaded with; `ComplexF64`
is safer at tight cutoffs. A real `type` is accepted for a real `H0`, but the
ComplexF32 deltas of the Hartree MPOs then promote the loop to the matching
complex type. ComplexF32 eigen-decompositions can NaN at very tight
cutoffs: a warning is emitted for a 32-bit `type` with `cutoff < 1e-6`; the
requested `cutoff` is used as-is.

The result carries `converged`, `iterations`, `rms_error`, `history`, the CPU
fields `rho_up`, `rho_dn`, `density_up_mpo`, `density_dn_mpo`, `H_up`, `H_dn`,
and their GPU counterparts `rho_up_gpu`, `rho_dn_gpu`, `density_up_mpo_gpu`,
`density_dn_mpo_gpu`, `H_up_mpo_gpu`, `H_dn_mpo_gpu`.

Post-convergence observables are intentionally separate. Use
[`get_scf_magnetization_gpu`](@ref) or [`get_scf_bands_gpu`](@ref) on the
returned result when you want those GPU-accelerated diagnostics.
"""
function scf_magnetic_hubbard_gpu(H0::TBHamiltonian, U::Union{Number, MPO};
                                  initial_up::Union{Nothing,MPS}=nothing,
                                  initial_dn::Union{Nothing,MPS}=nothing,
                                  background::Real = 0.5,
                                  Nel_up::Int = H0.N ÷ 2,
                                  Nel_dn::Int = H0.N ÷ 2,
                                  fermi::Real = 0.0,
                                  scale::Union{Nothing,Real} = H0.scale == 0.0 ? nothing : H0.scale,
                                  purification_scale_padding::Real = 1.05,
                                  max_scf_iter::Int = 30,
                                  scf_tol::Real = 1e-6,
                                  mix::Real = 0.4,
                                  maxdim::Int = 100,
                                  cutoff::Real = 1e-8,
                                  purif_maxiter::Int = 40,
                                  purif_tol::Real = 1e-6,
                                  type::Type{<:Number} = ComplexF32,
                                  dtype::Union{Nothing,Type{<:Number}} = nothing,
                                  verbose::Bool = true)
    _check_gpu("scf_magnetic_hubbard_gpu")
    gpu_type = _resolve_gpu_type("scf_magnetic_hubbard_gpu", type, dtype, cutoff)

    H0_up, H0_dn = _split_spin_channels(H0)
    sites = H0_up.sites
    scale === nothing &&
        error("scf_magnetic_hubbard_gpu: pass a concrete nonzero scale to keep the SCF loop GPU-resident.")
    scale_eff = Float64(scale) * Float64(purification_scale_padding)
    scale_eff == 0.0 &&
        error("scf_magnetic_hubbard_gpu: scale must be nonzero.")
    if initial_up === nothing || initial_dn === nothing
        rho_up, rho_dn = staggered_magnetic_initial(H0; background=background)
        initial_up === nothing || (rho_up = initial_up)
        initial_dn === nothing || (rho_dn = initial_dn)
    else
        rho_up, rho_dn = initial_up, initial_dn
    end

    rho_up_gpu = _to_gpu(rho_up, gpu_type)
    rho_dn_gpu = _to_gpu(rho_dn, gpu_type)
    bg_gpu = _to_gpu(constant_mps(collect(sites), background), gpu_type)
    H0_up_gpu = _to_gpu(H0_up.mpo, gpu_type)
    H0_dn_gpu = _to_gpu(H0_dn.mpo, gpu_type)
    Id_gpu = _to_gpu(MPO(collect(sites), "Id"), gpu_type)
    U_gpu = U isa MPO ? _to_gpu(U, gpu_type) : nothing

    history = NamedTuple[]
    density_up_mpo_gpu = nothing
    density_dn_mpo_gpu = nothing
    Hup_mpo_gpu = H0_up_gpu
    Hdn_mpo_gpu = H0_dn_gpu
    err = Inf

    function _result(converged::Bool, iters::Int)
        density_up_mpo = density_up_mpo_gpu === nothing ? nothing : _to_cpu_mpo(density_up_mpo_gpu)
        density_dn_mpo = density_dn_mpo_gpu === nothing ? nothing : _to_cpu_mpo(density_dn_mpo_gpu)
        Hup = _copy_with_mpo(H0_up, _to_cpu_mpo(Hup_mpo_gpu); scale=scale_eff, center=0.0)
        Hdn = _copy_with_mpo(H0_dn, _to_cpu_mpo(Hdn_mpo_gpu); scale=scale_eff, center=0.0)
        return (
            converged=converged,
            iterations=iters,
            rms_error=err,
            rho_up=_to_cpu_mps(rho_up_gpu),
            rho_dn=_to_cpu_mps(rho_dn_gpu),
            density_up_mpo=density_up_mpo,
            density_dn_mpo=density_dn_mpo,
            H_up=Hup,
            H_dn=Hdn,
            rho_up_gpu=rho_up_gpu,
            rho_dn_gpu=rho_dn_gpu,
            density_up_mpo_gpu=density_up_mpo_gpu,
            density_dn_mpo_gpu=density_dn_mpo_gpu,
            H_up_mpo_gpu=Hup_mpo_gpu,
            H_dn_mpo_gpu=Hdn_mpo_gpu,
            history=history,
        )
    end

    for iter in 1:max_scf_iter
        V_up_gpu = U isa MPO ?
            _hartree_mpo_from_density_gpu(rho_dn_gpu, U_gpu, sites, bg_gpu;
                                          maxdim=maxdim, cutoff=cutoff) :
            _local_hartree_from_density_gpu(rho_dn_gpu, sites, U, bg_gpu;
                                            maxdim=maxdim, cutoff=cutoff)
        V_dn_gpu = U isa MPO ?
            _hartree_mpo_from_density_gpu(rho_up_gpu, U_gpu, sites, bg_gpu;
                                          maxdim=maxdim, cutoff=cutoff) :
            _local_hartree_from_density_gpu(rho_up_gpu, sites, U, bg_gpu;
                                            maxdim=maxdim, cutoff=cutoff)

        Hup_mpo_gpu = +(H0_up_gpu, V_up_gpu; maxdim=maxdim, cutoff=Float64(cutoff))
        Hdn_mpo_gpu = +(H0_dn_gpu, V_dn_gpu; maxdim=maxdim, cutoff=Float64(cutoff))

        density_up_mpo_gpu = _mcweeny_purify_mpo_gpu(Hup_mpo_gpu, sites;
            ϵF=fermi, scale=scale_eff, center=0.0, Id_gpu=Id_gpu,
            maxdim=maxdim, cutoff=cutoff, maxiters=purif_maxiter,
            tol=Float64(purif_tol))
        density_dn_mpo_gpu = _mcweeny_purify_mpo_gpu(Hdn_mpo_gpu, sites;
            ϵF=fermi, scale=scale_eff, center=0.0, Id_gpu=Id_gpu,
            maxdim=maxdim, cutoff=cutoff, maxiters=purif_maxiter,
            tol=Float64(purif_tol))

        rho_up_new_gpu = density_profile_from_dm_gpu(density_up_mpo_gpu, sites;
                                                     maxdim=maxdim, cutoff=cutoff)
        rho_dn_new_gpu = density_profile_from_dm_gpu(density_dn_mpo_gpu, sites;
                                                     maxdim=maxdim, cutoff=cutoff)

        err_up = _rms_error_gpu(rho_up_new_gpu, rho_up_gpu)
        err_dn = _rms_error_gpu(rho_dn_new_gpu, rho_dn_gpu)
        err = sqrt((err_up^2 + err_dn^2) / 2)
        particle_err = abs(real(tr(density_up_mpo_gpu)) - float(Nel_up)) +
                       abs(real(tr(density_dn_mpo_gpu)) - float(Nel_dn))

        push!(history, (iter=iter, rms_error=err, rms_up=err_up, rms_dn=err_dn,
                        particle_error=particle_err))
        verbose && println("magnetic SCF (gpu) iter=$iter rms=$err particle_err=$particle_err")

        rho_up_mixed = +(mix * rho_up_new_gpu, (1.0 - mix) * rho_up_gpu;
                         maxdim=maxdim, cutoff=Float64(cutoff))
        rho_dn_mixed = +(mix * rho_dn_new_gpu, (1.0 - mix) * rho_dn_gpu;
                         maxdim=maxdim, cutoff=Float64(cutoff))

        rho_up_gpu, rho_dn_gpu = rho_up_mixed, rho_dn_mixed
        _gpu_gc!()
        err < scf_tol && return _result(true, iter)
    end

    return _result(false, max_scf_iter)
end

# ============================================================
# 3. Post-convergence observables
# ============================================================

# Thin 2D-grid wrapper around the shared geometry-aware planner
# spatial_sampling_plan (core/Utils.jl). Used by get_scf_magnetization_gpu. In
# :point mode, `groups` lists the sampled cells explicitly. In :block mode, the
# groups are nominal centers and the caller should use plan.a/plan.b with
# _eval_block_mps_gpu.
function _tb_spatial_plan_gpu(sites;
                              num_x::Int = 0,
                              num_y::Union{Nothing,Int} = nothing,
                              num_avg::Int = 1,
                              x_start::Int = 1,
                              x_end::Int = prod(dim(s) for s in sites),
                              x_groups = nothing,
                              box_half::Int = 0,
                              reduce::Symbol = :point,
                              Lx::Union{Nothing,Int} = nothing)
    L = length(sites)
    return spatial_sampling_plan(L;
        Lx       = something(Lx, div(L, 2)),
        grid     = x_groups === nothing,
        reduce   = reduce,
        num_x    = num_x, num_y = num_y, num_avg = num_avg,
        x_start  = x_start, x_end = x_end,
        x_groups = x_groups, box_half = box_half)
end

"""
    get_scf_magnetization_gpu(res; num_x=0, num_y=nothing, num_avg=1, x_start=1,
                              x_end=prod(dim(s) for s in res.H_up.sites),
                              x_groups=nothing, box_half=0, reduce=:point,
                              Lx=nothing)
        -> (values, centers, groups, n_up, n_dn, reduce, stride_x, stride_y)

Sample the converged magnetic SCF density matrices on GPU and extract only the
final scalar values. If `res` carries GPU density MPOs from
`scf_magnetic_hubbard_gpu`, they are reused directly; otherwise the CPU density
MPOs are uploaded once. Each sampled point is evaluated in the same big-endian
real-space convention as `binary_to_MPS`. `values = (n_up .- n_dn) ./ 2`, and
the sampling plan is the 2D grid of `spatial_sampling_plan` (`Lx` position
qubits along x, default half of them) unless `x_groups` is given.

Set `reduce=:block` to average over every unit cell in each coarse block by
tracing the within-block position bits on GPU. In block mode, `num_x` and
`num_y` must be powers of two, `x_groups` is rejected and `box_half` is ignored.
"""
function get_scf_magnetization_gpu(res;
                                   num_x::Int = 0,
                                   num_y::Union{Nothing,Int} = nothing,
                                   num_avg::Int = 1,
                                   x_start::Int = 1,
                                   x_end::Int = prod(dim(s) for s in res.H_up.sites),
                                   x_groups = nothing,
                                   box_half::Int = 0,
                                   reduce::Symbol = :point,
                                   Lx::Union{Nothing,Int} = nothing)
    reduce in (:point, :block) ||
        error("get_scf_magnetization_gpu: reduce must be :point or :block, got $reduce.")
    reduce === :block && x_groups !== nothing &&
        error("get_scf_magnetization_gpu: x_groups is not supported with reduce=:block.")
    reduce === :block && box_half > 0 &&
        @warn "get_scf_magnetization_gpu: box_half=$box_half is ignored with reduce=:block; each block is fully averaged."

    up_mpo = hasproperty(res, :density_up_mpo_gpu) && res.density_up_mpo_gpu !== nothing ?
        res.density_up_mpo_gpu : res.density_up_mpo
    dn_mpo = hasproperty(res, :density_dn_mpo_gpu) && res.density_dn_mpo_gpu !== nothing ?
        res.density_dn_mpo_gpu : res.density_dn_mpo

    up_mpo === nothing &&
        error("get_scf_magnetization_gpu: res.density_up_mpo is missing.")
    dn_mpo === nothing &&
        error("get_scf_magnetization_gpu: res.density_dn_mpo is missing.")

    sites = res.H_up.sites
    plan = _tb_spatial_plan_gpu(sites;
        num_x=num_x, num_y=num_y, num_avg=num_avg, x_start=x_start, x_end=x_end,
        x_groups=x_groups, box_half=box_half, reduce=reduce, Lx=Lx)
    centers, groups = plan.centers, plan.groups

    up_diag_gpu = density_profile_from_dm_gpu(up_mpo, sites)
    dn_diag_gpu = density_profile_from_dm_gpu(dn_mpo, sites)

    n_up, n_dn = if reduce === :block
        Lx_eff = something(Lx, div(length(sites), 2))
        Ly_eff = length(sites) - Lx_eff
        nbx = 2^plan.a
        nby = 2^plan.b
        norm = Float64(plan.stride_x * plan.stride_y)
        up_vals = Float64[
            _eval_block_mps_gpu(up_diag_gpu, ixp, iyp, plan.a, plan.b, Lx_eff, Ly_eff) / norm
            for iyp in 0:(nby - 1) for ixp in 0:(nbx - 1)
        ]
        dn_vals = Float64[
            _eval_block_mps_gpu(dn_diag_gpu, ixp, iyp, plan.a, plan.b, Lx_eff, Ly_eff) / norm
            for iyp in 0:(nby - 1) for ixp in 0:(nbx - 1)
        ]
        up_vals, dn_vals
    else
        up_vals = Float64[
            sum(_eval_mps_bigendian_gpu(up_diag_gpu, x - 1) for x in grp) / length(grp)
            for grp in groups
        ]
        dn_vals = Float64[
            sum(_eval_mps_bigendian_gpu(dn_diag_gpu, x - 1) for x in grp) / length(grp)
            for grp in groups
        ]
        up_vals, dn_vals
    end
    values = (n_up .- n_dn) ./ 2
    _gpu_gc!()
    return (values=values, centers=centers, groups=groups, n_up=n_up, n_dn=n_dn,
            reduce=reduce, stride_x=plan.stride_x, stride_y=plan.stride_y)
end

"""
    get_scf_bands_gpu(res, Ncheb, omega; kwargs...) -> (Ak, omega, ticks, labels)

Compute spin-summed mean-field bands from a converged magnetic SCF result. This
is deliberately separate from `scf_magnetic_hubbard_gpu`: it initializes from
the CPU `res.H_up`/`res.H_dn`, then each `get_bands_gpu` call uploads once and
keeps the Chebyshev/QFT accumulation on GPU, extracting only scalars. `kwargs`
are passed unchanged to both [`get_bands_gpu`](@ref) calls (including
`type`/`dtype`, complex only).
"""
function get_scf_bands_gpu(res, Ncheb::Int, omega; kwargs...)
    rb_up = get_bands_gpu(res.H_up, Ncheb, omega; kwargs...)
    rb_dn = get_bands_gpu(res.H_dn, Ncheb, omega; kwargs...)
    Ak_up = rb_up isa NamedTuple ? rb_up.Ak : rb_up
    Ak_dn = rb_dn isa NamedTuple ? rb_dn.Ak : rb_dn
    return (Ak = Ak_up .+ Ak_dn,
            omega = collect(omega),
            ticks = rb_up isa NamedTuple ? rb_up.ticks : nothing,
            labels = rb_up isa NamedTuple ? rb_up.labels : nothing)
end
