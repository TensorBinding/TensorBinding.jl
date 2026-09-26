# gpu/timeev.jl — GPU time-evolution trajectories: RK4 evolution of a density
# matrix under a non-Hermitian Hamiltonian (rk4_step_dm_nh_gpu,
# get_nh_density_trajectory_gpu) and TDVP evolution of a state
# (get_state_amplitude_trajectory_gpu), each with its GPU sampler. Moved from the
# former gpu/GPU_tk.jl.
#
# Main entry points: get_nh_density_trajectory_gpu,
# get_state_amplitude_trajectory_gpu, rk4_step_dm_nh_gpu.
# Depends on: core/Utils.jl (spatial_sampling_plan), core/TBSystem.jl
# (TBHamiltonian), solvers/Timeev.jl (the RK4 and TDVP steps _rk4_step,
# _tdvp_step), gpu/device.jl, gpu/primitives.jl.


# ============================================================
# 1. Density-matrix RK4 under a non-Hermitian Hamiltonian
# ============================================================

function _nh_von_neumann_rhs_gpu(H_gpu::MPO, Hdag_gpu::MPO, rho_gpu::MPO;
                                 maxdim::Int, cutoff::Real)
    ak = (cutoff=Float64(cutoff), maxdim=maxdim)
    Hrho    = apply(H_gpu, rho_gpu; ak...)
    rhoHdag = apply(rho_gpu, Hdag_gpu; ak...)
    diff = +(Hrho, ComplexF32(-1) * rhoHdag; ak...)
    ITensorMPS.truncate!(diff; cutoff=Float64(cutoff), maxdim=maxdim)
    return ComplexF32(0, -1) * diff
end

# One RK4 step of dρ/dt = -i(Hρ − ρH†) on GPU MPOs (H, H† and ρ already on GPU),
# through the CPU kernel _rk4_step (solvers/Timeev.jl). The step coefficients
# dt/2, dt, dt/6, 2 and the right-hand-side constants are ComplexF32 whatever the
# element type of the MPOs, and every MPO sum truncates with maxdim as well as
# cutoff (the CPU steps pass only cutoff).
function rk4_step_dm_nh_gpu(H_gpu::MPO, Hdag_gpu::MPO, rho_gpu::MPO, dt::Real;
                            maxdim::Int = 200,
                            cutoff::Real = 1e-8,
                            truncate_intermediates::Bool = true)
    coeffs = (ComplexF32(Float32(dt / 2)), ComplexF32(Float32(dt)),
              ComplexF32(Float32(dt / 6)), ComplexF32(2))
    rhs = (_, rho) -> _nh_von_neumann_rhs_gpu(H_gpu, Hdag_gpu, rho; maxdim=maxdim, cutoff=cutoff)
    rho_new = _rk4_step(rhs, rho_gpu, coeffs;
                        maxdim=maxdim, cutoff=Float64(cutoff),
                        truncate_intermediates=truncate_intermediates,
                        add_kwargs=(cutoff=Float64(cutoff), maxdim=maxdim))
    _gpu_gc!()
    return rho_new
end

function _sample_density_diag_gpu(rho_gpu::MPO, plan;
                                  maxdim::Int,
                                  cutoff::Real)
    diag_gpu = density_profile_from_dm_gpu(rho_gpu;
                                           maxdim=maxdim, cutoff=cutoff)
    vals = if plan.reduce === :block
        Lbits = length(diag_gpu)
        nb    = 2^plan.a
        norm  = Float64(plan.stride_x)
        Float64[_eval_block_mps_1d_gpu(diag_gpu, ixp, plan.a, Lbits) / norm
                for ixp in 0:(nb - 1)]
    else
        Float64[
            sum(_eval_mps_bigendian_gpu(diag_gpu, x - 1) for x in grp) / length(grp)
            for grp in plan.groups
        ]
    end
    _gpu_gc!()
    return vals
end

function _mpo_ket_siteinds(W::MPO)
    return Index[
        let (_, ket) = siteinds(W, i)
            ket
        end
        for i in 1:length(W)
    ]
end

"""
    get_nh_density_trajectory_gpu(H, rho0; nsteps, dt, sample_every=1,
                                  num_x=0, num_avg=1, reduce=:point,
                                  x_start=1, x_end=nothing, x_groups=nothing,
                                  maxdim=200, cutoff=1e-8,
                                  truncate_intermediates=true, dtype=ComplexF32,
                                  printinfo=false, verbose=false)
        -> (density, times, centers, groups, maxlinkdims)

GPU RK4 evolution of a density matrix under a static non-Hermitian Hamiltonian
using `d rho/dt = -i(H rho - rho Hdagger)`. `H` may be a `TBHamiltonian` or an
MPO; `rho0` may be a CPU or GPU MPO. The Hamiltonian and density matrix are
uploaded once and the RK4 loop stays on GPU. At every sampled time, the diagonal
of `rho(t)` is extracted on GPU and only scalar values at the requested groups
are copied back to CPU. `nsteps` and `dt` are required; samples are taken every
`sample_every` steps and at the last step.

`dtype` is the GPU element type, complex only: `ComplexF32` (default) or
`ComplexF64`; an MPO that is already on GPU keeps the type it was uploaded with.
The RK4 step (`rk4_step_dm_nh_gpu`) uses ComplexF32 step coefficients whatever
the `dtype`.

Sampling follows the 1D `spatial_sampling_plan` convention: use `num_x=0` to
sample all sites, or set `num_x` to a smaller number for coarse production
output. `num_avg > 1` averages a few sub-points per sampled spatial bin in
`:point` mode. With `reduce=:block`, `num_x` must be a power of two and each
output value is the GPU block average over a contiguous interval of size
`2^L / num_x`. `x_end=nothing` means the last site.
"""
function get_nh_density_trajectory_gpu(H, rho0::MPO;
                                       nsteps::Int,
                                       dt::Real,
                                       sample_every::Int = 1,
                                       num_x::Int = 0,
                                       num_avg::Int = 1,
                                       reduce::Symbol = :point,
                                       x_start::Int = 1,
                                       x_end::Union{Nothing,Int} = nothing,
                                       x_groups = nothing,
                                       maxdim::Int = 200,
                                       cutoff::Real = 1e-8,
                                       truncate_intermediates::Bool = true,
                                       dtype::Type{<:Complex} = ComplexF32,
                                       printinfo::Bool = false,
                                       verbose::Bool = false)
    _check_gpu("get_nh_density_trajectory_gpu")
    gpu_type = _resolve_gpu_type("get_nh_density_trajectory_gpu", dtype, nothing, cutoff)
    nsteps >= 0 || error("get_nh_density_trajectory_gpu: nsteps must be non-negative.")
    sample_every > 0 || error("get_nh_density_trajectory_gpu: sample_every must be positive.")
    reduce in (:point, :block) || error("get_nh_density_trajectory_gpu: reduce must be :point or :block.")

    H_mpo = H isa TBHamiltonian ? H.mpo : H
    sites = H isa TBHamiltonian ? H.sites : _mpo_ket_siteinds(H_mpo)
    Lbits = length(sites)
    Nsite = prod(dim(s) for s in sites)
    x_end_eff = x_end === nothing ? Nsite : Int(x_end)
    plan = spatial_sampling_plan(Lbits;
        grid=false, reduce=reduce, num_x=num_x, num_avg=num_avg,
        x_start=x_start, x_end=x_end_eff, x_groups=x_groups)
    centers, groups = plan.centers, plan.groups

    sample_steps = collect(0:sample_every:nsteps)
    if last(sample_steps) != nsteps
        push!(sample_steps, nsteps)
    end
    times = Float64[step * Float64(dt) for step in sample_steps]
    density = Matrix{Float64}(undef, length(centers), length(sample_steps))
    maxlinks = Vector{Int}(undef, length(sample_steps))

    H_gpu = _ensure_gpu_mpo(H_mpo, gpu_type; caller="get_nh_density_trajectory_gpu")
    Hdag_gpu = conj(swapprime(H_gpu, 0, 1))
    rho_gpu = _ensure_gpu_mpo(rho0, gpu_type; caller="get_nh_density_trajectory_gpu")

    sample_idx = 1
    density[:, sample_idx] = _sample_density_diag_gpu(rho_gpu, plan;
        maxdim=maxdim, cutoff=cutoff)
    maxlinks[sample_idx] = maxlinkdim(rho_gpu)
    printinfo && println("  [gpu] NH density sample step 0/$nsteps  t=0.0  maxlinkdim=$(maxlinks[sample_idx])")

    for step in 1:nsteps
        rho_gpu = rk4_step_dm_nh_gpu(H_gpu, Hdag_gpu, rho_gpu, dt;
            maxdim=maxdim, cutoff=cutoff,
            truncate_intermediates=truncate_intermediates)
        if step % sample_every == 0 || step == nsteps
            sample_idx += 1
            density[:, sample_idx] = _sample_density_diag_gpu(rho_gpu, plan;
                maxdim=maxdim, cutoff=cutoff)
            maxlinks[sample_idx] = maxlinkdim(rho_gpu)
            (verbose || printinfo) &&
                println("  [gpu] NH density sample step $step/$nsteps  t=$(round(step * Float64(dt), digits=6))  maxlinkdim=$(maxlinks[sample_idx])")
        elseif verbose
            println("  [gpu] NH density RK4 step $step/$nsteps  maxlinkdim=$(maxlinkdim(rho_gpu))")
        end
    end

    return (density=density, times=times, centers=centers, groups=groups,
            maxlinkdims=maxlinks)
end


# ============================================================
# 2. TDVP state amplitudes
# ============================================================

function _state_amplitude_component(z::Complex, component::Symbol)
    component === :real && return real(z)
    component === :imag && return imag(z)
    component === :abs  && return abs(z)
    component in (:abs2, :probability) && return abs2(z)
    error("_state_amplitude_component: unsupported component :$component. Use :real, :imag, :abs, :abs2, or :probability.")
end

function _sample_state_amplitudes_gpu(ψ_gpu::MPS, plan;
                                      component::Symbol,
                                      pointavg::Symbol = :complex)
    component in (:real, :imag, :abs, :abs2, :probability) ||
        error("_sample_state_amplitudes_gpu: unsupported component :$component.")
    pointavg in (:complex, :abs, :abs2) ||
        error("_sample_state_amplitudes_gpu: pointavg must be :complex, :abs, or :abs2.")

    if plan.reduce === :block
        Lbits = length(ψ_gpu)
        nb    = 2^plan.a
        norm  = Float64(plan.stride_x)
        amps  = ComplexF64[_eval_block_mps_1d_complex_gpu(ψ_gpu, ixp, plan.a, Lbits) / norm
                           for ixp in 0:(nb - 1)]
        return Float64[_state_amplitude_component(z, component) for z in amps]
    else
        if pointavg === :complex
            # Default: coherent average of complex amplitudes, then apply component.
            amps = ComplexF64[
                sum(_eval_mps_bigendian_complex_gpu(ψ_gpu, x - 1) for x in grp) / length(grp)
                for grp in plan.groups
            ]
            return Float64[_state_amplitude_component(z, component) for z in amps]
        else
            # Incoherent average: apply abs or abs2 per site before averaging.
            _paf = pointavg === :abs2 ? abs2 : abs
            return Float64[
                sum(_paf(_eval_mps_bigendian_complex_gpu(ψ_gpu, x - 1)) for x in grp) / length(grp)
                for grp in plan.groups
            ]
        end
    end
end

function _state_norm_gpu(ψ_gpu::MPS)
    n2 = real(inner(ψ_gpu, ψ_gpu))
    return sqrt(max(Float64(n2), 0.0))
end

"""
    get_state_amplitude_trajectory_gpu(H, psi0; nsteps, dt, sample_every=1,
                                       num_x=0, num_avg=1, reduce=:point,
                                       x_start=1, x_end=nothing, x_groups=nothing,
                                       component=:real, pointavg=:complex,
                                       normalize_each_step=false,
                                       maxdim=200, cutoff=1e-8,
                                       reverse_step=false, outputlevel=0, nsite=2,
                                       dtype=ComplexF32, printinfo=false,
                                       verbose=false)
        -> (amplitude, times, centers, groups, norms, maxlinkdims)

GPU TDVP evolution of a single-particle MPS state under the physical
Hamiltonian `H`. `ITensorMPS.tdvp(operator, t, init)` computes
`exp(t * operator) * init` (generator form), so the operator passed to `tdvp`
is `-im * H`, which implements the Schrödinger evolution `dψ/dt = -im * Hψ`;
therefore a loss term `-im * Γ` (Γ >= 0) damps the norm, matching
`evolve_with_tdvp(H::TBHamiltonian,...)` on CPU and the NH RK4 convention
`dρ/dt = -i(Hρ - ρH†)`. `H` may be a `TBHamiltonian` or an MPO. The Hamiltonian
and initial state are uploaded once, the TDVP loop stays on GPU, and only
sampled scalar amplitudes are copied back to CPU. `nsteps` and `dt` are
required; `nsite`, `reverse_step`, `outputlevel` and `normalize_each_step` are
passed to `tdvp` (the last also renormalizes the state after every step).

The returned `amplitude` matrix has rows = sampled positions and columns =
sampled times. By default it stores `real(<x|psi(t)>)`; `component` may be
`:real`, `:imag`, `:abs`, `:abs2` or `:probability` (the same as `:abs2`).

Sampling follows `spatial_sampling_plan` in 1D. `reduce=:point` samples
representative positions or explicit groups; `pointavg` sets how a group is
averaged: `:complex` (default) averages the complex amplitudes and then takes
`component`, while `:abs`/`:abs2` average `|ψ|`/`|ψ|²` site by site (then
`component` is not applied). `reduce=:block` returns the block-averaged complex
amplitude over contiguous intervals, then takes `component`.

`dtype` is the GPU element type, complex only: `ComplexF32` (default) or
`ComplexF64`; an MPS/MPO that is already on GPU keeps the type it was uploaded
with.
"""
function get_state_amplitude_trajectory_gpu(H, psi0::MPS;
                                            nsteps::Int,
                                            dt::Real,
                                            sample_every::Int = 1,
                                            num_x::Int = 0,
                                            num_avg::Int = 1,
                                            reduce::Symbol = :point,
                                            x_start::Int = 1,
                                            x_end::Union{Nothing,Int} = nothing,
                                            x_groups = nothing,
                                            component::Symbol = :real,
                                            pointavg::Symbol = :complex,
                                            normalize_each_step::Bool = false,
                                            maxdim::Int = 200,
                                            cutoff::Real = 1e-8,
                                            reverse_step::Bool = false,
                                            outputlevel::Int = 0,
                                            nsite::Int = 2,
                                            dtype::Type{<:Complex} = ComplexF32,
                                            printinfo::Bool = false,
                                            verbose::Bool = false)
    _check_gpu("get_state_amplitude_trajectory_gpu")
    gpu_type = _resolve_gpu_type("get_state_amplitude_trajectory_gpu", dtype, nothing, cutoff)
    nsteps >= 0 || error("get_state_amplitude_trajectory_gpu: nsteps must be non-negative.")
    sample_every > 0 || error("get_state_amplitude_trajectory_gpu: sample_every must be positive.")
    reduce in (:point, :block) || error("get_state_amplitude_trajectory_gpu: reduce must be :point or :block.")
    component in (:real, :imag, :abs, :abs2, :probability) ||
        error("get_state_amplitude_trajectory_gpu: unsupported component :$component.")

    H_mpo = H isa TBHamiltonian ? H.mpo : H
    Lbits = length(psi0)
    Nsite = prod(dim(s) for s in siteinds(psi0))
    x_end_eff = x_end === nothing ? Nsite : Int(x_end)
    plan = spatial_sampling_plan(Lbits;
        grid=false, reduce=reduce, num_x=num_x, num_avg=num_avg,
        x_start=x_start, x_end=x_end_eff, x_groups=x_groups)
    centers, groups = plan.centers, plan.groups

    sample_steps = collect(0:sample_every:nsteps)
    if last(sample_steps) != nsteps
        push!(sample_steps, nsteps)
    end
    times = Float64[step * Float64(dt) for step in sample_steps]
    amplitude = Matrix{Float64}(undef, length(centers), length(sample_steps))
    norms = Vector{Float64}(undef, length(sample_steps))
    maxlinks = Vector{Int}(undef, length(sample_steps))

    H_gpu = _ensure_gpu_mpo(H_mpo, gpu_type; caller="get_state_amplitude_trajectory_gpu")
    # ITensorMPS.tdvp(operator, t, init) computes exp(t*operator)*init (generator
    # form, no implicit sign flip), so -im*H gives dψ/dt = -im*Hψ, matching
    # evolve_with_tdvp(H::TBHamiltonian,...) (-im*H.mpo) and the NH RK4 convention
    # dρ/dt = -i(Hρ - ρH†). For H = H0 - iΓ this makes Γ>=0 lossy.
    generator_gpu = gpu_type(0, -1) * H_gpu
    ψ_gpu = _ensure_gpu_mps(psi0, gpu_type; caller="get_state_amplitude_trajectory_gpu")

    sample_idx = 1
    amplitude[:, sample_idx] = _sample_state_amplitudes_gpu(ψ_gpu, plan;
        component=component, pointavg=pointavg)
    norms[sample_idx] = _state_norm_gpu(ψ_gpu)
    maxlinks[sample_idx] = maxlinkdim(ψ_gpu)
    printinfo && println("  [gpu] state sample step 0/$nsteps  t=0.0  norm=$(round(norms[sample_idx], sigdigits=6))  maxlinkdim=$(maxlinks[sample_idx])")

    for step in 1:nsteps
        ψ_gpu = _tdvp_step(
            generator_gpu,
            ψ_gpu,
            dt;
            nsite=nsite,
            maxdim=maxdim,
            cutoff=Float64(cutoff),
            normalize=normalize_each_step,
            reverse_step=reverse_step,
            outputlevel=outputlevel,
        )
        normalize_each_step && normalize!(ψ_gpu)

        if step % sample_every == 0 || step == nsteps
            sample_idx += 1
            amplitude[:, sample_idx] = _sample_state_amplitudes_gpu(ψ_gpu, plan;
                component=component, pointavg=pointavg)
            norms[sample_idx] = _state_norm_gpu(ψ_gpu)
            maxlinks[sample_idx] = maxlinkdim(ψ_gpu)
            (verbose || printinfo) &&
                println("  [gpu] state sample step $step/$nsteps  t=$(round(step * Float64(dt), digits=6))  norm=$(round(norms[sample_idx], sigdigits=6))  maxlinkdim=$(maxlinks[sample_idx])")
            _gpu_gc!()
        elseif verbose
            println("  [gpu] state TDVP step $step/$nsteps  maxlinkdim=$(maxlinkdim(ψ_gpu))")
        end
    end

    return (amplitude=amplitude, times=times, centers=centers, groups=groups,
            norms=norms, maxlinkdims=maxlinks)
end
