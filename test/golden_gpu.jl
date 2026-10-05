using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test, Random
using TensorBinding: get_Hamiltonian, add_spin!, add_superconductivity!, add_loss!, TBHamiltonian

# Characterization ("golden") tests for the GPU code in src/gpu/*.jl (the former
# src/gpu/GPU_tk.jl).
#
# Every public *_gpu entry point is run on a tiny system (L = 2..4, Ncheb <= 12)
# and its output is pinned twice:
#   * against the value recorded when the data was generated (`gpu` in
#     test/data/gpu_golden.jl), with the case's `rtol`/`atol`;
#   * against the CPU counterpart evaluated at generation time (`cpu`), with the
#     case's `cpu_rtol`/`cpu_atol`. The CPU values are stored, not recomputed,
#     so the test only runs GPU code; the CPU code paths have their own goldens.
# MPS/MPO outputs are stored as dense vectors/matrices (big-endian over the site
# order, site 1 = most significant). Shapes and element types are compared
# exactly, integer and Bool fields with `==`, floating-point fields with
# `isapprox` (norm-wise). GPU arithmetic is not bit-reproducible, so ComplexF64
# and Float64 cases use rtol = 1e-8; the ComplexF32 cases (the default element
# type of most entry points, and the forced one-argument upload paths) use
# rtol = 1e-4. Most cases run in ComplexF64; each entry point that accepts real
# types has one Float64 case.
#
# Not pinned: the internal helpers, which are only reached through the entry
# points.
#
# Runtime: about 6 minutes cold on an RTX 4060 laptop GPU. Nearly all of it is
# first-call compilation of CUDA.jl/NDTensors kernels (one per element type and
# per operation family), not the tiny calls themselves.
#
# The data file is written by test/data/generate_gpu_golden.jl (see its header).
# A failure here means a GPU entry point changed its output. If that is a
# regression, fix the code; if it is intentional, regenerate the data in the same
# commit and review the diff case by case.
#
# The runner module below is shared with the generator, which includes this file
# with `GPU_GOLDEN_GENERATOR` defined so that only the module is loaded. The whole
# test is skipped (@test_skip) when CUDA.jl is missing or not functional, except
# the CPU-only case of `_reconstruct_ldos_moment_columns` (it lived in GPU_tk.jl
# and is now in src/solvers/kpm/kernels.jl).

module GPUGoldenRunner

using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test, Random
using NDTensors
using TensorBinding: get_Hamiltonian, add_spin!, add_superconductivity!, add_loss!, TBHamiltonian

const TB = TensorBinding

# ── CUDA access without a hard dependency ─────────────────────────────────────
const CUDA_PKGID = Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA")
cuda() = Base.loaded_modules[CUDA_PKGID]

_cast(::Type{T}, a) where {T<:Real} = T.(real.(a))
_cast(::Type{T}, a) where {T<:Complex} = T.(a)

"Upload a CPU MPO/MPS to the GPU with element type `T` (independent of TensorBinding's helpers)."
function upload(W::Union{MPO,MPS}, ::Type{T}) where {T<:Number}
    ts = [let t = dense(W[i]); itensor(cuda().CuArray(_cast(T, Array(t, inds(t)...))), inds(t)...) end
          for i in eachindex(W)]
    return W isa MPO ? MPO(ts) : MPS(ts)
end

"Copy one (GPU or CPU) ITensor to a dense CPU ITensor, keeping its element type."
function host(t::ITensor)
    td = dense(t)
    return itensor(Array(NDTensors.data(NDTensors.storage(ITensors.tensor(td)))), inds(td)...)
end

gpu_eltype(W::Union{MPO,MPS}) = eltype(NDTensors.data(NDTensors.storage(ITensors.tensor(W[1]))))

"Dense vector of an MPS, big-endian over the site order (site 1 = most significant)."
function dense_vector(ψ::MPS)
    ts = [host(ψ[i]) for i in eachindex(ψ)]
    s = siteinds(MPS(ts))
    A = Array(foldl(*, ts), s...)
    return vec(permutedims(A, ndims(A):-1:1))
end

"Dense matrix of an MPO (rows = primed indices), big-endian over the site order."
function dense_matrix(W::MPO)
    ts = [host(W[i]) for i in eachindex(W)]
    Wc = MPO(ts)
    kets = [only(filter(i -> plev(i) == 0, siteinds(Wc, n))) for n in eachindex(Wc)]
    n = length(kets)
    A = Array(foldl(*, ts), prime.(kets)..., kets...)
    d = prod(dim, kets)
    return reshape(permutedims(A, (n:-1:1..., 2n:-1:n+1...)), d, d)
end

groups_int(groups) = [collect(Int, g) for g in groups]
group_means(d, groups) = [sum(d[g]) / length(g) for g in groups]
block_means(d, nb) = (n = length(d) ÷ nb; [sum(d[(b - 1) * n + 1:b * n]) / n for b in 1:nb])

# Block averages of a 2D profile d[iy * 2^Lx + ix + 1] over nbx × nby coarse
# pixels, column order ixp + iyp * nbx + 1 (the GPU block layout).
function block_means_2d(d, Lx, Ly, nbx, nby)
    Nx, Ny = 2^Lx, 2^Ly
    sx, sy = Nx ÷ nbx, Ny ÷ nby
    return [sum(d[iy * Nx + ix + 1] for iy in iyp*sy:(iyp+1)*sy-1, ix in ixp*sx:(ixp+1)*sx-1) / (sx * sy)
            for iyp in 0:nby-1 for ixp in 0:nbx-1]
end

function message_prefix(err)
    hasproperty(err, :msg) || return nothing
    msg = string(getproperty(err, :msg))
    return String(first(first(split(msg, '\n')), 60))
end

# ── Case table ────────────────────────────────────────────────────────────────
# setup() builds the inputs (right after Random.seed!(case_seed(name))); gpu(inp)
# and cpu(inp) return NamedTuples whose fields are compared. `cpu` returns a
# subset of the GPU fields; cpu === nothing means there is no CPU counterpart.
# `throws = true` pins the exception type and message prefix of a failing call.
struct Case
    name::String
    setup::Function
    gpu::Function
    cpu::Union{Nothing,Function}
    rtol::Float64
    atol::Float64
    cpu_rtol::Float64
    cpu_atol::Float64
    throws::Bool
    needs_gpu::Bool
end

const CASES = Case[]

const TOL64 = (rtol = 1e-8, atol = 1e-10)
const TOL32 = (rtol = 1e-4, atol = 1e-5)

function case!(name, setup, gpu, cpu = nothing; tol = TOL64, cpu_rtol = 1e-7, cpu_atol = 1e-9,
               throws = false, needs_gpu = true)
    any(c -> c.name == name, CASES) && error("duplicate case name $name")
    push!(CASES, Case(name, setup, gpu, cpu, tol.rtol, tol.atol, cpu_rtol, cpu_atol, throws, needs_gpu))
end

case_seed(name) = 1000 + sum(Int, codeunits(name)) % 9000

function run_gpu(c::Case)
    Random.seed!(case_seed(c.name))
    inp = c.setup()
    if c.throws
        try
            c.gpu(inp)
        catch err
            return (exception = nameof(typeof(err)), message_prefix = message_prefix(err))
        end
        return (exception = :none, message_prefix = nothing)
    end
    return c.gpu(inp)
end

function run_cpu(c::Case)
    c.cpu === nothing && return nothing
    Random.seed!(case_seed(c.name))
    return c.cpu(c.setup())
end

# ── Shared inputs ─────────────────────────────────────────────────────────────
const OMEGA = collect(range(-1.8, 1.8; length=5))

chain(; L = 3, scale = 2.5) = get_Hamiltonian("chain_1d", 1.0; L = L, scale = scale)

# Lossy chain: H = H_chain - i diag(0.1 + 0.05 n), n = 0..2^L-1.
function nh_chain(; L = 3)
    H = chain(; L = L, scale = 3.0)
    add_loss!(H, n -> 0.1 + 0.05 * n)
    return H
end

# H = c·1: every random probe of the stochastic NH trace gives the exact trace.
function nh_identity(; L = 2)
    H0 = chain(; L = L, scale = 3.0)
    return TBHamiltonian(H0; mpo = (0.3 - 0.2im) * MPO(H0.sites, "Id"))
end

function spin_chain(; L = 3)
    H = chain(; L = L)
    add_spin!(H)
    H.scale = 2.5
    return H
end

function nambu_chain(; L = 3)
    H = chain(; L = L)
    add_superconductivity!(H, 0.3)
    H.scale = 2.8
    return H
end

honeycomb() = get_Hamiltonian("honeycomb", 1.0; L = 2, scale = 3.2)
square() = get_Hamiltonian("square_2d", 1.0; L = 4, scale = 4.4)
function bilayer(; geometry = false)
    H = TB.bilayer_hamiltonian(:square, 1, 1; t_inter = 0.3)
    H.scale = 3.0
    # bilayer_hamiltonian sets no geometry; get_bands_gpu needs one to infer D = 2.
    geometry && (H.geometry = get_Hamiltonian("square_2d", 1.0; L = 2, scale = 4.4).geometry)
    return H
end
exciton(; L = 2) = TB.exciton_hamiltonian("chain_1d", 1.0, x -> -1.0; L = L, scale = 4.5)
fibonacci() = TB.fibonacci_hamiltonian(4; A = 1.0, B = 2.0, model = :onsite, t = 0.6,
                                       boundary = :open, cutoff = 1e-12, maxdim = 100)

pure_dm(H, x) = (ψ = TB.binary_to_MPS(x, length(H.sites), H.sites); outer(ψ', ψ))

# ═════════════════════════════════════════════════════════════════════════════
# 0. CPU-only helper formerly in GPU_tk.jl (now solvers/kpm/kernels.jl)
# ═════════════════════════════════════════════════════════════════════════════
case!("reconstruct_ldos_moment_columns",
    () -> (moments = [1.0 2.0; 0.5 -1.0; -0.25 0.75],
           W = [1.0 2.0 3.0; 0.5 -1.0 4.0; 2.0 0.25 -2.0],
           denom = [2.0, 4.0, 0.0], valid = [true, true, false]),
    inp -> (ldos = TB._reconstruct_ldos_moment_columns(inp.moments, inp.W, inp.denom, inp.valid),);
    tol = (rtol = 1e-12, atol = 1e-14), needs_gpu = false)

# ═════════════════════════════════════════════════════════════════════════════
# 1. Diagonal extraction and density profiles
# ═════════════════════════════════════════════════════════════════════════════
case!("extract_diagonal_c64",
    () -> nh_chain(),
    H -> (d = TB.extract_diagonal_to_mps_gpu(upload(H.mpo, ComplexF64));
          (diag = dense_vector(d), eltype = gpu_eltype(d))),
    H -> (diag = dense_vector(TB.extract_diagonal_to_mps(H.mpo)),))

case!("extract_diagonal_f64",
    () -> (H = chain(); apply(H.mpo, H.mpo; cutoff = 1e-14)),
    W -> (d = TB.extract_diagonal_to_mps_gpu(upload(W, Float64));
          (diag = dense_vector(d), eltype = gpu_eltype(d))),
    W -> (diag = ComplexF64.(dense_vector(TB.extract_diagonal_to_mps(W))),))

function _dm_setup()
    H = chain()
    return (H = H, rho = apply(H.mpo, H.mpo; cutoff = 1e-14) + MPO(H.sites, "Id"))
end

case!("density_profile_gpu_input_direct_c64",
    _dm_setup,
    inp -> (p = TB.density_profile_from_dm_gpu(upload(inp.rho, ComplexF64));
            (profile = dense_vector(p), eltype = gpu_eltype(p))),
    inp -> (profile = dense_vector(TB.density_profile_from_dm(inp.rho, inp.H.sites)),))

case!("density_profile_gpu_input_complement_c64",
    _dm_setup,
    inp -> (p = TB.density_profile_from_dm_gpu(upload(inp.rho, ComplexF64), inp.H.sites;
                                               mode = :complement, cutoff = 1e-12);
            (profile = dense_vector(p), eltype = gpu_eltype(p))),
    inp -> (profile = dense_vector(TB.density_profile_from_dm(inp.rho, inp.H.sites; mode = :complement)),);
    cpu_rtol = 1e-5, cpu_atol = 1e-5)

# A CPU MPO is uploaded as ComplexF32 (the one-argument upload path).
case!("density_profile_cpu_input_direct_f32",
    _dm_setup,
    inp -> (p = TB.density_profile_from_dm_gpu(inp.rho);
            (profile = dense_vector(p), eltype = gpu_eltype(p))),
    inp -> (profile = dense_vector(TB.density_profile_from_dm(inp.rho, inp.H.sites)),);
    tol = TOL32, cpu_rtol = 1e-5, cpu_atol = 1e-5)

case!("density_profile_bad_mode_throws",
    _dm_setup,
    inp -> TB.density_profile_from_dm_gpu(upload(inp.rho, ComplexF64); mode = :bogus);
    throws = true)

# ═════════════════════════════════════════════════════════════════════════════
# 2. Non-Hermitian RK4 density-matrix evolution
# ═════════════════════════════════════════════════════════════════════════════
function _rk4_setup()
    H = nh_chain()
    return (H = H, rho = pure_dm(H, 2))
end

for (tag, trunc) in (("", true), ("_notrunc", false))
    case!("rk4_step_dm_nh_c64$tag",
        _rk4_setup,
        inp -> begin
            Hg = upload(inp.H.mpo, ComplexF64)
            r = TB.rk4_step_dm_nh_gpu(Hg, conj(swapprime(Hg, 0, 1)), upload(inp.rho, ComplexF64), 0.1;
                                      maxdim = 64, cutoff = 1e-12, truncate_intermediates = trunc)
            (rho = dense_matrix(r), eltype = gpu_eltype(r))
        end,
        inp -> (rho = dense_matrix(TB.rk4_step_dm_nh(_ -> inp.H.mpo, inp.rho, 0.0, 0.1;
                                                     maxdim = 64, cutoff = 1e-12,
                                                     truncate_intermediates = trunc)),);
        cpu_rtol = 1e-6, cpu_atol = 1e-8)
end

function _nh_density_cpu(H, rho0, kw; nsteps, dt, sample_every, groups = nothing, nblock = nothing)
    states = TB.evolve_rk4_dm_nh(_ -> H.mpo, rho0, nsteps, Float64(dt);
                                 maxdim = kw.maxdim, cutoff = kw.cutoff,
                                 truncate_intermediates = get(kw, :truncate_intermediates, true))
    steps = collect(0:sample_every:nsteps)
    last(steps) == nsteps || push!(steps, nsteps)
    cols = map(steps) do s
        d = real.(diag(dense_matrix(states[s + 1])))
        nblock === nothing ? group_means(d, groups) : block_means(d, nblock)
    end
    return (density = reduce(hcat, cols), times = Float64.(steps) .* dt)
end

_traj_out(r) = (density = r.density, times = r.times, centers = r.centers,
                groups = groups_int(r.groups), maxlinkdims = r.maxlinkdims)

case!("nh_density_traj_point_all_c64",
    _rk4_setup,
    inp -> _traj_out(TB.get_nh_density_trajectory_gpu(inp.H, inp.rho; nsteps = 3, dt = 0.1,
        sample_every = 2, maxdim = 64, cutoff = 1e-12, dtype = ComplexF64)),
    inp -> _nh_density_cpu(inp.H, inp.rho, (maxdim = 64, cutoff = 1e-12); nsteps = 3, dt = 0.1,
        sample_every = 2, groups = [[x] for x in 1:8]);
    cpu_rtol = 1e-6, cpu_atol = 1e-8)

case!("nh_density_traj_block_mpo_gpu_input_c64",
    _rk4_setup,
    inp -> _traj_out(TB.get_nh_density_trajectory_gpu(inp.H.mpo, upload(inp.rho, ComplexF64);
        nsteps = 2, dt = 0.1, reduce = :block, num_x = 4, truncate_intermediates = false,
        maxdim = 64, cutoff = 1e-12, dtype = ComplexF64)),
    inp -> _nh_density_cpu(inp.H, inp.rho, (maxdim = 64, cutoff = 1e-12, truncate_intermediates = false);
        nsteps = 2, dt = 0.1, sample_every = 1, nblock = 4);
    cpu_rtol = 1e-6, cpu_atol = 1e-8)

case!("nh_density_traj_window_navg_c64",
    _rk4_setup,
    inp -> _traj_out(TB.get_nh_density_trajectory_gpu(inp.H, inp.rho; nsteps = 2, dt = 0.1,
        num_x = 2, num_avg = 2, x_start = 2, x_end = 7, maxdim = 64, cutoff = 1e-12,
        dtype = ComplexF64)),
    inp -> (r = TB.spatial_sampling_plan(3; grid = false, reduce = :point, num_x = 2, num_avg = 2,
                                         x_start = 2, x_end = 7);
            _nh_density_cpu(inp.H, inp.rho, (maxdim = 64, cutoff = 1e-12); nsteps = 2, dt = 0.1,
                            sample_every = 1, groups = groups_int(r.groups)));
    cpu_rtol = 1e-6, cpu_atol = 1e-8)

case!("nh_density_traj_xgroups_c64",
    _rk4_setup,
    inp -> _traj_out(TB.get_nh_density_trajectory_gpu(inp.H, inp.rho; nsteps = 1, dt = 0.1,
        x_groups = [[1, 2], [3], [6, 7, 8]], cutoff = 1e-12, dtype = ComplexF64)),
    inp -> _nh_density_cpu(inp.H, inp.rho, (maxdim = 200, cutoff = 1e-12); nsteps = 1, dt = 0.1,
        sample_every = 1, groups = [[1, 2], [3], [6, 7, 8]]);
    cpu_rtol = 1e-6, cpu_atol = 1e-8)

# ═════════════════════════════════════════════════════════════════════════════
# 3. TDVP state-amplitude trajectories
# ═════════════════════════════════════════════════════════════════════════════
function _state_setup()
    H = nh_chain()
    return (H = H, psi0 = TB.binary_to_MPS(2, H.L, H.sites))
end

function _component(z, component)
    component === :real && return real(z)
    component === :imag && return imag(z)
    component === :abs && return abs(z)
    return abs2(z)
end

function _state_cpu(H, psi0; nsteps, dt, sample_every = 1, component = :real, pointavg = :complex,
                    normalize_each_step = false, maxdim, cutoff, groups = nothing, nblock = nothing)
    states = TB.evolve_with_tdvp(H, psi0, nsteps, dt; normalize_each_step = normalize_each_step,
                                 maxdim = maxdim, cutoff = cutoff)
    steps = collect(0:sample_every:nsteps)
    last(steps) == nsteps || push!(steps, nsteps)
    cols = map(steps) do s
        v = dense_vector(states[s + 1])
        if nblock !== nothing
            _component.(block_means(v, nblock), component)
        elseif pointavg === :complex
            _component.(group_means(v, groups), component)
        else
            f = pointavg === :abs2 ? abs2 : abs
            group_means(f.(v), groups)
        end
    end
    return (amplitude = reduce(hcat, cols), times = Float64.(steps) .* dt,
            norms = [norm(states[s + 1]) for s in steps])
end

_state_out(r) = (amplitude = r.amplitude, times = r.times, centers = r.centers,
                 groups = groups_int(r.groups), norms = r.norms, maxlinkdims = r.maxlinkdims)

const _ALL8 = [[x] for x in 1:8]

case!("state_traj_real_point_all_c64",
    _state_setup,
    inp -> _state_out(TB.get_state_amplitude_trajectory_gpu(inp.H, inp.psi0; nsteps = 3, dt = 0.1,
        sample_every = 2, maxdim = 32, cutoff = 1e-12, dtype = ComplexF64)),
    inp -> _state_cpu(inp.H, inp.psi0; nsteps = 3, dt = 0.1, sample_every = 2, maxdim = 32,
        cutoff = 1e-12, groups = _ALL8);
    cpu_rtol = 1e-6, cpu_atol = 1e-8)

case!("state_traj_imag_xgroups_normalize_mpo_c64",
    _state_setup,
    inp -> _state_out(TB.get_state_amplitude_trajectory_gpu(inp.H.mpo, inp.psi0; nsteps = 2, dt = 0.1,
        component = :imag, x_groups = [[1, 2], [3], [6, 7, 8]], normalize_each_step = true,
        maxdim = 32, cutoff = 1e-12, dtype = ComplexF64)),
    inp -> _state_cpu(inp.H, inp.psi0; nsteps = 2, dt = 0.1, component = :imag,
        normalize_each_step = true, maxdim = 32, cutoff = 1e-12, groups = [[1, 2], [3], [6, 7, 8]]);
    cpu_rtol = 1e-6, cpu_atol = 1e-8)

case!("state_traj_abs_block_c64",
    _state_setup,
    inp -> _state_out(TB.get_state_amplitude_trajectory_gpu(inp.H, inp.psi0; nsteps = 2, dt = 0.1,
        component = :abs, reduce = :block, num_x = 4, maxdim = 32, cutoff = 1e-12, dtype = ComplexF64)),
    inp -> _state_cpu(inp.H, inp.psi0; nsteps = 2, dt = 0.1, component = :abs, maxdim = 32,
        cutoff = 1e-12, nblock = 4);
    cpu_rtol = 1e-6, cpu_atol = 1e-8)

case!("state_traj_probability_pointavg_abs2_c64",
    _state_setup,
    inp -> _state_out(TB.get_state_amplitude_trajectory_gpu(inp.H, inp.psi0; nsteps = 2, dt = 0.1,
        component = :probability, pointavg = :abs2, num_x = 4, num_avg = 2, maxdim = 32,
        cutoff = 1e-12, dtype = ComplexF64)),
    inp -> (r = TB.spatial_sampling_plan(3; grid = false, reduce = :point, num_x = 4, num_avg = 2,
                                         x_start = 1, x_end = 8);
            _state_cpu(inp.H, inp.psi0; nsteps = 2, dt = 0.1, component = :probability,
                       pointavg = :abs2, maxdim = 32, cutoff = 1e-12, groups = groups_int(r.groups)));
    cpu_rtol = 1e-6, cpu_atol = 1e-8)

case!("state_traj_abs_pointavg_abs_c64",
    _state_setup,
    inp -> _state_out(TB.get_state_amplitude_trajectory_gpu(inp.H, inp.psi0; nsteps = 1, dt = 0.1,
        component = :abs, pointavg = :abs, x_groups = [[2, 3], [7]], maxdim = 32,
        cutoff = 1e-12, dtype = ComplexF64)),
    inp -> _state_cpu(inp.H, inp.psi0; nsteps = 1, dt = 0.1, component = :abs, pointavg = :abs,
        maxdim = 32, cutoff = 1e-12, groups = [[2, 3], [7]]);
    cpu_rtol = 1e-6, cpu_atol = 1e-8)

case!("state_traj_bad_component_throws",
    _state_setup,
    inp -> TB.get_state_amplitude_trajectory_gpu(inp.H, inp.psi0; nsteps = 1, dt = 0.1,
        component = :phase, dtype = ComplexF64);
    throws = true)

# ═════════════════════════════════════════════════════════════════════════════
# 4. Chebyshev MPO recurrence
# ═════════════════════════════════════════════════════════════════════════════
_tn_dense(list) = [x === nothing ? zeros(ComplexF64, 0, 0) : ComplexF64.(dense_matrix(x)) for x in list]

case!("kpm_tn_all_c64",
    () -> chain(),
    H -> ((list, sc, c) = TB.KPM_Tn_gpu(H.mpo, 5, H.sites; scale = 2.5, maxdim = 32, cutoff = 1e-12,
                                        type = ComplexF64, verbose = false);
          (Tn = _tn_dense(list), scale = sc, center = c, eltype = gpu_eltype(list[end]))),
    H -> ((list, sc, c) = TB.KPM_Tn(H.mpo, 5, H.sites; scale = 2.5, maxdim = 32, cutoff = 1e-12,
                                    verbose = false);
          (Tn = _tn_dense(list), scale = sc, center = c)))

case!("kpm_tn_keep_center_dtype_f64",
    () -> chain(),
    H -> ((list, sc, c) = TB.KPM_Tn_gpu(H.mpo, 5, H.sites; scale = 2.5, center = 0.3, maxdim = 32,
                                        cutoff = 1e-12, keep_indices = Set([1, 3, 6]),
                                        dtype = Float64, verbose = false);
          (Tn = _tn_dense(list), kept = [x !== nothing for x in list], scale = sc, center = c,
           eltype = gpu_eltype(list[end]))),
    H -> ((list, sc, c) = TB.KPM_Tn(H.mpo, 5, H.sites; scale = 2.5, center = 0.3, maxdim = 32,
                                    cutoff = 1e-12, verbose = false);
          (Tn = _tn_dense([k in (1, 3, 6) ? list[k] : nothing for k in eachindex(list)]),
           scale = sc, center = c)))

case!("kpm_tn_dmrg_scale_c64",
    () -> chain(),
    H -> ((list, sc, c) = TB.KPM_Tn_gpu(H.mpo, 3, H.sites; maxdim = 32, cutoff = 1e-12,
                                        type = ComplexF64, verbose = false);
          (Tn = _tn_dense(list), scale = sc, center = c)),
    H -> ((list, sc, c) = TB.KPM_Tn(H.mpo, 3, H.sites; maxdim = 32, cutoff = 1e-12, verbose = false);
          (Tn = _tn_dense(list), scale = sc, center = c));
    cpu_rtol = 1e-6, cpu_atol = 1e-8)

case!("kpm_tn_conflicting_types_throws",
    () -> chain(),
    H -> TB.KPM_Tn_gpu(H.mpo, 3, H.sites; scale = 2.5, type = ComplexF64, dtype = Float64,
                       verbose = false);
    throws = true)

# ═════════════════════════════════════════════════════════════════════════════
# 5. Band structure A(k, ω)
# ═════════════════════════════════════════════════════════════════════════════
_bands_out(r) = r isa NamedTuple ? (Ak = r.Ak, ticks = collect(Int, r.ticks), labels = collect(String, r.labels)) :
                                   (Ak = r,)

function bands_case!(name, setup, kw; tol = TOL64, cpu_rtol = 1e-7, cpu_atol = 1e-9, Ncheb = 12)
    case!(name, setup,
        H -> _bands_out(TB.get_bands_gpu(H, Ncheb, OMEGA; kw...)),
        H -> _bands_out(TB.get_bands(H, Ncheb, OMEGA; (k => v for (k, v) in pairs(kw) if k ∉ (:type, :dtype))...));
        tol = tol, cpu_rtol = cpu_rtol, cpu_atol = cpu_atol)
end

const BKW = (maxdim = 64, cutoff = 1e-12)

bands_case!("bands_chain_grid_c64", () -> chain(), (; BKW..., num_x = 4, type = ComplexF64))
bands_case!("bands_square_kpath_c64", square,
    (; BKW..., kpath = [:G, :X, :M, :G], kpath_lattice = :square, num_x = 2, type = ComplexF64))
bands_case!("bands_honeycomb_projsl_grid2d_c64", honeycomb,
    (; BKW..., proj_sl = 1, num_x = 2, num_y = 2, type = ComplexF64))
bands_case!("bands_spin_chain_projs_c64", () -> spin_chain(),
    (; BKW..., proj_s = 2, num_x = 4, type = ComplexF64))
bands_case!("bands_nambu_chain_c64", () -> nambu_chain(),
    (; BKW..., num_x = 4, type = ComplexF64))
bands_case!("bands_bilayer_projlayer_c64", () -> bilayer(; geometry = true),
    (; BKW..., proj_layer = 1, num_x = 2, num_y = 2, type = ComplexF64))
bands_case!("bands_chain_legacy_sublattice_mask_c64", () -> chain(),
    (; BKW..., sublattice = true, num_x = 4, type = ComplexF64))
bands_case!("bands_chain_dtype_alias_navg_lorentz_c64", () -> chain(),
    (; BKW..., num_x = 2, num_avg = 2, kernel = :lorentz, dtype = ComplexF64))

case!("bands_no_geometry_throws",
    bilayer,
    H -> TB.get_bands_gpu(H, 8, OMEGA; num_x = 2, type = ComplexF64);
    throws = true)

case!("bands_real_type_throws",
    () -> chain(),
    H -> TB.get_bands_gpu(H, 8, OMEGA; type = Float64, num_x = 4);
    throws = true)

# ═════════════════════════════════════════════════════════════════════════════
# 6. Spatial LDOS, MPO recurrence
# ═════════════════════════════════════════════════════════════════════════════
function ldos_case!(name, setup, kw; tol = TOL64, cpu_rtol = 1e-7, cpu_atol = 1e-9, Ncheb = 12)
    case!(name, setup,
        H -> (ldos = TB.get_ldos_spatial_gpu(H, Ncheb, OMEGA; kw...),),
        H -> (ldos = TB.get_ldos_spatial(H, Ncheb, OMEGA; mode = :mpo,
                  (k => v for (k, v) in pairs(kw) if k ∉ (:type, :dtype))...),);
        tol = tol, cpu_rtol = cpu_rtol, cpu_atol = cpu_atol)
end

const LKW = (maxdim = 64, cutoff = 1e-12)

ldos_case!("ldos_chain_point_all_c64", () -> chain(), (; LKW..., type = ComplexF64))
ldos_case!("ldos_chain_point_all_f64", () -> chain(), (; LKW..., type = Float64))
ldos_case!("ldos_chain_xgroups_lorentz_c64", () -> chain(),
    (; LKW..., x_groups = [[1, 2], [4], [6, 7, 8]], kernel = :lorentz, type = ComplexF64))
ldos_case!("ldos_chain_window_navg_dtype_c64", () -> chain(),
    (; LKW..., num_x = 2, num_avg = 2, x_start = 2, x_end = 7, dtype = ComplexF64))
ldos_case!("ldos_square_block_c64", square,
    (; LKW..., reduce = :block, num_x = 2, num_y = 2, type = ComplexF64))
ldos_case!("ldos_square_grid_box_c64", square,
    (; LKW..., grid = true, num_x = 2, num_y = 2, box_half = 1, type = ComplexF64))
ldos_case!("ldos_honeycomb_resolve_c64", honeycomb, (; LKW..., type = ComplexF64))
ldos_case!("ldos_honeycomb_average_c64", honeycomb, (; LKW..., sublattice = :average, type = ComplexF64))
ldos_case!("ldos_honeycomb_projsl2_c64", honeycomb, (; LKW..., proj_sl = 2, type = ComplexF64))
ldos_case!("ldos_spin_chain_projs_c64", () -> spin_chain(), (; LKW..., proj_s = 1, type = ComplexF64))
ldos_case!("ldos_nambu_chain_projnambu_c64", () -> nambu_chain(), (; LKW..., proj_nambu = 1, type = ComplexF64))
ldos_case!("ldos_bilayer_c64", bilayer, (; LKW..., type = ComplexF64))
ldos_case!("ldos_chain_default_f32", () -> chain(), (; maxdim = 64, cutoff = 1e-6);
    tol = TOL32, cpu_rtol = 1e-4, cpu_atol = 1e-5)

case!("ldos_block_on_1d_throws",
    () -> chain(),
    H -> TB.get_ldos_spatial_gpu(H, 8, OMEGA; reduce = :block, num_x = 2, type = ComplexF64);
    throws = true)

# ═════════════════════════════════════════════════════════════════════════════
# 7. Spatial LDOS, independent MPS recursions
# ═════════════════════════════════════════════════════════════════════════════
function _moments_cpu(H, Ncheb, groups; maxdim, cutoff)
    cols = map(groups) do g
        sum(g) do x
            ψ = TB.physical_site_state(H, x)
            Tn, _, _ = TB.KPM_Tn_mps(H, Ncheb - 1, ψ; maxdim = maxdim, cutoff = cutoff)
            [real(inner(ψ, t)) for t in Tn]
        end ./ length(g)
    end
    return reduce(hcat, cols)
end

case!("ldos_mps_chain_groups_moments_links_c64",
    () -> chain(),
    H -> ((ldos, mom, links) = TB.get_ldos_spatial_mps_gpu(H, 8, OMEGA; x_groups = [[1], [3, 4]],
              maxdim = 32, cutoff = 1e-12, type = ComplexF64, return_moments = true,
              return_maxlinkdim = true);
          (ldos = ldos, moments = mom, linkdims = links)),
    H -> (ldos = TB.get_ldos_spatial(H, 8, OMEGA; mode = :mps, x_groups = [[1], [3, 4]],
                                     maxdim = 32, cutoff = 1e-12),
          moments = _moments_cpu(H, 8, [[1], [3, 4]]; maxdim = 32, cutoff = 1e-12)))

case!("ldos_mps_chain_auto_window_hodc_c64",
    () -> chain(),
    H -> ((ldos, mom) = TB.get_ldos_spatial_mps_gpu(H, 8, OMEGA; num_x = 3, num_avg = 2, x_start = 2,
              x_end = 7, kernel = :hodc, maxdim = 32, cutoff = 1e-12, type = ComplexF64,
              return_moments = true);
          (ldos = ldos, moments = mom)),
    H -> (moments = _moments_cpu(H, 8, TB.spatial_sampling_plan(3; grid = false, num_x = 3,
                                     num_avg = 2, x_start = 2, x_end = 7).groups;
                                 maxdim = 32, cutoff = 1e-12),))

case!("ldos_mps_fibonacci_c64",
    fibonacci,
    H -> (ldos = TB.get_ldos_spatial_mps_gpu(H, 8, collect(range(-0.1, 3.1; length = 5));
              x_groups = [[1, 2], [4], [7, 8]], maxdim = 64, cutoff = 1e-12, type = ComplexF64),),
    H -> (ldos = TB.get_ldos_spatial(H, 8, collect(range(-0.1, 3.1; length = 5)); mode = :mps,
              x_groups = [[1, 2], [4], [7, 8]], maxdim = 64, cutoff = 1e-12),))

case!("ldos_mps_chain_f64_lorentz_links",
    () -> chain(),
    H -> ((ldos, links) = TB.get_ldos_spatial_mps_gpu(H, 8, OMEGA; x_groups = [2, 5], kernel = :lorentz,
              maxdim = 32, cutoff = 1e-12, type = Float64, return_maxlinkdim = true);
          (ldos = ldos, linkdims = links)),
    H -> (ldos = TB.get_ldos_spatial(H, 8, OMEGA; mode = :mps, x_groups = [[2], [5]], kernel = :lorentz,
                                     maxdim = 32, cutoff = 1e-12),))

case!("ldos_mps_chain_default_f32",
    () -> chain(),
    H -> (ldos = TB.get_ldos_spatial_mps_gpu(H, 8, OMEGA; x_groups = [1, 6], cutoff = 1e-6),),
    H -> (ldos = TB.get_ldos_spatial(H, 8, OMEGA; mode = :mps, x_groups = [[1], [6]], cutoff = 1e-6),);
    tol = TOL32, cpu_rtol = 1e-4, cpu_atol = 1e-5)

# ═════════════════════════════════════════════════════════════════════════════
# 8. Stochastic DOS
# ═════════════════════════════════════════════════════════════════════════════
function dos_case!(name, setup, kw; cpu = true, tol = TOL64, cpu_rtol = 1e-7, cpu_atol = 1e-9, Ncheb = 12)
    case!(name, setup,
        H -> (dos = TB.get_dos_stochastic_gpu(H, Ncheb, OMEGA; kw...),),
        cpu ? (H -> (dos = TB.get_dos_stochastic(H, Ncheb, OMEGA;
                         (k => v for (k, v) in pairs(kw) if k ∉ (:type, :dtype))...),)) : nothing;
        tol = tol, cpu_rtol = cpu_rtol, cpu_atol = cpu_atol)
end

const DKW = (maxdim = 32, cutoff = 1e-12)

dos_case!("dos_stoch_chain_trace_c64", () -> chain(), (; DKW..., N_sample = 4, type = ComplexF64))
dos_case!("dos_stoch_chain_sample_normalize_hodc_c64", () -> chain(),
    (; DKW..., N_sample = 3, seed = 7, dos_weighting = :sample, normalize = true, kernel = :hodc,
     type = ComplexF64))
dos_case!("dos_stoch_chain_trace_normalize_globalrng_f64", () -> chain(),
    (; DKW..., N_sample = 3, seed = nothing, normalize = true, type = Float64))
dos_case!("dos_stoch_spin_chain_projs_c64", () -> spin_chain(; L = 2),
    (; DKW..., N_sample = 3, spin_proj = true, proj_s = 1, type = ComplexF64))
dos_case!("dos_stoch_exciton_bound_c64", () -> exciton(),
    (; DKW..., N_sample = 3, N_bound = 2, type = ComplexF64))
dos_case!("dos_stoch_exciton_continuum_normalize_c64", () -> exciton(),
    (; DKW..., N_sample = 3, continuum_only = true, normalize = true, type = ComplexF64); cpu = false)
dos_case!("dos_stoch_exciton_continuum_bound_sample_c64", () -> exciton(),
    (; DKW..., N_sample = 2, N_bound = 2, continuum_only = true, dos_weighting = :sample,
     type = ComplexF64); cpu = false)
dos_case!("dos_stoch_chain_default_f32", () -> chain(), (; N_sample = 3, maxdim = 32, cutoff = 1e-6);
    tol = TOL32, cpu_rtol = 1e-4, cpu_atol = 1e-5)

case!("dos_stoch_continuum_non_exciton_throws",
    () -> chain(),
    H -> TB.get_dos_stochastic_gpu(H, 8, OMEGA; N_sample = 2, continuum_only = true, type = ComplexF64);
    throws = true)

# ═════════════════════════════════════════════════════════════════════════════
# 9. Non-Hermitian DOS (stochastic and deterministic diagonal trace)
# ═════════════════════════════════════════════════════════════════════════════
const NHKW = (maxdim = 64, cutoff = 1e-12)
const ZPTS = ComplexF64[0.2 + 0.1im, -0.5 + 0.3im, 0.0 - 0.2im]

_nh_scalar_cpu(H, z, n; scale, convention = :z_minus_H, block_placement = :post, kw...) =
    real(TB._nh_scalar_online(TB.hermitize(H; z = z, scale = scale, maxdim = 64, cutoff = 1e-12,
                                           convention = convention, block_placement = block_placement), n;
                              scale = scale, maxdim = 64, cutoff = 1e-12, kw...))

_grid_out(r) = (xgrid = r[1], ygrid = r[2], Z = r[3])

# Stochastic probes are drawn differently on CPU and GPU, so the nontrivial cases
# have no pointwise CPU counterpart; the H = c·1 cases do (every probe is exact).
case!("nh_grid_stoch_c64",
    () -> nh_chain(; L = 2),
    H -> _grid_out(TB.get_nh_dos_grid_gpu(H, (-1.0, 1.0), 2, (-0.5, 0.5), 2, 3; scale = 4.0,
                                          n_random = 2, NHKW...)))

case!("nh_grid_stoch_identity_c64",
    () -> nh_identity(),
    H -> _grid_out(TB.get_nh_dos_grid_gpu(H, (-1.0, 1.0), 2, (-0.5, 0.5), 2, 3; scale = 4.0,
                                          n_random = 1, NHKW...)),
    H -> ((x, y, Z) = TB.nh_spectrum_grid(H, (-1.0, 1.0), 2, (-0.5, 0.5), 2, 3; scale = 4.0,
                                          mode = :scalar, NHKW...);
          (xgrid = collect(x), ygrid = collect(y), Z = real.(Z))))

case!("nh_points_stoch_ids_dmrgscale_Hminusz_c64",
    () -> nh_chain(; L = 2),
    H -> (dos = TB.get_nh_dos_points_gpu(H, ZPTS, 3; point_ids = [10, 3, 7], seed = 5, n_random = 2,
                                          convention = :H_minus_z, NHKW...),))

case!("nh_points_stoch_globalrng_pre_c64",
    () -> nh_chain(; L = 2),
    H -> (dos = TB.get_nh_dos_points_gpu(H, ZPTS[1:2], 2; scale = 4.0, seed = nothing, n_random = 2,
                                          block_placement = :pre, NHKW...),))

case!("nh_points_stoch_identity_c64",
    () -> nh_identity(),
    H -> (dos = TB.get_nh_dos_points_gpu(H, ZPTS, 3; scale = 4.0, n_random = 1, NHKW...),),
    H -> (dos = [_nh_scalar_cpu(H, z, 3; scale = 4.0) for z in ZPTS],))

case!("nh_points_ids_length_throws",
    () -> nh_chain(; L = 2),
    H -> TB.get_nh_dos_points_gpu(H, ZPTS, 2; scale = 4.0, point_ids = [1, 2], NHKW...);
    throws = true)

case!("nh_points_diag_c64",
    () -> nh_chain(; L = 2),
    H -> (dos = TB.get_nh_dos_points_diag_trace_gpu(H, ZPTS, 3; scale = 4.0, NHKW...),),
    H -> (dos = [_nh_scalar_cpu(H, z, 3; scale = 4.0) for z in ZPTS],))

case!("nh_points_diag_rows_pre_Hminusz_ids_c64",
    () -> nh_chain(; L = 2),
    H -> (dos = TB.get_nh_dos_points_diag_trace_gpu(H, ZPTS[1:2], 3; scale = 4.0, point_ids = [4, 9],
              source_row = 1, source_col = 2, block_row = 1, block_col = 2, block_placement = :pre,
              convention = :H_minus_z, NHKW...),),
    H -> (dos = [_nh_scalar_cpu(H, z, 3; scale = 4.0, convention = :H_minus_z, block_placement = :pre,
                                source_row = 1, source_col = 2, block_row = 1, block_col = 2)
                 for z in ZPTS[1:2]],))

case!("nh_grid_diag_dmrgscale_c64",
    () -> nh_chain(; L = 2),
    H -> _grid_out(TB.get_nh_dos_grid_diag_trace_gpu(H, (-1.0, 1.0), 3, (-0.4, 0.4), 2, 3; NHKW...)),
    H -> ((x, y, Z) = TB.nh_spectrum_grid(H, (-1.0, 1.0), 3, (-0.4, 0.4), 2, 3; mode = :scalar, NHKW...);
          (xgrid = collect(x), ygrid = collect(y), Z = real.(Z)));
    cpu_rtol = 1e-6, cpu_atol = 1e-8)

case!("nh_grid_diag_bad_n_throws",
    () -> nh_chain(; L = 2),
    H -> TB.get_nh_dos_grid_diag_trace_gpu(H, (-1.0, 1.0), 2, (-0.4, 0.4), 2, 0; scale = 4.0, NHKW...);
    throws = true)

# ═════════════════════════════════════════════════════════════════════════════
# 10. Exciton LDOS and Chebyshev convergence
# ═════════════════════════════════════════════════════════════════════════════
const EKW = (maxdim = 32, cutoff = 1e-12)

function exciton_case!(name, kw, cpu_kw; L = 2, tol = TOL64, Ncheb = 10)
    case!(name, () -> exciton(; L = L),
        H -> (r = TB.get_exciton_ldos_spatial_gpu(H, Ncheb, OMEGA; kw...);
              r isa Tuple ? (ldos = r[1], linkdims = r[2]) : (ldos = r,)),
        cpu_kw === nothing ? nothing :
            (H -> (r = TB.get_exciton_ldos_spatial(H, Ncheb, OMEGA; cpu_kw(H)...);
                   (ldos = r isa Tuple ? r[1] : r,)));
        tol = tol, cpu_rtol = tol === TOL32 ? 1e-4 : 1e-7, cpu_atol = tol === TOL32 ? 1e-5 : 1e-9)
end

exciton_case!("exciton_ldos_xlist_c64", (; EKW..., X_list = [1, 3], type = ComplexF64),
    H -> (; EKW..., X_list = [1, 3]))
exciton_case!("exciton_ldos_Xgroups_hodc_links_c64",
    (; EKW..., X_groups = [[1, 2], [4]], kernel = :hodc, return_maxlinkdim = true, type = ComplexF64),
    H -> (; EKW..., X_groups = [[1, 2], [4]], kernel = :hodc))
exciton_case!("exciton_ldos_auto_1d_navg_c64", (; EKW..., num_x = 2, num_avg = 2, type = ComplexF64),
    H -> (; EKW..., num_x = 2, num_avg = 2))
exciton_case!("exciton_ldos_auto_default_c64", (; EKW..., type = ComplexF64), H -> (; EKW...))
exciton_case!("exciton_ldos_grid_2d_c64", (; EKW..., Lx = 1, num_x = 2, num_y = 2, type = ComplexF64),
    H -> (; EKW..., X_groups = TB.spatial_sampling_plan(H.L; Lx = 1, grid = true, num_x = 2,
                                                        num_y = 2).groups))
exciton_case!("exciton_ldos_xgroups_f64_lorentz", (; EKW..., x_groups = [2, 3], kernel = :lorentz,
                                                    type = Float64),
    H -> (; EKW..., x_groups = [2, 3], kernel = :lorentz))
exciton_case!("exciton_ldos_default_f32", (; X_list = [2], maxdim = 32, cutoff = 1e-6),
    H -> (; X_list = [2], maxdim = 32, cutoff = 1e-6); tol = TOL32)

case!("exciton_ldos_block_throws",
    () -> exciton(),
    H -> TB.get_exciton_ldos_spatial_gpu(H, 6, OMEGA; reduce = :block, type = ComplexF64);
    throws = true)

function _conv_cpu(H, X, Nmax; maxdim, cutoff)
    ψ = TB.mpsexciton(X, H.sites)
    Tn, _, _ = TB.KPM_Tn_mps(H, Nmax, ψ; maxdim = maxdim, cutoff = cutoff)
    return (mu_ref = [real(inner(ψ, Tn[n + 1])) for n in 1:Nmax],
            norm_ref = [norm(Tn[n + 1]) for n in 1:Nmax])
end

_conv_out(r) = (n = r.n, mu_ref = r.mu_ref, mu_test = r.mu_test, delta_mu = r.delta_mu,
                err_fidelity = r.err_fidelity, mdim_ref = r.mdim_ref, mdim_test = r.mdim_test,
                norm_ref = r.norm_ref, norm_test = r.norm_test)

case!("exciton_cheb_conv_single_c64",
    () -> exciton(),
    H -> _conv_out(TB.get_exciton_cheb_convergence_gpu(H, 2, 6; maxdim_test = 2, maxdim_ref = 32,
                                                       cutoff = 1e-12, type = ComplexF64)),
    H -> _conv_cpu(H, 2, 6; maxdim = 32, cutoff = 1e-12))

case!("exciton_cheb_conv_multi_c64",
    () -> exciton(),
    H -> (probes = [_conv_out(r) for r in TB.get_exciton_cheb_convergence_gpu(H, [1, 4], 4;
              maxdim_test = 3, maxdim_ref = 32, cutoff = 1e-12, dtype = ComplexF64)],),
    H -> (probes = [_conv_cpu(H, X, 4; maxdim = 32, cutoff = 1e-12) for X in (1, 4)],))

# ═════════════════════════════════════════════════════════════════════════════
# 11. Real-space Chern marker
# ═════════════════════════════════════════════════════════════════════════════
# chern8 (explicit scale; its default scale is excluded) has no geometry, so the
# coordinates are passed explicitly: x = i mod 4, y = i ÷ 4 for 0-based site i.
chern8() = get_Hamiltonian("chern8", (V = 1.0, t = 1.0); L = 4, scale = 5.0)
honeycomb_nnn() = get_Hamiltonian("honeycomb_nnn", (t = 1.0, t2 = 0.3 + 0.3im); L = 2, scale = 4.0)
const XF8 = (i, _) -> Float64(i % 4)
const YF8 = (i, _) -> Float64(i ÷ 4)

const CKW = (maxdim = 64, cutoff = 1e-10)

function chern_case!(name, setup, xy, kw, ucs; cpu_rtol = 1e-4, cpu_atol = 1e-5)
    case!(name, setup,
        H -> (C = ComplexF64[TB.chern_marker_gpu(H, xy...; kw...)(uc) for uc in ucs],),
        H -> (C = ComplexF64[TB.chern_marker(H, xy...; (k => v for (k, v) in pairs(kw) if k ∉ (:dtype,))...)(uc)
                             for uc in ucs],);
        cpu_rtol = cpu_rtol, cpu_atol = cpu_atol)
end

chern_case!("chern_chern8_mcweeny_c64", chern8, (XF8, YF8),
    (; CKW..., method = :mcweeny, dtype = ComplexF64), [1, 6, 11, 16])
chern_case!("chern_chern8_sp2_flat_c64", chern8, (XF8, YF8),
    (; CKW..., method = :sp2, Nel = 7, quenched = false, dtype = ComplexF64), [2, 7])
chern_case!("chern_chern8_kpm_lambda_c64", chern8, (XF8, YF8),
    (; CKW..., method = :kpm, Ncheb = 30, Lambda = 5, dtype = ComplexF64), [3, 10])
chern_case!("chern_honeycomb_nnn_sublattice_geometry_c64", honeycomb_nnn, (),
    (; CKW..., method = :mcweeny, dtype = ComplexF64), [1, 4])

case!("chern_unknown_method_throws",
    chern8,
    H -> TB.chern_marker_gpu(H, XF8, YF8; method = :bogus, dtype = ComplexF64, CKW...);
    throws = true)

case!("chern_no_geometry_throws",
    chern8,
    H -> TB.chern_marker_gpu(H; dtype = ComplexF64, CKW...);
    throws = true)

# ═════════════════════════════════════════════════════════════════════════════
# 12. Magnetic Hubbard SCF and its post-hoc observables
# ═════════════════════════════════════════════════════════════════════════════
const SKW = (max_scf_iter = 2, purif_maxiter = 20, purif_tol = 1e-4, mix = 0.25, maxdim = 40,
             cutoff = 1e-10, verbose = false)

scf_chain() = (H = chain(; L = 3); add_spin!(H); H.scale = 3.5; H)

_scf_core(res) = (converged = res.converged, iterations = res.iterations, rms_error = res.rms_error,
                  rho_up = ComplexF64.(dense_vector(res.rho_up)), rho_dn = ComplexF64.(dense_vector(res.rho_dn)),
                  density_up = ComplexF64.(dense_matrix(res.density_up_mpo)),
                  density_dn = ComplexF64.(dense_matrix(res.density_dn_mpo)),
                  H_up = ComplexF64.(dense_matrix(res.H_up.mpo)), H_dn = ComplexF64.(dense_matrix(res.H_dn.mpo)),
                  scale_up = res.H_up.scale, history_rms = [h.rms_error for h in res.history],
                  history_particle = [h.particle_error for h in res.history])

_mag_out(m) = (values = m.values, centers = m.centers, groups = groups_int(m.groups), n_up = m.n_up,
               n_dn = m.n_dn, reduce = m.reduce, stride_x = m.stride_x, stride_y = m.stride_y)

function _mag_cpu(res, groups)
    up = real.(diag(dense_matrix(res.density_up_mpo)))
    dn = real.(diag(dense_matrix(res.density_dn_mpo)))
    nu, nd = group_means(up, groups), group_means(dn, groups)
    return (values = (nu .- nd) ./ 2, n_up = nu, n_dn = nd)
end

# One SCF run feeds the magnetization (point groups, point grid) and band checks.
case!("scf_hubbard_local_U_pipeline_c64",
    () -> scf_chain(),
    H -> begin
        res = TB.scf_magnetic_hubbard_gpu(H, 2.0; SKW..., type = ComplexF64)
        m_groups = TB.get_scf_magnetization_gpu(res; x_groups = [[1, 2], [5], [7, 8]])
        m_grid = TB.get_scf_magnetization_gpu(res; num_x = 2, num_y = 2)
        bands = TB.get_scf_bands_gpu(res, 10, OMEGA; num_x = 4, maxdim = 64, cutoff = 1e-12, type = ComplexF64)
        (; _scf_core(res)..., mag_groups = _mag_out(m_groups), mag_grid = _mag_out(m_grid),
           bands_Ak = bands.Ak, bands_omega = bands.omega)
    end,
    H -> begin
        res = TB.scf_magnetic_hubbard(H, 2.0; density_method = :mcweeny, SKW...)
        # Post-hoc observables of the CPU result, evaluated densely / with get_bands.
        plan = TB.spatial_sampling_plan(3; Lx = 1, grid = true, num_x = 2, num_y = 2)
        Ak = TB.get_bands(res.H_up, 10, OMEGA; num_x = 4, maxdim = 64, cutoff = 1e-12) .+
             TB.get_bands(res.H_dn, 10, OMEGA; num_x = 4, maxdim = 64, cutoff = 1e-12)
        (; _scf_core(res)..., mag_groups = _mag_cpu(res, [[1, 2], [5], [7, 8]]),
           mag_grid = _mag_cpu(res, groups_int(plan.groups)), bands_Ak = Ak, bands_omega = OMEGA)
    end;
    cpu_rtol = 1e-4, cpu_atol = 1e-5)

case!("scf_hubbard_U_mpo_initial_up_c64",
    () -> begin
        H = scf_chain()
        pos = filter(s -> s != H.spin_s, H.sites)
        (H = H, U = 1.5 * MPO(pos, "Id"), init = TB.constant_mps(pos, 0.45))
    end,
    inp -> _scf_core(TB.scf_magnetic_hubbard_gpu(inp.H, inp.U; initial_up = inp.init, SKW...,
                                                 dtype = ComplexF64)),
    inp -> _scf_core(TB.scf_magnetic_hubbard(inp.H, inp.U; initial_up = inp.init,
                                             density_method = :mcweeny, SKW...));
    cpu_rtol = 1e-4, cpu_atol = 1e-5)

# A CPU SCF result (no GPU fields) is uploaded as ComplexF32; block reduction.
case!("scf_magnetization_cpu_res_block_f32",
    () -> TB.scf_magnetic_hubbard(scf_chain(), 2.0; density_method = :mcweeny, SKW...),
    res -> _mag_out(TB.get_scf_magnetization_gpu(res; reduce = :block, num_x = 2, num_y = 2)),
    res -> begin
        up = real.(diag(dense_matrix(res.density_up_mpo)))
        dn = real.(diag(dense_matrix(res.density_dn_mpo)))
        nu, nd = block_means_2d(up, 1, 2, 2, 2), block_means_2d(dn, 1, 2, 2, 2)
        (values = (nu .- nd) ./ 2, n_up = nu, n_dn = nd)
    end;
    tol = TOL32, cpu_rtol = 1e-5, cpu_atol = 1e-6)

case!("scf_hubbard_scale_nothing_throws",
    () -> (H = scf_chain(); H.scale = 0.0; H),
    H -> TB.scf_magnetic_hubbard_gpu(H, 2.0; SKW..., type = ComplexF64);
    throws = true)

case!("scf_magnetization_block_xgroups_throws",
    () -> TB.scf_magnetic_hubbard(scf_chain(), 2.0; density_method = :mcweeny, SKW..., max_scf_iter = 1),
    res -> TB.get_scf_magnetization_gpu(res; reduce = :block, x_groups = [[1]]);
    throws = true)

# ── Comparison ────────────────────────────────────────────────────────────────
_isfloatish(T) = T <: AbstractFloat || T <: Complex

"`nothing` when `got` matches `want`, else a String describing the first mismatch."
function mismatch(got, want; rtol, atol, strict = true, path = "")
    if want isa NamedTuple
        got isa NamedTuple || return "$path: expected a NamedTuple, got $(typeof(got))"
        for k in keys(want)
            haskey(got, k) || return "$path.$k: missing"
            m = mismatch(got[k], want[k]; rtol, atol, strict, path = "$path.$k")
            m === nothing || return m
        end
        return nothing
    elseif want isa AbstractArray && eltype(want) <: Number
        got isa AbstractArray || return "$path: expected an array, got $(typeof(got))"
        size(got) == size(want) || return "$path: size $(size(got)) != $(size(want))"
        !strict || eltype(got) == eltype(want) ||
            return "$path: eltype $(eltype(got)) != $(eltype(want))"
        if _isfloatish(eltype(want))
            isempty(want) && return nothing
            isapprox(got, want; rtol, atol, nans = true) && return nothing
            return "$path: max abs diff $(maximum(abs.(got .- want))) (norm $(norm(want)))"
        end
        return got == want ? nothing : "$path: $got != $want"
    elseif want isa AbstractVector || want isa Tuple
        (got isa AbstractVector || got isa Tuple) || return "$path: expected a collection, got $(typeof(got))"
        length(got) == length(want) || return "$path: length $(length(got)) != $(length(want))"
        for i in eachindex(want)
            m = mismatch(got[i], want[i]; rtol, atol, strict, path = "$path[$i]")
            m === nothing || return m
        end
        return nothing
    elseif want isa Number
        !strict || typeof(got) == typeof(want) || return "$path: type $(typeof(got)) != $(typeof(want))"
        _isfloatish(typeof(want)) || return got == want ? nothing : "$path: $got != $want"
        return isapprox(got, want; rtol, atol, nans = true) ? nothing : "$path: $got != $want"
    else
        return isequal(got, want) ? nothing : "$path: $(repr(got)) != $(repr(want))"
    end
end

function check_case(c::Case, entry)
    got = try
        run_gpu(c)
    catch err
        @error "GPU golden case now throws" case = c.name exception = (err, catch_backtrace())
        return false, false
    end
    m = mismatch(got, entry.gpu; rtol = c.rtol, atol = c.atol)
    m === nothing || @error "GPU output changed" case = c.name detail = m
    ok_cpu = true
    if entry.cpu !== nothing
        # CPU counterparts may use other element types (e.g. real profiles): values only.
        mc = mismatch(got, entry.cpu; rtol = c.cpu_rtol, atol = c.cpu_atol, strict = false)
        mc === nothing || @error "GPU output no longer matches the CPU counterpart" case = c.name detail = mc
        ok_cpu = mc === nothing
    end
    return m === nothing, ok_cpu
end

function run_tests(entries; cuda_functional::Bool)
    @testset "GPU outputs are pinned" begin
        names = [e.name for e in entries]
        @test allunique(names)
        @test sort(names) == sort([c.name for c in CASES])
        bycase = Dict(c.name => c for c in CASES)
        for e in entries
            c = get(bycase, e.name, nothing)
            c === nothing && continue
            if c.needs_gpu && !cuda_functional
                continue
            end
            ok_gpu, ok_cpu = check_case(c, e)
            @test ok_gpu
            @test ok_cpu
        end
        cuda_functional || @test_skip "CUDA.jl not functional"
    end
end

end # module GPUGoldenRunner

if !isdefined(@__MODULE__, :GPU_GOLDEN_GENERATOR)
    let cuda_functional = false
        if Base.find_package("CUDA") !== nothing
            try
                @eval using CUDA
                cuda_functional = Base.invokelatest(() -> CUDA.functional())
            catch
            end
        end
        entries = include(joinpath(@__DIR__, "data", "gpu_golden.jl"))
        Base.invokelatest(GPUGoldenRunner.run_tests, entries; cuda_functional = cuda_functional)
    end
end
