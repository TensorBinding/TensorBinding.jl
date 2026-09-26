# ============================================================
# device.jl — CUDA bridge, CPU/GPU transfers, GPU-residency checks
# ============================================================
# Moved from gpu/GPU_tk.jl. The first src/gpu/ file to be included, so it
# also carries the toolkit overview that opened GPU_tk.jl.
#
# ============================================================
# src/gpu/ — GPU production toolkit for TensorBinding
# ============================================================
#
# src/gpu/ is the GPU companion to the CPU solvers in src/solvers/ and
# src/core/Utils.jl.  Its purpose is to make LARGE PRODUCTION RUNS (big L,
# many Chebyshev moments, fine spatial/k grids) tractable by moving the
# dominant Chebyshev-recurrence / MPO-product cost onto a GPU via the
# NDTensors CUDA backend.  Every function here mirrors a CPU counterpart with
# a `_gpu` suffix and (unless its own docstring says otherwise) accepts the
# same keyword arguments and returns the same shape of result.
#
# REQUIREMENTS
#   using TensorBinding
#   using CUDA            # optional (not a package dependency); load it before
#                         # or after TensorBinding, but before the first *_gpu
#                         # call — without it, *_gpu calls raise an error
#                         # explaining how to load it
#   The *_gpu functions are not exported; call them qualified, e.g.
#   TensorBinding.get_bands_gpu(...).
#
# ENTRY POINTS (each documented in its own docstring)
#   Spectral / spatial maps
#     get_bands_gpu(H, Ncheb, ω; kpath=..., ...)          — A(k,ω) bands
#     get_ldos_spatial_gpu(H, Ncheb, ω; reduce=..., ...)  — A(r,ω) real-space LDOS
#                                                            (:point or :block sampling,
#                                                             sublattice :average/:resolve)
#     get_ldos_spatial_mps_gpu(H, Ncheb, ω; ...)          — A(r,ω), independent
#                                                            GPU MPS recursions (including
#                                                            projected position spaces)
#     get_dos_stochastic_gpu(H, Ncheb, ω; ...)            — stochastic-trace DOS
#     get_nh_dos_grid_gpu(H, xlims, nx, ylims, ny, n; ...) — NH stochastic DOS
#     get_nh_dos_points_gpu(H, z_points, n; ...)           — NH stochastic DOS at selected z
#     get_nh_dos_grid_diag_trace_gpu(H, xlims, nx, ylims, ny, n; ...)
#                                                           — NH deterministic diagonal-trace DOS
#     get_nh_dos_points_diag_trace_gpu(H, z_points, n; ...) — NH deterministic DOS at selected z
#     get_nh_density_trajectory_gpu(H, rho0; ...)          — NH density diag vs t
#     get_state_amplitude_trajectory_gpu(H, psi0; ...)     — TDVP state amplitudes vs t
#     get_exciton_ldos_spatial_gpu(H, Ncheb, ω; ...)      — A(X,ω) exciton LDOS
#   Topology
#     get_C_gpu(H, xfunc, yfunc; ...)                     — real-space Chern marker
#   Magnetic Hubbard SCF
#     scf_magnetic_hubbard_gpu(H0, U; ...)                — collinear mean-field loop
#     get_scf_magnetization_gpu(res; ...)                 — post-hoc <Sz>(r) map
#     get_scf_bands_gpu(res, Ncheb, ω; ...)                — post-hoc spin-summed bands
#
# GPU/CPU SPLIT (general pattern — see each function's docstring for specifics)
#   GPU : Chebyshev recurrence (KPM_Tn_gpu / inline recursions), weighted MPO
#         sums (_weighted_mpo_sum_gpu), MPO×MPO and MPO×MPS products +
#         truncation, Hadamard products (_hadamard_mpo_gpu), projections
#         (_project_aux_gpu), diagonal extraction
#         (extract_diagonal_to_mps_gpu) and real-space/QFT sampling.
#   CPU : one-time setup (Hamiltonian/operator construction, Tucker SVDs,
#         k-/spatial-group bookkeeping, KPM weight matrices, McWeeny initial
#         guesses) and the final per-ω scalar accumulation.
#
# PRECISION (WHY F32)
#   The NDTensors GPU backend requires Float32 storage, so every MPO/MPS
#   moved to GPU via _to_gpu_mpo / _to_gpu_mps is first cast to ComplexF32;
#   results moved back via _to_cpu_mpo / _to_cpu_mps are promoted to
#   ComplexF64. This is fine for the observables computed here, but
#   ComplexF32 eigendecomposition can produce NaN at very tight `cutoff` on
#   large systems — functions on this path warn (without altering the value)
#   if `cutoff` is below a recommended floor, typically 1e-4 to 1e-6
#   depending on the routine.
#
# REAL-SPACE / BIT-ORDERING CONVENTIONS
#   Real-space sampling (_eval_mps_bigendian_gpu, _eval_block_mps_gpu, used by
#   get_ldos_spatial_gpu and get_scf_magnetization_gpu) encodes the position
#   index MSB-first across the site list, matching the CPU
#   eval_mps/binary_to_MPS convention exactly. The QFT/bands path
#   (_eval_diag_mps_gpu, used by get_bands_gpu) instead uses the legacy
#   LSB-first convention required by the quantics-Fourier MPO. The two are
#   not interchangeable — see spatial_sampling_plan in Utils.jl for how the
#   shared sampler keeps them straight.
# ============================================================


# ============================================================
# CUDA bridge (no hard dependency)
# ============================================================

const _TB_CUDA = Ref{Union{Module,Nothing}}(nothing)

function _tb_cuda_module()
    if _TB_CUDA[] === nothing
        id = Base.PkgId(Base.UUID("052768ef-5323-5732-b1bb-66c8b64840ba"), "CUDA")
        _TB_CUDA[] = get(Base.loaded_modules, id, nothing)
    end
    return _TB_CUDA[]
end

function _check_gpu(caller::String = "")
    m = _tb_cuda_module()
    if m === nothing
        tag = isempty(caller) ? "" : " (called from $caller)"
        error("""
TensorBinding$tag: GPU functions require CUDA.jl.
Load it before calling any *_gpu function:

    using CUDA

Install once with:  ] add CUDA
""")
    end
end

function _gpu_gc!()
    m = _tb_cuda_module()
    m === nothing && return
    m.synchronize()
    GC.gc(false)
    m.reclaim()
end


# ============================================================
# MPO type conversion
# ============================================================

# CPU F64 → CPU F32  (prerequisite before cu())
function _mpo_to_f32(mpo::MPO)
    return MPO([
        let idx = inds(mpo[i])
            ITensor(ComplexF32.(Array(mpo[i], idx...)), idx)
        end
        for i in 1:length(mpo)
    ])
end

# CPU F64  →  GPU F32
function _to_gpu_mpo(mpo::MPO)
    _check_gpu("_to_gpu_mpo")
    return _tb_cuda_module().cu(_mpo_to_f32(mpo))
end

# CPU MPO → GPU with explicit dtype (complex or real).
# Complex: T.(arr) casts ComplexF64 → T directly.
# Real:    real.(arr) discards zero imaginary parts, then casts to T.
#          Only valid when the MPO is known to be real-valued.
function _to_gpu_mpo(mpo::MPO, T::Type{<:Complex})
    _check_gpu("_to_gpu_mpo")
    cuda = _tb_cuda_module()
    return MPO([
        let idx = inds(mpo[i])
            ITensors.itensor(cuda.CuArray(T.(Array(mpo[i], idx...))), idx...)
        end
        for i in 1:length(mpo)
    ])
end

function _to_gpu_mpo(mpo::MPO, T::Type{<:Real})
    _check_gpu("_to_gpu_mpo")
    cuda = _tb_cuda_module()
    return MPO([
        let idx = inds(mpo[i])
            ITensors.itensor(cuda.CuArray(T.(real.(Array(mpo[i], idx...)))), idx...)
        end
        for i in 1:length(mpo)
    ])
end

# CPU MPS  →  GPU F32 MPS
function _to_gpu_mps(mps::MPS)
    _check_gpu("_to_gpu_mps")
    m      = _tb_cuda_module()
    result = similar(mps)
    for j in 1:length(mps)
        idx    = inds(mps[j])
        arr    = Array(mps[j], idx...)        # CPU: typeassert safe
        result[j] = ITensors.itensor(m.cu(ComplexF32.(arr)), idx...)
    end
    return result
end

function _to_gpu_mps(mps::MPS, T::Type{<:Complex})
    _check_gpu("_to_gpu_mps")
    cuda = _tb_cuda_module()
    result = similar(mps)
    for j in 1:length(mps)
        idx = inds(mps[j])
        arr = Array(mps[j], idx...)
        result[j] = ITensors.itensor(cuda.CuArray(T.(arr)), idx...)
    end
    return result
end

function _to_gpu_mps(mps::MPS, T::Type{<:Real})
    _check_gpu("_to_gpu_mps")
    cuda = _tb_cuda_module()
    result = similar(mps)
    for j in 1:length(mps)
        idx = inds(mps[j])
        arr = Array(mps[j], idx...)
        result[j] = ITensors.itensor(cuda.CuArray(T.(real.(arr))), idx...)
    end
    return result
end

# Resolve the type/dtype kwarg pair into a single GPU element type and emit a
# tight-cutoff NaN warning for 32-bit types. Shared by the Hermitian/KPM GPU
# entry points that accept real OR complex element types (ComplexF32 default;
# ComplexF64 / Float32 / Float64 also valid — real types only for real H).
function _resolve_gpu_type(caller::String, type, dtype, cutoff)
    gpu_type = dtype === nothing ? type : dtype
    dtype !== nothing && dtype != type && type != ComplexF32 &&
        error("$caller: received both type=$type and dtype=$dtype; pass only one datatype keyword.")
    (gpu_type == ComplexF32 || gpu_type == Float32) && cutoff < 1e-6 &&
        @warn "$caller: cutoff=$cutoff with 32-bit $gpu_type may produce NaN on large systems; use a 64-bit dtype or cutoff ≥ 1e-4."
    return gpu_type
end

function _to_cpu_mps(mps::MPS)
    result = similar(mps)
    for j in 1:length(mps)
        T = mps[j]
        s = NDTensors.storage(ITensors.tensor(T))
        arr_cpu = Array(NDTensors.data(s))
        result[j] = ITensors.itensor(ComplexF64.(arr_cpu), inds(T)...)
    end
    return result
end

# GPU F32  →  CPU F64  (called after Hadamard products)
# Array(::ITensor, inds...) typeasserts the result as Array{T,N}, which fails
# for GPU tensors (CuArray ≠ Array).  Go through storage() → raw CuArray →
# Array (bulk copy) → itensor (no typeassert).
function _to_cpu_mpo(mpo::MPO)
    result = similar(mpo)
    for j in 1:length(mpo)
        T       = mpo[j]
        s       = NDTensors.storage(ITensors.tensor(T))  # Dense{F32, CuArray}
        arr_cpu = Array(NDTensors.data(s))               # CuArray → Array{F32,1}
        result[j] = ITensors.itensor(ComplexF64.(arr_cpu), inds(T)...)
    end
    return result
end


# ============================================================
# GPU-residency checks
# ============================================================

function _is_gpu_tensor(T::ITensor)
    storage = NDTensors.storage(ITensors.tensor(T))
    data = try
        NDTensors.data(storage)
    catch
        return false
    end
    return occursin("CuArray", string(typeof(data)))
end

function _ensure_gpu_mpo(W::MPO; caller::String = "_ensure_gpu_mpo")
    flags = [_is_gpu_tensor(W[i]) for i in 1:length(W)]
    all(flags) && return W
    any(flags) && error("$caller: mixed CPU/GPU MPO tensors are not supported.")
    return _to_gpu_mpo(W)
end

# Upload a CPU MPO with an explicit element type; already-GPU MPOs are returned
# untouched (the caller chose their type at upload time).
function _ensure_gpu_mpo(W::MPO, T::Type{<:Number}; caller::String = "_ensure_gpu_mpo")
    flags = [_is_gpu_tensor(W[i]) for i in 1:length(W)]
    all(flags) && return W
    any(flags) && error("$caller: mixed CPU/GPU MPO tensors are not supported.")
    return _to_gpu_mpo(W, T)
end

function _ensure_gpu_mps(ψ::MPS; caller::String = "_ensure_gpu_mps")
    flags = [_is_gpu_tensor(ψ[i]) for i in 1:length(ψ)]
    all(flags) && return ψ
    any(flags) && error("$caller: mixed CPU/GPU MPS tensors are not supported.")
    return _to_gpu_mps(ψ)
end

function _ensure_gpu_mps(ψ::MPS, T::Type{<:Number}; caller::String = "_ensure_gpu_mps")
    flags = [_is_gpu_tensor(ψ[i]) for i in 1:length(ψ)]
    all(flags) && return ψ
    any(flags) && error("$caller: mixed CPU/GPU MPS tensors are not supported.")
    return _to_gpu_mps(ψ, T)
end
