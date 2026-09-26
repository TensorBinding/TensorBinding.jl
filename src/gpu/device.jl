# gpu/device.jl — the CUDA bridge (CUDA.jl is looked up at run time; it is not a
# package dependency), CPU ↔ GPU transfers of MPOs and MPSs with a chosen element
# type, the resolution of the `type`/`dtype` keyword pair, and GPU-residency checks.
# Moved from the former gpu/GPU_tk.jl. This is the first src/gpu/ file included, so
# it also carries the overview of the whole GPU toolkit (below) that opened that file.
#
# Main entry points (internal; every other src/gpu/ file uses them): _check_gpu,
# _gpu_gc!, _to_gpu (the one upload, also the `to_device` hook of the CPU kernels the
# GPU wrappers call), _to_cpu_mpo / _to_cpu_mps, _resolve_gpu_type (with its halves
# _gpu_type and _warn_gpu_cutoff), _ensure_gpu, _gpu_log.
# Depends on: no other file of the package (ITensors/NDTensors only, and CUDA.jl
# found through Base.loaded_modules).
#
# THE src/gpu/ TOOLKIT
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
#   The main *_gpu entry points are exported (see the export list in
#   TensorBinding.jl); the others are called qualified, e.g.
#   TensorBinding.get_nh_dos_grid_diag_trace_gpu(...).
#
# ENTRY POINTS (each documented in its own docstring; defining file in brackets)
#   Chebyshev moments
#     KPM_Tn_gpu(H_mpo, N, sites; ...)                    — T_n(H̃) MPOs  [kpm.jl]
#   Spectral / spatial maps
#     get_bands_gpu(H, Ncheb, ω; kpath=..., ...)          — A(k,ω) bands  [bands.jl]
#     get_ldos_spatial_gpu(H, Ncheb, ω; reduce=..., ...)  — A(r,ω) real-space LDOS
#                                                            (:point or :block sampling,
#                                                             sublattice :average/:resolve)
#                                                            [kpm.jl]
#     get_ldos_spatial_mps_gpu(H, Ncheb, ω; ...)          — A(r,ω), independent
#                                                            GPU MPS recursions (including
#                                                            projected position spaces)
#                                                            [kpm.jl]
#     get_dos_stochastic_gpu(H, Ncheb, ω; ...)            — stochastic-trace DOS  [kpm.jl]
#     get_exciton_ldos_spatial_gpu(H, Ncheb, ω; ...)      — A(X,ω) exciton LDOS  [exciton.jl]
#     get_exciton_cheb_convergence_gpu(H, X, Ncheb_max; ...)
#                                                          — truncation check  [exciton.jl]
#   Non-Hermitian DOS  [nh.jl]
#     get_nh_dos_grid_gpu(H, xlims, nx, ylims, ny, n; ...) — NH stochastic DOS
#     get_nh_dos_points_gpu(H, z_points, n; ...)           — NH stochastic DOS at selected z
#     get_nh_dos_grid_diag_trace_gpu(H, xlims, nx, ylims, ny, n; ...)
#                                                           — NH deterministic diagonal-trace DOS
#     get_nh_dos_points_diag_trace_gpu(H, z_points, n; ...) — NH deterministic DOS at selected z
#   Time evolution  [timeev.jl]
#     get_nh_density_trajectory_gpu(H, rho0; ...)          — NH density diag vs t
#     get_state_amplitude_trajectory_gpu(H, psi0; ...)     — TDVP state amplitudes vs t
#   Topology  [topology.jl]
#     get_C_gpu(H, xfunc, yfunc; ...)                     — real-space Chern marker
#   Magnetic Hubbard SCF  [scf.jl]
#     scf_magnetic_hubbard_gpu(H0, U; ...)                — collinear mean-field loop
#     get_scf_magnetization_gpu(res; ...)                 — post-hoc <Sz>(r) map
#     get_scf_bands_gpu(res, Ncheb, ω; ...)                — post-hoc spin-summed bands
#   Diagonals  [primitives.jl]
#     extract_diagonal_to_mps_gpu(M), density_profile_from_dm_gpu(density_mpo, sites; ...)
#   Internal only: gpu/purification.jl (GPU McWeeny, used by scf.jl) and
#   gpu/conductivity.jl (Tucker/QFT/Hadamard helpers, no caller). What each file
#   calls into is in the source map of src/TensorBinding.jl.
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
#   (_weighted_mpo_sum_gpu and _hadamard_mpo_gpu are in gpu/conductivity.jl.)
#
# PRECISION (ELEMENT TYPES)
#   Every upload is _to_gpu(x, T), with element type T. The entry points take T
#   from their `type` keyword (alias `dtype`), or from `dtype` alone for
#   get_C_gpu and the NH-DOS and time-evolution entry points: ComplexF32 by
#   default (ComplexF64 for the NH-DOS entry points), and a real type only
#   where the docstring allows it, for a real Hamiltonian. A few internal steps
#   upload with ComplexF32 whatever the entry point's type (a CPU MPO passed to
#   density_profile_from_dm_gpu and its constant profile, the SCF Hartree deltas,
#   the identity of _purification_initial_guess_gpu, _mcweeny_purify_gpu, the
#   conductivity helpers). Results moved back via _to_cpu_mpo /
#   _to_cpu_mps are promoted to ComplexF64. ComplexF32 eigendecomposition can
#   produce NaN at very tight `cutoff` on large systems: the entry points warn
#   (without altering the value) for a 32-bit element type with `cutoff < 1e-6`
#   (_resolve_gpu_type; the NH diagonal-trace entry points do not warn).
#
# SHARED KERNELS
#   Where a CPU kernel runs unchanged on GPU tensors, the GPU function is a thin
#   wrapper that calls it with `to_device = _to_gpu`: diagonal extraction, the
#   diagonal MPO of a profile and the block/point evaluators (core/Utils.jl), the
#   stochastic DOS (solvers/kpm/dos.jl), the McWeeny/SP2 iterations and the
#   linear initial guess (physics/Purification.jl), the Chern operator assembly
#   (physics/Topology.jl), the NH block contraction and probe states
#   (core/AuxDOF.jl, physics/nh/kpm.jl). Keywords of those kernels carry the
#   truncations and element types in which the GPU steps differ. The rest is GPU
#   code over the shared recursion chebyshev_foreach: the LDOS/bands accumulations,
#   the NH recurrences, the exciton convergence check and the trajectories.
#
# REAL-SPACE / BIT-ORDERING CONVENTIONS
#   Real-space sampling (_eval_mps_bigendian_gpu, _eval_block_mps_gpu, used by
#   get_ldos_spatial_gpu and get_scf_magnetization_gpu) encodes the position
#   index MSB-first across the site list, matching the CPU
#   eval_mps/binary_to_MPS convention exactly. The QFT/bands path
#   (_eval_diag_mps_gpu, used by get_bands_gpu) instead uses the legacy
#   LSB-first convention required by the quantics-Fourier MPO. The two are
#   not interchangeable — see eval_mps and _eval_diag_mps in core/Utils.jl.


# ============================================================
# 1. CUDA bridge (no hard dependency)
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
# 2. CPU ↔ GPU transfers, the element-type keyword, progress lines
# ============================================================

# The entries of an upload with element type T: a complex T casts them directly; a
# real T drops their imaginary parts first, which is only valid for a tensor known
# to be real-valued.
_gpu_cast(::Type{T}, arr) where {T<:Complex} = T.(arr)
_gpu_cast(::Type{T}, arr) where {T<:Real}    = T.(real.(arr))

"""
    _to_gpu(x, T) -> x on the GPU

Upload a CPU ITensor, MPO or MPS with element type `T` (each tensor copied densely;
an MPO/MPS comes back with fresh orthogonality limits). The only CPU → GPU transfer
of the package, and the `to_device` hook the GPU wrappers pass to the kernels they
share with the CPU (whose CPU default `_on_host(x, T)` returns `x`).
"""
function _to_gpu(t::ITensor, ::Type{T}) where {T<:Number}
    _check_gpu("_to_gpu")
    idx  = inds(t)
    data = _gpu_cast(T, vec(Array(dense(t), idx...)))   # vec: also a 0-dim (scalar) tensor
    return ITensors.itensor(NDTensors.tensor(
        NDTensors.Dense(_tb_cuda_module().CuArray(data)), idx))
end
_to_gpu(W::MPO, ::Type{T}) where {T<:Number} = MPO([_to_gpu(W[i], T) for i in 1:length(W)])
_to_gpu(ψ::MPS, ::Type{T}) where {T<:Number} = MPS([_to_gpu(ψ[i], T) for i in 1:length(ψ)])

# The GPU element type of an entry point: its `type` keyword, or the alias `dtype`
# when given; both at once is an error unless they agree or `type` is left at its
# default ComplexF32. The complex-only entry points (get_C_gpu, the NH-DOS and
# trajectory entry points) have only `dtype` and pass `dtype, nothing`.
function _gpu_type(caller::String, type, dtype)
    gpu_type = dtype === nothing ? type : dtype
    dtype !== nothing && dtype != type && type != ComplexF32 &&
        error("$caller: received both type=$type and dtype=$dtype; pass only one datatype keyword.")
    return gpu_type
end

# The one precision warning of the GPU entry points: a 32-bit element type
# (ComplexF32 or Float32) with `cutoff < below`, where ComplexF32 eigendecompositions
# can produce NaN on large systems. `below` is 1e-6, except for the entry points that
# warned from a looser cutoff before (1e-4 for get_nh_dos_grid_gpu and
# get_nh_dos_points_gpu, 1e-5 for scf_magnetic_hubbard_gpu). The cutoff is used as given.
function _warn_gpu_cutoff(caller::String, gpu_type, cutoff; below::Real = 1e-6)
    (gpu_type == ComplexF32 || gpu_type == Float32) && cutoff < below &&
        @warn "$caller: cutoff=$cutoff with 32-bit $gpu_type may produce NaN on large systems; use a 64-bit dtype or cutoff ≥ 1e-4."
    return nothing
end

# `_gpu_type` then `_warn_gpu_cutoff`: the GPU entry points resolve their element-type
# keyword here (get_bands_gpu checks the type is complex in between,
# get_ldos_spatial_gpu warns just before its recursion, as they did; the NH
# diagonal-trace entry points take `dtype` as given and never warned).
function _resolve_gpu_type(caller::String, type, dtype, cutoff; below::Real = 1e-6)
    gpu_type = _gpu_type(caller, type, dtype)
    _warn_gpu_cutoff(caller, gpu_type, cutoff; below = below)
    return gpu_type
end

# A progress line of the GPU entry points, "[gpu] msg" indented by `indent` steps
# of two spaces; the callers guard it with their `verbose`/`printinfo` flags.
_gpu_log(msg::AbstractString; indent::Integer = 1) = println(" "^(2 * indent), "[gpu] ", msg)

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
# 3. GPU-residency checks
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

# An MPO or MPS on the GPU: returned untouched when all its tensors already are
# (the caller chose their element type at upload time), uploaded with element type
# `T` when none is; a mix of CPU and GPU tensors is an error.
function _ensure_gpu(x::Union{MPO,MPS}, ::Type{T}; caller::String) where {T<:Number}
    flags = [_is_gpu_tensor(x[i]) for i in 1:length(x)]
    all(flags) && return x
    any(flags) && error("$caller: mixed CPU/GPU $(x isa MPO ? "MPO" : "MPS") tensors are not supported.")
    return _to_gpu(x, T)
end
