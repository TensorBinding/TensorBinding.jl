# ============================================================
# GPU_tk.jl — GPU production toolkit for TensorBinding
# ============================================================
#
# This file is the GPU companion to the CPU solvers in src/solvers/ and
# src/core/Utils.jl.  Its purpose is to make LARGE PRODUCTION RUNS (big L,
# many Chebyshev moments, fine spatial/k grids) tractable by moving the
# dominant Chebyshev-recurrence / MPO-product cost onto a GPU via the
# NDTensors CUDA backend.  Every function here mirrors a CPU counterpart with
# a `_gpu` suffix and (unless its own docstring says otherwise) accepts the
# same keyword arguments and returns the same shape of result.
#
# REQUIREMENTS
#   using CUDA            # must be loaded *before* calling any *_gpu function
#   include("TensorBinding.jl"); using .TensorBinding
#
# ENTRY POINTS (each documented in its own docstring below)
#   Spectral / spatial maps
#     get_bands_gpu(H, Ncheb, ω; kpath=..., ...)          — A(k,ω) bands
#     get_ldos_spatial_gpu(H, Ncheb, ω; reduce=..., ...)  — A(r,ω) real-space LDOS
#                                                            (:point or :block sampling,
#                                                             sublattice :average/:resolve)
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


# ── 1. CUDA bridge (no hard dependency) ─────────────────────────────────────

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


# ── 2. MPO type conversion ───────────────────────────────────────────────────

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


# Build the two QFT operators for the given Hamiltonian and move them to GPU F32.
# Call once before the Tucker pairs loop so the build cost is amortised across
# all r_m × r_n pairs.
function _build_qft_ops_gpu(H::TBHamiltonian)
    pos_s = _pos_sites(H)
    R     = length(pos_s)
    FTirev_cpu = _embed_in_full_sites(H, fix_sites(
        MPO(TCI.reverse(QuanticsTCI.quanticsfouriermpo(R; sign=-1.0, normalize=true))), pos_s))
    FTrev_cpu  = _embed_in_full_sites(H, fix_sites(
        MPO(TCI.reverse(QuanticsTCI.quanticsfouriermpo(R; sign=+1.0, normalize=true))), pos_s))
    return _to_gpu_mpo(FTirev_cpu), _to_gpu_mpo(FTrev_cpu)
end

# Apply the QFT sandwich U·W·U† on GPU using pre-built GPU QFT operators.
# Returns a GPU F32 MPO.
function _apply_qft_conj_gpu(W::MPO, FTirev_gpu::MPO, FTrev_gpu::MPO;
                              tol::Real = 1e-9, maxdim::Int = 100)
    Op1 = apply(W, FTirev_gpu; cutoff=tol, maxdim=maxdim)
    Op2 = apply(swapprime(FTrev_gpu, 0 => 1), Op1; cutoff=tol, maxdim=maxdim)
    return ITensorMPS.truncate!(Op2; cutoff=tol, maxdim=maxdim)
end

# ── 3. GPU-safe primitives ───────────────────────────────────────────────────

# delta() produces a DiagTensor{Float64} (CPU).  When contracted with a
# Dense{ComplexF32} GPU tensor, NDTensors promotes the output to ComplexF64
# and the _contract! dispatch fails (all tensors in these contractions should
# share the GPU ComplexF32 element type).
# Fix: materialise the delta as a dense ComplexF32 GPU tensor.
function _make_delta_gpu(i::Index, j::Index, k::Index)
    d_dense = dense(delta(i, j, k))          # DiagStorage → DenseStorage
    idx     = inds(d_dense)
    arr     = Array(d_dense, idx...)
    return _tb_cuda_module().cu(ITensor(ComplexF32.(arr), idx))
end

function _onehot_gpu(p::Pair{<:Index,<:Integer}, T::Type{<:Complex}=ComplexF32)
    i = p.first
    v = Int(p.second)
    1 <= v <= dim(i) || error("_onehot_gpu: state $v is outside index dimension $(dim(i)).")
    arr = zeros(T, dim(i))
    arr[v] = one(T)
    return ITensors.itensor(_tb_cuda_module().CuArray(arr), i)
end

_onehot_gpu_f32(p::Pair{<:Index,<:Integer}) = _onehot_gpu(p, ComplexF32)

# GPU-safe Hadamard product: identical logic to _hadamard_mpo but uses
# _make_delta_gpu so all contractions stay within ComplexF32 on GPU.
function _hadamard_mpo_gpu(A::MPO, B::MPO, out_sites::Vector{<:Index};
                           maxdim::Int = typemax(Int), cutoff::Real = 0.0)
    L      = length(A)
    @assert length(B) == L && length(out_sites) == L
    sindsA = siteinds(A)
    sindsB = siteinds(B)

    links_B_old = Vector{Index}(undef, max(L - 1, 0))
    links_B_new = Vector{Index}(undef, max(L - 1, 0))
    for b in 1:L-1
        lB = only(commoninds(B[b], B[b+1]))
        links_B_old[b] = lB
        links_B_new[b] = sim(lB)
    end

    tens = Vector{ITensor}(undef, L)
    for n in 1:L
        bra_A, ket_A = _bra_ket(sindsA[n])
        bra_B, ket_B = _bra_ket(sindsB[n])
        bra_out = prime(out_sites[n])
        ket_out = out_sites[n]
        bra_B_f = sim(bra_B)
        ket_B_f = sim(ket_B)
        old_inds = Index[bra_B, ket_B]
        new_inds = Index[bra_B_f, ket_B_f]
        n > 1 && push!(old_inds, links_B_old[n-1]); n > 1 && push!(new_inds, links_B_new[n-1])
        n < L && push!(old_inds, links_B_old[n]);   n < L && push!(new_inds, links_B_new[n])
        B_n = replaceinds(B[n], old_inds, new_inds)
        # Contract delta tensors into A *before* multiplying B_n to avoid an
        # 8D intermediate. Old order: (A*B)→8D→*δ→6D→*δ→5D.
        # New order: (A*δ_bra*δ_ket)→6D→*B_n→6D.
        # The 8D path overflows int32 CUDA indexing for maxdim ≳ 115
        # (16·χ⁴ > 2³¹ when χ > ~115), causing ERROR_ILLEGAL_ADDRESS.
        W   = A[n] * _make_delta_gpu(bra_A, bra_B_f, bra_out)  # 4D→5D
        W   = W    * _make_delta_gpu(ket_A, ket_B_f, ket_out)  # 5D→6D
        W   = W    * B_n                                        # 6D→6D
        tens[n] = W
    end

    if L == 1
        mpo = MPO(tens)
        (maxdim < typemax(Int) || cutoff > 0.0) && ITensorMPS.truncate!(mpo; maxdim=maxdim, cutoff=cutoff)
        return mpo
    end
    Cs = Vector{ITensor}(undef, L - 1)
    for b in 1:L-1
        lA     = only(commoninds(A[b], A[b+1]))
        lB     = links_B_new[b]
        Cs[b]  = combiner(lA, lB; tags="Link,l=$b")
    end
    tens[1] = tens[1] * Cs[1]
    for n in 2:L-1
        tens[n] = tens[n] * Cs[n-1] * Cs[n]
    end
    tens[L] = tens[L] * Cs[L-1]
    mpo = MPO(tens)
    (maxdim < typemax(Int) || cutoff > 0.0) && ITensorMPS.truncate!(mpo; maxdim=maxdim, cutoff=cutoff)
    return mpo
end

# Evaluate an MPS element at bit-index `idx` entirely on GPU using the legacy
# LSB-first convention used by the GPU QFT/bands path.
#
# Basis vectors are built as explicit dense arrays matching the element type of
# A so that the contraction is GPU×GPU with a consistent dtype throughout.
function _eval_diag_mps_gpu(A::MPS, idx::Int)
    cuda = _tb_cuda_module()
    s    = siteinds(A)
    ElT  = eltype(A[1])
    acc  = cuda.cu(ITensor(one(ElT)))
    for i in 1:length(s)
        b     = (idx >> (i - 1)) & 1
        v_arr = zeros(ElT, dim(s[i]))
        v_arr[b + 1] = one(real(ElT))
        v   = cuda.cu(ITensor(v_arr, s[i]))
        acc *= A[i] * v
    end
    return real(scalar(acc))
end

# Real-space MPS element evaluation on GPU, matching binary_to_MPS/eval_mps:
# `idx` is encoded big-endian across the site order.
function _eval_mps_bigendian_gpu(A::MPS, idx::Int)
    cuda = _tb_cuda_module()
    s    = siteinds(A)
    ElT  = eltype(A[1])
    n    = length(s)
    acc  = cuda.cu(ITensor(one(ElT)))
    for i in 1:n
        b     = (idx >> (n - i)) & 1
        v_arr = zeros(ElT, dim(s[i]))
        v_arr[b + 1] = one(real(ElT))
        v   = cuda.cu(ITensor(v_arr, s[i]))
        acc *= A[i] * v
    end
    return real(scalar(acc))
end

function _eval_mps_bigendian_complex_gpu(A::MPS, idx::Int)
    cuda = _tb_cuda_module()
    s    = siteinds(A)
    ElT  = eltype(A[1])
    n    = length(s)
    acc  = cuda.cu(ITensor(one(ElT)))
    for i in 1:n
        b     = (idx >> (n - i)) & 1
        v_arr = zeros(ElT, dim(s[i]))
        v_arr[b + 1] = one(real(ElT))
        v   = cuda.cu(ITensor(v_arr, s[i]))
        acc *= A[i] * v
    end
    return ComplexF64(scalar(acc))
end

# Block-integrated MPS element on GPU (reduce=:block): sum the profile over one
# coarse block by tracing out the within-block position bits and pinning the
# block to the coarse pixel (ixp, iyp).  The big-endian position site order is
# [iy_MSB..iy_LSB, ix_MSB..ix_LSB] (sites 1..Ly carry iy, Ly+1..L carry ix), so
# we keep the top b bits of iy (sites 1..b) and top a bits of ix (sites
# Ly+1..Ly+a) as onehot, and contract every lower bit with [1,1] (a sum).
function _eval_block_mps_gpu(A::MPS, ixp::Int, iyp::Int,
                             a::Int, b::Int, Lx::Int, Ly::Int)
    cuda = _tb_cuda_module()
    s    = siteinds(A)
    ElT  = eltype(A[1])
    L    = Lx + Ly
    acc  = cuda.cu(ITensor(one(ElT)))
    for i in 1:L
        v_arr = zeros(ElT, dim(s[i]))
        if i <= b                       # keep: iy block bit (b - i)
            v_arr[((iyp >> (b - i)) & 1) + 1] = one(real(ElT))
        elseif i <= Ly                  # sum: iy within-block bit
            v_arr .= one(real(ElT))
        elseif i <= Ly + a              # keep: ix block bit (a - (i - Ly))
            v_arr[((ixp >> (a - (i - Ly))) & 1) + 1] = one(real(ElT))
        else                            # sum: ix within-block bit
            v_arr .= one(real(ElT))
        end
        v   = cuda.cu(ITensor(v_arr, s[i]))
        acc *= A[i] * v
    end
    return real(scalar(acc))
end

# 1D block-integrated MPS element on GPU. Pins the top `a` big-endian bits to
# the coarse block index `ixp` and traces the remaining lower bits with [1, 1].
function _eval_block_mps_1d_gpu(A::MPS, ixp::Int, a::Int, L::Int)
    cuda = _tb_cuda_module()
    s    = siteinds(A)
    ElT  = eltype(A[1])
    length(s) == L || error("_eval_block_mps_1d_gpu: MPS has $(length(s)) sites but L=$L.")
    acc  = cuda.cu(ITensor(one(ElT)))
    for i in 1:L
        v_arr = zeros(ElT, dim(s[i]))
        if i <= a
            v_arr[((ixp >> (a - i)) & 1) + 1] = one(real(ElT))
        else
            v_arr .= one(real(ElT))
        end
        v = cuda.cu(ITensor(v_arr, s[i]))
        acc *= A[i] * v
    end
    return real(scalar(acc))
end

function _eval_block_mps_1d_complex_gpu(A::MPS, ixp::Int, a::Int, L::Int)
    cuda = _tb_cuda_module()
    s    = siteinds(A)
    ElT  = eltype(A[1])
    length(s) == L || error("_eval_block_mps_1d_complex_gpu: MPS has $(length(s)) sites but L=$L.")
    acc  = cuda.cu(ITensor(one(ElT)))
    for i in 1:L
        v_arr = zeros(ElT, dim(s[i]))
        if i <= a
            v_arr[((ixp >> (a - i)) & 1) + 1] = one(real(ElT))
        else
            v_arr .= one(real(ElT))
        end
        v = cuda.cu(ITensor(v_arr, s[i]))
        acc *= A[i] * v
    end
    return ComplexF64(scalar(acc))
end

"""
    extract_diagonal_to_mps_gpu(M::MPO) -> MPS

GPU-resident analogue of `extract_diagonal_to_mps`. `M` is expected to already
be a GPU MPO. The returned MPS stays on GPU, with one-hot tensors matched to
the input tensor element type.
"""
# extract_diagonal_to_mps (in RPA_tk.jl) uses plain onehot() which returns a
# CPU DiagBlockSparse tensor.  Contracting a GPU MPO tensor with a CPU onehot
# fails (GPU×CPU mismatch). Here the one-hot basis vectors are explicitly dense
# GPU tensors with the same element type as the input MPO tensor.
function extract_diagonal_to_mps_gpu(M::MPO)::MPS
    N    = length(M)
    new_tensors = Vector{ITensor}(undef, N)
    for i in 1:N
        t      = M[i]
        s2, s1 = siteinds(M, i)   # s2 = bra (primed), s1 = ket
        d_s    = dim(s1)
        ElT    = eltype(t)
        v_inds = uniqueinds(t, s1, s2)

        res = ITensor(v_inds..., s1)   # zero tensor; type determined by first +=
        for v in 1:d_s
            ket_v = _onehot_gpu(s1 => v, ElT)
            bra_v = _onehot_gpu(s2 => v, ElT)
            slice = t * ket_v * bra_v
            res  += slice * ket_v
        end
        new_tensors[i] = res
    end
    return MPS(new_tensors)
end

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

"""
    density_profile_from_dm_gpu(density_mpo, sites=nothing; mode=:direct) -> MPS

GPU-resident analogue of `density_profile_from_dm`. If `density_mpo` is a CPU
MPO it is uploaded once; if it is already on GPU it is used in place. The
returned profile is a GPU MPS. `mode=:complement` returns `1 - diag(D)` on GPU.
"""
function density_profile_from_dm_gpu(density_mpo::MPO, sites=nothing;
                                     mode::Symbol = :direct,
                                     maxdim::Int = 100,
                                     cutoff::Real = 1e-8)
    _check_gpu("density_profile_from_dm_gpu")
    dm_gpu = _ensure_gpu_mpo(density_mpo; caller="density_profile_from_dm_gpu")
    diag_mps = extract_diagonal_to_mps_gpu(dm_gpu)
    mode === :direct && return diag_mps
    if mode === :complement
        profile_sites = sites === nothing ? collect(siteinds(diag_mps)) : collect(sites)
        one_mps = _to_gpu_mps(constant_mps(profile_sites, 1.0))
        return +(one_mps, -diag_mps; maxdim=maxdim, cutoff=cutoff)
    end
    error("Unsupported density extraction mode :$mode. Use :direct or :complement.")
end

function _nh_von_neumann_rhs_gpu(H_gpu::MPO, Hdag_gpu::MPO, rho_gpu::MPO;
                                 maxdim::Int, cutoff::Real)
    ak = (cutoff=Float64(cutoff), maxdim=maxdim)
    Hrho    = apply(H_gpu, rho_gpu; ak...)
    rhoHdag = apply(rho_gpu, Hdag_gpu; ak...)
    diff = +(Hrho, ComplexF32(-1) * rhoHdag; ak...)
    ITensorMPS.truncate!(diff; cutoff=Float64(cutoff), maxdim=maxdim)
    return ComplexF32(0, -1) * diff
end

function rk4_step_dm_nh_gpu(H_gpu::MPO, Hdag_gpu::MPO, rho_gpu::MPO, dt::Real;
                            maxdim::Int = 200,
                            cutoff::Real = 1e-8,
                            truncate_intermediates::Bool = true)
    ak = (cutoff=Float64(cutoff), maxdim=maxdim)
    halfdt = ComplexF32(Float32(dt / 2))
    dt_gpu = ComplexF32(Float32(dt))
    dt6    = ComplexF32(Float32(dt / 6))
    two    = ComplexF32(2)

    k1 = _nh_von_neumann_rhs_gpu(H_gpu, Hdag_gpu, rho_gpu; maxdim=maxdim, cutoff=cutoff)

    rho2 = +(rho_gpu, halfdt * k1; ak...)
    truncate_intermediates && ITensorMPS.truncate!(rho2; cutoff=Float64(cutoff), maxdim=maxdim)
    k2 = _nh_von_neumann_rhs_gpu(H_gpu, Hdag_gpu, rho2; maxdim=maxdim, cutoff=cutoff)

    rho3 = +(rho_gpu, halfdt * k2; ak...)
    truncate_intermediates && ITensorMPS.truncate!(rho3; cutoff=Float64(cutoff), maxdim=maxdim)
    k3 = _nh_von_neumann_rhs_gpu(H_gpu, Hdag_gpu, rho3; maxdim=maxdim, cutoff=cutoff)

    rho4 = +(rho_gpu, dt_gpu * k3; ak...)
    truncate_intermediates && ITensorMPS.truncate!(rho4; cutoff=Float64(cutoff), maxdim=maxdim)
    k4 = _nh_von_neumann_rhs_gpu(H_gpu, Hdag_gpu, rho4; maxdim=maxdim, cutoff=cutoff)

    ksum = +(k1, two * k2; ak...)
    ksum = +(ksum, two * k3; ak...)
    ksum = +(ksum, k4; ak...)
    ITensorMPS.truncate!(ksum; cutoff=Float64(cutoff), maxdim=maxdim)

    rho_new = +(rho_gpu, dt6 * ksum; ak...)
    ITensorMPS.truncate!(rho_new; cutoff=Float64(cutoff), maxdim=maxdim)
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
    get_nh_density_trajectory_gpu(H, rho0; nsteps, dt, sample_every, ...)
        -> (density, times, centers, groups, maxlinkdims)

GPU RK4 evolution of a density matrix under a static non-Hermitian Hamiltonian
using `d rho/dt = -i(H rho - rho Hdagger)`. `H` may be a `TBHamiltonian` or an
MPO; `rho0` may be a CPU or GPU MPO. The Hamiltonian and density matrix are
uploaded once and the RK4 loop stays on GPU. At every sampled time, the diagonal
of `rho(t)` is extracted on GPU and only scalar values at the requested groups
are copied back to CPU.

Sampling follows the 1D `spatial_sampling_plan` convention: use `num_x=0` to
sample all sites, or set `num_x` to a smaller number for coarse production
output. `num_avg > 1` averages a few sub-points per sampled spatial bin in
`:point` mode. With `reduce=:block`, `num_x` must be a power of two and each
output value is the GPU block average over a contiguous interval of size
`2^L / num_x`.
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
    get_state_amplitude_trajectory_gpu(H, psi0; nsteps, dt, sample_every, ...)
        -> (amplitude, times, centers, groups, norms, maxlinkdims)

GPU TDVP evolution of a single-particle MPS state under the physical
Hamiltonian `H`. `ITensorMPS.tdvp(operator, t, init)` computes
`exp(t * operator) * init` (generator form), so the operator passed to `tdvp`
is `-im * H`, which implements the Schrödinger evolution `dψ/dt = -im * Hψ`;
therefore a loss term `-im * Γ` (Γ >= 0) damps the norm, matching
`evolve_with_tdvp(H::TBHamiltonian,...)` on CPU and the NH RK4 convention
`dρ/dt = -i(Hρ - ρH†)`. `H` may be a `TBHamiltonian` or an MPO. The Hamiltonian
and initial state are uploaded once, the TDVP loop stays on GPU, and only
sampled scalar amplitudes are copied back to CPU.

The returned `amplitude` matrix has rows = sampled positions and columns =
sampled times. By default it stores `real(<x|psi(t)>)`, matching panel (d) of
`APSOS_NH_testing`. Set `component=:imag`, `:abs`, or `:probability` if needed.

Sampling follows `spatial_sampling_plan` in 1D. `reduce=:point` samples
representative positions or explicit groups. `reduce=:block` returns the
block-averaged complex amplitude over contiguous intervals.
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
        ψ_gpu = tdvp(
            generator_gpu,
            dt,
            ψ_gpu;
            time_step=dt,
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
                component=component)
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

get_nh_state_trajectory_gpu(H, psi0::MPS; kwargs...) =
    get_state_amplitude_trajectory_gpu(H, psi0; kwargs...)

function _mps_to_diagonal_mpo_gpu(mps::MPS, sites)::MPO
    N = length(mps)
    mpo_tensors = Vector{ITensor}(undef, N)
    for i in 1:N
        mps_t = mps[i]
        old_s = if N == 1
            only(siteinds(mps))
        elseif i == 1
            uniqueind(mps_t, mps[i+1])
        elseif i == N
            uniqueind(mps_t, mps[i-1])
        else
            uniqueind(mps_t, mps[i-1], mps[i+1])
        end
        s = sites[i]
        s_temp = Index(dim(s), "temp")
        mpo_tensors[i] = replaceind(mps_t, old_s => s_temp) *
                         _make_delta_gpu(s_temp, s, s')
    end
    return MPO(mpo_tensors)
end

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

# Project one auxiliary site out of a GPU MPO.
# Mirrors project_aux (CPU) but builds a dense ComplexF32 projector on GPU so
# every contraction stays on the GPU.  The contracted site is absorbed into the
# neighbouring site, returning an MPO with one fewer site.
#
# setelt() produces a DiagBlockSparse ITensor that cu() leaves on CPU — we
# therefore build the |sec><sec| projector as an explicit dense array instead.
function _project_aux_gpu(T::MPO, idx::Index, sec::Int; side::Symbol=:post)
    cuda = _tb_cuda_module()
    L    = length(T)
    n    = side == :post ? L : 1

    # Build dense |sec><sec| projector matching the element type of T so the
    # contraction is GPU×GPU with a consistent dtype throughout.
    ElT        = eltype(T[n])
    d          = dim(idx)
    proj_arr   = zeros(ElT, d, d)
    proj_arr[sec, sec] = one(ElT)
    proj       = cuda.cu(ITensor(proj_arr, idx, idx'))
    contracted = T[n] * proj   # removes idx & idx' from T[n]; leaves bond indices only

    L == 1 && return MPO([contracted])

    if side == :post
        absorbed = T[L-1] * contracted   # merge the dangling bond into site L-1
        return MPO(vcat([T[i] for i in 1:L-2], [absorbed]))
    else  # :pre
        absorbed = contracted * T[2]     # merge the dangling bond into site 2
        return MPO(vcat([absorbed], [T[i] for i in 3:L]))
    end
end

# Weighted MPO sum.  Accepts Vector{Union{MPO,Nothing}} so that callers can
# pass a sparse Tn_list produced by KPM_Tn_gpu with keep_indices set.
# nothing entries (inactive Tns) are silently skipped.
function _weighted_mpo_sum_gpu(weights::AbstractVector{<:Number},
                               mpos::AbstractVector;
                               maxdim::Int, cutoff::Real, weight_tol::Real = 1e-14)
    result = nothing
    for (w, mpo) in zip(weights, mpos)
        (abs(w) < weight_tol || isnothing(mpo)) && continue
        et = eltype(mpo[1])
        wc = convert(et <: Complex ? et : complex(et), w)
        if result === nothing
            result = wc * mpo
        else
            result = ITensorMPS.truncate!(+(result, wc * mpo; maxdim=maxdim); cutoff=cutoff)
        end
    end
    return result
end

# Density matrix: purify on CPU (expensive but complex-type safe), then
# convert to GPU F32 for use in the Tucker bubble pipeline.
function _get_density_matrix_gpu(H::TBHamiltonian, ϵF::Real,
                                  P_method::Symbol, Ncheb::Int,
                                  maxdim::Int, cutoff::Real,
                                  purify_method::Symbol, purify_maxdim::Int,
                                  purify_maxiters::Int, purify_tol::Float64,
                                  verbose::Bool)
    P = _get_density_matrix(H, ϵF, P_method, Ncheb, maxdim, cutoff,
                             purify_method, purify_maxdim, purify_maxiters,
                             purify_tol, verbose)
    return _to_gpu_mpo(P)
end


# ── 4. GPU Chebyshev recurrence ──────────────────────────────────────────────

"""
    KPM_Tn_gpu(H_mpo, N, sites; scale, center, maxdim, cutoff, keep_indices, verbose)
        -> (Tn_list, scale, center)

GPU version of `KPM_Tn`.  Moves the identity and scaled Hamiltonian MPOs to
GPU (ComplexF32) before the recurrence so all Tn tensors stay on GPU.
Requires `using CUDA`.

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
    Ham_n = (1 / scale) * +(H_mpo, (-center) * I_mpo; cutoff = cutoff)

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


# ── 5. Shared Tucker component builder ──────────────────────────────────────

# Computes C_tuck, B_tuck, A_tuck, E_tuck fully on GPU.
# All inputs (Tn1, Tn2, P1_gpu, P2_gpu) are expected to be GPU F32 MPOs.
function _build_tucker_components_gpu(Tn1, Tn2, P1_gpu, P2_gpu;
                                      U_m, V_n, r_m, r_n,
                                      maxdim, cutoff)
    C_tuck = [_weighted_mpo_sum_gpu(U_m[:, s1], Tn1; maxdim=maxdim, cutoff=cutoff)
              for s1 in 1:r_m]
    B_tuck = [_weighted_mpo_sum_gpu(conj.(V_n[:, s2]), Tn2; maxdim=maxdim, cutoff=cutoff)
              for s2 in 1:r_n]
    A_tuck = [isnothing(C_tuck[s1]) ? nothing :
              ITensorMPS.truncate!(apply(C_tuck[s1], P1_gpu; maxdim=maxdim, cutoff=cutoff); cutoff=cutoff)
              for s1 in 1:r_m]
    E_tuck = [isnothing(B_tuck[s2]) ? nothing :
              ITensorMPS.truncate!(apply(B_tuck[s2], P2_gpu; maxdim=maxdim, cutoff=cutoff); cutoff=cutoff)
              for s2 in 1:r_n]
    return C_tuck, B_tuck, A_tuck, E_tuck
end


# ── 6. GPU entry points ──────────────────────────────────────────────────────

"""
    get_bands_gpu(H, Ncheb, ω_phys_vals; kwargs...)
        -> Matrix{Float64}  or  NamedTuple(Ak, ticks, labels)

GPU-accelerated version of `get_bands`.

GPU handles: the full Chebyshev MPO recurrence (the dominant cost) and the
             QFT sandwich applied to each Chebyshev moment.
CPU handles: k-group setup, KPM weight matrix, final scalar accumulation.

Use `type=ComplexF32` or `type=ComplexF64` to choose the GPU tensor datatype.
`dtype=...` is accepted as an alias for consistency with the non-Hermitian GPU
entry points. ComplexF32 is faster, while ComplexF64 is safer at tight cutoffs
on large systems.

All keyword arguments are identical to the TBHamiltonian overload of
`get_bands`.  The return value is also identical: a plain `Matrix{Float64}`
when no `kpath` is given, or a `NamedTuple(Ak, ticks, labels)` when a
high-symmetry path is requested.

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

    _check_gpu("get_bands_gpu")
    gpu_type = dtype === nothing ? type : dtype
    dtype !== nothing && dtype != type && type != ComplexF32 &&
        error("get_bands_gpu: received both type=$type and dtype=$dtype; pass only one datatype keyword.")
    gpu_type == ComplexF32 && cutoff < 1e-6 &&
        @warn "get_bands_gpu: cutoff=$cutoff with ComplexF32 may produce NaN on large systems; use type=ComplexF64 or cutoff ≥ 1e-4."

    _ensure_scale!(H)
    nambu_proj, spin_proj, layer_proj, sublat_proj =
        _autoenable_proj(H, nambu_proj, spin_proj, layer_proj, sublat_proj)

    ω_resc = (collect(ω_phys_vals) .- H.center) ./ H.scale
    Nω     = length(ω_resc)
    valid  = [abs(ω) < 1.0 for ω in ω_resc]
    W_kpm  = _kpm_weight_matrix(Ncheb, ω_resc; kernel=kernel, lambda=lambda)

    # ── Auto-detect aux indices (mirrors the CPU TBHamiltonian overload) ────
    nambu_s_det, nambu_side_det = !isnothing(H.nambu_s) ?
        aux_site(H, :nambu) : (nothing, :pre)
    spin_s_det = H.spin_s
    layer_s_det, layer_side_det = !isnothing(H.layer_s) ?
        aux_site(H, :layer) : (nothing, :pre)
    sublat_s_det, sublat_side_det = !isnothing(H.sublattice_s) ?
        aux_site(H, :sublattice) : (nothing, :post)

    # ── L_pos: position qubits only (excluding aux sites) ───────────────────
    isnothing(H.geometry) && error("get_bands_gpu: H.geometry must be set (needed to infer D).")
    D     = length(H.geometry(1))
    L     = H.L
    L_pos = L - (spin_proj ? 1 : 0) - (!isnothing(nambu_s_det)  ? 1 : 0) -
                (!isnothing(layer_s_det)  ? 1 : 0) - (!isnothing(sublat_s_det) ? 1 : 0)

    # ── k-path shortcut ──────────────────────────────────────────────────────
    kpath_ticks = nothing; kpath_labels = nothing
    if !isnothing(kpath)
        isnothing(kpath_lattice) && error("get_bands_gpu: kpath requires kpath_lattice.")
        Lx_kp = isnothing(kpath_Lx) ? H.L ÷ 2 : Int(kpath_Lx)
        Ly_kp = H.L - Lx_kp
        k_groups_override, kpath_ticks, kpath_labels =
            kpath_setup(kpath_lattice, Lx_kp, Ly_kp, kpath; npts_per_segment=num_x)
    end

    # ── k-groups (same logic as low-level CPU get_bands) ────────────────────
    Lx_pos = D == 2 ? div(L_pos, 2) : 0
    N_pos  = 2^L_pos
    if !isnothing(k_groups_override)
        k_groups = k_groups_override
        num_x    = length(k_groups)
    elseif D == 1
        _xmax     = xmax === nothing ? N_pos - 1 : Int(xmax)
        xcenters  = ilinspace(xmin, _xmax, num_x)
        half_step = num_x > 1 ? (_xmax - xmin) / (2 * num_x) : 0
        offsets   = num_avg > 1 ? round.(Int, range(-half_step, half_step; length=num_avg)) : Int[0]
        k_groups  = [clamp.(xcenters[i] .+ offsets, 0, N_pos - 1) for i in 1:num_x]
    elseif D == 2
        Nx_loc = 2^Lx_pos; Ny_loc = 2^(L_pos - Lx_pos)
        num_x  = min(num_x, Nx_loc)
        _xmax  = xmax === nothing ? Nx_loc - 1 : Int(xmax)
        _ymax  = ymax === nothing ? Ny_loc - 1 : Int(ymax)
        xcenters = ilinspace(xmin, _xmax, Nx_loc)
        ycenters = ilinspace(ymin, _ymax, Ny_loc)
        hsx = num_x > 1 ? (_xmax - xmin) / (2 * num_x) : 0
        hsy = num_y > 1 ? (_ymax - ymin) / (2 * num_y) : 0
        x_offs = num_avg > 1 ? round.(Int, range(-hsx, hsx; length=num_avg)) : Int[0]
        y_offs = num_avg > 1 ? round.(Int, range(-hsy, hsy; length=num_avg)) : Int[0]
        k_groups = [
            begin
                xs = clamp.(xcenters[i] .+ x_offs, 0, Nx_loc - 1)
                ys = clamp.(ycenters[i] .+ y_offs, 0, Ny_loc - 1)
                [(y << Lx_pos) | x for (x, y) in zip(xs, ys)]
            end
            for i in 1:num_x
        ]
    else
        error("D must be 1 or 2")
    end

    Ak_w = zeros(Float64, Nω, num_x)

    # ── Position sites (used for QFT ops and optional sublattice masks) ──────
    aux_to_drop = Set{Index}()
    spin_proj && push!(aux_to_drop,
        isnothing(spin_s_det) ? H.sites[1] : spin_s_det::Index)
    !isnothing(nambu_s_det)  && push!(aux_to_drop, nambu_s_det::Index)
    !isnothing(layer_s_det)  && push!(aux_to_drop, layer_s_det::Index)
    !isnothing(sublat_s_det) && push!(aux_to_drop, sublat_s_det::Index)
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
    I_mpo_cpu = MPO(H.sites, "Id")
    Ham_n_cpu = (1 / H.scale) * +(H.mpo, (-H.center) * I_mpo_cpu; cutoff=cutoff)
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

    local _nambu_side = nambu_side_det
    local _layer_side = layer_side_det
    local _sublat_side = sublat_side_det
    local _spin_idx = isnothing(spin_s_det) ? H.sites[1] : spin_s_det

    # ── Online accumulation — fully on GPU ──────────────────────────────────
    # Workflow: prebuild everything on CPU (done above), then T_n stays on GPU
    # for the entire accumulate step.  Only the final scalar() calls transfer
    # numbers out of the GPU — no explicit MPO/MPS moves back to CPU.
    #
    # Per step:
    #   projection  → _project_aux_gpu  (dense typed projector, GPU throughout)
    #   QFT         → _apply_qft_conj_gpu  (pre-built GPU QFT operators)
    #   diagonal    → extract_diagonal_to_mps_gpu (same GPU dtype as input)
    #   sampling    → _eval_diag_mps_gpu  (scalars pulled out of GPU directly)
    function accumulate_Tn_gpu!(ak_accum, Tn_gpu, n)
        # Step 0: Nambu (BdG) projection
        after_nambu = nambu_proj ?
            [_project_aux_gpu(Tn_gpu, nambu_s_det::Index, sec; side=_nambu_side)
             for sec in (isnothing(proj_nambu) ? (1:2) : (proj_nambu:proj_nambu))] :
            MPO[Tn_gpu]

        # Step 1: spin projection
        after_spin = spin_proj ?
            [_project_aux_gpu(T, _spin_idx, sec; side=:pre)
             for T in after_nambu, sec in (isnothing(proj_s) ? (1:2) : (proj_s:proj_s))] :
            after_nambu

        # Step 1c: layer projection
        after_layer = if layer_proj
            n_lay     = dim(layer_s_det::Index)
            lay_range = isnothing(proj_layer) ? (1:n_lay) : (proj_layer:proj_layer)
            [_project_aux_gpu(T, layer_s_det::Index, sec; side=_layer_side)
             for T in after_spin for sec in lay_range]
        else
            after_spin
        end

        # Step 1b: sublattice aux projection
        after_sl_aux = if sublat_proj
            sl_range = isnothing(proj_sl) ? (1:dim(sublat_s_det::Index)) : (proj_sl:proj_sl)
            [_project_aux_gpu(T, sublat_s_det::Index, sec; side=_sublat_side)
             for T in after_layer for sec in sl_range]
        else
            after_layer
        end

        # Step 2: legacy sublattice mask sandwich (all GPU — masks pre-built above)
        if sublattice
            masks   = isnothing(proj_sl) ? [mask_A_gpu, mask_B_gpu] :
                      proj_sl == 1       ? [mask_A_gpu]              : [mask_B_gpu]
            sl_mpas = MPO[]
            for T in after_sl_aux, mask in masks
                push!(sl_mpas, apply(apply(mask, T; cutoff=cutoff, maxdim=maxdim), mask;
                                     cutoff=cutoff, maxdim=maxdim))
            end
        else
            sl_mpas = after_sl_aux
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
    Tkm2 = I_mpo_gpu
    Tkm1 = Ham_n_gpu
    two = gpu_type(2)
    negone = gpu_type(-1)

    accumulate_Tn_gpu!(Ak_w, Tkm2, 1)
    accumulate_Tn_gpu!(Ak_w, Tkm1, 2)

    for k in 3:Ncheb
        Tk = +(two * apply(Ham_n_gpu, Tkm1; cutoff=cutoff, maxdim=maxdim),
               negone * Tkm2; cutoff=cutoff, maxdim=maxdim)
        ITensorMPS.truncate!(Tk; cutoff=cutoff)
        accumulate_Tn_gpu!(Ak_w, Tk, k)
        Tkm2 = Tkm1
        Tkm1 = Tk
        _gpu_gc!()
        printinfo && (k % 10 == 0 || k == Ncheb) &&
            println("  [gpu] bands step $k/$Ncheb  maxlinkdim=$(maxlinkdim(Tkm1))")
    end

    # ── KPM normalization: 1 / (π² Ncheb √(1 − ε²)) ────────────────────────
    for iω in 1:Nω
        valid[iω] || continue
        Ak_w[iω, :] ./= (π^2 * Ncheb * sqrt(1 - ω_resc[iω]^2))
    end

    return isnothing(kpath_ticks) ? Ak_w :
           (Ak = Ak_w, ticks = kpath_ticks, labels = kpath_labels)
end


"""
    get_ldos_spatial_gpu(H, Ncheb, ω_phys_vals; kwargs...)
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

Keyword arguments are identical to `get_ldos_spatial` (`:mps` mode is not
available on GPU; only the single-pass MPO mode is implemented here). Pass
`type=ComplexF32` or `type=ComplexF64` to choose the GPU tensor datatype
consistently throughout the MPO recurrence, projections, and diagonal extraction.

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
    nambu_proj, spin_proj, layer_proj, sublat_proj =
        _autoenable_proj(H, nambu_proj, spin_proj, layer_proj, sublat_proj)

    # ── Aux site detection ───────────────────────────────────────────────────
    nambu_s_det,  nambu_side_det  = !isnothing(H.nambu_s)      ? aux_site(H, :nambu)      : (nothing, :pre)
    spin_s_det                    = H.spin_s
    layer_s_det,  layer_side_det  = !isnothing(H.layer_s)      ? aux_site(H, :layer)      : (nothing, :pre)
    sublat_s_det, sublat_side_det = !isnothing(H.sublattice_s) ? aux_site(H, :sublattice) : (nothing, :post)

    has_sublat = !isnothing(sublat_s_det)
    n_sub      = has_sublat ? dim(sublat_s_det::Index) : 1
    # Large-scale sampling traces out the sublattice (one value per unit cell);
    # atomic-scale / proj_sl=k resolves it into per-atom columns. See plan above.
    resolve_sl = has_sublat && (plan.resolve_sublattice || !isnothing(proj_sl))
    average_sl = has_sublat && !resolve_sl
    sl_fill    = has_sublat ?
        (isnothing(proj_sl) ? (1:n_sub) : (proj_sl:proj_sl)) :
        (1:1)

    # ── KPM setup ────────────────────────────────────────────────────────────
    ω_vals = (collect(ω_phys_vals) .- H.center) ./ H.scale
    Nω     = length(ω_vals)
    W      = _kpm_weight_matrix(Ncheb, ω_vals; kernel=kernel, lambda=lambda)
    valid  = [abs(ω) < 1.0 for ω in ω_vals]

    ng     = length(groups)
    n_cols = average_sl ? ng : ng * n_sub
    accum  = zeros(Float64, Nω, n_cols)

    # ── GPU operators ────────────────────────────────────────────────────────
    I_mpo_cpu = MPO(H.sites, "Id")
    Ham_n_cpu = (1 / H.scale) * +(H.mpo, (-H.center) * I_mpo_cpu; cutoff=cutoff)
    I_mpo_gpu = _to_gpu_mpo(I_mpo_cpu, gpu_type)
    Ham_n_gpu = _to_gpu_mpo(Ham_n_cpu, gpu_type)

    local _nambu_side  = nambu_side_det
    local _layer_side  = layer_side_det
    local _sublat_side = sublat_side_det
    local _spin_idx    = isnothing(spin_s_det) ? H.sites[1] : spin_s_det

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

    function accumulate_Tn_ldos_gpu!(ak_accum, Tn_gpu, n)
        after_nambu = nambu_proj ?
            [_project_aux_gpu(Tn_gpu, nambu_s_det::Index, sec; side=_nambu_side)
             for sec in (isnothing(proj_nambu) ? (1:2) : (proj_nambu:proj_nambu))] :
            MPO[Tn_gpu]

        after_spin = spin_proj ?
            [_project_aux_gpu(T, _spin_idx, sec; side=:pre)
             for T in after_nambu, sec in (isnothing(proj_s) ? (1:2) : (proj_s:proj_s))] :
            after_nambu

        after_layer = if layer_proj
            n_lay     = dim(layer_s_det::Index)
            lay_range = isnothing(proj_layer) ? (1:n_lay) : (proj_layer:proj_layer)
            [_project_aux_gpu(T, layer_s_det::Index, sec; side=_layer_side)
             for T in after_spin for sec in lay_range]
        else
            after_spin
        end

        if has_sublat
            # Resolved → per-atom column; averaged → fold all atoms into the
            # single per-pixel column u (mean over the n_sub atoms).
            for Tl in after_layer, s in sl_fill
                Tp       = _project_aux_gpu(Tl, sublat_s_det::Index, s; side=_sublat_side)
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
        else
            for Tp in after_layer
                diag_mps = ITensorMPS.truncate!(extract_diagonal_to_mps_gpu(Tp); cutoff=cutoff)
                for (u, val) in spatial_vals_gpu(diag_mps)
                    for iω in 1:Nω
                        valid[iω] || continue
                        ak_accum[iω, u] += W[n, iω] * val
                    end
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
        result[iω, :] = accum[iω, :] ./ (π^2 * Ncheb * sqrt(1 - ω_vals[iω]^2))
    end

    return result
end


"""
    get_dos_stochastic_gpu(H, Ncheb, ω_phys_vals; kwargs...)
        -> Vector{Float64}   length Nω

GPU-accelerated stochastic density of states via MPS Chebyshev KPM.

For each random sample the scaled Hamiltonian MPO lives on GPU and the product-
state MPS is transferred to GPU once before the recursion starts.
The Chebyshev moments ⟨ψ₀|T_n(H̃)|ψ₀⟩ are scalars pulled to CPU at each step.

Signature and optional kwargs are identical to `get_dos_stochastic` (CPU).
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

    _check_gpu("get_dos_stochastic_gpu")
    gpu_type = _resolve_gpu_type("get_dos_stochastic_gpu", type, dtype, cutoff)
    _ensure_scale!(H)
    dos_weighting in (:trace, :sample) ||
        error("get_dos_stochastic_gpu: dos_weighting must be :trace or :sample.")
    N_sample >= 0 || error("get_dos_stochastic_gpu: N_sample must be non-negative.")
    N_bound >= 0 || error("get_dos_stochastic_gpu: N_bound must be non-negative.")

    I_mpo_cpu = MPO(H.sites, "Id")
    Ham_n_cpu = (1 / H.scale) * +(H.mpo, (-H.center) * I_mpo_cpu; cutoff=cutoff)
    Ham_n_gpu = _to_gpu_mpo(Ham_n_cpu, gpu_type)

    D      = prod(ITensors.dim(s) for s in H.sites)
    N_phys = H.N
    is_exc = length(H.sites) == 2 * H.L
    continuum_only && !is_exc &&
        error("get_dos_stochastic_gpu: continuum_only=true requires an exciton Hamiltonian.")
    continuum_only && N_phys < 2 &&
        error("get_dos_stochastic_gpu: continuum_only=true requires H.N >= 2.")

    (; nambu_range, spin_range, layer_range, sl_range, any_aux_proj) =
        _aux_setup(H, nambu_proj, proj_nambu, spin_proj, proj_s,
                      layer_proj, proj_layer, sublat_proj, proj_sl)

    ω_vals = (collect(ω_phys_vals) .- H.center) ./ H.scale
    Nω     = length(ω_vals)
    W, denom = _dos_weight_matrix(Ncheb, ω_vals;
                                  kernel=kernel, lambda=lambda, eta=eta, m_order=m_order)
    valid  = [abs(ω) < 1.0 for ω in ω_vals]

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

    if any_aux_proj
        continuum_only &&
            error("get_dos_stochastic_gpu: continuum_only is not supported together with auxiliary projections.")
        D_eff = N_phys
        if N_sample > 0
            xs = rand(rng, 1:N_phys, N_sample)
            for (i, x) in enumerate(xs)
                for σ_n in nambu_range, σ_s in spin_range, σ_l in layer_range, σ_sl in sl_range
                    psi0_gpu = _to_gpu_mps(_ldos_make_psi0(H, x, σ_n, σ_s, σ_l, σ_sl), gpu_type)
                    χ = _run_kpm_mps_gpu!(psi0_gpu, accum_full, 1.0/N_sample)
                    (verbose || printinfo) && i % 10 == 0 &&
                        σ_n == first(nambu_range) && σ_s == first(spin_range) &&
                        σ_l == first(layer_range) && σ_sl == first(sl_range) &&
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

function _eval_fullsum_mps_1d_gpu(A::MPS)
    cuda = _tb_cuda_module()
    s    = siteinds(A)
    ElT  = eltype(A[1])
    acc  = cuda.cu(ITensor(one(ElT)))
    for i in 1:length(s)
        v_arr = fill(one(ElT), dim(s[i]))
        v = ITensors.itensor(cuda.CuArray(v_arr), s[i])
        acc *= A[i] * v
    end
    return real(scalar(acc))
end

function _contract_nh_block_gpu(W::MPO, block_s::Index;
                                row::Int = 2,
                                col::Int = 1,
                                dtype::Type{<:Complex} = ComplexF64)
    M = length(W)
    M >= 2 || error("_contract_nh_block_gpu requires an MPO with a block site and at least one physical site.")

    if siteind(W, M) == block_s
        bt = W[M] *
             _onehot_gpu(block_s => col, dtype) *
             _onehot_gpu(block_s' => row, dtype)
        tensors = ITensor[W[i] for i in 1:M-2]
        push!(tensors, W[M-1] * bt)
        return MPO(tensors)
    elseif siteind(W, 1) == block_s
        bt = W[1] *
             _onehot_gpu(block_s => col, dtype) *
             _onehot_gpu(block_s' => row, dtype)
        tensors = ITensor[W[2] * bt]
        for i in 3:M
            push!(tensors, W[i])
        end
        return MPO(tensors)
    end

    error("NH block index must be the first or last MPO site for _contract_nh_block_gpu.")
end

function _trace_nh_block_diagonal_gpu(P_gpu::MPO, block_s::Index;
                                      row::Int = 2,
                                      col::Int = 1,
                                      maxdim::Int = 100,
                                      cutoff::Real = 1e-8,
                                      dtype::Type{<:Complex} = ComplexF64)
    P_phys_gpu = _contract_nh_block_gpu(P_gpu, block_s;
        row=row, col=col, dtype=dtype)
    A_mps_gpu = ITensorMPS.truncate!(
        extract_diagonal_to_mps_gpu(P_phys_gpu);
        cutoff=Float64(cutoff), maxdim=maxdim)
    return _eval_fullsum_mps_1d_gpu(A_mps_gpu)
end

function _nh_diag_trace_scalar_online_gpu(NH::NonHermitianHamiltonian, n::Int;
                                          scale::Union{Nothing,Real} = nothing,
                                          maxdim::Int  = 100,
                                          cutoff::Real = 1e-8,
                                          source_row::Int = 2,
                                          source_col::Int = 1,
                                          block_row::Int  = 2,
                                          block_col::Int  = 1,
                                          dtype::Type{<:Complex} = ComplexF64,
                                          printinfo::Bool = false,
                                          verbose::Bool = false)
    _check_gpu("_nh_diag_trace_scalar_online_gpu")

    N  = 2 * n
    Hh = NH.hermitized
    sc = isnothing(scale) ? Hh.scale : Float64(scale)
    sc == 0.0 && error("_nh_diag_trace_scalar_online_gpu requires a nonzero scale.")
    n > 0 || error("_nh_diag_trace_scalar_online_gpu requires n > 0.")

    ak       = (cutoff=Float64(cutoff), maxdim=maxdim)
    A_op_gpu = _to_gpu_mpo(Hh.mpo / sc, dtype)
    S_gpu    = _to_gpu_mpo(nh_block_source(NH; row=source_row, col=source_col), dtype)
    I_gpu    = _to_gpu_mpo(MPO(Hh.sites, "Id"), dtype)
    weights  = nh_jackson_weights(N)
    two      = dtype(2)
    negone   = dtype(-1)
    zero     = dtype(0)

    verbose && println("    [gpu dtype=$dtype] A=$(eltype(A_op_gpu[1])) S=$(eltype(S_gpu[1])) I=$(eltype(I_gpu[1]))")

    Tkm2 = I_gpu
    Tkm1 = A_op_gpu
    Pkm2 = zero * S_gpu
    Pkm1 = S_gpu

    trace_acc = ComplexF64(weights[1]) *
        _trace_nh_block_diagonal_gpu(Pkm1, NH.block_s;
            row=block_row, col=block_col, maxdim=maxdim,
            cutoff=cutoff, dtype=dtype)

    for k in 3:N
        Tk = +(two * apply(A_op_gpu, Tkm1; ak...),
               negone * Tkm2; ak...)
        ITensorMPS.truncate!(Tk; ak...)

        Pk = +(+(two * apply(S_gpu, Tkm1; ak...),
                 two * apply(A_op_gpu, Pkm1; ak...);
                 ak...),
               negone * Pkm2; ak...)
        ITensorMPS.truncate!(Pk; ak...)

        if iseven(k)
            coeff = (-1)^(div(k, 2) - 1) * weights[k - 1]
            trace_acc += ComplexF64(coeff) *
                _trace_nh_block_diagonal_gpu(Pk, NH.block_s;
                    row=block_row, col=block_col, maxdim=maxdim,
                    cutoff=cutoff, dtype=dtype)
        end

        Tkm2 = Tkm1
        Tkm1 = Tk
        Pkm2 = Pkm1
        Pkm1 = Pk

        (verbose || (printinfo && k % 15 == 0)) &&
            println("    [gpu] NH scalar-diag cheb $k/$N  maxlinkdim(T)=$(maxlinkdim(Tkm1))  maxlinkdim(P)=$(maxlinkdim(Pkm1))")
    end

    _gpu_gc!()
    return real(trace_acc * 2.0 / (pi^2 * (N + 1)))
end

function _nh_diag_trace_online_gpu(NH::NonHermitianHamiltonian, n::Int;
                                   scale::Union{Nothing,Real} = nothing,
                                   maxdim::Int  = 100,
                                   cutoff::Real = 1e-8,
                                   source_row::Int = 2,
                                   source_col::Int = 1,
                                   block_row::Int  = 2,
                                   block_col::Int  = 1,
                                   dtype::Type{<:Complex} = ComplexF64,
                                   verbose::Bool = false)
    _check_gpu("_nh_diag_trace_online_gpu")

    N  = 2 * n
    Hh = NH.hermitized
    sc = isnothing(scale) ? Hh.scale : Float64(scale)
    sc == 0.0 && error("_nh_diag_trace_online_gpu requires a nonzero scale.")
    n > 0 || error("_nh_diag_trace_online_gpu requires n > 0.")

    ak       = (cutoff=Float64(cutoff), maxdim=maxdim)
    A_op_gpu = _to_gpu_mpo(Hh.mpo / sc, dtype)
    S_gpu    = _to_gpu_mpo(nh_block_source(NH; row=source_row, col=source_col), dtype)
    I_gpu    = _to_gpu_mpo(MPO(Hh.sites, "Id"), dtype)
    weights  = nh_jackson_weights(N)
    two      = dtype(2)
    negone   = dtype(-1)
    zero     = dtype(0)

    _diag(P_gpu) = ITensorMPS.truncate!(
        extract_diagonal_to_mps_gpu(
            _contract_nh_block_gpu(P_gpu, NH.block_s;
                row=block_row, col=block_col, dtype=dtype));
        ak...)

    Tkm2 = I_gpu
    Tkm1 = A_op_gpu
    Pkm2 = zero * S_gpu
    Pkm1 = S_gpu
    A_mps = dtype(weights[1]) * _diag(Pkm1)

    for k in 3:N
        Tk = +(two * apply(A_op_gpu, Tkm1; ak...),
               negone * Tkm2; ak...)
        ITensorMPS.truncate!(Tk; ak...)

        Pk = +(+(two * apply(S_gpu, Tkm1; ak...),
                 two * apply(A_op_gpu, Pkm1; ak...);
                 ak...),
               negone * Pkm2; ak...)
        ITensorMPS.truncate!(Pk; ak...)

        if iseven(k)
            coeff = dtype((-1)^(div(k, 2) - 1) * weights[k - 1])
            A_mps = +(A_mps, coeff * _diag(Pk); ak...)
            ITensorMPS.truncate!(A_mps; ak...)
        end

        Tkm2 = Tkm1
        Tkm1 = Tk
        Pkm2 = Pkm1
        Pkm1 = Pk

        verbose && println("    [gpu] NH diag order $k/$N  maxlinkdim(P)=$(maxlinkdim(Pkm1)) dtype(P)=$(eltype(Pkm1[1]))")
    end

    A_mps = dtype(2.0 / (pi^2 * (N + 1))) * A_mps
    dos = _eval_fullsum_mps_1d_gpu(A_mps)
    _gpu_gc!()
    return A_mps, dos
end

function _nh_random_probes_gpu_seed(sites::Vector{<:Index}, block_s::Index,
                                    ket_block::Int, bra_block::Int, rng,
                                    dtype::Type{<:Complex}=ComplexF64)
    N = length(sites)
    pos_rand = Dict(s => normalize(dtype.(randn(rng, Float64, dim(s)) .+
                                           1im .* randn(rng, Float64, dim(s))))
                    for s in sites if s != block_s)

    function _make(block_state)
        links = [Index(1, "Link,l=$i") for i in 1:N-1]
        tensors = Vector{ITensor}(undef, N)
        for i in 1:N
            s = sites[i]
            inds_i = Index[]
            i > 1 && push!(inds_i, links[i-1])
            push!(inds_i, s)
            i < N && push!(inds_i, links[i])
            T = ITensor(dtype, inds_i...)
            if s == block_s
                p = Pair{Index,Int}[]
                i > 1 && push!(p, links[i-1] => 1)
                push!(p, s => block_state)
                i < N && push!(p, links[i] => 1)
                T[p...] = one(dtype)
            else
                for (v, c) in enumerate(pos_rand[s])
                    p = Pair{Index,Int}[]
                    i > 1 && push!(p, links[i-1] => 1)
                    push!(p, s => v)
                    i < N && push!(p, links[i] => 1)
                    T[p...] = c
                end
            end
            tensors[i] = T
        end
        return MPS(tensors)
    end

    return _to_gpu_mps(_make(ket_block), dtype), _to_gpu_mps(_make(bra_block), dtype)
end

function _nh_stochastic_online_gpu(NH::NonHermitianHamiltonian, n::Int;
                                   scale::Union{Nothing,Real} = nothing,
                                   n_random::Int  = 10,
                                   maxdim::Int    = 100,
                                   cutoff::Real   = 1e-8,
                                   source_row::Int = 2,
                                   source_col::Int = 1,
                                   block_row::Int  = 2,
                                   block_col::Int  = 1,
                                   dtype::Type{<:Complex} = ComplexF64,
                                   rng = Random.default_rng(),
                                   verbose::Bool = false)
    _check_gpu("_nh_stochastic_online_gpu")
    dtype == ComplexF32 && cutoff < 1e-4 &&
        @warn "_nh_stochastic_online_gpu: cutoff=$cutoff with ComplexF32 may produce NaN; use dtype=ComplexF64 for large NH runs."

    N  = 2 * n
    Hh = NH.hermitized
    sc = isnothing(scale) ? Hh.scale : Float64(scale)
    sc == 0.0 && error("_nh_stochastic_online_gpu requires a nonzero scale.")
    n_random > 0 || error("_nh_stochastic_online_gpu requires n_random > 0.")

    A_op_gpu = _to_gpu_mpo(Hh.mpo / sc, dtype)
    S_gpu    = _to_gpu_mpo(nh_block_source(NH; row=source_row, col=source_col), dtype)
    weights  = nh_jackson_weights(N)
    D        = NH.parent.N

    dos_acc = ComplexF64(0)
    z_gpu   = dtype(0)
    two_gpu = dtype(2)
    negone_gpu = dtype(-1)
    apply_kwargs = (cutoff=Float64(cutoff), maxdim=maxdim)

    for ir in 1:n_random
        ket_probe, bra_probe = _nh_random_probes_gpu_seed(Hh.sites, NH.block_s,
                                                          source_col, block_row, rng,
                                                          dtype)
        tkm2 = ket_probe
        tkm1 = apply(A_op_gpu, ket_probe; apply_kwargs...)
        pkm2 = z_gpu * ket_probe
        pkm1 = apply(S_gpu, ket_probe; apply_kwargs...)

        partial_vals = zeros(ComplexF64, N)
        partial_vals[2] = inner(bra_probe, pkm1)

        for k in 3:N
            tk = +(two_gpu * apply(A_op_gpu, tkm1; apply_kwargs...),
                   negone_gpu * tkm2; apply_kwargs...)
            a_pkm1 = two_gpu * apply(A_op_gpu, pkm1; apply_kwargs...)
            pk_base = if iseven(k)
                s_tkm1 = two_gpu * apply(S_gpu, tkm1; apply_kwargs...)
                +(s_tkm1, a_pkm1; apply_kwargs...)
            else
                a_pkm1
            end
            pk = k == 3 ? pk_base : +(pk_base, negone_gpu * pkm2; apply_kwargs...)
            iseven(k) && (partial_vals[k] = inner(bra_probe, pk))
            tkm2 = tkm1
            tkm1 = tk
            pkm2 = pkm1
            pkm1 = pk
        end

        val = ComplexF64(0)
        for l in 2:2:N
            val += (-1)^(l ÷ 2 - 1) * weights[l - 1] * partial_vals[l]
        end
        dos_acc += val

        verbose && println("    [gpu] NH probe $ir/$n_random  maxlinkdim=$(maxlinkdim(tkm1))")
        _gpu_gc!()
    end

    return real(dos_acc * D * 2.0 / (π^2 * (N + 1) * n_random))
end

"""
    get_nh_dos_grid_gpu(H, xlims, nx, ylims, ny, n; scale=nothing,
                        nh_scale_padding=1.05, n_random, ...)
        -> (xgrid, ygrid, Z)

GPU stochastic non-Hermitian KPM spectral-weight grid. This mirrors
`nh_spectrum_grid(...; mode=:stochastic)`: for each complex point
`z = x + im*y`, the non-Hermitian Hamiltonian is hermitized on CPU, then the
dual-chain stochastic MPS recurrence runs on GPU. Only scalar moments are
copied back to CPU.

The integer `n` follows the existing NH convention: the partial recurrence runs
to order `2n`.
"""
function get_nh_dos_grid_gpu(H::TBHamiltonian, xlims, nx::Int, ylims, ny::Int, n::Int;
                             scale::Union{Nothing,Real} = nothing,
                             nh_scale_padding::Real = 1.05,
                             convention::Symbol      = :z_minus_H,
                             block_placement::Symbol = :post,
                             n_random::Int           = 10,
                             seed::Union{Int,Nothing}= 42,
                             maxdim::Int             = 100,
                             cutoff::Real            = 1e-8,
                             dmrg_nsweeps::Int       = 5,
                             dmrg_maxdim             = [10, 20, 40],
                             dmrg_linkdim::Int       = 4,
                             dtype::Type{<:Complex}  = ComplexF64,
                             verbose::Bool           = false,
                             printinfo::Bool         = false)
    _check_gpu("get_nh_dos_grid_gpu")
    dtype == ComplexF32 && cutoff < 1e-4 &&
        @warn "get_nh_dos_grid_gpu: cutoff=$cutoff with ComplexF32 may be unstable; use dtype=ComplexF64 for large NH runs."
    n > 0 || error("get_nh_dos_grid_gpu: n must be positive.")
    n_random > 0 || error("get_nh_dos_grid_gpu: n_random must be positive.")

    xgrid = collect(range(xlims[1], xlims[2]; length=nx))
    ygrid = collect(range(ylims[1], ylims[2]; length=ny))
    nh_scale = nh_kpm_scale(H, (ComplexF64(x, y) for x in xgrid for y in ygrid);
        scale=scale,
        padding=nh_scale_padding,
        maxdim=maxdim,
        cutoff=cutoff,
        convention=convention,
        block_placement=block_placement,
        dmrg_nsweeps=dmrg_nsweeps,
        dmrg_maxdim=dmrg_maxdim,
        dmrg_linkdim=dmrg_linkdim,
        printinfo=(verbose || printinfo))
    Z = Matrix{Float64}(undef, ny, nx)
    rng = seed === nothing ? Random.default_rng() : Random.MersenneTwister(seed)

    (verbose || printinfo) &&
        println("get_nh_dos_grid_gpu: $(nx)x$(ny)=$(nx*ny) points, NH order=2*$n, n_random=$n_random, scale=$nh_scale")

    for (ix, x) in enumerate(xgrid)
        (verbose || printinfo) &&
            println("  [gpu] NH grid col $(lpad(ix, ndigits(nx)))/$nx  Re(z)=$(round(x, digits=4))")
        for (iy, y) in enumerate(ygrid)
            NH = hermitize(H; z=x + 1im*y, scale=nh_scale, maxdim=maxdim,
                           cutoff=cutoff, convention=convention,
                           block_placement=block_placement)
            Z[iy, ix] = _nh_stochastic_online_gpu(NH, n;
                scale=nh_scale,
                n_random=n_random,
                maxdim=maxdim,
                cutoff=cutoff,
                dtype=dtype,
                rng=rng,
                verbose=verbose)
        end
    end

    return xgrid, ygrid, Z
end

nh_spectrum_grid_gpu(H::TBHamiltonian, xlims, nx::Int, ylims, ny::Int, n::Int; kwargs...) =
    get_nh_dos_grid_gpu(H, xlims, nx, ylims, ny, n; kwargs...)

"""
    get_nh_dos_points_gpu(H, z_points, n; scale=nothing,
                          nh_scale_padding=1.05, n_random, point_ids, ...)
        -> Vector{Float64}

GPU stochastic NH KPM at an explicit list of complex energies. This is the
array-job companion to `get_nh_dos_grid_gpu`: each `z_points[j]` is independent,
so production scripts can split a large grid over many GPUs and concatenate the
long-form CSV outputs afterward.

When `seed` is an integer, each point uses a deterministic seed
`seed + seed_stride * point_id`, where `point_id` defaults to the local point
index. Supplying global flattened grid indices as `point_ids` makes stochastic
samples reproducible independent of how the grid is tiled.
"""
function get_nh_dos_points_gpu(H::TBHamiltonian, z_points, n::Int;
                               scale::Union{Nothing,Real} = nothing,
                               nh_scale_padding::Real = 1.05,
                               convention::Symbol       = :z_minus_H,
                               block_placement::Symbol  = :post,
                               n_random::Int            = 10,
                               seed::Union{Int,Nothing} = 42,
                               seed_stride::Int         = 1_000_003,
                               point_ids                = nothing,
                               maxdim::Int              = 100,
                               cutoff::Real             = 1e-8,
                               dmrg_nsweeps::Int        = 5,
                               dmrg_maxdim              = [10, 20, 40],
                               dmrg_linkdim::Int        = 4,
                               dtype::Type{<:Complex}   = ComplexF64,
                               verbose::Bool            = false,
                               printinfo::Bool          = false)
    _check_gpu("get_nh_dos_points_gpu")
    dtype == ComplexF32 && cutoff < 1e-4 &&
        @warn "get_nh_dos_points_gpu: cutoff=$cutoff with ComplexF32 may be unstable; use dtype=ComplexF64 for large NH runs."
    n > 0 || error("get_nh_dos_points_gpu: n must be positive.")
    n_random > 0 || error("get_nh_dos_points_gpu: n_random must be positive.")

    z_list = collect(z_points)
    Nz = length(z_list)
    ids = isnothing(point_ids) ? collect(1:Nz) : collect(point_ids)
    length(ids) == Nz || error("get_nh_dos_points_gpu: point_ids length must match z_points length.")
    nh_scale = nh_kpm_scale(H, z_list;
        scale=scale,
        padding=nh_scale_padding,
        maxdim=maxdim,
        cutoff=cutoff,
        convention=convention,
        block_placement=block_placement,
        dmrg_nsweeps=dmrg_nsweeps,
        dmrg_maxdim=dmrg_maxdim,
        dmrg_linkdim=dmrg_linkdim,
        printinfo=(verbose || printinfo))

    values = Vector{Float64}(undef, Nz)
    (verbose || printinfo) &&
        println("get_nh_dos_points_gpu: $Nz points, NH order=2*$n, n_random=$n_random, scale=$nh_scale")

    for j in 1:Nz
        z = ComplexF64(z_list[j])
        point_id = Int(ids[j])
        (verbose || printinfo) &&
            println("  [gpu] NH point $j/$Nz  id=$point_id  z=$(round(real(z), digits=4)) + $(round(imag(z), digits=4))im")

        rng = if seed === nothing
            Random.default_rng()
        else
            Random.MersenneTwister(seed + seed_stride * point_id)
        end

        NH = hermitize(H; z=z, scale=nh_scale, maxdim=maxdim,
                       cutoff=cutoff, convention=convention,
                       block_placement=block_placement)
        values[j] = _nh_stochastic_online_gpu(NH, n;
            scale=nh_scale,
            n_random=n_random,
            maxdim=maxdim,
            cutoff=cutoff,
            dtype=dtype,
            rng=rng,
            verbose=verbose)
    end

    return values
end

"""
    get_nh_dos_points_diag_trace_gpu(H, z_points, n; scale=nothing,
                                     nh_scale_padding=1.05, point_ids, ...)
        -> Vector{Float64}

Deterministic GPU NH KPM at an explicit list of complex energies. For each
`z`, the hermitized NH problem is built on CPU, then the online MPO-MPO
recurrence runs on GPU and evaluates the total trace through diagonal
extraction plus a GPU-resident all-sites sum. This avoids stochastic probes.

The integer `n` follows the NH convention used elsewhere in this file: the
partial recurrence runs to order `2n`.
"""
function get_nh_dos_points_diag_trace_gpu(H::TBHamiltonian, z_points, n::Int;
                                          scale::Union{Nothing,Real} = nothing,
                                          nh_scale_padding::Real = 1.05,
                                          convention::Symbol       = :z_minus_H,
                                          block_placement::Symbol  = :post,
                                          point_ids                = nothing,
                                          maxdim::Int              = 100,
                                          cutoff::Real             = 1e-8,
                                          dmrg_nsweeps::Int        = 5,
                                          dmrg_maxdim              = [10, 20, 40],
                                          dmrg_linkdim::Int        = 4,
                                          dtype::Type{<:Complex}   = ComplexF64,
                                          source_row::Int          = 2,
                                          source_col::Int          = 1,
                                          block_row::Int           = 2,
                                          block_col::Int           = 1,
                                          verbose::Bool            = false,
                                          printinfo::Bool          = false)
    _check_gpu("get_nh_dos_points_diag_trace_gpu")
    n > 0 || error("get_nh_dos_points_diag_trace_gpu: n must be positive.")

    z_list = collect(z_points)
    Nz = length(z_list)
    ids = isnothing(point_ids) ? collect(1:Nz) : collect(point_ids)
    length(ids) == Nz ||
        error("get_nh_dos_points_diag_trace_gpu: point_ids length must match z_points length.")
    nh_scale = nh_kpm_scale(H, z_list;
        scale=scale,
        padding=nh_scale_padding,
        maxdim=maxdim,
        cutoff=cutoff,
        convention=convention,
        block_placement=block_placement,
        dmrg_nsweeps=dmrg_nsweeps,
        dmrg_maxdim=dmrg_maxdim,
        dmrg_linkdim=dmrg_linkdim,
        printinfo=(verbose || printinfo))

    values = Vector{Float64}(undef, Nz)
    (verbose || printinfo) &&
        println("get_nh_dos_points_diag_trace_gpu: $Nz points, NH order=2*$n, dtype=$dtype, scale=$nh_scale")

    for j in 1:Nz
        z = ComplexF64(z_list[j])
        point_id = Int(ids[j])
        (verbose || (printinfo && (j == 1 || j % 15 == 0 || j == Nz))) &&
            println("  [gpu] NH diag-trace point $j/$Nz  id=$point_id  z=$(round(real(z), digits=4)) + $(round(imag(z), digits=4))im")

        NH = hermitize(H; z=z, scale=nh_scale, maxdim=maxdim,
                       cutoff=cutoff, convention=convention,
                       block_placement=block_placement)
        values[j] = _nh_diag_trace_scalar_online_gpu(NH, n;
            scale=nh_scale,
            maxdim=maxdim,
            cutoff=cutoff,
            source_row=source_row,
            source_col=source_col,
            block_row=block_row,
            block_col=block_col,
            dtype=dtype,
            printinfo=printinfo,
            verbose=verbose)
    end

    return values
end

"""
    get_nh_dos_grid_diag_trace_gpu(H, xlims, nx, ylims, ny, n; scale=nothing,
                                   nh_scale_padding=1.05, ...)
        -> (xgrid, ygrid, Z)

Grid companion to `get_nh_dos_points_diag_trace_gpu`.
"""
function get_nh_dos_grid_diag_trace_gpu(H::TBHamiltonian, xlims, nx::Int, ylims, ny::Int, n::Int;
                                        scale::Union{Nothing,Real} = nothing,
                                        nh_scale_padding::Real = 1.05,
                                        convention::Symbol      = :z_minus_H,
                                        block_placement::Symbol = :post,
                                        maxdim::Int             = 100,
                                        cutoff::Real            = 1e-8,
                                        dmrg_nsweeps::Int       = 5,
                                        dmrg_maxdim             = [10, 20, 40],
                                        dmrg_linkdim::Int       = 4,
                                        dtype::Type{<:Complex}  = ComplexF64,
                                        verbose::Bool           = false,
                                        printinfo::Bool         = false)
    _check_gpu("get_nh_dos_grid_diag_trace_gpu")
    n > 0 || error("get_nh_dos_grid_diag_trace_gpu: n must be positive.")

    xgrid = collect(range(xlims[1], xlims[2]; length=nx))
    ygrid = collect(range(ylims[1], ylims[2]; length=ny))
    z_points = ComplexF64[]
    point_ids = Int[]
    for (ix, x) in enumerate(xgrid), (iy, y) in enumerate(ygrid)
        push!(z_points, ComplexF64(x, y))
        push!(point_ids, (ix - 1) * ny + iy)
    end

    values = get_nh_dos_points_diag_trace_gpu(H, z_points, n;
        scale=scale,
        convention=convention,
        block_placement=block_placement,
        point_ids=point_ids,
        maxdim=maxdim,
        cutoff=cutoff,
        nh_scale_padding=nh_scale_padding,
        dmrg_nsweeps=dmrg_nsweeps,
        dmrg_maxdim=dmrg_maxdim,
        dmrg_linkdim=dmrg_linkdim,
        dtype=dtype,
        verbose=verbose,
        printinfo=printinfo)

    Z = Matrix{Float64}(undef, ny, nx)
    for (v, pid) in zip(values, point_ids)
        ix = div(pid - 1, ny) + 1
        iy = mod(pid - 1, ny) + 1
        Z[iy, ix] = v
    end

    return xgrid, ygrid, Z
end


# Enumerate all block members for exciton block-reduce (positional averaging).
# For :block, spatial_sampling_plan gives singleton groups; this expands each to the
# full set of probe positions inside the coarse block, enumerated from plan.stride_x/y.
function _exciton_block_groups(plan, Lx::Union{Nothing,Int}, L::Int)
    nblocks = length(plan.centers)
    Wx = plan.stride_x
    if Lx === nothing
        return [[ixp * Wx + d + 1 for d in 0:Wx-1] for ixp in 0:nblocks-1]
    end
    a  = plan.a
    Wy = plan.stride_y
    Nx = 2^Lx
    return [let ixp = (iblock-1) % 2^a, iyp = (iblock-1) ÷ 2^a
                [ixp*Wx + dx + (iyp*Wy + dy)*Nx + 1 for dy in 0:Wy-1 for dx in 0:Wx-1]
            end
            for iblock in 1:nblocks]
end

"""
    get_exciton_ldos_spatial_gpu(H, Ncheb, ω_phys_vals;
                                 Lx, num_y, reduce,
                                 X_list, X_groups, num_x, num_avg, x_start, x_end,
                                 kernel, lambda, eta, m_order, maxdim, cutoff,
                                 verbose, printinfo)
        -> Matrix{Float64}   (Nω × n_cols)

GPU-accelerated spatial exciton LDOS A(X,ω) = ⟨X,X|δ(ω−H)|X,X⟩ via MPS Chebyshev KPM.
One GPU Chebyshev recursion runs per probe position X (electron = hole = X, 1-indexed
in 1:H.N) from |X,X⟩ = mpsexciton(X, H.sites); moments are scalars pulled to CPU.

**1D sampling** (default, `Lx=nothing`): `num_x` coarse positions over `[x_start, x_end]`
with `num_avg` sub-positions per coarse cell averaged per output pixel.

**2D grid** (`Lx` provided): positions are 1-indexed on the (Lx+Ly)-qubit quantics grid,
encoded as X = ix + iy·2^Lx + 1 (row-major, 0-indexed). `num_x × num_y` coarse cells
are sampled via `spatial_sampling_plan` with `num_avg` sub-positions per cell.
Output columns are row-major over coarse cells (iy outer, ix inner).

`X_list` / `X_groups` / `x_groups` bypass the automatic plan and pass positions directly.

`kernel=:hodc` selects HODC reconstruction (`eta`, `m_order`; `eta=0` → `1/(Ncheb+1)`).
Other kernels: `:jackson` (default), `:lorentz` (`lambda`), `:fejer`, `:dirichlet`.

Use `type=ComplexF32` (default, faster) or `type=ComplexF64` (safer at tight cutoffs
or on large systems where F32 eigendecomposition can produce NaN). `dtype` is accepted
as an alias for `type` for consistency with other GPU entry points.

`return_maxlinkdim=true` returns `(result, linkdims)` instead of just `result`, where
`linkdims::Vector{Int}` is the reached MPS bond dimension per output column (the χ the
Chebyshev recursion hit under the given `maxdim`/`cutoff`). Useful for cutoff/tolerance
studies where χ is the observable.

!!! note "Block averaging not supported"
    `reduce=:block` is **not available** for the exciton LDOS. In the MPO-based LDOS
    functions (`get_ldos_spatial_gpu`), block averaging is a cheap O(1) partial trace
    of the diagonal MPS over the within-block position bits. The exciton LDOS has no
    diagonal MPS representation: each probe requires an independent Chebyshev recursion
    from |X,X⟩, so block averaging would cost O(block_size) recursions per pixel —
    equivalent to computing every site. Use `reduce=:point` with `num_avg` for light
    spatial averaging (each output pixel averages `num_avg` nearby positions).
"""
function get_exciton_ldos_spatial_gpu(H::TBHamiltonian, Ncheb::Int, ω_phys_vals;
                                       Lx::Union{Nothing,Int}    = nothing,
                                       num_y::Union{Nothing,Int} = nothing,
                                       reduce::Symbol            = :point,
                                       X_list           = nothing,
                                       X_groups         = nothing,
                                       x_groups         = nothing,
                                       num_x::Int       = 0,
                                       num_avg::Int     = 1,
                                       x_start::Int     = 1,
                                       x_end::Int       = H.N,
                                       kernel::Symbol   = :jackson,
                                       lambda::Real     = 4.0,
                                       eta::Real        = 0.0,
                                       m_order::Int     = 4,
                                       maxdim::Int      = 100,
                                       cutoff::Real     = 1e-8,
                                       type::Type{<:Number}                  = ComplexF32,
                                       dtype::Union{Nothing,Type{<:Number}}  = nothing,
                                       verbose::Bool    = false,
                                       printinfo::Bool  = false,
                                       return_maxlinkdim::Bool = false)

    _check_gpu("get_exciton_ldos_spatial_gpu")
    gpu_type = dtype === nothing ? type : dtype
    dtype !== nothing && dtype != type && type != ComplexF32 &&
        error("get_exciton_ldos_spatial_gpu: received both type=$type and dtype=$dtype; pass only one datatype keyword.")
    gpu_type == ComplexF32 && cutoff < 1e-6 &&
        @warn "get_exciton_ldos_spatial_gpu: cutoff=$cutoff with ComplexF32 may produce NaN — use type=ComplexF64 or cutoff ≥ 1e-4."
    reduce === :block &&
        error("get_exciton_ldos_spatial_gpu: reduce=:block is not supported for the exciton LDOS. " *
              "Block averaging requires O(block_size) independent Chebyshev recursions per pixel, " *
              "making it equivalent to computing every site. Use reduce=:point with num_avg for " *
              "light spatial averaging. Block averaging is only available in MPO-based LDOS functions.")
    reduce === :point ||
        error("get_exciton_ldos_spatial_gpu: reduce must be :point, got $reduce.")
    _ensure_scale!(H)
    length(H.sites) == 2 * H.L ||
        error("get_exciton_ldos_spatial_gpu: H is not an exciton Hamiltonian (expected length(H.sites) == 2*H.L).")

    X_groups !== nothing && x_groups !== nothing &&
        error("get_exciton_ldos_spatial_gpu: pass only one of X_groups or x_groups.")
    X_list !== nothing && (X_groups !== nothing || x_groups !== nothing) &&
        error("get_exciton_ldos_spatial_gpu: pass either X_list or grouped positions, not both.")

    group_arg = X_groups !== nothing ? X_groups : x_groups
    groups = if group_arg !== nothing
        group_arg isa AbstractVector{<:AbstractVector} ?
            [collect(Int, grp) for grp in group_arg] :
            [[Int(x)] for x in group_arg]
    elseif X_list !== nothing
        [[Int(x)] for x in X_list]
    else
        _grid = Lx !== nothing
        _nx   = num_x <= 0 ? (_grid ? 8 : H.N) : num_x
        plan  = spatial_sampling_plan(H.L;
                    Lx      = Lx,
                    grid    = _grid,
                    reduce  = reduce,
                    num_x   = _nx,
                    num_y   = num_y,
                    num_avg = reduce === :point ? num_avg : 1,
                    x_start = x_start,
                    x_end   = x_end)
        reduce === :block ? _exciton_block_groups(plan, Lx, H.L) : plan.groups
    end
    isempty(groups) && error("get_exciton_ldos_spatial_gpu: no spatial groups were selected.")
    for grp in groups
        isempty(grp) && error("get_exciton_ldos_spatial_gpu: empty spatial group.")
        all(x -> 1 <= x <= H.N, grp) ||
            error("get_exciton_ldos_spatial_gpu: all positions must lie in 1:H.N.")
    end
    Xs = first.(groups)

    I_mpo_cpu = MPO(H.sites, "Id")
    Ham_n_cpu = (1 / H.scale) * +(H.mpo, (-H.center) * I_mpo_cpu; cutoff=cutoff)
    Ham_n_gpu = _to_gpu_mpo(Ham_n_cpu, gpu_type)

    ω_vals = (collect(ω_phys_vals) .- H.center) ./ H.scale
    Nω     = length(ω_vals)
    W, denom = _dos_weight_matrix(Ncheb, ω_vals;
                                  kernel=kernel, lambda=lambda, eta=eta, m_order=m_order)
    valid  = [abs(ω) < 1.0 for ω in ω_vals]

    nX           = length(groups)
    result       = zeros(Float64, Nω, nX)
    apply_kwargs = (cutoff=Float64(cutoff), maxdim=maxdim)

    printinfo && println("  [gpu] exciton ldos dtype=$gpu_type")

    linkdims = zeros(Int, nX)   # reached MPS bond dim per output column (see return_maxlinkdim)

    for (j, group) in enumerate(groups)
        last_linkdim = 0

        for X in group
            psi0_gpu = _to_gpu_mps(mpsexciton(X, H.sites), gpu_type)
            accum    = zeros(Float64, Nω)

            function kpm_step!(phi, n)
                mu = Float64(real(inner(psi0_gpu, phi)))
                for iω in 1:Nω
                    valid[iω] || continue
                    accum[iω] += W[n, iω] * mu
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

            last_linkdim = maxlinkdim(phi_km1)
            for iω in 1:Nω
                valid[iω] || continue
                result[iω, j] += accum[iω] / denom[iω]
            end

            _gpu_gc!()
        end

        for iω in 1:Nω
            result[iω, j] /= length(group)
        end
        linkdims[j] = last_linkdim
        (verbose || printinfo) && (j % 5 == 0 || j == nX) &&
            println("  [gpu] exciton ldos $j/$nX (X=$(Xs[j]), n_avg=$(length(group)))  maxlinkdim=$last_linkdim")
    end

    return return_maxlinkdim ? (result, linkdims) : result
end


"""
    get_exciton_cheb_convergence_gpu(H, X, Ncheb_max;
                                      maxdim_test, maxdim_ref, cutoff, printinfo)
        -> NamedTuple

Run two parallel GPU Chebyshev KPM recursions starting from |X,X⟩ = mpsexciton(X, H.sites):
a *reference* recursion at `maxdim_ref` and a *test* recursion at `maxdim_test`.
At each step n the two Chebyshev vectors φ_ref^n and φ_test^n are compared to yield:

  err_fidelity_n = 1 − |⟨φ_ref^n | φ_test^n⟩|² / (‖φ_ref^n‖² ‖φ_test^n‖²)

which grows from 0 (perfect agreement) towards 1 as truncation errors accumulate.

Returns a NamedTuple with Float64 / Int vectors of length Ncheb_max:
  n            — Chebyshev index (1-based)
  mu_ref       — KPM moment ⟨X,X|T_n(H̃)|X,X⟩ from reference recursion
  mu_test      — KPM moment from test recursion
  delta_mu     — |mu_ref − mu_test| (moment discrepancy)
  err_fidelity — infidelity 1 − fidelity as above
  mdim_ref     — maxlinkdim of φ_ref^n
  mdim_test    — maxlinkdim of φ_test^n
  norm_ref     — ‖φ_ref^n‖ (should stay ≤ 1; growth indicates instability)
  norm_test    — ‖φ_test^n‖

The Hamiltonian is rescaled internally: H̃ = (H − center·I) / scale, same as in
get_exciton_ldos_spatial_gpu. All MPS live on GPU in ComplexF32 throughout.

Use this to find the critical Ncheb beyond which `maxdim_test` is too small for a
given system size — the threshold is where err_fidelity departs significantly from 0
(e.g. > 0.01 for a tight criterion, > 0.1 for a loose one).
"""
function get_exciton_cheb_convergence_gpu(H::TBHamiltonian, X::Int, Ncheb_max::Int;
                                           maxdim_test::Int  = 100,
                                           maxdim_ref::Int   = 500,
                                           cutoff::Real      = 1e-4,
                                           type::Type{<:Number} = ComplexF32,
                                           dtype::Union{Nothing,Type{<:Number}} = nothing,
                                           printinfo::Bool   = false)
    _check_gpu("get_exciton_cheb_convergence_gpu")
    gpu_type = _resolve_gpu_type("get_exciton_cheb_convergence_gpu", type, dtype, cutoff)
    length(H.sites) == 2 * H.L ||
        error("get_exciton_cheb_convergence_gpu: H is not an exciton Hamiltonian (expected length(H.sites) == 2*H.L).")
    1 <= X <= H.N ||
        error("get_exciton_cheb_convergence_gpu: X=$X out of range 1:$(H.N).")
    maxdim_ref >= maxdim_test ||
        @warn "get_exciton_cheb_convergence_gpu: maxdim_ref=$maxdim_ref < maxdim_test=$maxdim_test; reference is no more accurate than test."

    _ensure_scale!(H)

    I_mpo_cpu  = MPO(H.sites, "Id")
    Ham_n_cpu  = (1 / H.scale) * +(H.mpo, (-H.center) * I_mpo_cpu; cutoff=Float64(cutoff))
    Ham_n_gpu  = _to_gpu_mpo(Ham_n_cpu, gpu_type)

    psi0_gpu   = _to_gpu_mps(mpsexciton(X, H.sites), gpu_type)

    ak_ref  = (cutoff=Float64(cutoff), maxdim=maxdim_ref)
    ak_test = (cutoff=Float64(cutoff), maxdim=maxdim_test)

    # Chebyshev recursion: T_0 = psi0, T_1 = H̃·psi0,  T_n = 2H̃·T_{n-1} − T_{n-2}
    phi_ref_km2  = nothing;  phi_ref_km1  = psi0_gpu
    phi_test_km2 = nothing;  phi_test_km1 = psi0_gpu

    n_vec        = Int[]
    mu_ref_vec   = Float64[]
    mu_test_vec  = Float64[]
    delta_mu_vec = Float64[]
    err_vec      = Float64[]
    mdim_ref_vec = Int[]
    mdim_test_vec = Int[]
    norm_ref_vec = Float64[]
    norm_test_vec = Float64[]

    for n in 1:Ncheb_max
        if n == 1
            phi_ref_new  = apply(Ham_n_gpu, phi_ref_km1;  ak_ref...)
            phi_test_new = apply(Ham_n_gpu, phi_test_km1; ak_test...)
        else
            phi_ref_new  = +(2 * apply(Ham_n_gpu, phi_ref_km1;  ak_ref...),
                             -phi_ref_km2;  ak_ref...)
            phi_test_new = +(2 * apply(Ham_n_gpu, phi_test_km1; ak_test...),
                             -phi_test_km2; ak_test...)
        end

        mu_ref   = Float64(real(inner(psi0_gpu, phi_ref_new)))
        mu_test  = Float64(real(inner(psi0_gpu, phi_test_new)))
        n2_ref   = Float64(real(inner(phi_ref_new,  phi_ref_new)))
        n2_test  = Float64(real(inner(phi_test_new, phi_test_new)))
        ovlp     = inner(phi_ref_new, phi_test_new)
        fid      = Float64(abs2(ovlp)) / (n2_ref * n2_test)
        err      = max(0.0, 1.0 - fid)

        push!(n_vec,         n)
        push!(mu_ref_vec,    mu_ref)
        push!(mu_test_vec,   mu_test)
        push!(delta_mu_vec,  abs(mu_ref - mu_test))
        push!(err_vec,       err)
        push!(mdim_ref_vec,  maxlinkdim(phi_ref_new))
        push!(mdim_test_vec, maxlinkdim(phi_test_new))
        push!(norm_ref_vec,  sqrt(max(0.0, n2_ref)))
        push!(norm_test_vec, sqrt(max(0.0, n2_test)))

        printinfo && (n % 10 == 0 || n == Ncheb_max) &&
            println("  [cheb_conv] n=$(lpad(n,3))  err=$(round(err; sigdigits=3))  mdim_ref=$(mdim_ref_vec[end])  mdim_test=$(mdim_test_vec[end])")

        phi_ref_km2  = phi_ref_km1;   phi_ref_km1  = phi_ref_new
        phi_test_km2 = phi_test_km1;  phi_test_km1 = phi_test_new
        _gpu_gc!()
    end

    return (; n=n_vec, mu_ref=mu_ref_vec, mu_test=mu_test_vec, delta_mu=delta_mu_vec,
              err_fidelity=err_vec, mdim_ref=mdim_ref_vec, mdim_test=mdim_test_vec,
              norm_ref=norm_ref_vec, norm_test=norm_test_vec)
end

# Multi-probe overload: run convergence for each X in X_probes and return a
# Vector of per-probe NamedTuples (same structure as the single-X version).
# The Hamiltonian rescaling and GPU MPO conversion happen once per call.
function get_exciton_cheb_convergence_gpu(H::TBHamiltonian,
                                           X_probes::AbstractVector{<:Integer},
                                           Ncheb_max::Int; kwargs...)
    return [get_exciton_cheb_convergence_gpu(H, Int(X), Ncheb_max; kwargs...)
            for X in X_probes]
end


# ============================================================
# GPU Chern marker
# ============================================================

"""
    get_C_gpu(H::TBHamiltonian, xfunc=nothing, yfunc=nothing; kwargs...) -> Function

GPU-accelerated real-space Chern marker.  Mirrors `get_C` exactly but runs all
MPO×MPO products (projector assembly and C1–C4 construction) on GPU.

Returns the same closure `C_at(uc::Int) -> ComplexF64` as `get_C`.

# Key differences from `get_C`
- All `apply`/`truncate!` operations run on GPU tensors.
- `dtype` (default `ComplexF32`) selects the GPU element type; the marker is
  intrinsically complex, so only `ComplexF32` / `ComplexF64` are accepted. Use
  `dtype=ComplexF64` to avoid NaN from ComplexF32 eigendecompositions on large
  systems at tight cutoffs (a warning is emitted for `ComplexF32` + `cutoff < 1e-6`).
- The projector is built on CPU first (via `_get_projector`), then moved to GPU.
  For method=:mcweeny this means the purification loop runs on GPU.
- `sequential` mode is not supported (non-sequential quenched is always used).

All other keyword arguments are identical to `get_C`.
"""
function get_C_gpu(H::TBHamiltonian, xfunc=nothing, yfunc=nothing;
                   method::Symbol   = :mcweeny,
                   fermi::Real      = 0.0,
                   l                = nothing,
                   Λ::Real          = 10,
                   Lambda           = nothing,
                   Nchebychev::Int  = 300,
                   maxdim::Int      = 500,
                   cutoff::Real     = 1e-8,
                   Nel              = nothing,
                   quenched::Bool   = true,
                   dtype::Type{<:Complex} = ComplexF32,
                   printinfo::Bool  = false)

    _check_gpu("get_C_gpu")
    gpu_type = _resolve_gpu_type("get_C_gpu", dtype, nothing, cutoff)
    Λ_val = Lambda !== nothing ? Float64(Lambda) : Float64(Λ)
    ak    = (cutoff=Float64(cutoff), maxdim=maxdim)

    # ── geometry ──────────────────────────────────────────────────────────────
    if xfunc === nothing || yfunc === nothing
        geom = H.geometry_uc !== nothing ? H.geometry_uc :
               H.geometry   !== nothing ? H.geometry   :
               error("get_C_gpu: H has no geometry; provide xfunc and yfunc explicitly.")
        xfunc === nothing && (xfunc = (i, _) -> geom(i + 1)[1])
        yfunc === nothing && (yfunc = (i, _) -> geom(i + 1)[2])
    end

    # ── sublattice bookkeeping (mirrors get_C_op_MPO_from_P) ──────────────────
    L       = H.L
    l_bits  = l === nothing ? div(L, 2) : l
    L_chain = 2^l_bits
    sites   = H.sites
    n_sub   = length(sites) > L ? dim(sites[L+1]) : 1
    has_sub = n_sub > 1
    pos_sites = has_sub ? collect(sites[1:L]) : collect(sites)
    sub_s     = has_sub ? sites[L+1] : nothing
    I_mat     = has_sub ? Matrix{Float64}(LinearAlgebra.I, n_sub, n_sub) : nothing

    xfunc_pos = has_sub ? ((i, Lc) -> xfunc(i * n_sub, Lc)) : xfunc
    yfunc_pos = has_sub ? ((i, Lc) -> yfunc(i * n_sub, Lc)) : yfunc

    a1x = xfunc_pos(1, L_chain) - xfunc_pos(0, L_chain)
    a1y = yfunc_pos(1, L_chain) - yfunc_pos(0, L_chain)
    a2x = xfunc_pos(L_chain, L_chain) - xfunc_pos(0, L_chain)
    a2y = yfunc_pos(L_chain, L_chain) - yfunc_pos(0, L_chain)
    A_cell = abs(a1x * a2y - a1y * a2x)

    # ── projector: build initial guess on CPU, purify on GPU ──────────────────
    printinfo && println("[gpu] Building initial projector guess (CPU)...")
    _ensure_scale!(H)
    P0_cpu = purification_initial_guess(H; ϵF=fermi, maxdim=maxdim, cutoff=cutoff)
    P = _to_gpu_mpo(P0_cpu, gpu_type)

    if method == :mcweeny
        printinfo && println("[gpu] McWeeny purification on GPU...")
        maxiters_mc = 30
        tol_mc      = 1e-5
        for iter in 1:maxiters_mc
            P2   = apply(P, P; ak...)
            ITensorMPS.truncate!(P2; cutoff=Float64(cutoff))
            err  = let diff = +(P2, -1.0 * P; cutoff=1e-12)
                       n = norm(diff); d = norm(P); d > 0 ? n / d : n
                   end
            printinfo && iter % 5 == 0 &&
                println("  McWeeny iter $iter: err=$err  maxlinkdim=$(maxlinkdim(P))")
            err < tol_mc && break
            P_inte = +(3.0 * P, -2.0 * P2; cutoff=Float64(cutoff))
            P = apply(P, P_inte; ak...)
            ITensorMPS.truncate!(P; cutoff=Float64(cutoff))
            _gpu_gc!()
        end
        H._density_cache = nothing   # don't cache GPU MPO in CPU field
    elseif method == :sp2
        Nel_val = Nel === nothing ? H.N ÷ 2 : Int(Nel)
        printinfo && println("[gpu] SP2 purification on GPU (Nel=$Nel_val)...")
        maxiters_sp = 40
        tol_sp      = 1e-5
        for iter in 1:maxiters_sp
            P2  = apply(P, P; ak...)
            ITensorMPS.truncate!(P2; cutoff=Float64(cutoff))
            err = let diff = +(P2, -1.0 * P; cutoff=1e-12)
                      n = norm(diff); d = norm(P); d > 0 ? n / d : n
                  end
            printinfo && println("  SP2 iter $iter: err=$err  maxlinkdim=$(maxlinkdim(P))")
            err < tol_sp && break
            tr_P2 = real(tr(P2))
            if tr_P2 >= Nel_val
                P = P2
            else
                P = +(2.0 * P, -1.0 * P2; ak...)
                ITensorMPS.truncate!(P; cutoff=Float64(cutoff))
            end
            _gpu_gc!()
        end
    elseif method == :KPM
        # KPM: use CPU projector, just move to GPU
        P_cpu = _get_projector(H; method=:KPM, fermi=fermi, Nchebychev=Nchebychev,
                               maxdim=maxdim, cutoff=cutoff)
        P = _to_gpu_mpo(P_cpu, gpu_type)
    else
        error("get_C_gpu: unknown method :$method. Choose :mcweeny, :sp2, or :KPM")
    end
    printinfo && println("[gpu] Projector ready, maxlinkdim=$(maxlinkdim(P))")

    # ── Q = I − P on GPU ──────────────────────────────────────────────────────
    I_gpu = _to_gpu_mpo(MPO(collect(sites), "Id"), gpu_type)
    Q = +(I_gpu, -1.0 * P; ak...)
    ITensorMPS.truncate!(Q; cutoff=Float64(cutoff))
    _gpu_gc!()

    # ── basis MPS closure (returns GPU MPS) ───────────────────────────────────
    make_alpha_gpu = if has_sub
        all_sites = collect(sites)
        alpha -> begin
            n_cell   = (alpha - 1) ÷ n_sub
            sub      = (alpha - 1) % n_sub + 1
            pos_bits = [((n_cell >> (L - i)) & 1) + 1 for i in 1:L]
            _to_gpu_mps(_product_state_mps(all_sites, [pos_bits; sub]), gpu_type)
        end
    else
        alpha -> _to_gpu_mps(binary_to_MPS(alpha - 1, L, collect(sites)), gpu_type)
    end

    if quenched
        # ── position operators on GPU ──────────────────────────────────────────
        sinX_gpu = _to_gpu_mpo(has_sub ?
            postpend_op(get_sinx_op(L, pos_sites, L_chain, Λ_val, xfunc_pos), sub_s, I_mat) :
            get_sinx_op(L, pos_sites, L_chain, Λ_val, xfunc_pos), gpu_type)
        cosX_gpu = _to_gpu_mpo(has_sub ?
            postpend_op(get_cosx_op(L, pos_sites, L_chain, Λ_val, xfunc_pos), sub_s, I_mat) :
            get_cosx_op(L, pos_sites, L_chain, Λ_val, xfunc_pos), gpu_type)
        sinY_gpu = _to_gpu_mpo(has_sub ?
            postpend_op(get_siny_op(L, pos_sites, L_chain, Λ_val, yfunc_pos), sub_s, I_mat) :
            get_siny_op(L, pos_sites, L_chain, Λ_val, yfunc_pos), gpu_type)
        cosY_gpu = _to_gpu_mpo(has_sub ?
            postpend_op(get_cosy_op(L, pos_sites, L_chain, Λ_val, yfunc_pos), sub_s, I_mat) :
            get_cosy_op(L, pos_sites, L_chain, Λ_val, yfunc_pos), gpu_type)
        printinfo && println("[gpu] Position operators on GPU.")

        # ── 8 intermediate MPO products ────────────────────────────────────────
        sinY_P = apply(sinY_gpu, P; ak...); cosY_P = apply(cosY_gpu, P; ak...)
        P_sinX = apply(P, sinX_gpu; ak...); P_cosX = apply(P, cosX_gpu; ak...)
        sinY_Q = apply(sinY_gpu, Q; ak...); cosY_Q = apply(cosY_gpu, Q; ak...)
        Q_sinX = apply(Q, sinX_gpu; ak...); Q_cosX = apply(Q, cosX_gpu; ak...)
        printinfo && println("[gpu] 8 intermediate MPO products done.")
        _gpu_gc!()

        # C1 = Q sinX P sinY Q − P sinX Q sinY P
        C1 = +(apply(apply(Q_sinX, P; ak...), sinY_Q; ak...),
               -apply(apply(P_sinX, Q; ak...), sinY_P; ak...); ak...)
        ITensorMPS.truncate!(C1; cutoff=Float64(cutoff))
        printinfo && println("[gpu] C1 done, maxlinkdim=$(maxlinkdim(C1))")
        _gpu_gc!()

        # C2 = Q cosX P cosY Q − P cosX Q cosY P
        C2 = +(apply(apply(Q_cosX, P; ak...), cosY_Q; ak...),
               -apply(apply(P_cosX, Q; ak...), cosY_P; ak...); ak...)
        ITensorMPS.truncate!(C2; cutoff=Float64(cutoff))
        printinfo && println("[gpu] C2 done, maxlinkdim=$(maxlinkdim(C2))")
        _gpu_gc!()

        # C3 = Q sinX P cosY Q − P sinX Q cosY P
        C3 = +(apply(apply(Q_sinX, P; ak...), cosY_Q; ak...),
               -apply(apply(P_sinX, Q; ak...), cosY_P; ak...); ak...)
        ITensorMPS.truncate!(C3; cutoff=Float64(cutoff))
        printinfo && println("[gpu] C3 done, maxlinkdim=$(maxlinkdim(C3))")
        _gpu_gc!()

        # C4 = Q cosX P sinY Q − P cosX Q sinY P
        C4 = +(apply(apply(Q_cosX, P; ak...), sinY_Q; ak...),
               -apply(apply(P_cosX, Q; ak...), sinY_P; ak...); ak...)
        ITensorMPS.truncate!(C4; cutoff=Float64(cutoff))
        printinfo && println("[gpu] C4 done. Closure ready.")
        _gpu_gc!()

        calculate_chern_number = uc -> begin
            sum(sub -> begin
                alpha    = (uc - 1) * n_sub + sub
                α        = make_alpha_gpu(alpha)
                x        = xfunc(alpha - 1, L_chain)
                y        = yfunc(alpha - 1, L_chain)
                cos_x, sin_x = cos(x / Λ_val), sin(x / Λ_val)
                cos_y, sin_y = cos(y / Λ_val), sin(y / Λ_val)
                ch  =  cos_x * cos_y * inner(α', C1, α)
                ch +=  sin_x * sin_y * inner(α', C2, α)
                ch -=  cos_x * sin_y * inner(α', C3, α)
                ch -=  sin_x * cos_y * inner(α', C4, α)
                ch * 2im * π * Λ_val^2
            end, 1:n_sub) / A_cell
        end

    else
        # flat (non-quenched) mode
        x_op = has_sub ?
            postpend_op(get_diagonal_mpo(L, pos_sites, i -> xfunc_pos(i-1, L_chain)), sub_s, I_mat) :
            get_diagonal_mpo(L, pos_sites, i -> xfunc_pos(i-1, L_chain))
        y_op = has_sub ?
            postpend_op(get_diagonal_mpo(L, pos_sites, i -> yfunc_pos(i-1, L_chain)), sub_s, I_mat) :
            get_diagonal_mpo(L, pos_sites, i -> yfunc_pos(i-1, L_chain))
        x_gpu = _to_gpu_mpo(x_op, gpu_type)
        y_gpu = _to_gpu_mpo(y_op, gpu_type)

        T1 = apply(Q, apply(x_gpu, apply(P, apply(y_gpu, Q; ak...); ak...); ak...); ak...)
        T2 = apply(P, apply(x_gpu, apply(Q, apply(y_gpu, P; ak...); ak...); ak...); ak...)
        C_op = 2im * π * +(T1, -1.0 * T2; ak...)
        ITensorMPS.truncate!(C_op; cutoff=Float64(cutoff))
        _gpu_gc!()

        calculate_chern_number = uc -> begin
            sum(sub -> begin
                alpha = (uc - 1) * n_sub + sub
                α     = make_alpha_gpu(alpha)
                inner(α', C_op, α)
            end, 1:n_sub) / A_cell
        end
    end

    return calculate_chern_number
end


# ============================================================
# GPU magnetic Hubbard SCF
# ============================================================

# GPU McWeeny purification of a (rescaled) single-channel Hamiltonian.
# Builds the initial guess on CPU, moves it to GPU, iterates the McWeeny map
# on GPU (F32), and returns the purified density matrix back on CPU
# (ComplexF64). Mirrors the purification loop in `get_C_gpu`.
function _mcweeny_purify_gpu(H::TBHamiltonian; ϵF::Real = 0.0,
                              fermi::Union{Nothing,Real} = nothing,
                              maxdim::Int, cutoff::Real,
                              maxiters::Int, tol::Real,
                              return_gpu::Bool = false)
    ak     = (cutoff = Float64(cutoff), maxdim = maxdim)
    epsF   = fermi === nothing ? ϵF : Float64(fermi)
    P0_cpu = purification_initial_guess(H; ϵF=epsF, maxdim=maxdim, cutoff=Float64(cutoff))
    P      = _to_gpu_mpo(P0_cpu)
    for iter in 1:maxiters
        P2  = apply(P, P; ak...)
        ITensorMPS.truncate!(P2; cutoff=Float64(cutoff))
        err = let diff = +(P2, -1.0 * P; cutoff=1e-12)
                  n = norm(diff); d = norm(P); d > 0 ? n / d : n
              end
        err < tol && break
        P_inte = +(3.0 * P, -2.0 * P2; cutoff=Float64(cutoff))
        P = apply(P, P_inte; ak...)
        ITensorMPS.truncate!(P; cutoff=Float64(cutoff))
        _gpu_gc!()
    end
    return return_gpu ? P : _to_cpu_mpo(P)
end

function _purification_initial_guess_gpu(H_mpo_gpu::MPO, sites;
                                         ϵF::Real,
                                         scale::Real,
                                         center::Real = 0.0,
                                         maxdim::Int,
                                         cutoff::Real,
                                         Id_gpu::Union{Nothing,MPO} = nothing)
    scale == 0 && error("_purification_initial_guess_gpu: scale must be non-zero.")
    Id = Id_gpu === nothing ? _to_gpu_mpo(MPO(collect(sites), "Id")) : Id_gpu
    coeff_I = 0.5 + (ϵF + center) / (2 * scale)
    coeff_H = -0.5 / scale
    ρ0 = +(coeff_I * Id, coeff_H * H_mpo_gpu; cutoff=Float64(cutoff))
    ITensorMPS.truncate!(ρ0; maxdim=maxdim, cutoff=Float64(cutoff))
    return ρ0
end

function _mcweeny_purify_mpo_gpu(H_mpo_gpu::MPO, sites;
                                 ϵF::Real,
                                 scale::Real,
                                 center::Real = 0.0,
                                 Id_gpu::Union{Nothing,MPO} = nothing,
                                 maxdim::Int,
                                 cutoff::Real,
                                 maxiters::Int,
                                 tol::Real)
    ak = (cutoff = Float64(cutoff), maxdim = maxdim)
    P = _purification_initial_guess_gpu(H_mpo_gpu, sites;
        ϵF=ϵF, scale=scale, center=center, maxdim=maxdim,
        cutoff=cutoff, Id_gpu=Id_gpu)
    for iter in 1:maxiters
        P2 = apply(P, P; ak...)
        ITensorMPS.truncate!(P2; cutoff=Float64(cutoff))
        err = let diff = +(P2, -1.0 * P; cutoff=1e-12)
            n = norm(diff); d = norm(P); d > 0 ? n / d : n
        end
        err < tol && break
        P_inte = +(3.0 * P, -2.0 * P2; cutoff=Float64(cutoff))
        P = apply(P, P_inte; ak...)
        ITensorMPS.truncate!(P; cutoff=Float64(cutoff))
        _gpu_gc!()
    end
    return P
end

"""
    scf_magnetic_hubbard_gpu(H0, U; kwargs...) -> NamedTuple

GPU-accelerated two-channel collinear magnetic mean-field loop for the on-site
Hubbard model. The SCF iteration keeps the density profiles, Hartree MPOs,
Hamiltonian MPOs, density matrices, RMS checks, and mixing on GPU; CPU objects
are built only for initialization and for the compatibility fields returned at
the end.

```text
H_up = H0_up + U·diag(n_dn − background)
H_dn = H0_dn + U·diag(n_up   − background)
```

Only `density_method=:mcweeny` is supported here (grand-canonical at `fermi`);
for particle-number-fixed SP2 use the CPU `scf_magnetic_hubbard`. A concrete
purification `scale` is required so the GPU initial guess can be formed without
estimating spectral bounds on CPU during the loop.

ComplexF32 eigen-decompositions can NaN at very tight cutoffs; a warning is
emitted if `cutoff < 1e-5`, but the requested `cutoff` is used as-is.

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
    cutoff < 1e-5 && @warn "scf_magnetic_hubbard_gpu: cutoff=$cutoff is below 1e-5; ComplexF32 eigen-decomposition may produce NaN — consider cutoff ≥ 1e-5."

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

    rho_up_gpu = _to_gpu_mps(rho_up, gpu_type)
    rho_dn_gpu = _to_gpu_mps(rho_dn, gpu_type)
    bg_gpu = _to_gpu_mps(constant_mps(collect(sites), background), gpu_type)
    H0_up_gpu = _to_gpu_mpo(H0_up.mpo, gpu_type)
    H0_dn_gpu = _to_gpu_mpo(H0_dn.mpo, gpu_type)
    Id_gpu = _to_gpu_mpo(MPO(collect(sites), "Id"), gpu_type)
    U_gpu = U isa MPO ? _to_gpu_mpo(U, gpu_type) : nothing

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

# Thin 2D-grid wrapper around the shared geometry-aware planner (Utils.jl).
# Used by get_scf_magnetization_gpu. In :point mode, `groups` lists the
# sampled cells explicitly. In :block mode, the groups are nominal centers and
# the caller should use plan.a/plan.b with _eval_block_mps_gpu.
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

function _tb_spatial_groups_gpu(sites; kwargs...)
    plan = _tb_spatial_plan_gpu(sites; kwargs...)
    return plan.centers, plan.groups
end

"""
    get_scf_magnetization_gpu(res; kwargs...) -> (values, centers, groups, n_up, n_dn)

Sample the converged magnetic SCF density matrices on GPU and extract only the
final scalar values. If `res` carries GPU density MPOs from
`scf_magnetic_hubbard_gpu`, they are reused directly; otherwise the CPU density
MPOs are uploaded once. Each sampled point is evaluated in the same big-endian
real-space convention as `binary_to_MPS`.

Set `reduce=:block` to average over every unit cell in each coarse block by
tracing the within-block position bits on GPU. In block mode, `num_x` and
`num_y` must be powers of two.
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
keeps the Chebyshev/QFT accumulation on GPU, extracting only scalars.
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
