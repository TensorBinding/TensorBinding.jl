# ============================================================
# primitives.jl — GPU-safe primitives
# ============================================================
# Moved from the former gpu/GPU_tk.jl: the QFT sandwich used by get_bands_gpu, dense GPU
# delta/one-hot tensors, MPS element evaluation (point, block, all-sites sum),
# diagonal extraction and density profiles, diagonal-MPO embedding and
# auxiliary-site projection.

# Apply the QFT sandwich U·W·U† on GPU using pre-built GPU QFT operators.
# Returns a GPU F32 MPO.
function _apply_qft_conj_gpu(W::MPO, FTirev_gpu::MPO, FTrev_gpu::MPO;
                              tol::Real = 1e-9, maxdim::Int = 100)
    Op1 = apply(W, FTirev_gpu; cutoff=tol, maxdim=maxdim)
    Op2 = apply(swapprime(FTrev_gpu, 0 => 1), Op1; cutoff=tol, maxdim=maxdim)
    return ITensorMPS.truncate!(Op2; cutoff=tol, maxdim=maxdim)
end

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

# Dense GPU one-hot vector on `p.first` with element type `T`, real or complex,
# so it matches the tensor it is contracted with (see extract_diagonal_to_mps_gpu).
function _onehot_gpu(p::Pair{<:Index,<:Integer}, T::Type{<:Number}=ComplexF32)
    i = p.first
    v = Int(p.second)
    1 <= v <= dim(i) || error("_onehot_gpu: state $v is outside index dimension $(dim(i)).")
    arr = zeros(T, dim(i))
    arr[v] = one(T)
    return ITensors.itensor(_tb_cuda_module().CuArray(arr), i)
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

# extract_diagonal_to_mps (in core/Utils.jl) uses plain onehot() which returns a
# CPU DiagBlockSparse tensor.  Contracting a GPU MPO tensor with a CPU onehot
# fails (GPU×CPU mismatch). Here the one-hot basis vectors are explicitly dense
# GPU tensors with the same element type as the input MPO tensor.
# (Kept above the docstring: a comment in between would detach it.)
"""
    extract_diagonal_to_mps_gpu(M::MPO) -> MPS

GPU-resident analogue of `extract_diagonal_to_mps`. `M` is expected to already
be a GPU MPO. The returned MPS stays on GPU, with one-hot tensors matched to
the input tensor element type.
"""
function extract_diagonal_to_mps_gpu(M::MPO)::MPS
    _check_gpu("extract_diagonal_to_mps_gpu")
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
