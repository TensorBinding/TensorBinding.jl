# gpu/primitives.jl — GPU-safe building blocks shared by the src/gpu/ entry points:
# the QFT sandwich used by get_bands_gpu, a dense GPU delta, MPS element evaluation
# (point, block, all-sites sum), diagonal extraction and density profiles,
# diagonal-MPO embedding and auxiliary-site projection. Moved from the former
# gpu/GPU_tk.jl. The evaluators, the extraction and the embedding are thin wrappers
# over the CPU kernels of core/Utils.jl (_eval_block_mps, _extract_diagonal,
# _mps_to_diagonal) with `to_device = _to_gpu`.
#
# Main entry points: extract_diagonal_to_mps_gpu and density_profile_from_dm_gpu
# (documented); internal: _apply_qft_conj_gpu, _make_delta_gpu, the _eval_*_gpu
# samplers, _mps_to_diagonal_mpo_gpu, _project_aux_gpu.
# Depends on: core/Utils.jl (the kernels above, constant_mps), gpu/device.jl (CUDA
# bridge, uploads).


# ============================================================
# 1. QFT sandwich
# ============================================================

# Apply the QFT sandwich U·W·U† on GPU using pre-built GPU QFT operators.
# Returns a GPU MPO with the element type of its inputs.
function _apply_qft_conj_gpu(W::MPO, FTirev_gpu::MPO, FTrev_gpu::MPO;
                              tol::Real = 1e-9, maxdim::Int = 100)
    Op1 = apply(W, FTirev_gpu; cutoff=tol, maxdim=maxdim)
    Op2 = apply(swapprime(FTrev_gpu, 0 => 1), Op1; cutoff=tol, maxdim=maxdim)
    return ITensorMPS.truncate!(Op2; cutoff=tol, maxdim=maxdim)
end


# ============================================================
# 2. Dense GPU delta
# ============================================================

# A dense ComplexF32 GPU delta (delta() is a CPU DiagTensor{Float64}); the Hadamard
# product of gpu/conductivity.jl contracts it into ComplexF32 GPU tensors.
_make_delta_gpu(i::Index, j::Index, k::Index) = _to_gpu(delta(i, j, k), ComplexF32)


# ============================================================
# 3. MPS element evaluation (point, block, all-sites sum)
# ============================================================

# Evaluate an MPS element at bit-index `idx` entirely on GPU using the legacy
# LSB-first convention used by the GPU QFT/bands path.
#
# Basis vectors are built as explicit dense arrays matching the element type of
# A so that the contraction is GPU×GPU with a consistent dtype throughout.
function _eval_diag_mps_gpu(A::MPS, idx::Int)
    s    = siteinds(A)
    ElT  = eltype(A[1])
    acc  = _to_gpu(ITensor(one(ElT)), ElT)
    for i in 1:length(s)
        b     = (idx >> (i - 1)) & 1
        v_arr = zeros(ElT, dim(s[i]))
        v_arr[b + 1] = one(real(ElT))
        v   = _to_gpu(ITensor(v_arr, s[i]), ElT)
        acc *= A[i] * v
    end
    return real(scalar(acc))
end

# The real-space samplers are the CPU block evaluator _eval_block_mps (core/Utils.jl)
# on GPU vectors. Big-endian site order [iy_MSB..iy_LSB, ix_MSB..ix_LSB]; the kept
# top bits are pinned to the pixel, the lower ones summed with [1, 1].
#   _eval_block_mps_gpu            one coarse 2D block (reduce=:block)
#   _eval_block_mps_1d_gpu         one 1D block: the top `a` of the L bits pinned
#   _eval_mps_bigendian_gpu        one element (every bit pinned), matching
#                                  binary_to_MPS/eval_mps
#   _eval_fullsum_mps_1d_gpu       the sum of all elements (no bit pinned), the
#                                  trace of the NH diagonal-trace DOS (gpu/nh.jl)
# The `_complex` variants return the ComplexF64 amplitude instead of its real part.
_eval_block_mps_gpu(A::MPS, ixp::Int, iyp::Int, a::Int, b::Int, Lx::Int, Ly::Int) =
    _eval_block_mps(A, ixp, iyp, a, b, Lx, Ly; to_device=_to_gpu)

function _eval_block_mps_1d_gpu(A::MPS, ixp::Int, a::Int, L::Int)
    length(A) == L || error("_eval_block_mps_1d_gpu: MPS has $(length(A)) sites but L=$L.")
    return _eval_block_mps(A, ixp, 0, a, 0, L, 0; to_device=_to_gpu)
end

function _eval_block_mps_1d_complex_gpu(A::MPS, ixp::Int, a::Int, L::Int)
    length(A) == L || error("_eval_block_mps_1d_complex_gpu: MPS has $(length(A)) sites but L=$L.")
    return _eval_block_mps(A, ixp, 0, a, 0, L, 0; to_device=_to_gpu, value=ComplexF64)
end

_eval_mps_bigendian_gpu(A::MPS, idx::Int) =
    _eval_block_mps(A, idx, 0, length(A), 0, length(A), 0; to_device=_to_gpu)

_eval_mps_bigendian_complex_gpu(A::MPS, idx::Int) =
    _eval_block_mps(A, idx, 0, length(A), 0, length(A), 0; to_device=_to_gpu, value=ComplexF64)

_eval_fullsum_mps_1d_gpu(A::MPS) = _eval_block_mps(A, 0, 0, 0, 0, length(A), 0; to_device=_to_gpu)


# ============================================================
# 4. Diagonal extraction, density profiles and diagonal MPOs
# ============================================================

"""
    extract_diagonal_to_mps_gpu(M::MPO) -> MPS

GPU-resident analogue of `extract_diagonal_to_mps`. `M` is expected to already
be a GPU MPO. The returned MPS stays on GPU, with one-hot tensors matched to
the input tensor element type.
"""
function extract_diagonal_to_mps_gpu(M::MPO)::MPS
    _check_gpu("extract_diagonal_to_mps_gpu")
    return _extract_diagonal(M; to_device=_to_gpu)
end

# The diagonal comes from the shared extraction kernel. The :complement branch is
# not density_profile_from_dm's: that one subtracts without truncation on the sites
# it is given, this one truncates the sum 1 − diag with `maxdim`/`cutoff`, uploads
# the constant profile as ComplexF32 and defaults `sites` to the diagonal's own.
# (Kept above the docstring: a comment in between would detach it.)
"""
    density_profile_from_dm_gpu(density_mpo, sites=nothing; mode=:direct,
                                maxdim=100, cutoff=1e-8) -> MPS

GPU-resident analogue of `density_profile_from_dm`. If `density_mpo` is a CPU
MPO it is uploaded once (as ComplexF32); if it is already on GPU it is used in
place. The returned profile is a GPU MPS. `mode=:complement` returns
`1 - diag(D)` on GPU, summed with `maxdim`/`cutoff` on the sites `sites`
(default: the site indices of the diagonal).
"""
function density_profile_from_dm_gpu(density_mpo::MPO, sites=nothing;
                                     mode::Symbol = :direct,
                                     maxdim::Int = 100,
                                     cutoff::Real = 1e-8)
    _check_gpu("density_profile_from_dm_gpu")
    dm_gpu = _ensure_gpu(density_mpo, ComplexF32; caller="density_profile_from_dm_gpu")
    diag_mps = extract_diagonal_to_mps_gpu(dm_gpu)
    mode === :direct && return diag_mps
    if mode === :complement
        profile_sites = sites === nothing ? collect(siteinds(diag_mps)) : collect(sites)
        one_mps = _to_gpu(constant_mps(profile_sites, 1.0), ComplexF32)
        return +(one_mps, -diag_mps; maxdim=maxdim, cutoff=cutoff)
    end
    error("Unsupported density extraction mode :$mode. Use :direct or :complement.")
end

# GPU analogue of mps_to_diagonal_mpo (core/Utils.jl), through its kernel: dense
# ComplexF32 GPU deltas, and a one-site MPS is accepted.
_mps_to_diagonal_mpo_gpu(mps::MPS, sites)::MPO =
    _mps_to_diagonal(mps, sites; to_device=_to_gpu, delta_type=ComplexF32, one_site=true)


# ============================================================
# 5. Auxiliary-site projection
# ============================================================

# Project one auxiliary site out of a GPU MPO.
# Mirrors project_aux (CPU, core/AuxDOF.jl) but builds a dense projector on GPU,
# with the element type of T, so every contraction stays on the GPU.  The
# contracted site is absorbed into the neighbouring site, returning an MPO with
# one fewer site.  get_bands_gpu and get_ldos_spatial_gpu pass it as the
# `project` step of _project_aux_sectors (core/AuxDOF.jl).
#
# setelt() produces a DiagBlockSparse ITensor that cu() leaves on CPU — we
# therefore build the |sec><sec| projector as an explicit dense array instead.
function _project_aux_gpu(T::MPO, idx::Index, sec::Int; side::Symbol=:post)
    L    = length(T)
    n    = side == :post ? L : 1

    # Build dense |sec><sec| projector matching the element type of T so the
    # contraction is GPU×GPU with a consistent dtype throughout.
    ElT        = eltype(T[n])
    d          = dim(idx)
    proj_arr   = zeros(ElT, d, d)
    proj_arr[sec, sec] = one(ElT)
    proj       = _to_gpu(ITensor(proj_arr, idx, idx'), ElT)
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
