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
# and the _contract! dispatch fails (all three tensors must share El).
# Fix: materialise the delta as a dense F32 GPU tensor.
function _make_delta_gpu(i::Index, j::Index, k::Index)
    d_dense = dense(delta(i, j, k))          # DiagStorage → DenseStorage
    idx     = inds(d_dense)
    arr     = Array(d_dense, idx...)
    return _tb_cuda_module().cu(ITensor(Float32.(arr), idx))
end

# GPU-safe Hadamard product: identical logic to _hadamard_mpo but uses
# _make_delta_gpu so all contractions stay within {ComplexF32, GPU}.
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

"""
    extract_diagonal_to_mps_gpu(M::MPO) -> MPS

GPU-resident analogue of `extract_diagonal_to_mps`. `M` is expected to already
be a GPU ComplexF32 MPO (e.g. a Chebyshev moment from `KPM_Tn_gpu`, after
`_apply_qft_conj_gpu` and/or `_project_aux_gpu`); the returned MPS is also on
GPU (ComplexF32).
"""
# extract_diagonal_to_mps (in RPA_tk.jl) uses plain onehot() which returns a
# CPU DiagBlockSparse tensor.  Contracting a GPU ComplexF32 MPO tensor with a
# CPU onehot fails (GPU×CPU mismatch).  Here we wrap each onehot call with
# cu() so NDTensors resolves the contraction entirely on the GPU.
# The zero ITensor `res` has no committed storage, so the first `+=` with a
# GPU ComplexF32 result promotes it to the correct GPU type.
function extract_diagonal_to_mps_gpu(M::MPO)::MPS
    cuda = _tb_cuda_module()
    N    = length(M)
    new_tensors = Vector{ITensor}(undef, N)
    for i in 1:N
        t      = M[i]
        s2, s1 = siteinds(M, i)   # s2 = bra (primed), s1 = ket
        d_s    = dim(s1)
        v_inds = uniqueinds(t, s1, s2)

        res = ITensor(v_inds..., s1)   # zero tensor; type determined by first +=
        for v in 1:d_s
            slice = t * cuda.cu(onehot(s1 => v)) * cuda.cu(onehot(s2 => v))
            res  += slice * cuda.cu(onehot(s1 => v))
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
                    verbose::Bool                       = true)

    _check_gpu("KPM_Tn_gpu")

    if isnothing(scale)
        scale, center = _estimate_spectral_bounds(H_mpo, sites;
                             dmrg_nsweeps = dmrg_nsweeps,
                             dmrg_maxdim  = dmrg_maxdim,
                             dmrg_linkdim = dmrg_linkdim)
    end

    I_mpo = MPO(sites, "Id")
    Ham_n = (1 / scale) * +(H_mpo, (-center) * I_mpo; cutoff = cutoff)

    I_mpo = _to_gpu_mpo(I_mpo)
    Ham_n = _to_gpu_mpo(Ham_n)

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

All GPU operators (Hamiltonian, identity, QFT pair, per-step aux projectors)
use ComplexF32 via `_to_gpu_mpo`/`_make_delta_gpu`, matching the rest of this
toolkit (see the PRECISION note at the top of GPU_tk.jl). ComplexF32
eigendecomposition can produce NaN at very tight `cutoff` on large systems —
if `Ak` comes back all-NaN, raise `cutoff` rather than lowering it.

All keyword arguments are identical to the TBHamiltonian overload of
`get_bands`.  The return value is also identical: a plain `Matrix{Float64}`
when no `kpath` is given, or a `NamedTuple(Ak, ticks, labels)` when a
high-symmetry path is requested.

Usage:
```julia
using CUDA
res = TensorBinding.get_bands_gpu(H, 500, omega;
        kpath=[:G, :M, :Kp, :G], kpath_lattice=:honeycomb,
        num_x=50, maxdim=200, printinfo=true)
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
                       printinfo::Bool   = false)

    _check_gpu("get_bands_gpu")

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
            mask_A_gpu = _to_gpu_mpo(_col_select_mpo(L_pos, 0, pos_sites_cpu; keep=:odd))
            mask_B_gpu = _to_gpu_mpo(_col_select_mpo(L_pos, 0, pos_sites_cpu; keep=:even))
        else
            Ly_pos = L_pos - Lx_pos
            mask_A_gpu = _to_gpu_mpo(_row_checker_mpo(Lx_pos, Ly_pos, pos_sites_cpu))
            mask_B_gpu = _to_gpu_mpo(MPO(pos_sites_cpu, "Id") -
                                     _row_checker_mpo(Lx_pos, Ly_pos, pos_sites_cpu))
        end
    end

    # ── GPU Ham and QFT operators ────────────────────────────────────────────
    I_mpo_cpu = MPO(H.sites, "Id")
    Ham_n_cpu = (1 / H.scale) * +(H.mpo, (-H.center) * I_mpo_cpu; cutoff=cutoff)
    I_mpo_gpu = _to_gpu_mpo(I_mpo_cpu)
    Ham_n_gpu = _to_gpu_mpo(Ham_n_cpu)

    # QFT operators sized for pos_sites_cpu (the post-projection site list).
    # Calling fix_sites maps the abstract QFT indices onto the actual pos_sites.
    R_pos      = length(pos_sites_cpu)
    FTirev_gpu = _to_gpu_mpo(fix_sites(
        MPO(TCI.reverse(QuanticsTCI.quanticsfouriermpo(R_pos; sign=-1.0, normalize=true))),
        pos_sites_cpu))
    FTrev_gpu  = _to_gpu_mpo(fix_sites(
        MPO(TCI.reverse(QuanticsTCI.quanticsfouriermpo(R_pos; sign=+1.0, normalize=true))),
        pos_sites_cpu))

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
    #   projection  → _project_aux_gpu  (dense ComplexF32 projector, GPU throughout)
    #   QFT         → _apply_qft_conj_gpu  (pre-built GPU QFT operators)
    #   diagonal    → extract_diagonal_to_mps_gpu (CPU array slice, back to GPU F32)
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

    accumulate_Tn_gpu!(Ak_w, Tkm2, 1)
    accumulate_Tn_gpu!(Ak_w, Tkm1, 2)

    for k in 3:Ncheb
        Tk = +(2 * apply(Ham_n_gpu, Tkm1; cutoff=cutoff, maxdim=maxdim),
               -Tkm2; cutoff=cutoff, maxdim=maxdim)
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
available on GPU; only the single-pass MPO mode is implemented here).

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
                               proj_sl           = nothing)

    _check_gpu("get_ldos_spatial_gpu")

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
    I_mpo_gpu = _to_gpu_mpo(I_mpo_cpu)
    Ham_n_gpu = _to_gpu_mpo(Ham_n_cpu)

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
    cutoff < 1e-6 && @warn "get_ldos_spatial_gpu: cutoff=$cutoff is below 1e-6; ComplexF32 eigendecomposition may produce NaN on large systems — consider cutoff ≥ 1e-4."
    gpu_cutoff = Float64(cutoff)
    Tkm2 = I_mpo_gpu
    Tkm1 = Ham_n_gpu

    accumulate_Tn_ldos_gpu!(accum, Tkm2, 1)
    accumulate_Tn_ldos_gpu!(accum, Tkm1, 2)

    for k in 3:Ncheb
        Tk = +(2 * apply(Ham_n_gpu, Tkm1; cutoff=gpu_cutoff, maxdim=maxdim),
               -Tkm2; cutoff=gpu_cutoff, maxdim=maxdim)
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
                                  proj_sl                  = nothing)

    _check_gpu("get_dos_stochastic_gpu")
    cutoff < 1e-6 && @warn "get_dos_stochastic_gpu: cutoff=$cutoff is below 1e-6; ComplexF32 eigendecomposition may produce NaN on large systems — consider cutoff ≥ 1e-4."
    _ensure_scale!(H)
    dos_weighting in (:trace, :sample) ||
        error("get_dos_stochastic_gpu: dos_weighting must be :trace or :sample.")
    N_sample >= 0 || error("get_dos_stochastic_gpu: N_sample must be non-negative.")
    N_bound >= 0 || error("get_dos_stochastic_gpu: N_bound must be non-negative.")

    I_mpo_cpu = MPO(H.sites, "Id")
    Ham_n_cpu = (1 / H.scale) * +(H.mpo, (-H.center) * I_mpo_cpu; cutoff=cutoff)
    Ham_n_gpu = _to_gpu_mpo(Ham_n_cpu)

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
                    psi0_gpu = _to_gpu_mps(_ldos_make_psi0(H, x, σ_n, σ_s, σ_l, σ_sl))
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
                psi0_gpu = _to_gpu_mps(_exciton_pair_mps_gpu_seed(xe, xh))
                χ = _run_kpm_mps_gpu!(psi0_gpu, accum_full, 1.0/N_sample)
                (verbose || printinfo) && i % 10 == 0 &&
                    println("  [gpu] dos continuum sample $i/$N_sample (xe=$xe, xh=$xh)  maxlinkdim=$χ")
            end
        else
            samples = rand(rng, 0:(D - 1), N_sample)
            for (i, k) in enumerate(samples)
                psi0_gpu = _to_gpu_mps(_basis_state_mps(k, H.sites))
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
            psi0_gpu = _to_gpu_mps(mpsexciton(x, H.sites))
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


"""
    get_exciton_ldos_spatial_gpu(H, Ncheb, ω_phys_vals; X_list, X_groups,
                                 num_x, num_avg, x_start, x_end, kernel,
                                 lambda, eta, m_order, maxdim, cutoff, verbose, printinfo)
        -> Matrix{Float64}   (Nω × n_X)

GPU-accelerated spatial exciton LDOS: for each bound exciton position `X` (electron
= hole = X, 1-indexed in `1:H.N`) the local spectral weight
`A(X, ω) = ⟨X,X|δ(ω−H)|X,X⟩` is reconstructed from the Chebyshev moments
`μ_n = ⟨X,X|T_n(H̃)|X,X⟩`.  One GPU MPS Chebyshev recursion is run per X starting
from `|X,X⟩ = mpsexciton(X, H.sites)`; moments are scalars pulled to CPU.

This is the spatial / batched GPU analog of the CPU `get_exciton_ldos` (same
reconstruction via the KPM weight matrix), columns ordered as `X_list` or as
the coarse centers of the generated spatial groups.

`X_list` selects the positions directly. `X_groups` selects explicit position
groups and averages all probes inside each group into one output column. If
neither is provided, `num_x` coarse groups are built over `x_start:x_end`; each
group contains `num_avg` subpositions with the same stride convention as
`get_ldos_spatial`. All positions are 1-indexed in `1:H.N`.

`kernel=:hodc` selects the Higher-Order Delta Chebyshev reconstruction (`eta`,
`m_order`); its weights carry the full normalisation (no `√(1−ω²)` denominator).
`eta=0` falls back to `1/(Ncheb+1)`.  Otherwise `kernel` is a convolution kernel
(`:jackson` default, `:lorentz` with `lambda`, `:fejer`, `:dirichlet`).
"""
function get_exciton_ldos_spatial_gpu(H::TBHamiltonian, Ncheb::Int, ω_phys_vals;
                                       X_list           = nothing,
                                       X_groups         = nothing,
                                       x_groups         = nothing,
                                       num_x::Int       = H.N,
                                       num_avg::Int     = 1,
                                       x_start::Int     = 1,
                                       x_end::Int       = H.N,
                                       kernel::Symbol   = :jackson,
                                       lambda::Real     = 4.0,
                                       eta::Real        = 0.0,
                                       m_order::Int     = 4,
                                       maxdim::Int      = 100,
                                       cutoff::Real     = 1e-8,
                                       verbose::Bool    = false,
                                       printinfo::Bool  = false)

    _check_gpu("get_exciton_ldos_spatial_gpu")
    cutoff < 1e-6 && @warn "get_exciton_ldos_spatial_gpu: cutoff=$cutoff is below 1e-6; ComplexF32 eigendecomposition may produce NaN on large systems — consider cutoff ≥ 1e-4."
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
        num_x > 0 || error("get_exciton_ldos_spatial_gpu: num_x must be positive.")
        num_avg > 0 || error("get_exciton_ldos_spatial_gpu: num_avg must be positive.")
        1 <= x_start <= x_end <= H.N ||
            error("get_exciton_ldos_spatial_gpu: expected 1 <= x_start <= x_end <= H.N.")
        window = x_end - x_start + 1
        num_x <= window ||
            error("get_exciton_ldos_spatial_gpu: num_x=$num_x exceeds sampling window length $window.")
        dx     = div(window, num_x)
        dx_sub = max(1, div(dx, num_avg))
        [[x_start + (i - 1) * dx + k * dx_sub
          for k in 0:num_avg-1
          if x_start + (i - 1) * dx + k * dx_sub <= x_end]
         for i in 1:num_x]
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
    Ham_n_gpu = _to_gpu_mpo(Ham_n_cpu)

    ω_vals = (collect(ω_phys_vals) .- H.center) ./ H.scale
    Nω     = length(ω_vals)
    W, denom = _dos_weight_matrix(Ncheb, ω_vals;
                                  kernel=kernel, lambda=lambda, eta=eta, m_order=m_order)
    valid  = [abs(ω) < 1.0 for ω in ω_vals]

    nX           = length(groups)
    result       = zeros(Float64, Nω, nX)
    apply_kwargs = (cutoff=Float64(cutoff), maxdim=maxdim)

    for (j, group) in enumerate(groups)
        last_linkdim = 0

        for X in group
            psi0_gpu = _to_gpu_mps(mpsexciton(X, H.sites))
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
        (verbose || printinfo) && (j % 5 == 0 || j == nX) &&
            println("  [gpu] exciton ldos $j/$nX (X=$(Xs[j]), n_avg=$(length(group)))  maxlinkdim=$last_linkdim")
    end

    return result
end


# ============================================================
# GPU Chern marker
# ============================================================

"""
    get_C_gpu(H::TBHamiltonian, xfunc=nothing, yfunc=nothing; kwargs...) -> Function

GPU-accelerated real-space Chern marker.  Mirrors `get_C` exactly but runs all
MPO×MPO products (projector assembly and C1–C4 construction) on GPU in F32.

Returns the same closure `C_at(uc::Int) -> ComplexF64` as `get_C`.

# Key differences from `get_C`
- All `apply`/`truncate!` operations run on GPU tensors (F32).
- `cutoff` is passed directly; a warning is emitted if `cutoff < 1e-6` since
  ComplexF32 eigendecompositions can produce NaN on large systems at tight cutoffs.
- The projector is built on CPU first (via `_get_projector`), then moved to GPU.
  For method=:mcweeny this means the purification loop runs on GPU.
- `sequential` mode is not supported (non-sequential quenched is always used).

All keyword arguments are identical to `get_C`.
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
                   printinfo::Bool  = false)

    _check_gpu("get_C_gpu")
    cutoff < 1e-6 && @warn "get_C_gpu: cutoff=$cutoff is below 1e-6; ComplexF32 eigendecomposition may produce NaN on large systems — consider cutoff ≥ 1e-4."
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
    P = _to_gpu_mpo(P0_cpu)

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
        P = _to_gpu_mpo(P_cpu)
    else
        error("get_C_gpu: unknown method :$method. Choose :mcweeny, :sp2, or :KPM")
    end
    printinfo && println("[gpu] Projector ready, maxlinkdim=$(maxlinkdim(P))")

    # ── Q = I − P on GPU ──────────────────────────────────────────────────────
    I_gpu = _to_gpu_mpo(MPO(collect(sites), "Id"))
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
            _to_gpu_mps(_product_state_mps(all_sites, [pos_bits; sub]))
        end
    else
        alpha -> _to_gpu_mps(binary_to_MPS(alpha - 1, L, collect(sites)))
    end

    if quenched
        # ── position operators on GPU ──────────────────────────────────────────
        sinX_gpu = _to_gpu_mpo(has_sub ?
            postpend_op(get_sinx_op(L, pos_sites, L_chain, Λ_val, xfunc_pos), sub_s, I_mat) :
            get_sinx_op(L, pos_sites, L_chain, Λ_val, xfunc_pos))
        cosX_gpu = _to_gpu_mpo(has_sub ?
            postpend_op(get_cosx_op(L, pos_sites, L_chain, Λ_val, xfunc_pos), sub_s, I_mat) :
            get_cosx_op(L, pos_sites, L_chain, Λ_val, xfunc_pos))
        sinY_gpu = _to_gpu_mpo(has_sub ?
            postpend_op(get_siny_op(L, pos_sites, L_chain, Λ_val, yfunc_pos), sub_s, I_mat) :
            get_siny_op(L, pos_sites, L_chain, Λ_val, yfunc_pos))
        cosY_gpu = _to_gpu_mpo(has_sub ?
            postpend_op(get_cosy_op(L, pos_sites, L_chain, Λ_val, yfunc_pos), sub_s, I_mat) :
            get_cosy_op(L, pos_sites, L_chain, Λ_val, yfunc_pos))
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
        x_gpu = _to_gpu_mpo(x_op)
        y_gpu = _to_gpu_mpo(y_op)

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
function _mcweeny_purify_gpu(H::TBHamiltonian; ϵF::Real,
                              maxdim::Int, cutoff::Real,
                              maxiters::Int, tol::Real,
                              return_gpu::Bool = false)
    ak     = (cutoff = Float64(cutoff), maxdim = maxdim)
    P0_cpu = purification_initial_guess(H; ϵF=ϵF, maxdim=maxdim, cutoff=Float64(cutoff))
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
                                  verbose::Bool = true)
    _check_gpu("scf_magnetic_hubbard_gpu")
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

    rho_up_gpu = _to_gpu_mps(rho_up)
    rho_dn_gpu = _to_gpu_mps(rho_dn)
    bg_gpu = _to_gpu_mps(constant_mps(collect(sites), background))
    H0_up_gpu = _to_gpu_mpo(H0_up.mpo)
    H0_dn_gpu = _to_gpu_mpo(H0_dn.mpo)
    Id_gpu = _to_gpu_mpo(MPO(collect(sites), "Id"))
    U_gpu = U isa MPO ? _to_gpu_mpo(U) : nothing

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
# Used by get_scf_magnetization_gpu; returns (centers, groups) of unit-cell
# indices laid out on a num_x × num_y grid (or x_groups override).
function _tb_spatial_groups_gpu(sites;
                                num_x::Int = 0,
                                num_y::Union{Nothing,Int} = nothing,
                                num_avg::Int = 1,
                                x_start::Int = 1,
                                x_end::Int = prod(dim(s) for s in sites),
                                x_groups = nothing,
                                box_half::Int = 0,
                                Lx::Union{Nothing,Int} = nothing)
    L = length(sites)
    plan = spatial_sampling_plan(L;
        Lx       = something(Lx, div(L, 2)),
        grid     = x_groups === nothing,
        num_x    = num_x, num_y = num_y, num_avg = num_avg,
        x_start  = x_start, x_end = x_end,
        x_groups = x_groups, box_half = box_half)
    return plan.centers, plan.groups
end

"""
    get_scf_magnetization_gpu(res; kwargs...) -> (values, centers, groups, n_up, n_dn)

Sample the converged magnetic SCF density matrices on GPU and extract only the
final scalar values. If `res` carries GPU density MPOs from
`scf_magnetic_hubbard_gpu`, they are reused directly; otherwise the CPU density
MPOs are uploaded once. Each sampled point is evaluated in the same big-endian
real-space convention as `binary_to_MPS`.
"""
function get_scf_magnetization_gpu(res;
                                   num_x::Int = 0,
                                   num_y::Union{Nothing,Int} = nothing,
                                   num_avg::Int = 1,
                                   x_start::Int = 1,
                                   x_end::Int = prod(dim(s) for s in res.H_up.sites),
                                   x_groups = nothing,
                                   box_half::Int = 0,
                                   Lx::Union{Nothing,Int} = nothing)
    up_mpo = hasproperty(res, :density_up_mpo_gpu) && res.density_up_mpo_gpu !== nothing ?
        res.density_up_mpo_gpu : res.density_up_mpo
    dn_mpo = hasproperty(res, :density_dn_mpo_gpu) && res.density_dn_mpo_gpu !== nothing ?
        res.density_dn_mpo_gpu : res.density_dn_mpo

    up_mpo === nothing &&
        error("get_scf_magnetization_gpu: res.density_up_mpo is missing.")
    dn_mpo === nothing &&
        error("get_scf_magnetization_gpu: res.density_dn_mpo is missing.")

    sites = res.H_up.sites
    centers, groups = _tb_spatial_groups_gpu(sites;
        num_x=num_x, num_y=num_y, num_avg=num_avg, x_start=x_start, x_end=x_end,
        x_groups=x_groups, box_half=box_half, Lx=Lx)

    up_diag_gpu = density_profile_from_dm_gpu(up_mpo, sites)
    dn_diag_gpu = density_profile_from_dm_gpu(dn_mpo, sites)

    n_up = Float64[
        sum(_eval_mps_bigendian_gpu(up_diag_gpu, x - 1) for x in grp) / length(grp)
        for grp in groups
    ]
    n_dn = Float64[
        sum(_eval_mps_bigendian_gpu(dn_diag_gpu, x - 1) for x in grp) / length(grp)
        for grp in groups
    ]
    values = (n_up .- n_dn) ./ 2
    _gpu_gc!()
    return (values=values, centers=centers, groups=groups, n_up=n_up, n_dn=n_dn)
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
