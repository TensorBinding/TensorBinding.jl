# ============================================================
# conductivity.jl — conductivity-only Tucker/QFT/Hadamard block
# ============================================================
# Moved from gpu/GPU_tk.jl. No other code in src/, test/ or examples/ calls
# these helpers (QFT operator build, GPU Hadamard product, weighted MPO sum,
# GPU density matrix, Tucker components); kept in the package for now.

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


# ============================================================
# Shared Tucker component builder
# ============================================================

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
