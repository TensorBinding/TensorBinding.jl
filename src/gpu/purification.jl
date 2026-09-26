# ============================================================
# purification.jl — GPU McWeeny purification
# ============================================================
# Moved from gpu/GPU_tk.jl: _mcweeny_purify_gpu and the GPU-resident
# _mcweeny_purify_mpo_gpu with its initial guess. The GPU SP2 loop is inline
# in get_C_gpu (topology.jl).

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
