# gpu/purification.jl — GPU McWeeny purification: _mcweeny_purify_gpu (from a
# TBHamiltonian, initial guess built on CPU) and the fully GPU-resident
# _mcweeny_purify_mpo_gpu with its initial guess _purification_initial_guess_gpu.
# Moved from the former gpu/GPU_tk.jl. The loop is the CPU kernel _mcweeny_iterate
# and the guess the CPU kernel _linear_density_guess (physics/Purification.jl), run
# on GPU MPOs; chern_marker_gpu (gpu/topology.jl) calls the same McWeeny and SP2 kernels.
#
# Main entry points (internal): _mcweeny_purify_mpo_gpu (used by
# scf_magnetic_hubbard_gpu in gpu/scf.jl) and _mcweeny_purify_gpu (called by
# examples/manuscript_files/scripts/nonequilibrium/manuscript_nhdens_gpu.jl).
# Depends on: core/TBSystem.jl (TBHamiltonian), physics/Purification.jl
# (purification_initial_guess, _mcweeny_iterate, _linear_density_guess),
# gpu/device.jl.


# ============================================================
# 1. McWeeny from a TBHamiltonian (CPU initial guess)
# ============================================================

# GPU McWeeny purification of a (rescaled) single-channel Hamiltonian.
# Builds the initial guess on CPU, moves it to GPU (ComplexF32), iterates the
# McWeeny map on GPU, and returns the purified density matrix back on CPU
# (ComplexF64), or on GPU with `return_gpu=true`. The GPU McWeeny steps truncate
# with `cutoff` only (the CPU ones with `maxdim` too) and free GPU memory after
# each iteration.
function _mcweeny_purify_gpu(H::TBHamiltonian; ϵF::Real = 0.0,
                              fermi::Union{Nothing,Real} = nothing,
                              maxdim::Int, cutoff::Real,
                              maxiters::Int, tol::Real,
                              return_gpu::Bool = false)
    epsF   = fermi === nothing ? ϵF : Float64(fermi)
    P0_cpu = purification_initial_guess(H; ϵF=epsF, maxdim=maxdim, cutoff=Float64(cutoff))
    P      = _mcweeny_iterate(_to_gpu(P0_cpu, ComplexF32); maxiters, maxdim,
                              cutoff=Float64(cutoff), tol, trunc=(:cutoff,),
                              after_step=_gpu_gc!)
    return return_gpu ? P : _to_cpu_mpo(P)
end


# ============================================================
# 2. GPU-resident McWeeny on an uploaded Hamiltonian MPO
# ============================================================

# Linear initial guess ρ₀ = (1/2 + (ϵF + center)/(2 scale))·I − H/(2 scale), formed
# on GPU (the GPU analogue of purification_initial_guess); the identity is uploaded
# as ComplexF32 unless `Id_gpu` is given.
function _purification_initial_guess_gpu(H_mpo_gpu::MPO, sites;
                                         ϵF::Real,
                                         scale::Real,
                                         center::Real = 0.0,
                                         maxdim::Int,
                                         cutoff::Real,
                                         Id_gpu::Union{Nothing,MPO} = nothing)
    scale == 0 && error("_purification_initial_guess_gpu: scale must be non-zero.")
    Id = Id_gpu === nothing ? _to_gpu(MPO(collect(sites), "Id"), ComplexF32) : Id_gpu
    return _linear_density_guess(H_mpo_gpu, Id; ϵF=ϵF, center=center, scale=scale,
                                 maxdim=maxdim, cutoff=Float64(cutoff))
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
    P = _purification_initial_guess_gpu(H_mpo_gpu, sites;
        ϵF=ϵF, scale=scale, center=center, maxdim=maxdim,
        cutoff=cutoff, Id_gpu=Id_gpu)
    return _mcweeny_iterate(P; maxiters, maxdim, cutoff=Float64(cutoff), tol,
                            trunc=(:cutoff,), after_step=_gpu_gc!)
end
