# gpu/topology.jl — the GPU real-space Chern marker get_C_gpu, the GPU mirror of
# get_C (physics/Topology.jl): a thin wrapper that runs the CPU kernels on GPU MPOs,
# the McWeeny/SP2 loops (_mcweeny_iterate, _sp2_iterate) and the marker assembly
# (_chern_marker), with the GPU truncations as keywords. One entry point, so the
# file has no sections. Moved from the former gpu/GPU_tk.jl.
#
# Main entry point: get_C_gpu.
# Depends on: core/TBSystem.jl, solvers/DMRG.jl (_ensure_scale!), physics/Topology.jl
# (_chern_marker, _get_projector), physics/Purification.jl
# (purification_initial_guess, _mcweeny_iterate, _sp2_iterate), gpu/device.jl.

"""
    get_C_gpu(H::TBHamiltonian, xfunc=nothing, yfunc=nothing;
              method=:mcweeny, fermi=0.0, l=nothing, Λ=10, Lambda=nothing,
              Nchebychev=300, maxdim=500, cutoff=1e-8,
              Nel=nothing, quenched=true, dtype=ComplexF32,
              printinfo=false) -> Function

GPU-accelerated real-space Chern marker.  Mirrors `get_C` but runs all
MPO×MPO products (projector assembly and C1–C4 construction) on GPU.

Returns the same closure `C_at(uc::Int) -> ComplexF64` as `get_C`.

# Key differences from `get_C`
- All `apply`/`truncate!` operations run on GPU tensors.
- `dtype` (default `ComplexF32`) selects the GPU element type; the marker is
  intrinsically complex, so only `ComplexF32` / `ComplexF64` are accepted. Use
  `dtype=ComplexF64` to avoid NaN from ComplexF32 eigendecompositions on large
  systems at tight cutoffs (a warning is emitted for `ComplexF32` + `cutoff < 1e-6`).
- For `method=:mcweeny` and `method=:sp2`, only the initial guess
  (`purification_initial_guess`) is built on CPU; it is moved to GPU and the
  purification loop runs there. For `method=:KPM` the whole projector is built
  on CPU (via `_get_projector`), then moved to GPU.
- `get_C`'s `sequential` keyword is not accepted; the quenched marker is always
  assembled from the C1–C4 MPOs.
- The default `method` is `:mcweeny` (`get_C` defaults to `:KPM`), and
  `printinfo` prints progress.

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

    _require_binary_position_space(H, "get_C_gpu")
    _check_gpu("get_C_gpu")
    gpu_type = _resolve_gpu_type("get_C_gpu", dtype, nothing, cutoff)
    Λ_val = Lambda !== nothing ? Float64(Lambda) : Float64(Λ)

    # ── geometry ──────────────────────────────────────────────────────────────
    if xfunc === nothing || yfunc === nothing
        geom = H.geometry_uc !== nothing ? H.geometry_uc :
               H.geometry   !== nothing ? H.geometry   :
               error("get_C_gpu: H has no geometry; provide xfunc and yfunc explicitly.")
        xfunc === nothing && (xfunc = (i, _) -> geom(i + 1)[1])
        yfunc === nothing && (yfunc = (i, _) -> geom(i + 1)[2])
    end

    # ── projector: build initial guess on CPU, purify on GPU ──────────────────
    printinfo && _gpu_log("Building initial projector guess (CPU)..."; indent=0)
    _ensure_scale!(H)
    P0_cpu = purification_initial_guess(H; ϵF=fermi, maxdim=maxdim, cutoff=cutoff)
    P = _to_gpu(P0_cpu, gpu_type)

    # The CPU McWeeny/SP2 loops (physics/Purification.jl) on GPU MPOs, with the GPU
    # truncations: each square and update truncated with `cutoff` only, the SP2
    # expansion 2P − P² summed with `cutoff` and `maxdim`; GPU memory is freed after
    # every iteration.
    if method == :mcweeny
        printinfo && _gpu_log("McWeeny purification on GPU..."; indent=0)
        P = _mcweeny_iterate(P; maxiters=30, maxdim, cutoff=Float64(cutoff), tol=1e-5,
                             trunc=(:cutoff,), after_step=_gpu_gc!,
                             progress = printinfo ? function (iter, err, ρ)
                                 iter % 5 == 0 &&
                                     println("  McWeeny iter $iter: err=$err  maxlinkdim=$(maxlinkdim(ρ))")
                             end : nothing)
        H._density_cache = nothing   # don't cache GPU MPO in CPU field
    elseif method == :sp2
        Nel_val = Nel === nothing ? H.N ÷ 2 : Int(Nel)
        printinfo && _gpu_log("SP2 purification on GPU (Nel=$Nel_val)..."; indent=0)
        P = _sp2_iterate(P, Nel_val; maxiters=40, maxdim, cutoff=Float64(cutoff), tol=1e-5,
                         trunc=(:cutoff,), add_trunc=(:cutoff, :maxdim), after_step=_gpu_gc!,
                         progress = printinfo ? function (iter, err, ρ)
                             println("  SP2 iter $iter: err=$err  maxlinkdim=$(maxlinkdim(ρ))")
                         end : nothing)
    elseif method == :KPM
        # KPM: use CPU projector, just move to GPU
        P_cpu = _get_projector(H; method=:KPM, fermi=fermi, Nchebychev=Nchebychev,
                               maxdim=maxdim, cutoff=cutoff)
        P = _to_gpu(P_cpu, gpu_type)
    else
        error("get_C_gpu: unknown method :$method. Choose :mcweeny, :sp2, or :KPM")
    end
    printinfo && _gpu_log("Projector ready, maxlinkdim=$(maxlinkdim(P))"; indent=0)

    # ── the CPU marker assembly on GPU MPOs ──────────────────────────────────
    # Q = I − P summed with `maxdim` and `cutoff`, then truncated with `cutoff`, as
    # are C1–C4 and the flat operator; operators and basis states uploaded with
    # gpu_type; GPU memory freed after each step.
    progress = printinfo ? function (stage, M)
        stage === :positions && _gpu_log("Position operators on GPU."; indent=0)
        stage === :products  && _gpu_log("8 intermediate MPO products done."; indent=0)
        stage in (:C1, :C2, :C3) && _gpu_log("$stage done, maxlinkdim=$(maxlinkdim(M))"; indent=0)
        stage === :C4        && _gpu_log("C4 done. Closure ready."; indent=0)
        return nothing
    end : nothing
    return _chern_marker(P, H.L, H.sites, xfunc, yfunc;
                         l, Λ=Λ_val, maxdim, cutoff=Float64(cutoff), quenched,
                         to_device=_to_gpu, device_type=gpu_type,
                         q_add=(:cutoff, :maxdim), q_trunc=(:cutoff,),
                         c_trunc=(:cutoff,), flat_trunc=(:cutoff,),
                         progress, after_step=_gpu_gc!)
end
