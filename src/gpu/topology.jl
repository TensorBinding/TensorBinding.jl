# gpu/topology.jl — the GPU real-space Chern marker get_C_gpu, the GPU mirror of
# get_C (physics/Topology.jl), with the McWeeny/SP2 purification loops run on GPU.
# One entry point, so the file has no sections. Moved from the former
# gpu/GPU_tk.jl.
#
# Main entry point: get_C_gpu.
# Depends on: core/Utils.jl (basis MPS, diagonal MPOs, postpend_op),
# core/TBSystem.jl, solvers/DMRG.jl (_ensure_scale!), physics/Topology.jl
# (position operators, _get_projector), physics/Purification.jl
# (purification_initial_guess), gpu/device.jl.

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
