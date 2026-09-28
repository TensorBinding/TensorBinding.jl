# gpu/exciton.jl — GPU exciton spectroscopy on the interleaved electron-hole
# register: the spatial exciton LDOS get_exciton_ldos_spatial_gpu and the
# truncation diagnostic get_exciton_cheb_convergence_gpu. Moved from the former
# gpu/GPU_tk.jl.
#
# Main entry points: get_exciton_ldos_spatial_gpu, get_exciton_cheb_convergence_gpu.
# Depends on: core/Utils.jl (mpsexciton, spatial_sampling_plan), core/TBSystem.jl,
# solvers/DMRG.jl (_ensure_scale!), solvers/kpm/kernels.jl (_kpm_energy_grid),
# solvers/kpm/recursion.jl (_scaled_hamiltonian, _run_kpm_mps!, _chebyshev_step),
# gpu/device.jl.


# ============================================================
# 1. Exciton LDOS
# ============================================================

"""
    get_exciton_ldos_spatial_gpu(H, Ncheb, ω_phys_vals;
                                 Lx=nothing, num_y=nothing, reduce=:point,
                                 X_list=nothing, X_groups=nothing, x_groups=nothing,
                                 num_x=0, num_avg=1, x_start=1, x_end=H.N,
                                 kernel=:jackson, lambda=4.0, eta=0.0, m_order=4,
                                 maxdim=100, cutoff=1e-8,
                                 type=ComplexF32, dtype=nothing,
                                 verbose=false, printinfo=false,
                                 return_maxlinkdim=false)
        -> Matrix{Float64}   (Nω × n_cols)

GPU-accelerated spatial exciton LDOS A(X,ω) = ⟨X,X|δ(ω−H)|X,X⟩ via MPS Chebyshev KPM.
One GPU Chebyshev recursion runs per probe position X (electron = hole = X, 1-indexed
in 1:H.N) from |X,X⟩ = mpsexciton(X, H.sites); moments are scalars pulled to CPU.

**1D sampling** (default, `Lx=nothing`): `num_x` coarse positions over `[x_start, x_end]`
with `num_avg` sub-positions per coarse cell averaged per output pixel. `num_x ≤ 0`
(the default) means every position (`H.N` columns) in 1D, and `num_x = 8` on the
2D grid.

**2D grid** (`Lx` provided): positions are 1-indexed on the (Lx+Ly)-qubit quantics grid,
encoded as X = ix + iy·2^Lx + 1 (row-major, 0-indexed). `num_x × num_y` coarse cells
are sampled via `spatial_sampling_plan` with `num_avg` sub-positions per cell.
Output columns are row-major over coarse cells (iy outer, ix inner).

`X_list` / `X_groups` / `x_groups` bypass the automatic plan and pass positions
directly (`X_groups` and `x_groups` are aliases; pass at most one of the three).

`kernel=:hodc` selects HODC reconstruction (`eta`, `m_order`; `eta=0` → `1/(Ncheb+1)`).
Other kernels: `:jackson` (default), `:lorentz` (`lambda`), `:fejer`, `:dirichlet`.

Use `type=ComplexF32` (default, faster) or `type=ComplexF64` (safer at tight cutoffs
or on large systems where F32 eigendecomposition can produce NaN; a warning is
emitted for a 32-bit type with `cutoff < 1e-6`), or `Float32`/`Float64` for a real `H`.
`dtype` is accepted as an alias for `type` for consistency with other GPU entry points.

`return_maxlinkdim=true` returns `(result, linkdims)` instead of just `result`, where
`linkdims[j]` is the `maxlinkdim` of the last Chebyshev vector of the last probe
in column `j` (the χ the recursion hit under the given `maxdim`/`cutoff`). Useful
for cutoff/tolerance studies where χ is the observable.

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
    gpu_type = _resolve_gpu_type("get_exciton_ldos_spatial_gpu", type, dtype, cutoff)
    reduce === :block &&
        error("get_exciton_ldos_spatial_gpu: reduce=:block is not supported for the exciton LDOS. " *
              "Block averaging requires O(block_size) independent Chebyshev recursions per pixel, " *
              "making it equivalent to computing every site. Use reduce=:point with num_avg for " *
              "light spatial averaging. Block averaging is only available in MPO-based LDOS functions.")
    reduce === :point ||
        error("get_exciton_ldos_spatial_gpu: reduce must be :point, got $reduce.")
    _ensure_scale!(H)
    _is_exciton_register(H) ||
        error("get_exciton_ldos_spatial_gpu: H is not an exciton Hamiltonian (expected the 2L-site electron-hole register of exciton_hamiltonian).")

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
                    num_avg = num_avg,
                    x_start = x_start,
                    x_end   = x_end)
        plan.groups
    end
    isempty(groups) && error("get_exciton_ldos_spatial_gpu: no spatial groups were selected.")
    for grp in groups
        all(x -> 1 <= x <= H.N, grp) ||
            error("get_exciton_ldos_spatial_gpu: all positions must lie in 1:H.N.")
    end
    Xs = first.(groups)

    Ham_n_cpu = _scaled_hamiltonian(H; cutoff=cutoff)
    Ham_n_gpu = _to_gpu(Ham_n_cpu, gpu_type)

    ω_vals, W, denom, valid = _kpm_energy_grid(H, Ncheb, ω_phys_vals;
                                               kernel=kernel, lambda=lambda, eta=eta,
                                               m_order=m_order, allow_hodc=true)
    Nω     = length(ω_vals)

    nX           = length(groups)
    result       = zeros(Float64, Nω, nX)

    printinfo && _gpu_log("exciton ldos dtype=$gpu_type")

    linkdims = zeros(Int, nX)   # reached MPS bond dim per output column (see return_maxlinkdim)

    for (j, group) in enumerate(groups)
        last_linkdim = 0

        for X in group
            psi0_gpu = _to_gpu(mpsexciton(X, H.sites), gpu_type)
            accum    = zeros(Float64, Nω)

            # The CPU online recursion on GPU tensors; weight 1.0 leaves each
            # W[n, iω] * μ_n unchanged.
            last_linkdim = _run_kpm_mps!(Ham_n_gpu, psi0_gpu, Ncheb, W, valid, accum;
                                         weight=1.0, cutoff=Float64(cutoff), maxdim=maxdim)
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
            _gpu_log("exciton ldos $j/$nX (X=$(Xs[j]), n_avg=$(length(group)))  maxlinkdim=$last_linkdim")
    end

    return return_maxlinkdim ? (result, linkdims) : result
end


# ============================================================
# 2. Chebyshev convergence diagnostics
# ============================================================

"""
    get_exciton_cheb_convergence_gpu(H, X, Ncheb_max;
                                      maxdim_test=100, maxdim_ref=500, cutoff=1e-4,
                                      type=ComplexF32, dtype=nothing,
                                      printinfo=false)
        -> NamedTuple
    get_exciton_cheb_convergence_gpu(H, X_probes::AbstractVector{<:Integer},
                                      Ncheb_max; kwargs...)
        -> Vector{NamedTuple}

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
get_exciton_ldos_spatial_gpu. All MPS live on GPU throughout, with element type
`type` (alias `dtype`): `ComplexF32` (default), `ComplexF64`, or
`Float32`/`Float64` for a real `H`. The vector method runs the single-probe
method once per entry of `X_probes`, with the same keywords.

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
    _is_exciton_register(H) ||
        error("get_exciton_cheb_convergence_gpu: H is not an exciton Hamiltonian (expected the 2L-site electron-hole register of exciton_hamiltonian).")
    1 <= X <= H.N ||
        error("get_exciton_cheb_convergence_gpu: X=$X out of range 1:$(H.N).")
    maxdim_ref >= maxdim_test ||
        @warn "get_exciton_cheb_convergence_gpu: maxdim_ref=$maxdim_ref < maxdim_test=$maxdim_test; reference is no more accurate than test."

    _ensure_scale!(H)

    Ham_n_cpu  = _scaled_hamiltonian(H; cutoff=Float64(cutoff))
    Ham_n_gpu  = _to_gpu(Ham_n_cpu, gpu_type)

    psi0_gpu   = _to_gpu(mpsexciton(X, H.sites), gpu_type)

    ak_ref  = (cutoff=Float64(cutoff), maxdim=maxdim_ref)
    ak_test = (cutoff=Float64(cutoff), maxdim=maxdim_test)

    # Chebyshev recursion: T_0 = psi0, T_1 = H̃·psi0,  T_n = 2H̃·T_{n-1} − T_{n-2}.
    # Two recursions in lockstep (compared at every order), so the loop is written
    # out here and each step is chebyshev_foreach's (_chebyshev_step).
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
            phi_ref_new  = _chebyshev_step(Ham_n_gpu, phi_ref_km1,  phi_ref_km2;
                                           apply_kwargs=ak_ref,  add_kwargs=ak_ref)
            phi_test_new = _chebyshev_step(Ham_n_gpu, phi_test_km1, phi_test_km2;
                                           apply_kwargs=ak_test, add_kwargs=ak_test)
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
# Each probe is a full call of the single-X method, so the Hamiltonian rescaling
# and GPU MPO conversion are repeated once per probe.
function get_exciton_cheb_convergence_gpu(H::TBHamiltonian,
                                           X_probes::AbstractVector{<:Integer},
                                           Ncheb_max::Int; kwargs...)
    return [get_exciton_cheb_convergence_gpu(H, Int(X), Ncheb_max; kwargs...)
            for X in X_probes]
end
