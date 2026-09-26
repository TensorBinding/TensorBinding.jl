# gpu/nh.jl — GPU non-Hermitian KPM density of states on the hermitized block
# Hamiltonian: the NH block contraction/trace helpers, the online deterministic
# diagonal-trace recurrences, the stochastic dual-chain recurrence, and the entry
# points get_nh_dos_{grid,points}_gpu (stochastic) and
# get_nh_dos_{grid,points}_diag_trace_gpu (deterministic). Moved from the former
# gpu/GPU_tk.jl.
#
# Main entry points: get_nh_dos_grid_gpu, get_nh_dos_points_gpu,
# get_nh_dos_points_diag_trace_gpu, get_nh_dos_grid_diag_trace_gpu.
# Depends on: core/TBSystem.jl, physics/nh/model.jl (NonHermitianHamiltonian,
# hermitize), physics/nh/kpm.jl (nh_kpm_scale, nh_block_source,
# nh_jackson_weights), gpu/device.jl, gpu/primitives.jl.


# ============================================================
# 1. NH block contraction and trace
# ============================================================

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


# ============================================================
# 2. Deterministic diagonal-trace recurrences
# ============================================================

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

# Same recurrence, but accumulates the diagonal MPS and returns it with the DOS.
# No caller in the package (the entry points use the scalar variant above).
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


# ============================================================
# 3. Stochastic dual-chain recurrence
# ============================================================

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


# ============================================================
# 4. Stochastic DOS entry points
# ============================================================

"""
    get_nh_dos_grid_gpu(H, xlims, nx, ylims, ny, n;
                        scale=nothing, nh_scale_padding=1.05,
                        convention=:z_minus_H, block_placement=:post,
                        n_random=10, seed=42, maxdim=100, cutoff=1e-8,
                        dmrg_nsweeps=5, dmrg_maxdim=[10, 20, 40], dmrg_linkdim=4,
                        dtype=ComplexF64, verbose=false, printinfo=false)
        -> (xgrid, ygrid, Z)

GPU stochastic non-Hermitian KPM spectral-weight grid. This mirrors
`nh_spectrum_grid(...; mode=:stochastic)`: for each complex point
`z = x + im*y`, the non-Hermitian Hamiltonian is hermitized on CPU, then the
dual-chain stochastic MPS recurrence runs on GPU. Only scalar moments are
copied back to CPU. `Z[iy, ix]` is the value at `xgrid[ix] + im*ygrid[iy]`.

One universal scale from `nh_kpm_scale` (`scale`, `nh_scale_padding`, `dmrg_*`)
is used for every point; one random-number generator (`seed`, or the global RNG
for `seed=nothing`) is shared by the whole grid. `dtype` is the GPU element type,
complex only: `ComplexF64` (default) or `ComplexF32` (warned below
`cutoff = 1e-4`).

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

"""
    get_nh_dos_points_gpu(H, z_points, n;
                          scale=nothing, nh_scale_padding=1.05,
                          convention=:z_minus_H, block_placement=:post,
                          n_random=10, seed=42, seed_stride=1_000_003,
                          point_ids=nothing, maxdim=100, cutoff=1e-8,
                          dmrg_nsweeps=5, dmrg_maxdim=[10, 20, 40], dmrg_linkdim=4,
                          dtype=ComplexF64, verbose=false, printinfo=false)
        -> Vector{Float64}

GPU stochastic NH KPM at an explicit list of complex energies. This is the
array-job companion to `get_nh_dos_grid_gpu`: each `z_points[j]` is independent,
so production scripts can split a large grid over many GPUs and concatenate the
long-form CSV outputs afterward. The scale and `dtype` keywords are those of
[`get_nh_dos_grid_gpu`](@ref).

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


# ============================================================
# 5. Deterministic diagonal-trace DOS entry points
# ============================================================

"""
    get_nh_dos_points_diag_trace_gpu(H, z_points, n;
                                     scale=nothing, nh_scale_padding=1.05,
                                     convention=:z_minus_H, block_placement=:post,
                                     point_ids=nothing, maxdim=100, cutoff=1e-8,
                                     dmrg_nsweeps=5, dmrg_maxdim=[10, 20, 40],
                                     dmrg_linkdim=4, dtype=ComplexF64,
                                     source_row=2, source_col=1,
                                     block_row=2, block_col=1,
                                     verbose=false, printinfo=false)
        -> Vector{Float64}

Deterministic GPU NH KPM at an explicit list of complex energies. For each
`z`, the hermitized NH problem is built on CPU, then the online MPO-MPO
recurrence runs on GPU and evaluates the total trace through diagonal
extraction plus a GPU-resident all-sites sum. This avoids stochastic probes.

The integer `n` follows the NH convention used elsewhere in this file: the
partial recurrence runs to order `2n`. The scale keywords are those of
[`get_nh_dos_grid_gpu`](@ref); `dtype` is complex only (`ComplexF64` default,
or `ComplexF32`). `source_row`/`source_col` select the block of the NH source
term and `block_row`/`block_col` the block whose diagonal is traced;
`point_ids` only labels the progress output.
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
    get_nh_dos_grid_diag_trace_gpu(H, xlims, nx, ylims, ny, n;
                                   scale=nothing, nh_scale_padding=1.05,
                                   convention=:z_minus_H, block_placement=:post,
                                   maxdim=100, cutoff=1e-8, dmrg_nsweeps=5,
                                   dmrg_maxdim=[10, 20, 40], dmrg_linkdim=4,
                                   dtype=ComplexF64, verbose=false, printinfo=false)
        -> (xgrid, ygrid, Z)

Grid companion to [`get_nh_dos_points_diag_trace_gpu`](@ref): evaluates it on the
`nx × ny` grid over `xlims × ylims` and returns `Z[iy, ix]` at
`xgrid[ix] + im*ygrid[iy]`. The source and block rows/columns keep their
defaults.
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
