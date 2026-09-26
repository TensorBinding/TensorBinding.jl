# nh/kpm.jl — non-Hermitian KPM on the hermitized block Hamiltonian: the
# universal Chebyshev scale (nh_kpm_scale), the partial Chebyshev recursion
# (nh_kpm_partials), Jackson reconstruction of the spectral function
# (nh_reconstruct_spectral_mps, nh_spectral_function), the online MPO / MPS /
# stochastic _nh_* helpers and nh_spectrum_grid. Split from the former physics/NH_tk.jl;
# the NonHermitianHamiltonian model and hermitize are in nh/model.jl.
#
# Entry points: nh_spectrum_grid, nh_spectral_function, nh_kpm_partials,
#   nh_reconstruct_spectral_mps, nh_kpm_scale, nh_block_source, contract_nh_block.
# Depends on: core/Utils.jl, core/TBSystem.jl, core/AuxDOF.jl (_project_end_site:
#   the block contraction), solvers/DMRG.jl, solvers/kpm/kernels.jl
#   (_kpm_kernel: the Jackson weights), physics/nh/model.jl.

# ============================================================
# 1. Non-Hermitian KPM scale
# ============================================================

"""
    nh_kpm_scale(H, z_points; scale=nothing, padding=1.05, maxdim=200,
                 cutoff=1e-8, convention=:z_minus_H, block_placement=:post,
                 dmrg_nsweeps=5, dmrg_maxdim=[10,20,40], dmrg_linkdim=4,
                 printinfo=false)

Return one universal, zero-centered Chebyshev scale for NH KPM over all
complex points in `z_points`.

For `scale=nothing` or `scale=0`, the parent operator norm is estimated from
the Hermitian zero-shift block Hamiltonian

```text
[ 0  -H ; -H'  0 ]
```

whose spectral radius is `||H||_2`. The scale then uses the triangle bound

```text
||zI - H||_2 <= |z| + ||H||_2
```

and returns `padding * (maximum(abs(z_points)) + ||H||_2)`. This is deliberately
conservative and uses the same scale for every point, keeping values comparable
across a grid. Passing a positive numeric `scale` bypasses the estimator.
"""
function nh_kpm_scale(H::TBHamiltonian, z_points;
                      scale::Union{Nothing,Real} = nothing,
                      padding::Real = 1.05,
                      maxdim::Int = 200,
                      cutoff::Real = 1e-8,
                      convention::Symbol = :z_minus_H,
                      block_placement::Symbol = :post,
                      dmrg_nsweeps::Int = 5,
                      dmrg_maxdim = [10, 20, 40],
                      dmrg_linkdim::Int = 4,
                      printinfo::Bool = false)
    if scale !== nothing
        sc = Float64(scale)
        sc < 0.0 && error("nh_kpm_scale: scale must be nonnegative, got $sc.")
        sc > 0.0 && return sc
    end
    padding >= 1.0 || error("nh_kpm_scale: padding must be >= 1, got $padding.")

    z_radius = 0.0
    for z in z_points
        z_radius = max(z_radius, abs(ComplexF64(z)))
    end

    NH0 = hermitize(H;
        z=0.0,
        scale=0.0,
        maxdim=maxdim,
        cutoff=cutoff,
        convention=convention,
        block_placement=block_placement)
    _ensure_scale!(NH0.hermitized;
        dmrg_nsweeps=dmrg_nsweeps,
        dmrg_maxdim=dmrg_maxdim,
        dmrg_linkdim=dmrg_linkdim)

    parent_norm_bound = abs(NH0.hermitized.center) + NH0.hermitized.scale
    parent_norm_bound > 0.0 ||
        error("nh_kpm_scale: estimated parent norm is zero; cannot build NH KPM scale.")

    nh_scale = Float64(padding) * (z_radius + parent_norm_bound)
    printinfo && println("nh_kpm_scale: parent_norm_bound=$parent_norm_bound, z_radius=$z_radius, padding=$padding, scale=$nh_scale")
    return nh_scale
end

function _nh_resolve_scale(NH::NonHermitianHamiltonian;
                           scale::Union{Nothing,Real} = nothing,
                           nh_scale_padding::Real = 1.05,
                           maxdim::Int = 200,
                           cutoff::Real = 1e-8,
                           convention::Symbol = :z_minus_H,
                           dmrg_nsweeps::Int = 5,
                           dmrg_maxdim = [10, 20, 40],
                           dmrg_linkdim::Int = 4,
                           printinfo::Bool = false)
    if scale === nothing
        NH.hermitized.scale > 0.0 && return NH.hermitized.scale
    else
        sc = Float64(scale)
        sc < 0.0 && error("_nh_resolve_scale: scale must be nonnegative, got $sc.")
        sc > 0.0 && return sc
    end

    return nh_kpm_scale(NH.parent, (NH.z,);
        scale=nothing,
        padding=nh_scale_padding,
        maxdim=maxdim,
        cutoff=cutoff,
        convention=convention,
        block_placement=NH.block_placement,
        dmrg_nsweeps=dmrg_nsweeps,
        dmrg_maxdim=dmrg_maxdim,
        dmrg_linkdim=dmrg_linkdim,
        printinfo=printinfo)
end

# ============================================================
# 2. Partial recursion and Jackson reconstruction (MPO)
# ============================================================

"""
    nh_block_source(NH; row=2, col=1) -> MPO
    nh_block_source(Hh, block_s; row=2, col=1) -> MPO

Build the off-diagonal block source `|row><col| x I`. The default `row=2,
col=1` matches the old `I_ldn` source used in the non-Hermitian KPM recursion.
"""
function nh_block_source(Hh::TBHamiltonian, block_s::Index; row::Int = 2, col::Int = 1)
    pos_sites = filter(!=(block_s), Hh.sites)
    I_pos = MPO(pos_sites, "Id")
    return last(Hh.sites) == block_s ?
        postpend_op(I_pos, block_s, row, col) :
        prepend_op(I_pos, block_s, row, col)
end

nh_block_source(NH::NonHermitianHamiltonian; row::Int = 2, col::Int = 1) =
    nh_block_source(NH.hermitized, NH.block_s; row=row, col=col)

"""
    nh_kpm_partials(Hh, n; source, scale=nothing, maxdim=100, cutoff=1e-8)
        -> Vector{MPO}
    nh_kpm_partials(NH, n; source=nothing, source_row=2, source_col=1, scale=nothing,
                    nh_scale_padding=1.05, maxdim=100, cutoff=1e-8, dmrg_nsweeps=5,
                    dmrg_maxdim=[10, 20, 40], dmrg_linkdim=4) -> Vector{MPO}

Compute the auxiliary "partial" Chebyshev recursion used by the old
non-Hermitian spectral algorithm. If `A = Hh / scale` and `S` is the block
source, the recurrence is

```text
P_0 = 0
P_1 = S
P_k = 2 S T_{k-1}(A) + 2 A P_{k-1} - P_{k-2}
```

while `T_k(A)` is advanced in parallel. The returned vector has length `2n`
and stores `P_0, P_1, ..., P_{2n-1}`.

The `NonHermitianHamiltonian` method builds `S` with `nh_block_source(NH;
row=source_row, col=source_col)` unless `source` is given, and resolves the scale
from `NH.hermitized.scale` or `nh_kpm_scale`.
"""
function nh_kpm_partials(Hh::TBHamiltonian, n::Int;
                         source::MPO,
                         scale::Union{Nothing,Real} = nothing,
                         maxdim::Int = 100,
                         cutoff::Real = 1e-8)
    N = 2 * n
    sc = isnothing(scale) ? Hh.scale : Float64(scale)
    sc == 0.0 && error("nh_kpm_partials requires a nonzero scale. Pass scale=... or set Hh.scale.")

    A = Hh.mpo / sc
    Tkm2 = MPO(Hh.sites, "Id")
    Tkm1 = A
    Pkm2 = 0.0 * source
    Pkm1 = source
    partials = MPO[Pkm2, Pkm1]

    for k in 3:N
        Pk = +(apply(2.0 * source, Tkm1; maxdim=maxdim, cutoff=cutoff),
               2.0 * apply(A, Pkm1; maxdim=maxdim, cutoff=cutoff);
               maxdim=maxdim, cutoff=cutoff)
        Pk = +(Pk, -Pkm2; maxdim=maxdim, cutoff=cutoff)

        Tk = +(2.0 * apply(A, Tkm1; maxdim=maxdim, cutoff=cutoff),
               -Tkm2; maxdim=maxdim, cutoff=cutoff)

        push!(partials, Pk)
        Pkm2, Pkm1 = Pkm1, Pk
        Tkm2, Tkm1 = Tkm1, Tk
    end

    return partials
end

function nh_kpm_partials(NH::NonHermitianHamiltonian, n::Int;
                         source::Union{Nothing,MPO} = nothing,
                         source_row::Int = 2,
                         source_col::Int = 1,
                         scale::Union{Nothing,Real} = nothing,
                         nh_scale_padding::Real = 1.05,
                         maxdim::Int = 100,
                         cutoff::Real = 1e-8,
                         dmrg_nsweeps::Int = 5,
                         dmrg_maxdim = [10, 20, 40],
                         dmrg_linkdim::Int = 4)
    S = isnothing(source) ? nh_block_source(NH; row=source_row, col=source_col) : source
    sc = _nh_resolve_scale(NH;
        scale=scale,
        nh_scale_padding=nh_scale_padding,
        maxdim=maxdim,
        cutoff=cutoff,
        dmrg_nsweeps=dmrg_nsweeps,
        dmrg_maxdim=dmrg_maxdim,
        dmrg_linkdim=dmrg_linkdim)
    return nh_kpm_partials(NH.hermitized, n; source=S, scale=sc,
                           maxdim=maxdim, cutoff=cutoff)
end

"""
    contract_nh_block(W, block_s; row=2, col=1) -> MPO

Extract the `(row, col)` block of an MPO whose first or last site is `block_s`,
returning an MPO on the remaining sites. The block position is auto-detected
(the last site is tried first); the contraction is `project_aux`'s, off the
diagonal (`_project_end_site`, core/AuxDOF.jl).
"""
function contract_nh_block(W::MPO, block_s::Index; row::Int = 2, col::Int = 1)
    M = length(W)
    M >= 2 || error("contract_nh_block requires an MPO with a block site and at least one physical site.")
    side = siteind(W, M) == block_s ? :post :
           siteind(W, 1) == block_s ? :pre  :
           error("NH block index must be the first or last MPO site for contract_nh_block.")
    return _project_end_site(W, block_s, row, col, side)
end

"""
    nh_preprocess_partials(partials, block_s; row=2, col=1) -> Vector{MPS}

For each partial MPO, extract the requested NH block and then extract its
diagonal as an MPS. This is the old `pre_process` step without global state.
"""
function nh_preprocess_partials(partials::AbstractVector{<:MPO}, block_s::Index;
                                row::Int = 2,
                                col::Int = 1)
    return [extract_diagonal_to_mps(contract_nh_block(P, block_s; row=row, col=col))
            for P in partials]
end

function nh_ones_mps(sites::Vector{<:Index})
    N = length(sites)
    N == 0 && return MPS(ITensor[])
    links = [Index(1, "Link,l=$i") for i in 1:N-1]
    tensors = Vector{ITensor}(undef, N)

    for i in 1:N
        inds_i = if N == 1
            (sites[i],)
        elseif i == 1
            (sites[i], links[i])
        elseif i == N
            (links[i-1], sites[i])
        else
            (links[i-1], sites[i], links[i])
        end
        T = ITensor(ComplexF64, inds_i...)
        for v in 1:dim(sites[i])
            if N == 1
                T[sites[i] => v] = 1.0
            elseif i == 1
                T[sites[i] => v, links[i] => 1] = 1.0
            elseif i == N
                T[links[i-1] => 1, sites[i] => v] = 1.0
            else
                T[links[i-1] => 1, sites[i] => v, links[i] => 1] = 1.0
            end
        end
        tensors[i] = T
    end
    return MPS(tensors)
end

# The Jackson weights g_l of the reconstructions below are the textbook kernel for
# N moments, unnormalised: _kpm_kernel(N + 1, :jackson)[1:N] (solvers/kpm/kernels.jl),
# = (N − k + 1)cos(πk/(N+1)) + sin(πk/(N+1))cot(π/(N+1)), k = 0 … N−1.

"""
    nh_reconstruct_spectral_mps(partials, n, block_s; maxdim=100,
                                row=2, col=1) -> (A_mps, dos)

Apply the old Jackson reconstruction to the even partial terms:

```text
A = 2/(pi^2 (2n+1)) * sum_l (-1)^(l/2-1) g_l diag(P_l)
```

where `l = 2, 4, ..., 2n` in one-based Julia indexing of the partial list.
Returns the diagonal spectral MPS and its summed value.
"""
function nh_reconstruct_spectral_mps(partials::AbstractVector{<:MPO}, n::Int,
                                     block_s::Index;
                                     maxdim::Int = 100,
                                     row::Int = 2,
                                     col::Int = 1)
    N = 2 * n
    length(partials) >= N || error("Expected at least $N partials, got $(length(partials)).")
    weights = _kpm_kernel(N + 1, :jackson)[1:N]   # Jackson weights, N moments
    diag_list = nh_preprocess_partials(partials, block_s; row=row, col=col)

    A = diag_list[1]
    for l in 2:2:N
        order = (-1)^((l ÷ 2) - 1)
        A = +(A, order * weights[l - 1] * diag_list[l]; maxdim=maxdim)
    end
    A *= 2.0 / (pi^2 * (N + 1))

    dos = inner(nh_ones_mps(siteinds(A))', A)
    return A, dos
end

"""
    nh_spectral_function(NH, n; scale=nothing, nh_scale_padding=1.05, maxdim=100,
                         cutoff=1e-8, dmrg_nsweeps=5, dmrg_maxdim=[10, 20, 40],
                         dmrg_linkdim=4, source_row=2, source_col=1,
                         block_row=2, block_col=1) -> (A_mps, dos, partials)

Convenience wrapper for the full non-Hermitian KPM spectral calculation at
the reference point stored in `NH.z`.
"""
function nh_spectral_function(NH::NonHermitianHamiltonian, n::Int;
                              scale::Union{Nothing,Real} = nothing,
                              nh_scale_padding::Real = 1.05,
                              maxdim::Int = 100,
                              cutoff::Real = 1e-8,
                              dmrg_nsweeps::Int = 5,
                              dmrg_maxdim = [10, 20, 40],
                              dmrg_linkdim::Int = 4,
                              source_row::Int = 2,
                              source_col::Int = 1,
                              block_row::Int = 2,
                              block_col::Int = 1)
    partials = nh_kpm_partials(NH, n; source_row=source_row, source_col=source_col,
                               scale=scale, nh_scale_padding=nh_scale_padding,
                               maxdim=maxdim, cutoff=cutoff,
                               dmrg_nsweeps=dmrg_nsweeps,
                               dmrg_maxdim=dmrg_maxdim,
                               dmrg_linkdim=dmrg_linkdim)
    A, dos = nh_reconstruct_spectral_mps(partials, n, NH.block_s;
                                         maxdim=maxdim, row=block_row, col=block_col)
    return A, dos, partials
end

# ============================================================
# 3. Online evaluators (MPO trace, diagonal, MPS probe, stochastic)
# ============================================================

"""
    _nh_kpm_probe_mps(sites, block_s, block_state, site_r) -> MPS

Product-state MPS on the hermitized block space. The block site carries
`block_state` (1-indexed); position sites are set to the big-endian binary
encoding of the 0-indexed physical site `site_r`. Bond dimension 1.

Works for both `:pre` (`sites = [block_s; pos_sites...]`) and `:post`
(`sites = [pos_sites...; block_s]`) layouts — placement is auto-detected.
"""
function _nh_kpm_probe_mps(sites::Vector{<:Index}, block_s::Index,
                             block_state::Int, site_r::Int)
    N = length(sites)
    L = N - 1  # number of position qubits
    postpend = (sites[end] == block_s)
    links   = [Index(1, "Link,l=$i") for i in 1:N-1]
    tensors = Vector{ITensor}(undef, N)
    for i in 1:N
        s      = sites[i]
        inds_i = Index[]
        i > 1 && push!(inds_i, links[i-1])
        push!(inds_i, s)
        i < N && push!(inds_i, links[i])
        T = ITensor(ComplexF64, inds_i...)
        v = if s == block_s
            block_state
        elseif postpend
            # Position at i=1:L; MSB at i=1 (bit L-1), LSB at i=L (bit 0)
            ((site_r >> (L - i)) & 1) + 1
        else
            # Position at i=2:N; MSB at i=2 (bit L-1), LSB at i=N (bit 0)
            ((site_r >> (L - i + 1)) & 1) + 1
        end
        pairs = Pair{Index,Int}[]
        i > 1 && push!(pairs, links[i-1] => 1)
        push!(pairs, s => v)
        i < N && push!(pairs, links[i]   => 1)
        T[pairs...] = 1.0
        tensors[i] = T
    end
    return MPS(tensors)
end


"""
    _nh_kpm_mps_ldos(NH, n, probe_site; scale, maxdim=100, cutoff=1e-8) -> Real

Online MPS NH KPM: compute the site-resolved spectral weight A(probe_site, z)
using the dual-chain MPS partial recursion, keeping only 4 MPS in memory at a time.

`probe_site` is a 0-indexed physical site. The probes are localized basis states:
  ket_probe = |block=1⟩ ⊗ |probe_site⟩
  bra_probe = |block=2⟩ ⊗ |probe_site⟩

so inner(bra_probe, p_k) = ⟨2, probe_site | P_k | 1, probe_site⟩, which is the
diagonal element of block_{2,1}(P_k) at site probe_site — the correct LDOS
contribution at that site.

Two chains are propagated on the hermitized block space:
  |t_k⟩ = T_k(A)|ket_probe⟩    (Chebyshev,  A = H_herm / scale)
  |p_k⟩ = P_k|ket_probe⟩        (NH partial sum)

with partial recurrence:
  |p_0⟩ = 0,  |p_1⟩ = S|ket_probe⟩
  |p_k⟩ = 2S|t_{k-1}⟩ + 2A|p_{k-1}⟩ − |p_{k-2}⟩

Cost per z-point: O(Ncheb × χ_H × χ_ψ) instead of O(Ncheb × χ_P²) for MPO mode.
"""
function _nh_kpm_mps_ldos(NH::NonHermitianHamiltonian, n::Int, probe_site::Int;
                           scale::Real,
                           maxdim::Int = 100,
                           cutoff::Real = 1e-8)
    N  = 2 * n
    Hh = NH.hermitized
    A  = Hh.mpo / scale
    S  = nh_block_source(NH)

    ket_probe = _nh_kpm_probe_mps(Hh.sites, NH.block_s, 1, probe_site)
    bra_probe = _nh_kpm_probe_mps(Hh.sites, NH.block_s, 2, probe_site)

    tkm2 = ket_probe
    tkm1 = apply(A, ket_probe; maxdim=maxdim, cutoff=cutoff)
    pkm2 = 0.0 * ket_probe
    pkm1 = apply(S, ket_probe; maxdim=maxdim, cutoff=cutoff)

    partial_vals = zeros(ComplexF64, N)
    partial_vals[2] = inner(bra_probe, pkm1)

    for k in 3:N
        tk     = +(2.0 * apply(A, tkm1; maxdim=maxdim, cutoff=cutoff),
                   -tkm2; maxdim=maxdim, cutoff=cutoff)
        s_tkm1 = 2.0 * apply(S, tkm1; maxdim=maxdim, cutoff=cutoff)
        a_pkm1 = 2.0 * apply(A, pkm1; maxdim=maxdim, cutoff=cutoff)
        pk     = +(+(s_tkm1, a_pkm1; maxdim=maxdim, cutoff=cutoff),
                   -pkm2;              maxdim=maxdim, cutoff=cutoff)
        partial_vals[k] = inner(bra_probe, pk)
        tkm2 = tkm1;  tkm1 = tk
        pkm2 = pkm1;  pkm1 = pk
    end

    weights = _kpm_kernel(N + 1, :jackson)[1:N]   # Jackson weights, N moments
    dos = ComplexF64(0)
    for l in 2:2:N
        dos += (-1)^(l ÷ 2 - 1) * weights[l - 1] * partial_vals[l]
    end
    return real(dos * 2.0 / (π^2 * (N + 1)))
end


"""
    _nh_scalar_online(NH, n; scale=nothing, nh_scale_padding=1.05, maxdim=100,
                      cutoff=1e-8, dmrg_nsweeps=5, dmrg_maxdim=[10, 20, 40],
                      dmrg_linkdim=4, source_row=2, source_col=1,
                      block_row=2, block_col=1) -> ComplexF64

Online NH KPM scalar DOS: run the partial Chebyshev recursion and accumulate
Tr[block_{2,1}(P_k)] contributions in a single pass, keeping only two partial
MPOs in memory at a time.

Avoids the O(N·χ_P²) memory cost of `nh_kpm_partials` and skips building the
intermediate A_mps entirely — each contributing step adds only one scalar to
the accumulator.
"""
function _nh_scalar_online(NH::NonHermitianHamiltonian, n::Int;
                            scale::Union{Nothing,Real} = nothing,
                            nh_scale_padding::Real = 1.05,
                            maxdim::Int  = 100,
                            cutoff::Real = 1e-8,
                            dmrg_nsweeps::Int = 5,
                            dmrg_maxdim = [10, 20, 40],
                            dmrg_linkdim::Int = 4,
                            source_row::Int = 2,
                            source_col::Int = 1,
                            block_row::Int  = 2,
                            block_col::Int  = 1)
    N  = 2 * n
    Hh = NH.hermitized
    sc = _nh_resolve_scale(NH;
        scale=scale,
        nh_scale_padding=nh_scale_padding,
        maxdim=maxdim,
        cutoff=cutoff,
        dmrg_nsweeps=dmrg_nsweeps,
        dmrg_maxdim=dmrg_maxdim,
        dmrg_linkdim=dmrg_linkdim)

    A_op    = Hh.mpo / sc
    source  = nh_block_source(NH; row=source_row, col=source_col)
    weights = _kpm_kernel(N + 1, :jackson)[1:N]   # Jackson weights, N moments
    ones_p  = nh_ones_mps(filter(!=(NH.block_s), Hh.sites))

    Tkm2 = MPO(Hh.sites, "Id")
    Tkm1 = A_op
    Pkm2 = 0.0 * source
    Pkm1 = source   # P_1

    _tr(P) = inner(ones_p',
                   extract_diagonal_to_mps(
                       contract_nh_block(P, NH.block_s; row=block_row, col=block_col)))

    dos = weights[1] * _tr(Pkm1)   # l=2 term: order=+1, weight=weights[1]

    for k in 3:N
        Tk = +(2.0 * apply(A_op,   Tkm1; maxdim=maxdim, cutoff=cutoff),
               -Tkm2; maxdim=maxdim, cutoff=cutoff)
        Pk = +(+(2.0 * apply(source, Tkm1; maxdim=maxdim, cutoff=cutoff),
                 2.0 * apply(A_op,   Pkm1; maxdim=maxdim, cutoff=cutoff);
                 maxdim=maxdim, cutoff=cutoff),
               -Pkm2; maxdim=maxdim, cutoff=cutoff)

        if iseven(k)
            dos += (-1)^(k ÷ 2 - 1) * weights[k - 1] * _tr(Pk)
        end

        Tkm2, Tkm1 = Tkm1, Tk
        Pkm2, Pkm1 = Pkm1, Pk
    end

    return dos * 2.0 / (π^2 * (N + 1))
end


"""
    _nh_diag_online(NH, n; scale=nothing, nh_scale_padding=1.05, maxdim=100,
                    cutoff=1e-8, dmrg_nsweeps=5, dmrg_maxdim=[10, 20, 40],
                    dmrg_linkdim=4, source_row=2, source_col=1,
                    block_row=2, block_col=1) -> (A_mps, dos)

Online NH KPM diagonal spectral function: run the partial Chebyshev recursion
and accumulate the site-resolved diagonal MPS A(r, z) in a single pass, keeping
only two partial MPOs in memory at a time.

Compared with `nh_kpm_partials` + `nh_reconstruct_spectral_mps`:
  - Memory: O(2 χ_P²) instead of O(N χ_P²).
  - Diagonal extractions: N/2 (only even Julia-index partials contribute).
"""
function _nh_diag_online(NH::NonHermitianHamiltonian, n::Int;
                          scale::Union{Nothing,Real} = nothing,
                          nh_scale_padding::Real = 1.05,
                          maxdim::Int  = 100,
                          cutoff::Real = 1e-8,
                          dmrg_nsweeps::Int = 5,
                          dmrg_maxdim = [10, 20, 40],
                          dmrg_linkdim::Int = 4,
                          source_row::Int = 2,
                          source_col::Int = 1,
                          block_row::Int  = 2,
                          block_col::Int  = 1)
    N  = 2 * n
    Hh = NH.hermitized
    sc = _nh_resolve_scale(NH;
        scale=scale,
        nh_scale_padding=nh_scale_padding,
        maxdim=maxdim,
        cutoff=cutoff,
        dmrg_nsweeps=dmrg_nsweeps,
        dmrg_maxdim=dmrg_maxdim,
        dmrg_linkdim=dmrg_linkdim)

    A_op    = Hh.mpo / sc
    source  = nh_block_source(NH; row=source_row, col=source_col)
    weights = _kpm_kernel(N + 1, :jackson)[1:N]   # Jackson weights, N moments

    Tkm2 = MPO(Hh.sites, "Id")
    Tkm1 = A_op
    Pkm2 = 0.0 * source
    Pkm1 = source   # P_1

    _diag(P) = extract_diagonal_to_mps(
        contract_nh_block(P, NH.block_s; row=block_row, col=block_col))
    

    A_mps = weights[1] * _diag(Pkm1)   # l=2 term: order=+1, weight=weights[1]

    for k in 3:N
        Tk = +(2.0 * apply(A_op,   Tkm1; maxdim=maxdim, cutoff=cutoff),
               -Tkm2; maxdim=maxdim, cutoff=cutoff)
        Pk = +(+(2.0 * apply(source, Tkm1; maxdim=maxdim, cutoff=cutoff),
                 2.0 * apply(A_op,   Pkm1; maxdim=maxdim, cutoff=cutoff);
                 maxdim=maxdim, cutoff=cutoff),
               -Pkm2; maxdim=maxdim, cutoff=cutoff)

        if iseven(k)
            A_mps = +(A_mps, ((-1)^(k ÷ 2 - 1) * weights[k - 1]) * _diag(Pk);
                      maxdim=maxdim)
        end

        Tkm2, Tkm1 = Tkm1, Tk
        Pkm2, Pkm1 = Pkm1, Pk
    end

    A_mps = A_mps * (2.0 / (π^2 * (N + 1)))
    dos   = inner(nh_ones_mps(siteinds(A_mps))', A_mps)
    return A_mps, dos
end


# Build a pair of product-state MPS (ket, bra) sharing the same random position
# state. Used by the stochastic trace estimator.
function _nh_random_probes(sites::Vector{<:Index}, block_s::Index,
                            ket_block::Int, bra_block::Int)
    N = length(sites)
    pos_rand = Dict(s => normalize(randn(ComplexF64, dim(s)))
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
            T = ITensor(ComplexF64, inds_i...)
            if s == block_s
                p = Pair{Index,Int}[]
                i > 1 && push!(p, links[i-1] => 1)
                push!(p, s => block_state)
                i < N && push!(p, links[i] => 1)
                T[p...] = 1.0
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

    return _make(ket_block), _make(bra_block)
end


"""
    _nh_stochastic_online(NH, n; scale=nothing, nh_scale_padding=1.05, n_random=10,
                          maxdim=100, cutoff=1e-8, dmrg_nsweeps=5,
                          dmrg_maxdim=[10, 20, 40], dmrg_linkdim=4, source_row=2,
                          source_col=1, block_row=2, block_col=1) -> Real

Stochastic trace NH KPM DOS: estimate Tr[block_{2,1}(P_k)] via Monte Carlo
averaging over `n_random` random product-state probes on the position sites.

Each realization draws |φ⟩ = ⊗_i (random local state) and runs the dual-chain
MPS recursion with ket = |1⟩_block ⊗ |φ⟩, bra = |2⟩_block ⊗ |φ⟩. The same
position state is shared between bra and ket so the estimator is unbiased:
  E[⟨2,φ|P_k|1,φ⟩] = Tr[P_k] / D,   D = 2^L

Cost: O(n_random × Ncheb × χ_H × χ_ψ) — no MPO×MPO products.
"""
function _nh_stochastic_online(NH::NonHermitianHamiltonian, n::Int;
                                scale::Union{Nothing,Real} = nothing,
                                nh_scale_padding::Real = 1.05,
                                n_random::Int  = 10,
                                maxdim::Int    = 100,
                                cutoff::Real   = 1e-8,
                                dmrg_nsweeps::Int = 5,
                                dmrg_maxdim = [10, 20, 40],
                                dmrg_linkdim::Int = 4,
                                source_row::Int = 2,
                                source_col::Int = 1,
                                block_row::Int  = 2,
                                block_col::Int  = 1)
    N  = 2 * n
    Hh = NH.hermitized
    sc = _nh_resolve_scale(NH;
        scale=scale,
        nh_scale_padding=nh_scale_padding,
        maxdim=maxdim,
        cutoff=cutoff,
        dmrg_nsweeps=dmrg_nsweeps,
        dmrg_maxdim=dmrg_maxdim,
        dmrg_linkdim=dmrg_linkdim)

    A_op    = Hh.mpo / sc
    S       = nh_block_source(NH; row=source_row, col=source_col)
    weights = _kpm_kernel(N + 1, :jackson)[1:N]   # Jackson weights, N moments
    D       = NH.parent.N   # 2^L = number of physical sites

    dos_acc = ComplexF64(0)

    for _ in 1:n_random
        ket_probe, bra_probe = _nh_random_probes(Hh.sites, NH.block_s,
                                                  source_col, block_row)
        tkm2 = ket_probe
        tkm1 = apply(A_op, ket_probe; maxdim=maxdim, cutoff=cutoff)
        pkm2 = 0.0 * ket_probe
        pkm1 = apply(S,    ket_probe; maxdim=maxdim, cutoff=cutoff)

        partial_vals = zeros(ComplexF64, N)
        partial_vals[2] = inner(bra_probe, pkm1)

        for k in 3:N
            tk = +(2.0 * apply(A_op, tkm1; maxdim=maxdim, cutoff=cutoff),
                   -tkm2; maxdim=maxdim, cutoff=cutoff)
            pk = +(+(2.0 * apply(S,    tkm1; maxdim=maxdim, cutoff=cutoff),
                     2.0 * apply(A_op, pkm1; maxdim=maxdim, cutoff=cutoff);
                     maxdim=maxdim, cutoff=cutoff),
                   -pkm2; maxdim=maxdim, cutoff=cutoff)
            partial_vals[k] = inner(bra_probe, pk)
            tkm2 = tkm1; tkm1 = tk
            pkm2 = pkm1; pkm1 = pk
        end

        val = ComplexF64(0)
        for l in 2:2:N
            val += (-1)^(l ÷ 2 - 1) * weights[l - 1] * partial_vals[l]
        end
        dos_acc += val
    end

    return real(dos_acc * D * 2.0 / (π^2 * (N + 1) * n_random))
end


# ============================================================
# 4. Complex-energy grid driver
# ============================================================

"""
    nh_spectrum_grid(H, xlims, nx, ylims, ny, n; scale=nothing,
                     nh_scale_padding=1.05, convention=:z_minus_H,
                     block_placement=:post, mode=:scalar, probe_site=0,
                     n_random=10, maxdim=100, cutoff=1e-8, dmrg_nsweeps=5,
                     dmrg_maxdim=[10, 20, 40], dmrg_linkdim=4, verbose=false)

Evaluate the NH KPM spectral weight on a rectangular complex energy grid.

**Modes**

| `mode`          | Algorithm                           | Returns                        |
|-----------------|-------------------------------------|--------------------------------|
| `:scalar`       | MPO×MPO (default)                   | `(xgrid, ygrid, Z)`            |
| `:mps`          | online dual-chain MPS, one site     | `(xgrid, ygrid, Z)`            |
| `:diag`         | MPO×MPO + diagonal extraction       | `(xgrid, ygrid, Z, Z_spatial)` |
| `:stochastic`   | stochastic trace, `n_random` probes | `(xgrid, ygrid, Z)`            |

- `:scalar` — full NH partial MPO recursion; total DOS. O(Ncheb × χ_P²).
- `:mps` — dual-chain MPS at a single site (`probe_site`, 0-indexed). LDOS at
  that site. O(χ_H × χ_ψ) per step.
- `:diag` — same as `:scalar` but also extracts site-resolved diagonal MPS A(r,z).
  Extra return `Z_spatial` has shape `(H.N, ny, nx)`.
- `:stochastic` — Monte Carlo trace: average over `n_random` random product-state
  probes. Total DOS estimate. O(n_random × Ncheb × χ_H × χ_ψ). No MPO×MPO products.

Set `verbose=true` to print one progress line per Re(z) column.
If `scale` is omitted or zero, a single conservative scale is estimated from
`nh_kpm_scale` and reused over the whole grid.
"""
function nh_spectrum_grid(H::TBHamiltonian, xlims, nx::Int, ylims, ny::Int, n::Int;
                          scale::Union{Nothing,Real} = nothing,
                          nh_scale_padding::Real = 1.05,
                          convention::Symbol      = :z_minus_H,
                          block_placement::Symbol = :post,
                          mode::Symbol            = :scalar,
                          probe_site::Int         = 0,
                          n_random::Int           = 10,
                          maxdim::Int             = 100,
                          cutoff::Real            = 1e-8,
                          dmrg_nsweeps::Int       = 5,
                          dmrg_maxdim             = [10, 20, 40],
                          dmrg_linkdim::Int       = 4,
                          verbose::Bool           = false)
    mode in (:scalar, :mps, :diag, :stochastic) ||
        error("Unknown mode :$mode for nh_spectrum_grid. Choose :scalar, :mps, :diag, or :stochastic.")

    xgrid = range(xlims[1], xlims[2]; length=nx)
    ygrid = range(ylims[1], ylims[2]; length=ny)
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
        printinfo=verbose)
    Z         = Matrix{ComplexF64}(undef, ny, nx)
    Z_spatial = (mode === :diag) ? zeros(Float64, H.N, ny, nx) : nothing

    verbose && println("nh_spectrum_grid [mode=:$mode]: $(nx)×$(ny)=$(nx*ny) points, Ncheb=$(2n), scale=$nh_scale")

    for (ix, x) in enumerate(xgrid)
        verbose && print("  col $(lpad(ix, ndigits(nx)))/$(nx)  Re(z)=$(round(x, digits=4)) ...")
        for (iy, y) in enumerate(ygrid)
            NH = hermitize(H; z=x + 1im*y, scale=nh_scale, maxdim=maxdim,
                           cutoff=cutoff, convention=convention,
                           block_placement=block_placement)
            if mode === :mps
                Z[iy, ix] = _nh_kpm_mps_ldos(NH, n, probe_site;
                                               scale=nh_scale, maxdim=maxdim, cutoff=cutoff)
            elseif mode === :diag
                A_mps, dos = _nh_diag_online(NH, n;
                                              scale=nh_scale, maxdim=maxdim, cutoff=cutoff)
                Z[iy, ix] = dos
                for i in 0:H.N-1
                    Z_spatial[i+1, iy, ix] = real(eval_mps(A_mps, i))
                end
            elseif mode === :stochastic
                Z[iy, ix] = _nh_stochastic_online(NH, n;
                                                   scale=nh_scale, n_random=n_random,
                                                   maxdim=maxdim, cutoff=cutoff)
            else  # :scalar
                Z[iy, ix] = _nh_scalar_online(NH, n;
                                               scale=nh_scale, maxdim=maxdim, cutoff=cutoff)
            end
        end
        verbose && println("  done")
    end

    return mode === :diag ? (xgrid, ygrid, Z, Z_spatial) : (xgrid, ygrid, Z)
end
