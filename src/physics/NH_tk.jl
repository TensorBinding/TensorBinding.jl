# NH_tk.jl -- non-Hermitian Hamiltonian helpers
#
# The core construction here hermitizes a non-Hermitian single-particle MPO by
# adding one dim-2 auxiliary block index:
#
#       H_NH(z) = [ 0        zI - H ;
#                  (zI - H)'     0  ]
#
# In TensorBinding terms this is
#       (zI - H) x |1><2| + (zI - H)' x |2><1|
# using postpend_op so the block site lives at the END of the site list:
#   hermitized.sites = [pos_sites..., block_s]
# Position sites therefore occupy indices 1:L with the standard big-endian
# quantics encoding, matching binary_to_MPS / eval_mps directly.

"""
    NonHermitianHamiltonian

Wrapper for a non-Hermitian `TBHamiltonian` together with its hermitized
block Hamiltonian.

Fields
------
- `parent`     : original `TBHamiltonian`, not modified in-place
- `z`          : complex reference point used in `zI - H`
- `block_s`    : dim-2 auxiliary Index tagged `"NHBlock"`
- `hermitized` : Hermitian `TBHamiltonian` on `[parent.sites...; block_s]`

The hermitized Hamiltonian can be passed to existing MPO/KPM routines. Avoid
calling tight-binding mutation helpers like `add_onsite!` on `hermitized`;
mutate `parent` first and call `hermitize` again.
"""
mutable struct NonHermitianHamiltonian
    parent          :: TBHamiltonian
    z               :: ComplexF64
    block_s         :: Index
    hermitized      :: TBHamiltonian
    block_placement :: Symbol   # :pre  → [block_s; pos_sites...]
                                # :post → [pos_sites...; block_s]
end

"""
    nh_block_index() -> Index

Create the dim-2 auxiliary block index used for non-Hermitian hermitization.
State 1 is the upper block, state 2 is the lower block. The index keeps a
`"Qubit"` tag so generic `MPO(sites, "Id")` construction remains compatible
with the existing KPM code paths.
"""
nh_block_index() = Index(2, "Qubit,NHBlock")

"""
    hermitized_hamiltonian(H; z=0, block_s=nh_block_index(), cutoff=1e-8,
                           maxdim=200, scale=0.0, block_placement=:post) -> TBHamiltonian

Return the Hermitian block Hamiltonian

```text
[ 0        zI - H ;
  (zI-H)'  0      ]
```

as a `TBHamiltonian`. `block_placement` controls where the auxiliary block site lives:
- `:post` (default) — site order `[H.sites...; block_s]`; position qubits occupy 1:L directly
- `:pre`            — site order `[block_s; H.sites...]`; original layout before postpend change

`scale=0.0` keeps the usual lazy KPM spectral-bound estimation.
"""
function hermitized_hamiltonian(H::TBHamiltonian;
                                z::Number = 0.0,
                                block_s::Index = nh_block_index(),
                                cutoff::Real = 1e-8,
                                maxdim::Int = 200,
                                scale::Real = 0.0,
                                convention::Symbol = :z_minus_H,
                                block_placement::Symbol = :post)
    convention in (:H_minus_z, :z_minus_H) ||
        error("convention must be :H_minus_z or :z_minus_H; got :$convention")
    I_H     = MPO(H.sites, "Id")
    H_shift = convention === :H_minus_z ?
        +(H.mpo, -ComplexF64(z) * I_H; cutoff=cutoff) :
        +(ComplexF64(z) * I_H, -H.mpo; cutoff=cutoff)
    H_adj   = swapprime(dag(H_shift), 0, 1)

    block_placement in (:pre, :post) ||
        error("block_placement must be :pre or :post; got :$block_placement")
    if block_placement === :post
        H_block = +(postpend_op(H_shift, block_s, 1, 2),
                    postpend_op(H_adj,   block_s, 2, 1); cutoff=cutoff)
        sites = [H.sites...; block_s]
    else
        H_block = +(prepend_op(H_shift, block_s, 1, 2),
                    prepend_op(H_adj,   block_s, 2, 1); cutoff=cutoff)
        sites = [block_s; H.sites...]
    end
    ITensorMPS.truncate!(H_block; cutoff=cutoff, maxdim=maxdim)

    return TBHamiltonian(H.L, H.N, sites, H_block,
                         H.geometry, H.geometry_uc,
                         Float64(scale), 0.0,
                         H.spin_s, H.nambu_s, H.layer_s, H.sublattice_s, :pre,
                         nothing, nothing, 0, nothing)
end

"""
    hermitize(H; z=0, cutoff=1e-8, maxdim=200, scale=0.0)
        -> NonHermitianHamiltonian

Build a `NonHermitianHamiltonian` wrapper without modifying `H`.
"""
function hermitize(H::TBHamiltonian;
                   z::Number = 0.0,
                   cutoff::Real = 1e-8,
                   maxdim::Int = 200,
                   scale::Real = 0.0,
                   convention::Symbol = :z_minus_H,
                   block_placement::Symbol = :post)
    block_s = nh_block_index()
    Hh = hermitized_hamiltonian(H;
                                z=z,
                                block_s=block_s,
                                cutoff=cutoff,
                                maxdim=maxdim,
                                scale=scale,
                                convention=convention,
                                block_placement=block_placement)
    return NonHermitianHamiltonian(H, ComplexF64(z), block_s, Hh, block_placement)
end

"""
    hermitize(NH; z=NH.z, cutoff=1e-8, maxdim=200, scale=0.0)

Rebuild the hermitized block Hamiltonian from `NH.parent`, optionally at a new
reference point `z`.
"""
function hermitize(NH::NonHermitianHamiltonian;
                   z::Number = NH.z,
                   cutoff::Real = 1e-8,
                   maxdim::Int = 200,
                   scale::Real = 0.0,
                   convention::Symbol = :z_minus_H,
                   block_placement::Symbol = NH.block_placement)
    return hermitize(NH.parent; z=z, cutoff=cutoff, maxdim=maxdim, scale=scale,
                     convention=convention, block_placement=block_placement)
end

function Base.show(io::IO, NH::NonHermitianHamiltonian)
    print(io, "NonHermitianHamiltonian | z=$(NH.z), " *
              "blockdim=$(ITensors.dim(NH.block_s)), " *
              "hermitized maxlinkdim=$(ITensorMPS.maxlinkdim(NH.hermitized.mpo))")
end

# ============================================================
# Non-Hermitian model-building helpers
# ============================================================

function _nh_position_sites_only(H::TBHamiltonian)
    (H.spin_s === nothing && H.nambu_s === nothing &&
     H.layer_s === nothing && H.sublattice_s === nothing) ||
        error("NH_tk model-building helpers currently expect a position-only TBHamiltonian. " *
              "Add non-Hermitian terms before adding spin/Nambu/layer/sublattice auxiliaries, " *
              "or use a custom MPO term directly.")
    return _pos_sites(H)
end

function _nh_diagonal_mpo(L::Int, sites, f; Lx=nothing, type=ComplexF64)
    if f isa Number
        return ComplexF64(f) * MPO(sites, "Id")
    elseif applicable(f, 0, 0)
        Lx !== nothing ||
            error("2D onsite function f(ix,iy) requires Lx=... so Nx=2^Lx is known.")
        Nx = 2^Lx
        return get_diagonal_mpo(L, sites,
                                i -> (n = round(Int, i) - 1; f(n % Nx, n ÷ Nx));
                                type=type)
    elseif applicable(f, 0)
        return get_diagonal_mpo(L, sites, i -> f(round(Int, i) - 1); type=type)
    else
        error("Unsupported onsite signature. Use a Number, f(n), or f(ix,iy).")
    end
end

function _nh_profile_diagonal_mpo(L::Int, sites, f; Lx=nothing, type=Float64)
    if f isa Number
        return get_diagonal_mpo(L, sites, _ -> f; type=type)
    elseif applicable(f, 0, 0)
        Lx !== nothing ||
            error("2D profile function f(ix,iy) requires Lx=... so Nx=2^Lx is known.")
        Nx = 2^Lx
        return get_diagonal_mpo(L, sites,
                                i -> (n = round(Int, i) - 1; f(n % Nx, n ÷ Nx));
                                type=type)
    elseif applicable(f, 0)
        return get_diagonal_mpo(L, sites, i -> f(round(Int, i) - 1); type=type)
    else
        error("Unsupported profile signature. Use a Number, f(n), or f(ix,iy).")
    end
end

function _nh_fullspace_diagonal_mpo(H::TBHamiltonian, f; Lx=nothing, type=Float64)
    all(dim(s) == 2 for s in H.sites) ||
        error("Full-space diagonal loss currently requires all H.sites to be dim-2. " *
              "For mixed-dimensional auxiliary spaces, build the desired MPO explicitly.")
    Lfull = length(H.sites)
    return _nh_profile_diagonal_mpo(Lfull, H.sites, f; Lx=Lx, type=type)
end

"""
    add_nh_onsite!(H, v; Lx=nothing, tol=1e-8, maxdim=200, type=ComplexF64)

Add a possibly complex onsite potential to a position-only `TBHamiltonian`.

`v` may be:
- a number, e.g. `1im * gamma`
- `v(n)` with `n = 0, ..., H.N-1`
- `v(ix, iy)` with 0-indexed coordinates, requiring `Lx=...`

This is the non-Hermitian counterpart of `add_onsite!`; unlike the generic
version it preserves complex values by default.
"""
function add_nh_onsite!(H::TBHamiltonian, v;
                        Lx=nothing,
                        tol::Real = 1e-8,
                        maxdim::Int = 200,
                        type = ComplexF64)
    pos_s = _nh_position_sites_only(H)
    term = _nh_diagonal_mpo(H.L, pos_s, v; Lx=Lx, type=type)
    H.mpo = +(H.mpo, term; cutoff=tol, maxdim=maxdim)
    ITensorMPS.truncate!(H.mpo; cutoff=tol, maxdim=maxdim)
    _invalidate_cache!(H)
    return H
end

"""
    loss_profile_mpo(H, f; Lx=nothing, type=Float64, space=:full)

Build the real diagonal profile MPO `diag(f)` used for loss/gain terms. This
function does not multiply by `im` and does not hermitize anything.

By default `space=:full`, so the diagonal is built on the full Hamiltonian
site space `H.sites`, with basis coordinate `n = 0, ..., 2^length(H.sites)-1`.
Use `space=:position` only when the profile should live on position qubits
before any auxiliary spaces are attached.
"""
function loss_profile_mpo(H::TBHamiltonian, f;
                          Lx=nothing,
                          type = Float64,
                          space::Symbol = :full)
    if space === :full
        _nh_fullspace_diagonal_mpo(H, f; Lx=Lx, type=type)
    elseif space === :position
        pos_s = _nh_position_sites_only(H)
        _nh_profile_diagonal_mpo(H.L, pos_s, f; Lx=Lx, type=type)
    else
        error("space must be :full or :position; got :$space")
    end
end

"""
    nh_imag_onsite_mpo(H, f; prefactor=1im, Lx=nothing, type=Float64,
                       space=:full)

Build `prefactor * diag(f)`. The real profile MPO is constructed first by
`loss_profile_mpo`, then the imaginary prefactor is multiplied afterward.
This returns a term on the original Hilbert space, not an NH hermitized block.
"""
function nh_imag_onsite_mpo(H::TBHamiltonian, f;
                            prefactor::Number = 1im,
                            Lx=nothing,
                            type = Float64,
                            space::Symbol = :full)
    return ComplexF64(prefactor) * loss_profile_mpo(H, f; Lx=Lx, type=type, space=space)
end

"""
    add_loss!(H, f; coefficient=-1im, space=:full, ...)

Add a loss/gain term `coefficient * diag(f)` to the original Hamiltonian MPO.
This only modifies `H.mpo`; it does not create the hermitized NH block.
"""
function add_loss!(H::TBHamiltonian, f;
                   coefficient::Number = -1im,
                   Lx=nothing,
                   tol::Real = 1e-8,
                   maxdim::Int = 200,
                   type = Float64,
                   space::Symbol = :full)
    term = ComplexF64(coefficient) * loss_profile_mpo(H, f; Lx=Lx, type=type, space=space)
    H.mpo = +(H.mpo, term; cutoff=tol, maxdim=maxdim)
    ITensorMPS.truncate!(H.mpo; cutoff=tol, maxdim=maxdim)
    _invalidate_cache!(H)
    return H
end

"""
    add_nh_imag_onsite!(H, f; prefactor=1im, space=:full, ...)

Backward-compatible name for adding `prefactor * diag(f)` to the original
Hamiltonian MPO. This is not hermitization.
"""
function add_nh_imag_onsite!(H::TBHamiltonian, f;
                             prefactor::Number = 1im,
                             kwargs...)
    return add_loss!(H, f; coefficient=prefactor, kwargs...)
end

"""
    add_nh_loss!(H, f; prefactor=-1im, space=:full, ...)

Convenience wrapper for onsite loss. By default this adds `-im * diag(f)`.
Use `prefactor=1im` for gain or for the convention `i*f(x)`.
"""
function add_nh_loss!(H::TBHamiltonian, f;
                      prefactor::Number = -1im,
                      kwargs...)
    return add_loss!(H, f; coefficient=prefactor, kwargs...)
end

function _nh_directional_hop(pos_s, N::Int, amplitude, nn::Integer, direction::Symbol;
                             L::Int,
                             tol::Real,
                             maxdim::Int,
                             type)
    nn >= 1 || error("nn must be >= 1 for directional hopping.")
    K_nn = direction === :forward ? shift_mpo(pos_s, nn; cyclic=false) :
        direction === :backward ? shift_mpo(pos_s, -nn; cyclic=false) :
        error("direction must be :forward or :backward.")

    if amplitude isa Number
        return ComplexF64(amplitude) * K_nn
    elseif applicable(amplitude, 0)
        A = get_diagonal_mpo(L, pos_s, i -> amplitude(round(Int, i) - 1); type=type)
        return direction === :forward ?
            apply(A, K_nn; cutoff=tol, maxdim=maxdim) :
            apply(K_nn, A; cutoff=tol, maxdim=maxdim)
    else
        error("Directional hopping amplitude must be a Number or f(n).")
    end
end

"""
    nh_nonreciprocal_hopping_mpo(H, t_forward, t_backward; nn=1, ...)

Build the position-space MPO

```text
t_forward  * K_+^nn + t_backward * K_-^nn
```

without imposing Hermiticity. `t_forward` and `t_backward` may be numbers or
site-dependent one-argument functions `t(n)` with 0-indexed `n`.
"""
function nh_nonreciprocal_hopping_mpo(H::TBHamiltonian, t_forward, t_backward;
                                      nn::Integer = 1,
                                      tol::Real = 1e-8,
                                      maxdim::Int = 200,
                                      type = ComplexF64)
    pos_s = _nh_position_sites_only(H)
    Hf = _nh_directional_hop(pos_s, H.N, t_forward, nn, :forward;
                             L=H.L, tol=tol, maxdim=maxdim, type=type)
    Hb = _nh_directional_hop(pos_s, H.N, t_backward, nn, :backward;
                             L=H.L, tol=tol, maxdim=maxdim, type=type)
    return +(Hf, Hb; cutoff=tol, maxdim=maxdim)
end

"""
    add_nh_nonreciprocal_hopping!(H, t_forward, t_backward; nn=1, ...)

Add asymmetric hopping directly to `H`. This is the skin-effect helper:
choose, for example, `t_forward=t*exp(g)` and `t_backward=t*exp(-g)`.
"""
function add_nh_nonreciprocal_hopping!(H::TBHamiltonian, t_forward, t_backward;
                                       nn::Integer = 1,
                                       tol::Real = 1e-8,
                                       maxdim::Int = 200,
                                       type = ComplexF64)
    term = nh_nonreciprocal_hopping_mpo(H, t_forward, t_backward;
                                        nn=nn, tol=tol, maxdim=maxdim, type=type)
    H.mpo = +(H.mpo, term; cutoff=tol, maxdim=maxdim)
    ITensorMPS.truncate!(H.mpo; cutoff=tol, maxdim=maxdim)
    _invalidate_cache!(H)
    return H
end

"""
    add_nh_skin_hopping!(H, t, g; nn=1, convention=:exp)

Convenience wrapper for non-reciprocal skin hopping.

- `convention=:exp` uses `t_R = t * exp(g)`, `t_L = t * exp(-g)`.
- `convention=:linear` uses `t_R = t + g`, `t_L = t - g`.
"""
function add_nh_skin_hopping!(H::TBHamiltonian, t, g;
                              nn::Integer = 1,
                              convention::Symbol = :exp,
                              tol::Real = 1e-8,
                              maxdim::Int = 200)
    t_forward, t_backward = if convention === :exp
        (t * exp(g), t * exp(-g))
    elseif convention === :linear
        (t + g, t - g)
    else
        error("Unknown skin convention :$convention. Use :exp or :linear.")
    end
    return add_nh_nonreciprocal_hopping!(H, t_forward, t_backward;
                                         nn=nn, tol=tol, maxdim=maxdim)
end

# ============================================================
# Non-Hermitian KPM partial recursion
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
    nh_kpm_partials(Hh, n; source, scale, maxdim=100, cutoff=1e-8)
        -> Vector{MPO}

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
                         maxdim::Int = 100,
                         cutoff::Real = 1e-8)
    S = isnothing(source) ? nh_block_source(NH; row=source_row, col=source_col) : source
    return nh_kpm_partials(NH.hermitized, n; source=S, scale=scale,
                           maxdim=maxdim, cutoff=cutoff)
end

"""
    contract_nh_block(W, block_s; row=2, col=1) -> MPO

Extract the `(row, col)` block of an MPO whose first or last site is `block_s`,
returning an MPO on the remaining sites. The block position is auto-detected.
"""
function contract_nh_block(W::MPO, block_s::Index; row::Int = 2, col::Int = 1)
    M = length(W)
    M >= 2 || error("contract_nh_block requires an MPO with a block site and at least one physical site.")
    if siteind(W, M) == block_s
        # Postpend: block is last
        bt = W[M] * onehot(block_s => col) * onehot(block_s' => row)
        tensors = ITensor[W[i] for i in 1:M-2]
        push!(tensors, W[M-1] * bt)
        return MPO(tensors)
    elseif siteind(W, 1) == block_s
        # Prepend: block is first
        bt = W[1] * onehot(block_s => col) * onehot(block_s' => row)
        tensors = ITensor[W[2] * bt]
        for i in 3:M
            push!(tensors, W[i])
        end
        return MPO(tensors)
    else
        error("NH block index must be the first or last MPO site for contract_nh_block.")
    end
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

function nh_jackson_weights(N::Int)
    return [((N - k + 1) * cos(pi * k / (N + 1)) +
             sin(pi * k / (N + 1)) / tan(pi / (N + 1))) for k in 0:N-1]
end

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
    weights = nh_jackson_weights(N)
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
    nh_reconstruct_spectral_mpo(partials, n, NH; maxdim=400, cutoff=1e-8)
        -> (ldos_mps, dos, rotated_mpo)

High-bond-dimension reconstruction of the non-Hermitian spectral object.

This mirrors the older all-site LDOS workflow: first reconstruct the full
partial MPO on the hermitized block space, then left-multiply by
`|1><2| x I` and take its trace / diagonal. Compared with
`nh_reconstruct_spectral_mps`, this keeps the full MPO until the end, so it can
represent all sites at once but usually needs a much larger `maxdim`.
"""
function nh_reconstruct_spectral_mpo(partials::AbstractVector{<:MPO}, n::Int,
                                     NH::NonHermitianHamiltonian;
                                     maxdim::Int = 400,
                                     cutoff::Real = 1e-8,
                                     rotate_row::Int = 1,
                                     rotate_col::Int = 2,
                                     diag_block::Int = 1)
    N = 2 * n
    length(partials) >= N || error("Expected at least $N partials, got $(length(partials)).")
    weights = nh_jackson_weights(N)

    A = partials[1]
    for l in 2:2:N
        order = (-1)^((l ÷ 2) - 1)
        A = +(A, order * weights[l - 1] * partials[l];
              maxdim=maxdim, cutoff=cutoff)
    end
    A *= 2.0 / (pi^2 * (N + 1))

    rotator = nh_block_source(NH; row=rotate_row, col=rotate_col)
    rotated = apply(rotator, A; maxdim=maxdim, cutoff=cutoff)
    dos = tr(rotated)

    ldos_block = contract_nh_block(rotated, NH.block_s;
                                   row=diag_block, col=diag_block)
    ldos_mps = extract_diagonal_to_mps(ldos_block)
    return ldos_mps, dos, rotated
end

"""
    nh_spectral_function(NH, n; scale, maxdim=100, cutoff=1e-8,
                         source_row=2, source_col=1, block_row=2, block_col=1)
        -> (A_mps, dos, partials)

Convenience wrapper for the full non-Hermitian KPM spectral calculation at
the reference point stored in `NH.z`.
"""
function nh_spectral_function(NH::NonHermitianHamiltonian, n::Int;
                              scale::Union{Nothing,Real} = nothing,
                              maxdim::Int = 100,
                              cutoff::Real = 1e-8,
                              source_row::Int = 2,
                              source_col::Int = 1,
                              block_row::Int = 2,
                              block_col::Int = 1)
    partials = nh_kpm_partials(NH, n; source_row=source_row, source_col=source_col,
                               scale=scale, maxdim=maxdim, cutoff=cutoff)
    A, dos = nh_reconstruct_spectral_mps(partials, n, NH.block_s;
                                         maxdim=maxdim, row=block_row, col=block_col)
    return A, dos, partials
end

"""
    nh_spectral_function_allsite_mpo(NH, n; scale, maxdim=400, cutoff=1e-8)
        -> (ldos_mps, dos, rotated_mpo, partials)

Alternative non-default NH KPM path that reconstructs the full spectral MPO
before extracting the LDOS. It can compute all sites in one object, but needs
larger bond dimensions for accuracy.
"""
function nh_spectral_function_allsite_mpo(NH::NonHermitianHamiltonian, n::Int;
                                          scale::Union{Nothing,Real} = nothing,
                                          maxdim::Int = 400,
                                          cutoff::Real = 1e-8,
                                          source_row::Int = 2,
                                          source_col::Int = 1,
                                          rotate_row::Int = 1,
                                          rotate_col::Int = 2,
                                          diag_block::Int = 1)
    partials = nh_kpm_partials(NH, n; source_row=source_row, source_col=source_col,
                               scale=scale, maxdim=maxdim, cutoff=cutoff)
    ldos_mps, dos, rotated = nh_reconstruct_spectral_mpo(
        partials, n, NH;
        maxdim=maxdim,
        cutoff=cutoff,
        rotate_row=rotate_row,
        rotate_col=rotate_col,
        diag_block=diag_block,
    )
    return ldos_mps, dos, rotated, partials
end

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
    _nh_kpm_mps_ldos(NH, n, probe_site; scale, maxdim, cutoff) -> Real

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

    weights = nh_jackson_weights(N)
    dos = ComplexF64(0)
    for l in 2:2:N
        dos += (-1)^(l ÷ 2 - 1) * weights[l - 1] * partial_vals[l]
    end
    return real(dos * 2.0 / (π^2 * (N + 1)))
end


"""
    _nh_scalar_online(NH, n; scale, maxdim, cutoff) -> ComplexF64

Online NH KPM scalar DOS: run the partial Chebyshev recursion and accumulate
Tr[block_{2,1}(P_k)] contributions in a single pass, keeping only two partial
MPOs in memory at a time.

Avoids the O(N·χ_P²) memory cost of `nh_kpm_partials` and skips building the
intermediate A_mps entirely — each contributing step adds only one scalar to
the accumulator.
"""
function _nh_scalar_online(NH::NonHermitianHamiltonian, n::Int;
                            scale::Union{Nothing,Real} = nothing,
                            maxdim::Int  = 100,
                            cutoff::Real = 1e-8,
                            source_row::Int = 2,
                            source_col::Int = 1,
                            block_row::Int  = 2,
                            block_col::Int  = 1)
    N  = 2 * n
    Hh = NH.hermitized
    sc = isnothing(scale) ? Hh.scale : Float64(scale)
    sc == 0.0 && error("_nh_scalar_online requires a nonzero scale.")

    A_op    = Hh.mpo / sc
    source  = nh_block_source(NH; row=source_row, col=source_col)
    weights = nh_jackson_weights(N)
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
    _nh_diag_online(NH, n; scale, maxdim, cutoff) -> (A_mps, dos)

Online NH KPM diagonal spectral function: run the partial Chebyshev recursion
and accumulate the site-resolved diagonal MPS A(r, z) in a single pass, keeping
only two partial MPOs in memory at a time.

Compared with `nh_kpm_partials` + `nh_reconstruct_spectral_mps`:
  - Memory: O(2 χ_P²) instead of O(N χ_P²).
  - Diagonal extractions: N/2 (only even Julia-index partials contribute).
"""
function _nh_diag_online(NH::NonHermitianHamiltonian, n::Int;
                          scale::Union{Nothing,Real} = nothing,
                          maxdim::Int  = 100,
                          cutoff::Real = 1e-8,
                          source_row::Int = 2,
                          source_col::Int = 1,
                          block_row::Int  = 2,
                          block_col::Int  = 1)
    N  = 2 * n
    Hh = NH.hermitized
    sc = isnothing(scale) ? Hh.scale : Float64(scale)
    sc == 0.0 && error("_nh_diag_online requires a nonzero scale.")

    A_op    = Hh.mpo / sc
    source  = nh_block_source(NH; row=source_row, col=source_col)
    weights = nh_jackson_weights(N)

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
    _nh_stochastic_online(NH, n; scale, n_random=10, maxdim, cutoff) -> Real

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
                                n_random::Int  = 10,
                                maxdim::Int    = 100,
                                cutoff::Real   = 1e-8,
                                source_row::Int = 2,
                                source_col::Int = 1,
                                block_row::Int  = 2,
                                block_col::Int  = 1)
    N  = 2 * n
    Hh = NH.hermitized
    sc = isnothing(scale) ? Hh.scale : Float64(scale)
    sc == 0.0 && error("_nh_stochastic_online requires a nonzero scale.")

    A_op    = Hh.mpo / sc
    S       = nh_block_source(NH; row=source_row, col=source_col)
    weights = nh_jackson_weights(N)
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


"""
    nh_spectrum_grid(H, xlims, nx, ylims, ny, n; scale, convention=:z_minus_H,
                     mode=:scalar, probe_site=0, n_random=10,
                     maxdim=100, cutoff=1e-8, verbose=false)

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
"""
function nh_spectrum_grid(H::TBHamiltonian, xlims, nx::Int, ylims, ny::Int, n::Int;
                          scale::Real,
                          convention::Symbol      = :z_minus_H,
                          block_placement::Symbol = :post,
                          mode::Symbol            = :scalar,
                          probe_site::Int         = 0,
                          n_random::Int           = 10,
                          maxdim::Int             = 100,
                          cutoff::Real            = 1e-8,
                          verbose::Bool           = false)
    mode in (:scalar, :mps, :diag, :stochastic) ||
        error("Unknown mode :$mode for nh_spectrum_grid. Choose :scalar, :mps, :diag, or :stochastic.")

    xgrid = range(xlims[1], xlims[2]; length=nx)
    ygrid = range(ylims[1], ylims[2]; length=ny)
    Z         = Matrix{ComplexF64}(undef, ny, nx)
    Z_spatial = (mode === :diag) ? zeros(Float64, H.N, ny, nx) : nothing

    verbose && println("nh_spectrum_grid [mode=:$mode]: $(nx)×$(ny)=$(nx*ny) points, Ncheb=$(2n)")

    for (ix, x) in enumerate(xgrid)
        verbose && print("  col $(lpad(ix, ndigits(nx)))/$(nx)  Re(z)=$(round(x, digits=4)) ...")
        for (iy, y) in enumerate(ygrid)
            NH = hermitize(H; z=x + 1im*y, scale=scale, maxdim=maxdim,
                           cutoff=cutoff, convention=convention,
                           block_placement=block_placement)
            if mode === :mps
                Z[iy, ix] = _nh_kpm_mps_ldos(NH, n, probe_site;
                                               scale=scale, maxdim=maxdim, cutoff=cutoff)
            elseif mode === :diag
                A_mps, dos = _nh_diag_online(NH, n;
                                              scale=scale, maxdim=maxdim, cutoff=cutoff)
                Z[iy, ix] = dos
                for i in 0:H.N-1
                    Z_spatial[i+1, iy, ix] = real(eval_mps(A_mps, i))
                end
            elseif mode === :stochastic
                Z[iy, ix] = _nh_stochastic_online(NH, n;
                                                   scale=scale, n_random=n_random,
                                                   maxdim=maxdim, cutoff=cutoff)
            else  # :scalar
                Z[iy, ix] = _nh_scalar_online(NH, n;
                                               scale=scale, maxdim=maxdim, cutoff=cutoff)
            end
        end
        verbose && println("  done")
    end

    return mode === :diag ? (xgrid, ygrid, Z, Z_spatial) : (xgrid, ygrid, Z)
end
