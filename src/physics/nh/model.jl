# nh/model.jl -- non-Hermitian Hamiltonian model: the NonHermitianHamiltonian
# wrapper and hermitization (hermitize, hermitized_hamiltonian), plus the
# non-Hermitian model-building helpers (add_nh_onsite!, loss_profile_mpo,
# add_loss!, non-reciprocal / skin hopping). Split from the former physics/NH_tk.jl; the
# NH KPM routines acting on the hermitized block Hamiltonian are in nh/kpm.jl.
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

# ============================================================
# Hermitization: NonHermitianHamiltonian wrapper
# ============================================================

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
    _require_binary_position_space(H, "hermitized_hamiltonian")
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

    # The block operator is not a physical Hamiltonian: interactions stay on the parent.
    return TBHamiltonian(H; sites=sites, mpo=H_block, scale=Float64(scale), center=0.0,
                         aux_side=:pre, interaction_mpo=nothing, fock_mpo=nothing)
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
    _require_binary_position_space(H, "add_nh_onsite!")
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
    _require_binary_position_space(H, "add_loss!")
    term = ComplexF64(coefficient) * loss_profile_mpo(H, f; Lx=Lx, type=type, space=space)
    H.mpo = +(H.mpo, term; cutoff=tol, maxdim=maxdim)
    ITensorMPS.truncate!(H.mpo; cutoff=tol, maxdim=maxdim)
    _invalidate_cache!(H)
    return H
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
    _require_binary_position_space(H, "nh_nonreciprocal_hopping_mpo")
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
