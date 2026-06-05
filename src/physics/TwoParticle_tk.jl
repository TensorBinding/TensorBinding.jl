# twoparticle_tk.jl — exciton and two-particle system construction

# ─────────────────────────────────────────────────────────────────
# 1.  High-level constructor (returns TBHamiltonian)
# ─────────────────────────────────────────────────────────────────

"""
    exciton_hamiltonian(geometry, params, Ufunc; L, [on_site, scale, tol_quantics,
        maxbonddim_quantics, tol, cutoff, maxdim, kwargs...]) -> TBHamiltonian

    exciton_hamiltonian(H_c, H_v, Ufunc; [on_site, scale, tol_quantics,
        maxbonddim_quantics, tol, cutoff, maxdim]) -> TBHamiltonian

Build an exciton Hamiltonian and wrap it in a `TBHamiltonian` for use with
TensorBinding's KPM, DMRG, and spectral tools.

**Site encoding** (`2L` sites total, `L = H_c.L`):
- Odd sites  (1, 3, …) : electron position qubits
- Even sites (2, 4, …) : hole position qubits (interleaved)

`TBHamiltonian.L = L` counts position qubits per sector;
`TBHamiltonian.sites` holds all `2L` interleaved MPO sites.

**Two calling modes**

1. **Geometry string** — both bands built from the same geometry and parameters
   via `get_Hamiltonian`; any extra `kwargs` are forwarded to it:
   ```julia
   H = exciton_hamiltonian("square_2d", t, Ufunc; L=Lx+Ly, Lx=Lx, Ly=Ly,
                            on_site = x -> V(x))
   ```

2. **Pre-built sectors** — pass explicit `TBHamiltonian` objects for the electron
   (`H_c`) and hole (`H_v`) bands when the two sectors differ (different hopping,
   disorder, external fields):
   ```julia
   H_c = get_Hamiltonian("chain_1d", t_c; L=L)
   H_v = get_Hamiltonian("chain_1d", t_v; L=L)
   H   = exciton_hamiltonian(H_c, H_v, Ufunc)
   ```

**Keyword arguments**
- `on_site`             : conduction band edge modulation `V(x)`, 1-indexed.
                          Applied as `+V` to electron and `−V` to valence sector
                          (type-I confinement). Compressed via QTCI.
- `scale`               : exciton spectral half-bandwidth (0.0 → lazy DMRG estimate).
- `tol_quantics`        : QTCI tolerance for `Ufunc` and `on_site`. Default `1e-8`.
- `maxbonddim_quantics` : QTCI max bond dimension. Default `100`.
- `tol`                 : MPO assembly truncation tolerance. Default `1e-8`.
- `cutoff`              : SVD cutoff for MPO arithmetic. Default `1e-8`.
- `maxdim`              : max bond dimension of the final MPO. Default `200`.
"""
function exciton_hamiltonian(geometry::String, params, Ufunc;
                              L::Int,
                              on_site              = nothing,
                              scale                = nothing,
                              tol_quantics         = 1e-8,
                              maxbonddim_quantics  = 100,
                              tol                  = 1e-8,
                              cutoff               = 1e-8,
                              maxdim               = 200,
                              kwargs...)
    H_c = get_Hamiltonian(geometry, params; L=L, tol=tol, maxdim=maxdim, kwargs...)
    H_v = get_Hamiltonian(geometry, params; L=L, tol=tol, maxdim=maxdim, kwargs...)
    return exciton_hamiltonian(H_c, H_v, Ufunc;
                               on_site             = on_site,
                               scale               = scale,
                               tol_quantics        = tol_quantics,
                               maxbonddim_quantics = maxbonddim_quantics,
                               tol=tol, cutoff=cutoff, maxdim=maxdim)
end

function exciton_hamiltonian(H_c::TBHamiltonian, H_v::TBHamiltonian, Ufunc;
                              on_site              = nothing,
                              scale                = nothing,
                              tol_quantics         = 1e-8,
                              maxbonddim_quantics  = 100,
                              tol                  = 1e-8,
                              cutoff               = 1e-8,
                              maxdim               = 200)
    H_exc_mpo = Exciton_Hamiltonian(H_c, H_v, Ufunc;
                                     on_site             = on_site,
                                     tol_quantics        = tol_quantics,
                                     maxbonddim_quantics = maxbonddim_quantics,
                                     tol=tol, cutoff=cutoff, maxdim=maxdim)
    sites_eh = collect(Iterators.flatten(zip(H_c.sites, H_v.sites)))
    sc       = something(scale, 0.0)   # 0.0 → lazy DMRG estimation on first KPM call
    # Inherit single-sector geometry from H_c (physical positions are the same).
    # L = position qubits per sector; N = physical positions; sites = 2L interleaved
    return TBHamiltonian(H_c.L, H_c.N, sites_eh, H_exc_mpo, H_c.geometry, sc, 0.0,
                         nothing, nothing, nothing, nothing, 0, nothing)
end


# ─────────────────────────────────────────────────────────────────
# 2.  Low-level MPO builder
# ─────────────────────────────────────────────────────────────────

"""
    Exciton_Hamiltonian(H_c, H_v, Ufunc; on_site, tol_quantics, maxbonddim_quantics,
                        tol, cutoff, maxdim) -> MPO

Build the exciton Hamiltonian on the interleaved 2L-site electron-hole space:

    H_exc = (H_c ⊗ I_h − I_e ⊗ H_v) + U

where `U` is the contact interaction diagonal MPO built from `Ufunc`.

`H_c` and `H_v` are `TBHamiltonian` objects for the electron and hole
single-particle sectors (any geometry: `"chain_1d"`, `"square_2d"`, etc.).
Both must have the same `L` and distinct site indices.
`Ufunc(x)` gives the interaction strength at site `x ∈ {1, …, 2^L}` (1-indexed).

**`on_site` keyword (optional):** a function `V(x)` representing the conduction
band edge modulation, compressed via QTCI. Applied as `+V` to the electron sector
and `−V` to the valence sector, so that the hole also feels `+V` after the
`H_c − H_v` subtraction (type-I semiconductor convention: both carriers confined).
`tol_quantics` and `maxbonddim_quantics` control the QTCI compression of `V`.

Examples
--------
```julia
# 1D, uniform hopping, contact interaction, Gaussian confinement
H_c = get_Hamiltonian("chain_1d", t; L=L)
H_v = get_Hamiltonian("chain_1d", t; L=L)
H_exc = Exciton_Hamiltonian(H_c, H_v, x -> -U;
                             on_site = x -> -V0 * exp(-((x - N/2)^2) / (2σ^2)))

# 2D square lattice
H_c = get_Hamiltonian("square_2d", t; L=Lx+Ly, Lx=Lx, Ly=Ly)
H_v = get_Hamiltonian("square_2d", t; L=Lx+Ly, Lx=Lx, Ly=Ly)
H_exc = Exciton_Hamiltonian(H_c, H_v, Ufunc; on_site = x -> dot_potential(x))
```
"""
function Exciton_Hamiltonian(H_c::TBHamiltonian, H_v::TBHamiltonian, Ufunc;
                              on_site              = nothing,
                              tol_quantics         = 1e-8,
                              maxbonddim_quantics  = 100,
                              tol                  = 1e-8,
                              cutoff               = 1e-8,
                              maxdim               = 200)
    @assert H_c.L == H_v.L "Electron and hole sectors must have the same system size"

    mpo_c = H_c.mpo
    mpo_v = H_v.mpo

    if on_site !== nothing
        # on_site(x) is compressed via QTCI on the 1-indexed grid {1, …, 2^L}.
        # +V on conduction band (electron), −V on valence band (hole).
        # After H_c − H_v, both carriers feel +V → type-I confinement.
        L     = H_c.L
        xvals = range(1, 2^L; length=2^L)
        V_c   = qtt_mpo(L, xvals, H_c.sites, on_site;
                        tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)
        V_v   = qtt_mpo(L, xvals, H_v.sites, on_site;
                        tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)
        mpo_c = +(mpo_c,       V_c; cutoff=tol)
        mpo_v = +(mpo_v, -1.0*V_v; cutoff=tol)
    end

    sites_eh    = collect(Iterators.flatten(zip(H_c.sites, H_v.sites)))
    kinetic_mpo = interleave_mpo(mpo_c, sites_eh, 0) - interleave_mpo(mpo_v, sites_eh, 1)
    # Use sites_eh directly (unprimed ket indices) — extracting sites from
    # siteinds(kinetic_mpo) risks grabbing the primed bra indices instead.
    interaction = build_interaction_op_exciton(H_c.L, sites_eh, Ufunc)
    return kinetic_mpo + interaction
end


"""
    hopping_mpo_exciton(H_c, H_v) -> MPO

Embed the electron (`H_c`) and hole (`H_v`) single-particle Hamiltonians
into the interleaved 2L-site electron-hole space:

    H_kin = H_c ⊗ I_h  -  I_e ⊗ H_v

`H_c` sits at odd sites (1, 3, 5, …) and `H_v` at even sites (2, 4, 6, …).
Both `TBHamiltonian` objects must have the same `L` and distinct site indices.
"""
function hopping_mpo_exciton(H_c::TBHamiltonian, H_v::TBHamiltonian)
    @assert H_c.L == H_v.L "Electron and hole sectors must have the same system size"
    sites_eh = collect(Iterators.flatten(zip(H_c.sites, H_v.sites)))
    return interleave_mpo(H_c.mpo, sites_eh, 0) - interleave_mpo(H_v.mpo, sites_eh, 1)
end


"""
    build_interaction_op_exciton(L, sites, Ufunc) -> MPO

Build the electron-hole interaction MPO on the 2L-site interleaved space.
`Ufunc(x)` gives the interaction strength when electron and hole are both
at position `x` (contact interaction). `sites` must be the 2L-site interleaved
site index vector.
"""
function build_interaction_op_exciton(L, sites, Ufunc)
    evals = range(1, 2^L, length=2^L)
    hvals = range(1, 2^L, length=2^L)

    o(x, y) = x == y ? Ufunc(x) : 0

    qtt     = quanticscrossinterpolate(Float64, o, [evals, hvals]; tolerance=1e-8)[1]
    tt      = TCI.tensortrain(qtt.tci)
    int_mps = MPS(tt)

    return -mps_to_diagonal_mpo(int_mps, sites)
end

