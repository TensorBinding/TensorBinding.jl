# bilayer_tk.jl — Bilayer / multilayer tight-binding Hamiltonians via MPO
#
# Constructs the interlayer coupling exactly (without TCI) for lattice-
# commensurate stackings by expressing the coupling as products of shift
# operators (generate_kin_u/d) and sublattice mask MPOs (get_diagonal_mpo).
# For general (e.g. twisted) interlayer potentials use twisted_tk.jl.
#
# Site encoding identical to twisted_tk.jl:
#   Site 1      : Layer index (dim = n_layers)
#   Sites 2…L+1 : L position qubits (quantics binary, row-major)
#
# With sublattice=true the site order is:
#   Site 1        : Layer index (dim = n_layers)
#   Sites 2...L+1 : L unit-cell position qubits
#   Site L+2      : Sublattice index
#
# Depends on: utils.jl, Hamiltonian.jl, 2D_lattice.jl, twisted_tk.jl


# ─────────────────────────────────────────────────────────────────
# 1.  Exact interlayer coupling builders
# ─────────────────────────────────────────────────────────────────

"""
    _bernal_interlayer_mpo(L, sites; t_inter=1.0, cutoff=1e-8) -> MPO

Build the interlayer coupling MPO for Bernal (AB) stacking on a honeycomb
lattice.  In the quantics site ordering, sublattice-A sites have 1-based odd
indices and sublattice-B sites have 1-based even indices.

Bernal stacking places each A site in layer 1 directly above a B site in
layer 2 (the B site of the same unit cell, index A+1).  The interlayer
operator in position space is therefore

    V = t_inter · (K_u D_A + D_A K_d)

where D_A is the A-sublattice projector and K_u/K_d are the ±1 shift
operators.  This is symmetric (Hermitian for real t_inter).
"""
function _bernal_interlayer_mpo(L::Int, sites;
                                 t_inter::Number = 1.0,
                                 cutoff::Real    = 1e-8)
    N   = 2^L
    D_A = get_diagonal_mpo(L, sites, x -> Float64(isodd(Int(x))))
    K_u = generate_kin_u(sites, N)
    K_d = generate_kin_d(sites, N)
    V   = +(t_inter * apply(K_u, D_A),
            conj(t_inter) * apply(D_A, K_d); cutoff=cutoff)
    ITensorMPS.truncate!(V; cutoff=cutoff)
    return V
end


"""
    _aa_interlayer_mpo(sites; t_inter=1.0) -> MPO

Build the interlayer coupling MPO for AA stacking: each site in layer 1
couples on-site to the same site in layer 2.  The coupling operator is
simply `t_inter * Identity`.
"""
_aa_interlayer_mpo(sites; t_inter::Number = 1.0) = t_inter * MPO(sites, "Id")

function _explicit_sublattice_monolayer(lattice::Symbol, Lx::Int, Ly::Int,
                                        pos_sites, sub_s;
                                        t::Number = 1.0,
                                        cutoff::Real = 1e-8,
                                        maxdim::Int = 200)
    lattice === :honeycomb ||
        error("sublattice=true is currently implemented for :honeycomb multilayers only.")
    H_mono = honeycomb_sublattice_hamiltonian(Lx, Ly, t; cutoff=cutoff, maxdim=maxdim)
    fix_sites(H_mono.mpo, [pos_sites; sub_s])
    rs = honeycomb_sublattice_positions(Lx, Ly)
    geom = let m = rs
        i -> m[i, :]
    end
    Nx_uc = 2^Lx
    sq3_2 = sqrt(3) / 2
    geom_uc = let Nx = Nx_uc, sq3_2 = sq3_2
        i -> begin
            n_cell = (i - 1) ÷ 2
            ix = n_cell % Nx
            iy = n_cell ÷ Nx
            [ix + iy * 0.5, iy * sq3_2]
        end
    end
    return H_mono.mpo, geom, geom_uc
end

function _explicit_sublattice_interlayer_pair(lattice::Symbol, stacking::Symbol,
                                              pos_sites, sub_s, layer_s,
                                              k::Int, l::Int;
                                              t_inter::Number = 1.0,
                                              cutoff::Real = 1e-8)
    Id_pos = MPO(pos_sites, "Id")
    n_sub  = dim(sub_s)

    V_fwd, V_bwd = if stacking === :AA
        I_sub = Matrix{ComplexF64}(I, n_sub, n_sub)
        (postpend_op(Id_pos, sub_s, t_inter * I_sub),
         postpend_op(Id_pos, sub_s, conj(t_inter) * I_sub))

    elseif stacking === :Bernal
        lattice === :honeycomb ||
            error(":Bernal stacking is defined only for :honeycomb lattice; got :$lattice.")
        n_sub == 2 ||
            error(":Bernal stacking with sublattice=true requires a 2-component honeycomb sublattice.")
        lower_A_to_upper_B = zeros(ComplexF64, n_sub, n_sub)
        upper_B_to_lower_A = zeros(ComplexF64, n_sub, n_sub)
        lower_A_to_upper_B[2, 1] = t_inter
        upper_B_to_lower_A[1, 2] = conj(t_inter)
        (postpend_op(Id_pos, sub_s, lower_A_to_upper_B),
         postpend_op(Id_pos, sub_s, upper_B_to_lower_A))

    else
        error("Unknown stacking :$stacking.  Supported: :AA, :Bernal.")
    end

    return +(prepend_layer_hopping(V_fwd, layer_s, l, k),
             prepend_layer_hopping(V_bwd, layer_s, k, l); cutoff=cutoff)
end


"""
    interlayer_mpo(lattice, stacking, Lx, Ly, sites;
                   t_inter=1.0, cutoff=1e-8) -> MPO

Build the position-space interlayer coupling MPO for the given `stacking`.
The returned operator V satisfies

    H_inter = t_inter · (|k⟩⟨l| ⊗ V + |l⟩⟨k| ⊗ V)

for each pair of adjacent layers k, l.

**Supported stackings**
- `:AA`     — on-site (identity in position space); any lattice
- `:Bernal` — A₁↔B₂ coupling within each unit cell; `:honeycomb` only

For general (non-commensurate) interlayer functions, pass a function
`f(i,j)` to `hopping2MPO` directly and use `prepend_layer_hopping`.
"""
function interlayer_mpo(lattice::Symbol, stacking::Symbol,
                        Lx::Int, Ly::Int, sites;
                        t_inter::Number = 1.0,
                        cutoff::Real    = 1e-8)
    L = Lx + Ly

    if stacking === :AA
        return _aa_interlayer_mpo(sites; t_inter=t_inter)

    elseif stacking === :Bernal
        lattice === :honeycomb ||
            error(":Bernal stacking is defined only for :honeycomb lattice; " *
                  "got :$lattice.")
        return _bernal_interlayer_mpo(L, sites; t_inter=t_inter, cutoff=cutoff)

    else
        error("Unknown stacking :$stacking.  Supported: :AA, :Bernal.\n" *
              "For custom stackings supply a function f(i,j) to hopping2MPO.")
    end
end


# ─────────────────────────────────────────────────────────────────
# 2.  Bilayer Hamiltonian
# ─────────────────────────────────────────────────────────────────

"""
    bilayer_hamiltonian(lattice, Lx, Ly;
        stacking=:AA, t_intra=1.0, t_inter=0.3,
        cutoff=1e-8, maxdim=200) -> (MPO, Vector{<:Index})

Build a bilayer tight-binding Hamiltonian as an MPO.

**Site encoding** (`L+1` sites total, `L = Lx + Ly`):
  - Site 1      : layer index (dim = 2)
  - Sites 2…L+1 : `L` position qubits (quantics binary, row-major)

**Arguments**
- `lattice`  : `:square`, `:triangular`, or `:honeycomb`
- `Lx`, `Ly` : each layer has `2^Lx × 2^Ly` sites

**Keyword arguments**
- `stacking` : `:AA` (on-site) or `:Bernal` (A₁↔B₂, honeycomb only)
- `t_intra`  : intra-layer NN hopping amplitude
- `t_inter`  : interlayer hopping amplitude
- `cutoff`   : MPO truncation cutoff
- `maxdim`   : maximum bond dimension of the final MPO

The assembled Hamiltonian is

    H = Σₖ Pₖ ⊗ H_mono  +  (|1⟩⟨2| + |2⟩⟨1|) ⊗ V

where V is the exact interlayer MPO for the chosen stacking.

Returns `(H_total, ext_sites)` where `ext_sites[1]` is the layer index
and `ext_sites[2:end]` are the `L` position qubits.
"""
function bilayer_hamiltonian(
    lattice::Symbol, Lx::Int, Ly::Int;
    stacking::Symbol = :AA,
    t_intra::Number  = 1.0,
    t_inter::Number  = 0.3,
    sublattice::Bool = false,
    cutoff::Real     = 1e-8,
    maxdim::Int      = 200,
)
    L = Lx + Ly

    if sublattice
        layer_s   = siteinds("Qubit", 1)[1]
        pos_sites = siteinds("Qubit", L)
        sub_s     = Index(2, "Honeycomb")
        ext_sites = [layer_s; pos_sites; sub_s]

        H_mono, geom, geom_uc =
            _explicit_sublattice_monolayer(lattice, Lx, Ly, pos_sites, sub_s;
                                           t=t_intra, cutoff=cutoff, maxdim=maxdim)
        H_intra = +(prepend_layer_projector(H_mono, layer_s, 1),
                    prepend_layer_projector(H_mono, layer_s, 2); cutoff=cutoff)

        H_inter = _explicit_sublattice_interlayer_pair(lattice, stacking,
                                                       pos_sites, sub_s, layer_s,
                                                       1, 2;
                                                       t_inter=t_inter,
                                                       cutoff=cutoff)

        H_total = +(H_intra, H_inter; cutoff=cutoff)
        ITensorMPS.truncate!(H_total; maxdim=maxdim, cutoff=cutoff)
        return TBHamiltonian(L, 2^L, ext_sites, H_total, geom, geom_uc,
                             0.0, 0.0, nothing, nothing, layer_s, sub_s, :pre,
                             nothing, nothing, 0, nothing)
    end

    # Layer encoded as a Qubit site (dim=2) so the full ext_sites vector
    # is all-Qubit — required for KPM_Tn / MPO(sites, "Id") to work.
    layer_s   = siteinds("Qubit", 1)[1]
    pos_sites = siteinds("Qubit", L)
    ext_sites = [layer_s; pos_sites]

    # Intralayer: P₁ ⊗ H_mono + P₂ ⊗ H_mono
    H_mono  = monolayer_hamiltonian(lattice, Lx, Ly, pos_sites;
                                    t=t_intra, cutoff=cutoff)
    H_intra = +(prepend_layer_projector(H_mono, layer_s, 1),
                prepend_layer_projector(H_mono, layer_s, 2); cutoff=cutoff)

    # Interlayer: (|1⟩⟨2| + |2⟩⟨1|) ⊗ V  (V built exactly, no TCI)
    V = interlayer_mpo(lattice, stacking, Lx, Ly, pos_sites;
                       t_inter=t_inter, cutoff=cutoff)
    H_inter = +(prepend_layer_hopping(V, layer_s, 1, 2),
                prepend_layer_hopping(V, layer_s, 2, 1); cutoff=cutoff)

    H_total = +(H_intra, H_inter; cutoff=cutoff)
    ITensorMPS.truncate!(H_total; maxdim=maxdim, cutoff=cutoff)
    # scale=0.0 → lazy DMRG estimation on first KPM_Tn call
    return TBHamiltonian(L, 2^L, ext_sites, H_total, nothing, 0.0, 0.0,
                         nothing, nothing, layer_s, nothing, 0, nothing)
end


# ─────────────────────────────────────────────────────────────────
# 3.  Multilayer Hamiltonian (nearest-neighbour layers only)
# ─────────────────────────────────────────────────────────────────

"""
    multilayer_hamiltonian(lattice, Lx, Ly, n_layers;
        stacking=:AA, t_intra=1.0, t_inter=0.3,
        cutoff=1e-8, maxdim=200) -> (MPO, Vector{<:Index})

Generalisation of `bilayer_hamiltonian` to `n_layers` layers.
The same `stacking` and `t_inter` are used for every adjacent pair.

Returns `(H_total, ext_sites)` with the same site encoding as
`bilayer_hamiltonian`, extended to a `dim = n_layers` layer index.
"""
function multilayer_hamiltonian(
    lattice::Symbol, Lx::Int, Ly::Int, n_layers::Int;
    stacking::Symbol = :AA,
    t_intra::Number  = 1.0,
    t_inter::Number  = 0.3,
    sublattice::Bool = false,
    cutoff::Real     = 1e-8,
    maxdim::Int      = 200,
)
    n_layers ≥ 2 || error("Need at least 2 layers; got $n_layers.")
    L = Lx + Ly

    if sublattice
        layer_s   = Index(n_layers, "Layer")
        pos_sites = siteinds("Qubit", L)
        sub_s     = Index(2, "Honeycomb")
        ext_sites = [layer_s; pos_sites; sub_s]

        H_mono, geom, geom_uc =
            _explicit_sublattice_monolayer(lattice, Lx, Ly, pos_sites, sub_s;
                                           t=t_intra, cutoff=cutoff, maxdim=maxdim)
        H_intra = prepend_layer_projector(H_mono, layer_s, 1)
        for k in 2:n_layers
            H_intra = +(H_intra,
                        prepend_layer_projector(H_mono, layer_s, k); cutoff=cutoff)
        end

        H_inter = nothing
        for k in 1:(n_layers - 1)
            term = _explicit_sublattice_interlayer_pair(lattice, stacking,
                                                        pos_sites, sub_s, layer_s,
                                                        k, k + 1;
                                                        t_inter=t_inter,
                                                        cutoff=cutoff)
            H_inter = H_inter === nothing ? term : +(H_inter, term; cutoff=cutoff)
        end

        H_total = +(H_intra, H_inter; cutoff=cutoff)
        ITensorMPS.truncate!(H_total; maxdim=maxdim, cutoff=cutoff)
        return TBHamiltonian(L, 2^L, ext_sites, H_total, geom, geom_uc,
                             0.0, 0.0, nothing, nothing, layer_s, sub_s, :pre,
                             nothing, nothing, 0, nothing)
    end

    layer_s   = Index(n_layers, "Layer")
    pos_sites = siteinds("Qubit", L)
    ext_sites = [layer_s; pos_sites]

    # Intralayer
    H_mono  = monolayer_hamiltonian(lattice, Lx, Ly, pos_sites;
                                    t=t_intra, cutoff=cutoff)
    H_intra = prepend_layer_projector(H_mono, layer_s, 1)
    for k in 2:n_layers
        H_intra = +(H_intra,
                    prepend_layer_projector(H_mono, layer_s, k); cutoff=cutoff)
    end

    # Interlayer: only adjacent layers k ↔ k+1
    V = interlayer_mpo(lattice, stacking, Lx, Ly, pos_sites;
                       t_inter=t_inter, cutoff=cutoff)
    H_inter = nothing
    for k in 1:(n_layers - 1)
        term = +(prepend_layer_hopping(V, layer_s, k,   k+1),
                 prepend_layer_hopping(V, layer_s, k+1, k  ); cutoff=cutoff)
        H_inter = H_inter === nothing ? term : +(H_inter, term; cutoff=cutoff)
    end

    H_total = +(H_intra, H_inter; cutoff=cutoff)
    ITensorMPS.truncate!(H_total; maxdim=maxdim, cutoff=cutoff)
    # scale=0.0 → lazy DMRG estimation on first KPM_Tn call
    return TBHamiltonian(L, 2^L, ext_sites, H_total, nothing, 0.0, 0.0,
                         nothing, nothing, layer_s, nothing, 0, nothing)
end
