# Supercond.jl — Pairing MPO builders and spin/BdG assemblers for MPO Hamiltonians
#
# The spin/Nambu indices, operator tables and prepend/postpend helpers that
# used to open this file live in core/AuxDOF.jl.
#
# Entry points: pairingNNN, pairing2MPO, spin_hamiltonian, bdg_hamiltonian,
#   bdg_spin_hamiltonian.
# Depends on: core/Utils.jl (the "sigma_plus"/"sigma_minus" ops), core/MPOTools.jl,
#   core/Hamiltonian.jl, core/AuxDOF.jl.
#
# Follows the same prepend-core pattern as lattice/Twisted.jl: an auxiliary site
# (spin or particle/hole) is prepended to a position-qubit MPO, extending
# it by one site.  Multiple prepends can be chained:
#
#   [nambu_s, spin_s, pos_qubits...]   ← BdG with spin (call prepend_spin first)
#   [spin_s,  pos_qubits...]           ← spin-resolved tight-binding
#   [nambu_s, pos_qubits...]           ← spinless BdG
#
# Operator convention (both spin and Nambu use 2-state 1-indexed basis):
#   spin:  state 1 = ↑,        state 2 = ↓
#   Nambu: state 1 = particle,  state 2 = hole

# ============================================================
# 1. Antisymmetric pairing MPO builders
# ============================================================

"""
    pairingNNN(L, sites, hopping, nn; apply_kwargs=NamedTuple()) -> MPO

Antisymmetric analogue of `kineticNNN` for superconducting pairing.

Returns

    H_pair = hopping · Kf^nn  −  Kb^nn · dag(hopping)

where `Kf` / `Kb` are the quantics forward / backward unit-shift operators.
The sign flip relative to `kineticNNN` (`+` → `−`) produces an antisymmetric
matrix `Δ(i,j) = −Δ(j,i)`, satisfying the Fermi constraint `c_i†c_j† = −c_j†c_i†`.

For uniform pairing `hopping = Δ·I`:

    H_pair = Δ · (Kf^nn − Kb^nn)

`H_pair` is passed directly to `bdg_hamiltonian` or `bdg_spin_hamiltonian`;
the `τ_- ⊗ H_pair†` (hole-particle) term is built automatically there.
"""
function pairingNNN(L, sites, hopping::MPO, nn::Integer; apply_kwargs = NamedTuple())
    @assert L == length(sites) "L must equal length(sites)"
    @assert nn ≥ 1             "nn must be ≥ 1"
    kinetic_1 = OpSum()
    kinetic_2 = OpSum()
    for i in 1:L
        os = OpSum()
        os += 1, "sigma_plus", L - (i-1)
        for j in 1:L-i;     os *= ("Id",          j); end
        for j in (L+2-i):L; os *= ("sigma_minus",  j); end
        kinetic_1 += os
    end
    for i in 1:L
        os = OpSum()
        os += 1, "sigma_minus", L - (i-1)
        for j in 1:L-i;     os *= ("Id",          j); end
        for j in (L+2-i):L; os *= ("sigma_plus",  j); end
        kinetic_2 += os
    end
    k1 = MPO(kinetic_1, sites)
    k2 = MPO(kinetic_2, sites)
    An = compose_power(k1, nn; side=:right, apply_kwargs)
    Am = compose_power(k2, nn; side=:left,  apply_kwargs)
    return +(apply(hopping, An; apply_kwargs...),
             -1.0 * apply(Am, dag(hopping); apply_kwargs...); cutoff=1e-12)
end


"""
    pairing2MPO(f, N, sites; tol=1e-8, type=ComplexF64,
                initial_positions=[], unfoldingscheme=:interleaved, check=false) -> MPO

Antisymmetric analogue of `hopping2MPO` for superconducting pairing.

Compresses a general N×N pairing matrix `Δ[i,j] = f(i,j)` into an MPO via
Quantics TCI.  `f` is passed directly to TCI — the caller is responsible for
providing an antisymmetric function (`f(i,j) = −f(j,i)`).

For nearest-neighbour pairing prefer `pairingNNN` — it is exact and avoids TCI.
Use `pairing2MPO` for longer-range or d-wave / p±ip patterns.

`H_pair` is passed directly to `bdg_hamiltonian` or `bdg_spin_hamiltonian`.
`check=true` runs the sampled self-check of `hopping2MPO` (a sparse `f`, such as a
short-range pairing, can make the default QTCI pivots miss whole bond classes);
`add_superconductivity!(H, Δ; type=:custom)` turns it on.
"""
function pairing2MPO(f, N, sites; tol=1e-8, initial_positions=[],
                     type=ComplexF64, unfoldingscheme=:interleaved, check::Bool=false)
    return hopping2MPO(f, N, sites; tol=tol, initial_positions=initial_positions,
                       type=type, unfoldingscheme=unfoldingscheme, check=check)
end


# ============================================================
# 2. Higher-level assemblers
# ============================================================

"""
    spin_hamiltonian(H_up, H_down, spin_s;
                     H_Zeeman=nothing, cutoff=nothing) -> MPO

Build a spin-resolved Hamiltonian on `[spin_s; pos_sites…]`:

    H = P_↑ ⊗ H_up  +  P_↓ ⊗ H_down  [+  S_z ⊗ H_Zeeman]

`H_up`, `H_down` are MPOs on the same position sites (they may differ for
spin-orbit coupling or magnetic exchange).  `H_Zeeman` is an optional
position-MPO encoding a local magnetic field `h(x)`; it enters as
`S_z ⊗ H_Zeeman` so spin-↑ gains `+½ h(x)` and spin-↓ gains `−½ h(x)`.

`cutoff` is the relative cutoff of the sums, by default `min(1e-8, 0.1/D)`, D the dimension
of the spinful site space (`1e-8` up to D = 2^23; see "Large systems" in
[`get_Hamiltonian`](@ref)).
"""
function spin_hamiltonian(H_up::MPO, H_down::MPO, spin_s::Index;
                          H_Zeeman::Union{MPO, Nothing} = nothing,
                          cutoff::Union{Nothing,Real} = nothing)
    cutoff = _mpo_cutoff(cutoff, dim(spin_s) * _site_space_dim(_mpo_ket_sites(H_up)),
                         "spin_hamiltonian"; keyword="cutoff")
    H = +(prepend_spin(H_up,   spin_s, :Pup),
          prepend_spin(H_down, spin_s, :Pdn); cutoff=cutoff)
    isnothing(H_Zeeman) || (H = +(H, prepend_spin(H_Zeeman, spin_s, :Sz); cutoff=cutoff))
    return H
end


"""
    bdg_hamiltonian(H_kin, H_pair, nambu_s; cutoff=nothing) -> MPO

Build a **spinless** Bogoliubov–de Gennes Hamiltonian on `[nambu_s; pos_sites…]`, in
the Nambu basis `(c, c†)`:

    H_BdG = P_p ⊗ H_kin  +  P_h ⊗ (−H_kin*)  +  τ_+ ⊗ H_pair  +  τ_- ⊗ H_pair†

For a real `H_kin` the first two terms are `τ_z ⊗ H_kin`, built as one term; up to
v0.2.0 every `H_kin` got `τ_z ⊗ H_kin`, whose hole block `−H_kin` is wrong when `H_kin`
has imaginary entries (Peierls phases, complex hoppings).

`H_kin` is the single-particle kinetic/hopping MPO measured from the chemical
potential (`H_kin = H_tb − μ·I`).  `H_pair` encodes the pairing amplitude
`Δ(i,j)`.

**Note**: for spinless fermions `c_i† c_j† = −c_j† c_i†`, so the pairing matrix
must satisfy `Δ(i,j) = −Δ(j,i)`.  On-site (s-wave) pairing is therefore
**forbidden**; the minimal allowed symmetry is **p-wave** (nearest-neighbour,
antisymmetric).  For the Kitaev chain with uniform p-wave amplitude `Δ` on the
`L` position sites `sites`:
```julia
H_pair = pairingNNN(L, sites, Δ * MPO(sites, "Id"), 1)   # Δ(i,i+1) = +Δ, Δ(i+1,i) = −Δ
```
`H_pair` must hold both entries of each antisymmetric pair itself; the h.c. term
`τ_- ⊗ H_pair†` is built automatically.  For a general `Δ(i,j)` use `pairing2MPO`.

The result is Hermitian for any `H_kin = H_kin†` and any complex `H_pair`.

`cutoff` is the relative cutoff of the sums, by default `min(1e-8, 0.1/D)`, D the dimension
of the BdG site space (`1e-8` up to D = 2^23; see "Large systems" in
[`get_Hamiltonian`](@ref)).
"""
function bdg_hamiltonian(H_kin::MPO, H_pair::MPO, nambu_s::Index;
                         cutoff::Union{Nothing,Real} = nothing)
    cutoff = _mpo_cutoff(cutoff, dim(nambu_s) * _site_space_dim(_mpo_ket_sites(H_kin)),
                         "bdg_hamiltonian"; keyword="cutoff")
    H_pair_adj = swapprime(dag(H_pair), 0, 1)
    return +(+(_nambu_normal_block(H_kin, nambu_s, :pre; cutoff=cutoff),
               prepend_nambu(H_pair,        nambu_s, :tp); cutoff=cutoff),
               prepend_nambu(H_pair_adj,    nambu_s, :tm); cutoff=cutoff)
end


"""
    bdg_spin_hamiltonian(H_kin_up, H_kin_down, H_pair, spin_s, nambu_s;
                         H_soc=nothing, cutoff=nothing) -> MPO

Build a **spin-½ singlet** BdG Hamiltonian on `[nambu_s, spin_s; pos_sites…]`.

**Nambu–spin convention**: `Ψ = (c_↑, c_↓, c†_↑, c†_↓)ᵀ`, the basis in which the
singlet pairing is `i·σ_y ⊗ Δ` (up to v0.2.0 this docstring named the basis
`(c_↑, c_↓, c†_↓, −c†_↑)`, in which the singlet pairing would be `1 ⊗ Δ` instead).

    H_BdG = N(P_↑ ⊗ H_kin_up)  +  N(P_↓ ⊗ H_kin_down)
          + τ_+⊗(i·σ_y) ⊗ H_pair  +  τ_-⊗(−i·σ_y) ⊗ H_pair†
          [+ N(S_z ⊗ H_soc)]

with `N(M) = P_p ⊗ M + P_h ⊗ (−M*)`, the particle and hole blocks. For a real `M`,
`N(M) = τ_z ⊗ M`, built as one term; up to v0.2.0 every `M` got `τ_z ⊗ M`, whose hole
block is wrong when `M` has imaginary entries.

The first two lines are the kinetic energy (allowing spin-dependent fields,
e.g. Zeeman: pass `H_kin_up = H_tb − (μ+h)·I`, `H_kin_down = H_tb − (μ−h)·I`).

The pairing lines implement singlet Cooper-pair creation via the antisymmetric
spin factor `i·σ_y = [[0,1],[−1,0]]`.  For on-site s-wave pairing, build
`H_pair` as a diagonal MPO with `Δ` on the diagonal.

`H_soc` (optional) adds an Ising-type spin-orbit coupling `N(S_z ⊗ H_soc)`.

The result is Hermitian for real or complex `H_pair` and any `H_kin_up/dn`.

`cutoff` is the relative cutoff of the sums, by default `min(1e-8, 0.1/D)`, D the dimension
of the spinful BdG site space (`1e-8` up to D = 2^23; see "Large systems" in
[`get_Hamiltonian`](@ref)).
"""
function bdg_spin_hamiltonian(
    H_kin_up::MPO, H_kin_down::MPO, H_pair::MPO,
    spin_s::Index, nambu_s::Index;
    H_soc::Union{MPO, Nothing} = nothing,
    cutoff::Union{Nothing,Real} = nothing,
)
    cutoff = _mpo_cutoff(cutoff,
                         dim(nambu_s) * dim(spin_s) * _site_space_dim(_mpo_ket_sites(H_kin_up)),
                         "bdg_spin_hamiltonian"; keyword="cutoff")
    # N(P_↑ ⊗ H_kin_up) and N(P_↓ ⊗ H_kin_down): τ_z ⊗ … for real terms
    H = +(_nambu_normal_block(prepend_spin(H_kin_up,   spin_s, :Pup), nambu_s, :pre; cutoff=cutoff),
          _nambu_normal_block(prepend_spin(H_kin_down, spin_s, :Pdn), nambu_s, :pre; cutoff=cutoff);
          cutoff=cutoff)

    # τ_+ ⊗ (i·σ_y) ⊗ Δ  +  τ_- ⊗ (−i·σ_y) ⊗ Δ†
    # i·σ_y = [[0,1],[-1,0]] (real matrix): P_↑ pairs with h-↓, P_↓ pairs with h-↑ (singlet)
    H_pair_adj = swapprime(dag(H_pair), 0, 1)
    H_tp = prepend_nambu(prepend_spin(H_pair,       spin_s, :iSy),  nambu_s, :tp)
    H_tm = prepend_nambu(prepend_spin(H_pair_adj,   spin_s, :miSy), nambu_s, :tm)
    H    = +(+(H, H_tp; cutoff=cutoff), H_tm; cutoff=cutoff)

    # Optional Ising SOC: N(S_z ⊗ H_soc)
    if !isnothing(H_soc)
        H = +(H, _nambu_normal_block(prepend_spin(H_soc, spin_s, :Sz), nambu_s, :pre; cutoff=cutoff);
              cutoff=cutoff)
    end
    return H
end
