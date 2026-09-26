# Supercond_tk.jl — Pairing MPO builders and spin/BdG assemblers for MPO Hamiltonians
#
# The spin/Nambu indices, operator tables and prepend/postpend helpers that
# were sections 0–2 of this file live in core/AuxDOF.jl.
#
# Follows the same prepend-core pattern as twisted_tk.jl: an auxiliary site
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

# ─────────────────────────────────────────────────────────────────
# 3.  Antisymmetric pairing MPO builders
# ─────────────────────────────────────────────────────────────────

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
                initial_positions=[], unfoldingscheme=:interleaved) -> MPO

Antisymmetric analogue of `hopping2MPO` for superconducting pairing.

Compresses a general N×N pairing matrix `Δ[i,j] = f(i,j)` into an MPO via
Quantics TCI.  `f` is passed directly to TCI — the caller is responsible for
providing an antisymmetric function (`f(i,j) = −f(j,i)`).

For nearest-neighbour pairing prefer `pairingNNN` — it is exact and avoids TCI.
Use `pairing2MPO` for longer-range or d-wave / p±ip patterns.

`H_pair` is passed directly to `bdg_hamiltonian` or `bdg_spin_hamiltonian`.
"""
function pairing2MPO(f, N, sites; tol=1e-8, initial_positions=[],
                     type=ComplexF64, unfoldingscheme=:interleaved)
    return hopping2MPO(f, N, sites; tol=tol, initial_positions=initial_positions,
                       type=type, unfoldingscheme=unfoldingscheme)
end


# ─────────────────────────────────────────────────────────────────
# 4.  Higher-level assemblers
# ─────────────────────────────────────────────────────────────────

"""
    spin_hamiltonian(H_up, H_down, spin_s;
                     H_Zeeman=nothing, cutoff=1e-8) -> MPO

Build a spin-resolved Hamiltonian on `[spin_s; pos_sites…]`:

    H = P_↑ ⊗ H_up  +  P_↓ ⊗ H_down  [+  S_z ⊗ H_Zeeman]

`H_up`, `H_down` are MPOs on the same position sites (they may differ for
spin-orbit coupling or magnetic exchange).  `H_Zeeman` is an optional
position-MPO encoding a local magnetic field `h(x)`; it enters as
`S_z ⊗ H_Zeeman` so spin-↑ gains `+½ h(x)` and spin-↓ gains `−½ h(x)`.
"""
function spin_hamiltonian(H_up::MPO, H_down::MPO, spin_s::Index;
                          H_Zeeman::Union{MPO, Nothing} = nothing,
                          cutoff::Real = 1e-8)
    H = +(prepend_spin(H_up,   spin_s, :Pup),
          prepend_spin(H_down, spin_s, :Pdn); cutoff=cutoff)
    isnothing(H_Zeeman) || (H = +(H, prepend_spin(H_Zeeman, spin_s, :Sz); cutoff=cutoff))
    return H
end


"""
    bdg_hamiltonian(H_kin, H_pair, nambu_s; cutoff=1e-8) -> MPO

Build a **spinless** Bogoliubov–de Gennes Hamiltonian on `[nambu_s; pos_sites…]`:

    H_BdG = τ_z ⊗ H_kin  +  τ_+ ⊗ H_pair  +  τ_- ⊗ dag(H_pair)

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
"""
function bdg_hamiltonian(H_kin::MPO, H_pair::MPO, nambu_s::Index;
                         cutoff::Real = 1e-8)
    H_pair_adj = swapprime(dag(H_pair), 0, 1)
    return +(+(prepend_nambu(H_kin,         nambu_s, :tz),
               prepend_nambu(H_pair,        nambu_s, :tp); cutoff=cutoff),
               prepend_nambu(H_pair_adj,    nambu_s, :tm); cutoff=cutoff)
end


"""
    bdg_spin_hamiltonian(H_kin_up, H_kin_down, H_pair, spin_s, nambu_s;
                         H_soc=nothing, cutoff=1e-8) -> MPO

Build a **spin-½ singlet** BdG Hamiltonian on `[nambu_s, spin_s; pos_sites…]`.

**Nambu–spin convention**: `Ψ = (c_↑, c_↓, c†_↓, −c†_↑)ᵀ` (standard BCS).

    H_BdG = τ_z⊗P_↑ ⊗ H_kin_up  +  τ_z⊗P_↓ ⊗ H_kin_down
          + τ_+⊗(i·σ_y) ⊗ H_pair  +  τ_-⊗(−i·σ_y) ⊗ H_pair†
          [+ τ_z⊗S_z ⊗ H_soc]

The first two lines are the kinetic energy (allowing spin-dependent fields,
e.g. Zeeman: pass `H_kin_up = H_tb − (μ+h)·I`, `H_kin_down = H_tb − (μ−h)·I`).

The pairing lines implement singlet Cooper-pair creation via the antisymmetric
spin factor `i·σ_y = [[0,1],[−1,0]]`.  For on-site s-wave pairing, build
`H_pair` as a diagonal MPO with `Δ` on the diagonal.

`H_soc` (optional) adds an Ising-type spin-orbit coupling `τ_z⊗S_z⊗H_soc`.

The result is Hermitian for real or complex `H_pair` and any `H_kin_up/dn`.
"""
function bdg_spin_hamiltonian(
    H_kin_up::MPO, H_kin_down::MPO, H_pair::MPO,
    spin_s::Index, nambu_s::Index;
    H_soc::Union{MPO, Nothing} = nothing,
    cutoff::Real = 1e-8,
)
    # τ_z ⊗ P_↑ ⊗ H_kin_up  and  τ_z ⊗ P_↓ ⊗ H_kin_down
    H = +(prepend_nambu(prepend_spin(H_kin_up,   spin_s, :Pup), nambu_s, :tz),
          prepend_nambu(prepend_spin(H_kin_down, spin_s, :Pdn), nambu_s, :tz); cutoff=cutoff)

    # τ_+ ⊗ (i·σ_y) ⊗ Δ  +  τ_- ⊗ (−i·σ_y) ⊗ Δ†
    # i·σ_y = [[0,1],[-1,0]] (real matrix): P_↑ pairs with h-↓, P_↓ pairs with h-↑ (singlet)
    H_pair_adj = swapprime(dag(H_pair), 0, 1)
    H_tp = prepend_nambu(prepend_spin(H_pair,       spin_s, :iSy),  nambu_s, :tp)
    H_tm = prepend_nambu(prepend_spin(H_pair_adj,   spin_s, :miSy), nambu_s, :tm)
    H    = +(+(H, H_tp; cutoff=cutoff), H_tm; cutoff=cutoff)

    # Optional Ising SOC: τ_z ⊗ S_z ⊗ H_soc
    if !isnothing(H_soc)
        H = +(H, prepend_nambu(prepend_spin(H_soc, spin_s, :Sz), nambu_s, :tz); cutoff=cutoff)
    end
    return H
end
