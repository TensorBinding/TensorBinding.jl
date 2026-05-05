# Supercond_tk.jl — Spin and Nambu auxiliary degrees of freedom for MPO Hamiltonians
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
# 0.  Symbol dispatch for prepend_op (defined in utils.jl)
# ─────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────
# 1.  Spin-½ site index and operators
# ─────────────────────────────────────────────────────────────────

"""
    spin_index() -> Index

Create a dim-2 Index tagged "Spin" (state 1 = ↑, state 2 = ↓).
Pass the result as `spin_s` to all `prepend_spin` calls.
"""
spin_index() = Index(2, "Spin")


# 2×2 spin-½ operator matrices, ComplexF64 throughout for uniformity.
# Basis: |↑⟩ = 1, |↓⟩ = 2.
const _SPIN_OPS = Dict{Symbol, Matrix{ComplexF64}}(
    :Id   => [1   0;  0   1],
    :Pup  => [1   0;  0   0],          # |↑⟩⟨↑|  — spin-up projector
    :Pdn  => [0   0;  0   1],          # |↓⟩⟨↓|  — spin-down projector
    :Sz   => [1/2 0;  0  -1/2],        # S_z = ½σ_z
    :Sp   => [0   1;  0   0],          # S_+ = |↑⟩⟨↓|  (spin-flip ↓→↑)
    :Sm   => [0   0;  1   0],          # S_- = |↓⟩⟨↑|  (spin-flip ↑→↓)
    :Sx   => [0   1/2; 1/2  0],        # S_x = ½σ_x
    :Sy   => [0  -1im/2; 1im/2  0],    # S_y = ½σ_y
    :iSy  => [0   1;  -1   0],         # i·σ_y  — singlet pairing spin factor
    :miSy => [0  -1;   1   0],         # (i·σ_y)† = −i·σ_y
)


"""
    prepend_op(H_mpo, s, op::Symbol) -> MPO
    postpend_op(H_mpo, s, op::Symbol) -> MPO

Symbol dispatch for indices tagged `"Spin"` or `"Nambu"`.
Looks up `op` in the appropriate operator dictionary and calls
the matrix-form `prepend_op` / `postpend_op`.

Spin ops (index tagged `"Spin"`):
`:Id`, `:Pup`, `:Pdn`, `:Sz`, `:Sp`, `:Sm`, `:Sx`, `:Sy`, `:iSy`, `:miSy`

Nambu ops (index tagged `"Nambu"`):
`:Id`, `:Pp`, `:Ph`, `:tz`, `:tx`, `:ty`, `:tp`, `:tm`
"""
function prepend_op(H_mpo::MPO, s::Index, op::Symbol)
    if hastags(s, "Spin")
        mat = get(_SPIN_OPS, op, nothing)
        isnothing(mat) && error("Unknown Spin op :$op.  Known: $(sort(collect(keys(_SPIN_OPS))))")
    elseif hastags(s, "Nambu")
        mat = get(_NAMBU_OPS, op, nothing)
        isnothing(mat) && error("Unknown Nambu op :$op.  Known: $(sort(collect(keys(_NAMBU_OPS))))")
    else
        error("Symbol-based prepend_op requires a \"Spin\" or \"Nambu\" tagged index; got tags: $(tags(s))")
    end
    return prepend_op(H_mpo, s, mat)
end

function postpend_op(H_mpo::MPO, s::Index, op::Symbol)
    if hastags(s, "Spin")
        mat = get(_SPIN_OPS, op, nothing)
        isnothing(mat) && error("Unknown Spin op :$op.  Known: $(sort(collect(keys(_SPIN_OPS))))")
    elseif hastags(s, "Nambu")
        mat = get(_NAMBU_OPS, op, nothing)
        isnothing(mat) && error("Unknown Nambu op :$op.  Known: $(sort(collect(keys(_NAMBU_OPS))))")
    else
        error("Symbol-based postpend_op requires a \"Spin\" or \"Nambu\" tagged index; got tags: $(tags(s))")
    end
    return postpend_op(H_mpo, s, mat)
end


"""
    prepend_spin(H_mpo, spin_s, op) -> MPO

Prepend a spin-½ operator on `spin_s` (created with `spin_index()`).
`op` is a `Symbol` from the table below or an explicit 2×2 matrix.
Equivalent to `prepend_op(H_mpo, spin_s, op)`.

| Symbol  | Matrix                  | Typical use                      |
|---------|-------------------------|----------------------------------|
| `:Id`   | I₂                      | Spin-degenerate term             |
| `:Pup`  | diag(1,0)               | Spin-up projector                |
| `:Pdn`  | diag(0,1)               | Spin-down projector              |
| `:Sz`   | diag(½,−½)              | Zeeman / exchange field          |
| `:Sp`   | \\|↑⟩⟨↓\\|              | Spin-flip ↓→↑                   |
| `:Sm`   | \\|↓⟩⟨↑\\|              | Spin-flip ↑→↓                   |
| `:Sx`   | ½σ_x                    | In-plane exchange                |
| `:Sy`   | ½σ_y                    | In-plane exchange                |
| `:iSy`  | i·σ_y = [[0,1],[−1,0]] | Singlet pairing spin factor      |
| `:miSy` | −i·σ_y                 | h.c. of singlet pairing          |

Basis: state 1 = ↑, state 2 = ↓.
"""
prepend_spin(H::MPO, s::Index, op::Symbol)           = prepend_op(H, s, op)
prepend_spin(H::MPO, s::Index, mat::AbstractMatrix)  = prepend_op(H, s, mat)

"""
    postpend_spin(H_mpo, spin_s, op) -> MPO

Append a spin-½ operator on `spin_s` to the *end* of `H_mpo`.
`op` is a `Symbol` (same table as `prepend_spin`) or an explicit 2×2 matrix.
Equivalent to `postpend_op(H_mpo, spin_s, op)`.
"""
postpend_spin(H::MPO, s::Index, op::Symbol)          = postpend_op(H, s, op)
postpend_spin(H::MPO, s::Index, mat::AbstractMatrix) = postpend_op(H, s, mat)


# ─────────────────────────────────────────────────────────────────
# 2.  Nambu (particle–hole) site index and operators
# ─────────────────────────────────────────────────────────────────

"""
    nambu_index() -> Index

Create a dim-2 Index tagged "Nambu" (state 1 = particle, state 2 = hole).
Pass the result as `nambu_s` to all `prepend_nambu` calls.
"""
nambu_index() = Index(2, "Nambu")


# 2×2 Nambu operator matrices, ComplexF64 throughout.
# Basis: |particle⟩ = 1, |hole⟩ = 2.
const _NAMBU_OPS = Dict{Symbol, Matrix{ComplexF64}}(
    :Id => [1   0;  0   1],
    :Pp => [1   0;  0   0],            # |p⟩⟨p|  — particle-sector projector
    :Ph => [0   0;  0   1],            # |h⟩⟨h|  — hole-sector projector
    :tz => [1   0;  0  -1],            # τ_z  — kinetic sign in BdG
    :tx => [0   1;  1   0],            # τ_x  — real pairing (spinless p-wave)
    :ty => [0  -1im; 1im  0],          # τ_y  — imaginary / chiral pairing
    :tp => [0   1;  0   0],            # τ_+ = |p⟩⟨h|  — pairing Δ
    :tm => [0   0;  1   0],            # τ_- = |h⟩⟨p|  — pairing Δ† (h.c.)
)


"""
    prepend_nambu(H_mpo, nambu_s, op) -> MPO

Prepend a Nambu (particle–hole) operator on `nambu_s` (created with `nambu_index()`).
`op` is a `Symbol` from the table below or an explicit 2×2 matrix.
Equivalent to `prepend_op(H_mpo, nambu_s, op)`.

| Symbol | Matrix      | Typical use                    |
|--------|-------------|--------------------------------|
| `:Id`  | I₂          | Particle + hole                |
| `:Pp`  | diag(1,0)   | Particle-sector projector      |
| `:Ph`  | diag(0,1)   | Hole-sector projector          |
| `:tz`  | diag(1,−1)  | Kinetic τ_z in BdG             |
| `:tp`  | \\|p⟩⟨h\\|  | Pairing amplitude Δ            |
| `:tm`  | \\|h⟩⟨p\\|  | Pairing Δ† (h.c.)              |
| `:tx`  | σ_x         | Real pairing (spinless p-wave) |
| `:ty`  | σ_y         | Imaginary / chiral pairing     |

Basis: state 1 = particle, state 2 = hole.
"""
prepend_nambu(H::MPO, s::Index, op::Symbol)          = prepend_op(H, s, op)
prepend_nambu(H::MPO, s::Index, mat::AbstractMatrix) = prepend_op(H, s, mat)

"""
    postpend_nambu(H_mpo, nambu_s, op) -> MPO

Append a Nambu operator on `nambu_s` to the *end* of `H_mpo`.
`op` is a `Symbol` (same table as `prepend_nambu`) or an explicit 2×2 matrix.
Equivalent to `postpend_op(H_mpo, nambu_s, op)`.
"""
postpend_nambu(H::MPO, s::Index, op::Symbol)          = postpend_op(H, s, op)
postpend_nambu(H::MPO, s::Index, mat::AbstractMatrix) = postpend_op(H, s, mat)


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
antisymmetric).  For the Kitaev chain with uniform p-wave amplitude `Δ`:
```julia
H_pair = hopping2MPO(L, [(i, i+1, Δ) for i in 0:N-2], sites)   # Δ(i,i+1) = +Δ
```
The h.c. term `τ_- ⊗ H_pair†` (which carries `Δ(j,i) = −Δ`) is built automatically.

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
