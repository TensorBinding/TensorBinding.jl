# core/AuxDOF.jl — auxiliary degrees of freedom (spin, Nambu, layer, sublattice)
#
# Owns the machinery that adds, detects and projects out the auxiliary sites of
# a TBHamiltonian.  An auxiliary site (spin or particle/hole) is prepended (or
# postpended) to a position-qubit MPO, extending it by one site.  Multiple
# prepends can be chained:
#
#   [nambu_s, spin_s, pos_qubits...]   ← BdG with spin (call prepend_spin first)
#   [spin_s,  pos_qubits...]           ← spin-resolved tight-binding
#   [nambu_s, pos_qubits...]           ← spinless BdG
#
# Operator convention (both spin and Nambu use 2-state 1-indexed basis):
#   spin:  state 1 = ↑,        state 2 = ↓
#   Nambu: state 1 = particle,  state 2 = hole
#
# Main entry points: add_spin!, add_zeeman!, add_superconductivity!, add_soc!,
# spin_index, nambu_index, prepend_spin/postpend_spin,
# prepend_nambu/postpend_nambu, project_aux, aux_site, probe_state.
#
# Contents by section (sections 1–8 and 10 were moved in Tier 1 of the
# reorganisation from the files named on the right; Tier 2 added the shared
# kernels of sections 8–10):
#   1–2.  spin_index, _SPIN_OPS, Symbol prepend_op/postpend_op,
#         prepend_spin/postpend_spin, nambu_index, _NAMBU_OPS,
#         prepend_nambu/postpend_nambu            ← physics/Supercond.jl
#   3–6.  add_spin!, add_zeeman!, add_superconductivity!, add_soc!
#                                                 ← core/TBSystem.jl
#   7.    project_aux, _autoenable_proj, aux_site ← the former physics/QFT_tk.jl
#   8.    the sector projectors: _project_end_site (behind project_aux and
#         contract_nh_block), _block_projector and _absorb_aux_site (behind
#         _project_aux_block ← physics/SCF.jl and _project_spin_sector
#         ← physics/rpa/dyson.jl)
#   9.    AuxProjection, _aux_projection, _project_aux_sectors, _probe_sectors,
#         _aux_setup (the spectral methods' eight aux keywords, in one value)
#   10.   _ldos_make_psi0 (← solvers/kpm/ldos.jl), probe_state
#
# Depends on: Utils, Hamiltonian, TBSystem, hopping2d*, Supercond* (a * marks a
# file included later; see the source map in TensorBinding.jl).  Included right
# after core/TBSystem.jl, this file needs only the ITensors types and
# TBHamiltonian at definition time.  Its callees are resolved at run time: the
# matrix-form prepend_op/postpend_op, get_diagonal_mpo, _basis_state_mps and
# mpsexciton (core/Utils.jl), hopping2MPO (core/Hamiltonian.jl), _pos_sites,
# _invalidate_cache!, _require_binary_position_space and physical_site_state
# (core/TBSystem.jl), generate_kin_u/d (lattice/hopping2d.jl) and
# pairingNNN/pairing2MPO (physics/Supercond.jl).


# ============================================================
# 1. Spin-½ site index and operators; Symbol methods of prepend_op/postpend_op
# ============================================================

"""
    spin_index() -> Index

Create a dim-2 Index tagged "Spin" (state 1 = ↑, state 2 = ↓).
Pass the result as `s` to all `prepend_spin` / `postpend_spin` calls.
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
    prepend_spin(H, s, op) -> MPO

Prepend a spin-½ operator on the spin index `s` (created with `spin_index()`).
`op` is a `Symbol` from the table below or an explicit 2×2 matrix.
Equivalent to `prepend_op(H, s, op)`.

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
    postpend_spin(H, s, op) -> MPO

Append a spin-½ operator on the spin index `s` to the *end* of `H`.
`op` is a `Symbol` (same table as `prepend_spin`) or an explicit 2×2 matrix.
Equivalent to `postpend_op(H, s, op)`.
"""
postpend_spin(H::MPO, s::Index, op::Symbol)          = postpend_op(H, s, op)
postpend_spin(H::MPO, s::Index, mat::AbstractMatrix) = postpend_op(H, s, mat)


# ============================================================
# 2. Nambu (particle–hole) site index and operators
# ============================================================

"""
    nambu_index() -> Index

Create a dim-2 Index tagged "Nambu" (state 1 = particle, state 2 = hole).
Pass the result as `s` to all `prepend_nambu` / `postpend_nambu` calls.
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
    prepend_nambu(H, s, op) -> MPO

Prepend a Nambu (particle–hole) operator on the Nambu index `s` (created with
`nambu_index()`). `op` is a `Symbol` from the table below or an explicit 2×2
matrix. Equivalent to `prepend_op(H, s, op)`.

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
    postpend_nambu(H, s, op) -> MPO

Append a Nambu operator on the Nambu index `s` to the *end* of `H`.
`op` is a `Symbol` (same table as `prepend_nambu`) or an explicit 2×2 matrix.
Equivalent to `postpend_op(H, s, op)`.
"""
postpend_nambu(H::MPO, s::Index, op::Symbol)          = postpend_op(H, s, op)
postpend_nambu(H::MPO, s::Index, mat::AbstractMatrix) = postpend_op(H, s, mat)


# ============================================================
# 3. Spin extension
# ============================================================

"""
    add_spin!(H; cutoff=1e-8, maxdim=200, position=:pre) -> H

Extend `H` to a spin-½ degenerate system by adding a spin-½ index in front of
(`position=:pre`) or behind (`position=:post`) the existing sites.
The resulting Hamiltonian is `I_spin ⊗ H` (both spin sectors identical).

No-op if `H` is already spinful (`H.spin_s !== nothing`).
Invalidates all caches.
"""
function add_spin!(H::TBHamiltonian; cutoff::Real=1e-8, maxdim::Int=200,
                   position::Symbol=:pre)
    _require_binary_position_space(H, "add_spin!")
    H.spin_s === nothing || return H
    spin_s = spin_index()
    if position === :pre
        H.mpo   = prepend_spin(H.mpo, spin_s, :Id)
        H.sites = [spin_s; H.sites]
    else
        H.mpo   = postpend_spin(H.mpo, spin_s, :Id)
        H.sites = [H.sites; spin_s]
    end
    ITensorMPS.truncate!(H.mpo; maxdim=maxdim, cutoff=cutoff)
    H.spin_s   = spin_s
    H.aux_side = position
    _invalidate_cache!(H)
    return H
end


# ============================================================
# 4. Zeeman coupling
# ============================================================

"""
    add_zeeman!(H, h; direction=:z, tol=1e-8, maxdim=200, position=nothing) -> H

Add a Zeeman coupling `h · Sα` to `H`.  Calls `add_spin!` automatically if
`H` is not yet spinful, placing the spin index at `position` (default
`H.aux_side`).

`h` can be:
- a `Number`    — uniform field amplitude `h₀`
- a `Function`  — spatially varying `h(i)`, `i ∈ {1, …, N}` (1-indexed)

`direction`: `:x`, `:y`, or `:z` (default).

If `add_superconductivity!` was already called, the Zeeman term is wrapped in
`τ_z` so it enters with opposite sign in the hole sector, as required in BdG.

Examples
--------
```julia
add_zeeman!(H, 0.1)                         # uniform h = 0.1 along z
add_zeeman!(H, i -> 0.05 * sin(2π*i/H.N))  # oscillating field
add_zeeman!(H, 0.05; direction=:x)          # in-plane
```
"""
function add_zeeman!(H::TBHamiltonian, h;
                     direction::Symbol = :z,
                     tol::Real  = 1e-8,
                     maxdim::Int = 200,
                     position::Union{Nothing,Symbol} = nothing)
    _require_binary_position_space(H, "add_zeeman!")
    direction in (:x, :y, :z) ||
        error("direction must be :x, :y, or :z; got :$direction")
    pos = something(position, H.aux_side)
    add_spin!(H; cutoff=tol, maxdim=maxdim, position=pos)

    spin_op = direction == :z ? :Sz : direction == :x ? :Sx : :Sy
    pos_s   = _pos_sites(H)
    h_mpo   = h isa Number ? h * MPO(pos_s, "Id") :
                             get_diagonal_mpo(H.L, pos_s, h)

    if H.aux_side === :pre
        H_Z = prepend_spin(h_mpo, H.spin_s, spin_op)
        H.nambu_s !== nothing && (H_Z = prepend_nambu(H_Z, H.nambu_s, :tz))
    else
        H_Z = postpend_spin(h_mpo, H.spin_s, spin_op)
        H.nambu_s !== nothing && (H_Z = postpend_nambu(H_Z, H.nambu_s, :tz))
    end

    H.mpo = +(H.mpo, H_Z; maxdim=maxdim, cutoff=tol)
    ITensorMPS.truncate!(H.mpo; maxdim=maxdim, cutoff=tol)
    _invalidate_cache!(H)
    return H
end


# ============================================================
# 5. Superconducting pairing (BdG extension)
# ============================================================

"""
    add_superconductivity!(H, Δ; type=:swave, tol=1e-8, maxdim=200,
                           position=nothing) -> H

Extend `H` to a Bogoliubov–de Gennes (BdG) Hamiltonian by prepending
(`position=:pre`) or postpending (`position=:post`) a Nambu (particle–hole)
index; `position` defaults to `H.aux_side`.

The BdG structure is:
    H_BdG = τ_z ⊗ H_kin  +  τ_+ ⊗ H_pair  +  τ_- ⊗ H_pair†

- **Spinless + p-wave** (`type=:pwave`, or auto-selected when spinless + `:swave`):
  `H_pair = Δ·(K_forward − K_backward)`, the antisymmetric nearest-neighbour
  matrix required by Fermi statistics (`Δ(i,j) = −Δ(j,i)`).  This is the
  Kitaev chain.  `Δ` must be a `Number`.
- **Spinful + s-wave** (`add_spin!` called first, `type=:swave`):
  `H_pair = (i·σ_y)_spin ⊗ Δ(r)`, the standard BCS singlet Cooper-pair operator.
  On-site (s-wave) pairing is allowed here because the antisymmetry is carried
  by the spin singlet factor `i·σ_y`.  `Δ` can be a `Number` or 1-arg `Function`.
- **Custom** (`type=:custom`): arbitrary pairing matrix via 2-arg function `Δ(i,j)`,
  compressed with TCI.

**Note on spinless s-wave**: on-site pairing is forbidden for spinless fermions
(`Δ(i,i) = 0` by antisymmetry).  Calling with `type=:swave` on a spinless chain
automatically redirects to `:pwave` (uniform `Δ`) or errors (spatially varying `Δ`).

`type`:
- `:swave`  (default) — diagonal pairing for spinful chains; auto-redirects to
  `:pwave` for spinless chains when `Δ isa Number`
- `:pwave`  — antisymmetric NN pairing `Δ·(K_f − K_b)` for spinless chains; `Δ` must be a `Number`
- `:custom` — arbitrary `Δ(i,j)`; pass a 2-arg function

Errors if BdG has already been applied.  Invalidates all caches.

Examples
--------
```julia
add_superconductivity!(H_spinless, 0.1)              # auto p-wave (Kitaev chain)
add_superconductivity!(H_spinless, 0.1; type=:pwave) # explicit p-wave
add_superconductivity!(H_spinful,  0.1)              # singlet s-wave (spinful required)
add_superconductivity!(H_spinful,  i -> i < N÷2 ? 0.1 : 0.0)  # spatially varying s-wave
add_superconductivity!(H, (i,j) -> ...; type=:custom)          # general pairing
```
"""
function add_superconductivity!(H::TBHamiltonian, Δ;
                                type::Symbol = :swave,
                                tol::Real    = 1e-8,
                                maxdim::Int  = 200,
                                position::Union{Nothing,Symbol} = nothing)
    _require_binary_position_space(H, "add_superconductivity!")
    H.nambu_s === nothing ||
        error("BdG already applied (H.nambu_s is set). Cannot apply twice.")

    pos   = something(position, H.aux_side)
    pos_s = _pos_sites(H)

    # ── Spinless + :swave redirect ───────────────────────────────────────────
    if H.spin_s === nothing && type === :swave
        if Δ isa Number
            @info("On-site (s-wave) pairing is forbidden for spinless fermions " *
                   "(Δ(i,i) = 0 by Fermi antisymmetry).  " *
                   "Constructing nearest-neighbour p-wave instead.")
            type = :pwave
        else
            error("On-site (s-wave) pairing is forbidden for spinless fermions.  " *
                  "For spatially varying spinless pairing use type=:custom with a 2-arg Function Δ(i,j).")
        end
    end

    # ── Build the pairing MPO in position space ──────────────────────────────
    H_pair_pos = if type === :pwave
        H.spin_s === nothing ||
            error("type=:pwave is only for spinless chains.  " *
                  "For spinful p-wave use type=:custom with a 2-arg Function Δ(i,j).")
        Δ isa Number ||
            error("For type=:pwave, Δ must be a Number.  " *
                  "For spatially varying spinless pairing use type=:custom with a 2-arg Function Δ(i,j).")
        # H_pair = Δ·(Kf − Kb): antisymmetric NN pairing matrix.
        # The τ- ⊗ H_pair† term handles the hole-particle sector automatically.
        pairingNNN(H.L, pos_s, Δ * MPO(pos_s, "Id"), 1)
    elseif type === :swave
        Δ isa Number   ? Δ * MPO(pos_s, "Id")            :
        Δ isa Function ? get_diagonal_mpo(H.L, pos_s, Δ) :
        error("For type=:swave, Δ must be a Number or a 1-arg Function.")
    elseif type === :custom
        Δ isa Function ||
            error("For type=:custom, Δ must be a 2-arg Function Δ(i,j).")
        pairing2MPO(Δ, H.N, pos_s; tol=tol, type=ComplexF64)
    else
        error("Unknown pairing type :$type.  Use :swave, :pwave, or :custom.")
    end

    # ── Lift pairing to spin space if needed ─────────────────────────────────
    H_pair = if H.spin_s !== nothing
        pos === :pre ? prepend_spin(H_pair_pos,  H.spin_s, :iSy) :
                       postpend_spin(H_pair_pos, H.spin_s, :iSy)
    else
        H_pair_pos
    end

    H_pair_adj = swapprime(dag(H_pair), 0, 1)

    # ── BdG assembly ─────────────────────────────────────────────────────────
    nambu_s = nambu_index()
    if pos === :pre
        H_bdg = +(+(prepend_nambu(H.mpo,      nambu_s, :tz),
                    prepend_nambu(H_pair,     nambu_s, :tp); cutoff=tol),
                    prepend_nambu(H_pair_adj, nambu_s, :tm); cutoff=tol)
        H.sites = [nambu_s; H.sites]
    else
        H_bdg = +(+(postpend_nambu(H.mpo,      nambu_s, :tz),
                    postpend_nambu(H_pair,     nambu_s, :tp); cutoff=tol),
                    postpend_nambu(H_pair_adj, nambu_s, :tm); cutoff=tol)
        H.sites = [H.sites; nambu_s]
    end
    ITensorMPS.truncate!(H_bdg; maxdim=maxdim, cutoff=tol)

    Δ_scale    = Δ isa Number ? abs(Δ) : 1.0
    H.mpo      = H_bdg
    H.nambu_s  = nambu_s
    H.aux_side = pos
    H.scale    = H.scale + Δ_scale * 1.1   # rough update; user can override
    _invalidate_cache!(H)
    return H
end


# ============================================================
# 6. Spin-orbit coupling
# ============================================================

"""
    add_soc!(H, λ; type=:rashba, direction=:z, tol=1e-8, maxdim=200,
             position=nothing) -> H

Add spin-orbit coupling to `H`.  Calls `add_spin!` automatically if needed,
placing the spin index at `position` (default `H.aux_side`).

`type`:
- `:rashba` — nearest-neighbour Rashba SOC on the position chain:
              `λ · (S_y ⊗ K_u − S_y ⊗ K_d)` where `K_u/K_d` are the ±1 shift
              operators.  `λ` must be a scalar.  Breaks SU(2) spin symmetry
              while preserving time-reversal.
- `:ising`  — diagonal Ising SOC `λ(i) · S_z` (equivalent to a position-dependent
              Zeeman along z; useful for Kane–Mele type models).
- `:custom` — arbitrary position-space MPO `λ_mpo` tensor-producted with the
              spin operator given by `direction` (`:x`, `:y`, or `:z`).
              `λ` may be a Number, a 1-arg `Function λ(i)`, or a 2-arg
              `Function λ(i,j)` (the last compressed via TCI).
              For the result to be Hermitian, the position-space matrix must
              itself be Hermitian: `λ(i,j) = conj(λ(j,i))`.  Diagonal and
              real-symmetric inputs satisfy this automatically.

Examples
--------
```julia
add_soc!(H, 0.05)                             # Rashba λ=0.05
add_soc!(H, i -> 0.1*cos(2π*i/H.N); type=:ising)
add_soc!(H, (i,j)->...; type=:custom, direction=:y)
```
"""
function add_soc!(H::TBHamiltonian, λ;
                  type::Symbol      = :rashba,
                  direction::Symbol = :z,
                  tol::Real         = 1e-8,
                  maxdim::Int       = 200,
                  position::Union{Nothing,Symbol} = nothing)
    _require_binary_position_space(H, "add_soc!")
    pos = something(position, H.aux_side)
    add_spin!(H; cutoff=tol, maxdim=maxdim, position=pos)
    pos_s = _pos_sites(H)

    spin_prepend = H.aux_side === :pre ? prepend_spin : postpend_spin

    H_soc = if type === :ising
        λ_mpo = λ isa Number ? λ * MPO(pos_s, "Id") :
                               get_diagonal_mpo(H.L, pos_s, λ)
        spin_prepend(λ_mpo, H.spin_s, :Sz)

    elseif type === :rashba
        λ isa Number || error("Rashba SOC requires a scalar λ; got $(typeof(λ)).")
        K_u = generate_kin_u(pos_s, H.N)
        K_d = generate_kin_d(pos_s, H.N)
        # λ·(iσ_y) ⊗ (K_u − K_d): both factors anti-Hermitian → product Hermitian.
        # :Sy (Hermitian) ⊗ anti-Hermitian would give a non-Hermitian term.
        +(spin_prepend( λ * K_u, H.spin_s, :iSy),
          spin_prepend(-λ * K_d, H.spin_s, :iSy); cutoff=tol)

    elseif type === :custom
        direction in (:x, :y, :z) ||
            error("direction must be :x, :y, or :z; got :$direction")
        spin_op = direction == :z ? :Sz : direction == :x ? :Sx : :Sy
        λ_mpo = if λ isa Number
            λ * MPO(pos_s, "Id")
        elseif λ isa Function && applicable(λ, 1)
            get_diagonal_mpo(H.L, pos_s, λ)
        elseif λ isa Function
            hopping2MPO(λ, H.N, pos_s; tol=tol, type=ComplexF64)
        else
            error("λ must be a Number or a Function.")
        end
        spin_prepend(λ_mpo, H.spin_s, spin_op)

    else
        error("Unknown SOC type :$type.  Use :rashba, :ising, or :custom.")
    end

    H.mpo = +(H.mpo, H_soc; maxdim=maxdim, cutoff=tol)
    ITensorMPS.truncate!(H.mpo; maxdim=maxdim, cutoff=tol)
    _invalidate_cache!(H)
    return H
end


# ============================================================
# 7. Auxiliary index projection utilities
# ============================================================
#
# Any auxiliary DOF (spin, Nambu, layer, sublattice) added with prepend_op /
# postpend_op lives at the first or last site of the MPO as a dim-1-bonded
# tensor.  project_aux is the removal step of the projection chain
# _project_aux_sectors (section 9), which get_bands (physics/qft/bands.jl) and
# get_ldos_spatial (solvers/kpm/ldos.jl) apply to every Chebyshev operator.
#
# project_aux(W, aux_s, sec; side)
#   Contracts the projector |sec⟩⟨sec| onto the bra (aux_s') and ket (aux_s)
#   physical indices of the aux tensor (the kernel _project_end_site,
#   section 8).  The resulting link is absorbed into the adjacent position
#   site, returning an (L−1)-site MPO.
#   `side=:pre` for prepended indices (spin, Nambu, layer);
#   `side=:post` for postpended indices (sublattice).
#
# aux_site(H, which) -> (Index, Symbol)
#   Extracts the auxiliary Index and its side (:pre or :post) from H.sites.
#   `which` ∈ :spin, :nambu, :layer, :sublattice.
#   Used by _aux_projection (section 9) to auto-detect the auxiliary indices
#   of a TBHamiltonian without user intervention.

"""
    project_aux(W, aux_s, σ; side=:pre) -> MPO

Remove an auxiliary site from MPO `W` by projecting onto state `σ`.

- `side=:pre`  — aux site is at position 1 (prepended, e.g. spin).
- `side=:post` — aux site is at the last position (postpended, e.g. sublattice).

Contracts the projector |σ⟩⟨σ| on both bra and ket physical indices of the
aux tensor; the resulting dim-1 link is absorbed into the adjacent position
site.  Returns an (L−1)-site MPO suitable for `conjugate_by_qft`.
"""
project_aux(W::MPO, aux_s::Index, σ::Integer; side::Symbol = :pre) =
    _project_end_site(W, aux_s, σ, σ, side)

# Nothing-overloads: give Julia a compilable method when the Index is nothing,
# so branches in get_bands can be type-checked without a MethodError.
project_aux(::MPO, ::Nothing, ::Integer; side::Symbol=:pre) =
    error("sublat_proj=true requires sublat_s to be set (detected from H.sublattice_s)")


"""
    _autoenable_proj(H, nambu_proj, spin_proj, layer_proj, sublat_proj)
        -> (nambu_proj, spin_proj, layer_proj, sublat_proj)

Enable projection flags for any auxiliary DOF detected on `H`, emitting one
`@info` record per auto-enabled flag.  Called by `_aux_projection` (with
`autoenable=true`), which every `TBHamiltonian` spectral method but the
stochastic DOS uses before any aux-index logic runs.
"""
function _autoenable_proj(H::TBHamiltonian,
                           nambu_proj::Bool, spin_proj::Bool,
                           layer_proj::Bool, sublat_proj::Bool)
    if !isnothing(H.nambu_s) && !nambu_proj
        @info "H.nambu_s detected; auto-enabling nambu_proj=true " *
              "(pass proj_nambu=1/2 to select particle/hole sector)."
        nambu_proj = true
    end
    if !isnothing(H.spin_s) && !spin_proj
        @info "H.spin_s detected; auto-enabling spin_proj=true " *
              "(pass proj_s=1/2 to select ↑/↓ sector)."
        spin_proj = true
    end
    if !isnothing(H.layer_s) && !layer_proj
        @info "H.layer_s detected; auto-enabling layer_proj=true " *
              "(pass proj_layer=k to select a layer)."
        layer_proj = true
    end
    if !isnothing(H.sublattice_s) && !sublat_proj
        @info "H.sublattice_s detected; auto-enabling sublat_proj=true " *
              "(pass proj_sl=k to select a sublattice)."
        sublat_proj = true
    end
    return nambu_proj, spin_proj, layer_proj, sublat_proj
end


"""
    aux_site(H, which) -> (Index, Symbol)

Return the auxiliary `Index` and its position side (`:pre` or `:post`) for
the named auxiliary degree of freedom in `H`.

`which` ∈ `:spin`, `:sublattice`, `:nambu`, `:layer`.

Useful for passing the correct arguments to `project_aux` without manually
inspecting `H.sites`.

```julia
s, side = aux_site(H, :sublattice)
W_A = project_aux(W, s, 1; side=side)   # sublattice-A channel
```
"""
function aux_site(H::TBHamiltonian, which::Symbol)
    s = which === :spin       ? H.spin_s        :
        which === :sublattice ? H.sublattice_s  :
        which === :nambu      ? H.nambu_s       :
        which === :layer      ? H.layer_s       :
        error("Unknown auxiliary type :$which.  Use :spin, :sublattice, :nambu, or :layer.")
    isnothing(s) && error("H has no $which auxiliary index.")
    pos  = findfirst(==(s), H.sites)
    isnothing(pos) && error("Auxiliary index not found in H.sites — this is a bug.")
    side = pos == 1             ? :pre  :
           pos == length(H.sites) ? :post :
           error("Auxiliary $which index found at interior position $pos (unsupported).")
    return s, side
end


# ============================================================
# 8. Auxiliary sector projectors
# ============================================================
#
# Each projector removes one aux site from an MPO: it contracts that site with
# |row⟩⟨col| on (s', s) and absorbs the remaining link tensor into a neighbour.
# Two kernels do it, because their callers differ in the element type and the
# contraction order they have always used:
#
#   _project_end_site   one-hot pair (Float64, so a real operator stays real),
#                       an end site, absorbed as `W[neighbour] * block`
#                       → project_aux, contract_nh_block (physics/nh/kpm.jl)
#   _absorb_aux_site    a ComplexF64 projector (_block_projector), any site,
#                       absorbed as `block * W[2]` or `W[pos − 1] * block`
#                       → _project_aux_block, _project_spin_sector
#
# Merging the two would make project_aux's result complex for a real operator
# (the golden tests record element types), or the SCF/RPA blocks real. The
# wrappers keep what else differs: how
# the site is found (given side, detected end, index or tag, "Spin" tag), their
# checks and error messages, and _project_spin_sector's TBHamiltonian copy. The
# GPU twins _project_aux_gpu (gpu/primitives.jl) and _contract_nh_block_gpu
# (gpu/nh.jl) build dense device projectors instead (one-hot tensors do not move
# to the GPU) and stay with the GPU code.

"""
    _project_end_site(W, s, row, col, side) -> MPO

Contract the end site of `W` that carries `s` (the first site for `side=:pre`,
the last for any other `side`) with the one-hot pair ⟨row| on `s'` and |col⟩ on
`s`, and absorb the link tensor left over into the neighbouring site, as
`W[neighbour] * block`. Returns the MPO without that site. Nothing is checked:
the caller guarantees that `s` sits there (see project_aux, contract_nh_block).
"""
function _project_end_site(W::MPO, s::Index, row::Integer, col::Integer, side::Symbol)
    L = length(W)
    if side === :pre
        block = W[1] * onehot(s' => row) * onehot(s => col)
        return MPO(ITensor[W[2] * block; [W[i] for i in 3:L]])
    else
        block = W[L] * onehot(s' => row) * onehot(s => col)
        return MPO(ITensor[[W[i] for i in 1:L-2]; W[L-1] * block])
    end
end

# The ComplexF64 projector |row⟩⟨col| on (s', s) of _project_aux_block and
# _project_spin_sector.
function _block_projector(s::Index, row::Int, col::Int)
    proj = ITensor(ComplexF64, s', s)
    proj[s' => row, s => col] = 1.0
    return proj
end

# Contract site `pos` of `mpo` with `proj` and absorb the result into site 2
# (`pos == 1`, as `block * W[2]`) or site `pos − 1` (as `W[pos − 1] * block`).
# Returns the tensors of the remaining sites, in order.
function _absorb_aux_site(mpo::MPO, pos::Int, proj::ITensor)
    tensors    = ITensor[mpo[i] for i in eachindex(mpo)]
    contracted = tensors[pos] * proj
    if pos == 1
        tensors[2] = contracted * tensors[2]
        return tensors[2:end]
    end
    tensors[pos - 1] = tensors[pos - 1] * contracted
    return vcat(tensors[1:pos - 1], tensors[pos + 1:end])
end

"""
    _project_aux_block(mpo, aux_s, row, col; tag="") -> MPO

The `(row, col)` block of the aux site `aux_s` of `mpo`: that site is contracted
with the ComplexF64 projector |row⟩⟨col| and absorbed into a neighbour. The site
is the first one that carries `aux_s`, or an index tagged `tag` when `tag` is
not empty; it may be anywhere in the MPO. A one-site MPO gives an empty MPO.
"""
function _project_aux_block(mpo::MPO, aux_s::Index, row::Int, col::Int; tag::String="")
    aux_pos = findfirst(n -> any(i -> i == aux_s || (!isempty(tag) && hastags(i, tag)),
                                 siteinds(mpo, n)),
                        1:length(mpo))
    aux_pos === nothing && error("_project_aux_block: auxiliary index not found")

    proj = _block_projector(aux_s, row, col)
    length(mpo) == 1 && return MPO(ITensor[])
    return MPO(_absorb_aux_site(mpo, aux_pos, proj))
end


"""
    _project_spin_sector(H, sector) -> TBHamiltonian

Project a spinful `TBHamiltonian` onto spin sector `sector` (1 = ↑, 2 = ↓)
by contracting the spin site tensor with the projector |sector⟩⟨sector|.

The spin index is identified by its "Spin" tag, so the function is robust
to whether spin is prepended or postpended.  The contracted tensor is
absorbed into its neighbour, leaving a valid L-qubit MPO.

Returns a new `TBHamiltonian` with `spin_s = nothing` and fresh (empty)
caches; `scale` and `center` are reset to 0.0 so `_ensure_scale!` will
re-estimate them on the first KPM call. All other fields (`Lx`,
`interaction_mpo`, `fock_mpo`, `position_space`, …) are copied from `H`.
"""
function _project_spin_sector(H::TBHamiltonian, sector::Int)
    H.spin_s === nothing &&
        error("_project_spin_sector: H is not spinful (spin_s is nothing)")
    s = H.spin_s

    spin_pos = findfirst(n -> any(i -> hastags(i, "Spin"), siteinds(H.mpo, n)),
                         1:length(H.mpo))
    spin_pos === nothing && error("_project_spin_sector: spin Index not found in MPO")

    kept = _absorb_aux_site(H.mpo, spin_pos, _block_projector(s, sector, sector))
    # For a spin site inside the MPO (a postpended spin followed by a postpended
    # Nambu site) the sites after it are dropped, as they always were; see
    # docs/dev/REORGANISATION_TODO.md.
    new_tensors = spin_pos == 1 ? kept : kept[1:spin_pos - 1]

    new_sites = filter(i -> !hastags(i, "Spin"), H.sites)

    # interaction_mpo / fock_mpo live on the position sites (add_interaction!), which
    # the projection leaves untouched, so they are kept along with Lx and position_space.
    return TBHamiltonian(H; sites=new_sites, mpo=MPO(new_tensors),
                         scale=0.0, center=0.0, spin_s=nothing)
end


# ============================================================
# 9. The auxiliary projection of a spectral method: AuxProjection
# ============================================================
#
# The spectral methods (get_ldos_online, get_ldos_spatial, get_dos_stochastic,
# get_bands and the GPU get_bands_gpu, get_ldos_spatial_gpu,
# get_dos_stochastic_gpu) take the same eight keywords, a flag and a sector
# selector per auxiliary DOF: nambu_proj/proj_nambu, spin_proj/proj_s,
# layer_proj/proj_layer, sublat_proj/proj_sl. The keywords stay in every public
# signature; inside, one AuxProjection carries them together with the Index and
# the side of each DOF on the MPO. _aux_projection builds it from H (the
# low-level get_bands from its explicit index keywords), _project_aux_sectors
# applies it to a Chebyshev operator T_n (the MPO methods), and _probe_sectors
# lists the sectors of the probe states (the MPS methods, with probe_state).
#
# A struct rather than keywords forwarded to _aux_setup: the projection chain
# needs a flag, a selector, an Index and a side for each of the four DOFs, and
# the low-level get_bands supplies the indices itself instead of detecting them
# on H; one value lets both feed the same chain.

"""
    AuxDOFProjection(on, sector, index, side)

How a spectral method treats one auxiliary DOF: `on` projects it (the `*_proj`
flag, after `_autoenable_proj`), `sector` is the `proj_*` selector (`nothing`
sums every sector, an integer keeps one), `index` is its Index on the MPO
(`nothing` when absent) and `side` the end of the MPO it sits at (`:pre` first,
`:post` last).
"""
struct AuxDOFProjection
    on::Bool
    sector
    index
    side::Symbol
end

"""
    AuxProjection(nambu, spin, layer, sublat)

The auxiliary projection of a spectral method: one `AuxDOFProjection` per DOF
(the field `sublat` is the sublattice). Built by `_aux_projection(H; …)` from
the eight public keywords, or by the low-level `get_bands` from its own.
"""
struct AuxProjection
    nambu::AuxDOFProjection
    spin::AuxDOFProjection
    layer::AuxDOFProjection
    sublat::AuxDOFProjection
end

"""
    _aux_projection(H; nambu_proj=false, proj_nambu=nothing, spin_proj=false,
                    proj_s=nothing, layer_proj=false, proj_layer=nothing,
                    sublat_proj=false, proj_sl=nothing, autoenable=true)
        -> AuxProjection

The auxiliary projection a spectral method applies to `H`, from its eight
public keywords. With `autoenable=true` the flag of every DOF present on `H` is
switched on (`_autoenable_proj`, one `@info` per flag it enables); the stochastic
DOS passes `false`. The Nambu, layer and sublattice sites are located with
`aux_site`. The spin site is always taken to be the first one (`side = :pre`):
its side is not detected, which is the postpended-spin segfault listed in
docs/dev/REORGANISATION_TODO.md.
"""
function _aux_projection(H::TBHamiltonian;
                         nambu_proj::Bool  = false, proj_nambu = nothing,
                         spin_proj::Bool   = false, proj_s     = nothing,
                         layer_proj::Bool  = false, proj_layer = nothing,
                         sublat_proj::Bool = false, proj_sl    = nothing,
                         autoenable::Bool  = true)
    if autoenable
        nambu_proj, spin_proj, layer_proj, sublat_proj =
            _autoenable_proj(H, nambu_proj, spin_proj, layer_proj, sublat_proj)
    end
    nambu_s,  nambu_side  = !isnothing(H.nambu_s)      ? aux_site(H, :nambu)      : (nothing, :pre)
    layer_s,  layer_side  = !isnothing(H.layer_s)      ? aux_site(H, :layer)      : (nothing, :pre)
    sublat_s, sublat_side = !isnothing(H.sublattice_s) ? aux_site(H, :sublattice) : (nothing, :post)
    return AuxProjection(AuxDOFProjection(nambu_proj,  proj_nambu, nambu_s,  nambu_side),
                         AuxDOFProjection(spin_proj,   proj_s,     H.spin_s, :pre),
                         AuxDOFProjection(layer_proj,  proj_layer, layer_s,  layer_side),
                         AuxDOFProjection(sublat_proj, proj_sl,    sublat_s, sublat_side))
end

# Whether any DOF is projected (the probe-state methods then fix the aux sectors).
_any_projected(aux::AuxProjection) =
    aux.nambu.on || aux.spin.on || aux.layer.on || aux.sublat.on

# All `n` sectors, or the selected one.
_sector_range(sector, n) = isnothing(sector) ? (1:n) : (sector:sector)

"""
    _project_aux_sectors(T, aux; project=project_aux, spin_index=nothing,
                         sublattice=aux.sublat.on)

The projections of the operator `T` onto every requested auxiliary sector,
outermost DOF first: Nambu (sectors `1:2`, or `proj_nambu`), spin (`1:2`, or
`proj_s`), layer (all its sectors, or `proj_layer`) and, when `sublattice` is
true, the sublattice (all, or `proj_sl`). A DOF whose flag is off is left alone.
Returns an iterator of `(T_proj, s)` pairs, `s` the sublattice sector (`1` when
the sublattice is not projected), in the order of the four hand-written chains
it replaced: the spin step runs over its sectors outermost (a two-dimensional
comprehension), the others over their sectors innermost, so the Nambu sector
varies fastest across the spin step. The Nambu, spin and layer lists are built
when it is called, each sublattice projection when the iteration reaches it.

- `project(T, index, σ; side)` removes one aux site: `project_aux` on the CPU,
  `_project_aux_gpu` on the GPU.
- `spin_index` is the spin Index to project; `nothing` takes `aux.spin.index`,
  which must then be set (get_ldos_spatial). get_bands and the GPU methods pass
  their fallback, `sites[1]` for a Hamiltonian without spin.
- `sublattice`: get_bands projects the sublattice when `sublat_proj` is on, the
  spatial LDOS whenever `H` has a sublattice index.
"""
function _project_aux_sectors(T::MPO, aux::AuxProjection;
                              project          = project_aux,
                              spin_index       = nothing,
                              sublattice::Bool = aux.sublat.on)
    (; nambu, spin, layer, sublat) = aux
    Ts = nambu.on ?
        [project(T, nambu.index::Index, σ; side=nambu.side)
         for σ in _sector_range(nambu.sector, 2)] :
        MPO[T]
    if spin.on
        s_idx = isnothing(spin_index) ? spin.index::Index : spin_index
        Ts = [project(t, s_idx, σ; side=spin.side)
              for t in Ts, σ in _sector_range(spin.sector, 2)]
    end
    if layer.on
        n_lay = dim(layer.index::Index)
        Ts = [project(t, layer.index::Index, σ; side=layer.side)
              for t in Ts for σ in _sector_range(layer.sector, n_lay)]
    end
    sublattice || return ((t, 1) for t in Ts)
    sl_range = _sector_range(sublat.sector, dim(sublat.index::Index))
    return ((project(t, sublat.index::Index, σ; side=sublat.side), σ)
            for t in Ts for σ in sl_range)
end

# The sectors a probe loop runs over for one DOF: all `nsectors` (default: the
# dimension of its index) or the selected one when the DOF is projected and
# present on H, else the placeholder 1:1.
function _probe_range(d::AuxDOFProjection, nsectors = nothing)
    (d.on && !isnothing(d.index)) || return 1:1
    return _sector_range(d.sector, something(nsectors, dim(d.index::Index)))
end

"""
    _probe_sectors(aux) -> Vector

The auxiliary sectors `(σ_n, σ_s, σ_l, σ_sl)` (Nambu, spin, layer, sublattice) a
probe-state method sums over, Nambu outermost and sublattice innermost: for each
projected DOF present on `H` all its sectors (the spin sectors are `1:2`) or the
selected one, `1` for the others. `[nothing]` when no DOF is projected: one plain
position probe. Each entry is the third argument of `probe_state`.
"""
function _probe_sectors(aux::AuxProjection)
    _any_projected(aux) || return [nothing]
    rn, rs = _probe_range(aux.nambu), _probe_range(aux.spin, 2)
    rl, rsl = _probe_range(aux.layer), _probe_range(aux.sublat)
    return [(σ_n, σ_s, σ_l, σ_sl) for σ_n in rn for σ_s in rs for σ_l in rl for σ_sl in rsl]
end

"""
    _aux_setup(H, nambu_proj, proj_nambu, spin_proj, proj_s,
               layer_proj, proj_layer, sublat_proj, proj_sl) -> NamedTuple

The flat view of `_aux_projection(H; …, autoenable=false)` with the probe ranges.
Fields: `nambu_s_det`, `nambu_side_det`, `spin_s_det`, `layer_s_det`,
`layer_side_det`, `sublat_s_det`, `sublat_side_det` (the indices and sides),
`nambu_range`, `spin_range`, `layer_range`, `sl_range` (the sector ranges of
`_probe_sectors`) and `any_aux_proj`. The package itself now uses the
`AuxProjection`; the KPM golden test pins this view.
"""
function _aux_setup(H::TBHamiltonian,
                    nambu_proj::Bool, proj_nambu,
                    spin_proj::Bool,  proj_s,
                    layer_proj::Bool, proj_layer,
                    sublat_proj::Bool, proj_sl)
    aux = _aux_projection(H; nambu_proj, proj_nambu, spin_proj, proj_s,
                             layer_proj, proj_layer, sublat_proj, proj_sl,
                             autoenable=false)
    (; nambu, spin, layer, sublat) = aux
    return (; nambu_s_det = nambu.index, nambu_side_det = nambu.side,
              spin_s_det  = spin.index,
              layer_s_det = layer.index, layer_side_det = layer.side,
              sublat_s_det = sublat.index, sublat_side_det = sublat.side,
              nambu_range = _probe_range(nambu), spin_range = _probe_range(spin, 2),
              layer_range = _probe_range(layer), sl_range = _probe_range(sublat),
              any_aux_proj = _any_projected(aux))
end


# ============================================================
# 10. Probe states of the MPS methods
# ============================================================

"""
    _ldos_make_psi0(H, x, σ_n, σ_s, σ_l, σ_sl) -> MPS

Product-state MPS over all `H.sites` for KPM evaluation in LDOS `:mps` mode.

- Position sites encode `x-1` in big-endian binary (first position site = MSB).
- Auxiliary sites are set to 1-based sector indices:
  `σ_n` (nambu), `σ_s` (spin), `σ_l` (layer), `σ_sl` (sublattice).
  Indices for absent aux dofs are ignored.
"""
function _ldos_make_psi0(H::TBHamiltonian, x::Int,
                          σ_n::Int, σ_s::Int, σ_l::Int, σ_sl::Int)
    k       = 0
    pos_bit = H.L - 1   # bit index: MSB of (x-1) goes to the first position site
    for s in H.sites
        k *= dim(s)
        if     !isnothing(H.nambu_s)      && s == H.nambu_s;      k += σ_n  - 1
        elseif !isnothing(H.spin_s)       && s == H.spin_s;       k += σ_s  - 1
        elseif !isnothing(H.layer_s)      && s == H.layer_s;      k += σ_l  - 1
        elseif !isnothing(H.sublattice_s) && s == H.sublattice_s; k += σ_sl - 1
        else
            k += (x - 1) >> pos_bit & 1
            pos_bit -= 1
        end
    end
    return _basis_state_mps(k, H.sites)
end

"""
    probe_state(H, x) -> MPS
    probe_state(H, x, σ) -> MPS

The probe state |x⟩ of the MPS Chebyshev methods at unit cell `x`
(`get_ldos_online`, `get_ldos_spatial(mode=:mps)`, the stochastic DOS). With
`σ = nothing` (the default) it is the position probe: `physical_site_state(H, x)`
when `H.sites` are the `H.L` position sites, the exciton pair state |x, x⟩
(`mpsexciton`) when `H` has more sites. With `σ = (σ_n, σ_s, σ_l, σ_sl)`, an entry
of `_probe_sectors`, it is the binary product state with the auxiliary sites set
to those sectors (`_ldos_make_psi0`).
"""
probe_state(H::TBHamiltonian, x::Integer, ::Nothing = nothing) =
    length(H.sites) == H.L ? physical_site_state(H, x) : mpsexciton(x, H.sites)

probe_state(H::TBHamiltonian, x::Integer, σ::NTuple{4,Integer}) =
    _ldos_make_psi0(H, x, σ...)
