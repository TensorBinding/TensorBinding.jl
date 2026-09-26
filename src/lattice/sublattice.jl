# sublattice.jl — lattices with an explicit sublattice index (kagome, Lieb,
# honeycomb, honeycomb NNN, dice/T3, SSH chain): the *_hamiltonian builders,
# each returning a TBHamiltonian with the sublattice site postpended
# (H.sublattice_s, H.aux_side = :post). get_Hamiltonian("kagome", ...) and the
# other sublattice geometry names call them; the matching *_positions tables
# live in lattice/geometry.jl.
#
# Entry points: kagome_hamiltonian, lieb_hamiltonian,
#   honeycomb_sublattice_hamiltonian, honeycomb_nnn_hamiltonian,
#   dice_hamiltonian, ssh_sublattice_hamiltonian.
# Internals: _sublattice_setup (the sites and identity every builder starts from)
#   and _sublattice_bond (one Hermitian pair of inter-cell hops).
#
# Depends on: core/Utils.jl (shift_pair_mpos, postpend_op), core/MPOTools.jl
# (sum_mpos), core/TBSystem.jl (TBHamiltonian) and lattice/masks2d.jl
# (_row_break_mpo).
#
# Every builder sums its terms left to right (intra-cell first, then the bond
# types in the order they are listed), each partial sum compressed at `cutoff`,
# and truncates the total once with `maxdim`; that order is part of the output.
#
# Split from the former lattice/2Dlattice_tk.jl.

# ============================================================
# 1. Shared pieces of the builders
# ============================================================

"""
    _sublattice_setup(Lx, Ly, n_sub, tag; cutoff, maxdim) -> NamedTuple

What every sublattice builder starts from, for `2^Lx × 2^Ly` unit cells: `Nx = 2^Lx`,
`L = Lx + Ly`, `N = 2^L`, the `L` position qubits `pos_sites`, the dim-`n_sub`
sublattice index `sub_s` (tagged `tag`), `all_sites = [pos_sites; sub_s]`, the
identity `Id` on the positions, `cutoff` and the apply keywords
`apkw = (; cutoff, maxdim)`.
"""
function _sublattice_setup(Lx::Integer, Ly::Integer, n_sub::Integer, tag::AbstractString;
                           cutoff::Real, maxdim::Integer)
    Nx = 2^Lx
    L  = Lx + Ly
    N  = 2^L
    pos_sites = siteinds("Qubit", L)
    sub_s     = Index(n_sub, tag)
    all_sites = [pos_sites; sub_s]
    Id   = MPO(pos_sites, "Id")
    apkw = (; cutoff = cutoff, maxdim = maxdim)
    return (; Nx, L, N, pos_sites, sub_s, all_sites, Id, cutoff, apkw)
end


"""
    _sublattice_bond(S, q, amp, op; brk=nothing, cyclic=false) -> MPO

One inter-cell bond type of a sublattice builder with setup `S` (see
`_sublattice_setup`): the Hermitian pair

    amp · (K·brk ⊗ O)  +  conj(amp) · (brk·K† ⊗ O†),    K = shift_mpo(S.pos_sites, q; cyclic),

summed at `S.cutoff`. The sublattice operator is `O = |a⟩⟨b|` for `op = (a, b)` (the
backward hop gets `|b⟩⟨a|`) or the matrix `op` (the backward hop gets its adjoint).
`brk`, a row-break mask (`_row_break_mpo`), is applied on the source side of each
hop with `S.apkw`; without it the shift is used bare.
"""
function _sublattice_bond(S, q::Integer, amp::Number, op;
                          brk::Union{Nothing,MPO} = nothing, cyclic::Bool = false)
    K, Kdag = shift_pair_mpos(S.pos_sites, q; cyclic=cyclic)
    fwd = amp       * _postpend_bond(brk === nothing ? K    : apply(K, brk; S.apkw...),
                                     S.sub_s, op, false)
    bwd = conj(amp) * _postpend_bond(brk === nothing ? Kdag : apply(brk, Kdag; S.apkw...),
                                     S.sub_s, op, true)
    return +(fwd, bwd; cutoff=S.cutoff)
end

# The sublattice operator of _sublattice_bond on the forward or (back=true) backward hop.
_postpend_bond(M::MPO, s::Index, (a, b)::Tuple{Int,Int}, back::Bool) =
    back ? postpend_op(M, s, b, a) : postpend_op(M, s, a, b)
_postpend_bond(M::MPO, s::Index, O::AbstractMatrix, back::Bool) =
    postpend_op(M, s, back ? adjoint(O) : O)


# ============================================================
# 2. Kagome lattice
# ============================================================

"""
    kagome_hamiltonian(Lx, Ly, t=1.0; t_AB=t, t_AC=t, t_BC=t,
                       cutoff=1e-8, maxdim=200) -> TBHamiltonian

Build a kagomé tight-binding Hamiltonian as a `TBHamiltonian`.

**Encoding** (L+1 sites, L = Lx+Ly):
- Sites 1…L : position qubits for 2^L unit cells (row-major: n = ix + iy·2^Lx)
- Site  L+1 : dim-3 "Kagome" sublattice index A=1, B=2, C=3 (postpended)

**Bond amplitudes**

Each bond type controls both the intra-cell matrix entry and the matching
inter-cell hopping term along the corresponding lattice direction:

| Kwarg | Bond  | Intra-cell | Inter-cell direction       |
|-------|-------|------------|---------------------------|
| `t_AB`| A↔B   | yes        | x  (shift ±1)             |
| `t_AC`| A↔C   | yes        | y  (shift ±Nx)            |
| `t_BC`| B↔C   | yes        | diag (shift ±(Nx-1))      |

All three default to `t` (uniform kagomé).  For anisotropic / breathing kagomé
pass individual values:
```julia
H = kagome_hamiltonian(Lx, Ly; t_AB=1.0, t_AC=0.8, t_BC=0.6)
```

**Flat band**: at `E = −2t` (uniform case); dispersive bands reach up to `+4t`.
Boundary wrapping is suppressed.  Real-space coordinates: `kagome_positions(Lx, Ly)`.
`H.sublattice_s` stores the dim-3 sublattice index; `H.aux_side = :post`.
"""
function kagome_hamiltonian(Lx::Integer, Ly::Integer, t::Number = 1.0;
                             t_AB::Number = t,
                             t_AC::Number = t,
                             t_BC::Number = t,
                             cutoff::Real = 1e-8,
                             maxdim::Int  = 200)
    S = _sublattice_setup(Lx, Ly, 3, "Kagome"; cutoff, maxdim)

    brk_xp = _row_break_mpo(Lx, Ly, S.pos_sites; which=:xplus)   # zeros ix = Nx-1
    brk_xn = _row_break_mpo(Lx, Ly, S.pos_sites; which=:xplain)  # zeros ix = 0

    # ── Intra-cell: 3×3 bond matrix (A=1, B=2, C=3) ──────────────────────────
    # t_AB: A-B bond,  t_AC: A-C bond,  t_BC: B-C bond
    H_intra = postpend_op(S.Id, S.sub_s,
        Float64[0 t_AB t_AC; t_AB 0 t_BC; t_AC t_BC 0])

    # ── Inter-cell x: B(n) ↔ A(n+1), shift ±1 — uses t_AB ───────────────────
    H_x = _sublattice_bond(S, 1, t_AB, (1, 2); brk=brk_xp)

    # ── Inter-cell y: C(n) ↔ A(n+Nx), shift ±Nx — uses t_AC ─────────────────
    H_y = _sublattice_bond(S, S.Nx, t_AC, (1, 3))

    # ── Inter-cell diagonal: C(n) ↔ B(n+Nx-1), shift ±(Nx-1) — uses t_BC ────
    H_d = _sublattice_bond(S, S.Nx - 1, t_BC, (2, 3); brk=brk_xn)

    # ── Assembly ───────────────────────────────────────────────────────────────
    H_total = sum_mpos((H_intra, H_x, H_y, H_d); cutoff=cutoff)
    ITensorMPS.truncate!(H_total; maxdim=maxdim, cutoff=cutoff)

    scale = 4.5 * max(abs(t_AB), abs(t_AC), abs(t_BC))
    return TBHamiltonian(; L=S.L, N=S.N, sites=S.all_sites, mpo=H_total, scale,
                         sublattice_s=S.sub_s, aux_side=:post)
end


# ============================================================
# 3. Lieb lattice
# ============================================================

"""
    lieb_hamiltonian(Lx, Ly, t=1.0; t_AB=t, t_AC=t, cutoff=1e-8, maxdim=200) -> TBHamiltonian

Build a Lieb tight-binding Hamiltonian as a `TBHamiltonian`.

**Encoding** (L+1 sites, L = Lx+Ly):
- Sites 1…L : position qubits for 2^L unit cells on a square Bravais lattice
- Site  L+1 : dim-3 "Lieb" sublattice index A=1 (corner), B=2 (x-edge), C=3 (y-edge)

**Bond amplitudes**

| Kwarg | Bond | Intra-cell | Inter-cell direction    |
|-------|------|------------|-------------------------|
| `t_AB`| A↔B  | yes        | x  (shift ±1)           |
| `t_AC`| A↔C  | yes        | y  (shift ±Nx)          |

No B-C bond exists (corner connects to edges only).  Both default to `t`.
```julia
H = lieb_hamiltonian(Lx, Ly; t_AB=1.0, t_AC=0.5)  # anisotropic Lieb
```

**Flat band** at E=0; dispersive bands at ±2√(t_AB²+t_AC²)/√2 (approx ±2t uniform).
Real-space coordinates: `lieb_positions(Lx, Ly)`.
`H.sublattice_s` stores the dim-3 index; `H.aux_side = :post`.
"""
function lieb_hamiltonian(Lx::Integer, Ly::Integer, t::Number = 1.0;
                           t_AB::Number = t,
                           t_AC::Number = t,
                           cutoff::Real = 1e-8,
                           maxdim::Int  = 200)
    S = _sublattice_setup(Lx, Ly, 3, "Lieb"; cutoff, maxdim)

    brk_xp = _row_break_mpo(Lx, Ly, S.pos_sites; which=:xplus)

    # ── Intra-cell: A↔B (t_AB) and A↔C (t_AC) ───────────────────────────────
    H_intra = postpend_op(S.Id, S.sub_s,
        Float64[0 t_AB t_AC; t_AB 0 0; t_AC 0 0])

    # ── Inter-cell x: B(n) ↔ A(n+1), shift ±1 — uses t_AB ───────────────────
    H_x = _sublattice_bond(S, 1, t_AB, (1, 2); brk=brk_xp)

    # ── Inter-cell y: C(n) ↔ A(n+Nx), shift ±Nx — uses t_AC ─────────────────
    H_y = _sublattice_bond(S, S.Nx, t_AC, (1, 3))

    # ── Assembly ───────────────────────────────────────────────────────────────
    H_total = sum_mpos((H_intra, H_x, H_y); cutoff=cutoff)
    ITensorMPS.truncate!(H_total; maxdim=maxdim, cutoff=cutoff)

    scale = 2.5 * max(abs(t_AB), abs(t_AC))
    return TBHamiltonian(; L=S.L, N=S.N, sites=S.all_sites, mpo=H_total, scale,
                         sublattice_s=S.sub_s, aux_side=:post)
end


# ============================================================
# 4. Honeycomb lattice (NN, and NN + NNN)
# ============================================================

"""
    honeycomb_sublattice_hamiltonian(Lx, Ly, t=1.0; cutoff=1e-8, maxdim=200) -> TBHamiltonian

Build a uniform honeycomb tight-binding Hamiltonian with an explicit 2-component
sublattice index, as a `TBHamiltonian`.

**Encoding** (L+1 sites total, L = Lx+Ly):
- Sites 1…L : L position qubits for 2^L unit cells on a triangular Bravais lattice
              (row-major: n = ix + iy·2^Lx)
- Site  L+1 : dim-2 "Honeycomb" sublattice index (A=1, B=2), postpended

**Hopping structure** (uniform amplitude `t`):

*Intra-cell* — one bond per unit cell:
  A-B  (same unit cell)

*Inter-cell*:
  x (shift +1 ): B(n) ↔ A(n+1)  — break at ix=Nx-1
  y (shift +Nx): B(n) ↔ A(n+Nx) — no x-break needed (pure y step)

The spectrum has two Dirac cones touching at E=0 (gapless for uniform t).
Use `honeycomb_sublattice_positions(Lx, Ly)` for real-space atom coordinates.
The sublattice index is stored in `H.sublattice_s`; `H.aux_side = :post`.
"""
function honeycomb_sublattice_hamiltonian(Lx::Integer, Ly::Integer, t::Number = 1.0;
                                           cutoff::Real = 1e-8,
                                           maxdim::Int  = 200)
    S = _sublattice_setup(Lx, Ly, 2, "Honeycomb"; cutoff, maxdim)

    brk_xp = _row_break_mpo(Lx, Ly, S.pos_sites; which=:xplus)

    # ── Intra-cell: A↔B within the same unit cell ────────────────────────────
    H_intra = postpend_op(S.Id, S.sub_s, t * Float64[0 1; 1 0])

    # ── Inter-cell x: B(n) ↔ A(n+1), shift ±1 ───────────────────────────────
    # Break suppresses B(Nx-1) ↔ A(0) wrap-around across row boundary
    H_x = _sublattice_bond(S, 1, t, (1, 2); brk=brk_xp)

    # ── Inter-cell y: B(n) ↔ A(n+Nx), shift ±Nx ─────────────────────────────
    H_y = _sublattice_bond(S, S.Nx, t, (1, 2))

    # ── Assembly ───────────────────────────────────────────────────────────────
    H_total = sum_mpos((H_intra, H_x, H_y); cutoff=cutoff)
    ITensorMPS.truncate!(H_total; maxdim=maxdim, cutoff=cutoff)

    # Honeycomb spectrum: Dirac bands at ±3t bandwidth
    scale = 3.5 * abs(t)
    return TBHamiltonian(; L=S.L, N=S.N, sites=S.all_sites, mpo=H_total, scale,
                         sublattice_s=S.sub_s, aux_side=:post)
end


"""
    honeycomb_nnn_hamiltonian(Lx, Ly, t=1.0, t2=0.0; cutoff=1e-8, maxdim=200) -> TBHamiltonian

Build a honeycomb tight-binding Hamiltonian with both nearest-neighbor (NN)
and next-nearest-neighbor (NNN) hopping, as a `TBHamiltonian`.

Encoding is identical to `honeycomb_sublattice_hamiltonian`: L+1 sites, with
the last site being the dim-2 sublattice index (A=1, B=2, postpended).

**Hopping structure**

*NN* (amplitude `t`): same three bonds as `honeycomb_sublattice_hamiltonian`
(intra-cell A↔B, x-shift B↔A, y-shift B↔A).

*NNN* (amplitude `t2`): connects same-sublattice atoms along the three
triangular Bravais directions.  The sublattice operator is the 2×2 identity
(both A↔A and B↔B hop with the same amplitude `t2`):

- x-direction (shift ±1):          A(n) ↔ A(n±1),  B(n) ↔ B(n±1)
- y-direction (shift ±Nx):         A(n) ↔ A(n±Nx), B(n) ↔ B(n±Nx)
- diagonal (shift ±(1−Nx)):        A(n) ↔ A(n±(1−Nx)), same for B

`t2` may be complex; `conj(t2)` is used for the backward hop so that the
Hamiltonian is Hermitian.  For Haldane-type NNN (sublattice-dependent phases)
construct the NN and NNN terms manually.
"""
function honeycomb_nnn_hamiltonian(Lx::Integer, Ly::Integer,
                                   t::Number = 1.0, t2::Number = 0.0;
                                   cutoff::Real = 1e-8,
                                   maxdim::Int  = 200)
    S = _sublattice_setup(Lx, Ly, 2, "Honeycomb"; cutoff, maxdim)

    brk_xp = _row_break_mpo(Lx, Ly, S.pos_sites; which=:xplus)

    # ── NN terms (same as honeycomb_sublattice_hamiltonian) ───────────────────
    H_intra = postpend_op(S.Id, S.sub_s, t * Float64[0 1; 1 0])
    H_x     = _sublattice_bond(S, 1,    t, (1, 2); brk=brk_xp)
    H_y     = _sublattice_bond(S, S.Nx, t, (1, 2))

    # ── NNN terms: sublattice matrix = I₂ (A↔A and B↔B with same amplitude) ──
    I2 = Float64[1 0; 0 1]

    # ±a₁ (x-direction, shift ±1)
    H_nnn_x = _sublattice_bond(S, 1, t2, I2; brk=brk_xp)

    # ±a₂ (y-direction, shift ±Nx)
    H_nnn_y = _sublattice_bond(S, S.Nx, t2, I2)

    # ±(a₁−a₂) (diagonal, shift +(1−Nx) and −(1−Nx))
    H_nnn_d = _sublattice_bond(S, 1 - S.Nx, t2, I2; brk=brk_xp)

    # ── Assembly ───────────────────────────────────────────────────────────────
    H_total = sum_mpos((H_intra, H_x, H_y, H_nnn_x, H_nnn_y, H_nnn_d); cutoff=cutoff)
    ITensorMPS.truncate!(H_total; maxdim=maxdim, cutoff=cutoff)

    scale = 3.5 * abs(t) + 3.5 * abs(t2)
    return TBHamiltonian(; L=S.L, N=S.N, sites=S.all_sites, mpo=H_total, scale,
                         sublattice_s=S.sub_s, aux_side=:post)
end


# ============================================================
# 5. Dice (T3) lattice
# ============================================================

"""
    dice_hamiltonian(Lx, Ly, t=1.0; t_AB=t, t_AC=t, cutoff=1e-8, maxdim=200) -> TBHamiltonian

Build a dice (T3) tight-binding Hamiltonian as a `TBHamiltonian`.

**Encoding** (L+1 sites, L = Lx+Ly):
- Sites 1…L : position qubits for 2^L unit cells on a triangular Bravais lattice
- Site  L+1 : dim-3 "Dice" sublattice index A=1 (hub), B=2 (rim), C=3 (rim)

**Bond amplitudes**

The hub A has coordination 6 (three B neighbors, three C neighbors).
Each kwarg controls all bonds of that type (intra- and inter-cell):

| Kwarg | Bond | Intra-cell | Inter-cell directions              |
|-------|------|------------|-------------------------------------|
| `t_AB`| A↔B  | yes        | x (shift ±1), y (shift ±Nx)        |
| `t_AC`| A↔C  | no         | x, y, diagonal (shift ±(Nx+1))     |

Both default to `t` (uniform dice).
```julia
H = dice_hamiltonian(Lx, Ly; t_AB=1.0, t_AC=0.7)  # hub-to-B ≠ hub-to-C
```

**Spectrum**: doubly degenerate flat band at E=0; dispersive bands reaching ±3t.
Real-space coordinates: `dice_positions(Lx, Ly)`.
`H.sublattice_s` stores the dim-3 index; `H.aux_side = :post`.
"""
function dice_hamiltonian(Lx::Integer, Ly::Integer, t::Number = 1.0;
                           t_AB::Number = t,
                           t_AC::Number = t,
                           cutoff::Real = 1e-8,
                           maxdim::Int  = 200)
    S = _sublattice_setup(Lx, Ly, 3, "Dice"; cutoff, maxdim)

    brk_xp = _row_break_mpo(Lx, Ly, S.pos_sites; which=:xplus)   # zeros ix = Nx-1

    # ── Intra-cell: A↔B only (t_AB); no A-C intra-cell bond ─────────────────
    H_intra = postpend_op(S.Id, S.sub_s,
        Float64[0 t_AB 0; t_AB 0 0; 0 0 0])

    # ── Inter-cell x: B(n) ↔ A(n+1) (t_AB) and C(n) ↔ A(n+1) (t_AC), shift ±1
    H_xB = _sublattice_bond(S, 1, t_AB, (1, 2); brk=brk_xp)
    H_xC = _sublattice_bond(S, 1, t_AC, (1, 3); brk=brk_xp)

    # ── Inter-cell y: B(n) ↔ A(n+Nx) (t_AB) and C(n) ↔ A(n+Nx) (t_AC), shift ±Nx
    H_yB = _sublattice_bond(S, S.Nx, t_AB, (1, 2))
    H_yC = _sublattice_bond(S, S.Nx, t_AC, (1, 3))

    # ── Inter-cell diagonal: C(n) ↔ A(n+Nx+1) (t_AC), shift ±(Nx+1) ─────────
    H_dC = _sublattice_bond(S, S.Nx + 1, t_AC, (1, 3); brk=brk_xp)

    # ── Assembly ───────────────────────────────────────────────────────────────
    H_total = sum_mpos((H_intra, H_xB, H_xC, H_yB, H_yC, H_dC); cutoff=cutoff)
    ITensorMPS.truncate!(H_total; maxdim=maxdim, cutoff=cutoff)

    scale = 4.5 * max(abs(t_AB), abs(t_AC))
    return TBHamiltonian(; L=S.L, N=S.N, sites=S.all_sites, mpo=H_total, scale,
                         sublattice_s=S.sub_s, aux_side=:post)
end


# ============================================================
# 6. SSH chain with explicit sublattice index
# ============================================================

"""
    ssh_sublattice_hamiltonian(L, t=1.0, d=0.0; cutoff=1e-8, maxdim=200) -> TBHamiltonian

Build an SSH (Su-Schrieffer-Heeger) tight-binding Hamiltonian with an explicit
2-component sublattice index, as a `TBHamiltonian`.

**Encoding** (L+1 sites total):
- Sites 1…L : L position qubits for 2^L unit cells
- Site  L+1 : dim-2 "SSH" sublattice index (A=1, B=2), postpended

**Hopping structure**:
- *Intra-cell* (amplitude `t+d`): A↔B within each unit cell
- *Inter-cell* (amplitude `t-d`): B(n) ↔ A(n+1)

**Geometry** (unit cell width = 1, 1-indexed site `i` over `2·2^L` atoms):
- A atom in unit cell `n = (i-1)÷2`: position `[n]`
- B atom in unit cell `n`: position `[n + 0.5]`

`geometry_uc` returns `[n]` for every atom in unit cell `n` (same for A and B).

The chain has periodic boundary conditions (B(N-1) ↔ A(0) inter-cell bond from
the binary-increment wrap-around), consistent with all other QTT Hamiltonians.
"""
function ssh_sublattice_hamiltonian(L::Integer, t::Number = 1.0, d::Number = 0.0;
                                    cutoff::Real = 1e-8,
                                    maxdim::Int  = 200)
    # A 1D chain: the setup of a 2^L × 2^0 grid.
    S = _sublattice_setup(L, 0, 2, "SSH"; cutoff, maxdim)

    t1 = t + d   # intra-cell hopping amplitude
    t2 = t - d   # inter-cell hopping amplitude

    # Intra-cell: A(n) ↔ B(n) — Hermitian matrix [0 t1; conj(t1) 0]
    H_intra = postpend_op(S.Id, S.sub_s, ComplexF64[0 t1; conj(t1) 0])

    # Inter-cell: B(n) ↔ A(n+1), i.e. K_u ⊗ |A⟩⟨B| + K_d ⊗ |B⟩⟨A|, with the
    # periodic binary increment K_u (the B(N-1) ↔ A(0) bond wraps around)
    H_inter = _sublattice_bond(S, 1, t2, (1, 2); cyclic=true)

    H_total = sum_mpos((H_intra, H_inter); cutoff=cutoff)
    ITensorMPS.truncate!(H_total; maxdim=maxdim, cutoff=cutoff)

    scale = (abs(t1) + abs(t2)) * 1.1

    geom_f    = let
        i -> [Float64(div(i - 1, 2)) + 0.5 * ((i - 1) % 2)]
    end
    geom_uc_f = let
        i -> [Float64(div(i - 1, 2))]
    end

    return TBHamiltonian(; L=S.L, N=S.N, sites=S.all_sites, mpo=H_total, geometry=geom_f,
                         geometry_uc=geom_uc_f, scale, sublattice_s=S.sub_s, aux_side=:post)
end
