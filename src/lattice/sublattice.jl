# sublattice.jl — lattices with an explicit sublattice index (kagome, Lieb,
# honeycomb, honeycomb NNN, dice/T3, SSH chain): the *_hamiltonian builders,
# each returning a TBHamiltonian with the sublattice site postpended, and the
# matching *_positions tables. Split from lattice/2Dlattice_tk.jl.

# ============================================================
# 1. Kagome lattice
# ============================================================

"""
    kagome_positions(Lx, Ly) -> Matrix{Float64}

Return the (3·2^L × 2) real-space atom-position matrix for a kagomé lattice
of 2^Lx × 2^Ly unit cells (L = Lx+Ly), consistent with the MPO site ordering.

For total 1-indexed site i:
  n_cell  = (i-1) ÷ 3          (0-indexed unit cell, row-major)
  s       = (i-1) % 3 + 1      (sublattice: A=1, B=2, C=3)
  ix = n_cell % Nx,  iy = n_cell ÷ Nx

Atom positions (lattice vectors a₁=(1,0), a₂=(½,√3/2)):
  A: (ix + iy/2,        iy·√3/2       )
  B: (ix + iy/2 + ½,    iy·√3/2       )
  C: (ix + iy/2 + ¼,    iy·√3/2 + √3/4)
"""
function kagome_positions(Lx::Int, Ly::Int)
    Nx    = 2^Lx
    N_uc  = 2^(Lx + Ly)
    rs    = Matrix{Float64}(undef, 3 * N_uc, 2)
    sq3_2 = sqrt(3) / 2
    sq3_4 = sqrt(3) / 4
    for n in 0:N_uc-1
        ix   = n % Nx
        iy   = div(n, Nx)
        ax   = ix + iy * 0.5
        ay   = iy * sq3_2
        base = 3n + 1
        rs[base,   :] = [ax,        ay        ]   # A
        rs[base+1, :] = [ax + 0.5,  ay        ]   # B
        rs[base+2, :] = [ax + 0.25, ay + sq3_4]   # C
    end
    return rs
end


"""
    kagome_hamiltonian(Lx, Ly[, t]; t_AB, t_AC, t_BC, cutoff, maxdim) -> TBHamiltonian

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
    Nx = 2^Lx
    L  = Lx + Ly
    N  = 2^L

    pos_sites = siteinds("Qubit", L)
    kag_s     = Index(3, "Kagome")
    all_sites = [pos_sites; kag_s]

    Id   = MPO(pos_sites, "Id")
    apkw = (; cutoff = cutoff, maxdim = maxdim)

    brk_xp = _row_break_mpo(Lx, Ly, pos_sites; which=:xplus)   # zeros ix = Nx-1
    brk_xn = _row_break_mpo(Lx, Ly, pos_sites; which=:xplain)  # zeros ix = 0

    # ── Intra-cell: 3×3 bond matrix (A=1, B=2, C=3) ──────────────────────────
    # t_AB: A-B bond,  t_AC: A-C bond,  t_BC: B-C bond
    H_intra = postpend_op(Id, kag_s,
        Float64[0 t_AB t_AC; t_AB 0 t_BC; t_AC t_BC 0])

    # ── Inter-cell x: B(n) ↔ A(n+1), shift ±1 — uses t_AB ───────────────────
    K_x = shift_mpo(pos_sites, 1; cyclic=false)
    D_x = shift_adjoint_mpo(K_x)
    H_x = +(t_AB        * postpend_op(apply(K_x, brk_xp;  apkw...), kag_s, 1, 2),
             conj(t_AB) * postpend_op(apply(brk_xp, D_x; apkw...), kag_s, 2, 1); cutoff=cutoff)

    # ── Inter-cell y: C(n) ↔ A(n+Nx), shift ±Nx — uses t_AC ─────────────────
    ku_y = shift_mpo(pos_sites, Nx; cyclic=false)
    kd_y = shift_adjoint_mpo(ku_y)
    H_y  = +(t_AC        * postpend_op(ku_y, kag_s, 1, 3),
              conj(t_AC) * postpend_op(kd_y, kag_s, 3, 1); cutoff=cutoff)

    # ── Inter-cell diagonal: C(n) ↔ B(n+Nx-1), shift ±(Nx-1) — uses t_BC ────
    ku_d = shift_mpo(pos_sites, Nx - 1; cyclic=false)
    kd_d = shift_adjoint_mpo(ku_d)
    H_d  = +(t_BC        * postpend_op(apply(ku_d, brk_xn; apkw...), kag_s, 2, 3),
              conj(t_BC) * postpend_op(apply(brk_xn, kd_d; apkw...), kag_s, 3, 2); cutoff=cutoff)

    # ── Assembly ───────────────────────────────────────────────────────────────
    H_total = +(H_intra, H_x;    cutoff=cutoff)
    H_total = +(H_total, H_y;    cutoff=cutoff)
    H_total = +(H_total, H_d;    cutoff=cutoff)
    ITensorMPS.truncate!(H_total; maxdim=maxdim, cutoff=cutoff)

    scale = 4.5 * max(abs(t_AB), abs(t_AC), abs(t_BC))
    return TBHamiltonian(L, N, all_sites, H_total, nothing, scale, 0.0,
                         nothing, nothing, nothing, kag_s, :post, nothing, nothing, 0, nothing)
end


# ============================================================
# 2. Lieb lattice
# ============================================================

"""
    lieb_positions(Lx, Ly) -> Matrix{Float64}

Return the (3·2^L × 2) real-space atom-position matrix for a Lieb lattice
of 2^Lx × 2^Ly unit cells (L = Lx+Ly), consistent with the MPO site ordering.

For total 1-indexed site i:
  n_cell  = (i-1) ÷ 3          (0-indexed unit cell, row-major)
  s       = (i-1) % 3 + 1      (sublattice: A=1, B=2, C=3)
  ix = n_cell % Nx,  iy = n_cell ÷ Nx

Atom positions (lattice vectors a₁=(1,0), a₂=(0,1)):
  A: (ix,       iy      )   corner
  B: (ix + 0.5, iy      )   x-edge center
  C: (ix,       iy + 0.5)   y-edge center
"""
function lieb_positions(Lx::Int, Ly::Int)
    Nx   = 2^Lx
    N_uc = 2^(Lx + Ly)
    rs   = Matrix{Float64}(undef, 3 * N_uc, 2)
    for n in 0:N_uc-1
        ix   = n % Nx
        iy   = div(n, Nx)
        base = 3n + 1
        rs[base,   :] = [ix,       iy       ]   # A
        rs[base+1, :] = [ix + 0.5, iy       ]   # B
        rs[base+2, :] = [ix,       iy + 0.5 ]   # C
    end
    return rs
end


"""
    lieb_hamiltonian(Lx, Ly[, t]; t_AB, t_AC, cutoff, maxdim) -> TBHamiltonian

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
    Nx = 2^Lx
    L  = Lx + Ly
    N  = 2^L

    pos_sites = siteinds("Qubit", L)
    lieb_s    = Index(3, "Lieb")
    all_sites = [pos_sites; lieb_s]

    Id   = MPO(pos_sites, "Id")
    apkw = (; cutoff = cutoff, maxdim = maxdim)

    brk_xp = _row_break_mpo(Lx, Ly, pos_sites; which=:xplus)

    # ── Intra-cell: A↔B (t_AB) and A↔C (t_AC) ───────────────────────────────
    H_intra = postpend_op(Id, lieb_s,
        Float64[0 t_AB t_AC; t_AB 0 0; t_AC 0 0])

    # ── Inter-cell x: B(n) ↔ A(n+1), shift ±1 — uses t_AB ───────────────────
    K_x = shift_mpo(pos_sites, 1; cyclic=false)
    D_x = shift_adjoint_mpo(K_x)
    H_x = +(t_AB        * postpend_op(apply(K_x, brk_xp;  apkw...), lieb_s, 1, 2),
             conj(t_AB) * postpend_op(apply(brk_xp, D_x; apkw...), lieb_s, 2, 1); cutoff=cutoff)

    # ── Inter-cell y: C(n) ↔ A(n+Nx), shift ±Nx — uses t_AC ─────────────────
    ku_y = shift_mpo(pos_sites, Nx; cyclic=false)
    kd_y = shift_adjoint_mpo(ku_y)
    H_y  = +(t_AC        * postpend_op(ku_y, lieb_s, 1, 3),
              conj(t_AC) * postpend_op(kd_y, lieb_s, 3, 1); cutoff=cutoff)

    # ── Assembly ───────────────────────────────────────────────────────────────
    H_total = +(H_intra, H_x;    cutoff=cutoff)
    H_total = +(H_total, H_y;    cutoff=cutoff)
    ITensorMPS.truncate!(H_total; maxdim=maxdim, cutoff=cutoff)

    scale = 2.5 * max(abs(t_AB), abs(t_AC))
    return TBHamiltonian(L, N, all_sites, H_total, nothing, scale, 0.0,
                         nothing, nothing, nothing, lieb_s, :post, nothing, nothing, 0, nothing)
end


# ============================================================
# 3. Honeycomb sublattice lattice
# ============================================================

"""
    honeycomb_sublattice_positions(Lx, Ly) -> Matrix{Float64}

Return the (2·2^L × 2) real-space atom-position matrix for a honeycomb
lattice of 2^Lx × 2^Ly unit cells (L = Lx+Ly), consistent with the MPO
site ordering.

For total 1-indexed site i:
  n_cell = (i-1) ÷ 2          (0-indexed unit cell, row-major)
  s      = (i-1) % 2 + 1      (sublattice: A=1, B=2)
  ix = n_cell % Nx,  iy = n_cell ÷ Nx

Atom positions (triangular Bravais vectors a₁=(1,0), a₂=(½,√3/2)):
  A: (ix + iy/2,       iy·√3/2          )
  B: (ix + iy/2 + ½,   iy·√3/2 + √3/6  )   displaced along the intra-cell bond
"""
function honeycomb_sublattice_positions(Lx::Int, Ly::Int)
    Nx    = 2^Lx
    N_uc  = 2^(Lx + Ly)
    rs    = Matrix{Float64}(undef, 2 * N_uc, 2)
    sq3_2 = sqrt(3) / 2
    sq3_6 = sqrt(3) / 6
    for n in 0:N_uc-1
        ix   = n % Nx
        iy   = div(n, Nx)
        ax   = ix + iy * 0.5
        ay   = iy * sq3_2
        base = 2n + 1
        rs[base,   :] = [ax,        ay         ]   # A
        rs[base+1, :] = [ax + 0.5,  ay + sq3_6 ]   # B
    end
    return rs
end


"""
    honeycomb_sublattice_hamiltonian(Lx, Ly[, t]; cutoff, maxdim) -> TBHamiltonian

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
    Nx = 2^Lx
    L  = Lx + Ly
    N  = 2^L

    pos_sites = siteinds("Qubit", L)
    hc_s      = Index(2, "Honeycomb")
    all_sites = [pos_sites; hc_s]

    Id   = MPO(pos_sites, "Id")
    apkw = (; cutoff = cutoff, maxdim = maxdim)

    brk_xp = _row_break_mpo(Lx, Ly, pos_sites; which=:xplus)

    # ── Intra-cell: A↔B within the same unit cell ────────────────────────────
    H_intra = postpend_op(Id, hc_s, t * Float64[0 1; 1 0])

    # ── Inter-cell x: B(n) ↔ A(n+1), shift ±1 ───────────────────────────────
    # Break suppresses B(Nx-1) ↔ A(0) wrap-around across row boundary
    K_x = shift_mpo(pos_sites, 1; cyclic=false)
    D_x = shift_adjoint_mpo(K_x)
    H_x = +(t        * postpend_op(apply(K_x, brk_xp;  apkw...), hc_s, 1, 2),
             conj(t) * postpend_op(apply(brk_xp, D_x; apkw...), hc_s, 2, 1); cutoff=cutoff)

    # ── Inter-cell y: B(n) ↔ A(n+Nx), shift ±Nx ─────────────────────────────
    ku_y = shift_mpo(pos_sites, Nx; cyclic=false)
    kd_y = shift_adjoint_mpo(ku_y)
    H_y  = +(t        * postpend_op(ku_y, hc_s, 1, 2),
              conj(t) * postpend_op(kd_y, hc_s, 2, 1); cutoff=cutoff)

    # ── Assembly ───────────────────────────────────────────────────────────────
    H_total = +(H_intra, H_x;    cutoff=cutoff)
    H_total = +(H_total, H_y;    cutoff=cutoff)
    ITensorMPS.truncate!(H_total; maxdim=maxdim, cutoff=cutoff)

    # Honeycomb spectrum: Dirac bands at ±3t bandwidth
    scale = 3.5 * abs(t)
    return TBHamiltonian(L, N, all_sites, H_total, nothing, scale, 0.0,
                         nothing, nothing, nothing, hc_s, :post, nothing, nothing, 0, nothing)
end


"""
    honeycomb_nnn_hamiltonian(Lx, Ly[, t[, t2]]; cutoff, maxdim) -> TBHamiltonian

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
    Nx = 2^Lx
    L  = Lx + Ly
    N  = 2^L

    pos_sites = siteinds("Qubit", L)
    hc_s      = Index(2, "Honeycomb")
    all_sites = [pos_sites; hc_s]

    Id   = MPO(pos_sites, "Id")
    apkw = (; cutoff = cutoff, maxdim = maxdim)

    brk_xp = _row_break_mpo(Lx, Ly, pos_sites; which=:xplus)

    K_x = shift_mpo(pos_sites, 1; cyclic=false)
    D_x = shift_adjoint_mpo(K_x)
    ku_y = shift_mpo(pos_sites, Nx; cyclic=false)
    kd_y = shift_adjoint_mpo(ku_y)

    # ── NN terms (same as honeycomb_sublattice_hamiltonian) ───────────────────
    H_intra = postpend_op(Id, hc_s, t * Float64[0 1; 1 0])

    H_x = +(t        * postpend_op(apply(K_x, brk_xp;  apkw...), hc_s, 1, 2),
             conj(t) * postpend_op(apply(brk_xp, D_x; apkw...), hc_s, 2, 1); cutoff=cutoff)

    H_y = +(t        * postpend_op(ku_y, hc_s, 1, 2),
             conj(t) * postpend_op(kd_y, hc_s, 2, 1); cutoff=cutoff)

    # ── NNN terms: sublattice matrix = I₂ (A↔A and B↔B with same amplitude) ──
    I2 = Float64[1 0; 0 1]

    # ±a₁ (x-direction, shift ±1)
    H_nnn_x = +(t2        * postpend_op(apply(K_x, brk_xp;  apkw...), hc_s, I2),
                conj(t2)  * postpend_op(apply(brk_xp, D_x; apkw...), hc_s, I2); cutoff=cutoff)

    # ±a₂ (y-direction, shift ±Nx)
    H_nnn_y = +(t2        * postpend_op(ku_y, hc_s, I2),
                conj(t2)  * postpend_op(kd_y, hc_s, I2); cutoff=cutoff)

    # ±(a₁−a₂) (diagonal, shift +(1−Nx) and −(1−Nx))
    K_diag = shift_mpo(pos_sites, 1 - Nx; cyclic=false)
    D_diag = shift_adjoint_mpo(K_diag)
    K_fwd = apply(K_diag, brk_xp; apkw...)
    K_bwd = apply(brk_xp, D_diag; apkw...)
    H_nnn_d = +(t2        * postpend_op(K_fwd, hc_s, I2),
                conj(t2)  * postpend_op(K_bwd, hc_s, I2); cutoff=cutoff)

    # ── Assembly ───────────────────────────────────────────────────────────────
    H_total = +(H_intra, H_x;     cutoff=cutoff)
    H_total = +(H_total, H_y;     cutoff=cutoff)
    H_total = +(H_total, H_nnn_x; cutoff=cutoff)
    H_total = +(H_total, H_nnn_y; cutoff=cutoff)
    H_total = +(H_total, H_nnn_d; cutoff=cutoff)
    ITensorMPS.truncate!(H_total; maxdim=maxdim, cutoff=cutoff)

    scale = 3.5 * abs(t) + 3.5 * abs(t2)
    return TBHamiltonian(L, N, all_sites, H_total, nothing, scale, 0.0,
                         nothing, nothing, nothing, hc_s, :post, nothing, nothing, 0, nothing)
end


# ============================================================
# 4. Dice (T3) lattice
# ============================================================

"""
    dice_positions(Lx, Ly) -> Matrix{Float64}

Return the (3·2^L × 2) real-space atom-position matrix for a dice (T3)
lattice of 2^Lx × 2^Ly unit cells (L = Lx+Ly), consistent with the MPO
site ordering.

For total 1-indexed site i:
  n_cell = (i-1) ÷ 3          (0-indexed unit cell, row-major)
  s      = (i-1) % 3 + 1      (sublattice: A=1 hub, B=2 rim, C=3 rim)
  ix = n_cell % Nx,  iy = n_cell ÷ Nx

Atom positions (triangular Bravais vectors a₁=(1,0), a₂=(½,√3/2)):
  A: (ix + iy/2,        iy·√3/2        )   at 0·(a₁+a₂)/3
  B: (ix + iy/2 + ½,    iy·√3/2 + √3/6)   at 1·(a₁+a₂)/3
  C: (ix + iy/2 + 1,    iy·√3/2 + √3/3)   at 2·(a₁+a₂)/3
"""
function dice_positions(Lx::Int, Ly::Int)
    Nx    = 2^Lx
    N_uc  = 2^(Lx + Ly)
    rs    = Matrix{Float64}(undef, 3 * N_uc, 2)
    sq3_2 = sqrt(3) / 2
    sq3_6 = sqrt(3) / 6
    sq3_3 = sqrt(3) / 3
    for n in 0:N_uc-1
        ix   = n % Nx
        iy   = div(n, Nx)
        ax   = ix + iy * 0.5
        ay   = iy * sq3_2
        base = 3n + 1
        rs[base,   :] = [ax,        ay        ]   # A: origin
        rs[base+1, :] = [ax + 0.5,  ay + sq3_6]   # B: (a1+a2)/3
        rs[base+2, :] = [ax + 1.0,  ay + sq3_3]   # C: 2(a1+a2)/3
    end
    return rs
end


"""
    dice_hamiltonian(Lx, Ly[, t]; t_AB, t_AC, cutoff, maxdim) -> TBHamiltonian

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
    Nx = 2^Lx
    L  = Lx + Ly
    N  = 2^L

    pos_sites = siteinds("Qubit", L)
    dice_s    = Index(3, "Dice")
    all_sites = [pos_sites; dice_s]

    Id   = MPO(pos_sites, "Id")
    apkw = (; cutoff = cutoff, maxdim = maxdim)

    brk_xp = _row_break_mpo(Lx, Ly, pos_sites; which=:xplus)   # zeros ix = Nx-1

    # ── Intra-cell: A↔B only (t_AB); no A-C intra-cell bond ─────────────────
    H_intra = postpend_op(Id, dice_s,
        Float64[0 t_AB 0; t_AB 0 0; 0 0 0])

    # ── Inter-cell x: B(n) ↔ A(n+1) (t_AB) and C(n) ↔ A(n+1) (t_AC), shift ±1
    K_x = shift_mpo(pos_sites, 1; cyclic=false)
    D_x = shift_adjoint_mpo(K_x)
    H_xB = +(t_AB        * postpend_op(apply(K_x, brk_xp;  apkw...), dice_s, 1, 2),
              conj(t_AB) * postpend_op(apply(brk_xp, D_x; apkw...), dice_s, 2, 1); cutoff=cutoff)
    H_xC = +(t_AC        * postpend_op(apply(K_x, brk_xp;  apkw...), dice_s, 1, 3),
              conj(t_AC) * postpend_op(apply(brk_xp, D_x; apkw...), dice_s, 3, 1); cutoff=cutoff)

    # ── Inter-cell y: B(n) ↔ A(n+Nx) (t_AB) and C(n) ↔ A(n+Nx) (t_AC), shift ±Nx
    ku_y = shift_mpo(pos_sites, Nx; cyclic=false)
    kd_y = shift_adjoint_mpo(ku_y)
    H_yB = +(t_AB        * postpend_op(ku_y, dice_s, 1, 2),
              conj(t_AB) * postpend_op(kd_y, dice_s, 2, 1); cutoff=cutoff)
    H_yC = +(t_AC        * postpend_op(ku_y, dice_s, 1, 3),
              conj(t_AC) * postpend_op(kd_y, dice_s, 3, 1); cutoff=cutoff)

    # ── Inter-cell diagonal: C(n) ↔ A(n+Nx+1) (t_AC), shift ±(Nx+1) ─────────
    ku_d = shift_mpo(pos_sites, Nx + 1; cyclic=false)
    kd_d = shift_adjoint_mpo(ku_d)
    H_dC = +(t_AC        * postpend_op(apply(ku_d, brk_xp; apkw...), dice_s, 1, 3),
              conj(t_AC) * postpend_op(apply(brk_xp, kd_d; apkw...), dice_s, 3, 1); cutoff=cutoff)

    # ── Assembly ───────────────────────────────────────────────────────────────
    H_total = +(H_intra, H_xB;  cutoff=cutoff)
    H_total = +(H_total, H_xC;  cutoff=cutoff)
    H_total = +(H_total, H_yB;  cutoff=cutoff)
    H_total = +(H_total, H_yC;  cutoff=cutoff)
    H_total = +(H_total, H_dC;  cutoff=cutoff)
    ITensorMPS.truncate!(H_total; maxdim=maxdim, cutoff=cutoff)

    scale = 4.5 * max(abs(t_AB), abs(t_AC))
    return TBHamiltonian(L, N, all_sites, H_total, nothing, scale, 0.0,
                         nothing, nothing, nothing, dice_s, :post, nothing, nothing, 0, nothing)
end


# ============================================================
# 5. SSH chain with explicit sublattice index
# ============================================================

"""
    ssh_sublattice_hamiltonian(L[, t[, d]]; cutoff, maxdim) -> TBHamiltonian

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
    N  = 2^L

    pos_sites = siteinds("Qubit", L)
    ssh_s     = Index(2, "SSH")
    all_sites = [pos_sites; ssh_s]

    t1 = t + d   # intra-cell hopping amplitude
    t2 = t - d   # inter-cell hopping amplitude

    ku = generate_kin_u(pos_sites, N)
    kd = generate_kin_d(pos_sites, N)
    Id = MPO(pos_sites, "Id")

    # Intra-cell: A(n) ↔ B(n) — Hermitian matrix [0 t1; conj(t1) 0]
    H_intra = postpend_op(Id, ssh_s, ComplexF64[0 t1; conj(t1) 0])

    # Inter-cell: B(n) ↔ A(n+1), i.e. K_u ⊗ |A⟩⟨B| + K_d ⊗ |B⟩⟨A|
    H_inter = +(t2       * postpend_op(ku, ssh_s, 1, 2),
                conj(t2) * postpend_op(kd, ssh_s, 2, 1); cutoff=cutoff)

    H_total = +(H_intra, H_inter; cutoff=cutoff)
    ITensorMPS.truncate!(H_total; maxdim=maxdim, cutoff=cutoff)

    scale = (abs(t1) + abs(t2)) * 1.1

    geom_f    = let
        i -> [Float64(div(i - 1, 2)) + 0.5 * ((i - 1) % 2)]
    end
    geom_uc_f = let
        i -> [Float64(div(i - 1, 2))]
    end

    return TBHamiltonian(L, N, all_sites, H_total, geom_f, geom_uc_f, scale, 0.0,
                         nothing, nothing, nothing, ssh_s, :post, nothing, nothing, 0, nothing)
end
