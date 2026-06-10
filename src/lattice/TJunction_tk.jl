# TJunction_tk.jl — T/Y-junction Hamiltonians via a spin-1 (dim-3) chain-label index
#
# A T-junction (Y-junction) connects three 1D chains at a common site.
# The spin-1 index (dim-3) postpended to the MPO labels the branch:
#   state 1 ↔ |-1⟩ branch
#   state 2 ↔ | 0⟩ branch
#   state 3 ↔ |+1⟩ branch
#
# All three branches share the same intra-chain Hamiltonian.  The junction
# coupling is a 3×3 operator acting at a designated junction site.
#
# Site ordering after add_tjunction!:
#   [pos_qubits..., tj_s]    (tj_s = Index(3, "TJunction"))
#
# Atom index convention for H.geometry (matches kagome — branch-fast):
#   combined 1-indexed atom i:
#     n = (i-1) ÷ 3   (0-indexed chain position, 0…N-1)
#     s = (i-1) % 3 + 1   (branch: 1=|-1⟩, 2=|0⟩, 3=|+1⟩)
#   Total atoms: 3*N where N = 2^L.
#
# Depends on: utils.jl, Hamiltonian.jl, TBSystem.jl


# ─────────────────────────────────────────────────────────────────
# 1.  TJunction site index
# ─────────────────────────────────────────────────────────────────

"""
    tjunction_index() -> Index

Create a dim-3 Index tagged "TJunction" for a Y/T-junction of three chains.
States: 1 = branch |-1⟩, 2 = branch |0⟩, 3 = branch |+1⟩.
"""
tjunction_index() = Index(3, "TJunction")


# ─────────────────────────────────────────────────────────────────
# 2.  Real-space positions
# ─────────────────────────────────────────────────────────────────

"""
    tjunction_positions(N, junction_site) -> Matrix{Float64}

Return a `(3N × 2)` real-space position matrix for a Y-junction of three
chains, each of length `N`, meeting at `junction_site` (0-indexed).

**Atom index convention** (branch-fast, matching kagome):
  atom `i` (1-indexed):
    `n = (i-1) ÷ 3`      — 0-indexed chain position (0…N-1)
    `s = (i-1) % 3 + 1`  — branch (1=|-1⟩, 2=|0⟩, 3=|+1⟩)

**Branch directions** (branches radiate symmetrically from the junction):
  branch 1: angle 0°
  branch 2: angle 120°
  branch 3: angle 240°

The junction site is placed at the origin; position along each branch is
the signed distance `n − junction_site` (positive = away from junction).
"""
function tjunction_positions(N::Int, junction_site::Int)
    rs = Matrix{Float64}(undef, 3 * N, 2)
    for i in 1:3*N
        n   = (i - 1) ÷ 3           # 0-indexed chain position
        s   = (i - 1) % 3 + 1       # branch (1, 2, 3)
        θ   = (s - 1) * 2π / 3      # branch angle: 0°, 120°, 240°
        d   = Float64(n - junction_site)   # signed distance from junction
        rs[i, :] = [d * cos(θ), d * sin(θ)]
    end
    return rs
end


# ─────────────────────────────────────────────────────────────────
# 3.  Internal helper: exact site-projector MPO
# ─────────────────────────────────────────────────────────────────

# Build the rank-1 projector |n><n| for 0-indexed site n on L position qubits.
# Site ordering: pos_sites[1] = MSB, pos_sites[L] = LSB.
# Returns a bond-dim-1 MPO (product of single-qubit projectors); no QTCI needed.
function _site_projector_mpo(L::Int, pos_sites, n::Int)
    0 <= n < 2^L ||
        error("Site index n=$n is out of range [0, $(2^L - 1)].")
    b0 = (n >> (L - 1)) & 1
    os  = OpSum()
    os += 1, b0 == 1 ? "sigma_d" : "sigma_u", 1
    for k in 2:L
        b  = (n >> (L - k)) & 1
        op = b == 1 ? "sigma_d" : "sigma_u"
        os *= 1, op, k
    end
    return MPO(os, pos_sites)
end


# ─────────────────────────────────────────────────────────────────
# 4.  add_tjunction! — in-place extension of an existing TBHamiltonian
# ─────────────────────────────────────────────────────────────────

"""
    add_tjunction!(H, t_j; junction_site=0, coupling=nothing,
                   cutoff=1e-8, maxdim=200) -> H

Extend a 1D `TBHamiltonian` to a T/Y-junction by postpending a dim-3
"TJunction" index.  The three states |1⟩≡|-1⟩, |2⟩≡|0⟩, |3⟩≡|+1⟩ label
three identical branches; all share the same intra-chain Hamiltonian.

A junction coupling is added at `junction_site` (0-indexed):

    H_TJ = I₃ ⊗ H_chain  +  J(t_j) ⊗ |junction_site⟩⟨junction_site|

The default 3×3 coupling matrix is the symmetric Y-junction:

    J = t_j * [0 1 1 ; 1 0 1 ; 1 1 0]

Pass a custom 3×3 matrix via `coupling` to override (e.g., for a linear
T-junction where only two of the three branches are mutually coupled).

**Requirements**: `H` must have no existing auxiliary indices
(spin, Nambu, layer, sublattice).

**Geometry**: `H.geometry` is set to a 2D Y-junction layout with the three
branches radiating at 0°, 120°, 240° from the origin (junction site).
See `tjunction_positions` for the atom index convention.

After the call, `H.sublattice_s` stores the TJunction index and
`H.aux_side = :post`.  All caches are invalidated.

Examples
--------
```julia
H = get_Hamiltonian("chain_1d", 1.0; L=8)

# Symmetric Y-junction at the left end (site 0)
add_tjunction!(H, 0.5)

# Junction at the right end (site N-1)
H2 = get_Hamiltonian("chain_1d", 1.0; L=8)
add_tjunction!(H2, 0.5; junction_site=H2.N - 1)

# Linear T-junction: branch 1 ↔ 2 and 2 ↔ 3, but not 1 ↔ 3
H3 = get_Hamiltonian("chain_1d", 1.0; L=8)
add_tjunction!(H3, 0.5; coupling=[0 1 0; 1 0 1; 0 1 0])
```
"""
function add_tjunction!(H::TBHamiltonian, t_j;
                        junction_site::Int = 0,
                        coupling           = nothing,
                        cutoff::Real       = 1e-8,
                        maxdim::Int        = 200)
    isnothing(H.sublattice_s) && isnothing(H.layer_s) &&
        isnothing(H.spin_s)   && isnothing(H.nambu_s) ||
        error("add_tjunction! requires H with no existing auxiliary indices " *
              "(spin, Nambu, layer, or sublattice).")

    0 <= junction_site < H.N ||
        error("junction_site=$junction_site is out of range [0, $(H.N-1)].")

    pos_s = _pos_sites(H)   # all L position-qubit indices
    tj_s  = tjunction_index()

    # --- Intra-chain: three identical copies of H via I₃ ----------------------
    H_intra = postpend_op(H.mpo, tj_s, Matrix{Float64}(I, 3, 3))

    # --- Junction coupling: J_mat ⊗ |junction_site><junction_site| ------------
    P_junc = _site_projector_mpo(H.L, pos_s, junction_site)
    J_mat  = isnothing(coupling) ?
             t_j * [0.0 1.0 1.0; 1.0 0.0 1.0; 1.0 1.0 0.0] :
             coupling
    H_junc = postpend_op(P_junc, tj_s, J_mat)

    # --- Geometry: 2D Y-junction, branches at 0°, 120°, 240° -----------------
    # Atom index convention (branch-fast):
    #   i (1-indexed) → n = (i-1)÷3 (chain pos, 0-indexed), s = (i-1)%3+1 (branch)
    #   position: d = n - junction_site (signed distance from junction along branch)
    rs = tjunction_positions(H.N, junction_site)
    H.geometry = let m = rs; i -> m[i, :]; end

    # --- Assemble and update struct -------------------------------------------
    H.mpo          = +(H_intra, H_junc; cutoff=cutoff, maxdim=maxdim)
    ITensorMPS.truncate!(H.mpo; maxdim=maxdim, cutoff=cutoff)
    H.sites        = [pos_s; tj_s]
    H.sublattice_s = tj_s
    H.aux_side     = :post
    _invalidate_cache!(H)
    return H
end


# ─────────────────────────────────────────────────────────────────
# 5.  tjunction_hamiltonian — standalone constructor
# ─────────────────────────────────────────────────────────────────

"""
    tjunction_hamiltonian(L, t, t_j; junction_site=0, coupling=nothing,
                          boundary=:open, cutoff=1e-8, maxdim=200)
        -> TBHamiltonian

Build a T/Y-junction Hamiltonian: three identical 1D nearest-neighbour chains
with hopping amplitude `t` meeting at a common junction site.

**Encoding** (`L+1` sites total):
- Sites 1…L : `L` position qubits (chain length = `2^L` per branch)
- Site  L+1 : dim-3 "TJunction" index (postpended), stored in `H.sublattice_s`

**Chain-label states** (in `H.sublattice_s`):
  1 ↔ branch |-1⟩,  2 ↔ branch |0⟩,  3 ↔ branch |+1⟩

**Geometry** (`H.geometry`): 2D Y-junction with branches at 0°, 120°, 240°
from the junction site (placed at the origin).  See `tjunction_positions`.

**Arguments**
- `L`            : number of position qubits (chain length = `2^L` per branch)
- `t`            : intra-chain NN hopping amplitude
- `t_j`          : junction coupling amplitude

**Keyword arguments**
- `junction_site` : 0-indexed site where the chains meet (default `0`, left end)
- `coupling`      : custom 3×3 coupling matrix (default: symmetric Y-junction
                    `t_j * [0 1 1; 1 0 1; 1 1 0]`)
- `boundary`      : `:open` (default) or `:periodic` for the intra-chain hoppings
- `cutoff`        : MPO truncation cutoff
- `maxdim`        : maximum MPO bond dimension

Examples
--------
```julia
# Symmetric Y-junction, junction at left end (site 0)
H = tjunction_hamiltonian(8, 1.0, 0.5)

# Junction at the right end (site 2^L - 1)
H = tjunction_hamiltonian(8, 1.0, 0.5; junction_site=255)

# Linear T-junction (branches 1↔2 and 2↔3 only)
H = tjunction_hamiltonian(8, 1.0, 0.5; coupling=[0 1 0; 1 0 1; 0 1 0])
```
"""
function tjunction_hamiltonian(L::Int, t::Number, t_j::Number;
                                junction_site::Int = 0,
                                coupling           = nothing,
                                boundary::Symbol   = :open,
                                cutoff::Real       = 1e-8,
                                maxdim::Int        = 200)
    H = get_Hamiltonian("chain_1d", t; L=L, boundary=boundary)
    add_tjunction!(H, t_j;
                   junction_site = junction_site,
                   coupling      = coupling,
                   cutoff        = cutoff,
                   maxdim        = maxdim)
    # Rough spectral scale: chain bandwidth + junction contribution
    j_scale = isnothing(coupling) ? abs(t_j) : maximum(abs.(coupling))
    H.scale = 2.5 * abs(t) + 2.0 * j_scale
    return H
end


# ─────────────────────────────────────────────────────────────────
# 6.  tjunction_lattice_hamiltonian — triangular lattice of T-junctions
# ─────────────────────────────────────────────────────────────────

"""
    tjunction_lattice_hamiltonian(Lx, Ly, L, t, t_j, t_inter;
                                   junction_site=0, coupling=nothing,
                                   boundary=:open, cutoff=1e-8, maxdim=200)
        -> TBHamiltonian

Build a triangular lattice of T/Y-junctions with `2^Lx × 2^Ly` unit cells,
where each unit cell is a T-junction with three arms of `2^L` sites.

**Site ordering** (`Lx + Ly + L + 1` sites total):
- Sites 1..Ly          : iy bits of the triangular lattice (MSB first)
- Sites Ly+1..Ly+Lx    : ix bits of the triangular lattice (MSB first)
- Sites Ly+Lx+1..Ly+Lx+L : chain position qubits within each arm (MSB first)
- Site  Ly+Lx+L+1      : dim-3 TJunction sublattice index (branch 1/2/3)

**Hamiltonian structure:**
1. *Intra-arm*: nearest-neighbor hopping `t` along each chain.
2. *Junction coupling*: `t_j * [0 1 1; 1 0 1; 1 1 0]` at `junction_site`
   (acts on the branch index at the shared chain site).
3. *Inter-cell*: hopping `t_inter` between the far end (`2^L - 1`) of one arm
   and the far end of the appropriate arm in the adjacent unit cell:
   - x-shift    (-1)  : arm 1 (330°) ↔ arm 3 (210°)
   - y-shift    (+Nx) : arm 3 (210°) ↔ arm 2 (90°)
   - diagonal (+Nx-1) : arm 2 (90°)  ↔ arm 1 (330°)

**Arguments**
- `Lx`, `Ly`   : log₂ of unit cell count along x and y
- `L`          : log₂ of sites per arm
- `t`          : intra-arm NN hopping amplitude
- `t_j`        : junction coupling amplitude
- `t_inter`    : inter-unit-cell hopping amplitude at arm ends

**Keyword arguments**
- `junction_site` : 0-indexed junction position within each arm (default `0`)
- `coupling`      : custom 3×3 coupling matrix (overrides `t_j` default)
- `boundary`      : `:open` or `:periodic` for intra-arm hopping
- `cutoff`, `maxdim` : MPO truncation parameters

`H.sublattice_s` stores the dim-3 TJunction index; `H.aux_side = :post`.
`H.L = Lx + Ly + L`, `H.N = 2^(Lx+Ly) * 2^L` (unit_cells × arm_length).

Examples
--------
```julia
# 4×4 triangular lattice of T-junctions, arms of length 8
H = tjunction_lattice_hamiltonian(2, 2, 3, 1.0, 0.5, 0.8)

# Weaker inter-cell coupling
H = tjunction_lattice_hamiltonian(2, 2, 3, 1.0, 0.5, 0.3; junction_site=0)
```
"""
function tjunction_lattice_hamiltonian(Lx::Int, Ly::Int, L::Int,
                                        t::Number, t_j::Number, t_inter::Number;
                                        junction_site::Int = 0,
                                        coupling           = nothing,
                                        boundary::Symbol   = :open,
                                        cutoff::Real       = 1e-8,
                                        maxdim::Int        = 200)
    Nx     = 2^Lx
    Llat   = Lx + Ly
    Nchain = 2^L
    Nlat   = 2^Llat

    # site indices: [iy_bits(Ly)..., ix_bits(Lx)..., chain(L)..., tj_s]
    lat_sites   = siteinds("Qubit", Llat)
    chain_sites = siteinds("Qubit", L)
    tj_s        = tjunction_index()
    all_sites   = [lat_sites; chain_sites; tj_s]

    I2   = Float64[1 0; 0 1]
    I3   = Matrix{Float64}(I, 3, 3)
    apkw = (; cutoff=cutoff, maxdim=maxdim)

    # extend MPO on chain_sites to [lat_sites..., chain_sites] via prepend
    extend_chain(m) = (for ls in reverse(lat_sites); m = prepend_op(m, ls, I2); end; m)

    # extend MPO on lat_sites to [lat_sites..., chain_sites] via postpend
    extend_lat(m)   = (for cs in chain_sites;        m = postpend_op(m, cs, I2); end; m)

    # ── 1. Intra-arm hopping: I_lat ⊗ (t * H_chain) ⊗ I_3 ───────────────────
    H_chain     = t * kinetic_1d_nn(L, chain_sites; boundary=boundary)
    H_chain_ext = postpend_op(extend_chain(H_chain), tj_s, I3)

    # ── 2. Junction coupling: I_lat ⊗ P_junc ⊗ J_mat ────────────────────────
    P_junc = _site_projector_mpo(L, chain_sites, junction_site)
    J_mat  = isnothing(coupling) ?
             t_j * [0.0 1.0 1.0; 1.0 0.0 1.0; 1.0 1.0 0.0] :
             coupling
    H_junc_ext = postpend_op(extend_chain(P_junc), tj_s, J_mat)

    # ── 3. Inter-cell coupling: t_inter * K_d ⊗ P_end ⊗ |s><s| ─────────────
    # End-of-arm projector extended to [lat..., chain]
    P_end_lc = extend_chain(_site_projector_mpo(L, chain_sites, Nchain - 1))

    # Direct shifts on lat_sites with exact per-cell valid-source masking.
    # Composing binary shift MPOs (as was done before) leaks carry-bits across
    # row boundaries and generates spurious intra-cell hoppings.
    Id_lat     = MPO(lat_sites, "Id")
    brk_xp_lat = _row_break_mpo(Lx, Ly, lat_sites; which=:xplus)
    K1 = _shift_mpo(-1,  0, Id_lat, Id_lat, Id_lat, brk_xp_lat, Nx; apkw=apkw)
    K2 = _shift_mpo( 0,  1, Id_lat, Id_lat, Id_lat, brk_xp_lat, Nx; apkw=apkw)
    K3 = _shift_mpo(-1,  1, Id_lat, Id_lat, Id_lat, brk_xp_lat, Nx; apkw=apkw)
    D1 = _shift_mpo( 1,  0, Id_lat, Id_lat, Id_lat, brk_xp_lat, Nx; apkw=apkw)
    D2 = _shift_mpo( 0, -1, Id_lat, Id_lat, Id_lat, brk_xp_lat, Nx; apkw=apkw)
    D3 = _shift_mpo( 1, -1, Id_lat, Id_lat, Id_lat, brk_xp_lat, Nx; apkw=apkw)

    # inter_hop: forward hop is K_f ⊗ P_end ⊗ |arm_dst><arm_src|; h.c. uses K_b
    function inter_hop(Kf, Kb, arm_dst, arm_src)
        KPf = apply(extend_lat(Kf), P_end_lc; apkw...)
        KPb = apply(extend_lat(Kb), P_end_lc; apkw...)
        H_f = t_inter        * postpend_op(KPf, tj_s, arm_dst, arm_src)
        H_b = conj(t_inter)  * postpend_op(KPb, tj_s, arm_src, arm_dst)
        return +(H_f, H_b; cutoff=cutoff)
    end

    # ── 4. Assembly ───────────────────────────────────────────────────────────
    # Inter-cell bond assignment (arm angles 330°, 90°, 210°):
    #   x-shift    (-1)   : arm 3 (210°) [src] → arm 1 (330°) [dst at ix-1]
    #   y-shift    (+Nx)  : arm 2 (90°)  [src] → arm 3 (210°) [dst at iy+1]
    #   diagonal (+Nx-1)  : arm 2 (90°)  [src] → arm 1 (330°) [dst at ix-1,iy+1]
    H_total = +(H_chain_ext, H_junc_ext;             cutoff=cutoff)
    H_total = +(H_total,     inter_hop(K1, D1, 1, 3); cutoff=cutoff)
    H_total = +(H_total,     inter_hop(K2, D2, 3, 2); cutoff=cutoff)
    H_total = +(H_total,     inter_hop(K3, D3, 1, 2); cutoff=cutoff)
    ITensorMPS.truncate!(H_total; maxdim=maxdim, cutoff=cutoff)

    # ── 5. Geometry ───────────────────────────────────────────────────────────
    # ITensors column-major ordering (tj_s fastest, then chain LSB, ..., lat LSB, ...):
    #   k = i - 1
    #   tj_idx  = k % 3                     (0,1,2 → branch 1,2,3)
    #   chain_n = (k ÷ 3) % Nchain         (0-indexed chain position)
    #   lat_n   = k ÷ (3 * Nchain)         (= ix + iy * Nx, 0-indexed unit cell)
    #
    # Unit cell on triangular Bravais lattice (a1=(1,0), a2=(1/2,√3/2)):
    #   r_cell = [ix + iy*0.5, iy*√3/2]
    # Arm at angle θ = tj_idx * 2π/3, scaled so arm end ≈ 0.45 lattice units from cell:
    arm_scale = 0.45 / max(1, Nchain - 1 - junction_site)
    sq3_2     = sqrt(3) / 2

    rs = Matrix{Float64}(undef, 3 * Nchain * Nlat, 2)
    for i in 1:(3 * Nchain * Nlat)
        k        = i - 1
        tj_idx   = k % 3
        chain_n  = (k ÷ 3) % Nchain
        lat_n    = k ÷ (3 * Nchain)
        ix       = lat_n % Nx
        iy       = lat_n ÷ Nx
        θ        = [330.0, 90.0, 210.0][tj_idx + 1] * π / 180.0
        d        = (Float64(chain_n - junction_site) + 0.5) * arm_scale
        rs[i, :] = [ix + iy * 0.5, iy * sq3_2] + d * [cos(θ), sin(θ)]
    end

    j_scale = isnothing(coupling) ? abs(t_j) : maximum(abs.(coupling))
    scale   = 2.5 * abs(t) + 2.0 * j_scale + 2.0 * abs(t_inter)

    return TBHamiltonian(Llat + L, Nlat * Nchain, all_sites, H_total,
                         let m = rs; i -> m[i, :]; end, nothing, scale, 0.0,
                         nothing, nothing, nothing, tj_s, :post,
                         nothing, nothing, 0, nothing)
end
