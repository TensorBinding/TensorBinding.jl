# geometry.jl — real-space geometry of the lattice models: the i -> position
# closures that preset Hamiltonians store in H.geometry, the *_positions tables of
# the preset Bravais lattices, of the explicit-sublattice lattices (kagome, Lieb,
# honeycomb, dice) and of the T/Y junction, and the dispatchers that pick one.
#
# Entry points: lattice_positions, square_positions, triangular_positions,
#   triangular_bravais_positions, honeycomb_positions, kagome_positions,
#   lieb_positions, honeycomb_sublattice_positions, dice_positions,
#   tjunction_positions.
# Internals: the _*_geometry closures (the model registry in core/ModelRegistry.jl
#   assigns one to each preset; _preset_geometry reads it there); _closure_positions and
#   _basis_positions, which tabulate a closure or a Bravais lattice with a basis;
#   _geom_positions; _resolve_2d_geometry, for add_hopping_2D! (lattice/NNNeighbor.jl).
#
# Depends on: core/TBSystem.jl (TBHamiltonian).
#
# Gathered, unchanged, from core/TBSystem.jl, lattice/sublattice.jl, the interim
# lattice/model_registry.jl (now core/ModelRegistry.jl), lattice/Twisted.jl,
# lattice/TJunction.jl and lattice/NNNeighbor.jl. The geometry_uc closures stay
# inline in the builders that set them. Tier 2 folded the eight position loops of
# sections 2 and 3 into _closure_positions and _basis_positions (same values, bit
# for bit).

# ============================================================
# 1. Geometry closures for the preset lattices
# ============================================================

# Each closure maps a 1-indexed site i to its position vector.

_chain_geometry() = i -> Float64[i]

function _square_geometry(Nx)
    return i -> Float64[(i-1) % Nx, (i-1) ÷ Nx]
end

function _tri_geometry(Nx)
    function pos(i)
        ix = (i-1) % Nx
        iy = (i-1) ÷ Nx
        x  = Float64(ix) + 0.5 * (iy % 2)
        y  = iy * sqrt(3) / 2
        return Float64[x, y]
    end
    return pos
end

function _tri_bravais_geometry(Nx)
    function pos(i)
        ix = (i-1) % Nx
        iy = (i-1) ÷ Nx
        x  = Float64(ix) + 0.5 * iy
        y  = iy * sqrt(3) / 2
        return Float64[x, y]
    end
    return pos
end

function _hex_geometry(Nx)
    function pos(i)
        ix = (i-1) % Nx
        iy = (i-1) ÷ Nx
        x  = 3.0*(ix÷2) + Float64(ix%2) + (iy%2) * (Float64(ix%2) - 0.5)
        y  = iy * sqrt(3)/2
        return Float64[x, y]
    end
    return pos
end


# ============================================================
# 2. Position tables for the preset lattices
# ============================================================

# The N × 2 table of a position closure `g` (row i = g(i)); the tables below are
# the preset closures of section 1 evaluated at every site.
function _closure_positions(g, N::Int)
    rs = Matrix{Float64}(undef, N, 2)
    for i in 1:N; rs[i, :] = g(i); end
    return rs
end

"""
    honeycomb_positions(L; Lx=L÷2) -> Matrix{Float64}

Generate `N = 2^L` physical honeycomb positions consistent with the
quantics row-major encoding `n = ix + iy * 2^Lx`, bond length = 1.

The lattice is an armchair ribbon: even rows have intra-row bonds
`(2k, 2k+1)` and odd rows have intra-row bonds `(2k+1, 2k+2)`, with
all inter-row bonds `(iy, ix) ↔ (iy+1, ix)`.

Returns an `N × 2` matrix where row `i` (1-indexed) is the 2D position
of quantics site `i-1`.
"""
honeycomb_positions(L::Int; Lx::Int = L ÷ 2) = _closure_positions(_hex_geometry(2^Lx), 2^L)

"""
    square_positions(L; Lx=L÷2) -> Matrix{Float64}

Physical positions for the `2^L`-site square lattice in quantics row-major
encoding `n = ix + iy·2^Lx`.  Site `i` (1-indexed) maps to `(ix, iy)`.
"""
square_positions(L::Int; Lx::Int = L ÷ 2) = _closure_positions(_square_geometry(2^Lx), 2^L)

"""
    triangular_positions(L; Lx=L÷2) -> Matrix{Float64}

Physical positions for the `2^L`-site triangular lattice in quantics row-major
encoding `n = ix + iy·2^Lx`, bond length = 1.  Odd rows are offset by 0.5 in x:
`x = ix + 0.5·(iy % 2)`,  `y = iy·√3/2`.
"""
triangular_positions(L::Int; Lx::Int = L ÷ 2) = _closure_positions(_tri_geometry(2^Lx), 2^L)

"""
    triangular_bravais_positions(L; Lx=L÷2) -> Matrix{Float64}

Physical positions for the `2^L`-site Bravais triangular lattice in quantics
row-major encoding `n = ix + iy·2^Lx`, bond length = 1.
Bravais vectors a1=(1,0), a2=(1/2,√3/2):  `x = ix + iy/2`,  `y = iy·√3/2`.
"""
triangular_bravais_positions(L::Int; Lx::Int = L ÷ 2) =
    _closure_positions(_tri_bravais_geometry(2^Lx), 2^L)


# ============================================================
# 3. Position tables for the explicit-sublattice lattices
# ============================================================

"""
    _basis_positions(Lx, Ly, a1, a2, basis) -> Matrix{Float64}

Atom positions of `2^Lx × 2^Ly` unit cells with an `n_sub`-atom basis, one row per
atom in the MPO order of the sublattice builders: atom `i = n_sub·n + s` (1-indexed)
is basis atom `s` of cell `n = ix + iy·2^Lx`, at `ix·a1 + iy·a2 + basis[s]`
(`a1`, `a2` the Bravais vectors, `basis` the offsets).
"""
function _basis_positions(Lx::Int, Ly::Int, a1, a2, basis)
    Nx    = 2^Lx
    N_uc  = 2^(Lx + Ly)
    n_sub = length(basis)
    rs    = Matrix{Float64}(undef, n_sub * N_uc, 2)
    for n in 0:N_uc-1
        ix = n % Nx
        iy = div(n, Nx)
        ax = ix * a1[1] + iy * a2[1]
        ay = ix * a1[2] + iy * a2[2]
        for (s, b) in enumerate(basis)
            rs[n_sub * n + s, :] = [ax + b[1], ay + b[2]]
        end
    end
    return rs
end

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
    sq3_2 = sqrt(3) / 2
    sq3_4 = sqrt(3) / 4
    return _basis_positions(Lx, Ly, (1.0, 0.0), (0.5, sq3_2),
                            ((0.0,  0.0  ),     # A
                             (0.5,  0.0  ),     # B
                             (0.25, sq3_4)))    # C
end


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
    return _basis_positions(Lx, Ly, (1.0, 0.0), (0.0, 1.0),
                            ((0.0, 0.0),     # A
                             (0.5, 0.0),     # B
                             (0.0, 0.5)))    # C
end


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
    sq3_2 = sqrt(3) / 2
    sq3_6 = sqrt(3) / 6
    return _basis_positions(Lx, Ly, (1.0, 0.0), (0.5, sq3_2),
                            ((0.0, 0.0  ),     # A
                             (0.5, sq3_6)))    # B
end


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
    sq3_2 = sqrt(3) / 2
    sq3_6 = sqrt(3) / 6
    sq3_3 = sqrt(3) / 3
    return _basis_positions(Lx, Ly, (1.0, 0.0), (0.5, sq3_2),
                            ((0.0, 0.0  ),     # A: origin
                             (0.5, sq3_6),     # B: (a1+a2)/3
                             (1.0, sq3_3)))    # C: 2(a1+a2)/3
end


# ============================================================
# 4. Position tables by lattice name (for spatial LDOS plots)
# ============================================================

_geom_positions(::Val{:honeycomb}, Lx, Ly) = honeycomb_sublattice_positions(Lx, Ly)
_geom_positions(::Val{:kagome},    Lx, Ly) = kagome_positions(Lx, Ly)
_geom_positions(::Val{:lieb},      Lx, Ly) = lieb_positions(Lx, Ly)
_geom_positions(::Val{:dice},      Lx, Ly) = dice_positions(Lx, Ly)


# ============================================================
# 5. Real-space lattice positions (optionally rotated)
# ============================================================

"""
    lattice_positions(lattice, Lx, Ly; angle_deg=0.0) -> Matrix{Float64}

Return an N×2 matrix of real-space positions for a 2^Lx × 2^Ly patch of
`lattice ∈ {:square, :triangular, :honeycomb}`, optionally rotated by
`angle_deg` degrees about the geometric centroid of the lattice.

Site ordering matches the quantics row-major encoding: `n = ix + iy·2^Lx` (0-indexed).
"""
function lattice_positions(lattice::Symbol, Lx::Int, Ly::Int;
                           angle_deg::Real = 0.0)
    L = Lx + Ly
    rs = if lattice === :square
        square_positions(L; Lx=Lx)
    elseif lattice === :triangular
        triangular_positions(L; Lx=Lx)
    elseif lattice === :honeycomb
        honeycomb_positions(L; Lx=Lx)
    else
        error("Unknown lattice :$lattice.  Choose :square, :triangular, or :honeycomb.")
    end

    if !iszero(angle_deg)
        θ      = angle_deg * π / 180
        c, s   = cos(θ), sin(θ)
        R      = [c -s; s c]
        center = vec(sum(rs, dims=1)) / size(rs, 1)
        rs     = Matrix{Float64}(((R * (rs' .- center)) .+ center)')
    end
    return rs
end


# ============================================================
# 6. T/Y-junction positions
# ============================================================

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


# ============================================================
# 7. Geometry resolution for add_hopping_2D!
# ============================================================

function _resolve_2d_geometry(H::TBHamiltonian, lattice, geometry, Lx::Int, Ly::Int)
    if geometry !== nothing
        if geometry isa AbstractMatrix
            n_geom = H.sublattice_s === nothing ? H.N : dim(H.sublattice_s) * H.N
            size(geometry, 1) == n_geom ||
                error("geometry matrix has $(size(geometry, 1)) rows, expected $n_geom.")
            return let m = Float64.(geometry)
                i -> m[i, :]
            end
        elseif geometry isa Function
            return geometry
        else
            error("geometry must be a function i -> r_i or an Nxd matrix.")
        end
    end

    H.geometry !== nothing && return H.geometry

    lattice !== nothing ||
        error("Layered add_hopping_2D! needs `lattice=:square/:triangular/:honeycomb` " *
              "or `geometry=...` because H.geometry is not set.")
    lat = lattice isa Symbol ? lattice : Symbol(lattice)
    rs = H.sublattice_s === nothing ? lattice_positions(lat, Lx, Ly) :
         lat === :honeycomb         ? honeycomb_sublattice_positions(Lx, Ly) :
         error("lattice=:$lat with H.sublattice_s is not supported by add_hopping_2D!.")
    n_geom = H.sublattice_s === nothing ? H.N : dim(H.sublattice_s) * H.N
    size(rs, 1) == n_geom ||
        error("geometry for :$lat returned $(size(rs, 1)) sites, expected $n_geom.")
    return let m = Float64.(rs)
        i -> m[i, :]
    end
end
