# kpath.jl — High-symmetry k-path utilities for 2D get_bands
#
# Contains kpath_2d, hsk_honeycomb / hsk_square / hsk_triangular, _hs_label,
# _hsk and kpath_setup.  Split from the former physics/QFT_tk.jl; the overview
# and file map of physics/qft/ are at the top of bands.jl.
#
# Entry points: kpath_setup, kpath_2d, hsk_honeycomb, hsk_square, hsk_triangular.
# Depends on: nothing else in the package.
#
# The preferred user interface is the kpath kwarg in get_bands(H::TBHamiltonian):
#   res = get_bands(H, Ncheb, 2, omega;
#                   kpath=[:G, :M, :Kp, :G], kpath_lattice=:honeycomb, num_x=30)
#   # res.Ak — (Nω × Nk) spectral function
#   # res.ticks, res.labels — ready for xticks=(res.ticks, res.labels)

# ============================================================
# 1. High-symmetry k-points and explicit paths (2D)
# ============================================================
#
#   kpath_2d(hs_points, Lx; npts_per_segment)
#       Takes explicit (kx_idx, ky_idx) tuples, interpolates between them.
#       Returns (k_groups, tick_positions) for use with k_groups_override.
#
#   hsk_honeycomb / hsk_square / hsk_triangular(Lx, Ly)
#       Named-tuple dictionaries of standard high-symmetry k-points in
#       quantics integer units (0-based; verified analytically for honeycomb,
#       approximate for triangular).  Use Latin symbol names (G, M, K, Kp, X)
#       — no non-ASCII input required.

"""
    kpath_2d(hs_points, Lx; npts_per_segment=20) -> (k_groups, tick_positions)

Build k-groups for `get_bands(H, Ncheb, 2, ω; k_groups_override=k_groups)`
sampling along a high-symmetry path.

`hs_points` is an ordered list of `(kx_idx, ky_idx)` integer tuples defining
the path vertices (0-based, kx_idx ∈ [0, 2^Lx−1]).  `npts_per_segment` points
are linearly interpolated between each consecutive pair of vertices.

Returns:
- `k_groups`      : `Vector{Vector{Int}}` — single-element groups in the
  `(ky << Lx) | kx` linear-index format expected by `get_bands`.
  Pass as `k_groups_override`.
- `tick_positions` : 1-based indices into `k_groups` at each vertex;
  use as x-axis tick positions when plotting.

```julia
# Low-level: explicit tuples
hs = hsk_honeycomb(Lx, Ly)
kg, ticks = kpath_2d([hs.G, hs.M, hs.Kp, hs.G], Lx; npts_per_segment=30)
Ak = get_bands(H, Ncheb, 2, omega; k_groups_override=kg)

# High-level shortcut (preferred):
res = get_bands(H, Ncheb, 2, omega;
                kpath=[:G, :M, :Kp, :G], kpath_lattice=:honeycomb, num_x=30)
# heatmap(1:size(res.Ak,2), omega, res.Ak; xticks=(res.ticks, res.labels))
```
"""
function kpath_2d(hs_points, Lx::Int; npts_per_segment::Int = 20)
    k_list   = Int[]
    tick_pos = Int[]
    for seg in 1:length(hs_points)-1
        kx1, ky1 = hs_points[seg]
        kx2, ky2 = hs_points[seg+1]
        push!(tick_pos, length(k_list) + 1)
        for t in range(0, 1; length = npts_per_segment + 1)[1:end-1]
            kx = round(Int, kx1 + t * (kx2 - kx1))
            ky = round(Int, ky1 + t * (ky2 - ky1))
            push!(k_list, (ky << Lx) | kx)
        end
    end
    push!(tick_pos, length(k_list) + 1)   # final vertex
    kx, ky = hs_points[end]
    push!(k_list, (ky << Lx) | kx)
    return [[k] for k in k_list], tick_pos
end


"""
    hsk_honeycomb(Lx, Ly) -> NamedTuple

High-symmetry k-points for the honeycomb sublattice Hamiltonian
(`honeycomb_sublattice_hamiltonian`) in quantics integer units
(kx_idx ∈ [0, 2^Lx−1], ky_idx ∈ [0, 2^Ly−1]).

**Derivation.**  The Bloch off-diagonal element is
    h(k) = t ( 1 + e^{2πi kx/Nx} + e^{2πi ky/Ny} )
Dirac points h=0 require θx = 2π/3 AND θy = 4π/3 (or their conjugates):
    K  : (kx_idx, ky_idx) = (Nx/3, 2Ny/3)  →  Cartesian (2π/3, 2π/√3)
    K' : (kx_idx, ky_idx) = (2Nx/3, Ny/3)  →  Cartesian (4π/3, 0)

Because Nx = 2^Lx is never divisible by 3, the K/K' indices are rounded
to the nearest integer.  Use a large Lx (≥4) for a good approximation.

M is the edge midpoint adjacent to K' along the kx axis:
    M  : (Nx/2, Ny/4)  →  Cartesian (π, 0)   — |h|=1, saddle point

| Point | Symbol | Cartesian (b₁/b₂ frame)  | Quantics index             |
|-------|--------|--------------------------|----------------------------|
| Γ     | `G`    | (0, 0)                   | (0, 0)                     |
| M     | `M`    | (π, 0)                   | (Nx÷2, Ny÷4)              |
| K     | `K`    | (2π/3, 2π/√3)            | (round(Nx/3), round(2Ny/3))|
| K'    | `Kp`   | (4π/3, 0)                | (round(2Nx/3), round(Ny/3))|

Standard G–M–Kp–G path (along the kx direction, K' corner at (4π/3,0)):
```julia
hs = hsk_honeycomb(Lx, Ly)
kg, ticks = kpath_2d([hs.G, hs.M, hs.Kp, hs.G], Lx; npts_per_segment=30)
Ak = get_bands(H, Ncheb, 2, omega; k_groups_override=kg)
# or using the kpath shortcut:
res = get_bands(H, Ncheb, 2, omega; kpath=[:G, :M, :Kp, :G],
                kpath_lattice=:honeycomb, num_x=30)
```
"""
function hsk_honeycomb(Lx::Int, Ly::Int)
    Nx, Ny = 2^Lx, 2^Ly
    return (
        G  = (0,                       0                      ),   # Gamma — zone centre
        M  = (Nx ÷ 2,                  Ny ÷ 4                 ),   # (π,  0)         Cartesian
        K  = (round(Int, Nx / 3),      round(Int, 2Ny / 3)    ),   # (2π/3, 2π/√3)  Cartesian
        Kp = (round(Int, 2Nx / 3),     round(Int, Ny / 3)     ),   # (4π/3, 0)       Cartesian
    )
end


"""
    hsk_square(Lx, Ly) -> NamedTuple

High-symmetry k-points for a 2D square lattice in quantics integer units.

| Point | Symbol | Meaning              | Quantics index   |
|-------|--------|----------------------|------------------|
| Γ     | `G`    | zone centre          | (0, 0)           |
| X     | `X`    | zone-edge midpoint   | (Nx÷2, 0)       |
| M     | `M`    | zone corner          | (Nx÷2, Ny÷2)    |

Standard path: `[:G, :X, :M, :G]`
"""
function hsk_square(Lx::Int, Ly::Int)
    Nx, Ny = 2^Lx, 2^Ly
    return (
        G = (0,      0     ),   # Gamma — zone centre
        X = (Nx÷2,   0     ),
        M = (Nx÷2,   Ny÷2  ),
    )
end


"""
    hsk_triangular(Lx, Ly) -> NamedTuple

Approximate high-symmetry k-points for a 2D triangular lattice in quantics
integer units.

| Point | Symbol | Quantics index   |
|-------|--------|------------------|
| Γ     | `G`    | (0, 0)           |
| M     | `M`    | (Nx÷2, 0)       |
| K     | `K`    | (Nx÷3, Ny÷3)   |

Standard path: `[:G, :M, :K, :G]`
"""
function hsk_triangular(Lx::Int, Ly::Int)
    Nx, Ny = 2^Lx, 2^Ly
    return (
        G = (0,      0     ),   # Gamma — zone centre
        M = (Nx÷2,   0     ),
        K = (Nx÷3,   Ny÷3  ),
    )
end


# ============================================================
# 2. Symbol-based path setup (the get_bands kpath shortcut)
# ============================================================
#
#   _hs_label(sym)            — symbol → display string (G → "Γ", Kp → "K'")
#   _hsk(lattice, Lx, Ly)     — dispatch to the right hsk_* function
#   kpath_setup(lattice, ...) — builds (k_groups, ticks, labels) from symbols

# Symbol → display string for axis tick labels.
# Use Latin aliases (G, Kp, …) in code; display shows the traditional notation.
_hs_label(s::Symbol) = s === :G  ? "Γ"  :
                       s === :M  ? "M"  :
                       s === :K  ? "K"  :
                       s === :Kp ? "K'" :
                       s === :X  ? "X"  :
                       s === :R  ? "R"  :
                       s === :A  ? "A"  :
                       string(s)

# Dispatch kpath symbols → (kx_idx, ky_idx) integer pairs via hsk_* functions.
_hsk(lattice::Symbol, Lx::Int, Ly::Int) =
    lattice === :honeycomb  ? hsk_honeycomb(Lx, Ly)  :
    lattice === :square     ? hsk_square(Lx, Ly)     :
    lattice === :triangular ? hsk_triangular(Lx, Ly) :
    error("Unknown kpath_lattice :$lattice.  Use :honeycomb, :square, or :triangular.")

"""
    kpath_setup(lattice, Lx, Ly, path_syms; npts_per_segment=20)
        -> (k_groups, ticks, labels)

Build k-path inputs for `get_bands` from a list of high-symmetry symbols.
`path_syms` is a vector of symbols such as `[:G, :M, :Kp, :G]`
(use `G` for Γ — the Latin alias avoids non-ASCII input).
`npts_per_segment` is the number of points between each consecutive pair.

Returns `(k_groups, ticks, labels)` ready to pass to `get_bands` as
`k_groups_override`, and to `heatmap` as `xticks=(ticks, labels)`.
"""
function kpath_setup(lattice::Symbol, Lx::Int, Ly::Int,
                     path_syms::AbstractVector{Symbol};
                     npts_per_segment::Int = 20)
    hs     = _hsk(lattice, Lx, Ly)
    path   = [getfield(hs, s) for s in path_syms]
    labels = [_hs_label(s) for s in path_syms]
    kg, ticks = kpath_2d(path, Lx; npts_per_segment = npts_per_segment)
    return kg, ticks, labels
end
