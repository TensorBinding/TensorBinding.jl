# Flake.jl — smooth flake masking for TBHamiltonian: signed-distance functions
# (SDFs) for 1D and 2D shapes, their CSG combinations, and mask_hamiltonian, which
# restricts a Hamiltonian to the region where an SDF is positive.
#
# Entry points: mask_hamiltonian; sdf_interval, sdf_disk, sdf_rect, sdf_halfplane,
#   sdf_annulus, sdf_convex_polygon; sdf_union, sdf_intersect, sdf_subtract.
#
# Depends on: core/TBSystem.jl (TBHamiltonian, _invalidate_cache!) and core/Utils.jl
# (postpend_op, for the per-sublattice masks).
#
# mask_hamiltonian applies a smooth diagonal mask M learned via QTCI:
#
#     M_i = σ( sdf(rᵢ) / sigma ),   H_flake = M · H · M
#
# where sdf(r) is a signed-distance function (positive inside the flake),
# sigma is the smoothing half-width in lattice units, and σ is the logistic
# sigmoid.  Because sigma can be much smaller than 1 lattice spacing, QTCI
# resolves the boundary sharply without a hard cutoff.
#
# Typical workflow
# ---------------
#   H = get_Hamiltonian("square_2d", 1.0; L=8, Lx=4)   # 16 × 16 sites (Lx, L are qubit counts)
#   H_disk = mask_hamiltonian(H, sdf_disk(8.0, 8.0, 6.0); sigma=0.3)
#
# SDFs compose via CSG helpers (sdf_union / sdf_intersect / sdf_subtract).
# Call mask_hamiltonian *before* add_spin! / add_superconductivity!.

# ============================================================
# 1. 1-D SDF primitives
# ============================================================

"""
    sdf_interval(x_lo, x_hi) -> x -> Float64

Signed distance function for the interval `[x_lo, x_hi]`.
Positive inside, negative outside.
"""
sdf_interval(x_lo, x_hi) = (x) -> min(x - x_lo, x_hi - x)


# ============================================================
# 2. 2-D SDF primitives
# ============================================================

"""
    sdf_disk(cx, cy, r) -> (x, y) -> Float64

Signed distance function for a disk of radius `r` centred at `(cx, cy)`.
Positive inside, negative outside.
"""
sdf_disk(cx, cy, r) = (x, y) -> r - sqrt((x - cx)^2 + (y - cy)^2)


"""
    sdf_rect(cx, cy, w, h) -> (x, y) -> Float64

Signed distance function for an axis-aligned rectangle with full width `w`
and full height `h`, centred at `(cx, cy)`.  Positive inside.
"""
sdf_rect(cx, cy, w, h) = (x, y) -> min(w/2 - abs(x - cx), h/2 - abs(y - cy))


"""
    sdf_halfplane(nx, ny, d) -> (x, y) -> Float64

Signed distance function for the half-plane  `nx·x + ny·y ≥ d`.
`(nx, ny)` need not be unit-length — normalisation is applied internally.
Positive on the side where the inequality holds.
"""
function sdf_halfplane(nx::Real, ny::Real, d::Real)
    len = sqrt(nx^2 + ny^2)
    return (x, y) -> (nx*x + ny*y - d) / len
end


"""
    sdf_annulus(cx, cy, r_in, r_out) -> (x, y) -> Float64

Signed distance function for a circular annulus with inner radius `r_in`
and outer radius `r_out`, centred at `(cx, cy)`.
"""
sdf_annulus(cx, cy, r_in, r_out) =
    sdf_subtract(sdf_disk(cx, cy, r_out), sdf_disk(cx, cy, r_in))


"""
    sdf_convex_polygon(vertices) -> (x, y) -> Float64

Signed distance function for the convex polygon with the given `vertices`
(a vector of `(x, y)` tuples, listed counter-clockwise or clockwise: the
orientation is read from the signed area). Positive inside, like the other SDFs.

Implemented as the minimum signed distance to each edge's inward half-plane.
Collinear vertices (zero area) are an error.
"""
function sdf_convex_polygon(vertices::AbstractVector)
    length(vertices) ≥ 3 || error("sdf_convex_polygon: need at least 3 vertices.")
    n = length(vertices)
    # Twice the signed area (shoelace): positive for counter-clockwise vertices.
    area2 = sum(vertices[i][1] * vertices[mod1(i + 1, n)][2] -
                vertices[mod1(i + 1, n)][1] * vertices[i][2] for i in 1:n)
    iszero(area2) && error("sdf_convex_polygon: the vertices are collinear (zero area).")
    ccw = area2 > 0
    return (x, y) -> begin
        d = Inf
        for i in 1:n
            x1, y1 = vertices[i]
            x2, y2 = vertices[mod1(i + 1, n)]
            ex, ey  = x2 - x1, y2 - y1
            # Inward normal: the edge vector rotated by 90° CCW for a counter-clockwise
            # polygon, 90° CW for a clockwise one. (The CCW case used the CW rotation,
            # the outward normal, so the documented order gave an SDF negative everywhere.)
            nx_, ny_ = ccw ? (-ey, ex) : (ey, -ex)
            len = sqrt(nx_^2 + ny_^2)
            d   = min(d, (nx_*(x - x1) + ny_*(y - y1)) / len)
        end
        return d
    end
end


# ============================================================
# 3. CSG operations
# ============================================================

"""
    sdf_union(f, g) -> SDF

Boolean union: positive where *either* `f` or `g` is positive (`max(f, g)`).
"""
sdf_union(f, g) = (args...) -> max(f(args...), g(args...))

"""
    sdf_intersect(f, g) -> SDF

Boolean intersection: positive where *both* `f` and `g` are positive (`min(f, g)`).
"""
sdf_intersect(f, g) = (args...) -> min(f(args...), g(args...))

"""
    sdf_subtract(f, g) -> SDF

Boolean subtraction: positive inside `f` and outside `g` (`min(f, -g)`).
"""
sdf_subtract(f, g) = (args...) -> min(f(args...), -g(args...))


# ============================================================
# 4. Mask application
# ============================================================

"""
    mask_hamiltonian(H, sdf; sigma=0.3, tol=1e-8, maxdim=200, cutoff=1e-8)
        -> TBHamiltonian

Restrict `H` to a flake geometry defined by `sdf` by applying the smooth
diagonal mask

    M_i = σ( sdf(rᵢ) / sigma ),   H_flake = M · H · M

where `rᵢ = H.geometry(i)` is the real-space position of site `i` (1-indexed),
`σ` is the logistic sigmoid, and `sigma` controls boundary sharpness in the
same coordinate units as `H.geometry`.

The mask MPS is learned via QTCI, so even sub-lattice-spacing smoothing
(`sigma ≪ 1`) is captured accurately without explicit enumeration.

**Keyword arguments**
- `sigma`   : sigmoid half-width (lattice units). Default `0.3`.
- `tol`     : QTCI tolerance for the mask MPS. Default `1e-8`.
- `maxdim`  : max bond dim during the two M·H·M products. Default `200`.
- `cutoff`  : SVD truncation cutoff. Default `1e-8`.

**Restrictions**
- Requires `H.geometry` to be set (all preset geometries provide this).
- Must be called **before** `add_spin!`, `add_superconductivity!`, or
  bilayer construction.
- On a sublattice Hamiltonian (kagome, Lieb, honeycomb, dice, …; the sublattice
  index postpended) every atom is masked at its own position `H.geometry(i)`:
  one QTCI mask per sublattice, `M = Σ_s M_s ⊗ |s⟩⟨s|`.

**Examples**
```julia
H = get_Hamiltonian("square_2d", 1.0; L=8, Lx=4)   # 16 × 16 sites

# Disk flake
H_disk = mask_hamiltonian(H, sdf_disk(8.0, 8.0, 6.0); sigma=0.3)

# Triangular flake (intersection of three half-planes)
tri = sdf_intersect(
    sdf_intersect(sdf_halfplane(0, 1, 2.0),
                  sdf_halfplane(-sqrt(3)/2, -0.5, -8.0)),
                  sdf_halfplane( sqrt(3)/2, -0.5, -8.0))
H_tri = mask_hamiltonian(H, tri; sigma=0.2)

# Ring (disk minus inner disk)
H_ring = mask_hamiltonian(H, sdf_annulus(8.0, 8.0, 3.0, 7.0); sigma=0.3)
```
"""
function mask_hamiltonian(H::TBHamiltonian, sdf;
                          sigma::Real  = 0.3,
                          tol::Real    = 1e-8,
                          maxdim::Int  = 200,
                          cutoff::Real = 1e-8)
    H.spin_s === nothing && H.nambu_s === nothing && H.layer_s === nothing ||
        error("mask_hamiltonian: call before add_spin!, add_superconductivity!, " *
              "or bilayer construction (auxiliary indices not yet supported).")
    isnothing(H.geometry) &&
        error("mask_hamiltonian: H.geometry is not set. " *
              "Provide geometry= when constructing, or use a preset model.")

    L     = H.L
    N     = H.N
    sub   = H.sublattice_s

    # Logistic sigmoid with numerical clamp
    _sig(x) = 1 / (1 + exp(-clamp(x, -500.0, 500.0)))

    # Mask value for 1-indexed atom i (geometry(i) splatted into sdf)
    mask_site(i) = _sig(sdf(H.geometry(i)...) / sigma)

    mask_mpo = if sub === nothing
        _diagonal_mask_mpo(mask_site, N, H.sites; tol)
    else
        # One mask per sublattice on the position qubits, M = Σ_s M_s ⊗ |s⟩⟨s|, with
        # M_s(n) the mask at atom n_sub·n + s (0-based cell n), summed exactly.
        (H.aux_side === :post && H.sites[end] == sub) ||
            error("mask_hamiltonian: the sublattice index must be the last site " *
                  "(aux_side = :post), as the sublattice builders make it.")
        n_sub = dim(sub)
        pos   = H.sites[1:L]
        terms = [postpend_op(_diagonal_mask_mpo(i -> mask_site(n_sub * (i - 1) + s), N, pos;
                                                tol), sub, s) for s in 1:n_sub]
        foldl((a, b) -> +(a, b; alg="directsum"), terms)
    end

    # H_flake = M · H · M  (Hermitian, boundary hoppings smoothly suppressed)
    Hm = apply(mask_mpo, H.mpo; maxdim=maxdim, cutoff=cutoff)
    Hm = apply(Hm, mask_mpo;    maxdim=maxdim, cutoff=cutoff)
    ITensorMPS.truncate!(Hm; maxdim=maxdim, cutoff=cutoff)

    H_new     = deepcopy(H)
    H_new.mpo = Hm
    _invalidate_cache!(H_new)
    return H_new
end

# Diagonal MPO diag(m(1), …, m(N)) on the L qubit `sites` (N = 2^L, site 1 the most
# significant bit), learned by QTCI at tolerance `tol` over x = 0, …, N-1 (x ↦ m(x + 1)).
function _diagonal_mask_mpo(m, N, sites; tol)
    mask_raw(x) = m(round(Int, x) + 1)
    xvals       = range(0, N - 1; length=N)
    qtt, _, _   = quanticscrossinterpolate(Float64, mask_raw, xvals; tolerance=tol)
    mask_mps    = MPS(TCI.tensortrain(qtt.tci); sites=sites)

    # Promote MPS → diagonal MPO via _asdiagonal on each site tensor
    mask_mpo = outer(mask_mps', mask_mps)
    for i in 1:length(sites)
        mask_mpo[i] = Quantics._asdiagonal(mask_mps[i], sites[i])
    end
    return mask_mpo
end
