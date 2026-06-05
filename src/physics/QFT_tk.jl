# QFT_tk.jl — Momentum-space band structure via online Chebyshev KPM
#
# ─────────────────────────────────────────────────────────────────────────────
# Overview
# ─────────────────────────────────────────────────────────────────────────────
# The quantics representation encodes a 1D or 2D real-space position index as
# a binary string across L qubit sites.  Conjugating any real-space MPO W by
# the Quantum Fourier Transform gives its momentum-space counterpart:
#
#   Ã(k,ω) = U · δ(ω − H) · U†
#
# The diagonal ⟨k|Ã(ω)|k⟩ is the k-resolved spectral function A(k,ω).
#
# ─────────────────────────────────────────────────────────────────────────────
# Online Chebyshev accumulation
# ─────────────────────────────────────────────────────────────────────────────
# `get_bands` runs a single Chebyshev recurrence over the full MPO space.
# At each step n the current T_n passes through five composable projection
# stages before the QFT is applied:
#
#   Step 0  nambu_proj   — project Nambu (BdG particle/hole) auxiliary index
#   Step 1  spin_proj    — project spin auxiliary index
#   Step 1c layer_proj   — project layer auxiliary index (bilayer/multilayer)
#   Step 1b sublat_proj  — project sublattice auxiliary index (kagome, Lieb, …)
#   Step 2  sublattice   — legacy mask sandwich (preset models without aux index)
#   Step 3  QFT + diagonal extraction + KPM weight accumulation
#
# All steps are independent and optional; any combination is valid.
# Peak memory: O(3 MPOs) regardless of Ncheb.
#
# ─────────────────────────────────────────────────────────────────────────────
# Auxiliary DOF projection  (section 4b)
# ─────────────────────────────────────────────────────────────────────────────
# Models with auxiliary DOFs (spin, Nambu, layer, sublattice) have an extra
# site at the front (`:pre`) or back (`:post`) of the MPO.  `project_aux`
# removes it by contracting |σ⟩⟨σ| onto the auxiliary tensor, returning an
# (L−1)-site position-only MPO ready for `conjugate_by_qft`.
#
# When `H::TBHamiltonian` is passed to `get_bands`, all auxiliary indices are
# auto-detected from the struct fields (H.spin_s, H.nambu_s, H.layer_s,
# H.sublattice_s) and never need to be passed manually.
#
# ─────────────────────────────────────────────────────────────────────────────
# High-symmetry k-path shortcut  (section 3b)
# ─────────────────────────────────────────────────────────────────────────────
# The `kpath` kwarg in the `TBHamiltonian` overload of `get_bands` eliminates
# the manual kpath setup:
#
#   res = get_bands(H, Ncheb, 2, omega;
#                   kpath=[:G, :M, :Kp, :G], kpath_lattice=:honeycomb, num_x=30)
#   # res.Ak, res.ticks, res.labels  ← all path metadata included
#
# ─────────────────────────────────────────────────────────────────────────────
# Encoding conventions
# ─────────────────────────────────────────────────────────────────────────────
# 1D  — sites 1…L hold x bits, LSB at site 1 (quantics QFT convention).
# 2D  — sites 1…Ly hold iy bits (MSB first), sites Ly+1…L hold ix bits
#        (MSB first); linear index n = ix + iy·2^Lx (row-major).
#
# ─────────────────────────────────────────────────────────────────────────────
# Dependencies outside this file
# ─────────────────────────────────────────────────────────────────────────────
# fix_sites, _kpm_kernel               → utils.jl
# extract_diagonal_to_mps              → utils.jl
# _row_checker_mpo, _col_select_mpo    → 2D_lattice.jl
# TBHamiltonian, _ensure_scale!        → TBSystem.jl
#
# ─────────────────────────────────────────────────────────────────────────────
# File structure
# ─────────────────────────────────────────────────────────────────────────────
# 1.  QFT conjugation            conjugate_by_qft
# 2.  Legacy sublattice projectors projop_2DSL, projop_1DSL
# 3.  Internal utilities         ilinspace, _eval_diag_mps, sample_diag,
#                                _kpm_weight_matrix
# 3b. High-symmetry k-path       kpath_2d, hsk_honeycomb/square/triangular,
#                                kpath_setup, _hs_label, _hsk
# 4.  Online band structure      get_bands (low-level MPO version)
# 4b. Aux index projection       project_aux, project_spin, aux_site
# 5.  High-level overload        get_bands (TBHamiltonian version)
# 6.  Legacy reference code      old get_bands (inner-product approach)


# ============================================================
# 1. QFT conjugation
# ============================================================

"""
    conjugate_by_qft(W; tol=1e-9, maxdim=100) -> MPO

Return `U · W · U†` where `U` is the Quantum Fourier Transform MPO built from
`QuanticsTCI.quanticsfouriermpo` (normalised, with `TCI.reverse` applied).

`TCI.reverse` places the LSB at site 1, matching the quantics encoding used
throughout this codebase.  The resulting k-space MPO has the same site
structure as `W` but with momenta as the diagonal degree of freedom.

Calling this on the Chebyshev spectral MPO T_n and then extracting the
diagonal gives the k-resolved contribution to A(k,ω).
"""
function conjugate_by_qft(W; tol=1e-9, maxdim::Int=100)
    sites  = getindex.(siteinds(W), 2)
    R      = length(sites)
    FTirev = fix_sites(MPO(TCI.reverse(QuanticsTCI.quanticsfouriermpo(R; sign=-1.0, normalize=true))), sites)
    FTrev  = fix_sites(MPO(TCI.reverse(QuanticsTCI.quanticsfouriermpo(R; sign=+1.0, normalize=true))), sites)
    Op1    = apply(W,                        FTirev; cutoff=tol, maxdim=maxdim)
    Op2    = apply(swapprime(FTrev, 0 => 1), Op1;   cutoff=tol, maxdim=maxdim)
    return TCI.truncate(Op2; cutoff=tol, maxdim=maxdim)
end


"""
    conjugate_by_qft(H::TBHamiltonian, W::MPO; tol, maxdim) -> MPO

TBHamiltonian-aware version of `conjugate_by_qft`.  Applies `U·W·U†` where
`U` is the QFT acting **only on the position (Qubit) sites** of `H`, with
identity operators at all auxiliary sites (Layer, spin, sublattice, Nambu).

Use this overload whenever `W` lives on the full `H.sites` space (including
aux indices), as is the case in the bubble pipeline after `replace_sites`.
"""
function conjugate_by_qft(H::TBHamiltonian, W::MPO; tol=1e-9, maxdim::Int=100)
    pos_s = _pos_sites(H)
    R     = length(pos_s)
    FTirev_pos = fix_sites(MPO(TCI.reverse(QuanticsTCI.quanticsfouriermpo(R; sign=-1.0, normalize=true))), pos_s)
    FTrev_pos  = fix_sites(MPO(TCI.reverse(QuanticsTCI.quanticsfouriermpo(R; sign=+1.0, normalize=true))), pos_s)
    FTirev = _embed_in_full_sites(H, FTirev_pos)
    FTrev  = _embed_in_full_sites(H, FTrev_pos)
    Op1    = apply(W,                        FTirev; cutoff=tol, maxdim=maxdim)
    Op2    = apply(swapprime(FTrev, 0 => 1), Op1;   cutoff=tol, maxdim=maxdim)
    return TCI.truncate(Op2; cutoff=tol, maxdim=maxdim)
end


"""
    _embed_in_full_sites(H, mpo_pos) -> MPO

Embed `mpo_pos` (which lives on `_pos_sites(H)`) into the full `H.sites`
space by prepending/appending dim-1 identity tensors at each auxiliary site.
"""
function _embed_in_full_sites(H::TBHamiltonian, mpo_pos::MPO)
    pos_set  = Set(_pos_sites(H))
    first_pos = findfirst(s -> s ∈ pos_set, H.sites)
    last_pos  = findlast( s -> s ∈ pos_set, H.sites)
    pre_aux  = H.sites[1:first_pos-1]
    post_aux = H.sites[last_pos+1:end]
    result   = mpo_pos
    for s in reverse(pre_aux)
        result = mpo_kron(MPO([dense(delta(s, prime(s)))]), result)
    end
    for s in post_aux
        result = mpo_kron(result, MPO([dense(delta(s, prime(s)))]))
    end
    return result
end


"""
    _embed_displacement_in_full_sites(H, mpo_pos) -> MPO

Like `_embed_in_full_sites` but pads auxiliary sites with all-ones matrices
instead of identity.  Required for current-operator construction: the
displacement (xᵣ − xᵣ′) depends only on position, so it must be broadcast
uniformly across all auxiliary (sublattice, layer, spin, Nambu) index pairs,
including off-diagonal ones where physical hoppings exist.

Using identity at an aux site with off-diagonal hoppings (e.g. sublattice A↔B
in honeycomb, or inter-layer tunneling in bilayers) would set those current
matrix elements to zero and give σ = 0.
"""
function _embed_displacement_in_full_sites(H::TBHamiltonian, mpo_pos::MPO)
    pos_set   = Set(_pos_sites(H))
    first_pos = findfirst(s -> s ∈ pos_set, H.sites)
    last_pos  = findlast( s -> s ∈ pos_set, H.sites)
    pre_aux   = H.sites[1:first_pos-1]
    post_aux  = H.sites[last_pos+1:end]
    result    = mpo_pos
    for s in reverse(pre_aux)
        ones_t = dense(ITensor(ones(Float64, dim(s), dim(s)), prime(s), s))
        result = mpo_kron(MPO([ones_t]), result)
    end
    for s in post_aux
        ones_t = dense(ITensor(ones(Float64, dim(s), dim(s)), prime(s), s))
        result = mpo_kron(result, MPO([ones_t]))
    end
    return result
end


# ============================================================
# 2. Legacy sublattice projection  (mask sandwich)
#
# These wrappers apply the mask sandwich O_SL = mask · O · mask to project
# an MPO onto one of two sublattices.  They are used by `get_bands` when
# `sublattice=true` (Step 2 in the pipeline).
#
# WHEN TO USE:
#   `sublattice=true` / `projop_*SL`  — for PRESET models built by
#   `build_hamiltonian` / `monolayer_hamiltonian` (HUniform2Dhex, H2DChernhex,
#   HUniform2Dtri, …).  These encode the sublattice structure implicitly in
#   the hopping MPO; H.sublattice_s is nothing.
#
#   `sublat_proj=true`                — for models with an EXPLICIT sublattice
#   auxiliary index (honeycomb_sublattice_hamiltonian, kagome_hamiltonian,
#   lieb_hamiltonian, dice_hamiltonian).  H.sublattice_s is set.
#
# The mask MPOs are bond-dimension 1 (single-site operators from 2D_lattice.jl)
# and negligibly cheap to apply.
#
# 2D — checkerboard sublattices:
#   SL=1 → (ix+iy) even  (_row_checker_mpo)
#   SL=2 → (ix+iy) odd   (Id − _row_checker_mpo)
#
# 1D — alternating-site sublattices:
#   SL=1 → even sites (ix % 2 == 0)   (_col_select_mpo, keep=:odd)
#   SL=2 → odd  sites (ix % 2 == 1)   (_col_select_mpo, keep=:even)
#   (`:odd`/`:even` labels refer to the LSB qubit state, not the site index)
# ============================================================

"""
    projop_2DSL(O, sites, Lx, Ly, SL) -> MPO

Project MPO `O` (on a `2^Lx × 2^Ly` lattice) onto sublattice `SL`
(1 = even checkerboard, 2 = odd checkerboard) by sandwiching with the
corresponding diagonal mask: `mask · O · mask`.
"""
function projop_2DSL(O::MPO, sites, Lx, Ly, SL::Integer)
    mask = SL == 1 ? _row_checker_mpo(Lx, Ly, sites) :
                     MPO(sites, "Id") - _row_checker_mpo(Lx, Ly, sites)
    Oproj = apply(mask, O; cutoff=1e-8, maxdim=100)
    Oproj = apply(Oproj, mask; cutoff=1e-8, maxdim=100)
    return Oproj
end

"""
    projop_1DSL(O, sites, Lx, SL) -> MPO

Project MPO `O` (on a `2^Lx` chain) onto sublattice `SL`
(1 = even sites, 2 = odd sites) by sandwiching with the corresponding
diagonal mask: `mask · O · mask`.
"""
function projop_1DSL(O::MPO, sites, Lx, SL::Integer)
    mask = SL == 1 ? _col_select_mpo(Lx, 0, sites; keep=:odd) :
                     _col_select_mpo(Lx, 0, sites; keep=:even)
    Oproj = apply(mask, O; cutoff=1e-8, maxdim=100)
    Oproj = apply(Oproj, mask; cutoff=1e-8, maxdim=100)
    return Oproj
end


# ============================================================
# 3. Internal utilities
#
# ilinspace       — evenly-spaced integer grid for k-center placement
# _eval_diag_mps  — fast diagonal evaluation without constructing basis MPS
# sample_diag     — batch evaluation over a contiguous range (convenience)
# _kpm_weight_matrix — precomputed Chebyshev-KPM weights W[n, iω]
# ============================================================

"""
    ilinspace(xmin, xmax, num_x) -> Vector{Int}

Return `num_x` as almost evenly spaced integers in `[xmin, xmax]`, inclusive,
with a preference for the endpoints.  Used to build the k-point center
grid for band-structure sampling.
"""
function ilinspace(xmin, xmax, num_x::Int)
    xvals = xmin:xmax
    _N = length(xvals)
    @assert 1 ≤ num_x ≤ _N
    num_x == 1 && return [0]
    step = (_N - 1) ÷ (num_x - 1)
    return collect(xmin:step:(xmin+step*(num_x-1)))
end


"""
    _eval_diag_mps(A, x) -> Float64

Evaluate the diagonal MPS `A` at the 0-indexed position `x` using a
LSB-first bit encoding (site 1 = bit 0 of x).  Equivalent to
`inner(binary_MPS(x), A)` but avoids constructing the full basis MPS.
"""
function _eval_diag_mps(A::MPS, x::Int)
    L     = length(A)
    sites = siteinds(A)
    acc   = ITensor(1.0)
    for i in 1:L
        b    = (x >> (i - 1)) & 1     # bit i-1 of x, LSB first
        acc *= A[i] * setelt(sites[i] => b + 1)
    end
    return real(scalar(acc))
end


"""
    sample_diag(Tn_k, ikstart, ikend) -> Vector{Float32}

Extract the diagonal of MPO `Tn_k` as an MPS and evaluate it at every
integer index in `ikstart:ikend`.  Convenience wrapper around
`_eval_diag_mps`; used when all k-points in a contiguous range are needed.
"""
function sample_diag(Tn_k::MPO, ikstart::Int, ikend::Int)
    A_mps = extract_diagonal_to_mps(Tn_k)
    A_mps = ITensorMPS.truncate!(A_mps; cutoff=1e-10)
    vals  = zeros(Float32, ikend - ikstart + 1)
    for (iloc, idx) in enumerate(ikstart:ikend)
        vals[iloc] = _eval_diag_mps(A_mps, idx)
    end
    return vals
end


"""
    _kpm_weight_matrix(Ncheb, ω_vals; kernel=:jackson, lambda=4.0) -> Matrix{Float64}

Precompute the full KPM weight matrix `W[n, iω]` for fast in-loop accumulation.

```
W[n, iω] = c_n · g_n · cos((n-1) · arccos(ω_iω))
```

- `c_n = 1` for n=1, `c_n = 2` otherwise (Chebyshev expansion factor)
- `g_n` = kernel damping: Jackson (default, finite-size ringing suppressed)
  or Lorentz (controlled width `lambda`, smoother tails)
- Entries for `|ω| ≥ 1` are set to zero (outside the spectral support)

Pre-computing W avoids recomputing cos((n-1)·arccos(ω)) inside the inner loop,
which is called Ncheb × Nω times.
"""
function _kpm_weight_matrix(Ncheb::Int, ω_vals; kernel::Symbol=:jackson, lambda::Real=4.0)
    kweights = _kpm_kernel(Ncheb, kernel; lambda=lambda)
    Nω = length(ω_vals)
    W = zeros(Float64, Ncheb, Nω)
    for iω in 1:Nω
        abs(ω_vals[iω]) >= 1.0 && continue
        for n in 1:Ncheb
            W[n, iω] = (n == 1 ? 1.0 : 2.0) * kweights[n] * cos((n-1) * acos(ω_vals[iω]))
        end
    end
    return W
end


# ============================================================
# 3b. High-symmetry k-path utilities  (2D)
#
# Low-level:
#   kpath_2d(hs_points, Lx; npts_per_segment)
#       Takes explicit (kx_idx, ky_idx) tuples, interpolates between them.
#       Returns (k_groups, tick_positions) for use with k_groups_override.
#
#   hsk_honeycomb / hsk_square / hsk_triangular(Lx, Ly)
#       Named-tuple dictionaries of standard high-symmetry k-points in
#       quantics integer units (0-based; verified analytically for honeycomb,
#       approximate for triangular).  Use Latin symbol names (G, M, K, Kp, X)
#       — no non-ASCII input required.
#
# High-level (called internally by get_bands kpath shortcut):
#   _hsk(lattice, Lx, Ly)     — dispatch to the right hsk_* function
#   _hs_label(sym)            — symbol → display string (G → "Γ", Kp → "K'")
#   kpath_setup(lattice, ...)  — builds (k_groups, ticks, labels) from symbols
#
# The preferred user interface is the kpath kwarg in get_bands(H::TBHamiltonian):
#   res = get_bands(H, Ncheb, 2, omega;
#                   kpath=[:G, :M, :Kp, :G], kpath_lattice=:honeycomb, num_x=30)
#   # res.Ak — (Nω × Nk) spectral function
#   # res.ticks, res.labels — ready for xticks=(res.ticks, res.labels)
# ============================================================

"""
    kpath_2d(hs_points, Lx; npts_per_segment=20) -> (k_groups, tick_positions)

Build k-groups for `get_bands(…; D=2, k_groups_override=k_groups)` sampling
along a high-symmetry path.

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


# ── High-symmetry path helpers ───────────────────────────────────────────────

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


# ============================================================
# 4. Online band structure  —  get_bands
#
# ── Chebyshev recurrence ────────────────────────────────────────────────────
# Runs on the FULL MPO space (all auxiliary sites included):
#   T_0 = I,  T_1 = H̃,  T_n = 2 H̃ T_{n-1} − T_{n-2}    (H̃ = (H−center)/scale)
#
# ── Projection pipeline (five composable steps) ─────────────────────────────
# At each step n, T_n is passed through the following stages.  Each stage
# builds a list of position-only MPOs; every MPO in the final list is QFT'd,
# sampled, and its contribution added to Ak_w.
#
#   Step 0  nambu_proj  — Nambu (BdG particle/hole) aux index
#       Outermost aux (prepended last), projected first.
#       Sectors: 1=particle, 2=hole.  kwarg: proj_nambu.
#
#   Step 1  spin_proj   — spin aux index
#       After Nambu removal, spin is at site 1 of the reduced MPO.
#       spin_s_aux carries the explicit spin Index to avoid ambiguity when
#       both Nambu and spin are prepended.
#       Channels: 1=↑, 2=↓.  kwarg: proj_s.
#
#   Step 1c layer_proj  — layer aux index (bilayer / multilayer)
#       For bilayer_hamiltonian / twisted_bilayer_hamiltonian models (H.layer_s).
#       Sectors: 1…n_layers.  kwarg: proj_layer.
#
#   Step 1b sublat_proj — sublattice aux index (explicit aux models only)
#       For honeycomb_sublattice_hamiltonian, kagome_hamiltonian,
#       lieb_hamiltonian, dice_hamiltonian (H.sublattice_s set).
#       Sectors: 1…dim(sublat_s).  kwarg: proj_sl.
#
#   Step 2  sublattice  — LEGACY mask sandwich (preset models, no aux index)
#       For HUniform2Dhex, H2DChernhex, HUniform2Dtri, HQC2Dsquare, …
#       The sublattice structure is implicit in the hopping MPO.
#       Use this when H.sublattice_s is nothing.  kwarg: proj_sl (shared).
#
#   Step 3  QFT + diagonal extraction + KPM weight accumulation  (always)
#       T_k   = conjugate_by_qft(T_proj)
#       A_mps = extract_diagonal_to_mps(T_k)
#       for each k-group:  s = mean(_eval_diag_mps(A_mps, x) for x in group)
#       ak_accum[iω, ik] += W[n, iω] * s
#
# ── k-point groups ───────────────────────────────────────────────────────────
# Default (grid) mode: num_x centres placed with ilinspace in [xmin,xmax];
#   each centre is averaged over num_avg offset points (±half_step).
#   2D: offsets zipped diagonally, combined as (iy << Lx) | ix.
#
# Path mode: pass k_groups_override (from kpath_2d) or use the kpath kwarg
#   in the TBHamiltonian overload; this bypasses all grid parameters.
#
# ── Projection count per Chebyshev step ──────────────────────────────────────
#   nambu(×2) × spin(×2) × layer(×n) × sublat_aux(×dim) × sublat_mask(×2)
#   All contributions are summed unless a specific sector is selected via the
#   corresponding proj_* kwarg.
# ============================================================

"""
    get_bands(H_mpo, scale, center, sites, Ncheb, D, ω_vals; kwargs...) -> Matrix{Float64}

Memory-efficient band structure via online Chebyshev KPM accumulation.
See the section 4 block comment above for the full four-step projection pipeline.

# Arguments
- `H_mpo`        : unscaled Hamiltonian MPO on all sites (position + any aux).
- `scale, center`: energy rescaling so that H̃ = (H−center)/scale ∈ (−1, 1).
- `sites`        : the full site list of `H_mpo` including any aux indices.
                   Position-only site count is inferred as `L_pos = L − n_aux`.
- `Ncheb`        : number of Chebyshev moments.
- `D`            : spatial dimension (1 or 2).
- `ω_vals`       : rescaled energies ∈ (−1, 1) at which to evaluate A(k,ω).

# Projection keyword arguments
Each projection flag is independent; any combination is valid.

**Nambu (BdG particle/hole) projection — Step 0:**
- `nambu_proj`   : project each T_n onto Nambu sectors (default `false`).
- `proj_nambu`   : `1` = particle only, `2` = hole only, `nothing` = sum both.
- `nambu_s`      : the Nambu `Index` (auto-detected from `H.nambu_s` via the
                   `TBHamiltonian` overload).
- `nambu_side`   : `:pre` (default) or `:post` — position of the Nambu site.

**Spin projection — Step 1:**
- `spin_proj`    : project each T_n onto spin channels (default `false`).
- `proj_s`       : `1` = ↑ only, `2` = ↓ only, `nothing` = sum both.
- `spin_s_aux`   : explicit spin `Index`; when `nothing`, falls back to
                   `sites[1]`.  Set automatically by the `TBHamiltonian` overload
                   so that spin is correctly identified even when Nambu is also
                   prepended at site 1.

**Layer projection — Step 1c (bilayer / multilayer):**
- `layer_proj`   : project each T_n onto individual layers (default `false`).
- `proj_layer`   : `k` = layer k only, `nothing` = sum all layers.
- `layer_s`      : the layer `Index` (auto-detected from `H.layer_s`).
- `layer_side`   : `:pre` (default) — layer is always prepended.

**Sublattice auxiliary projection — Step 1b (kagome, Lieb, honeycomb):**
- `sublat_proj`  : project each T_n onto sublattice aux sectors (default `false`).
- `proj_sl`      : `k` = sublattice k only, `nothing` = sum all.  Shared with
                   the legacy `sublattice` flag (Step 2).
- `sublat_s`     : the sublattice `Index` (auto-detected from `H.sublattice_s`).
- `sublat_side`  : `:post` (default) or `:pre` — position of the sublattice site.

**Legacy sublattice mask projection — Step 2 (2-sublattice models without aux index):**
- `sublattice`   : apply a mask sandwich `mask · T_n · mask` (default `false`).
- `proj_sl`      : `1` = mask A only, `2` = mask B only, `nothing` = both.

# k-point sampling keyword arguments
- `xmin, xmax, num_x` : grid in x (1D) or kx (2D).  Default: full range, 10 pts.
- `ymin, ymax, num_y` : grid in ky (2D only).
- `num_avg`      : number of offset points averaged around each center (default 1).

# Truncation and performance
- `kernel`       : KPM broadening kernel (`:jackson` or `:lorentz`).
- `lambda`       : Lorentz kernel width (ignored for Jackson).
- `tol, maxdim, cutoff` : MPO truncation parameters passed to `apply` and `truncate!`.
- `printinfo`    : print `maxlinkdim` every 10 Chebyshev steps (default `false`).

# Returns
`Matrix{Float64}` of shape `(Nω, num_x)`.
"""
function get_bands(H_mpo::MPO, scale::Real, center::Real, sites,
                          Ncheb::Int, D::Int, ω_vals;
                          spin_proj::Bool   = false,
                          proj_s            = nothing,
                          spin_s_aux        = nothing,
                          nambu_proj::Bool  = false,
                          proj_nambu        = nothing,
                          nambu_s           = nothing,
                          nambu_side::Symbol  = :pre,
                          layer_proj::Bool  = false,
                          proj_layer        = nothing,
                          layer_s           = nothing,
                          layer_side::Symbol  = :pre,
                          sublattice::Bool  = false,
                          proj_sl           = nothing,
                          sublat_proj::Bool = false,
                          sublat_s          = nothing,
                          sublat_side::Symbol = :post,
                          k_groups_override   = nothing,
                          xmin::Int       = 0,
                          xmax            = nothing,
                          num_x::Int      = 10,
                          num_avg::Int    = 1,
                          ymin::Int       = 0,
                          ymax            = nothing,
                          num_y::Int      = 10,
                          kernel::Symbol  = :jackson,
                          lambda::Real    = 4.0,
                          tol::Real       = 1e-9,
                          maxdim::Int     = 100,
                          cutoff::Real    = 1e-10,
                          printinfo::Bool = false)

    L = length(sites)
    # Nambu and sublattice are always internal aux DOFs, never position qubits.
    # Spin is subtracted only when spin_proj=true (it may be part of the physical encoding).
    L_pos = L - (spin_proj ? 1 : 0) - (!isnothing(nambu_s) ? 1 : 0) -
                (!isnothing(layer_s) ? 1 : 0) - (!isnothing(sublat_s) ? 1 : 0)
    N     = 2^L_pos

    # ── Scaled Hamiltonian ────────────────────────────────────────────────────
    # sites already includes all aux sites; MPO(sites, "Id") is correctly sized.
    I_mpo = MPO(sites, "Id")
    Ham_n = (1 / scale) * +(H_mpo, (-center) * I_mpo; cutoff = cutoff)

    # ── KPM weight matrix  W[n, iω] ──────────────────────────────────────────
    Nω    = length(ω_vals)
    valid = [abs(ω) < 1.0 for ω in ω_vals]
    W     = _kpm_weight_matrix(Ncheb, ω_vals; kernel = kernel, lambda = lambda)

    # Lx is needed for both the 2D k-group builder and the sublattice mask builder;
    # compute it unconditionally so it is always in scope when D==2.
    Lx = D == 2 ? div(L_pos, 2) : 0

    # ── Build k-point groups ──────────────────────────────────────────────────
    # k_groups_override (from kpath_2d) bypasses the grid sampling entirely.
    if !isnothing(k_groups_override)
        k_groups = k_groups_override
        num_x    = length(k_groups)
    elseif D == 1
        _xmax     = xmax === nothing ? N - 1 : Int(xmax)
        xcenters  = ilinspace(xmin, _xmax, num_x)
        half_step = num_x > 1 ? (_xmax - xmin) / (2 * num_x) : 0
        offsets   = num_avg > 1 ? round.(Int, range(-half_step, half_step; length=num_avg)) : Int[0]
        k_groups  = [clamp.(xcenters[i] .+ offsets, 0, N - 1) for i in 1:num_x]
    elseif D == 2
        Lx     = div(L_pos, 2)   # also computed above; repeated here keeps the branch self-contained
        Nx_loc = 2^Lx
        Ny_loc = 2^(L_pos - Lx)
        num_x  = min(num_x, Nx_loc)   # can't have more output pts than grid positions
        _xmax  = xmax === nothing ? Nx_loc - 1 : Int(xmax)
        _ymax  = ymax === nothing ? Ny_loc - 1 : Int(ymax)
        xcenters    = ilinspace(xmin, _xmax, Nx_loc)
        ycenters    = ilinspace(ymin, _ymax, Ny_loc)
        half_step_x = num_x > 1 ? (_xmax - xmin) / (2 * num_x) : 0
        half_step_y = num_y > 1 ? (_ymax - ymin) / (2 * num_y) : 0
        x_offs = num_avg > 1 ? round.(Int, range(-half_step_x, half_step_x; length=num_avg)) : Int[0]
        y_offs = num_avg > 1 ? round.(Int, range(-half_step_y, half_step_y; length=num_avg)) : Int[0]
        k_groups = [
            begin
                xs = clamp.(xcenters[i] .+ x_offs, 0, Nx_loc - 1)
                ys = clamp.(ycenters[i] .+ y_offs, 0, Ny_loc - 1)
                [(y << Lx) | x for (x, y) in zip(xs, ys)]  # diagonal zip in 2D k-space
            end
            for i in 1:num_x
        ]
    else
        error("D must be 1 or 2")
    end

    Ak_w = zeros(Float64, Nω, num_x)

    # ── Precompute sublattice masks once (reused every Chebyshev step) ────────
    # Masks are applied to the position-only MPO (after all aux projections),
    # so they must be built from the position sites only.
    # pos_sites = position qubits only, for legacy sublattice mask building.
    # All known aux indices are excluded regardless of whether their projection
    # is active — a "Kagome" or "Spin" tagged index must never reach OpSum.
    aux_to_drop = Set{Index}()
    spin_proj             && push!(aux_to_drop, sites[1])
    !isnothing(nambu_s)   && push!(aux_to_drop, nambu_s::Index)
    !isnothing(layer_s)   && push!(aux_to_drop, layer_s::Index)
    !isnothing(sublat_s)  && push!(aux_to_drop, sublat_s::Index)
    pos_sites = filter(s -> s ∉ aux_to_drop, sites)
    if sublattice
        if D == 1
            mask_A = _col_select_mpo(L_pos, 0, pos_sites; keep=:odd)   # even sites (ix % 2 == 0)
            mask_B = _col_select_mpo(L_pos, 0, pos_sites; keep=:even)  # odd  sites (ix % 2 == 1)
        else
            Ly = L_pos - Lx
            mask_A = _row_checker_mpo(Lx, Ly, pos_sites)                           # (ix+iy) even
            mask_B = MPO(pos_sites, "Id") - _row_checker_mpo(Lx, Ly, pos_sites)    # (ix+iy) odd
        end
    end

    # ── Online accumulation: project → QFT → sample → accumulate ─────────────
    # Four independent, composable projection steps build a list of position
    # MPOs; every MPO in the list is QFT'd, sampled, and its contribution summed.
    #
    #  Step 0  nambu_proj         → project aux Nambu (BdG) index  (×1 or ×2)
    #  Step 1  spin_proj          → project aux spin index          (×1 or ×2)
    #  Step 1c layer_proj         → project layer index             (×1 … ×n_layers)
    #  Step 1b sublat_proj        → project aux sublattice index    (×1 … ×dim)
    #  Step 2  sublattice (legacy)→ apply mask sandwich             (×1 or ×2)
    #
    # Aux sites are projected in outermost-first order (nambu → spin → layer →
    # sublat).  After each removal the next aux moves to position 1 of the
    # reduced MPO, so project_aux(:pre) always lands on the right site.
    local _nambu_side = nambu_side
    local _layer_side = layer_side
    local _sublat_side = sublat_side
    local _spin_idx    = isnothing(spin_s_aux) ? sites[1] : spin_s_aux
    function accumulate_Tn!(ak_accum, Tn, n)
        # Step 0: Nambu (BdG particle/hole) projection — outermost aux, project first.
        # proj_nambu=nothing → sum particle+hole; proj_nambu=1/2 → select one sector.
        after_nambu = nambu_proj ? [project_aux(Tn, nambu_s::Index, sec; side=_nambu_side)
                                    for sec in (isnothing(proj_nambu) ? (1:2) : (proj_nambu:proj_nambu))] : MPO[Tn]

        # Step 1: spin aux projection.
        # Uses spin_s_aux (explicit Index) when provided, falls back to sites[1].
        # proj_s=nothing → sum both channels; proj_s=1/2 → select one.
        after_spin = spin_proj ? [project_aux(T, _spin_idx, sec; side=:pre)
                                  for T in after_nambu, sec in (isnothing(proj_s) ? (1:2) : (proj_s:proj_s))] : after_nambu

        # Step 1c: layer projection (bilayer / multilayer with H.layer_s).
        # proj_layer=nothing → sum all layers; proj_layer=k → select layer k.
        after_layer = if layer_proj
            n_lay = dim(layer_s::Index)
            lay_range = isnothing(proj_layer) ? (1:n_lay) : (proj_layer:proj_layer)
            [project_aux(T, layer_s::Index, sec; side=_layer_side)
             for T in after_spin for sec in lay_range]
        else
            after_spin
        end

        # Step 1b: sublattice aux projection (kagome/Lieb/honeycomb with H.sublattice_s).
        # proj_sl=nothing → sum all sublattices; proj_sl=k → select sublattice k.
        after_sl_aux = if sublat_proj
            sl_range = isnothing(proj_sl) ? (1:dim(sublat_s::Index)) : (proj_sl:proj_sl)
            [project_aux(T, sublat_s::Index, sec; side=_sublat_side)
             for T in after_layer for sec in sl_range]
        else
            after_layer
        end

        # Step 2: legacy sublattice mask projection (for 2-sublattice models without aux index)
        # proj_sl=nothing applies both masks; proj_sl=1/2 selects one.
        if sublattice
            masks = isnothing(proj_sl) ? [mask_A, mask_B] :
                    proj_sl == 1       ? [mask_A]          : [mask_B]
            sl_mpas = MPO[]
            for T in after_sl_aux, mask in masks
                push!(sl_mpas, apply(apply(mask, T; cutoff=cutoff, maxdim=maxdim), mask; cutoff=cutoff, maxdim=maxdim))
            end
        else
            sl_mpas = after_sl_aux
        end

        # Step 3: QFT + diagonal sample + accumulate for every MPO in the list
        for T in sl_mpas
            Tn_k  = conjugate_by_qft(T; tol=tol, maxdim=maxdim)
            A_mps = ITensorMPS.truncate!(extract_diagonal_to_mps(Tn_k); cutoff=cutoff)
            for (ik, xs) in enumerate(k_groups)
                s = sum(_eval_diag_mps(A_mps, x) for x in xs) / length(xs)
                for ie in 1:Nω
                    ak_accum[ie, ik] += W[n, ie] * s
                end
            end
        end
    end

    # ── Chebyshev recurrence  T_0 = I,  T_1 = H̃,  T_n = 2H̃T_{n-1} − T_{n-2}
    # The recurrence runs on the full MPO space (L+1 sites when spin_proj=true).
    # Projection happens inside accumulate_Tn! so T_n itself is never modified.
    Tkm2 = I_mpo   # T_0
    Tkm1 = Ham_n   # T_1

    accumulate_Tn!(Ak_w, Tkm2, 1)
    accumulate_Tn!(Ak_w, Tkm1, 2)

    for k in 3:Ncheb
        Tk = +(2 * apply(Ham_n, Tkm1; cutoff=cutoff), -Tkm2; maxdim=maxdim)
        Tk = ITensorMPS.truncate!(Tk; cutoff=cutoff)
        accumulate_Tn!(Ak_w, Tk, k)
        Tkm2 = Tkm1
        Tkm1 = Tk
        printinfo && (k % 10 == 0 || k == Ncheb) &&
            println("Online KPM step $k/$Ncheb  maxlinkdim=$(maxlinkdim(Tkm1))")
    end

    # ── Normalization: divide by the KPM DOS weight ───────────────────────────
    for iω in 1:Nω
        valid[iω] || continue
        Ak_w[iω, :] ./= (π^2 * Ncheb * sqrt(1 - ω_vals[iω]^2))
    end

    return Ak_w
end


"""
    get_bands(H, Ncheb, D, ω_phys_vals; kwargs...)
        -> Matrix{Float64}  or  NamedTuple(Ak, ticks, labels)

High-level overload of `get_bands` for a `TBHamiltonian`.

Physical energies `ω_phys_vals` are rescaled via `H.scale` and `H.center`.

**Auto-detection:** All auxiliary site Indices (Nambu, spin, layer, sublattice)
and their positions (:pre/:post) are read from the struct fields and excluded
from position k-space automatically — no manual index passing required.

**Projection kwargs** (forwarded verbatim to the low-level MPO method):
`spin_proj`, `proj_s`, `nambu_proj`, `proj_nambu`, `layer_proj`, `proj_layer`,
`sublat_proj`, `proj_sl`, `sublattice`.

**High-symmetry k-path shortcut** — replaces the manual `hsk_*` + `kpath_2d`
+ `k_groups_override` boilerplate with a single call:

```julia
res = get_bands(H, Ncheb, 2, omega;
                kpath=[:G, :M, :Kp, :G], kpath_lattice=:honeycomb, num_x=30)
```

- `kpath`         : symbol vector defining the path.  Use Latin aliases:
                    `G`=Γ, `M`, `K`, `Kp`=K', `X` — no special characters needed.
- `kpath_lattice` : `:honeycomb`, `:square`, or `:triangular`.
- `kpath_Lx`      : Lx for the 2D grid; defaults to `H.L ÷ 2`.
- `num_x`         : **reused as `npts_per_segment`** when `kpath` is given.

When `kpath` is set the return value is a `NamedTuple`:
  `(Ak = Matrix{Float64}(Nω×Nk),  ticks = Vector{Int},  labels = Vector{String})`

  ```julia
  heatmap(1:size(res.Ak,2), omega, res.Ak; xticks=(res.ticks, res.labels))
  vline!(p, res.ticks; ls=:dash, color=:white)
  ```

Otherwise returns `Matrix{Float64}` as usual (backward-compatible).
"""
function get_bands(H::TBHamiltonian, Ncheb::Int, D::Int, ω_phys_vals;
                          kpath             = nothing,
                          kpath_lattice     = nothing,
                          kpath_Lx          = nothing,
                          spin_proj::Bool   = false,
                          proj_s            = nothing,
                          nambu_proj::Bool  = false,
                          proj_nambu        = nothing,
                          layer_proj::Bool  = false,
                          proj_layer        = nothing,
                          sublattice::Bool  = false,
                          proj_sl           = nothing,
                          sublat_proj::Bool = false,
                          k_groups_override   = nothing,
                          xmin::Int       = 0,
                          xmax            = nothing,
                          num_x::Int      = 10,
                          num_avg::Int    = 1,
                          ymin::Int       = 0,
                          ymax            = nothing,
                          num_y::Int      = 10,
                          kernel::Symbol  = :jackson,
                          lambda::Real    = 4.0,
                          tol::Real       = 1e-9,
                          maxdim::Int     = 100,
                          cutoff::Real    = 1e-10,
                          printinfo::Bool = false)

    _ensure_scale!(H)
    nambu_proj, spin_proj, layer_proj, sublat_proj =
        _autoenable_proj(H, nambu_proj, spin_proj, layer_proj, sublat_proj)

    ω_resc = (collect(ω_phys_vals) .- H.center) ./ H.scale

    # ── High-symmetry path shortcut ──────────────────────────────────────────
    # When `kpath` is provided, build k_groups_override from symbols and use
    # num_x as npts_per_segment.  Returns a NamedTuple with tick info.
    kpath_ticks = nothing; kpath_labels = nothing
    if !isnothing(kpath)
        isnothing(kpath_lattice) && error(
            "kpath requires kpath_lattice (:honeycomb, :square, or :triangular).")
        Lx_kp = isnothing(kpath_Lx) ? H.L ÷ 2 : Int(kpath_Lx)
        Ly_kp = H.L - Lx_kp
        k_groups_override, kpath_ticks, kpath_labels =
            kpath_setup(kpath_lattice, Lx_kp, Ly_kp, kpath; npts_per_segment = num_x)
    end

    # Auto-detect all aux indices so the low-level function can exclude them
    # from L_pos and pos_sites regardless of which projections are active.
    nambu_s_det, nambu_side_det = !isnothing(H.nambu_s) ?
        aux_site(H, :nambu) : (nothing, :pre)

    spin_s_det = H.spin_s   # may be nothing; low-level falls back to sites[1] when nothing

    layer_s_det, layer_side_det = !isnothing(H.layer_s) ?
        aux_site(H, :layer) : (nothing, :pre)

    sublat_s_det, sublat_side_det = !isnothing(H.sublattice_s) ?
        aux_site(H, :sublattice) : (nothing, :post)

    Ak_w = get_bands(H.mpo, H.scale, H.center, H.sites, Ncheb, D, ω_resc;
                            spin_proj  = spin_proj,  proj_s     = proj_s,
                            spin_s_aux = spin_s_det,
                            nambu_proj = nambu_proj, proj_nambu = proj_nambu,
                            nambu_s    = nambu_s_det, nambu_side = nambu_side_det,
                            layer_proj = layer_proj, proj_layer  = proj_layer,
                            layer_s    = layer_s_det, layer_side = layer_side_det,
                            sublattice = sublattice, proj_sl    = proj_sl,
                            sublat_proj = sublat_proj,
                            sublat_s    = sublat_s_det,
                            sublat_side = sublat_side_det,
                            k_groups_override = k_groups_override,
                            xmin = xmin, xmax = xmax,
                            num_x = num_x, num_avg = num_avg,
                            ymin = ymin, ymax = ymax, num_y = num_y,
                            kernel = kernel, lambda = lambda,
                            tol = tol, maxdim = maxdim, cutoff = cutoff,
                            printinfo = printinfo)

    # When kpath was used, return a NamedTuple carrying the tick info so the
    # caller can use result.Ak, result.ticks, result.labels directly in plots.
    return isnothing(kpath_ticks) ? Ak_w :
           (Ak = Ak_w, ticks = kpath_ticks, labels = kpath_labels)
end


"""
    get_bands(H, Ncheb, ω_phys_vals; kwargs...)

Convenience overload that infers the spatial dimension `D` from `H.geometry`
(via `length(H.geometry(1))`), so callers do not need to pass `D` explicitly.
All keyword arguments are forwarded unchanged to the 4-argument form.

Errors if `H.geometry` is `nothing` (custom or geometry-free Hamiltonians must
still pass `D` explicitly via the 4-argument form).
"""
function get_bands(H::TBHamiltonian, Ncheb::Int, ω_phys_vals; kwargs...)
    isnothing(H.geometry) &&
        error("get_bands without explicit D requires H.geometry to be set. " *
              "Pass D (1 or 2) as the third argument, or set H.geometry.")
    D = length(H.geometry(1))
    return get_bands(H, Ncheb, D, ω_phys_vals; kwargs...)
end


# ============================================================
# 4b. Auxiliary index projection utilities
#
# Any auxiliary DOF (spin, Nambu, layer, sublattice) added with prepend_op /
# postpend_op lives at the first or last site of the MPO as a dim-1-bonded
# tensor.  The functions below implement the removal step used in Steps 0–1c
# of the get_bands projection pipeline.
#
# project_aux(W, aux_s, sec; side)
#   Contracts the projector |sec⟩⟨sec| onto the bra (aux_s') and ket (aux_s)
#   physical indices of the aux tensor.  The resulting dim-1 link is absorbed
#   into the adjacent position site, returning an (L−1)-site MPO.
#   `side=:pre` for prepended indices (spin, Nambu, layer);
#   `side=:post` for postpended indices (sublattice).
#
#   project_spin — convenience alias for the :pre case (spin is always prepended).
#
# aux_site(H, which) -> (Index, Symbol)
#   Extracts the auxiliary Index and its side (:pre or :post) from H.sites.
#   `which` ∈ :spin, :nambu, :layer, :sublattice.
#   Used by the TBHamiltonian overload to auto-detect all auxiliary indices
#   and pass them to the low-level get_bands without user intervention.
# ============================================================

"""
    project_aux(W, aux_s, σ; side=:pre) -> MPO

Remove an auxiliary site from MPO `W` by projecting onto state `σ`.

- `side=:pre`  — aux site is at position 1 (prepended, e.g. spin).
- `side=:post` — aux site is at the last position (postpended, e.g. sublattice).

Contracts the projector |σ⟩⟨σ| on both bra and ket physical indices of the
aux tensor; the resulting dim-1 link is absorbed into the adjacent position
site.  Returns an (L−1)-site MPO suitable for `conjugate_by_qft`.
"""
function project_aux(W::MPO, aux_s::Index, σ::Integer; side::Symbol = :pre)
    L        = length(W)
    pos      = side === :pre ? 1 : L
    aux_proj = W[pos] * setelt(aux_s' => σ) * setelt(aux_s => σ)
    new_tensors = Vector{ITensor}(undef, L - 1)
    if side === :pre
        new_tensors[1] = W[2] * aux_proj
        for i in 2:L-1; new_tensors[i] = W[i+1]; end
    else  # :post
        for i in 1:L-2; new_tensors[i] = W[i]; end
        new_tensors[L-1] = W[L-1] * aux_proj
    end
    return MPO(new_tensors)
end

# Nothing-overloads: give Julia a compilable method when the Index is nothing,
# so branches in get_bands can be type-checked without a MethodError.
project_aux(::MPO, ::Nothing, ::Integer; side::Symbol=:pre) =
    error("sublat_proj=true requires sublat_s to be set (detected from H.sublattice_s)")

# Convenience alias — spin is always prepended (:pre)
"""
    project_spin(W, spin_s, σ) -> MPO

Convenience wrapper for `project_aux` when the auxiliary site is prepended
(spin at site 1).  Equivalent to `project_aux(W, spin_s, σ; side=:pre)`.
"""
project_spin(W::MPO, spin_s::Index,   σ::Integer) = project_aux(W, spin_s, σ; side=:pre)
project_spin(W::MPO, ::Nothing, ::Integer) =
    error("spin_proj=true requires spin_s — detected via sites[1] when spin_proj=true")


"""
    _autoenable_proj(H, nambu_proj, spin_proj, layer_proj, sublat_proj)
        -> (nambu_proj, spin_proj, layer_proj, sublat_proj)

Enable projection flags for any auxiliary DOF detected on `H`, printing one
info line per auto-enabled flag.  Called at the top of every `TBHamiltonian`
spectral method before any aux-index logic runs.
"""
function _autoenable_proj(H::TBHamiltonian,
                           nambu_proj::Bool, spin_proj::Bool,
                           layer_proj::Bool, sublat_proj::Bool)
    if !isnothing(H.nambu_s) && !nambu_proj
        println("Info: H.nambu_s detected; auto-enabling nambu_proj=true ",
                "(pass proj_nambu=1/2 to select particle/hole sector).")
        nambu_proj = true
    end
    if !isnothing(H.spin_s) && !spin_proj
        println("Info: H.spin_s detected; auto-enabling spin_proj=true ",
                "(pass proj_s=1/2 to select ↑/↓ sector).")
        spin_proj = true
    end
    if !isnothing(H.layer_s) && !layer_proj
        println("Info: H.layer_s detected; auto-enabling layer_proj=true ",
                "(pass proj_layer=k to select a layer).")
        layer_proj = true
    end
    if !isnothing(H.sublattice_s) && !sublat_proj
        println("Info: H.sublattice_s detected; auto-enabling sublat_proj=true ",
                "(pass proj_sl=k to select a sublattice).")
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
# 5. Legacy — kept for reference, not part of the public API
# ============================================================

# ── get_spect_k / get_spect_k_doubled (inner-product approach, superseded) ──
# These evaluated the band structure by constructing basis MPS |k⟩ and
# computing ⟨k|Ã(ω)|k⟩ directly.  Correct but O(N) inner products per ω.
# Replaced by the diagonal-extraction approach in get_bands.

# function get_spect_k(W; tol=1e-9, maxdim::Int=100)
#     Akop  = conjugate_by_qft(W; tol=tol, maxdim=maxdim)
#     sites = getindex.(siteinds(W), 2)
#     L     = length(sites)
#     N     = 2^L
#     # LSB at site 1 — matches the quantics QFT convention (see KPM_LDOS_1D notebook)
#     lsb_state(k) = [string((k >> (i-1)) & 1) for i in 1:L]
#     mpsk(k) = MPS(sites, lsb_state(Int(k)))
#     kvals   = range(0, N - 1; length=N)
#     return [inner(mpsk(k)', Akop, mpsk(k)) for k in kvals]
# end

# function get_spect_k_doubled(W; tol=1e-9, maxdim::Int=100)
#     Akop  = conjugate_by_qft(W; tol=tol, maxdim=maxdim)
#     sites = getindex.(siteinds(W), 2)
#     L     = div(length(sites), 2)
#     N     = 2^L
#     lsb_state(k) = [string((k >> (i-1)) & 1) for i in 1:L]
#     mpsk(k)  = MPS(sites[1:L],    lsb_state(Int(k)))
#     mpsk1(k) = MPS(sites[L+1:2L], lsb_state(Int(k)))
#     mpsk2(k) = mps_kron(mpsk(k), mpsk1(k))
#     kvals    = range(0, N - 1; length=N)
#     return [inner(mpsk2(k)', Akop, mpsk2(k)) for k in kvals]
# end


#= ── OLD get_bands (offline inner-product approach) — kept for reference ─────
#
# Key difference from the current get_bands:
#   OLD: loop over ω first → for each ω build δ(ω−H) via get_ldos_w_from_Tn,
#        conjugate with QFT, then loop over all k to compute inner products.
#        Memory: O(1 MPO) per ω, but O(N) inner products × Nω evaluations.
#   NEW: loop over Chebyshev steps → project, QFT, sample diagonal, accumulate.
#        Memory: O(3 MPOs), no inner products, scales to large N.
#
# The TBHamiltonian overload below also handled auxiliary (spin/orbital) sites
# via prepend_op / postpend_op — that generality is not yet ported to get_bands.

function get_bands(Tn_list, Ncheb::Int, sites, ω_vals;
                   tol=1e-9, maxdim::Int=100)
    L   = length(sites)
    N   = 2^L
    Nω  = length(ω_vals)
    FTirev = fix_sites(MPO(TCI.reverse(
        QuanticsTCI.quanticsfouriermpo(L; sign=-1.0, normalize=true))), sites)
    FTrev  = fix_sites(MPO(TCI.reverse(
        QuanticsTCI.quanticsfouriermpo(L; sign=+1.0, normalize=true))), sites)
    lsb_state(k) = [string((k >> (i-1)) & 1) for i in 1:L]
    mpsk = [MPS(sites, lsb_state(k)) for k in 0:N-1]
    Ak_w = zeros(Float64, N, Nω)
    for (iω, ω) in enumerate(ω_vals)
        abs(ω) >= 1.0 && continue
        δH = get_ldos_w_from_Tn(Tn_list, Ncheb, ω; maxdim=maxdim)
        Op1  = apply(δH,                        FTirev; cutoff=tol, maxdim=maxdim)
        Akop = apply(swapprime(FTrev, 0 => 1),  Op1;   cutoff=tol, maxdim=maxdim)
        for k in 0:N-1
            Ak_w[k+1, iω] = real(inner(mpsk[k+1]', Akop, mpsk[k+1]))
        end
    end
    return Ak_w
end

function get_bands(H::TBHamiltonian, ω_phys_vals;
                   aux_proj = nothing, tol=1e-9, maxdim::Int=100)
    H._tn_cache === nothing &&
        error("No Chebyshev cache found.  Call KPM_Tn(H, Ncheb; ...) first.")
    pos_sites = _pos_sites(H)
    Lpos      = length(pos_sites)
    Npos      = 2^Lpos
    pos_set   = Set(pos_sites)
    aux_sites = filter(s -> s ∉ pos_set, H.sites)
    FTirev = fix_sites(MPO(TCI.reverse(
        QuanticsTCI.quanticsfouriermpo(Lpos; sign=-1.0, normalize=true))), pos_sites)
    FTrev  = fix_sites(MPO(TCI.reverse(
        QuanticsTCI.quanticsfouriermpo(Lpos; sign=+1.0, normalize=true))), pos_sites)
    if H.aux_side === :pre
        for s in reverse(aux_sites)
            Id = Matrix{Float64}(LinearAlgebra.I, dim(s), dim(s))
            FTirev = prepend_op(FTirev, s, Id)
            FTrev  = prepend_op(FTrev,  s, Id)
        end
    else
        for s in aux_sites
            Id = Matrix{Float64}(LinearAlgebra.I, dim(s), dim(s))
            FTirev = postpend_op(FTirev, s, Id)
            FTrev  = postpend_op(FTrev,  s, Id)
        end
    end
    aux_combos = if isnothing(aux_proj) || isempty(aux_sites)
        collect(Iterators.product((1:dim(s) for s in aux_sites)...))
    else
        proj = aux_proj isa Integer ? fill(Int(aux_proj), length(aux_sites)) :
                                      collect(Int, aux_proj)
        [Tuple(proj)]
    end
    pos_kvals(k) = [((k >> (i-1)) & 1) + 1 for i in 1:Lpos]
    all_sites_ord = H.aux_side === :pre ? [aux_sites; pos_sites] : [pos_sites; aux_sites]
    kmps = Dict{Any, Vector{MPS}}()
    for σ_combo in aux_combos
        σ_vals = collect(Int, σ_combo)
        states = Vector{MPS}(undef, Npos)
        for k in 0:Npos-1
            all_vals = H.aux_side === :pre ? [σ_vals; pos_kvals(k)] : [pos_kvals(k); σ_vals]
            states[k+1] = _product_state_mps(all_sites_ord, all_vals)
        end
        kmps[σ_combo] = states
    end
    ω_resc = (collect(ω_phys_vals) .- H.center) ./ H.scale
    Nω     = length(ω_resc)
    Ak_w   = zeros(Float64, Npos, Nω)
    for (iω, ω) in enumerate(ω_resc)
        abs(ω) >= 1.0 && continue
        δH   = get_ldos_w_from_Tn(H._tn_cache, H._tn_Ncheb, ω; maxdim=maxdim)
        Op1  = apply(δH,                       FTirev; cutoff=tol, maxdim=maxdim)
        Akop = apply(swapprime(FTrev, 0 => 1), Op1;   cutoff=tol, maxdim=maxdim)
        for σ_combo in aux_combos, k in 0:Npos-1
            psi = kmps[σ_combo][k+1]
            Ak_w[k+1, iω] += real(inner(psi', Akop, psi))
        end
    end
    return Ak_w
end
=# # ── END OLD get_bands ────────────────────────────────────────────────────────
