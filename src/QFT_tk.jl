# QFT_tk.jl — Momentum-space band structure via online Chebyshev KPM
#
# Overview
# --------
# The quantics representation encodes a 1D or 2D real-space index as a
# binary string across L qubit sites.  Conjugating any real-space MPO W
# by the Quantum Fourier Transform (QFT) gives its momentum-space image:
#
#   Ã(k,ω) = U · δ(ω − H) · U†
#
# The diagonal ⟨k|Ã(ω)|k⟩ is the k-resolved spectral function A(k,ω).
#
# Design: online Chebyshev accumulation
# --------------------------------------
# Instead of storing all Ncheb Chebyshev MPOs and post-processing them
# (the old "offline" approach), get_bands runs a single Chebyshev
# recurrence T_0, T_1, … T_Ncheb and, at each step n:
#   1. Optionally project T_n onto sublattice A and B.
#   2. Conjugate (each projected) T_n by the QFT.
#   3. Extract the diagonal as an MPS and evaluate at the chosen k-points.
#   4. Accumulate the KPM-weighted contribution into A(k,ω).
#
# Peak memory: O(3 MPOs) regardless of Ncheb.
#
# Encoding conventions
# --------------------
# 1D  — sites 1…L hold x bits, LSB at site 1 (quantics QFT convention).
# 2D  — sites 1…Ly hold iy bits (MSB first), sites Ly+1…L hold ix bits
#        (MSB first); linear index n = ix + iy * 2^Lx (row-major).
#        Sublattice masks from 2D_lattice.jl respect this convention. ## Note: not sure about this description,     
#                                                                              need to double-check the bit ordering in 2D and update docs accordingly.
#
# Dependencies outside this file
# --------------------------------
# fix_sites, _kpm_kernel               → utils.jl
# extract_diagonal_to_mps              → utils.jl
# _row_checker_mpo, _col_select_mpo    → 2D_lattice.jl
# TBHamiltonian, _ensure_scale!        → Hamiltonian.jl


# ============================================================
# 1. QFT conjugation
# ============================================================

"""
    conjugate_by_qft(W; tol=1e-9, maxdim=100) -> MPO

Return `U · W · U†` where `U` is the Quantum Fourier Transform MPO
built from `QuanticsTCI.quanticsfouriermpo`.

The `TCI.reverse` call places the LSB at site 1 to match the quantics
encoding used throughout this codebase. Note that the ordering in k-space is the reverse of the site ordering in real space
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


# ============================================================
# 2. Sublattice projection
#
# For now only works in 1D and 2D lattices using a checkerboard or alternating-site sublattice decomposition
#
# These wrappers project an MPO O onto one of two sublattices by
# sandwiching it with the corresponding diagonal mask MPO:
#   O_SL = mask · O · mask
#
# The mask MPOs (`_row_checker_mpo`, `_col_select_mpo`) are defined in
# 2D_lattice.jl and operate on a single qubit site each, so they are
# bond-dimension 1 and cheap to apply.
#
# 2D — checkerboard sublattices of a 2^Lx × 2^Ly lattice:
#   SL=1 → sites where (ix+iy) is even  (_row_checker_mpo)
#   SL=2 → sites where (ix+iy) is odd   (Id − _row_checker_mpo)
#
# 1D — alternating sublattices of a 2^L chain:
#   SL=1 → even sites (ix % 2 == 0)    (_col_select_mpo, keep=:odd)
#   SL=2 → odd  sites (ix % 2 == 1)    (_col_select_mpo, keep=:even)
#   Note: the `:odd`/`:even` labels in _col_select_mpo refer to the
#   parity of the LSB qubit state (0 or 1), not of the site index.
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
    _kpm_weight_matrix(Ncheb, ω_vals; kernel, lambda) -> Matrix{Float64}

Precompute the KPM weight matrix `W[n, iω]` for fast accumulation.

`W[n, iω] = c_n · g_n · cos((n-1) · arccos(ω_iω))`

where `c_n = 1` for n=1 and 2 otherwise, and `g_n` is the kernel
damping factor (Jackson or Lorentz).  Entries for `|ω| ≥ 1` are left
at zero.

Note: later could include HODC kernel
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
# 4. Online band structure  —  get_bands
#
# Algorithm (for each Chebyshev step n = 1 … Ncheb):
#   T_n = 2 H̃ T_{n-1} − T_{n-2}       (recurrence, H̃ = (H−center)/scale)
#   if sublattice:
#       T_n^A = mask_A · T_n · mask_A   (project onto sublattice A)
#       T_n^B = mask_B · T_n · mask_B   (project onto sublattice B)
#       for each sublattice s ∈ {A,B}:
#           T_n^s → QFT → diagonal MPS → sample at k_groups → accumulate
#   else:
#       T_n → QFT → diagonal MPS → sample at k_groups → accumulate
#
# k-point sampling mirrors projdos_from_Tn_mpsk:
#   num_x centers placed with ilinspace in [xmin, xmax]; each center
#   is averaged over num_avg nearby offset points (±half_step).
#   2D: diagonal zip of x-offsets and y-offsets, combined via (iy<<Lx)|ix.
# ============================================================

"""
    get_bands(H_mpo, scale, center, sites, Ncheb, D, ω_vals; kwargs...) -> Matrix{Float64}

Memory-efficient band structure via online Chebyshev KPM accumulation.

# Arguments
- `H_mpo`      : real-space Hamiltonian MPO (unscaled).
- `scale, center` : energy rescaling so that H̃ = (H−center)/scale ∈ (−1,1).
- `sites`      : ITensor site indices (Qubit).
- `Ncheb`      : number of Chebyshev moments.
- `D`          : spatial dimension (1 or 2).
- `ω_vals`     : rescaled energies ∈ (−1, 1) at which to evaluate A(k,ω).

# Keyword arguments
- `sublattice` : if `true`, project each T_n onto sublattice A and B
                 before the QFT and sum contributions (default `false`).
- `xmin, xmax, num_x` : k-point grid in the x (1D) or kx (2D) direction.
- `ymin, ymax, num_y` : k-point grid in the ky direction (2D only).
- `num_avg`    : number of offset points averaged around each center.
- `kernel`     : KPM broadening kernel (`:jackson` or `:lorentz`).
- `lambda`     : Lorentz kernel width (only used when `kernel=:lorentz`).
- `tol, maxdim, cutoff` : MPO truncation parameters.
- `printinfo`  : print progress every 10 Chebyshev steps.

# Returns
`Matrix{Float64}` of shape `(Nω, num_x)`.
"""
function get_bands(H_mpo::MPO, scale::Real, center::Real, sites,
                          Ncheb::Int, D::Int, ω_vals;
                          sublattice::Bool = false,
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
    N = 2^L

    # ── Scaled Hamiltonian ────────────────────────────────────────────────────
    I_mpo = MPO(sites, "Id")
    Ham_n = (1 / scale) * +(H_mpo, (-center) * I_mpo; cutoff = cutoff)

    # ── KPM weight matrix  W[n, iω] ──────────────────────────────────────────
    Nω    = length(ω_vals)
    valid = [abs(ω) < 1.0 for ω in ω_vals]
    W     = _kpm_weight_matrix(Ncheb, ω_vals; kernel = kernel, lambda = lambda)

    # ── Build k-point groups (mirrors projdos_from_Tn_mpsk sampling) ─────────
    if D == 1
        _xmax     = xmax === nothing ? N - 1 : Int(xmax)
        xcenters  = ilinspace(xmin, _xmax, num_x)
        half_step = num_x > 1 ? (_xmax - xmin) / (2 * num_x) : 0
        offsets   = num_avg > 1 ? round.(Int, range(-half_step, half_step; length=num_avg)) : Int[0]
        k_groups  = [clamp.(xcenters[i] .+ offsets, 0, N - 1) for i in 1:num_x]
    elseif D == 2
        Lx     = div(L, 2)
        Nx_loc = 2^Lx
        Ny_loc = 2^(L - Lx)
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
    # Masks are low bond-dim diagonal MPOs; building them once is negligible cost.
    if sublattice
        if D == 1
            mask_A = _col_select_mpo(L, 0, sites; keep=:odd)   # even sites (ix % 2 == 0)
            mask_B = _col_select_mpo(L, 0, sites; keep=:even)  # odd  sites (ix % 2 == 1)
        else
            Ly = L - Lx
            mask_A = _row_checker_mpo(Lx, Ly, sites)                       # (ix+iy) even
            mask_B = MPO(sites, "Id") - _row_checker_mpo(Lx, Ly, sites)    # (ix+iy) odd
        end
    end

    # ── Online accumulation: project → QFT → sample → accumulate ─────────────
    # Single function handles both the plain and sublattice-projected cases.
    # The sublattice sum sA+sB is folded into `s` before accumulation so only
    # one pass over the (Nω × num_x) accumulator is needed per Chebyshev step.
    function accumulate_Tn!(ak_accum, Tn, n)
        if sublattice
            Tn_A = apply(apply(mask_A, Tn; cutoff=cutoff, maxdim=maxdim), mask_A; cutoff=cutoff, maxdim=maxdim)
            Tn_B = apply(apply(mask_B, Tn; cutoff=cutoff, maxdim=maxdim), mask_B; cutoff=cutoff, maxdim=maxdim)
            Tn_k_A = conjugate_by_qft(Tn_A; tol=tol, maxdim=maxdim)
            Tn_k_B = conjugate_by_qft(Tn_B; tol=tol, maxdim=maxdim)
            A_mps_A = ITensorMPS.truncate!(extract_diagonal_to_mps(Tn_k_A); cutoff=cutoff)
            A_mps_B = ITensorMPS.truncate!(extract_diagonal_to_mps(Tn_k_B); cutoff=cutoff)
            for (ik, xs) in enumerate(k_groups)
                s = sum(_eval_diag_mps(A_mps_A, x) + _eval_diag_mps(A_mps_B, x) for x in xs) / length(xs)
                for ie in 1:Nω
                    ak_accum[ie, ik] += W[n, ie] * s
                end
            end
        else
            Tn_k  = conjugate_by_qft(Tn; tol=tol, maxdim=maxdim)
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
    # A single recurrence is run regardless of sublattice; projection happens
    # inside accumulate_Tn! so T_n itself is never modified.
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
    get_bands(H, Ncheb, D, ω_phys_vals; kwargs...) -> Matrix{Float64}

High-level overload of `get_bands` for a `TBHamiltonian`.

Physical energies `ω_phys_vals` are rescaled automatically using
`H.scale` and `H.center`.  All keyword arguments are forwarded to the
low-level MPO method.  See that method for full documentation.
"""
function get_bands(H::TBHamiltonian, Ncheb::Int, D::Int, ω_phys_vals;
                          sublattice::Bool= false,
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
    ω_resc = (collect(ω_phys_vals) .- H.center) ./ H.scale

    return get_bands(H.mpo, H.scale, H.center, H.sites, Ncheb, D, ω_resc;
                            sublattice = sublattice, xmin = xmin, xmax = xmax,
                            num_x = num_x, num_avg = num_avg,
                            ymin = ymin, ymax = ymax, num_y = num_y,
                            kernel = kernel, lambda = lambda,
                            tol = tol, maxdim = maxdim, cutoff = cutoff,
                            printinfo = printinfo)
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
