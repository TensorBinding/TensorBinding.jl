# QPI_tk.jl — Quasiparticle Interference via single-impurity scattering
#
# Pipeline:
#   1. Build impurity potential V·|x₀⟩⟨x₀| analytically via OpSum (bond dim 1).
#   2. Compute diagonal LDOS MPS for clean and perturbed Hamiltonians via KPM.
#   3. Subtract: δA_mps = ldos_imp - ldos_clean  (real-space diagonal LDOS difference).
#   4. Optionally apply a smooth disk window (apodization) to suppress edge effects.
#   5. Apply forward QFT MPO directly to δA_mps → δÃ_mps  (no conjugate_by_qft needed:
#      we already have the diagonal, not a full MPO).
#   6. QPI(k, ω) = |⟨k|δÃ_mps⟩|².
#
# Dependencies:
#   central_index, _pos_sites, _invalidate_cache!  → TBSystem.jl
#   KPM_Tn, get_ldos_spectrum                      → KPM_tk.jl
#   fix_sites                                      → Utils.jl
#   sdf_disk, sdf_interval                         → Flake_tk.jl
#   QuanticsTCI.quanticsfouriermpo, TCI.reverse    → external


"""
    _impurity_mpo(x0, L, sites, V) -> MPO

Build the on-site impurity potential `V · |x₀−1⟩⟨x₀−1|` as a bond-dimension-1 MPO
using `OpSum` — exact, no QTCI required.

`x0` is 1-indexed.  The quantics encoding is big-endian (matching `binary_to_MPS`):
site `i` carries bit `(L−i)` of the 0-indexed address `x0−1`, so site 1 holds the MSB.

Uses `"projUp"` (|0⟩⟨0|) and `"projDn"` (|1⟩⟨1|) operators defined for `SiteType"Qubit"`.
"""
function _impurity_mpo(x0::Int, L::Int, sites::Vector{<:Index}, V::Real)
    n      = x0 - 1   # 0-indexed quantics address
    os     = OpSum()
    op_seq = Any[V]
    for i in 1:L
        b = (n >> (L - i)) & 1   # big-endian: site 1 = MSB (bit L-1)
        push!(op_seq, b == 0 ? "projUp" : "projDn", i)
    end
    os += tuple(op_seq...)
    return MPO(os, sites)
end


"""
    get_qpi(H, Ncheb, ω_phys_vals;
            impurity_site=nothing, V=1.0,
            window_fraction=nothing, window_sigma=1.5,
            kernel=:jackson, lambda=4.0,
            maxdim=100, cutoff=1e-8, verbose=false)
        -> Matrix{Float64}

Compute the Quasiparticle Interference (QPI) pattern from a single on-site impurity.

**Algorithm**

1. Place impurity `V · |x₀−1⟩⟨x₀−1|` at site `x0` (default: geometric center via
   `central_index`).  The impurity MPO is built analytically from `OpSum` (no QTCI).
2. Run KPM Chebyshev expansion for the clean (`H`) and perturbed (`H_imp`) systems
   and extract the site-resolved diagonal LDOS as `Vector{MPS}` via `get_ldos_spectrum`.
3. For each energy `ω`:
   - Subtract LDOS MPS: `δA_mps = ldos_imp[ω] − ldos_clean[ω]`.
   - If `window_fraction` is set, multiply `δA_mps` element-wise by a smooth disk mask
     (built via QTCI using the same SDF machinery as `mask_hamiltonian`).  The mask
     equals 1 deep in the bulk and rolls off to 0 at the boundary over `window_sigma`
     lattice units, suppressing Gibbs-like ringing from open edges.
   - Apply the forward QFT MPO (position → momentum) **directly** to `δA_mps`.
     This works because we already have the diagonal of the spectral function as an MPS;
     no `conjugate_by_qft` (which operates on full MPOs) is needed.
   - Read out `QPI(k, ω) = |⟨k|δÃ_mps⟩|²` using LSB-first bit encoding
     (site 1 = bit 0), consistent with the `TCI.reverse` QFT convention.

**Arguments**
- `H`            : `TBHamiltonian` with no auxiliary DOFs (spinless, no Nambu/layer/sublattice).
- `Ncheb`        : Number of Chebyshev moments for KPM.
- `ω_phys_vals`  : Physical energies to evaluate.

**Keyword arguments**
- `impurity_site`   : 1-indexed site index for the impurity (default: `central_index(H)`).
- `V`               : Impurity potential strength.
- `impurity_mode`   : `:delta` (default) — exact rank-1 projector `V·|x₀⟩⟨x₀|` via `OpSum`;
                      `:gaussian` — smooth Gaussian potential `V·exp(-|r−r₀|²/2σ²)` built
                      via `add_onsite!` (lower bond dimension for the KPM recursion).
                      Requires `H.geometry` to be set when `:gaussian`.
- `sigma`           : Gaussian half-width in physical distance units (only used when
                      `impurity_mode=:gaussian`).  Default: `1.5`.
- `window_fraction` : If set, build a circular (2D) or interval (1D) apodization mask
                      with radius `window_fraction × min_half_extent` of the bounding box.
                      Requires `H.geometry` to be set.  Default: `nothing` (no windowing).
- `window_sigma`    : Sigmoid roll-off half-width in lattice units.  Default: `1.5`.
- `kernel`          : KPM kernel (`:jackson` or `:lorentz`).
- `lambda`          : Lorentz kernel width.
- `maxdim`          : Maximum bond dimension for MPS/MPO operations.
- `cutoff`          : SVD truncation cutoff.
- `verbose`         : Print progress.

**Returns** `Matrix{Float64}` of shape `(Nω, N)` where `N = H.N`.
Column index `k+1` (1-based) corresponds to 0-indexed momentum `k`.

**Example**
```julia
H     = get_Hamiltonian("square_2d", 1.0; L=8, Lx=4)
ωlist = [0.0]
qpi   = get_qpi(H, 80, ωlist; V=2.0, window_fraction=0.8)
# Reshape to 2D: reshape(qpi[1,:], 2^Lx, 2^Ly)'
```
"""
function get_qpi(H::TBHamiltonian, Ncheb::Int, ω_phys_vals;
                 impurity_site::Union{Int,Nothing}  = nothing,
                 V::Real             = 1.0,
                 impurity_mode::Symbol = :delta,
                 sigma::Real         = 1.5,
                 window_fraction::Union{Real,Nothing} = nothing,
                 window_sigma::Real  = 1.5,
                 kernel::Symbol      = :jackson,
                 lambda::Real        = 4.0,
                 maxdim::Int         = 100,
                 cutoff::Real        = 1e-8,
                 verbose::Bool       = false)

    # ── Sanity: no aux DOFs ───────────────────────────────────────────────────
    (H.spin_s !== nothing || H.nambu_s !== nothing ||
     H.layer_s !== nothing || H.sublattice_s !== nothing) &&
        error("get_qpi: auxiliary DOFs (spin, Nambu, layer, sublattice) are not supported. " *
              "Call get_qpi on the position-only Hamiltonian before add_spin! etc.")

    # ── Impurity site (1-indexed) ─────────────────────────────────────────────
    x0 = isnothing(impurity_site) ? central_index(H) : impurity_site
    (x0 < 1 || x0 > H.N) &&
        error("impurity_site=$x0 out of range [1, $(H.N)].")

    impurity_mode ∈ (:delta, :gaussian) ||
        error("get_qpi: impurity_mode must be :delta or :gaussian, got :$impurity_mode.")
    (impurity_mode === :gaussian && isnothing(H.geometry)) &&
        error("get_qpi: impurity_mode=:gaussian requires H.geometry to be set.")

    verbose && println("QPI: impurity at site $x0 / $(H.N),  V=$V,  mode=:$impurity_mode,  Ncheb=$Ncheb")

    # ── Build perturbed Hamiltonian ───────────────────────────────────────────
    H_imp = deepcopy(H)
    if impurity_mode === :delta
        imp_mpo   = _impurity_mpo(x0, H.L, H.sites, V)
        H_imp.mpo = +(H_imp.mpo, imp_mpo; cutoff=cutoff)
    else  # :gaussian
        r0 = collect(Float64, H.geometry(x0))
        gauss_fn(n) = V * exp(-sum((collect(Float64, H.geometry(n + 1)) .- r0).^2) / (2sigma^2))
        add_onsite!(H_imp, gauss_fn; tol=cutoff, maxdim=maxdim)
    end
    _invalidate_cache!(H_imp)

    # ── Online KPM: Chebyshev recursion + diagonal LDOS accumulation ──────────
    # Only 3 MPOs live at a time per system; no full Tn cache stored.
    _ensure_scale!(H)
    _ensure_scale!(H_imp)

    I_mpo      = MPO(H.sites, "Id")
    Ham_clean  = (1/H.scale)     * +(H.mpo,     (-H.center)     * I_mpo; cutoff=cutoff)
    Ham_imp_sc = (1/H_imp.scale) * +(H_imp.mpo, (-H_imp.center) * I_mpo; cutoff=cutoff)

    ω_clean = (collect(ω_phys_vals) .- H.center)     ./ H.scale
    ω_imp   = (collect(ω_phys_vals) .- H_imp.center) ./ H_imp.scale
    Nω_loc  = length(ω_phys_vals)
    W_c     = _kpm_weight_matrix(Ncheb, ω_clean; kernel=kernel, lambda=lambda)
    W_i     = _kpm_weight_matrix(Ncheb, ω_imp;   kernel=kernel, lambda=lambda)
    valid_c = [abs(ω) < 1.0 for ω in ω_clean]
    valid_i = [abs(ω) < 1.0 for ω in ω_imp]

    ldos_clean = Vector{Union{Nothing, MPS}}(nothing, Nω_loc)
    ldos_imp   = Vector{Union{Nothing, MPS}}(nothing, Nω_loc)

    function _accum!(acc, diag_n, W, valid, n)
        for iω in 1:Nω_loc
            valid[iω] || continue
            w = W[n, iω]; iszero(w) && continue
            acc[iω] = acc[iω] === nothing ? w * diag_n :
                ITensorMPS.truncate!(+(acc[iω], w * diag_n; maxdim=maxdim); cutoff=cutoff)
        end
    end

    function _run_online_ldos!(acc, Ham_sc, W, valid, ω_vals, label)
        Tkm2 = I_mpo;  Tkm1 = Ham_sc
        _accum!(acc, ITensorMPS.truncate!(extract_diagonal_to_mps(Tkm2); cutoff=cutoff), W, valid, 1)
        _accum!(acc, ITensorMPS.truncate!(extract_diagonal_to_mps(Tkm1); cutoff=cutoff), W, valid, 2)
        for k in 3:Ncheb
            Tk = ITensorMPS.truncate!(+(2 * apply(Ham_sc, Tkm1; cutoff=cutoff),
                                         -Tkm2; maxdim=maxdim); cutoff=cutoff)
            _accum!(acc, ITensorMPS.truncate!(extract_diagonal_to_mps(Tk); cutoff=cutoff), W, valid, k)
            Tkm2 = Tkm1;  Tkm1 = Tk
            verbose && (k % 10 == 0 || k == Ncheb) &&
                println("  QPI ", label, " $k/$Ncheb  maxlinkdim=$(maxlinkdim(Tkm1))")
        end
        for iω in 1:Nω_loc
            valid[iω] && acc[iω] !== nothing || continue
            acc[iω] = ITensorMPS.truncate!(
                acc[iω] / (π^2 * Ncheb * sqrt(1 - ω_vals[iω]^2)); cutoff=cutoff)
        end
    end

    verbose && println("QPI: online KPM clean…")
    _run_online_ldos!(ldos_clean, Ham_clean,  W_c, valid_c, ω_clean, "clean")
    verbose && println("QPI: online KPM impurity…")
    _run_online_ldos!(ldos_imp,   Ham_imp_sc, W_i, valid_i, ω_imp,   "imp")

    # ── Position sites and QFT MPO ────────────────────────────────────────────
    L_pos  = H.L
    pos_s  = H.sites

    # sign=+1 → U_{kr} = e^{+2πikr/N}/√N  (physics forward QFT).
    # TCI.reverse places LSB at site 1, matching the quantics QFT convention.
    FT_mpo = fix_sites(
        MPO(QuanticsTCI.quanticsfouriermpo(L_pos; sign=+1.0, normalize=true)),
        pos_s)

    # ── Spatial window MPO (optional, built once, reused for all ω) ──────────
    # Smooth disk (2D) or interval (1D) mask learned via QTCI; same SDF machinery
    # as mask_hamiltonian.  Applied to δA_mps before the QFT to focus on bulk
    # signal and avoid Gibbs-like ringing from open boundaries.
    window_mpo = nothing
    if !isnothing(window_fraction)
        if isnothing(H.geometry)
            @warn "get_qpi: window_fraction requires H.geometry — windowing skipped."
        else
            verbose && println("QPI: building spatial window (fraction=$window_fraction, σ=$window_sigma)…")
            _sig_w(x) = 1 / (1 + exp(-clamp(x, -500.0, 500.0)))

            # Bounding box via min/max of all site coordinates
            p0      = collect(Float64, H.geometry(1))
            D_geom  = length(p0)
            lo      = copy(p0);  hi = copy(p0)
            for i in 2:H.N
                p = collect(Float64, H.geometry(i))
                lo .= min.(lo, p);  hi .= max.(hi, p)
            end
            ctr      = (lo .+ hi) ./ 2
            hw       = (hi .- lo) ./ 2   # per-dimension half-widths
            radius_w = window_fraction * minimum(hw)

            if D_geom == 2
                sdf_w = sdf_disk(ctr[1], ctr[2], radius_w)
                mfn   = i -> _sig_w(sdf_w(H.geometry(i)...) / window_sigma)
            else
                sdf_w = sdf_interval(ctr[1] - radius_w, ctr[1] + radius_w)
                mfn   = i -> _sig_w(sdf_w(H.geometry(i)[1]) / window_sigma)
            end

            xvals_w     = range(0, H.N - 1; length=H.N)
            qtt_w, _, _ = quanticscrossinterpolate(Float64,
                               x -> mfn(round(Int, x) + 1), xvals_w; tolerance=1e-6)
            w_mps       = MPS(TCI.tensortrain(qtt_w.tci); sites=pos_s)

            # Promote MPS → diagonal MPO (same as mask_hamiltonian)
            w_mpo = outer(w_mps', w_mps)
            for i in 1:L_pos
                w_mpo[i] = Quantics._asdiagonal(w_mps[i], pos_s[i])
            end
            window_mpo = w_mpo
        end
    end

    # ── Per-energy QPI accumulation ───────────────────────────────────────────
    Nω  = length(ω_phys_vals)
    qpi = zeros(Float64, Nω, H.N)

    # Bond-dim-1 product state with every amplitude = 1; inner(ones_mps, v) = Σ_r v(r).
    ones_mps = MPS([ITensor([1.0, 1.0], pos_s[i]) for i in 1:L_pos])

    for iω in 1:Nω
        (ldos_clean[iω] === nothing || ldos_imp[iω] === nothing) && continue

        # Difference LDOS: δA(r,ω) = A_imp(r,ω) − A_clean(r,ω)
        δA_mps = ITensorMPS.truncate!(
            +(ldos_imp[iω], -1.0 * ldos_clean[iω]; maxdim=maxdim, cutoff=cutoff);
            cutoff=cutoff)

        # Remove spatial mean (= tr[δA]/N) so that the q=0 component is zero.
        s      = real(inner(ones_mps, δA_mps))
        δA_mps = ITensorMPS.truncate!(
            +(δA_mps, (-s / H.N) * ones_mps; maxdim=maxdim, cutoff=cutoff);
            cutoff=cutoff)

        # Apply spatial window before QFT to suppress edge scattering
        if !isnothing(window_mpo)
            δA_mps = apply(window_mpo, δA_mps; cutoff=cutoff, maxdim=maxdim)
        end

        # Apply QFT unitary: δÃ_mps = U|δA⟩
        δÃ_mps = apply(FT_mpo, δA_mps; cutoff=cutoff, maxdim=maxdim)

        # QPI(k,ω) = |δÃ(k,ω)|²  evaluated at every momentum index.
        # Read out with LSB-first encoding (site 1 = bit 0), consistent with the
        # QFT MPO convention after TCI.reverse — same as _eval_diag_mps in QFT_tk.jl.
        for k in 0:H.N-1
            acc = ITensor(1.0)
            for i in 1:L_pos
                b   = (k >> (i - 1)) & 1   # LSB-first: site 1 = bit 0
                acc *= δÃ_mps[i] * setelt(pos_s[i] => b + 1)
            end
            qpi[iω, k+1] = abs2(scalar(acc))
        end

        verbose && iω % 10 == 0 &&
            println("QPI: processed $iω / $Nω energies")
    end

    return qpi
end
