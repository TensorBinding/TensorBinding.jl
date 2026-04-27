# QFT_tk.jl — Quantum Fourier Transform utilities
#
# In the quantics representation the L qubit sites encode real-space
# position in binary.  The QFT maps any real-space MPO W to momentum
# space via U · W · U†.
#
# Band structure workflow
# -----------------------
# The correct way to get A(k, ω) is to conjugate δ(ω − H), NOT H itself.
# Conjugating H directly only gives ε(k) for a perfectly translationally
# invariant system and breaks down with truncation or disorder.
# Instead:
#   1. Build Tn_list from KPM_Tn(H/scale, Ncheb, sites)
#   2. Call get_bands(Tn_list, Ncheb, sites, ω_vals)
#      which for each ω builds δ(ω−H) via get_ldos_w_from_Tn,
#      conjugates it with the QFT, and extracts ⟨k|Ã(ω)|k⟩.
#
# fix_sites lives in utils.jl.

# ============================================================
# QFT conjugation
# ============================================================

"""
    conjugate_by_qft(W; tol=1e-9, maxdim=100) -> MPO

Return `U · W · U†` where `U` is the Quantum Fourier Transform MPO
built from `QuanticsTCI.quanticsfouriermpo`.

This maps any real-space MPO `W` to its momentum-space representation.
Pass `W = δ(ω−H)` (the spectral MPO from `get_ldos_w_from_Tn`) to get
the k-resolved spectral function.  Passing `W = H` directly only
gives the correct ε(k) for perfectly translationally invariant systems.
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
    get_spect_k(W; tol=1e-9, maxdim=100) -> Vector

Compute the momentum-space spectrum of `W` (a single-band 1D system)
by conjugating with the QFT and evaluating the diagonal.

Returns a length-`N` vector of eigenvalues indexed by `k = 0, …, N-1`.
"""
function get_spect_k(W; tol=1e-9, maxdim::Int=100)
    Akop  = conjugate_by_qft(W; tol=tol, maxdim=maxdim)
    sites = getindex.(siteinds(W), 2)
    L     = length(sites)
    N     = 2^L
    # LSB at site 1 — matches the quantics QFT convention (see KPM_LDOS_1D notebook)
    lsb_state(k) = [string((k >> (i-1)) & 1) for i in 1:L]
    mpsk(k) = MPS(sites, lsb_state(Int(k)))
    kvals   = range(0, N - 1; length=N)
    return [inner(mpsk(k)', Akop, mpsk(k)) for k in kvals]
end


"""
    get_spect_k_doubled(W; tol=1e-9, maxdim=100) -> Vector

Like `get_spect_k` but for a doubled system (e.g. two-band or
spin-resolved), where `W` lives on `2L` sites and k-space is
indexed over the first `L` sites tensored with the second `L`.
"""
function get_spect_k_doubled(W; tol=1e-9, maxdim::Int=100)
    Akop  = conjugate_by_qft(W; tol=tol, maxdim=maxdim)
    sites = getindex.(siteinds(W), 2)
    L     = div(length(sites), 2)
    N     = 2^L
    lsb_state(k) = [string((k >> (i-1)) & 1) for i in 1:L]
    mpsk(k)  = MPS(sites[1:L],    lsb_state(Int(k)))
    mpsk1(k) = MPS(sites[L+1:2L], lsb_state(Int(k)))
    mpsk2(k) = mps_kron(mpsk(k), mpsk1(k))
    kvals    = range(0, N - 1; length=N)
    return [inner(mpsk2(k)', Akop, mpsk2(k)) for k in kvals]
end


"""
    get_bands(Tn_list, Ncheb, sites, ω_vals; tol=1e-9, maxdim=100) -> Matrix{Float64}
    get_bands(H::TBHamiltonian, Tn_list, Ncheb, ω_phys_vals; tol=1e-9, maxdim=100) -> Matrix{Float64}

Compute the k-resolved spectral function

    A(k, ω) = ⟨k| U δ(ω−H) U† |k⟩

by conjugating the KPM spectral MPO `δ(ω−H)` with the QFT for each ω.

Returns a `(N_k × N_ω)` matrix where `N_k = 2^L` and `N_ω = length(ω_vals)`.

**Low-level form**: `ω_vals` must be in **rescaled** units `∈ (−1, 1)`.

**High-level form**: pass a `TBHamiltonian` as the first argument and supply
`ω_phys_vals` in **physical** energy units.  The Chebyshev list and order are
taken from the cache set by a prior `KPM_Tn(H, Ncheb; ...)` call.  Spin,
Nambu, and layer auxiliary DOF are handled via `aux_proj` (see high-level
overload docstring).  Errors if no cache exists.

Typical usage (high-level)
--------------------------
```julia
KPM_Tn(H, Ncheb; maxdim=40)          # builds and caches the Tn list
ω_phys = range(-4.0, 4.0; length=120)
Ak_w   = get_bands(H, ω_phys)
```
"""
function get_bands(Tn_list, Ncheb::Int, sites, ω_vals;
                   tol=1e-9, maxdim::Int=100)
    L   = length(sites)
    N   = 2^L
    Nω  = length(ω_vals)

    # Pre-build the QFT and its conjugate once — they depend only on `sites`
    FTirev = fix_sites(MPO(TCI.reverse(
        QuanticsTCI.quanticsfouriermpo(L; sign=-1.0, normalize=true))), sites)
    FTrev  = fix_sites(MPO(TCI.reverse(
        QuanticsTCI.quanticsfouriermpo(L; sign=+1.0, normalize=true))), sites)

    # Pre-build all k-state MPS — LSB at site 1, matching the quantics QFT convention
    lsb_state(k) = [string((k >> (i-1)) & 1) for i in 1:L]
    mpsk = [MPS(sites, lsb_state(k)) for k in 0:N-1]

    Ak_w = zeros(Float64, N, Nω)

    for (iω, ω) in enumerate(ω_vals)
        abs(ω) >= 1.0 && continue   # outside Chebyshev support

        # δ(ω − H) in real space
        δH = get_ldos_w_from_Tn(Tn_list, Ncheb, ω; maxdim=maxdim)

        # Conjugate: Ã(ω) = U δ(ω−H) U†
        Op1  = apply(δH,                        FTirev; cutoff=tol, maxdim=maxdim)
        Akop = apply(swapprime(FTrev, 0 => 1),  Op1;   cutoff=tol, maxdim=maxdim)

        # Diagonal: A(k, ω) = ⟨k|Ã(ω)|k⟩
        for k in 0:N-1
            Ak_w[k+1, iω] = real(inner(mpsk[k+1]', Akop, mpsk[k+1]))
        end
    end

    return Ak_w
end


"""
    get_bands(H::TBHamiltonian, ω_phys_vals;
              aux_proj=nothing, tol=1e-9, maxdim=100) -> Matrix{Float64}

High-level band-structure calculation that works for plain, spinful, BdG,
and layered `TBHamiltonian`s.

The QFT acts only on the `L` position qubits; identity operators are
prepended/appended on any auxiliary (spin, Nambu, layer) sites so that the
full Ã(ω) = U_ext · δ(ω−H) · U_ext† lives on the complete site space.

**`aux_proj`** controls what happens to the auxiliary DOF:
- `nothing` (default) — **trace**: `A(k,ω) = Σ_σ ⟨σ,k|Ã|σ,k⟩`
- `[σ₁, σ₂, …]` — **project**: fix each auxiliary site to the given 1-based
  state index.  The vector must have one entry per auxiliary site in the order
  they appear in `H.sites`.  For a single aux site pass e.g. `[1]` (spin-up /
  particle sector).

Returns a `(N_k × N_ω)` matrix where `N_k = 2^L`.

Requires a prior `KPM_Tn(H, Ncheb; …)` call to populate `H._tn_cache`.
"""
function get_bands(H::TBHamiltonian, ω_phys_vals;
                   aux_proj = nothing,
                   tol=1e-9, maxdim::Int=100)
    H._tn_cache === nothing &&
        error("No Chebyshev cache found.  Call KPM_Tn(H, Ncheb; ...) first.")

    pos_sites = _pos_sites(H)
    Lpos      = length(pos_sites)
    Npos      = 2^Lpos
    pos_set   = Set(pos_sites)

    # Auxiliary sites in H.sites order (spin, Nambu, layer, …)
    aux_sites = filter(s -> s ∉ pos_set, H.sites)

    # ── Build QFT on pos_sites then extend with I on each aux site ────────────
    FTirev = fix_sites(MPO(TCI.reverse(
        QuanticsTCI.quanticsfouriermpo(Lpos; sign=-1.0, normalize=true))), pos_sites)
    FTrev  = fix_sites(MPO(TCI.reverse(
        QuanticsTCI.quanticsfouriermpo(Lpos; sign=+1.0, normalize=true))), pos_sites)

    if H.aux_side === :pre
        for s in reverse(aux_sites)       # innermost first → outermost last
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

    # ── Aux state combinations: trace or project ──────────────────────────────
    aux_combos = if isnothing(aux_proj) || isempty(aux_sites)
        collect(Iterators.product((1:dim(s) for s in aux_sites)...))
    else
        proj = aux_proj isa Integer ? fill(Int(aux_proj), length(aux_sites)) :
                                      collect(Int, aux_proj)
        length(proj) == length(aux_sites) ||
            error("aux_proj has $(length(proj)) entries but H has $(length(aux_sites)) " *
                  "auxiliary site(s).  Pass one state index per aux site.")
        [Tuple(proj)]
    end

    # ── Pre-build (σ_combo, k) product-state MPS ─────────────────────────────
    # k is encoded LSB-first on pos_sites: site i gets bit (i-1) of k.
    pos_kvals(k) = [((k >> (i-1)) & 1) + 1 for i in 1:Lpos]  # 1-indexed

    all_sites_ord = H.aux_side === :pre ? [aux_sites; pos_sites] :
                                          [pos_sites; aux_sites]

    kmps = Dict{Any, Vector{MPS}}()
    for σ_combo in aux_combos
        σ_vals = collect(Int, σ_combo)
        states = Vector{MPS}(undef, Npos)
        for k in 0:Npos-1
            all_vals = H.aux_side === :pre ? [σ_vals; pos_kvals(k)] :
                                             [pos_kvals(k); σ_vals]
            states[k+1] = _product_state_mps(all_sites_ord, all_vals)
        end
        kmps[σ_combo] = states
    end

    # ── Main ω loop ───────────────────────────────────────────────────────────
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
