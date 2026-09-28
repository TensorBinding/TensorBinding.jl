# conjugation.jl — QFT conjugation of MPOs into momentum space
#
# Contains conjugate_by_qft (plain and TBHamiltonian-aware), the aux-site
# embedding helpers _embed_in_full_sites / _embed_displacement_in_full_sites,
# the two-particle conjugate_by_qft_exciton and the k-space diagonal get_spect_k.
# Split from the former physics/QFT_tk.jl (get_spect_k from the former
# physics/RPA_tk.jl); the overview and file map of physics/qft/ are at the top
# of bands.jl.
#
# Entry points: conjugate_by_qft, conjugate_by_qft_exciton, get_spect_k.
# Depends on: core/Utils.jl, core/MPOTools.jl, core/TBSystem.jl.

# ============================================================
# 1. Single-particle QFT conjugation
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
    sites  = _mpo_ket_sites(W)
    R      = length(sites)
    FTirev = fix_sites(MPO(TCI.reverse(QuanticsTCI.quanticsfouriermpo(R; sign=-1.0, normalize=true))), sites)
    FTrev  = fix_sites(MPO(TCI.reverse(QuanticsTCI.quanticsfouriermpo(R; sign=+1.0, normalize=true))), sites)
    Op1    = apply(W,                        FTirev; cutoff=tol, maxdim=maxdim)
    Op2    = apply(swapprime(FTrev, 0 => 1), Op1;   cutoff=tol, maxdim=maxdim)
    return TCI.truncate(Op2; cutoff=tol, maxdim=maxdim)
end


"""
    conjugate_by_qft(H::TBHamiltonian, W::MPO; tol=1e-9, maxdim=100) -> MPO

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
# 2. Exciton QFT conjugation
# ============================================================

"""
    conjugate_by_qft_exciton(H::TBHamiltonian, W; tol=1e-9, maxdim=100) -> MPO

Two-particle (electron–hole) analogue of [`conjugate_by_qft`](@ref) for the
interleaved `2L`-site exciton encoding produced by `Exciton_Hamiltonian` /
`exciton_hamiltonian`, where one carrier occupies the odd sites (1, 3, …) and
the other the even sites (2, 4, …).

Returns `U · W · U†` where `U = U_e ⊗ U_h` is the product of two **independent**
Quantum Fourier Transforms, one acting on each carrier's `R = H.L` position
qubits.  Conjugating the exciton spectral MPO `T_n` by `U` rotates *both*
carriers into momentum space; extracting the diagonal then resolves the joint
`(kₑ, k_h)` weight (apply a relative-/centre-of-mass-momentum shift first to read
a fixed total momentum `q`).

The site list comes from `_pos_sites(H)` — the same canonical accessor the
single-particle `conjugate_by_qft(H, W)` overload uses — rather than being
reverse-engineered from `siteinds(W)`.

The construction mirrors the single-particle `conjugate_by_qft` but builds the
full two-particle transform first:

1. Two single-particle QFT cores from `QuanticsTCI.quanticsfouriermpo`
   (`TCI.reverse`, `normalize=true`) — `sign=-1` and `sign=+1`, identical to
   `conjugate_by_qft` so the per-carrier convention matches.
2. The same core is interleaved onto both carrier registers — the even sites
   (`interleave_mpo(…, 0)`) and the odd sites (`interleave_mpo(…, 1)`) — and the
   two halves are multiplied into the full transforms `U` and `U†` (`= U_e ⊗ U_h`).
   The product contracts away the interleaved identity tensors, leaving dense cores.
3. `fix_sites` re-canonicalises the combined MPOs onto the physical `sites`, then
   `swapprime` undoes the bra↔ket swap `interleave_mpo` introduces (needed because
   the reversed QFT is not symmetric, so the swap is a genuine transpose).
4. The conjugation `U · W · U†` is applied exactly as in the single-particle
   routine (`swapprime` left-multiply); the result stays on `sites`.

Verified to machine precision against the dense `U·W·U†` (with `U` taken from the
single-particle `conjugate_by_qft`) for `R = 2, 3, 4`.  As with the 2L-site MPOs
involved, raise `maxdim` (and/or tighten `tol`) for larger `R`; the default
`maxdim=100` starts truncating around `R ≳ 4`.

See also [`conjugate_by_qft`](@ref), `Exciton_Hamiltonian`, `interleave_mpo`,
`fix_sites`.
"""
function conjugate_by_qft_exciton(H::TBHamiltonian, W; tol=1e-9, maxdim::Int=100)
    sites = _pos_sites(H)
    L2    = length(sites)
    iseven(L2) || error("conjugate_by_qft_exciton expects an even site count " *
                        "(2L interleaved electron/hole qubits); got $L2.")
    R = L2 ÷ 2

    # Single-particle QFT cores, identical to conjugate_by_qft.
    FT1  = MPO(TCI.reverse(QuanticsTCI.quanticsfouriermpo(R; sign=-1.0, normalize=true)))  # U†-side
    FT1i = MPO(TCI.reverse(QuanticsTCI.quanticsfouriermpo(R; sign=+1.0, normalize=true)))  # U-side

    # Interleave the same core onto both registers (even sites via n=0, odd via
    # n=1), multiply the two halves into the full two-particle transform / inverse,
    # fix_sites onto the physical sites, then `swapprime` to undo the bra↔ket
    # swap that `interleave_mpo` introduces (it lays legs down unprimed-first).
    # Without it the non-symmetric reversed QFT comes out transposed and the
    # conjugation is wrong; with it these match the single-particle convention.
    FTirev = swapprime(fix_sites(apply(interleave_mpo(FT1,  sites, 0),
                                       interleave_mpo(FT1,  sites, 1);
                                       cutoff=tol, maxdim=maxdim), sites), 0 => 1)  # U†  (sign -1)
    FTrev  = swapprime(fix_sites(apply(interleave_mpo(FT1i, sites, 0),
                                       interleave_mpo(FT1i, sites, 1);
                                       cutoff=tol, maxdim=maxdim), sites), 0 => 1)  # U   (sign +1)

    # U · W · U†  — identical structure to single-particle conjugate_by_qft.
    Op1 = apply(W, FTirev; cutoff=tol, maxdim=maxdim)
    Op2 = apply(swapprime(FTrev, 0 => 1), Op1; cutoff=tol, maxdim=maxdim)
    return TCI.truncate(Op2; cutoff=tol, maxdim=maxdim)
end


# ============================================================
# 3. k-space diagonal of an MPO
# ============================================================

"""
    get_spect_k(W; tol=1e-9, maxdim=100) -> Vector{ComplexF64}

Extract the k-space diagonal of MPO `W` as a dense vector of 2^L values.

Conjugates `W` by the QFT (giving W̃ = QFT·W·QFT†), extracts the diagonal
as an MPS, then evaluates each ⟨k|W̃|k⟩ for the 2^L quantics k-indices.
Uses LSB-first quantics convention (site 1 = least significant bit).
"""
function get_spect_k(W::MPO; tol::Real=1e-9, maxdim::Int=100)
    Wk   = conjugate_by_qft(W; tol=tol, maxdim=maxdim)
    diag = extract_diagonal_to_mps(Wk)
    L    = length(diag)
    N    = 2^L
    sd   = siteinds(diag)
    return ComplexF64[inner(MPS(sd, [string((k >> (i-1)) & 1) for i in 1:L]), diag)
                      for k in 0:N-1]
end
