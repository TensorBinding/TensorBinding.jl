# MPOTools.jl — MPO composition and embedding helpers shared across TensorBinding.
#
# Contents: the MPO Kronecker product (mpo_kron), the interleaving site plumbing
# (interleave_mpo, interleave_mpo_tb, swap_every_other_legs, collapse_mpo_pairs)
# used by the RPA bubbles, the Krylov Green's function, the exciton QFT and the
# two-particle Hamiltonian, MPO powers by squaring (compose_power), the exact
# rank-1 site projector (_site_projector_mpo) and the left-to-right compressed
# sum of MPO terms (sum_mpos) that the lattice builders assemble with, with its
# checked variant _checked_sum_mpos (the sublattice builders).
#
# Main entry points: mpo_kron, interleave_mpo, interleave_mpo_tb, compose_power,
#   sum_mpos.
#
# Depends on: Utils (_bra_ket, _mpo_site_pair, the sigma_d/sigma_u ops).
#
# Moved verbatim in Tier 1 of docs/dev/REORGANISATION_TODO.md from the interim
# physics/rpa/plumbing.jl (split out of the former physics/RPA_tk.jl),
# core/Hamiltonian.jl and lattice/TJunction.jl.

# ============================================================
# 1. Tensor product utilities (MPO Kronecker product)
# ============================================================

"""
    mpo_kron(A, B) -> MPO

Concatenate two MPOs into a single MPO on the combined site space,
joined by a bond-dimension-1 link.
"""
function mpo_kron(A::MPO, B::MPO)
    LA = length(A)
    LB = length(B)
    M  = MPO([ITensor() for _ in 1:(LA+LB)], 1, LA+LB)
    for j in 1:LA;  M[j]    = A[j];  end
    for j in 1:LB;  M[LA+j] = B[j];  end
    link     = Index(1, "Link_AB")
    M[LA]   *= delta(link)
    M[LA+1] *= delta(link)
    return M
end

# ============================================================
# 2. Site-index manipulation helpers
# ============================================================

"""
    interleave_mpo_tb(op, sites_A, sites_B, which) -> MPO

Generalization of `interleave_mpo` for heterogeneous site spaces (Layer,
Qubit, Honeycomb, etc.).  Embeds an `N`-site MPO `op` into the `2N`-site
product space whose physical indices are ordered as

    [sites_A[1], sites_B[1], sites_A[2], sites_B[2], …, sites_A[N], sites_B[N]]

so every consecutive pair `(sites_A[i], sites_B[i])` shares the same
dimension regardless of site type.

- `which = :A` : `op` acts on `sites_A` (odd positions), identity on `sites_B`
- `which = :B` : `op` acts on `sites_B` (even positions), identity on `sites_A`
"""
function interleave_mpo_tb(op::MPO,
                            sites_A::Vector{<:Index},
                            sites_B::Vector{<:Index},
                            which::Symbol)
    N = length(op)
    @assert length(sites_A) == length(sites_B) == N "sites_A, sites_B and op must all have length N"
    sites_combined = reduce(vcat, [[sa, sb] for (sa, sb) in zip(sites_A, sites_B)])
    n = (which == :A) ? 1 : 0
    return interleave_mpo(op, sites_combined, n)
end


"""
    swap_every_other_legs(MPOin, newsites) -> MPO

Replace site indices and additionally swap bra↔ket on every odd-numbered
site (the first register of an interleaved pair).  Used to convert the 2L-site
bubble MPO from the interleaved ordering into the form expected by
`collapse_mpo_pairs`. The legs are read by prime level (`_mpo_site_pair`), not in
storage order.
"""
function swap_every_other_legs(MPOin::MPO, newsites)
    L2      = length(MPOin)
    @assert length(newsites) == L2
    T = MPO(L2)
    for n in 1:L2
        ket, bra = _mpo_site_pair(MPOin, n)
        new_s    = newsites[n]
        if iseven(n)
            T[n] = MPOin[n] * delta(bra, prime(new_s)) * delta(ket, new_s)
        else
            T[n] = MPOin[n] * delta(bra, new_s)        * delta(ket, prime(new_s))
        end
    end
    return T
end


"""
    collapse_mpo_pairs(mpo2L, out_sites) -> MPO

Merge each consecutive pair of sites `(2n-1, 2n)` in a `2L`-site MPO
into a single site of an `L`-site MPO by contracting and tying the
shared bra and ket indices to `out_sites[n]`.
"""
function collapse_mpo_pairs(mpo2L::MPO, out_sites)
    L2 = length(mpo2L)
    @assert iseven(L2) "Input MPO must have even length (2L)."
    L = L2 ÷ 2
    @assert length(out_sites) == L
    sinds = siteinds(mpo2L)
    T = MPO(L)
    for n in 1:L
        bra1, ket1 = _bra_ket(sinds[2n-1])
        bra2, ket2 = _bra_ket(sinds[2n])
        snew = out_sites[n]
        W    = mpo2L[2n-1] * mpo2L[2n]
        W   *= delta(bra1, bra2, prime(snew))
        W   *= delta(ket1, ket2, snew)
        T[n] = W
    end
    return T
end

# ============================================================
# 3. Interleaving (embed an L-site MPO into a 2L-site space)
# ============================================================

"""
    interleave_mpo(target_mpo, phys_sites, n) -> MPO

Embed an `L`-site MPO into a `2L`-site space by interleaving it with
identity operators.  `phys_sites` must have length `2L`.

- `n = 0` : operator sits at even positions (2, 4, 6, …), identities at odd
- `n = 1` : operator sits at odd positions (1, 3, 5, …), identities at even

The embedded operator is `target_mpo` itself, not its transpose: the ket leg of
tensor `i` becomes `phys_sites[k]` and the bra leg `phys_sites[k]'`, whichever
order the tensor stores them in (`_mpo_site_pair`; for the unrelated legs of a
converted QTCI tensor train the first leg is the bra).

**Note**: `phys_sites` must be interleaved as `[A[1], B[1], A[2], B[2], …]`
so that each operator site lands on an index with the correct dimension.
For heterogeneous site spaces (layer, sublattice, …), use `interleave_mpo_tb`
which builds the interleaved site list automatically.
"""
function interleave_mpo(target_mpo, phys_sites, n)
    N_old = length(target_mpo)
    N_new = 2 * N_old
    @assert length(phys_sites) == N_new

    new_mpo  = MPO(phys_sites)
    link_map = Dict{Index, Vector{Index}}()
    for k in 1:N_old-1
        ol          = linkind(target_mpo, k)
        d           = dim(ol)
        link_map[ol] = [Index(d, "Link,l=$(2k-1)"), Index(d, "Link,l=$(2k)")]
    end

    for i in 1:N_old
        idx_orig  = (n == 1) ? 2i-1 : 2i
        idx_ident = (n == 1) ? 2i   : 2i-1

        # Ket → phys, bra → phys' by prime level (mapping siteinds(target_mpo, i) in
        # storage order embedded the transpose of every tensor stored (s', s)).
        ket, bra = _mpo_site_pair(target_mpo, i)
        W = replaceinds(target_mpo[i], (ket, bra) =>
                                       (phys_sites[idx_orig], phys_sites[idx_orig]'))
        if i > 1
            ol_left = linkind(target_mpo, i-1)
            W = replaceind(W, ol_left => link_map[ol_left][2])
        end
        if i < N_old
            ol_right = linkind(target_mpo, i)
            W = replaceind(W, ol_right => link_map[ol_right][1])
        end
        new_mpo[idx_orig] = W

        if idx_ident == 1 || idx_ident == N_new
            new_mpo[idx_ident] = delta(phys_sites[idx_ident], phys_sites[idx_ident]')
        else
            ol     = linkind(target_mpo, idx_ident ÷ 2)
            l_left  = link_map[ol][1]
            l_right = link_map[ol][2]
            new_mpo[idx_ident] = delta(l_left, l_right) *
                                  delta(phys_sites[idx_ident], phys_sites[idx_ident]')
        end
    end
    return new_mpo
end


# ============================================================
# 4. Exponentiation-by-squaring for MPO composition
# ============================================================

"""
    compose_power(base, nn; side=:right, apply_kwargs=NamedTuple()) -> MPO

Compose `base` with itself `nn` times using **exponentiation-by-squaring** (O(log n) applies).
Replaces the old `arbitarty_offline` helper which used O(n) sequential applies.

- `side = :right`: `acc = apply(acc, base)` at each set bit
- `side = :left`: `acc = apply(base, acc)` at each set bit

`apply_kwargs` (e.g. `(; cutoff=1e-8, maxdim=200)`) are forwarded to every `apply` call.
`nn = 0` returns the identity MPO; `nn = 1` returns `base` unchanged.
"""
function compose_power(base::MPO, nn::Integer;
                       side::Symbol    = :right,
                       apply_kwargs    = NamedTuple())
    @assert nn >= 0 "nn must be non-negative"
    nn == 0 && return MPO(siteinds(base), "Id")
    nn == 1 && return base
    acc = nothing
    cur = base
    k   = nn
    while k > 0
        if (k & 1) == 1
            acc = acc === nothing ? cur :
                  side === :right ? apply(acc, cur; apply_kwargs...) :
                                    apply(cur, acc; apply_kwargs...)
        end
        k >>>= 1
        k > 0 && (cur = apply(cur, cur; apply_kwargs...))
    end
    return acc::MPO
end


# ============================================================
# 5. Exact site-projector MPO
# ============================================================

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


# ============================================================
# 6. Sequential compressed sum of MPO terms
# ============================================================

"""
    sum_mpos(terms; kwargs...) -> MPO

Sum the MPOs in `terms` from left to right, `((t₁ + t₂) + t₃) + …`, compressing each
partial sum as `+(a, b; kwargs...)` does (`cutoff`, `maxdim`, …). A single term is
returned as it is.

The order is part of the result: every partial sum is truncated, so summing the same
terms in another order (or with the multi-argument `+(t₁, t₂, t₃; …)`) moves the
output at the cutoff level. `terms` may be a lazy generator; each term is then built
just before it is added.
"""
function sum_mpos(terms; kwargs...)
    next = iterate(terms)
    next === nothing && throw(ArgumentError("sum_mpos needs at least one term"))
    total, state = next
    while (next = iterate(terms, state)) !== nothing
        term, state = next
        total = +(total, term; kwargs...)
    end
    return total
end

"""
    _checked_sum_mpos(terms; cutoff, maxdim, rtol=1e-11) -> MPO

`sum_mpos(terms; cutoff)` followed by `truncate!(·; maxdim, cutoff)`, the assembly of the
sublattice builders, checked against the exact direct sum `H` of `terms`: the result is
returned as it is unless it lies further than `rtol ‖H‖` (Frobenius) from `H` while the
SVD truncation of `H` at the same `cutoff` and `maxdim` stays within `rtol ‖H‖`; then that
SVD truncation is returned instead. Where the truncation does remove weight, or the sum
is right, the output is that of `sum_mpos` bit for bit.

`+` compresses every partial sum with ITensors' density-matrix algorithm, which projects
on the eigenvectors of a Hermitian eigensolver; for (nearly) degenerate eigenvalues LAPACK
can return them non-orthonormal, and the sum then carries spurious entries (1e-5 in
`honeycomb_sublattice_hamiltonian(2, 1)`, whatever the cutoff). `terms` is iterated twice.
"""
function _checked_sum_mpos(terms; cutoff::Real, maxdim::Integer, rtol::Real = 1e-11)
    total = sum_mpos(terms; cutoff=cutoff)
    ITensorMPS.truncate!(total; maxdim=maxdim, cutoff=cutoff)
    exact = foldl((a, b) -> +(a, b; alg="directsum"), terms)
    ref   = ITensorMPS.truncate(exact; maxdim=maxdim, cutoff=cutoff)
    tol   = rtol * norm(exact)
    (_mpo_distance(ref, exact) > tol || _mpo_distance(total, exact) <= tol) && return total
    return ref
end

# Frobenius distance ‖A - B‖ of two MPOs on the same sites, read off the orthogonality
# centre of their direct-sum difference (accurate to rounding, unlike a difference of
# inner products).
function _mpo_distance(A::MPO, B::MPO)
    Δ = +(A, -1 * B; alg="directsum")
    orthogonalize!(Δ, 1)
    return norm(Δ[1])
end
