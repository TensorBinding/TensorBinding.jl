# utils.jl — shared infrastructure used across TensorBinding
#
# Functions here are pure plumbing: binary ↔ MPS conversions,
# site-index manipulation, diagonal MPO construction, and debug
# helpers.  No physics lives here.

# ============================================================
# Operator extensions (defined once to avoid duplicate definitions)
# ============================================================

ITensors.op(::OpName"sigma_plus", ::SiteType"Qubit") =
    [0 1
     0 0]

ITensors.op(::OpName"sigma_minus", ::SiteType"Qubit") =
    [0 0
     1 0]

# ============================================================
# Binary / index utilities
# ============================================================

"""
    to_binary_vector(n, L) -> Vector{String}

Convert non-negative integer `n` to a length-`L` vector of `"0"`/`"1"`
strings (big-endian), suitable as state labels for `MPS(sites, state)`.

# Example
```julia
to_binary_vector(5, 4)   # → ["0", "1", "0", "1"]
```
"""
function to_binary_vector(n::Integer, L::Integer)
    return map(string, collect(lpad(string(n; base=2), L, '0')))
end


"""
    binary_to_MPS(n, L, sites) -> MPS

Return the computational-basis state |n⟩ as an `L`-site MPS, where
`n` is encoded in big-endian binary across the `L` qubit sites.

# Example
```julia
sites = siteinds("Qubit", 4)
psi   = binary_to_MPS(5, 4, sites)   # |0101⟩
```
"""
function binary_to_MPS(n::Integer, L::Integer, sites)
    return MPS(sites, to_binary_vector(n, L))
end

# ============================================================
# MPO / MPS site-index manipulation
# ============================================================

"""
    fix_sites(mpo, sites) -> MPO

Replace the site indices of `mpo` (typically built from a TCI
tensor train whose indices do not match the system's physical sites)
with `sites`.  Modifies `mpo` in-place and returns it.
"""
function fix_sites(mpo, sites)
    oldsites      = getindex.(siteinds(mpo), 2)   # unprimed (ket)
    oldsitesprime = getindex.(siteinds(mpo), 1)   # primed   (bra)
    for i in eachindex(mpo)
        mpo[i] = replaceind(mpo[i], oldsites[i]      => sites[i])
        mpo[i] = replaceind(mpo[i], oldsitesprime[i] => sites[i]')
    end
    return mpo
end


"""
    custom_mpo(mps, new_sites) -> MPO

Convert a `2N`-site MPS produced by QTCI on an interleaved 2D
quantics grid into an `N`-site MPO by contracting each pair of
tensors `(2i-1, 2i)` and mapping old site indices to `new_sites[i]`.

The first index of each pair becomes the bra (primed) site and the
second becomes the ket (unprimed) site, consistent with ITensors
MPO conventions.
"""
function custom_mpo(mps, new_sites)
    N     = length(mps)
    new_N = N ÷ 2
    @assert new_N == length(new_sites) "MPS has $N sites but new_sites has $(length(new_sites)) sites; expected $new_N."
    new_mpo = MPO(new_N)
    for i in 1:new_N
        A          = mps[2i - 1]
        B          = mps[2i]
        combined_T = A * B
        old_s1     = siteind(mps, 2i - 1)   # → bra (primed)
        old_s2     = siteind(mps, 2i)       # → ket (unprimed)
        new_mpo[i] = replaceinds(combined_T,
                                 [old_s1, old_s2] => [new_sites[i]', new_sites[i]])
    end
    return new_mpo
end


"""
    custom_mps(qtt, sites) -> MPS

Replace the site indices of an MPS obtained from a 1D TCI tensor
train with the physical `sites` of the target system.
"""
function custom_mps(qtt, sites)
    old_mps = ITensors.MPS(qtt)
    N       = length(old_mps)
    new_mps = MPS(N)
    for i in 1:N
        old_s   = siteind(old_mps, i)
        new_mps[i] = replaceinds(old_mps[i], [old_s] => [sites[i]])
    end
    return new_mps
end


"""
    mps2mpo(L, sites, density_mps) -> MPO

Convert a diagonal MPS (a function sampled on computational-basis
states) into a diagonal MPO by calling `Quantics._asdiagonal` on
each site tensor.
"""
function mps2mpo(L, sites, density_mps)
    density_mpo = outer(density_mps', density_mps)
    for i in 1:L
        density_mpo.data[i] = Quantics._asdiagonal(density_mps.data[i], sites[i])
    end
    return density_mpo
end


"""
    get_diagonal_mpo(L, sites, f; type=Float64) -> MPO

Build an MPO that is diagonal in the computational basis with entry
`f(x)` at site `x ∈ {1, …, 2^L}`, using QTCI to compress the
function into an MPS and then promoting it to a diagonal MPO.

This is the primary way to represent any on-site potential or
position-dependent modulation in the quantics framework.

# Example
```julia
# Uniform on-site energy gradient
pot = get_diagonal_mpo(L, sites, x -> 0.01 * x)
```
"""
function get_diagonal_mpo(L, sites, f; type=Float64)
    xvals = range(1, 2^L; length=2^L)
    qtt, _, _ = quanticscrossinterpolate(type, f, xvals; tolerance=1e-8)
    tt         = TensorCrossInterpolation.tensortrain(qtt.tci)
    density_mps = MPS(tt; sites)
    return mps2mpo(L, sites, density_mps)
end

# ============================================================
# Debug / validation utilities
# ============================================================

"""
    matrix_checker(mpo, L, sites, i, j) -> Number

Return the matrix element ⟨i|mpo|j⟩ by constructing computational-
basis MPS.  Intended for small-system validation only.
"""
function matrix_checker(mpo, L, sites, i, j)
    psii = binary_to_MPS(Int(i), L, sites)
    psij = binary_to_MPS(Int(j), L, sites)
    return inner(psii, apply(mpo, psij))
end


"""
    get_matrix(mpo, L, sites) -> Matrix

Return the full 2^L × 2^L matrix representation of `mpo` by
evaluating all matrix elements via `matrix_checker`.  Feasible
only for small `L` (≲ 8).
"""
function get_matrix(mpo, L, sites)
    sz  = 2^L
    mat = Matrix{ComplexF64}(undef, sz, sz)
    for i in 0:sz-1, j in 0:sz-1
        mat[i+1, j+1] = matrix_checker(mpo, L, sites, i, j)
    end
    return mat
end
