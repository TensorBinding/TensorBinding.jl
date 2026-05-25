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


# ---------------------------------------------------------------------
# Shift MPO:  (Q f)(x) = f(x + q)  on a binary-encoded chain
# ---------------------------------------------------------------------

function build_shift_mpo(sites, q::Integer,cyclic=true)
    N      = length(sites)
    q_bits = [(q >> (N - i)) & 1 for i in 1:N]
    links  = [Index(2, "Link,l$n") for n in 0:N+1]
    mpo    = MPO(sites)

    for n in N:-1:1
        s     = sites[n]
        l_in  = links[n+1]
        l_out = links[n]
        T     = ITensor(s', s, l_in, l_out)
        qn    = q_bits[n]

        for cin in 0:1, s_val in 0:1
            total   = s_val + qn + cin
            res_val = total % 2
            cout    = total ÷ 2
            T[s' => (res_val + 1), s => (s_val + 1),
              l_in => (cin == 1 ? 1 : 2), l_out => (cout == 1 ? 1 : 2)] = 1.0
        end
        mpo[n] = T
    end

    mpo[N] *= onehot(links[N+1] => 2)
    if cyclic
        mpo[1] *= (onehot(links[1] => 1) + onehot(links[1] => 2)) # cyclic
    else
        mpo[1] *= onehot(links[1] => 2)
    end

    return mpo
end


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
    fused_mpo(mps, new_sites) -> MPO

Convert an `N`-site MPS with dim-4 physical indices produced by QTCI on a
`:fused` 2D quantics grid into an `N`-site MPO.  Each dim-4 physical index
encodes one (bra-bit, ket-bit) pair; a combiner splits it into
`new_sites[i]'` (bra) and `new_sites[i]` (ket).
"""
function fused_mpo(mps, new_sites)
    N = length(mps)
    @assert N == length(new_sites) "MPS has $N sites but new_sites has $(length(new_sites)) sites."
    new_mpo = MPO(N)
    for i in 1:N
        T     = mps[i]
        old_s = siteind(mps, i)           # dim-4 fused index
        comb  = combiner(new_sites[i]', new_sites[i])
        c_idx = combinedind(comb)
        new_mpo[i] = replaceind(T, old_s, c_idx) * dag(comb)
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



# Exciton basis state |x, x⟩ on the interleaved electron-hole chain.
# x is 1-indexed (x ∈ {1, …, 2^LPhys}), consistent with get_diagonal_mpo
# and add_onsite! conventions in TensorBinding.
function mpsexciton(x, sites)
    L     = length(sites)
    LPhys = div(L, 2)
    bits  = to_binary_vector(Int(x) - 1, LPhys)   # shift to 0-indexed for binary encoding

    elechole = Vector{String}(undef, L)
    for i in 1:LPhys
        elechole[2i - 1] = bits[i]
        elechole[2i]     = bits[i]
    end

    return MPS(sites, elechole)
end



# ---------------------------------------------------------------------
# MPS → diagonal MPO conversion
# ---------------------------------------------------------------------

"""
    mps_to_diagonal_mpo(mps, sites) -> MPO

Convert an MPS into a diagonal MPO on `sites` by replacing each physical
index with a bra–ket pair tied by a 3-leg delta.  Used to convert the
output of a 2D QTCI (encoded as a flat MPS) into a diagonal MPO on the
interleaved (e.g. electron-hole) site space.
"""
function mps_to_diagonal_mpo(mps, sites)
    N          = length(mps)
    mpo_tensors = Vector{ITensor}(undef, N)
    for i in 1:N
        mps_t = mps[i]
        old_s = if i == 1
            uniqueind(mps_t, mps[i+1])
        elseif i == N
            uniqueind(mps_t, mps[i-1])
        else
            uniqueind(mps_t, mps[i-1], mps[i+1])
        end
        s              = sites[i]
        s_temp         = Index(dim(s), "temp")
        mpo_tensors[i] = replaceind(mps_t, old_s => s_temp) * delta(s_temp, s, s')
    end
    return MPO(mpo_tensors)
end

# ============================================================
# Auxiliary site prepend — unified prepend_op
# ============================================================

"""
    prepend_op(H_mpo, s, mat)       -> MPO   explicit matrix
    prepend_op(H_mpo, s, op::Symbol)-> MPO   named op (Spin / Nambu index)
    prepend_op(H_mpo, s, k, l)      -> MPO   sparse |k⟩⟨l|  (Layer / any index)
    prepend_op(H_mpo, s, k)         -> MPO   projector |k⟩⟨k|

Prepend a single-site operator on index `s` to `H_mpo`, extending it from
L sites to L+1 sites.  The returned MPO has site indices `[s; original…]`.

**Dispatch rules**
- Matrix form: `mat[i,j]` = ⟨i|op|j⟩ (1-indexed).  Element type is preserved.
- Symbol form: named operator looked up by the type of `s` (tag `"Spin"` or
  `"Nambu"`).  Defined in Supercond_tk.jl after the op dictionaries.
- Integer pair `(k, l)`: places a single 1 at row `k`, col `l` in a
  `dim(s) × dim(s)` zero matrix.  Covers layer hops and projectors for any
  dimension Layer index.
- Single integer `k`: shorthand for the projector `|k⟩⟨k|`.
"""
function prepend_op(H_mpo::MPO, s::Index, mat::AbstractMatrix{T}) where T <: Number
    Lh    = length(H_mpo)
    bond0 = Index(1, "Link,l=0")
    Op    = ITensor(T, s', s, bond0)
    for j in axes(mat, 2), i in axes(mat, 1)
        iszero(mat[i, j]) || (Op[s' => i, s => j, bond0 => 1] = mat[i, j])
    end
    delta0 = ITensor(bond0);  delta0[bond0 => 1] = 1.0
    H1_ext = H_mpo[1] * delta0
    ext    = MPO(Lh + 1)
    ext[1] = Op
    ext[2] = H1_ext
    for k in 3:Lh+1
        ext[k] = H_mpo[k-1]
    end
    return ext
end
prepend_op(H::MPO, s::Index, mat::AbstractMatrix) =
    prepend_op(H, s, ComplexF64.(mat))

function prepend_op(H_mpo::MPO, s::Index, k::Int, l::Int)
    mat = zeros(Float64, dim(s), dim(s))
    mat[k, l] = 1.0
    return prepend_op(H_mpo, s, mat)
end
prepend_op(H_mpo::MPO, s::Index, k::Int) = prepend_op(H_mpo, s, k, k)


"""
    postpend_op(H_mpo, s, mat)        -> MPO   explicit matrix
    postpend_op(H_mpo, s, op::Symbol) -> MPO   named op (Spin / Nambu index)
    postpend_op(H_mpo, s, k, l)       -> MPO   sparse |k⟩⟨l|  (Layer / any index)
    postpend_op(H_mpo, s, k)          -> MPO   projector |k⟩⟨k|

Append a single-site operator on index `s` to the *end* of `H_mpo`, extending
it from L sites to L+1 sites.  The returned MPO has site indices `[original…; s]`.

Symmetric counterpart of `prepend_op`; dispatch rules are identical.
"""
function postpend_op(H_mpo::MPO, s::Index, mat::AbstractMatrix{T}) where T <: Number
    Lh       = length(H_mpo)
    bond_end = Index(1, "Link,l=$Lh")
    Op       = ITensor(T, s', s, bond_end)
    for j in axes(mat, 2), i in axes(mat, 1)
        iszero(mat[i, j]) || (Op[s' => i, s => j, bond_end => 1] = mat[i, j])
    end
    delta_end = ITensor(bond_end);  delta_end[bond_end => 1] = 1.0
    HLast_ext = H_mpo[Lh] * delta_end
    ext       = MPO(Lh + 1)
    for k in 1:Lh-1;  ext[k] = H_mpo[k];  end
    ext[Lh]   = HLast_ext
    ext[Lh+1] = Op
    return ext
end
postpend_op(H::MPO, s::Index, mat::AbstractMatrix) =
    postpend_op(H, s, ComplexF64.(mat))

function postpend_op(H_mpo::MPO, s::Index, k::Int, l::Int)
    mat = zeros(Float64, dim(s), dim(s))
    mat[k, l] = 1.0
    return postpend_op(H_mpo, s, mat)
end
postpend_op(H_mpo::MPO, s::Index, k::Int) = postpend_op(H_mpo, s, k, k)


# ============================================================
# Debug / validation utilities
# ============================================================

# Build a product-state MPS with an explicit 1-indexed value per site.
# Works for any site types (Qubit, Layer, Spin, Nambu …).
function _product_state_mps(sites::Vector{<:Index}, vals::Vector{Int})
    n = length(sites)
    links   = [Index(1, "Link,l=$i") for i in 1:n-1]
    tensors = Vector{ITensor}(undef, n)
    if n == 1
        t = ITensor(sites[1]);  t[sites[1] => vals[1]] = 1.0
        tensors[1] = t
    else
        t = ITensor(sites[1], links[1])
        t[sites[1] => vals[1], links[1] => 1] = 1.0
        tensors[1] = t
        for i in 2:n-1
            t = ITensor(links[i-1], sites[i], links[i])
            t[links[i-1] => 1, sites[i] => vals[i], links[i] => 1] = 1.0
            tensors[i] = t
        end
        t = ITensor(links[n-1], sites[n])
        t[links[n-1] => 1, sites[n] => vals[n]] = 1.0
        tensors[n] = t
    end
    return MPS(tensors)
end


# Build a product-state MPS for basis state k (0-indexed, big-endian across
# sites) without using string state names.  Works for any site types.
function _basis_state_mps(k::Int, sites::Vector{<:Index})
    n    = length(sites)
    dims = dim.(sites)
    vals = Vector{Int}(undef, n)
    rem  = k
    for i in n:-1:1          # peel off LSB first (big-endian storage)
        vals[i] = rem % dims[i] + 1   # 1-based ITensors convention
        rem      = rem ÷ dims[i]
    end
    links   = [Index(1, "Link,l=$i") for i in 1:n-1]
    tensors = Vector{ITensor}(undef, n)
    if n == 1
        t = ITensor(sites[1]);  t[sites[1] => vals[1]] = 1.0
        tensors[1] = t
    else
        t = ITensor(sites[1], links[1])
        t[sites[1] => vals[1], links[1] => 1] = 1.0
        tensors[1] = t
        for i in 2:n-1
            t = ITensor(links[i-1], sites[i], links[i])
            t[links[i-1] => 1, sites[i] => vals[i], links[i] => 1] = 1.0
            tensors[i] = t
        end
        t = ITensor(links[n-1], sites[n])
        t[links[n-1] => 1, sites[n] => vals[n]] = 1.0
        tensors[n] = t
    end
    return MPS(tensors)
end


"""
    matrix_checker(mpo, sites, i, j) -> Number
    matrix_checker(mpo, L, sites, i, j) -> Number   (L ignored)

Return the matrix element ⟨i|mpo|j⟩.  Works for any site types
(Qubit, Layer, Spin, Nambu …).  Intended for small-system validation.
"""
function matrix_checker(mpo, sites, i, j)
    psii = _basis_state_mps(Int(i), sites)
    psij = _basis_state_mps(Int(j), sites)
    return inner(psii, apply(mpo, psij))
end
matrix_checker(mpo, ::Int, sites, i, j) = matrix_checker(mpo, sites, i, j)


"""
    get_matrix(mpo, sites) -> Matrix{ComplexF64}
    get_matrix(mpo, L, sites) -> Matrix{ComplexF64}   (L ignored)

Return the full `D × D` dense matrix of `mpo`, where `D = ∏ dim(sᵢ)`.
Works for any site types (Qubit, Layer, Spin, Nambu …).
Feasible only for small systems (D ≲ 512).
"""
function get_matrix(mpo, sites)
    sz  = prod(dim(s) for s in sites)
    mat = Matrix{ComplexF64}(undef, sz, sz)
    for i in 0:sz-1, j in 0:sz-1
        mat[i+1, j+1] = matrix_checker(mpo, sites, i, j)
    end
    return mat
end
get_matrix(mpo, ::Int, sites) = get_matrix(mpo, sites)

