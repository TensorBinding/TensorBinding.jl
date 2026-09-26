# MetallicMean.jl — metallic-mean quasicrystal position spaces
#
# The metallic-mean word with parameter m is the fixed point of the substitution
# A -> A^m B, B -> A (m = 1 Fibonacci, m = 2 silver mean / Pell, m = 3 bronze mean).
# Sites are labelled by their expansion in the numeration system with basis
#     q_0 = 1,  q_1 = m + 1,  q_(l+1) = m q_l + q_(l-1),
# digits in 0:m and the admissibility rule "a digit m must be followed by 0".
# L digits enumerate exactly q_L physical sites inside the ambient (m+1)^L
# register of Qudit sites. The letter at site n is B iff the least significant
# digit of n is m, so both the word and the validity indicator are
# bond-dimension-2 automaton MPS and the Hamiltonian MPO is exact at any L.
# For m = 1 this reproduces the Zeckendorf construction of Fibonacci.jl on
# dimension-2 Qudit sites.

"""
    MetallicMeanPositionSpace(m, projector)

Projected position space of the metallic-mean chain with parameter `m`.
`projector` is the identity on the `q_L` admissible digit strings embedded in
the ambient `(m+1)^L` Qudit register.
"""
struct MetallicMeanPositionSpace <: AbstractPositionSpace
    m::Int
    projector::MPO
end

ambient_dimension(space::MetallicMeanPositionSpace, H::TBHamiltonian) =
    big(space.m + 1)^H.L

"""
    metallic_mean_number(m, n) -> BigInt

Return `q_n` for the metallic mean with parameter `m`, where `q_0 = 1`,
`q_1 = m + 1`, and `q_(n+1) = m q_n + q_(n-1)`. For `m = 1` this is `F_(n+2)`;
for `m = 2` it is the sequence `1, 3, 7, 17, 41, …` of the silver-mean chain.
"""
function metallic_mean_number(m::Integer, n::Integer)
    m >= 1 || throw(ArgumentError("metallic-mean parameter m must be at least 1"))
    n >= 0 || throw(ArgumentError("n must be non-negative"))
    a, b = big(1), big(m + 1)
    for _ in 1:n
        a, b = b, m * b + a
    end
    return a
end

"""Number `q_L` of admissible length-`L` digit strings for the metallic mean `m`."""
metallic_mean_site_count(m::Integer, L::Integer) = Int(metallic_mean_number(m, L))

"""
    metallic_mean_digits(m, n, L) -> Vector{Int}

Length-`L`, most-significant-first expansion of the physical site label
`n in 0:q_L-1` in the metallic-mean numeration system: digits in `0:m` with
weights `q_(L-1), …, q_0`, obtained greedily. The result never contains a
digit `m` followed by a nonzero digit.
"""
function metallic_mean_digits(m::Integer, n::Integer, L::Integer)
    L >= 1 || throw(ArgumentError("L must be positive"))
    N = metallic_mean_site_count(m, L)
    0 <= n < N || throw(ArgumentError("site label must satisfy 0 <= n < $N"))
    digits = zeros(Int, L)
    remainder = big(n)
    for position in 1:L
        weight = metallic_mean_number(m, L - position)
        digit, remainder = divrem(remainder, weight)
        digits[position] = Int(digit)
    end
    iszero(remainder) || error("metallic-mean conversion failed for m=$m, n=$n, L=$L")
    return digits
end

# Bond-dimension-2 automaton MPS on (m+1)-dimensional sites. Link state 1 means
# "the last digit read was in 0:m-1 (or nothing was read yet)", link state 2
# means "the last digit read was m". The amplitude is zero on inadmissible
# strings, A on admissible strings whose last digit is below m, and B on
# admissible strings ending in m.
function _metallic_mean_automaton_mps(m::Integer, sites; A=0.0, B=1.0)
    L = length(sites)
    L >= 2 || throw(ArgumentError("metallic-mean chains require L >= 2"))
    d = m + 1
    all(s -> dim(s) == d, sites) ||
        throw(ArgumentError("metallic-mean sites must have local dimension m+1 = $d"))
    T = promote_type(Float64, typeof(A), typeof(B))
    links = [Index(2, "MMAutomaton,Link,l=$i") for i in 1:(L - 1)]
    word = MPS(sites)

    first = ITensor(T, sites[1], links[1])
    for σ in 0:(m - 1)
        first[sites[1] => σ + 1, links[1] => 1] = one(T)
    end
    first[sites[1] => d, links[1] => 2] = one(T)
    word[1] = first

    for i in 2:(L - 1)
        bulk = ITensor(T, links[i - 1], sites[i], links[i])
        for σ in 0:(m - 1)
            bulk[links[i - 1] => 1, sites[i] => σ + 1, links[i] => 1] = one(T)
        end
        bulk[links[i - 1] => 1, sites[i] => d, links[i] => 2] = one(T)
        bulk[links[i - 1] => 2, sites[i] => 1, links[i] => 1] = one(T)  # after m only 0
        word[i] = bulk
    end

    last = ITensor(T, links[end], sites[end])
    for σ in 0:(m - 1)
        last[links[end] => 1, sites[end] => σ + 1] = A
    end
    last[links[end] => 1, sites[end] => d] = B
    last[links[end] => 2, sites[end] => 1] = A
    word[end] = last
    return word
end

# Local digit operators on a Qudit of dimension d = m + 1 (1-based matrix
# index = digit + 1). ITensors dispatches Qudit operators on the site dimension.
#   MMLower    = Σ_{k=1}^{m} |k-1><k|   lowers any nonzero digit by one
#   MMP0       = |0><0|                 projector on the digit 0
#   MMRaiseTop = |m><0|                 turns a 0 into the top digit m
function ITensors.op(::OpName"MMLower", ::SiteType"Qudit", d::Int)
    mat = zeros(Float64, d, d)
    for k in 1:(d - 1)
        mat[k, k + 1] = 1.0
    end
    return mat
end
function ITensors.op(::OpName"MMP0", ::SiteType"Qudit", d::Int)
    mat = zeros(Float64, d, d)
    mat[1, 1] = 1.0
    return mat
end
function ITensors.op(::OpName"MMRaiseTop", ::SiteType"Qudit", d::Int)
    mat = zeros(Float64, d, d)
    mat[d, 1] = 1.0
    return mat
end

"""
    metallic_mean_decrement_mpo(m, sites; boundary=:open) -> MPO

Physical decrement `K|n> = |n-1>` in the metallic-mean numeration system. The
`i`-th term fires when the least significant nonzero digit sits at position `i`:
that digit is lowered by one and the (all-zero) tail becomes `m, 0, m, 0, …`,
which encodes `q_k - 1`. With periodic boundaries the only added transition is
`|0> -> |q_L - 1>`; wrapping never occurs at an inadmissible register state.
"""
function metallic_mean_decrement_mpo(m::Integer, sites; boundary::Symbol=:open)
    boundary in (:open, :periodic) ||
        throw(ArgumentError("boundary must be :open or :periodic"))
    L = length(sites)
    shifts = OpSum()
    for i in 1:L
        term = OpSum()
        term += 1.0, "MMLower", i
        for j in (i + 1):L
            term *= (isodd(j - i) ? "MMRaiseTop" : "MMP0", j)
        end
        shifts += term
    end

    if boundary === :periodic
        last_digits = metallic_mean_digits(m, metallic_mean_site_count(m, L) - 1, L)
        op_for(digit) = digit == 0 ? "MMP0" :
                        digit == m ? "MMRaiseTop" :
                        error("unexpected digit $digit in the expansion of q_L - 1")
        wrap = OpSum()
        wrap += 1.0, op_for(last_digits[1]), 1
        for i in 2:L
            wrap *= (op_for(last_digits[i]), i)
        end
        shifts += wrap
    end
    return MPO(shifts, sites)
end

function physical_projector(space::MetallicMeanPositionSpace, H::TBHamiltonian)
    length(H.sites) == H.L ||
        error("MetallicMeanPositionSpace currently supports position-only Hamiltonians")
    return copy(space.projector)
end

function physical_site_state(space::MetallicMeanPositionSpace, H::TBHamiltonian, x::Integer)
    1 <= x <= H.N || throw(BoundsError(1:H.N, x))
    length(H.sites) == H.L ||
        error("MetallicMeanPositionSpace currently supports position-only Hamiltonians")
    return _product_state_mps(H.sites, metallic_mean_digits(space.m, x - 1, H.L) .+ 1)
end

function site_axis(::MetallicMeanPositionSpace, H::TBHamiltonian;
                   ordering::Symbol=:physical, kwargs...)
    ordering === :physical || throw(ArgumentError(
        "ordering=:$ordering is not available for MetallicMeanPositionSpace; " *
        "only :physical is defined (conumbering is currently Fibonacci-only)"))
    return collect(0:(H.N - 1))
end

function site_permutation(::MetallicMeanPositionSpace, H::TBHamiltonian;
                          ordering::Symbol=:physical, kwargs...)
    ordering === :physical || throw(ArgumentError(
        "ordering=:$ordering is not available for MetallicMeanPositionSpace; " *
        "only :physical is defined (conumbering is currently Fibonacci-only)"))
    return collect(1:H.N)
end

"""
    metallic_mean_bond_symbol(m, L, bond) -> Symbol

Return `:A` or `:B` for the 1-indexed bond beginning at `bond` in the canonical
`q_L`-bond periodic approximant of the metallic-mean chain. Bond `N` joins site
`N` to site `1` when periodic boundaries are used.
"""
function metallic_mean_bond_symbol(m::Integer, L::Integer, bond::Integer)
    N = metallic_mean_site_count(m, L)
    1 <= bond <= N || throw(BoundsError(1:N, bond))
    return metallic_mean_digits(m, bond - 1, L)[end] == m ? :B : :A
end

"""
    metallic_mean_hamiltonian(m, L; A, B, model=:hopping, t=1.0, onsite=0.0,
                              boundary=:periodic, scale=nothing, padding=1.05,
                              cutoff=1e-12, maxdim=200) -> TBHamiltonian

Construct the metallic-mean chain with parameter `m` (`A -> A^m B`, `B -> A`)
in its projected numeration position space on `L` Qudit sites of dimension
`m + 1`. The chain has `H.N = q_L` physical sites.

- `model=:onsite`: `A` and `B` are onsite energies and `t` is uniform hopping.
- `model=:hopping`: `A` and `B` are bond amplitudes and `onsite` is uniform.

The Hamiltonian is assembled as `P (V + T K + h.c.) P`, where `P` is the
validity projector, `T`/`V` the diagonal word MPO, and `K` the decrement
[`metallic_mean_decrement_mpo`](@ref). `m = 1` reproduces the Fibonacci chain of
[`fibonacci_hamiltonian`](@ref) on dimension-2 Qudit sites. The default periodic
boundary closes the physical `q_L`-site approximant.
"""
function metallic_mean_hamiltonian(
    m::Integer, L::Integer; A, B,
    model::Symbol=:hopping,
    t::Number=1.0,
    onsite::Number=0.0,
    boundary::Symbol=:periodic,
    scale=nothing,
    padding::Real=1.05,
    cutoff::Real=1e-12,
    maxdim::Integer=200,
)
    m >= 1 || throw(ArgumentError("metallic-mean parameter m must be at least 1"))
    L >= 2 || throw(ArgumentError("metallic-mean chains require L >= 2"))
    model in (:onsite, :hopping) ||
        throw(ArgumentError("model must be :onsite or :hopping"))
    boundary in (:open, :periodic) ||
        throw(ArgumentError("boundary must be :open or :periodic"))
    padding > 1 || throw(ArgumentError("padding must be greater than 1"))
    if model === :onsite
        isreal(A) && isreal(B) ||
            throw(ArgumentError("onsite metallic-mean values A and B must be real"))
    else
        isreal(onsite) ||
            throw(ArgumentError("the uniform onsite energy must be real"))
    end

    sites = siteinds("Qudit", L; dim=m + 1)
    word_mps = _metallic_mean_automaton_mps(m, sites; A, B)
    valid_mps = _metallic_mean_automaton_mps(m, sites; A=1.0, B=1.0)
    word = mps_to_diagonal_mpo(word_mps, sites)
    P = mps_to_diagonal_mpo(valid_mps, sites)
    K = metallic_mean_decrement_mpo(m, sites; boundary)

    V, TK = if model === :onsite
        word, t * K
    else
        onsite * P, apply(word, K; cutoff=cutoff, maxdim=maxdim)
    end
    hopping = +(TK, shift_adjoint_mpo(TK); cutoff=cutoff, maxdim=maxdim)
    Hraw = +(V, hopping; cutoff=cutoff, maxdim=maxdim)
    mpo = apply(P, apply(Hraw, P; cutoff=cutoff, maxdim=maxdim);
                cutoff=cutoff, maxdim=maxdim)
    ITensorMPS.truncate!(mpo; cutoff=cutoff, maxdim=maxdim)

    center, halfwidth = if model === :onsite
        lo, hi = extrema((Float64(real(A)), Float64(real(B))))
        ((lo + hi) / 2, (hi - lo) / 2 + 2abs(t))
    else
        (Float64(real(onsite)), 2max(abs(A), abs(B)))
    end
    scale_value = isnothing(scale) ? padding * Float64(halfwidth) : Float64(scale)
    scale_value > 0 || throw(ArgumentError("KPM scale must be positive"))

    N = metallic_mean_site_count(m, L)
    H = TBHamiltonian(L, N, sites, mpo, _chain_geometry(),
                      scale_value, Float64(center),
                      nothing, nothing, nothing, nothing, 0, nothing)
    H.position_space = MetallicMeanPositionSpace(Int(m), P)
    return H
end

function _build_metallic_mean(params, L::Integer;
                              m=nothing, scale=nothing, tol=1e-12, maxdim=200,
                              kwargs...)
    m === nothing && throw(ArgumentError(
        "get_Hamiltonian(\"metallic_mean\", …) requires the keyword m " *
        "(m=1 Fibonacci, m=2 silver mean, m=3 bronze mean, …)"))
    p = if params isa NamedTuple
        Dict{Symbol,Any}(pairs(params))
    elseif params isa AbstractDict
        Dict{Symbol,Any}(Symbol(k) => v for (k, v) in pairs(params))
    else
        throw(ArgumentError("metallic-mean parameters must be a NamedTuple or dictionary containing A and B"))
    end
    haskey(p, :A) && haskey(p, :B) ||
        throw(ArgumentError("metallic-mean parameters must contain A and B"))
    allowed = Set((:A, :B, :t, :onsite))
    unknown = setdiff(Set(keys(p)), allowed)
    isempty(unknown) || throw(ArgumentError("unknown metallic-mean parameters: $(collect(unknown))"))
    return metallic_mean_hamiltonian(
        m, L; A=p[:A], B=p[:B],
        t=get(p, :t, 1.0), onsite=get(p, :onsite, 0.0),
        scale=scale, cutoff=tol, maxdim=maxdim, kwargs...,
    )
end

# Dense small-system oracle used only by the test suite.
function _dense_metallic_mean_hamiltonian(
    m::Integer, L::Integer; A, B,
    model::Symbol=:hopping,
    t::Number=1.0,
    onsite::Number=0.0,
    boundary::Symbol=:periodic,
)
    model in (:onsite, :hopping) ||
        throw(ArgumentError("model must be :onsite or :hopping"))
    boundary in (:open, :periodic) ||
        throw(ArgumentError("boundary must be :open or :periodic"))
    N = metallic_mean_site_count(m, L)
    word = [metallic_mean_digits(m, n, L)[end] == m ? B : A for n in 0:(N - 1)]
    diagonal = model === :onsite ? word : fill(onsite, N)
    bonds = model === :onsite ? fill(t, N - 1) : word[1:(N - 1)]
    H = zeros(ComplexF64, N, N)
    H[diagind(H)] .= diagonal
    for n in 1:(N - 1)
        H[n, n + 1] = bonds[n]
        H[n + 1, n] = conj(bonds[n])
    end
    if boundary === :periodic
        wrap = model === :onsite ? t : word[end]
        H[N, 1] = wrap
        H[1, N] = conj(wrap)
    end
    return H
end
