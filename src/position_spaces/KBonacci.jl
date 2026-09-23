# KBonacci.jl — k-bonacci quasicrystal position spaces
#
# The k-bonacci word on the alphabet a_1, …, a_k (written A, B, C, …) is the
# fixed point of the substitution a_i -> a_1 a_(i+1) for i < k and a_k -> a_1:
# k = 2 is Fibonacci (A -> AB, B -> A), k = 3 Tribonacci (A -> AB, B -> AC,
# C -> A), k = 4 Tetranacci, and so on. Sites are labelled by their binary
# expansion in the k-bonacci numeration system with weights
#     w_l = 2^l for l < k,      w_l = w_(l-1) + w_(l-2) + … + w_(l-k) for l >= k,
# and the admissibility rule "no k consecutive ones" (k = 2: Zeckendorf).
# w_l counts the admissible length-l strings, so L digits enumerate exactly
# w_L physical sites inside the ambient 2^L Qubit register (w_L = F_(L+2) for
# k = 2 and the Tribonacci number T_(L+3) for k = 3). The letter at site n is
# a_(r+1), where r in 0:k-1 is the number of trailing ones of n, so the word and
# the validity indicator are k-state automaton MPS (bond dimension k) and the
# Hamiltonian MPO is exact at any L. For k = 2 the construction coincides with
# Fibonacci.jl on the same Qubit sites; the qubit digit operators FibLower,
# FibRaise and FibP0 defined there are reused here.

"""
    KBonacciPositionSpace(k, projector)

Projected position space of the k-bonacci chain of order `k`. `projector` is
the identity on the `w_L` admissible binary strings (no `k` consecutive ones)
embedded in the ambient `2^L` Qubit register.
"""
struct KBonacciPositionSpace <: AbstractPositionSpace
    k::Int
    projector::MPO
end

ambient_dimension(::KBonacciPositionSpace, H::TBHamiltonian) = big(2)^H.L

"""
    kbonacci_number(k, n) -> BigInt

Return `w_n`, the number of length-`n` binary strings without `k` consecutive
ones: `w_n = 2^n` for `n < k` and `w_n = w_(n-1) + … + w_(n-k)` otherwise. In the
usual seeding of the k-bonacci sequence (`k - 1` zeros followed by a one) this is
its `(n + k)`-th term: `F_(n+2)` for `k = 2`, the Tribonacci number `T_(n+3)` for
`k = 3` (`1, 2, 4, 7, 13, 24, …`), the Tetranacci number for `k = 4`
(`1, 2, 4, 8, 15, 29, …`).
"""
function kbonacci_number(k::Integer, n::Integer)
    k >= 2 || throw(ArgumentError("k-bonacci order k must be at least 2"))
    n >= 0 || throw(ArgumentError("n must be non-negative"))
    n < k && return big(2)^n
    window = [big(2)^l for l in 0:(k - 1)]      # w_0, …, w_(k-1)
    for _ in k:n
        push!(window, sum(window))
        popfirst!(window)
    end
    return window[end]
end

"""Number `w_L` of admissible length-`L` binary strings of the k-bonacci chain."""
kbonacci_site_count(k::Integer, L::Integer) = Int(kbonacci_number(k, L))

"""
    kbonacci_digits(k, n, L) -> Vector{Int}

Length-`L`, most-significant-first binary expansion of the physical site label
`n in 0:w_L-1` in the k-bonacci numeration system, with weights
`w_(L-1), …, w_0`, obtained greedily. The result never contains `k`
consecutive ones. For `k = 2` this is [`fibonacci_zeckendorf_digits`](@ref).
"""
function kbonacci_digits(k::Integer, n::Integer, L::Integer)
    L >= 1 || throw(ArgumentError("L must be positive"))
    N = kbonacci_site_count(k, L)
    0 <= n < N || throw(ArgumentError("site label must satisfy 0 <= n < $N"))
    digits = zeros(Int, L)
    remainder = big(n)
    for position in 1:L
        weight = kbonacci_number(k, L - position)
        if remainder >= weight
            digits[position] = 1
            remainder -= weight
        end
    end
    iszero(remainder) || error("k-bonacci conversion failed for k=$k, n=$n, L=$L")
    return digits
end

# Number of trailing ones of a digit string; this selects the letter.
function _kbonacci_trailing_ones(digits::AbstractVector{<:Integer})
    r = 0
    for d in Iterators.reverse(digits)
        d == 1 || break
        r += 1
    end
    return r
end

"""Letter symbols `[:A, :B, …]` of the k-bonacci alphabet (`k <= 26`)."""
function kbonacci_letters(k::Integer)
    2 <= k <= 26 || throw(ArgumentError("letter symbols are defined for 2 <= k <= 26"))
    return [Symbol('A' + i) for i in 0:(k - 1)]
end

# k-state automaton MPS on Qubit sites. Link state s (1-based) means "the string
# read so far ends in s-1 ones" (state 1 after a 0 or before reading anything).
# A digit 0 resets to state 1, a digit 1 advances s -> s+1 and is forbidden from
# state k, which annihilates every string with k consecutive ones. The amplitude
# of an admissible string is values[r+1] with r its number of trailing ones.
function _kbonacci_automaton_mps(k::Integer, sites; values)
    L = length(sites)
    k >= 2 || throw(ArgumentError("k-bonacci order k must be at least 2"))
    L >= 2 || throw(ArgumentError("k-bonacci chains require L >= 2"))
    length(values) == k ||
        throw(ArgumentError("values must hold one amplitude per letter (k = $k), got $(length(values))"))
    all(s -> dim(s) == 2, sites) ||
        throw(ArgumentError("k-bonacci sites must be binary (Qubit) sites"))
    T = promote_type(Float64, map(typeof, Tuple(values))...)
    links = [Index(k, "KBAutomaton,Link,l=$i") for i in 1:(L - 1)]
    word = MPS(sites)

    first = ITensor(T, sites[1], links[1])
    first[sites[1] => 1, links[1] => 1] = one(T)
    first[sites[1] => 2, links[1] => 2] = one(T)
    word[1] = first

    for i in 2:(L - 1)
        bulk = ITensor(T, links[i - 1], sites[i], links[i])
        for s in 1:k
            bulk[links[i - 1] => s, sites[i] => 1, links[i] => 1] = one(T)
        end
        for s in 1:(k - 1)
            bulk[links[i - 1] => s, sites[i] => 2, links[i] => s + 1] = one(T)
        end
        word[i] = bulk
    end

    last = ITensor(T, links[end], sites[end])
    for s in 1:k
        last[links[end] => s, sites[end] => 1] = T(values[1])
    end
    for s in 1:(k - 1)
        last[links[end] => s, sites[end] => 2] = T(values[s + 1])
    end
    word[end] = last
    return word
end

"""
    kbonacci_decrement_mpo(k, sites; boundary=:open) -> MPO

Physical decrement `K|n> = |n-1>` in the k-bonacci numeration system. The
`i`-th term fires when the least significant one sits at position `i`: that
digit is cleared and the (all-zero) tail is rewritten with the repeating
pattern `1, …, 1, 0` (`k - 1` ones then a zero), which encodes `w_j - 1` on `j`
digits. With periodic boundaries the only added transition is
`|0> -> |w_L - 1>`; the ambient register state `2^L - 1` is never wrapped.
Uses the qubit operators `FibLower`, `FibRaise`, `FibP0` of Fibonacci.jl.
"""
function kbonacci_decrement_mpo(k::Integer, sites; boundary::Symbol=:open)
    k >= 2 || throw(ArgumentError("k-bonacci order k must be at least 2"))
    boundary in (:open, :periodic) ||
        throw(ArgumentError("boundary must be :open or :periodic"))
    L = length(sites)
    shifts = OpSum()
    for i in 1:L
        term = OpSum()
        term += 1.0, "FibLower", i
        for j in (i + 1):L
            term *= ((j - i) % k == 0 ? "FibP0" : "FibRaise", j)
        end
        shifts += term
    end

    if boundary === :periodic
        last_digits = kbonacci_digits(k, kbonacci_site_count(k, L) - 1, L)
        op_for(digit) = isone(digit) ? "FibRaise" : "FibP0"
        wrap = OpSum()
        wrap += 1.0, op_for(last_digits[1]), 1
        for i in 2:L
            wrap *= (op_for(last_digits[i]), i)
        end
        shifts += wrap
    end
    return MPO(shifts, sites)
end

function physical_projector(space::KBonacciPositionSpace, H::TBHamiltonian)
    length(H.sites) == H.L ||
        error("KBonacciPositionSpace currently supports position-only Hamiltonians")
    return copy(space.projector)
end

function physical_site_state(space::KBonacciPositionSpace, H::TBHamiltonian, x::Integer)
    1 <= x <= H.N || throw(BoundsError(1:H.N, x))
    length(H.sites) == H.L ||
        error("KBonacciPositionSpace currently supports position-only Hamiltonians")
    return MPS(H.sites, string.(kbonacci_digits(space.k, x - 1, H.L)))
end

function site_axis(::KBonacciPositionSpace, H::TBHamiltonian;
                   ordering::Symbol=:physical, kwargs...)
    ordering === :physical || throw(ArgumentError(
        "ordering=:$ordering is not available for KBonacciPositionSpace; " *
        "only :physical is defined (conumbering is currently Fibonacci-only)"))
    return collect(0:(H.N - 1))
end

function site_permutation(::KBonacciPositionSpace, H::TBHamiltonian;
                          ordering::Symbol=:physical, kwargs...)
    ordering === :physical || throw(ArgumentError(
        "ordering=:$ordering is not available for KBonacciPositionSpace; " *
        "only :physical is defined (conumbering is currently Fibonacci-only)"))
    return collect(1:H.N)
end

"""
    kbonacci_bond_symbol(k, L, bond) -> Symbol

Return the letter (`:A`, `:B`, `:C`, …) of the 1-indexed bond beginning at
`bond` in the canonical `w_L`-bond periodic approximant of the k-bonacci chain:
the letter of physical site `bond - 1`, selected by its number of trailing ones.
Bond `N` joins site `N` to site `1` when periodic boundaries are used.
"""
function kbonacci_bond_symbol(k::Integer, L::Integer, bond::Integer)
    N = kbonacci_site_count(k, L)
    1 <= bond <= N || throw(BoundsError(1:N, bond))
    r = _kbonacci_trailing_ones(kbonacci_digits(k, bond - 1, L))
    return kbonacci_letters(k)[r + 1]
end

"""
    kbonacci_hamiltonian(k, L; values, model=:hopping, t=1.0, onsite=0.0,
                         boundary=:periodic, scale=nothing, padding=1.05,
                         cutoff=1e-12, maxdim=200) -> TBHamiltonian

Construct the k-bonacci chain of order `k` (`a_i -> a_1 a_(i+1)`, `a_k -> a_1`)
in its projected numeration position space on `L` Qubit sites. `values` holds
one amplitude per letter `a_1, …, a_k` (i.e. `A, B, C, …`) and the chain has
`H.N = w_L` physical sites.

- `model=:onsite`: `values` are onsite energies and `t` is uniform hopping.
- `model=:hopping`: `values` are bond amplitudes and `onsite` is uniform.

The Hamiltonian is assembled as `P (V + T K + h.c.) P`, where `P` is the
validity projector, `T`/`V` the diagonal word MPO, and `K` the decrement
[`kbonacci_decrement_mpo`](@ref). `k = 2` reproduces the Fibonacci chain of
[`fibonacci_hamiltonian`](@ref) with `values = (A, B)`; `k = 3` is the
Tribonacci chain. The default periodic boundary closes the physical `w_L`-site
approximant.
"""
function kbonacci_hamiltonian(
    k::Integer, L::Integer; values,
    model::Symbol=:hopping,
    t::Number=1.0,
    onsite::Number=0.0,
    boundary::Symbol=:periodic,
    scale=nothing,
    padding::Real=1.05,
    cutoff::Real=1e-12,
    maxdim::Integer=200,
)
    k >= 2 || throw(ArgumentError("k-bonacci order k must be at least 2"))
    L >= 2 || throw(ArgumentError("k-bonacci chains require L >= 2"))
    length(values) == k ||
        throw(ArgumentError("values must hold exactly k = $k letter amplitudes, got $(length(values))"))
    all(v -> v isa Number, values) ||
        throw(ArgumentError("values must be numbers"))
    model in (:onsite, :hopping) ||
        throw(ArgumentError("model must be :onsite or :hopping"))
    boundary in (:open, :periodic) ||
        throw(ArgumentError("boundary must be :open or :periodic"))
    padding > 1 || throw(ArgumentError("padding must be greater than 1"))
    amplitudes = collect(values)
    if model === :onsite
        all(isreal, amplitudes) ||
            throw(ArgumentError("onsite k-bonacci values must be real"))
    else
        isreal(onsite) ||
            throw(ArgumentError("the uniform onsite energy must be real"))
    end

    sites = siteinds("Qubit", L; conserve_qns=false)
    word_mps = _kbonacci_automaton_mps(k, sites; values=amplitudes)
    valid_mps = _kbonacci_automaton_mps(k, sites; values=ones(k))
    word = mps_to_diagonal_mpo(word_mps, sites)
    P = mps_to_diagonal_mpo(valid_mps, sites)
    K = kbonacci_decrement_mpo(k, sites; boundary)

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
        lo, hi = extrema(Float64.(real.(amplitudes)))
        ((lo + hi) / 2, (hi - lo) / 2 + 2abs(t))
    else
        (Float64(real(onsite)), 2maximum(abs.(amplitudes)))
    end
    scale_value = isnothing(scale) ? padding * Float64(halfwidth) : Float64(scale)
    scale_value > 0 || throw(ArgumentError("KPM scale must be positive"))

    N = kbonacci_site_count(k, L)
    H = TBHamiltonian(L, N, sites, mpo, _chain_geometry(),
                      scale_value, Float64(center),
                      nothing, nothing, nothing, nothing, 0, nothing)
    H.position_space = KBonacciPositionSpace(Int(k), P)
    return H
end

function _build_kbonacci(params, L::Integer;
                         k=nothing, scale=nothing, tol=1e-12, maxdim=200,
                         kwargs...)
    k === nothing && throw(ArgumentError(
        "get_Hamiltonian(\"kbonacci\", …) requires the keyword k " *
        "(k=2 Fibonacci, k=3 Tribonacci, k=4 Tetranacci, …)"))
    k >= 2 || throw(ArgumentError("k-bonacci order k must be at least 2"))
    p = if params isa NamedTuple
        Dict{Symbol,Any}(pairs(params))
    elseif params isa AbstractDict
        Dict{Symbol,Any}(Symbol(key) => v for (key, v) in pairs(params))
    else
        throw(ArgumentError("k-bonacci parameters must be a NamedTuple or dictionary " *
                            "containing values=(a_1, …, a_k) or the letter keys A, B, C, …"))
    end
    letters = k <= 26 ? kbonacci_letters(k) : Symbol[]
    values = if haskey(p, :values)
        any(letter -> haskey(p, letter), letters) &&
            throw(ArgumentError("give either values=(…) or the letter keys $(join(letters, ", ")), not both"))
        p[:values]
    elseif !isempty(letters) && all(letter -> haskey(p, letter), letters)
        [p[letter] for letter in letters]
    else
        throw(ArgumentError("k-bonacci parameters must contain values=(a_1, …, a_k)" *
                            (isempty(letters) ? "" : " or all of the letter keys $(join(letters, ", "))")))
    end
    allowed = Set((:values, :t, :onsite, letters...))
    unknown = setdiff(Set(keys(p)), allowed)
    isempty(unknown) || throw(ArgumentError("unknown k-bonacci parameters: $(collect(unknown))"))
    return kbonacci_hamiltonian(
        k, L; values,
        t=get(p, :t, 1.0), onsite=get(p, :onsite, 0.0),
        scale=scale, cutoff=tol, maxdim=maxdim, kwargs...,
    )
end

# Dense small-system oracle used only by the test suite.
function _dense_kbonacci_hamiltonian(
    k::Integer, L::Integer; values,
    model::Symbol=:hopping,
    t::Number=1.0,
    onsite::Number=0.0,
    boundary::Symbol=:periodic,
)
    model in (:onsite, :hopping) ||
        throw(ArgumentError("model must be :onsite or :hopping"))
    boundary in (:open, :periodic) ||
        throw(ArgumentError("boundary must be :open or :periodic"))
    length(values) == k ||
        throw(ArgumentError("values must hold exactly k = $k letter amplitudes"))
    N = kbonacci_site_count(k, L)
    amplitudes = collect(values)
    word = [amplitudes[_kbonacci_trailing_ones(kbonacci_digits(k, n, L)) + 1]
            for n in 0:(N - 1)]
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
