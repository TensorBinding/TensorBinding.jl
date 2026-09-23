# Fibonacci.jl — Fibonacci position space and Hamiltonian constructors
#
# Fibonacci chains use Zeckendorf strings (no adjacent ones) inside an ambient
# 2^L qubit register. The validity projector is therefore the physical identity
# for every projected-space solver operation.

"""
    FibonacciPositionSpace(projector)

Zeckendorf-encoded Fibonacci position space. `projector` is the identity on the
`F_(L+2)` valid strings embedded in the ambient `2^L` qubit register.
"""
struct FibonacciPositionSpace <: AbstractPositionSpace
    projector::MPO
end

ambient_dimension(::FibonacciPositionSpace, H::TBHamiltonian) = big(2)^H.L

"""Return the `n`th Fibonacci number with `F_0=0` and `F_1=1`."""
function fibonacci_number(n::Integer)
    n >= 0 || throw(ArgumentError("n must be non-negative"))
    a, b = big(0), big(1)
    for _ in 1:n
        a, b = b, a + b
    end
    return a
end

"""Number `F_(L+2)` of valid length-`L` Zeckendorf strings."""
fibonacci_site_count(L::Integer) = Int(fibonacci_number(L + 2))

"""
    fibonacci_zeckendorf_digits(n, L) -> Vector{Int}

Length-`L`, most-significant-first Zeckendorf representation of the physical
site label `n in 0:F_(L+2)-1`.
"""
function fibonacci_zeckendorf_digits(n::Integer, L::Integer)
    L >= 1 || throw(ArgumentError("L must be positive"))
    N = fibonacci_site_count(L)
    0 <= n < N || throw(ArgumentError("site label must satisfy 0 <= n < $N"))
    digits = zeros(Int, L)
    remainder = big(n)
    for (position, k) in enumerate((L + 1):-1:2)
        weight = fibonacci_number(k)
        if weight <= remainder
            digits[position] = 1
            remainder -= weight
        end
    end
    iszero(remainder) || error("Zeckendorf conversion failed for n=$n, L=$L")
    return digits
end

function _fibonacci_automaton_mps(sites; A=0.0, B=1.0)
    L = length(sites)
    L >= 2 || throw(ArgumentError("Fibonacci chains require L >= 2"))
    T = promote_type(Float64, typeof(A), typeof(B))
    links = [Index(2, "FibAutomaton,Link,l=$i") for i in 1:(L - 1)]
    word = MPS(sites)

    first = ITensor(T, sites[1], links[1])
    first[sites[1] => 1, links[1] => 1] = one(T)
    first[sites[1] => 2, links[1] => 2] = one(T)
    word[1] = first

    for i in 2:(L - 1)
        bulk = ITensor(T, links[i - 1], sites[i], links[i])
        bulk[links[i - 1] => 1, sites[i] => 1, links[i] => 1] = one(T)
        bulk[links[i - 1] => 2, sites[i] => 1, links[i] => 1] = one(T)
        bulk[links[i - 1] => 1, sites[i] => 2, links[i] => 2] = one(T)
        word[i] = bulk
    end

    last = ITensor(T, links[end], sites[end])
    last[links[end] => 1, sites[end] => 1] = A
    last[links[end] => 2, sites[end] => 1] = A
    last[links[end] => 1, sites[end] => 2] = B
    word[end] = last
    return word
end

ITensors.op(::OpName"FibLower", ::SiteType"Qubit") = [0 1; 0 0]
ITensors.op(::OpName"FibRaise", ::SiteType"Qubit") = [0 0; 1 0]
ITensors.op(::OpName"FibP0", ::SiteType"Qubit") = [1 0; 0 0]

"""
    fibonacci_decrement_mpo(sites; boundary=:open) -> MPO

Physical decrement `K|n> = |n-1>` in the Zeckendorf basis. With periodic
boundaries the only added automaton transition is `|0> -> |F_(L+2)-1>`;
wrapping never occurs at the ambient binary state `2^L-1`.
"""
function fibonacci_decrement_mpo(sites; boundary::Symbol=:open)
    boundary in (:open, :periodic) ||
        throw(ArgumentError("boundary must be :open or :periodic"))
    L = length(sites)
    shifts = OpSum()
    for i in 1:L
        term = OpSum()
        term += 1.0, "FibLower", i
        for j in (i + 1):L
            term *= (isodd(j - i) ? "FibRaise" : "FibP0", j)
        end
        shifts += term
    end

    if boundary === :periodic
        last_digits = fibonacci_zeckendorf_digits(fibonacci_site_count(L) - 1, L)
        wrap = OpSum()
        first_op = isone(last_digits[1]) ? "FibRaise" : "FibP0"
        wrap += 1.0, first_op, 1
        for i in 2:L
            op = isone(last_digits[i]) ? "FibRaise" : "FibP0"
            wrap *= (op, i)
        end
        shifts += wrap
    end
    return MPO(shifts, sites)
end

_fibonacci_mpo_adjoint(A::MPO) = swapprime(dag(A), 0, 1)

function physical_projector(space::FibonacciPositionSpace, H::TBHamiltonian)
    length(H.sites) == H.L ||
        error("FibonacciPositionSpace currently supports position-only Hamiltonians")
    return copy(space.projector)
end

function physical_site_state(::FibonacciPositionSpace, H::TBHamiltonian, x::Integer)
    1 <= x <= H.N || throw(BoundsError(1:H.N, x))
    length(H.sites) == H.L ||
        error("FibonacciPositionSpace currently supports position-only Hamiltonians")
    return MPS(H.sites, string.(fibonacci_zeckendorf_digits(x - 1, H.L)))
end

"""
    fibonacci_bond_symbol(L, bond) -> Symbol

Return `:A` or `:B` for the 1-indexed bond beginning at `bond` in the
canonical `F_(L+2)`-bond periodic approximant. Bond `N` joins site `N` to
site `1` when periodic boundaries are used.
"""
function fibonacci_bond_symbol(L::Integer, bond::Integer)
    L >= 2 || throw(ArgumentError("Fibonacci chains require L >= 2"))
    N = fibonacci_site_count(L)
    1 <= bond <= N || throw(BoundsError(1:N, bond))
    return iszero(fibonacci_zeckendorf_digits(bond - 1, L)[end]) ? :A : :B
end

"""
    fibonacci_site_environment(L, site; boundary=:periodic) -> Symbol

Classify a site from its adjacent bonds. `:atomic` means `AA`, while
`:molecular_AB` and `:molecular_BA` retain the orientation of the molecular
site. Open-chain endpoints return `:boundary`.
"""
function fibonacci_site_environment(L::Integer, site::Integer;
                                    boundary::Symbol=:periodic)
    boundary in (:open, :periodic) ||
        throw(ArgumentError("boundary must be :open or :periodic"))
    N = fibonacci_site_count(L)
    1 <= site <= N || throw(BoundsError(1:N, site))
    boundary === :open && site in (1, N) && return :boundary
    left = fibonacci_bond_symbol(L, site == 1 ? N : site - 1)
    right = fibonacci_bond_symbol(L, site)
    left === :A && right === :A && return :atomic
    left === :A && right === :B && return :molecular_AB
    left === :B && right === :A && return :molecular_BA
    error("invalid Fibonacci bond environment $left$right at site $site")
end

function _fibonacci_conumber_rank(L::Integer, site::Integer;
                                  orientation::Symbol=:standard,
                                  alignment::Symbol=:atomic,
                                  origin::Integer=0)
    L >= 2 || throw(ArgumentError("Fibonacci chains require L >= 2"))
    orientation in (:standard, :reversed) ||
        throw(ArgumentError("orientation must be :standard or :reversed"))
    alignment in (:atomic, :raw) ||
        throw(ArgumentError("alignment must be :atomic or :raw"))
    N = fibonacci_site_count(L)
    1 <= site <= N || throw(BoundsError(1:N, site))
    multiplier = Int(fibonacci_number(L)) # F_(n-2), with N=F_n
    x = site - 1
    rank = Int(mod((big(x) + origin) * multiplier, N))

    # For the canonical bond word used by the MPO, the AA acceptance window is
    # [0,F_(L-1)-1] for even L and [1,F_(L-1)] for odd L in raw standard
    # conumbers. Move it between the two F_L molecular windows. This is a
    # cyclic cut of perpendicular space, not a change of conumber multiplier.
    if alignment === :atomic
        rank = Int(mod(big(rank) + multiplier - Int(isodd(L)), N))
    end
    orientation === :reversed && (rank = N - 1 - rank)
    return rank
end

"""
    fibonacci_conumber(L, site; orientation=:standard, alignment=:atomic,
                        centered=true, origin=0) -> Int

Conumber of a single 1-indexed physical site, evaluated without constructing a
Hamiltonian or allocating an `F_(L+2)`-element permutation.

`alignment=:atomic` makes the conumber cut compatible with TensorBinding's
canonical Fibonacci bond phase: the `AA` sites form one central block, between
the `AB` and `BA` molecular blocks. `alignment=:raw` exposes the unshifted
modular residue. `centered=true` labels the ordered ranks around zero; it does
not perform an additional cyclic permutation.
"""
function fibonacci_conumber(L::Integer, site::Integer;
                             orientation::Symbol=:standard,
                             alignment::Symbol=:atomic,
                             centered::Bool=true,
                             origin::Integer=0)
    N = fibonacci_site_count(L)
    rank = _fibonacci_conumber_rank(
        L, site; orientation, alignment, origin,
    )
    return centered ? rank - fld(N, 2) : rank
end

"""Inverse of [`fibonacci_conumber`](@ref), returning a 1-indexed site."""
function fibonacci_site_from_conumber(L::Integer, conumber::Integer;
                                       orientation::Symbol=:standard,
                                       alignment::Symbol=:atomic,
                                       centered::Bool=true,
                                       origin::Integer=0)
    L >= 2 || throw(ArgumentError("Fibonacci chains require L >= 2"))
    orientation in (:standard, :reversed) ||
        throw(ArgumentError("orientation must be :standard or :reversed"))
    alignment in (:atomic, :raw) ||
        throw(ArgumentError("alignment must be :atomic or :raw"))
    N = fibonacci_site_count(L)
    rank = centered ? conumber + fld(N, 2) : conumber
    0 <= rank < N || throw(BoundsError(0:(N - 1), rank))
    orientation === :reversed && (rank = N - 1 - rank)
    multiplier = Int(fibonacci_number(L))
    if alignment === :atomic
        rank = mod(rank - multiplier + Int(isodd(L)), N)
    end
    x = Int(mod(big(rank) * invmod(multiplier, N) - origin, N))
    return x + 1
end

"""
    fibonacci_rg_partition(L; depth=0, centered=true)

Return the molecular–atomic–molecular conumber intervals after `depth`
successive atomic deflations. Each deflation maps `L -> L-3`. The returned
ranges are embedded in the original conumber ordering, so a zoom should slice
these ranges directly rather than re-conumbering the selected sites as a new
canonical chain.
"""
function fibonacci_rg_partition(L::Integer; depth::Integer=0,
                                centered::Bool=true)
    L >= 2 || throw(ArgumentError("Fibonacci chains require L >= 2"))
    depth >= 0 || throw(ArgumentError("depth must be non-negative"))
    effective_L = Int(L)
    window_first = 0
    for _ in 1:depth
        effective_L >= 5 ||
            throw(ArgumentError("depth=$depth deflates L=$L below the supported L=2 approximant"))
        window_first += Int(fibonacci_number(effective_L))
        effective_L -= 3
    end

    molecular_count = Int(fibonacci_number(effective_L))
    atomic_count = Int(fibonacci_number(effective_L - 1))
    window_count = fibonacci_site_count(effective_L)
    @assert 2molecular_count + atomic_count == window_count

    left = window_first:(window_first + molecular_count - 1)
    atomic = (last(left) + 1):(last(left) + atomic_count)
    right = (last(atomic) + 1):(window_first + window_count - 1)
    window = window_first:(window_first + window_count - 1)
    shift = centered ? fld(fibonacci_site_count(L), 2) : 0
    shift_range(r) = (first(r) - shift):(last(r) - shift)
    return (;
        depth, effective_L, window_count, molecular_count, atomic_count,
        window_ranks=window, left_molecular_ranks=left,
        atomic_ranks=atomic, right_molecular_ranks=right,
        window_axis=shift_range(window),
        left_molecular_axis=shift_range(left),
        atomic_axis=shift_range(atomic),
        right_molecular_axis=shift_range(right),
    )
end

"""
    fibonacci_atomic_depth(L, site; kwargs...) -> Int

Number of consecutive atomic deflations containing `site`. This directly
tests whether a site remains inside the nested central atomic windows.
"""
function fibonacci_atomic_depth(L::Integer, site::Integer;
                                orientation::Symbol=:standard,
                                origin::Integer=0)
    rank = _fibonacci_conumber_rank(
        L, site; orientation, alignment=:atomic, origin,
    )
    effective_L = Int(L)
    window_first = 0
    depth = 0
    while effective_L >= 2
        molecular_count = Int(fibonacci_number(effective_L))
        atomic_count = Int(fibonacci_number(effective_L - 1))
        atomic_first = window_first + molecular_count
        atomic_last = atomic_first + atomic_count - 1
        atomic_first <= rank <= atomic_last || break
        depth += 1
        window_first = atomic_first
        effective_L -= 3
    end
    return depth
end

function _fibonacci_conumbering(H::TBHamiltonian;
                                orientation::Symbol=:standard,
                                centered::Bool=true,
                                origin::Integer=0,
                                alignment::Symbol=:atomic)
    H.position_space isa FibonacciPositionSpace ||
        throw(ArgumentError("conumbering requires FibonacciPositionSpace"))
    N = H.N
    ranks = [_fibonacci_conumber_rank(
                 H.L, site; orientation, alignment, origin,
             ) for site in 1:N]
    labels = centered ? ranks .- fld(N, 2) : ranks
    permutation = sortperm(labels)
    axis = labels[permutation]
    @assert length(unique(axis)) == N
    multiplier = orientation === :standard ?
        Int(fibonacci_number(H.L)) : Int(fibonacci_number(H.L + 1))
    return (; axis, permutation, ranks, labels, multiplier,
              orientation, centered, origin, alignment)
end

function site_axis(::FibonacciPositionSpace, H::TBHamiltonian;
                   ordering::Symbol=:physical,
                   orientation::Symbol=:standard,
                   centered::Bool=true,
                   origin::Integer=0,
                   alignment::Symbol=:atomic,
                   kwargs...)
    ordering === :physical && return collect(0:(H.N - 1))
    ordering === :conumber ||
        throw(ArgumentError("ordering must be :physical or :conumber"))
    return _fibonacci_conumbering(
        H; orientation, centered, origin, alignment,
    ).axis
end

function site_permutation(::FibonacciPositionSpace, H::TBHamiltonian;
                          ordering::Symbol=:physical,
                          orientation::Symbol=:standard,
                          centered::Bool=true,
                          origin::Integer=0,
                          alignment::Symbol=:atomic,
                          kwargs...)
    ordering === :physical && return collect(1:H.N)
    ordering === :conumber ||
        throw(ArgumentError("ordering must be :physical or :conumber"))
    return _fibonacci_conumbering(
        H; orientation, centered, origin, alignment,
    ).permutation
end

"""
    fibonacci_hamiltonian(L; A, B, model=:hopping, t=1.0, onsite=0.0,
                          boundary=:periodic, scale=nothing, padding=1.05,
                          cutoff=1e-12, maxdim=200) -> TBHamiltonian

Construct a Fibonacci chain in the projected Zeckendorf position space.

- `model=:onsite`: `A` and `B` are onsite energies and `t` is uniform hopping.
- `model=:hopping`: `A` and `B` are bond amplitudes and `onsite` is uniform.

The default periodic boundary closes the physical `F_(L+2)`-site approximant.
For odd `F_(L+2)`, a periodic hopping chain is an odd cycle and therefore is not
exactly chiral even when `onsite=0`.
"""
function fibonacci_hamiltonian(
    L::Integer; A, B,
    model::Symbol=:hopping,
    t::Number=1.0,
    onsite::Number=0.0,
    boundary::Symbol=:periodic,
    scale=nothing,
    padding::Real=1.05,
    cutoff::Real=1e-12,
    maxdim::Integer=200,
)
    L >= 2 || throw(ArgumentError("Fibonacci chains require L >= 2"))
    model in (:onsite, :hopping) ||
        throw(ArgumentError("model must be :onsite or :hopping"))
    boundary in (:open, :periodic) ||
        throw(ArgumentError("boundary must be :open or :periodic"))
    padding > 1 || throw(ArgumentError("padding must be greater than 1"))
    if model === :onsite
        isreal(A) && isreal(B) ||
            throw(ArgumentError("onsite Fibonacci values A and B must be real"))
    else
        isreal(onsite) ||
            throw(ArgumentError("the uniform onsite energy must be real"))
    end

    sites = siteinds("Qubit", L; conserve_qns=false)
    word_mps = _fibonacci_automaton_mps(sites; A, B)
    valid_mps = _fibonacci_automaton_mps(sites; A=1.0, B=1.0)
    word = mps_to_diagonal_mpo(word_mps, sites)
    P = mps_to_diagonal_mpo(valid_mps, sites)
    K = fibonacci_decrement_mpo(sites; boundary)

    V, TK = if model === :onsite
        word, t * K
    else
        onsite * P, apply(word, K; cutoff=cutoff, maxdim=maxdim)
    end
    hopping = +(TK, _fibonacci_mpo_adjoint(TK); cutoff=cutoff, maxdim=maxdim)
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

    N = fibonacci_site_count(L)
    H = TBHamiltonian(L, N, sites, mpo, _chain_geometry(),
                      scale_value, Float64(center),
                      nothing, nothing, nothing, nothing, 0, nothing)
    H.position_space = FibonacciPositionSpace(P)
    return H
end

function _build_fibonacci(params, L::Integer;
                          scale=nothing, tol=1e-12, maxdim=200, kwargs...)
    p = if params isa NamedTuple
        Dict{Symbol,Any}(pairs(params))
    elseif params isa AbstractDict
        Dict{Symbol,Any}(Symbol(k) => v for (k, v) in pairs(params))
    else
        throw(ArgumentError("fibonacci parameters must be a NamedTuple or dictionary containing A and B"))
    end
    haskey(p, :A) && haskey(p, :B) ||
        throw(ArgumentError("fibonacci parameters must contain A and B"))
    allowed = Set((:A, :B, :t, :onsite))
    unknown = setdiff(Set(keys(p)), allowed)
    isempty(unknown) || throw(ArgumentError("unknown fibonacci parameters: $(collect(unknown))"))
    return fibonacci_hamiltonian(
        L; A=p[:A], B=p[:B],
        t=get(p, :t, 1.0), onsite=get(p, :onsite, 0.0),
        scale=scale, cutoff=tol, maxdim=maxdim, kwargs...,
    )
end

# Dense small-system oracle used only by the test suite.
function _dense_fibonacci_hamiltonian(
    L::Integer; A, B,
    model::Symbol=:hopping,
    t::Number=1.0,
    onsite::Number=0.0,
    boundary::Symbol=:periodic,
)
    model in (:onsite, :hopping) ||
        throw(ArgumentError("model must be :onsite or :hopping"))
    boundary in (:open, :periodic) ||
        throw(ArgumentError("boundary must be :open or :periodic"))
    N = fibonacci_site_count(L)
    word = [iszero(fibonacci_zeckendorf_digits(n, L)[end]) ? A : B
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
