# FibonacciSampling.jl -- scalable sampling plans for Fibonacci LDOS jobs

"""
    fibonacci_ldos_sampling_plan(L; depth=0, num_x=100, num_avg=1,
                                 orientation=:standard, alignment=:atomic,
                                 centered=true, origin=0)
    fibonacci_ldos_sampling_plan(H::TBHamiltonian; kwargs...)

Build a deterministic, bounded-size LDOS sampling plan for an `L`-qubit
Fibonacci approximant. At `depth == 0` the plan covers the complete
`F_(L+2)` conumber interval. Each additional depth selects the nested atomic
renormalization window from [`fibonacci_rg_partition`](@ref), with
`effective_L == L - 3depth`.

The selected inherited interval is split into `min(num_x, window_count)`
contiguous integer intervals whose widths differ by at most one. Up to
`num_avg` equidistant conumbers (including both interval endpoints when there
is more than one sample) are chosen in each interval and mapped directly to
physical sites with [`fibonacci_site_from_conumber`](@ref). No full conumber
permutation or other `F_(L+2)`-element array is constructed; storage is
proportional to the requested output and sample counts.

Returned fields useful to an LDOS/Slurm/HDF5 workflow include:

- `groups`: physical-site vectors to pass as `x_groups` with
  `ordering=:physical`;
- `centers`: physical sites at the representative interval conumbers;
- `conumber_axis`: those representative conumbers in the original `L`
  coordinate system;
- `intervals`, `interval_first`, and `interval_last`: represented inherited
  conumber intervals;
- `sample_conumbers`: the inherited conumbers corresponding to `groups`;
- `sample_sites_flat`, `sample_conumbers_flat`, one-based `group_offsets`, and
  zero-based `group_offsets_zero`: flat representations convenient for Julia
  and Python/HDF5 consumers respectively (`flat[group_offsets[i]:
  group_offsets[i+1]-1]` reconstructs group `i` in Julia);
- `column_indices`: stable one-based output-column identifiers;
- `depth`, `effective_L`, the original `L` and `N`, requested/actual sampling
  counts, and all conumber conventions.

`intervals` and every field containing `conumber` use the requested `centered`
label convention. The corresponding `*_rank*` fields are always uncentered
ranks in `0:N-1`. Thus a zoom always retains its original-`L` coordinates;
the selected sites are never re-conumbered as an independent shorter chain.

When `num_avg` is larger than an interval, that interval is sampled at every
integer conumber and its `group_sizes` entry is smaller than `num_avg`.
`num_x` in the result is the actual number of output columns, while
`num_x_requested` records the input value.

Nested (`depth > 0`) windows are defined only for the canonical atomic phase,
so they require `alignment=:atomic` and `origin=0`. Reversing the orientation
is supported because it maps the canonical atomic interval onto itself.
"""
function fibonacci_ldos_sampling_plan(
    L::Integer;
    depth::Integer=0,
    num_x::Integer=100,
    num_avg::Integer=1,
    orientation::Symbol=:standard,
    alignment::Symbol=:atomic,
    centered::Bool=true,
    origin::Integer=0,
)
    num_x > 0 || throw(ArgumentError("num_x must be positive"))
    num_avg > 0 || throw(ArgumentError("num_avg must be positive"))
    orientation in (:standard, :reversed) ||
        throw(ArgumentError("orientation must be :standard or :reversed"))
    alignment in (:atomic, :raw) ||
        throw(ArgumentError("alignment must be :atomic or :raw"))
    if depth > 0 && (alignment !== :atomic || !iszero(origin))
        throw(ArgumentError(
            "depth > 0 requires alignment=:atomic and origin=0 so the selected " *
            "window remains the canonical nested atomic renormalization window",
        ))
    end

    # Work in uncentered ranks while partitioning. This keeps the RG embedding
    # independent of how callers choose to label the inherited conumber axis.
    partition = fibonacci_rg_partition(L; depth, centered=false)
    N = fibonacci_site_count(L)
    window_rank_first = first(partition.window_ranks)
    window_rank_last = last(partition.window_ranks)
    window_count = partition.window_count
    ncolumns = min(Int(num_x), window_count)

    # Tile the window exactly. Putting the remainder in the first intervals is
    # deterministic and makes every width either floor(W/n) or ceil(W/n).
    base_width, remainder = divrem(window_count, ncolumns)
    rank_intervals = Vector{UnitRange{Int}}(undef, ncolumns)
    cursor = window_rank_first
    for column in 1:ncolumns
        width = base_width + Int(column <= remainder)
        rank_intervals[column] = cursor:(cursor + width - 1)
        cursor += width
    end
    @assert cursor == window_rank_last + 1

    shift = centered ? fld(N, 2) : 0
    rank_to_conumber(rank::Int) = rank - shift
    to_axis_interval(interval::UnitRange{Int}) =
        rank_to_conumber(first(interval)):rank_to_conumber(last(interval))

    intervals = [to_axis_interval(interval) for interval in rank_intervals]
    interval_first = first.(intervals)
    interval_last = last.(intervals)
    interval_rank_first = first.(rank_intervals)
    interval_rank_last = last.(rank_intervals)

    # Integer samples are as uniformly spaced as possible. With one requested
    # sample use the lower integer midpoint; with two or more include endpoints.
    function equidistant_ranks(interval::UnitRange{Int})
        width = length(interval)
        count = min(Int(num_avg), width)
        lo = first(interval)
        count == 1 && return Int[lo + fld(width - 1, 2)]
        return Int[lo + fld(k * (width - 1), count - 1)
                   for k in 0:(count - 1)]
    end

    sample_ranks = [equidistant_ranks(interval) for interval in rank_intervals]
    sample_conumbers = [[rank_to_conumber(rank) for rank in ranks]
                         for ranks in sample_ranks]
    center_ranks = Int[first(interval) + fld(length(interval) - 1, 2)
                       for interval in rank_intervals]
    conumber_axis = rank_to_conumber.(center_ranks)

    site_from_rank(rank::Int) = fibonacci_site_from_conumber(
        L, rank_to_conumber(rank);
        orientation, alignment, centered, origin,
    )
    groups = [[site_from_rank(rank) for rank in ranks] for ranks in sample_ranks]
    centers = site_from_rank.(center_ranks)

    group_sizes = length.(groups)
    group_offsets = Vector{Int}(undef, ncolumns + 1)
    group_offsets[1] = 1
    for column in 1:ncolumns
        group_offsets[column + 1] = group_offsets[column] + group_sizes[column]
    end
    group_offsets_zero = group_offsets .- 1
    group_offsets_base = 1
    total_samples = group_offsets[end] - 1
    sample_sites_flat = Vector{Int}(undef, total_samples)
    sample_conumbers_flat = Vector{Int}(undef, total_samples)
    sample_ranks_flat = Vector{Int}(undef, total_samples)
    for column in 1:ncolumns
        destination = group_offsets[column]:(group_offsets[column + 1] - 1)
        sample_sites_flat[destination] = groups[column]
        sample_conumbers_flat[destination] = sample_conumbers[column]
        sample_ranks_flat[destination] = sample_ranks[column]
    end

    window_first = rank_to_conumber(window_rank_first)
    window_last = rank_to_conumber(window_rank_last)
    metadata = (;
        format="TensorBinding.fibonacci_ldos_sampling_plan",
        format_version=1,
        L=Int(L),
        N,
        depth=Int(depth),
        effective_L=partition.effective_L,
        window_count,
        window_first,
        window_last,
        window_rank_first,
        window_rank_last,
        num_x=ncolumns,
        num_x_requested=Int(num_x),
        num_avg=Int(num_avg),
        total_samples,
        group_offsets_base,
        orientation=String(orientation),
        alignment=String(alignment),
        centered,
        origin=Int(origin),
    )

    return (;
        groups,
        centers,
        conumber_axis,
        intervals,
        interval_first,
        interval_last,
        interval_rank_first,
        interval_rank_last,
        sample_conumbers,
        sample_ranks,
        sample_sites_flat,
        sample_conumbers_flat,
        sample_ranks_flat,
        group_offsets,
        group_offsets_zero,
        group_offsets_base,
        group_sizes,
        column_indices=collect(1:ncolumns),
        L=Int(L),
        N,
        depth=Int(depth),
        effective_L=partition.effective_L,
        window_count,
        window_first,
        window_last,
        window_rank_first,
        window_rank_last,
        num_x=ncolumns,
        num_x_requested=Int(num_x),
        num_avg=Int(num_avg),
        total_samples,
        orientation,
        alignment,
        centered,
        origin=Int(origin),
        metadata,
    )
end

function fibonacci_ldos_sampling_plan(H::TBHamiltonian; kwargs...)
    H.position_space isa FibonacciPositionSpace ||
        throw(ArgumentError("fibonacci LDOS sampling requires FibonacciPositionSpace"))
    expected_N = fibonacci_site_count(H.L)
    H.N == expected_N ||
        throw(ArgumentError("Hamiltonian has N=$(H.N), expected F_(L+2)=$expected_N for L=$(H.L)"))
    return fibonacci_ldos_sampling_plan(H.L; kwargs...)
end
