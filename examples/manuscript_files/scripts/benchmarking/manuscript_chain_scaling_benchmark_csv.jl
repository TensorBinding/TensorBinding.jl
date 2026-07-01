using SparseArrays
using LinearAlgebra
using Printf

include(joinpath(@__DIR__, "../../../src/TensorBinding.jl"))
using .TensorBinding

function csv_value(x)
    if x isa AbstractFloat
        isnan(x) && return "NaN"
        isinf(x) && return signbit(x) ? "-Inf" : "Inf"
    end

    s = string(x)
    if occursin(",", s) || occursin("\"", s) || occursin("\n", s)
        return "\"" * replace(s, "\"" => "\"\"") * "\""
    end
    return s
end

function write_rows_csv(path, header, rows)
    mkpath(dirname(path))
    open(path, "w") do io
        println(io, join(header, ","))
        for row in rows
            println(io, join(csv_value.(row), ","))
        end
    end
    return path
end

function jackson_kernel(ncheb)
    return [
        (((ncheb - n + 1) * cos(pi * n / (ncheb + 1)) +
          sin(pi * n / (ncheb + 1)) / tan(pi / (ncheb + 1))) / (ncheb + 1))
        for n in 0:(ncheb - 1)
    ]
end

function kpm_ldos(mu, omega, scale, ncheb)
    g = jackson_kernel(ncheb)
    omega_scaled = omega / scale
    abs(omega_scaled) >= 1 && return 0.0

    return (2 / (pi * scale * sqrt(1 - omega_scaled^2))) *
           sum(
               g[n + 1] * (n == 0 ? 1.0 : 2.0) * mu[n + 1] *
               cos(n * acos(omega_scaled))
               for n in 0:(ncheb - 1)
           )
end

function fit_powerlaw(n_vals, y_vals)
    log_n = log10.(Float64.(n_vals))
    log_y = log10.(Float64.(y_vals))
    coeffs = [ones(length(log_n)) log_n] \ log_y
    return 10.0^coeffs[1], coeffs[2]
end

function compute_fit_outputs(lvals, nplot, series; fit_from_l=6, show_from_l=5)
    fit_from = lvals .>= fit_from_l
    show_fit = lvals .>= show_from_l
    fit_lvals = lvals[show_fit]
    fit_nplot = nplot[show_fit]
    fit_values = Vector{Vector{Float64}}()
    fit_params = []

    for (method, values) in series
        valid = isfinite.(values) .& (values .> 0) .& fit_from

        if count(valid) >= 3
            a, b = fit_powerlaw(nplot[valid], values[valid])
            push!(fit_values, a .* fit_nplot .^ b)
            push!(fit_params, (method, a, b, count(valid), minimum(lvals[valid]), maximum(lvals[valid])))
        else
            push!(fit_values, fill(NaN, length(fit_nplot)))
            push!(fit_params, (method, NaN, NaN, count(valid), NaN, NaN))
        end
    end

    fit_rows = (
        (fit_lvals[idx], Int64(2)^fit_lvals[idx], (values[idx] for values in fit_values)...)
        for idx in eachindex(fit_lvals)
    )

    return fit_rows, fit_params
end

function warmup_benchmark(t_hop, hscale, ncheb, omega_bm)
    l_warmup = 4
    n_warmup = 2^l_warmup
    k_warmup = 2^3

    h_tb = TensorBinding.get_Hamiltonian("chain_1d", t_hop; L=l_warmup)
    h_tb.scale = hscale
    TensorBinding.get_ldos_online(h_tb, ncheb, k_warmup, [omega_bm]; maxdim=50, cutoff=1e-8)

    h_sparse = spdiagm(-1 => fill(-t_hop, n_warmup - 1), 1 => fill(-t_hop, n_warmup - 1))
    e_k = zeros(n_warmup)
    e_k[k_warmup] = 1.0
    v0 = copy(e_k)
    v1 = (1 / hscale) .* (h_sparse * e_k)
    mu = [v0[k_warmup], v1[k_warmup]]
    for _ in 3:ncheb
        v2 = (2 / hscale) .* (h_sparse * v1) .- v0
        push!(mu, v2[k_warmup])
        v0 = v1
        v1 = v2
    end
    kpm_ldos(mu, omega_bm, hscale, ncheb)

    f = eigen(SymTridiagonal(zeros(n_warmup), fill(-t_hop, n_warmup - 1)))
    eta = pi * hscale / ncheb
    sum(abs2.(f.vectors[k_warmup, :]) .* (eta / pi) ./ ((omega_bm .- f.values).^2 .+ eta^2))

    return nothing
end

function run_timing_benchmark(lvals, t_hop, hscale, ncheb, omega_bm, l_max_sparse, l_max_ed)
    times_tb = fill(NaN, length(lvals))
    times_sparse = fill(NaN, length(lvals))
    times_ed = fill(NaN, length(lvals))

    warmup_benchmark(t_hop, hscale, ncheb, omega_bm)
    println("Warmup done - starting timing benchmark")

    for (idx, lval) in enumerate(lvals)
        nsites = 2^lval
        k_center = nsites ÷ 2
        GC.gc()

        times_tb[idx] = @elapsed begin
            h_tb = TensorBinding.get_Hamiltonian("chain_1d", t_hop; L=lval)
            h_tb.scale = hscale
            TensorBinding.get_ldos_online(h_tb, ncheb, k_center, [omega_bm]; maxdim=100, cutoff=1e-8)
        end

        if lval <= l_max_sparse
            h_sparse = spdiagm(-1 => fill(-t_hop, nsites - 1), 1 => fill(-t_hop, nsites - 1))
            times_sparse[idx] = @elapsed begin
                e_k = zeros(nsites)
                e_k[k_center] = 1.0
                v0 = copy(e_k)
                v1 = (1 / hscale) .* (h_sparse * e_k)
                mu = zeros(ncheb)
                mu[1] = v0[k_center]
                mu[2] = v1[k_center]
                for n in 3:ncheb
                    v2 = (2 / hscale) .* (h_sparse * v1) .- v0
                    mu[n] = v2[k_center]
                    v0 = v1
                    v1 = v2
                end
                kpm_ldos(mu, omega_bm, hscale, ncheb)
            end
        end

        if lval <= l_max_ed
            times_ed[idx] = @elapsed begin
                f = eigen(SymTridiagonal(zeros(nsites), fill(-t_hop, nsites - 1)))
                eta = pi * hscale / ncheb
                sum(abs2.(f.vectors[k_center, :]) .* (eta / pi) ./ ((omega_bm .- f.values).^2 .+ eta^2))
            end
        end

        @printf(
            "L=%2d  N=%9d  MPO KPM: %7.3fs  Sparse KPM: %s  ED: %s\n",
            lval,
            nsites,
            times_tb[idx],
            lval <= l_max_sparse ? @sprintf("%7.3fs", times_sparse[idx]) : "      N/A",
            lval <= l_max_ed ? @sprintf("%7.3fs", times_ed[idx]) : "    N/A",
        )
    end

    return times_tb, times_sparse, times_ed
end

function run_memory_benchmark(lvals, t_hop)
    mem_tb = fill(NaN, length(lvals))
    mem_sparse = zeros(length(lvals))
    mem_ed = zeros(length(lvals))

    println("Starting memory benchmark")

    for (idx, lval) in enumerate(lvals)
        nsites = 2^lval
        GC.gc()

        h_tmp = TensorBinding.get_Hamiltonian("chain_1d", t_hop; L=lval)
        mem_tb[idx] = Base.summarysize(h_tmp.mpo) + 2 * lval * (2 * 2 * 2) * sizeof(ComplexF64)
        h_tmp = nothing
        GC.gc()

        nnz = 2 * (nsites - 1)
        mem_sparse[idx] = nnz * 8 + nnz * 8 + (nsites + 1) * 8 + 2 * nsites * 8
        mem_ed[idx] = Float64(nsites)^2 * sizeof(Float64) + nsites * sizeof(Float64)

        @printf(
            "L=%2d  N=%9d  MPO: %7.1f kB  Sparse: %.3g B  ED: %.3g B\n",
            lval,
            nsites,
            mem_tb[idx] / 1e3,
            mem_sparse[idx],
            mem_ed[idx],
        )
    end

    return mem_tb, mem_sparse, mem_ed
end

function export_fit_csvs(output_dir, basename, lvals, nplot, series, value_suffix)
    fit_rows, fit_params = compute_fit_outputs(lvals, nplot, series)
    methods = first.(series)

    fit_csv = write_rows_csv(
        joinpath(output_dir, "$(basename)_fit.csv"),
        ["L", "N", ["$(method)_fit_$(value_suffix)" for method in methods]...],
        fit_rows,
    )

    fit_params_csv = write_rows_csv(
        joinpath(output_dir, "$(basename)_fit_params.csv"),
        ["method", "coefficient_a", "exponent_b", "n_fit_points", "min_fit_L", "max_fit_L"],
        fit_params,
    )

    return fit_csv, fit_params_csv
end

function main()
    t_hop = 1.0
    hscale = 2.1
    ncheb = 100
    omega_bm = 0.0
    l_max_ed = 15
    l_max_sparse = 25
    lvals = collect(2:40)
    nplot = [2.0^lval for lval in lvals]

    output_dir = joinpath(@__DIR__, "results", "benchmarking")
    mkpath(output_dir)

    times_tb, times_sparse, times_ed =
        run_timing_benchmark(lvals, t_hop, hscale, ncheb, omega_bm, l_max_sparse, l_max_ed)

    timing_csv = write_rows_csv(
        joinpath(output_dir, "chain_1d_ldos_timing.csv"),
        ["L", "N", "mpo_kpm_seconds", "sparse_kpm_seconds", "exact_diag_seconds"],
        ((lval, Int64(2)^lval, times_tb[idx], times_sparse[idx], times_ed[idx])
         for (idx, lval) in enumerate(lvals)),
    )

    timing_fit_csv, timing_fit_params_csv = export_fit_csvs(
        output_dir,
        "chain_1d_ldos_timing",
        lvals,
        nplot,
        [
            ("mpo_kpm", times_tb),
            ("sparse_kpm", times_sparse),
            ("exact_diag", times_ed),
        ],
        "seconds",
    )

    mem_tb, mem_sparse, mem_ed = run_memory_benchmark(lvals, t_hop)

    memory_csv = write_rows_csv(
        joinpath(output_dir, "chain_1d_memory.csv"),
        ["L", "N", "mpo_kpm_bytes", "sparse_kpm_bytes", "exact_diag_bytes"],
        ((lval, Int64(2)^lval, mem_tb[idx], mem_sparse[idx], mem_ed[idx])
         for (idx, lval) in enumerate(lvals)),
    )

    memory_fit_csv, memory_fit_params_csv = export_fit_csvs(
        output_dir,
        "chain_1d_memory",
        lvals,
        nplot,
        [
            ("mpo_kpm", mem_tb),
            ("sparse_kpm", mem_sparse),
            ("exact_diag", mem_ed),
        ],
        "bytes",
    )

    println("Saved timing data CSV to $timing_csv")
    println("Saved timing fit CSV to $timing_fit_csv")
    println("Saved timing fit parameter CSV to $timing_fit_params_csv")
    println("Saved memory data CSV to $memory_csv")
    println("Saved memory fit CSV to $memory_fit_csv")
    println("Saved memory fit parameter CSV to $memory_fit_params_csv")
end

main()
