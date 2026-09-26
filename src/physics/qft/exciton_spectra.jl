# exciton_spectra.jl — Exciton momentum-space spectra (MPS-KPM)
#
# Contains get_exciton_bands (coherent pair probes at total momentum Q) and
# get_exciton_continuum (stochastic trace over |k, Q-k>).  Both run on an MPO
# already conjugated by conjugate_by_qft_exciton (conjugation.jl).  Moved
# verbatim from the end of section 4 of the former physics/QFT_tk.jl; the overview and
# file map of physics/qft/ are at the top of bands.jl.

# ============================================================
# Exciton momentum-space spectra (MPS-KPM)
# ============================================================

"""
    get_exciton_bands(H_QFT, H, Ncheb, omega_phys_vals; Q_list, Q_groups,
                      num_q, num_avg, q_start, q_end, kernel,
                      lambda, eta, m_order, maxdim, cutoff,
                      verbose, printinfo) -> Matrix{Float64}

CPU momentum-space exciton bands from an already-QFT-conjugated exciton MPO.
This is the MPS-KPM analogue of `get_exciton_ldos_spatial`: for each total
momentum label `Q` it runs an online MPS Chebyshev recursion from
`mpsexcitonQ(Q, H.sites) = (1 / sqrt(N)) * sum_k |k, -k + Q>` and accumulates
all requested energies in one pass. The exciton must not use the MPO-KPM
`get_bands` pipeline.

`H_QFT` supplies the MPO used in the recurrence. The original real-space
`H::TBHamiltonian` supplies metadata (`sites`, `N`, `L`, `scale`, `center`) and
sets the basis convention for `mpsexcitonQ`.

Rows are energies, columns are total momenta/groups. `Q_list` selects momenta
directly. `Q_groups` (or alias `q_groups`) averages several momentum probes into
one output column. `K_list`, `K_groups`, `k_groups`, `num_k`, `k_start`, and
`k_end` are accepted as backward-compatible aliases.
"""
function get_exciton_bands(H_QFT::MPO, H::TBHamiltonian, Ncheb::Int, omega_phys_vals;
                           Q_list           = nothing,
                           Q_groups         = nothing,
                           q_groups         = nothing,
                           K_list           = nothing,
                           K_groups         = nothing,
                           k_groups         = nothing,
                           num_q            = nothing,
                           num_k::Int       = H.N,
                           num_avg::Int     = 1,
                           q_start          = nothing,
                           q_end            = nothing,
                           k_start::Int     = 1,
                           k_end::Int       = H.N,
                           kernel::Symbol   = :jackson,
                           lambda::Real     = 4.0,
                           eta::Real        = 0.0,
                           m_order::Int     = 4,
                           maxdim::Int      = 100,
                           cutoff::Real     = 1e-8,
                           verbose::Bool    = false,
                           printinfo::Bool  = false)
    _require_binary_position_space(H, "get_exciton_bands")
    _ensure_scale!(H)
    length(H.sites) == 2 * H.L ||
        error("get_exciton_bands: H is not an exciton Hamiltonian (expected length(H.sites) == 2*H.L).")
    length(H_QFT) == length(H.sites) ||
        error("get_exciton_bands: H_QFT must live on the same number of sites as H.sites.")

    list_count = count(!isnothing, (Q_list, K_list))
    group_count = count(!isnothing, (Q_groups, q_groups, K_groups, k_groups))
    list_count <= 1 ||
        error("get_exciton_bands: pass only one of Q_list or K_list.")
    group_count <= 1 ||
        error("get_exciton_bands: pass only one of Q_groups, q_groups, K_groups, or k_groups.")
    list_count == 1 && group_count == 1 &&
        error("get_exciton_bands: pass either a momentum list or grouped momenta, not both.")

    list_arg = Q_list !== nothing ? Q_list : K_list
    group_arg = Q_groups !== nothing ? Q_groups :
                q_groups !== nothing ? q_groups :
                K_groups !== nothing ? K_groups : k_groups
    num_q_eff = num_q === nothing ? num_k : Int(num_q)
    q_start_eff = q_start === nothing ? k_start : Int(q_start)
    q_end_eff = q_end === nothing ? k_end : Int(q_end)

    groups = if group_arg !== nothing
        spatial_sampling_plan(H.L; x_groups=group_arg).groups
    elseif list_arg !== nothing
        [[Int(q)] for q in list_arg]
    else
        num_q_eff > 0 || error("get_exciton_bands: num_q must be positive.")
        num_avg > 0 || error("get_exciton_bands: num_avg must be positive.")
        1 <= q_start_eff <= q_end_eff <= H.N ||
            error("get_exciton_bands: expected 1 <= q_start <= q_end <= H.N.")
        window = q_end_eff - q_start_eff + 1
        num_q_eff <= window ||
            error("get_exciton_bands: num_q=$num_q_eff exceeds sampling window length $window.")
        # 1D point layout of the shared planner (core/Utils.jl): stride
        # window ÷ num_q with num_avg sub-probes per coarse cell.
        spatial_sampling_plan(H.L; num_x=num_q_eff, num_avg,
                              x_start=q_start_eff, x_end=q_end_eff).groups
    end

    isempty(groups) && error("get_exciton_bands: no momentum groups were selected.")
    for grp in groups
        isempty(grp) && error("get_exciton_bands: empty momentum group.")
        all(q -> 1 <= q <= H.N, grp) ||
            error("get_exciton_bands: all momenta must lie in 1:H.N.")
    end

    I_mpo = MPO(H.sites, "Id")
    Ham_n = (1 / H.scale) * +(H_QFT, (-H.center) * I_mpo; cutoff=cutoff)

    omega_vals = (collect(omega_phys_vals) .- H.center) ./ H.scale
    Nomega     = length(omega_vals)
    W, denom   = _dos_weight_matrix(Ncheb, omega_vals;
                                    kernel=kernel, lambda=lambda,
                                    eta=eta, m_order=m_order)
    valid      = [abs(omega) < 1.0 for omega in omega_vals]

    nQ     = length(groups)
    Qs     = first.(groups)
    result = zeros(Float64, Nomega, nQ)

    for (j, group) in enumerate(groups)
        last_linkdim = 0
        accum_group  = zeros(Float64, Nomega)

        for Q in group
            psi0 = mpsexcitonQ(Q, H.sites)
            last_linkdim = _run_kpm_mps!(Ham_n, psi0, Ncheb, W, valid, accum_group;
                                         weight=1.0 / length(group),
                                         cutoff=cutoff, maxdim=maxdim)
        end

        for iomega in 1:Nomega
            valid[iomega] || continue
            result[iomega, j] = accum_group[iomega] / denom[iomega]
        end

        (verbose || printinfo) && (j % 5 == 0 || j == nQ) &&
            println("  exciton bands $j/$nQ (Q=$(Qs[j]), n_avg=$(length(group)))  maxlinkdim=$last_linkdim")
    end

    return result
end

"""
    get_exciton_continuum(H_QFT, H, Ncheb, omega_phys_vals;
                          Q_list, num_q, q_start, q_end,
                          N_sample, k_list, seed, normalize,
                          kernel, lambda, eta, m_order, maxdim, cutoff,
                          verbose, printinfo) -> Matrix{Float64}

Stochastic MPS-KPM trace over the electron-hole continuum at fixed total
momentum. For each selected total momentum `Q`, this estimates

    A_cont(Q, omega) = (1 / N) * sum_k <k, Q-k| delta(omega - H_QFT) |k, Q-k>

with random-phase trace probes built by `mpsexcitonQTrace(Q, H.sites)`. This is
an incoherent trace over relative momentum, unlike `get_exciton_bands`, which
probes the coherent pair state `mpsexcitonQ(Q, H.sites)`.

`H_QFT` supplies the already-QFT-conjugated MPO used in the MPS recursion. The
original `H::TBHamiltonian` supplies metadata (`sites`, `N`, `L`, `scale`,
`center`) and the exciton site convention. The exciton continuum remains an
MPS-KPM calculation: no exciton MPO-KPM / `get_bands` path is used.

Rows are energies, columns are total momenta. If `k_list` is provided, those
1-indexed relative momenta are used deterministically for every `Q`; otherwise
`N_sample` compact random-phase trace probes are drawn for each `Q`. Each trace
probe is a randomized superposition over all `|k,Q-k>` states, so this avoids a
full KPM recursion for every explicit relative momentum.

With `normalize=true` (default), the result estimates the per-relative-momentum
average `(1/N) * Tr_Q`. With `normalize=false`, it is multiplied by `H.N` and
estimates the total fixed-`Q` trace `Tr_Q`.
"""
function get_exciton_continuum(H_QFT::MPO, H::TBHamiltonian, Ncheb::Int, omega_phys_vals;
                               Q_list::Union{Nothing,AbstractVector} = nothing,
                               num_q::Int       = H.N,
                               q_start::Int     = 1,
                               q_end::Int       = H.N,
                               N_sample::Int    = 4,
                               k_list           = nothing,
                               seed::Union{Int,Nothing} = 42,
                               normalize::Bool  = true,
                               kernel::Symbol   = :jackson,
                               lambda::Real     = 4.0,
                               eta::Real        = 0.0,
                               m_order::Int     = 4,
                               maxdim::Int      = 100,
                               cutoff::Real     = 1e-8,
                               verbose::Bool    = false,
                               printinfo::Bool  = false)
    _require_binary_position_space(H, "get_exciton_continuum")
    _ensure_scale!(H)
    length(H.sites) == 2 * H.L ||
        error("get_exciton_continuum: H is not an exciton Hamiltonian (expected length(H.sites) == 2*H.L).")
    length(H_QFT) == length(H.sites) ||
        error("get_exciton_continuum: H_QFT must live on the same number of sites as H.sites.")

    Qs = if Q_list !== nothing
        collect(Int, Q_list)
    else
        num_q > 0 || error("get_exciton_continuum: num_q must be positive.")
        1 <= q_start <= q_end <= H.N ||
            error("get_exciton_continuum: expected 1 <= q_start <= q_end <= H.N.")
        window = q_end - q_start + 1
        num_q <= window ||
            error("get_exciton_continuum: num_q=$num_q exceeds sampling window length $window.")
        unique(round.(Int, range(q_start, q_end; length=num_q)))
    end
    isempty(Qs) && error("get_exciton_continuum: no total momenta were selected.")
    all(Q -> 1 <= Q <= H.N, Qs) ||
        error("get_exciton_continuum: all total momenta must lie in 1:H.N.")

    k_samples_fixed = if k_list === nothing
        nothing
    else
        ks = collect(Int, k_list)
        isempty(ks) && error("get_exciton_continuum: k_list must not be empty.")
        all(k -> 1 <= k <= H.N, ks) ||
            error("get_exciton_continuum: all relative momenta in k_list must lie in 1:H.N.")
        ks
    end
    if k_list === nothing && N_sample <= 0
        error("get_exciton_continuum: N_sample must be positive when k_list is not provided.")
    end

    I_mpo = MPO(H.sites, "Id")
    Ham_n = (1 / H.scale) * +(H_QFT, (-H.center) * I_mpo; cutoff=cutoff)

    omega_vals = (collect(omega_phys_vals) .- H.center) ./ H.scale
    Nomega     = length(omega_vals)
    W, denom   = _dos_weight_matrix(Ncheb, omega_vals;
                                    kernel=kernel, lambda=lambda,
                                    eta=eta, m_order=m_order)
    valid      = [abs(omega) < 1.0 for omega in omega_vals]

    rng    = seed === nothing ? Random.default_rng() : Random.MersenneTwister(seed)
    result = zeros(Float64, Nomega, length(Qs))

    for (j, Q) in enumerate(Qs)
        accum_Q      = zeros(Float64, Nomega)
        last_linkdim = 0
        n_used       = 0

        if k_samples_fixed === nothing
            weight = 1.0 / N_sample
            n_used = N_sample
            for isample in 1:N_sample
                psi0 = mpsexcitonQTrace(Q, H.sites; rng=rng)
                last_linkdim = _run_kpm_mps!(Ham_n, psi0, Ncheb, W, valid, accum_Q;
                                             weight=weight, cutoff=cutoff, maxdim=maxdim)
                verbose && (isample % 5 == 0 || isample == N_sample) &&
                    println("  exciton continuum Q=$Q trace probe $isample/$N_sample  maxlinkdim=$last_linkdim")
            end
        else
            ks     = k_samples_fixed
            weight = 1.0 / length(ks)
            n_used = length(ks)
            for (isample, k) in enumerate(ks)
                psi0 = mpsexcitonKQ(k, Q, H.sites)
                last_linkdim = _run_kpm_mps!(Ham_n, psi0, Ncheb, W, valid, accum_Q;
                                             weight=weight, cutoff=cutoff, maxdim=maxdim)
                verbose && (isample % 15 == 0 || isample == length(ks)) &&
                    println("  exciton continuum Q=$Q basis sample $isample/$(length(ks)) (k=$k)  maxlinkdim=$last_linkdim")
            end
        end

        for iomega in 1:Nomega
            valid[iomega] || continue
            result[iomega, j] = accum_Q[iomega] / denom[iomega]
        end
        if !normalize
            result[:, j] .*= H.N
        end

        (verbose || printinfo) && (j % 5 == 0 || j == length(Qs)) &&
            println("  exciton continuum $j/$(length(Qs)) (Q=$Q, n_probe=$n_used)  maxlinkdim=$last_linkdim")
    end

    return result
end
