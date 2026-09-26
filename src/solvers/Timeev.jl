# solvers/Timeev.jl — time evolution: TDVP, propagator MPOs, density-matrix RK4
#
# Contents: the kernels every routine below is built on, shared with
# gpu/timeev.jl (one TDVP step _tdvp_step, the trajectory loop _trajectory, one
# RK4 step _rk4_step); the short-time propagator U(dt) = e^{-iH dt} as an MPO,
# sampled column by column with TDVP and compressed by TCI
# (build_tdvp_propagator_mpo); state evolution by TDVP or by repeated application
# of U (tdvp_evolve, apply_mpo_to_mps, evolve_with_propagator, evolve_with_tdvp,
# evolve_with_tdvp_timedep); basis-overlap diagnostics (compute_basis_overlaps,
# basis_amplitude, phase_aligned_distance, check_tdvp_vs_U_mpo); RK4 for a
# density-matrix MPO under dρ/dt = −i[H(t), ρ] and under the non-Hermitian
# −i(Hρ − ρH†) (rk4_step_dm_timedep, evolve_rk4_dm_timedep, rk4_step_dm_nh,
# evolve_rk4_dm_nh); and observables along a density-matrix trajectory
# (dm_expect, observables_trajectory, timedep_observable_trajectory, purity,
# bond_current_x, central_x_bond, …).
#
# build_tdvp_propagator_mpo, tdvp_evolve, evolve_with_tdvp and check_tdvp_vs_U_mpo
# take an MPO already multiplied by −im; their TBHamiltonian methods (section 8)
# apply it. evolve_with_tdvp_timedep and the RK4 functions take the physical H(t).
#
# Entry points: build_tdvp_propagator_mpo, tdvp_evolve, evolve_with_tdvp,
#   evolve_with_tdvp_timedep, evolve_with_propagator, evolve_rk4_dm_timedep,
#   evolve_rk4_dm_nh, observables_trajectory, bond_current_x_trajectory
# Depends on: core/Utils.jl (binary_to_MPS, matrix_checker), core/Hamiltonian.jl
#   (hopping2MPO), core/TBSystem.jl (TBHamiltonian).

using ITensors
using ITensorMPS

# ============================================================
# 1. Shared kernels: TDVP step, trajectory loop, RK4 step
# ============================================================

# One TDVP step of size `dt` under the generator `H` (−im·H for Schrödinger
# evolution): every tdvp call in this file and in gpu/timeev.jl goes through here.
# With `normalize=true`, tdvp ends each half-sweep with `normalize!` of the state,
# so the result needs no second normalisation. Extra keywords (`updater_kwargs`)
# go to tdvp.
function _tdvp_step(H, psi, dt; nsite, maxdim, cutoff, normalize, reverse_step,
                    outputlevel, kwargs...)
    return tdvp(
        H,
        dt,
        psi;
        time_step = dt,
        nsite = nsite,
        maxdim = maxdim,
        cutoff = cutoff,
        normalize = normalize,
        reverse_step = reverse_step,
        outputlevel = outputlevel,
        kwargs...,
    )
end

# The trajectory loop of every evolve_* function: returns [x0, x1, …, x_nsteps]
# as a Vector{T}, with x_step = advance(step, x_{step-1}). `snapshot` makes the
# stored copies and the working copy of x0: `copy` for the MPS loops, `deepcopy`
# for the density-matrix loops.
function _trajectory(advance, x0, nsteps::Integer, ::Type{T}; snapshot = copy) where {T}
    states = Vector{T}(undef, nsteps + 1)
    states[1] = snapshot(x0)

    x = snapshot(x0)
    for step in 1:nsteps
        x = advance(step, x)
        states[step + 1] = snapshot(x)
    end

    return states
end

# One classical RK4 step of dρ/dt = f(ρ) for an MPO ρ, shared by
# rk4_step_dm_timedep, rk4_step_dm_nh and rk4_step_dm_nh_gpu (gpu/timeev.jl).
# `rhs(stage, ρ)` returns f at stage 1–4 (stages 2 and 3 are the midpoint).
# `coeffs = (dt/2, dt, dt/6, 2)` are the tableau factors, passed in so that the GPU
# step keeps its ComplexF32 constants. Every MPO sum truncates with `add_kwargs`
# (the CPU steps pass `cutoff` only, the GPU step `cutoff` and `maxdim`); the
# intermediate states (if `truncate_intermediates`), the k-sum and the result are
# then truncated to `maxdim`, `cutoff`.
function _rk4_step(rhs, ρ::MPO, coeffs; maxdim, cutoff, truncate_intermediates,
                   add_kwargs = (; cutoff = cutoff))
    halfdt, fulldt, sixthdt, two = coeffs

    k1 = rhs(1, ρ)

    ρ2 = +(ρ, halfdt * k1; add_kwargs...)
    truncate_intermediates && ITensorMPS.truncate!(ρ2; maxdim=maxdim, cutoff=cutoff)
    k2 = rhs(2, ρ2)

    ρ3 = +(ρ, halfdt * k2; add_kwargs...)
    truncate_intermediates && ITensorMPS.truncate!(ρ3; maxdim=maxdim, cutoff=cutoff)
    k3 = rhs(3, ρ3)

    ρ4 = +(ρ, fulldt * k3; add_kwargs...)
    truncate_intermediates && ITensorMPS.truncate!(ρ4; maxdim=maxdim, cutoff=cutoff)
    k4 = rhs(4, ρ4)

    k_sum = +(k1, two * k2; add_kwargs...)
    k_sum = +(k_sum, two * k3; add_kwargs...)
    k_sum = +(k_sum, k4; add_kwargs...)
    ITensorMPS.truncate!(k_sum; maxdim=maxdim, cutoff=cutoff)

    ρ_new = +(ρ, sixthdt * k_sum; add_kwargs...)
    ITensorMPS.truncate!(ρ_new; maxdim=maxdim, cutoff=cutoff)

    return ρ_new
end


# ============================================================
# 2. TDVP propagator MPO
# ============================================================

"""
    build_tdvp_propagator_mpo(H, dt, L, sites; maxdim=50, cutoff=1e-8,
                              reverse_step=true, outputlevel=0, nsite=2,
                              cross_tol=1e-8, initial_positions=[],
                              use_diagonal_pivots=false, expand_basis=true,
                              cache_columns=true, interpolation_type=ComplexF64)
        -> MPO

Build an MPO approximation of the short-time propagator `U(dt) = e^{-iH dt}` by
sampling matrix elements `⟨i|U(dt)|j⟩` via TDVP and compressing with TCI.

`H` must already be multiplied by `-im` for Schrödinger evolution.
TDVP runs once per column `j` that TCI samples: `U|j⟩` is kept, and every other element
of that column is an overlap with it (`cache_columns`).  The diagonal is dominant for
small `dt`; TCI starts from `(1, 1)` and QuanticsTCI moves its random initial pivots to
large elements, so it lands on the diagonal without seeding.

Each sample evolves a basis state `|j⟩`, an MPS of bond dimension 1.  TDVP cannot leave
the tangent space of that state, so on its own it drops every hop that flips three or
more qubits (the carry chains `0111 → 1000` of the quantics encoding, such as the middle
bond of a chain).  By default `|j⟩` first gets the Krylov basis of `H|j⟩, H²|j⟩`
(`ITensorMPS.expand(...; alg="global_krylov")`).  The samples then match the dense
`exp(-iH dt)` to TDVP accuracy (chain_1d, `dt = 0.05`: Frobenius error 7e-5 at L = 3 and
1.3e-4 at L = 6, limited by `cutoff`; 0.07 and 0.27 without the expansion).  The
expansion makes each TDVP run about 4x more expensive (L = 8-10: 15-16 ms against 3.5-5 ms).

!!! warning
    From about L = 5, the QTCI fit in `hopping2MPO` can miss those isolated carry-chain
    elements even though they are sampled correctly.  In 1D-chain tests it missed them
    (max error ≈ `dt`) for most random seeds at L = 6 and for every seed at L = 8.  It does
    the same on a plain hopping matrix.  For a chain, seeding `initial_positions` with the
    elements near the middle boundary,
    `[(N÷2+a, N÷2+b) for a in -3:4 for b in -3:4 if abs(a-b) <= 4]`, fixed L = 6.  From
    L = 8, the final `truncate!(cutoff=1e-8)` in `hopping2MPO` also drops the second-order
    elements (error ≈ `dt²/2`), because that cutoff is relative to ‖U‖² = N.  Check the
    result against a dense `exp(-iH dt)` at small L.

## Keyword arguments
- `maxdim`, `cutoff`    : TDVP truncation parameters. Defaults `50`, `1e-8`.
- `reverse_step`        : Evolve the bond tensor backwards between two-site updates, as the
                          TDVP projector splitting requires. Default `true` (the ITensorMPS
                          default). `false` counts terms of `H` twice, so sampled elements are
                          off at O(dt) (some hops come out 1.5x too large); it warns.
- `outputlevel`, `nsite` : passed to `tdvp`. Defaults `0`, `2`.
- `cross_tol`           : TCI interpolation tolerance. Default `1e-8`.
- `initial_positions`   : TCI pivots `(i, j)` (1-indexed) passed to `hopping2MPO`.
                          Default `[]` (none).
- `use_diagonal_pivots` : Seed TCI with all N diagonal positions `(i, i)` when
                          `initial_positions` is empty. Default `false`:
                          seeding makes TCI sample every column, so it costs N TDVP runs
                          (L = 8: 256 against 182-238 unseeded; L = 10: 1024 against 334),
                          and it does not make TCI find the off-diagonal structure more
                          reliably.
- `expand_basis`        : Expand each basis state with its Krylov vectors before TDVP (see
                          above; needs `H::MPO`). Default `true`.
- `cache_columns`       : Keep each evolved column `U|j⟩` (one MPS of a few kB per sampled
                          `j`), so TDVP runs once per column instead of once per sampled
                          element (L = 8: about 200 runs against 1500-2300).  The samples, and
                          so the MPO, are the same either way; `false` saves the memory.
                          Default `true`.
- `interpolation_type`  : Element type for TCI sampling. Default `ComplexF64`.

To reproduce the samples and seeding used before these defaults changed, pass
`reverse_step=false, expand_basis=false, use_diagonal_pivots=true`.

A `TBHamiltonian` overload applies `-im` internally:
`build_tdvp_propagator_mpo(H::TBHamiltonian, dt; ...)`.
"""
function build_tdvp_propagator_mpo(
    H, dt, L, sites;
    maxdim = 50,
    cutoff = 1e-8,
    reverse_step = true,
    outputlevel = 0,
    nsite = 2,
    cross_tol = 1e-8,
    initial_positions = [],
    use_diagonal_pivots = false,  # true seeds all N diagonal positions: TDVP on all N columns
    expand_basis = true,
    cache_columns = true,
    interpolation_type = ComplexF64,
)
    N = 2^L
    reverse_step || @warn "build_tdvp_propagator_mpo: reverse_step=false skips TDVP's backward bond evolution and counts terms of H twice; the propagator is wrong at O(dt)."

    # Opt-in: the N diagonal pivots cost O(N) TDVP runs and are not needed for TCI to
    # find the near-identity structure (see the docstring).
    if use_diagonal_pivots && isempty(initial_positions)
        initial_positions = [(i, i) for i in 1:N]
    end

    # TCI samples many rows i of each column j: evolve each |j> once and keep U|j>.
    evolved_columns = Dict{Int,MPS}()

    function evolve_column(j::Int)
        psi_j = TensorBinding.binary_to_MPS(j - 1, L, sites)
        if expand_basis
            # |j> has bond dimension 1: without the Krylov basis of H|j>, H^2|j> TDVP
            # cannot reach hops that flip three or more qubits (carry chains 0111 -> 1000).
            psi_j = ITensorMPS.expand(psi_j, H; alg = "global_krylov")
        end

        return _tdvp_step(
            H,
            psi_j,
            dt;
            nsite = nsite,
            maxdim = maxdim,
            cutoff = cutoff,
            normalize = false,   # must be false: normalization is state-dependent and breaks linearity
            reverse_step = reverse_step,
            outputlevel = outputlevel,
        )
    end

    function func(i, j)
        psi_i = TensorBinding.binary_to_MPS(Int(i - 1), L, sites)
        psi_j_evolved = cache_columns ?
            get!(() -> evolve_column(Int(j)), evolved_columns, Int(j)) :
            evolve_column(Int(j))

        return inner(psi_i, psi_j_evolved)
    end

    U_mpo = TensorBinding.hopping2MPO(
        func,
        N,
        sites;
        tol = cross_tol,
        initial_positions = initial_positions,
        type = interpolation_type,
    )

    return U_mpo
end


# ============================================================
# 3. State evolution: TDVP and propagator MPO
# ============================================================

"""
    tdvp_evolve(H, psi, dt; maxdim=200, cutoff=1e-10, normalize=true,
                reverse_step=false, outputlevel=0, nsite=2) -> MPS

Apply one TDVP step to `psi` under Hamiltonian `H` for time `dt`.

`H` must already carry the `-im` prefactor for Schrödinger evolution.
When `normalize=true` the output is normalised (`tdvp` normalises the state at the
end of each half-sweep).

A `TBHamiltonian` overload applies `-im` internally:
`tdvp_evolve(H::TBHamiltonian, psi, dt; ...)`.
"""
function tdvp_evolve(
    H,
    psi,
    dt;
    maxdim = 200,
    cutoff = 1e-10,
    normalize = true,
    reverse_step = false,
    outputlevel = 0,
    nsite = 2,
)
    return _tdvp_step(
        H,
        psi,
        dt;
        nsite = nsite,
        maxdim = maxdim,
        cutoff = cutoff,
        normalize = normalize,
        reverse_step = reverse_step,
        outputlevel = outputlevel,
    )
end


"""
    apply_mpo_to_mps(U_mpo, psi; cutoff=1e-12, maxdim=500, normalize=true) -> MPS

Apply a propagator MPO `U_mpo` to the MPS `psi` with optional truncation and
normalisation.  Used to advance a state by one time step when `U_mpo` was
prebuilt by `build_tdvp_propagator_mpo`.
"""
function apply_mpo_to_mps(U_mpo, psi; cutoff=1e-12, maxdim=500, normalize=true)
    psi_out = apply(U_mpo, psi; cutoff=cutoff, maxdim=maxdim)
    if normalize
        nrm = sqrt(real(inner(psi_out, psi_out)))
        psi_out = psi_out / nrm
    end
    return psi_out
end


"""
    evolve_with_propagator(U_mpo, psi0, nsteps; normalize_each_step=true,
                           cutoff=1e-8, maxdim=10_000) -> Vector{MPS}

Apply the fixed MPO propagator `U_mpo` repeatedly for `nsteps` steps,
returning the full trajectory `[psi(0), psi(1*dt), ..., psi(nsteps*dt)]`.

Useful when the same short-time propagator is reused at every step (time-independent H).
For efficiency the MPO is built once via `build_tdvp_propagator_mpo`; this function
then applies it `nsteps` times.
"""
function evolve_with_propagator(U_mpo, psi0, nsteps;
    normalize_each_step = true,
    cutoff = 1e-8,
    maxdim = 10_000,
)
    return _trajectory(psi0, nsteps, MPS) do _, psi
        psi = apply(U_mpo, psi; cutoff = cutoff, maxdim = maxdim)
        ITensorMPS.truncate!(psi; cutoff = cutoff, maxdim = maxdim)
        if normalize_each_step
            normalize!(psi)
        end
        return psi
    end
end


"""
    evolve_with_tdvp(H, psi0, nsteps, dt; normalize_each_step=true, maxdim=200,
                     cutoff=1e-10, reverse_step=false, outputlevel=0, nsite=2)
        -> Vector{MPS}

Run a TDVP loop for `nsteps` steps of size `dt` under a fixed Hamiltonian `H`,
returning `[psi(0), psi(dt), ..., psi(nsteps*dt)]`.

`H` must carry the `-im` prefactor for Schrödinger evolution.

A `TBHamiltonian` overload applies `-im` internally:
`evolve_with_tdvp(H::TBHamiltonian, psi0, nsteps, dt; ...)`.
"""
function evolve_with_tdvp(H, psi0, nsteps, dt;
    normalize_each_step = true,
    maxdim = 200,
    cutoff = 1e-10,
    reverse_step = false,
    outputlevel = 0,
    nsite = 2,
)
    return _trajectory(psi0, nsteps, MPS) do _, psi
        return _tdvp_step(
            H,
            psi,
            dt;
            nsite = nsite,
            maxdim = maxdim,
            cutoff = cutoff,
            normalize = normalize_each_step,
            reverse_step = reverse_step,
            outputlevel = outputlevel,
        )
    end
end


"""
    evolve_with_tdvp_timedep(Hoft, psi0, nsteps, dt; normalize_each_step=true,
                             maxdim=200, cutoff=1e-10, reverse_step=false,
                             outputlevel=0, nsite=2, krylovdim=20, tol=1e-10)
        -> Vector{MPS}

TDVP loop for a time-dependent Hamiltonian `H(t)`.

`Hoft` is a callable `t::Float64 -> MPO`.  On each interval `[t, t+dt]` the
Hamiltonian is frozen at the midpoint `t + dt/2` (midpoint rule).  `Hoft` must
return the physical Hamiltonian; the `-im` prefactor is applied internally.
`krylovdim` and `tol` go to the TDVP exponentiation step (`updater_kwargs`);
the other keywords are as in `evolve_with_tdvp`.

Returns `[psi(0), psi(dt), ..., psi(nsteps*dt)]`.
"""
function evolve_with_tdvp_timedep(Hoft, psi0, nsteps, dt;
    normalize_each_step = true,
    maxdim = 200,
    cutoff = 1e-10,
    reverse_step = false,
    outputlevel = 0,
    nsite = 2,
    krylovdim = 20,
    tol = 1e-10,
)
    return _trajectory(psi0, nsteps, MPS) do step, psi
        t_mid = (step - 1) * dt + dt / 2
        Hmid = Hoft(t_mid)

        return _tdvp_step(
            -im * Hmid,
            psi,
            dt;
            nsite = nsite,
            maxdim = maxdim,
            cutoff = cutoff,
            normalize = normalize_each_step,
            reverse_step = reverse_step,
            outputlevel = outputlevel,
            updater_kwargs = (; tol = tol, krylovdim = krylovdim, eager = true),
        )
    end
end


# ============================================================
# 4. Basis overlaps and TDVP-vs-propagator checks
# ============================================================

"""
    compute_basis_overlaps(states, L, sites)
        -> NamedTuple(overlaps, abs_overlaps, probabilities, norms)

For each MPS in `states`, compute overlaps with all `2^L` computational basis states.

Returns a named tuple with fields:
- `overlaps`      : `(nsteps+1) × 2^L` matrix of `ComplexF64` amplitudes `⟨j|ψ(t)⟩`
- `abs_overlaps`  : element-wise absolute values
- `probabilities` : `|⟨j|ψ(t)⟩|²`
- `norms`         : `⟨ψ(t)|ψ(t)⟩` at each step
"""
function compute_basis_overlaps(states, L, sites)
    nsteps_plus_1 = length(states)
    nbasis = 2^L

    overlaps = Matrix{ComplexF64}(undef, nsteps_plus_1, nbasis)
    norms = zeros(Float64, nsteps_plus_1)

    basis_states = [TensorBinding.binary_to_MPS(i - 1, L, sites) for i in 1:nbasis]

    for step in 1:nsteps_plus_1
        psi = states[step]
        norms[step] = real(inner(psi, psi))
        for i in 1:nbasis
            overlaps[step, i] = inner(basis_states[i], psi)
        end
    end

    return (
        overlaps     = overlaps,
        abs_overlaps = abs.(overlaps),
        probabilities = abs2.(overlaps),
        norms        = norms,
    )
end


"""
    basis_amplitude(psi, n, L, sites) -> ComplexF64

Return the amplitude `⟨n|ψ⟩` where `|n⟩` is the `n`-th computational basis state
(0-indexed big-endian quantics encoding over `L` qubit `sites`).
"""
function basis_amplitude(psi, n, L, sites)
    phi = TensorBinding.binary_to_MPS(n, L, sites)
    return inner(phi, psi)
end


"""
    phase_aligned_distance(psi_a, psi_b) -> Float64

Phase-insensitive distance between two (unnormalised) MPS states:

    d = min_{φ} ‖â − e^{iφ} b̂‖ = √(2 − 2|⟨â|b̂⟩|)

where `â = psi_a/‖psi_a‖`.  Returns `Inf` when either state has zero norm.
Useful for comparing TDVP and MPO propagator trajectories independent of
any global phase accumulated during time evolution.
"""
function phase_aligned_distance(psi_a, psi_b)
    na2 = real(inner(psi_a, psi_a))
    nb2 = real(inner(psi_b, psi_b))

    if na2 <= 0 || nb2 <= 0
        return Inf
    end

    ov = inner(psi_a, psi_b) / (sqrt(na2) * sqrt(nb2))
    return sqrt(max(0.0, 2.0 - 2.0 * abs(ov)))
end


"""
    check_tdvp_vs_U_mpo(H, U_mpo, dt, L, sites;
                        test_states=[0, 1, 3, 7, 13, 29, 57, 2^L - 1],
                        tdvp_maxdim=200, tdvp_cutoff=1e-10, tdvp_normalize=true,
                        tdvp_reverse_step=false, tdvp_outputlevel=0, tdvp_nsite=2,
                        apply_maxdim=500, apply_cutoff=1e-12,
                        print_sample_amplitudes=true, sample_amplitudes=[0, 1, 2, 3])
        -> (max_overlap_error, max_phase_error)
    check_tdvp_vs_U_mpo(H::TBHamiltonian, U_mpo::MPO, dt; kwargs...)

Validate that `U_mpo` agrees with direct TDVP on a set of computational basis states.
Prints per-state overlap errors and phase-aligned distances, then returns the maxima.

## Keyword arguments
- `test_states`             : 0-indexed basis states `|n⟩` to evolve both ways.
- `tdvp_*`                  : `maxdim`, `cutoff`, `normalize`, `reverse_step`,
                              `outputlevel`, `nsite` of the reference `tdvp_evolve`
                              step (`tdvp_normalize` also normalises the `U_mpo` result).
- `apply_maxdim`, `apply_cutoff` : truncation of `apply_mpo_to_mps`.
- `print_sample_amplitudes` : also print `⟨m|ψ⟩` of both results for each `m` in
                              `sample_amplitudes`.

The reference is one TDVP step from the bare basis state, so it has the errors described
in `build_tdvp_propagator_mpo`: it drops the hops that flip three or more qubits, and
with the default `tdvp_reverse_step=false` it also counts terms of `H` twice.  An error
of order `dt` therefore does not mean `U_mpo` is wrong (chain_1d, L = 4, `dt = 0.05`: a
phase error of 0.056 for a `U_mpo` within 8e-5 of `exp(-iH dt)`).  At small L, compare
with a dense `exp(-iH dt)` instead.

The `TBHamiltonian` method applies `-im` internally and takes `L`, `sites` from `H`.
"""
function check_tdvp_vs_U_mpo(
    H,
    U_mpo,
    dt,
    L,
    sites;
    test_states = [0, 1, 3, 7, 13, 29, 57, 2^L - 1],
    tdvp_maxdim = 200,
    tdvp_cutoff = 1e-10,
    tdvp_normalize = true,
    tdvp_reverse_step = false,
    tdvp_outputlevel = 0,
    tdvp_nsite = 2,
    apply_maxdim = 500,
    apply_cutoff = 1e-12,
    print_sample_amplitudes = true,
    sample_amplitudes = [0, 1, 2, 3],
)
    println("Checking TDVP evolution against applying U_mpo")
    println("L = $L, dt = $dt")
    println()

    max_overlap_error = 0.0
    max_phase_error = 0.0

    for n in test_states
        println("Input basis state n = $n")

        psi0 = TensorBinding.binary_to_MPS(n, L, sites)

        psi_tdvp = tdvp_evolve(
            H, psi0, dt;
            maxdim = tdvp_maxdim,
            cutoff = tdvp_cutoff,
            normalize = tdvp_normalize,
            reverse_step = tdvp_reverse_step,
            outputlevel = tdvp_outputlevel,
            nsite = tdvp_nsite,
        )

        psi_mpo = apply_mpo_to_mps(
            U_mpo, psi0;
            cutoff = apply_cutoff,
            maxdim = apply_maxdim,
            normalize = tdvp_normalize,
        )

        n_tdvp = sqrt(real(inner(psi_tdvp, psi_tdvp)))
        n_mpo  = sqrt(real(inner(psi_mpo,  psi_mpo)))
        ov_norm = inner(psi_tdvp, psi_mpo) / (n_tdvp * n_mpo)

        overlap_error = abs(1 - abs(ov_norm))
        phase_error   = phase_aligned_distance(psi_tdvp, psi_mpo)

        max_overlap_error = max(max_overlap_error, overlap_error)
        max_phase_error   = max(max_phase_error,   phase_error)

        println("  ||psi_tdvp||          = ", n_tdvp)
        println("  ||psi_mpo||           = ", n_mpo)
        println("  normalized overlap    = ", ov_norm)
        println("  1 - |overlap|         = ", overlap_error)
        println("  phase-aligned error   = ", phase_error)

        if print_sample_amplitudes
            println("  Sample output amplitudes:")
            for m in sample_amplitudes
                a_tdvp = basis_amplitude(psi_tdvp, m, L, sites)
                a_mpo  = basis_amplitude(psi_mpo,  m, L, sites)
                println("    <$(m)|psi_tdvp> = ", a_tdvp,
                        "    <$(m)|psi_mpo> = ", a_mpo,
                        "    diff = ", abs(a_tdvp - a_mpo))
            end
        end

        println()
    end

    println("Summary")
    println("  max over tests of 1 - |overlap|       = ", max_overlap_error)
    println("  max over tests of phase-aligned error = ", max_phase_error)

    return max_overlap_error, max_phase_error
end


# ============================================================
# 5. Density-matrix RK4 (Hermitian H(t))
# ============================================================

# dρ/dt = -i(Hρ - ρ H_right): H_right = H gives the commutator of a Hermitian H,
# H_right = H† the non-Hermitian right-hand side (section 6).
function _von_neumann_rhs(H::MPO, H_right::MPO, ρ::MPO; maxdim::Int, cutoff::Float64)
    Hρ   = apply(H, ρ; maxdim=maxdim, cutoff=cutoff)
    ρH   = apply(ρ, H_right; maxdim=maxdim, cutoff=cutoff)
    comm = +(Hρ, -1.0 * ρH; cutoff=cutoff)
    ITensorMPS.truncate!(comm; maxdim=maxdim, cutoff=cutoff)
    return -1.0im * comm
end

# dρ/dt = -i[H, ρ] RHS for Hermitian H
_von_neumann_rhs(H::MPO, ρ::MPO; maxdim::Int, cutoff::Float64) =
    _von_neumann_rhs(H, H, ρ; maxdim=maxdim, cutoff=cutoff)

# One RK4 step of dρ/dt = rhs(H(t), ρ), shared by rk4_step_dm_timedep and
# rk4_step_dm_nh: `Hoft` is evaluated once each at t, t + dt/2 and t + dt, in that
# order, and the midpoint H serves stages 2 and 3.
function _rk4_step_dm(rhs, Hoft, ρ::MPO, t::Float64, dt::Float64;
    maxdim::Int, cutoff::Float64, truncate_intermediates::Bool,
)
    H0   = Hoft(t)
    Hmid = Hoft(t + dt / 2)
    H1   = Hoft(t + dt)
    Hstage = (H0, Hmid, Hmid, H1)

    return _rk4_step((stage, x) -> rhs(Hstage[stage], x; maxdim=maxdim, cutoff=cutoff),
                     ρ, (dt / 2, dt, dt / 6, 2.0);
                     maxdim=maxdim, cutoff=cutoff,
                     truncate_intermediates=truncate_intermediates)
end

# The trajectory loop of evolve_rk4_dm_timedep and evolve_rk4_dm_nh: step k
# advances ρ from t = (k - 1)·dt with `rk4_step`; `label` opens the verbose line.
function _evolve_rk4_dm(rk4_step, label, Hoft, ρ0::MPO, nsteps::Int, dt::Float64;
    maxdim::Int, cutoff::Float64, truncate_intermediates::Bool, verbose::Bool,
)
    return _trajectory(ρ0, nsteps, MPO; snapshot = deepcopy) do step, ρ
        t = (step - 1) * dt
        verbose && println("$label step $step / $nsteps,  t = $t,  maxlinkdim = $(ITensorMPS.maxlinkdim(ρ))")
        return rk4_step(Hoft, ρ, t, dt;
            maxdim=maxdim,
            cutoff=cutoff,
            truncate_intermediates=truncate_intermediates,
        )
    end
end


"""
    rk4_step_dm_timedep(Hoft, ρ::MPO, t::Float64, dt::Float64; maxdim=200,
                        cutoff=1e-10, truncate_intermediates=true) -> MPO

Single RK4 step for `dρ/dt = -i[H(t), ρ]` with a time-dependent Hamiltonian MPO.
`H` is evaluated at `t`, `t+dt/2`, and `t+dt` per the classical RK4 tableau.
"""
function rk4_step_dm_timedep(Hoft, ρ::MPO, t::Float64, dt::Float64;
    maxdim::Int     = 200,
    cutoff::Float64 = 1e-10,
    truncate_intermediates::Bool = true,
)
    return _rk4_step_dm(_von_neumann_rhs, Hoft, ρ, t, dt;
        maxdim=maxdim,
        cutoff=cutoff,
        truncate_intermediates=truncate_intermediates,
    )
end


"""
    evolve_rk4_dm_timedep(Hoft, ρ0::MPO, nsteps::Int, dt::Float64; maxdim=200,
                          cutoff=1e-10, truncate_intermediates=true, verbose=false)
        -> Vector{MPO}

Evolve a density-matrix MPO `ρ0` under `dρ/dt = -i[H(t), ρ]` for `nsteps` steps
of size `dt` using RK4.

`Hoft` is a callable `t::Float64 -> MPO` returning the physical Hamiltonian at time `t`.
Returns the full trajectory `[ρ(0), ρ(dt), ..., ρ(nsteps*dt)]` as a `Vector{MPO}`.

## Keyword arguments
- `maxdim`, `cutoff`          : Truncation for intermediate MPO sums and products.
                                Defaults `200`, `1e-10`.
- `truncate_intermediates`    : Truncate after each RK4 sub-step to control bond growth.
                                Default `true`.
- `verbose`                   : Print step/bond-dim progress. Default `false`.
"""
function evolve_rk4_dm_timedep(Hoft, ρ0::MPO, nsteps::Int, dt::Float64;
    maxdim::Int     = 200,
    cutoff::Float64 = 1e-10,
    truncate_intermediates::Bool = true,
    verbose::Bool   = false,
)
    return _evolve_rk4_dm(rk4_step_dm_timedep, "RK4", Hoft, ρ0, nsteps, dt;
        maxdim=maxdim,
        cutoff=cutoff,
        truncate_intermediates=truncate_intermediates,
        verbose=verbose,
    )
end


# ============================================================
# 6. Density-matrix RK4 (non-Hermitian H(t))
# ============================================================

# dρ/dt = -i(Hρ - ρH†) RHS for non-Hermitian H.
# H† is formed by swapping prime levels and conjugating: conj(swapprime(H, 0, 1)).
_nh_von_neumann_rhs(H::MPO, ρ::MPO; maxdim::Int, cutoff::Float64) =
    _von_neumann_rhs(H, conj(swapprime(H, 0, 1)), ρ; maxdim=maxdim, cutoff=cutoff)


"""
    rk4_step_dm_nh(Hoft, ρ::MPO, t::Float64, dt::Float64; maxdim=200, cutoff=1e-10,
                   truncate_intermediates=true) -> MPO

Single RK4 step for the non-Hermitian von Neumann equation

    dρ/dt = -i(H(t) ρ − ρ H(t)†)

`Hoft` is a callable `t -> MPO`; pass `(_ -> H.mpo)` for a static NH Hamiltonian.
"""
function rk4_step_dm_nh(Hoft, ρ::MPO, t::Float64, dt::Float64;
    maxdim::Int     = 200,
    cutoff::Float64 = 1e-10,
    truncate_intermediates::Bool = true,
)
    return _rk4_step_dm(_nh_von_neumann_rhs, Hoft, ρ, t, dt;
        maxdim=maxdim,
        cutoff=cutoff,
        truncate_intermediates=truncate_intermediates,
    )
end


"""
    evolve_rk4_dm_nh(Hoft, ρ0::MPO, nsteps::Int, dt::Float64; maxdim=200,
                     cutoff=1e-10, truncate_intermediates=true, verbose=false)
        -> Vector{MPO}

RK4 evolution of a density-matrix MPO under the non-Hermitian von Neumann equation

    dρ/dt = -i(H(t) ρ − ρ H(t)†)

`Hoft` is a callable `t::Float64 -> MPO`.  For `H = H₀ - iΓ` the anti-Hermitian part
causes `Tr(ρ)` to decay whenever `Γ > 0`, modelling lossy open systems.

Returns `[ρ(0), ρ(dt), ..., ρ(nsteps*dt)]` as a `Vector{MPO}`.
See `evolve_rk4_dm_timedep` for keyword-argument descriptions.
"""
function evolve_rk4_dm_nh(Hoft, ρ0::MPO, nsteps::Int, dt::Float64;
    maxdim::Int     = 200,
    cutoff::Float64 = 1e-10,
    truncate_intermediates::Bool = true,
    verbose::Bool   = false,
)
    return _evolve_rk4_dm(rk4_step_dm_nh, "RK4-NH", Hoft, ρ0, nsteps, dt;
        maxdim=maxdim,
        cutoff=cutoff,
        truncate_intermediates=truncate_intermediates,
        verbose=verbose,
    )
end


# ============================================================
# 7. Density-matrix observables
# ============================================================

"""
    dm_expect(O, ρ) -> Float64

Compute `Tr(O ρ)` for a Hermitian operator MPO `O` and density-matrix MPO `ρ`,
using `inner(O, ρ) = Tr(O† ρ) = Tr(O ρ)`.
"""
function dm_expect(O::MPO, ρ::MPO)
    return real(inner(O, ρ))
end


"""
    observables_trajectory(ops, states) -> Dict

Measure a named collection of Hermitian operator MPOs along a density-matrix trajectory.

`ops` is a `NamedTuple` or `Dict` mapping labels to MPOs, e.g. `(J=J_mpo, N=N_mpo)`.
Returns a `Dict` mapping each label to a `Vector{Float64}` of expectation values.
"""
function observables_trajectory(ops, states::Vector{MPO})
    return Dict(k => [dm_expect(v, ρ) for ρ in states] for (k, v) in pairs(ops))
end


"""
    timedep_observable_trajectory(Oft, states, dt) -> Vector{Float64}

Measure a time-dependent Hermitian operator `O(t)` along a density-matrix trajectory.

`Oft` is a callable `t -> MPO`.  State `n` is assigned time `t_n = (n-1)*dt`.
Covers e.g. `⟨H(t)⟩ = Tr(H(t) ρ(t))` under a driven Hamiltonian.
"""
function timedep_observable_trajectory(Oft, states::Vector{MPO}, dt::Float64)
    return [dm_expect(Oft((step - 1) * dt), ρ) for (step, ρ) in enumerate(states)]
end


"""
    purity(ρ) -> Float64

Return the purity `Tr(ρ²) = inner(ρ, ρ)`.
Equals 1 for a pure state and decreases as the system becomes mixed.
"""
function purity(ρ::MPO)
    return real(inner(ρ, ρ))
end

"""
    purity_trajectory(states) -> Vector{Float64}

Return `purity(ρ)` for each MPO in `states`.
"""
function purity_trajectory(states::Vector{MPO})
    return [purity(ρ) for ρ in states]
end


"""
    bond_current_x(ρ, j, tx, L, sites) -> ComplexF64

Compute the x-direction bond current

    Jⱼˣ = i tₓ (ρⱼ,ⱼ₊₁ − ρⱼ₊₁,ⱼ)

for a single bond at 0-indexed site `j` in density-matrix MPO `ρ`.
`tx` is the hopping amplitude.
"""
function bond_current_x(ρ::MPO, j::Int, tx::Number, L::Int, sites)
    ρ_fwd = TensorBinding.matrix_checker(ρ, L, sites, j,     j + 1)
    ρ_bwd = TensorBinding.matrix_checker(ρ, L, sites, j + 1, j    )
    return im * tx * (ρ_fwd - ρ_bwd)
end


"""
    bond_current_x_trajectory(states, j, tx, L, sites; dt=1.0) -> Vector{ComplexF64}

Compute the x-direction bond current `Jⱼˣ(t)` along a trajectory of density-matrix MPOs.

`tx` may be a scalar or a callable `t -> hopping amplitude` for a time-dependent drive.
States are assumed spaced by `dt`: `t_n = (n-1)*dt`.
"""
function bond_current_x_trajectory(
    states::Vector{MPO},
    j::Int,
    tx,
    L::Int,
    sites;
    dt::Float64 = 1.0,
)
    return [bond_current_x(ρ, j, tx isa Number ? tx : tx((step - 1) * dt), L, sites)
            for (step, ρ) in enumerate(states)]
end


"""
    central_x_bond(L; Nx=nothing) -> Int

Return the 0-indexed site `j` of the central x-direction bond (`j → j+1`).

- 1D (`Nx=nothing`): central bond at `j = 2^L ÷ 2 − 1`.
- 2D row-major (`Nx = 2^Lx`): bond at the centre column of the centre row,
  `j = (Ny÷2)*Nx + Nx÷2 − 1` where `Ny = 2^L ÷ Nx`.
"""
function central_x_bond(L::Int; Nx::Union{Int,Nothing} = nothing)
    N = 2^L
    if Nx === nothing
        return N ÷ 2 - 1
    else
        Ny = N ÷ Nx
        return (Ny ÷ 2) * Nx + Nx ÷ 2 - 1
    end
end


# ============================================================
# 8. TBHamiltonian overloads (apply -im internally)
# ============================================================

function build_tdvp_propagator_mpo(H::TBHamiltonian, dt; kwargs...)
    return build_tdvp_propagator_mpo(-im * H.mpo, dt, H.L, H.sites; kwargs...)
end

function tdvp_evolve(H::TBHamiltonian, psi::MPS, dt::Real; kwargs...)
    return tdvp_evolve(-im * H.mpo, psi, dt; kwargs...)
end

function evolve_with_tdvp(H::TBHamiltonian, psi0::MPS, nsteps::Int, dt::Real; kwargs...)
    return evolve_with_tdvp(-im * H.mpo, psi0, nsteps, dt; kwargs...)
end

function check_tdvp_vs_U_mpo(H::TBHamiltonian, U_mpo::MPO, dt; kwargs...)
    return check_tdvp_vs_U_mpo(-im * H.mpo, U_mpo, dt, H.L, H.sites; kwargs...)
end
