using ITensors
using ITensorMPS

"""
    build_tdvp_propagator_mpo(H, dt, L, sites; maxdim, cutoff, reverse_step,
                              outputlevel, nsite, cross_tol, initial_positions,
                              use_diagonal_pivots, interpolation_type) -> MPO

Build an MPO approximation of the short-time propagator `U(dt) = e^{-iH dt}` by
sampling matrix elements `⟨i|U(dt)|j⟩` via TDVP and compressing with TCI.

`H` must already be multiplied by `-im` for Schrödinger evolution.
The diagonal is dominant for small `dt`; by default TCI is seeded with all diagonal
pivots (`use_diagonal_pivots=true`) so the near-identity structure is captured first.

## Keyword arguments
- `maxdim`, `cutoff`    : TDVP truncation parameters.
- `cross_tol`           : TCI interpolation tolerance.
- `use_diagonal_pivots` : Seed TCI with the N diagonal positions. Default `true`.
- `interpolation_type`  : Element type for TCI sampling. Default `ComplexF64`.

A `TBHamiltonian` overload applies `-im` internally:
`build_tdvp_propagator_mpo(H::TBHamiltonian, dt; ...)`.
"""
function build_tdvp_propagator_mpo(
    H, dt, L, sites;
    maxdim = 50,
    cutoff = 1e-8,
    reverse_step = false,
    outputlevel = 0,
    nsite = 2,
    cross_tol = 1e-8,
    initial_positions = [],
    use_diagonal_pivots = true,   # seed TCI with diagonal to capture near-identity structure
    interpolation_type = ComplexF64,
)
    N = 2^L

    # For a short-time propagator the diagonal is dominant.  Seeding TCI with all
    # diagonal positions ensures it captures that structure before exploring off-diagonal.
    if use_diagonal_pivots && isempty(initial_positions)
        initial_positions = [(i, i) for i in 1:N]
    end

    function func(i, j)
        psi_i = TensorBinding.binary_to_MPS(Int(i - 1), L, sites)
        psi_j = TensorBinding.binary_to_MPS(Int(j - 1), L, sites)

        psi_j_evolved = tdvp(
            H,
            dt,
            psi_j;
            time_step = dt,
            nsite = nsite,
            maxdim = maxdim,
            cutoff = cutoff,
            normalize = false,   # must be false: normalization is state-dependent and breaks linearity
            reverse_step = reverse_step,
            outputlevel = outputlevel,
        )

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


"""
    tdvp_evolve(H, psi, dt; maxdim, cutoff, normalize, reverse_step,
                outputlevel, nsite) -> MPS

Apply one TDVP step to `psi` under Hamiltonian `H` for time `dt`.

`H` must already carry the `-im` prefactor for Schrödinger evolution.
When `normalize=true` the output is renormalised after the step.

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
    psi_out = tdvp(
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
    )

    if normalize
        nrm = sqrt(real(inner(psi_out, psi_out)))
        psi_out = psi_out / nrm
    end

    return psi_out
end


"""
    apply_mpo_to_mps(U_mpo, psi; cutoff, maxdim, normalize) -> MPS

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
    evolve_with_propagator(U_mpo, psi0, nsteps; normalize_each_step,
                           cutoff, maxdim) -> Vector{MPS}

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
    states = Vector{MPS}(undef, nsteps + 1)
    states[1] = copy(psi0)

    psi = copy(psi0)
    for step in 1:nsteps
        psi = apply(U_mpo, psi; cutoff = cutoff, maxdim = maxdim)
        truncate!(psi; cutoff = cutoff, maxdim = maxdim)
        if normalize_each_step
            normalize!(psi)
        end
        states[step + 1] = copy(psi)
    end

    return states
end


"""
    evolve_with_tdvp(H, psi0, nsteps, dt; normalize_each_step, maxdim,
                     cutoff, reverse_step, outputlevel, nsite) -> Vector{MPS}

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
    states = Vector{MPS}(undef, nsteps + 1)
    states[1] = copy(psi0)

    psi = copy(psi0)
    for step in 1:nsteps
        psi = tdvp(
            H,
            dt,
            psi;
            time_step = dt,
            nsite = nsite,
            maxdim = maxdim,
            cutoff = cutoff,
            normalize = normalize_each_step,
            reverse_step = reverse_step,
            outputlevel = outputlevel,
        )
        if normalize_each_step
            normalize!(psi)
        end
        states[step + 1] = copy(psi)
    end

    return states
end


"""
    evolve_with_tdvp_timedep(Hoft, psi0, nsteps, dt; normalize_each_step,
                             maxdim, cutoff, reverse_step, outputlevel,
                             nsite, krylovdim, tol) -> Vector{MPS}

TDVP loop for a time-dependent Hamiltonian `H(t)`.

`Hoft` is a callable `t::Float64 -> MPO`.  On each interval `[t, t+dt]` the
Hamiltonian is frozen at the midpoint `t + dt/2` (midpoint rule).  `Hoft` must
return the physical Hamiltonian; the `-im` prefactor is applied internally.

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
    states = Vector{MPS}(undef, nsteps + 1)
    states[1] = copy(psi0)

    psi = copy(psi0)
    for step in 1:nsteps
        t_mid = (step - 1) * dt + dt / 2
        Hmid = Hoft(t_mid)

        psi = tdvp(
            -im * Hmid,
            dt,
            psi;
            time_step = dt,
            nsite = nsite,
            maxdim = maxdim,
            cutoff = cutoff,
            normalize = normalize_each_step,
            reverse_step = reverse_step,
            outputlevel = outputlevel,
            updater_kwargs = (; tol = tol, krylovdim = krylovdim, eager = true),
        )
        if normalize_each_step
            normalize!(psi)
        end
        states[step + 1] = copy(psi)
    end

    return states
end


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
    check_tdvp_vs_U_mpo(H, U_mpo, dt, L, sites; test_states, ...) -> (max_overlap_error, max_phase_error)

Validate that `U_mpo` agrees with direct TDVP on a set of computational basis states.
Prints per-state overlap errors and phase-aligned distances, then returns the maxima.

A `TBHamiltonian` overload is available.
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


# dρ/dt = -i[H, ρ] RHS for Hermitian H
function _von_neumann_rhs(H::MPO, ρ::MPO; maxdim::Int, cutoff::Float64)
    Hρ   = apply(H, ρ; maxdim=maxdim, cutoff=cutoff)
    ρH   = apply(ρ, H; maxdim=maxdim, cutoff=cutoff)
    comm = +(Hρ, -1.0 * ρH; cutoff=cutoff)
    ITensorMPS.truncate!(comm; maxdim=maxdim, cutoff=cutoff)
    return -1.0im * comm
end


"""
    rk4_step_dm_timedep(Hoft, ρ, t, dt; maxdim, cutoff,
                        truncate_intermediates) -> MPO

Single RK4 step for `dρ/dt = -i[H(t), ρ]` with a time-dependent Hamiltonian MPO.
`H` is evaluated at `t`, `t+dt/2`, and `t+dt` per the classical RK4 tableau.
"""
function rk4_step_dm_timedep(Hoft, ρ::MPO, t::Float64, dt::Float64;
    maxdim::Int     = 200,
    cutoff::Float64 = 1e-10,
    truncate_intermediates::Bool = true,
)
    H0   = Hoft(t)
    Hmid = Hoft(t + dt / 2)
    H1   = Hoft(t + dt)

    k1 = _von_neumann_rhs(H0,   ρ;  maxdim=maxdim, cutoff=cutoff)

    ρ2 = +(ρ, (dt / 2) * k1; cutoff=cutoff)
    truncate_intermediates && ITensorMPS.truncate!(ρ2; maxdim=maxdim, cutoff=cutoff)
    k2 = _von_neumann_rhs(Hmid, ρ2; maxdim=maxdim, cutoff=cutoff)

    ρ3 = +(ρ, (dt / 2) * k2; cutoff=cutoff)
    truncate_intermediates && ITensorMPS.truncate!(ρ3; maxdim=maxdim, cutoff=cutoff)
    k3 = _von_neumann_rhs(Hmid, ρ3; maxdim=maxdim, cutoff=cutoff)

    ρ4 = +(ρ, dt * k3; cutoff=cutoff)
    truncate_intermediates && ITensorMPS.truncate!(ρ4; maxdim=maxdim, cutoff=cutoff)
    k4 = _von_neumann_rhs(H1,   ρ4; maxdim=maxdim, cutoff=cutoff)

    k_sum = +(k1, 2.0 * k2; cutoff=cutoff)
    k_sum = +(k_sum, 2.0 * k3; cutoff=cutoff)
    k_sum = +(k_sum, k4; cutoff=cutoff)
    ITensorMPS.truncate!(k_sum; maxdim=maxdim, cutoff=cutoff)

    ρ_new = +(ρ, (dt / 6) * k_sum; cutoff=cutoff)
    ITensorMPS.truncate!(ρ_new; maxdim=maxdim, cutoff=cutoff)

    return ρ_new
end


"""
    evolve_rk4_dm_timedep(Hoft, ρ0, nsteps, dt; maxdim, cutoff,
                          truncate_intermediates, verbose) -> Vector{MPO}

Evolve a density-matrix MPO `ρ0` under `dρ/dt = -i[H(t), ρ]` for `nsteps` steps
of size `dt` using RK4.

`Hoft` is a callable `t::Float64 -> MPO` returning the physical Hamiltonian at time `t`.
Returns the full trajectory `[ρ(0), ρ(dt), ..., ρ(nsteps*dt)]` as a `Vector{MPO}`.

## Keyword arguments
- `maxdim`, `cutoff`          : Truncation for intermediate MPO sums and products.
- `truncate_intermediates`    : Truncate after each RK4 sub-step to control bond growth.
- `verbose`                   : Print step/bond-dim progress.
"""
function evolve_rk4_dm_timedep(Hoft, ρ0::MPO, nsteps::Int, dt::Float64;
    maxdim::Int     = 200,
    cutoff::Float64 = 1e-10,
    truncate_intermediates::Bool = true,
    verbose::Bool   = false,
)
    states = Vector{MPO}(undef, nsteps + 1)
    states[1] = deepcopy(ρ0)

    ρ = deepcopy(ρ0)
    for step in 1:nsteps
        t = (step - 1) * dt
        verbose && println("RK4 step $step / $nsteps,  t = $t,  maxlinkdim = $(ITensorMPS.maxlinkdim(ρ))")
        ρ = rk4_step_dm_timedep(Hoft, ρ, t, dt;
            maxdim=maxdim,
            cutoff=cutoff,
            truncate_intermediates=truncate_intermediates,
        )
        states[step + 1] = deepcopy(ρ)
    end

    return states
end


# dρ/dt = -i(Hρ - ρH†) RHS for non-Hermitian H.
# H† is formed by swapping prime levels and conjugating: conj(swapprime(H, 0, 1)).
function _nh_von_neumann_rhs(H::MPO, ρ::MPO; maxdim::Int, cutoff::Float64)
    Hdag  = conj(swapprime(H, 0, 1))
    Hρ    = apply(H,    ρ; maxdim=maxdim, cutoff=cutoff)
    ρHdag = apply(ρ, Hdag; maxdim=maxdim, cutoff=cutoff)
    diff  = +(Hρ, -1.0 * ρHdag; cutoff=cutoff)
    ITensorMPS.truncate!(diff; maxdim=maxdim, cutoff=cutoff)
    return -1.0im * diff
end


"""
    rk4_step_dm_nh(Hoft, ρ, t, dt; maxdim, cutoff, truncate_intermediates) -> MPO

Single RK4 step for the non-Hermitian von Neumann equation

    dρ/dt = -i(H(t) ρ − ρ H(t)†)

`Hoft` is a callable `t -> MPO`; pass `(_ -> H.mpo)` for a static NH Hamiltonian.
"""
function rk4_step_dm_nh(Hoft, ρ::MPO, t::Float64, dt::Float64;
    maxdim::Int     = 200,
    cutoff::Float64 = 1e-10,
    truncate_intermediates::Bool = true,
)
    H0   = Hoft(t)
    Hmid = Hoft(t + dt / 2)
    H1   = Hoft(t + dt)

    k1 = _nh_von_neumann_rhs(H0,   ρ;  maxdim=maxdim, cutoff=cutoff)

    ρ2 = +(ρ, (dt / 2) * k1; cutoff=cutoff)
    truncate_intermediates && ITensorMPS.truncate!(ρ2; maxdim=maxdim, cutoff=cutoff)
    k2 = _nh_von_neumann_rhs(Hmid, ρ2; maxdim=maxdim, cutoff=cutoff)

    ρ3 = +(ρ, (dt / 2) * k2; cutoff=cutoff)
    truncate_intermediates && ITensorMPS.truncate!(ρ3; maxdim=maxdim, cutoff=cutoff)
    k3 = _nh_von_neumann_rhs(Hmid, ρ3; maxdim=maxdim, cutoff=cutoff)

    ρ4 = +(ρ, dt * k3; cutoff=cutoff)
    truncate_intermediates && ITensorMPS.truncate!(ρ4; maxdim=maxdim, cutoff=cutoff)
    k4 = _nh_von_neumann_rhs(H1,   ρ4; maxdim=maxdim, cutoff=cutoff)

    k_sum = +(k1, 2.0 * k2; cutoff=cutoff)
    k_sum = +(k_sum, 2.0 * k3; cutoff=cutoff)
    k_sum = +(k_sum, k4; cutoff=cutoff)
    ITensorMPS.truncate!(k_sum; maxdim=maxdim, cutoff=cutoff)

    ρ_new = +(ρ, (dt / 6) * k_sum; cutoff=cutoff)
    ITensorMPS.truncate!(ρ_new; maxdim=maxdim, cutoff=cutoff)

    return ρ_new
end


"""
    evolve_rk4_dm_nh(Hoft, ρ0, nsteps, dt; maxdim, cutoff,
                     truncate_intermediates, verbose) -> Vector{MPO}

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
    states = Vector{MPO}(undef, nsteps + 1)
    states[1] = deepcopy(ρ0)

    ρ = deepcopy(ρ0)
    for step in 1:nsteps
        t = (step - 1) * dt
        verbose && println("RK4-NH step $step / $nsteps,  t = $t,  maxlinkdim = $(ITensorMPS.maxlinkdim(ρ))")
        ρ = rk4_step_dm_nh(Hoft, ρ, t, dt;
            maxdim=maxdim,
            cutoff=cutoff,
            truncate_intermediates=truncate_intermediates,
        )
        states[step + 1] = deepcopy(ρ)
    end

    return states
end


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
    compare_propagator_and_tdvp_heatmaps(U_mpo, H, psi0, L, sites, nsteps; ...)

Full comparison of MPO-propagator and TDVP trajectories: evolves `psi0` with both
methods for `nsteps` steps, renders heatmaps of `|⟨x|ψ(t)⟩|` and `|⟨x|ψ(t)⟩|²`,
and returns all trajectory data and agreement metrics as a named tuple.

A `TBHamiltonian` overload is available.
"""
function compare_propagator_and_tdvp_heatmaps(U_mpo, H, psi0, L, sites, nsteps;
    normalize_each_step = true,
    dt = 0.1,
    plot_initial_overlap = true,
    mpo_cutoff = 1e-8,
    mpo_maxdim = 10_000,
    tdvp_maxdim = 200,
    tdvp_cutoff = 1e-10,
    tdvp_reverse_step = false,
    tdvp_outputlevel = 0,
    tdvp_nsite = 2,
)
    mpo_states = evolve_with_propagator(
        U_mpo, psi0, nsteps;
        normalize_each_step = normalize_each_step,
        cutoff = mpo_cutoff,
        maxdim = mpo_maxdim,
    )

    tdvp_states = evolve_with_tdvp(
        H, psi0, nsteps, dt;
        normalize_each_step = normalize_each_step,
        maxdim = tdvp_maxdim,
        cutoff = tdvp_cutoff,
        reverse_step = tdvp_reverse_step,
        outputlevel = tdvp_outputlevel,
        nsite = tdvp_nsite,
    )

    mpo_data  = compute_basis_overlaps(mpo_states,  L, sites)
    tdvp_data = compute_basis_overlaps(tdvp_states, L, sites)

    nbasis     = 2^L
    steps_axis = 0:nsteps
    basis_axis = 0:(nbasis - 1)

    abs_diff  = abs.(mpo_data.abs_overlaps  .- tdvp_data.abs_overlaps)
    prob_diff = abs.(mpo_data.probabilities .- tdvp_data.probabilities)

    state_overlaps      = Vector{ComplexF64}(undef, nsteps + 1)
    state_overlap_abs   = zeros(Float64, nsteps + 1)
    state_phase_distance = zeros(Float64, nsteps + 1)

    for step in 1:(nsteps + 1)
        psi_mpo  = mpo_states[step]
        psi_tdvp = tdvp_states[step]

        n_mpo  = sqrt(real(inner(psi_mpo,  psi_mpo)))
        n_tdvp = sqrt(real(inner(psi_tdvp, psi_tdvp)))

        ov = inner(psi_tdvp, psi_mpo) / (n_tdvp * n_mpo)
        state_overlaps[step]       = ov
        state_overlap_abs[step]    = abs(ov)
        state_phase_distance[step] = phase_aligned_distance(psi_tdvp, psi_mpo)
    end

    p1 = heatmap(basis_axis, steps_axis, mpo_data.abs_overlaps;
        xlabel="x", ylabel="step", title="MPO: |<x|ψ(step)>|", colorbar_title="magnitude")
    p2 = heatmap(basis_axis, steps_axis, tdvp_data.abs_overlaps;
        xlabel="x", ylabel="step", title="TDVP: |<x|ψ(step)>|", colorbar_title="magnitude")
    p3 = heatmap(basis_axis, steps_axis, mpo_data.probabilities;
        xlabel="x", ylabel="step", title="MPO: |<x|ψ(step)>|²", colorbar_title="probability")
    p4 = heatmap(basis_axis, steps_axis, tdvp_data.probabilities;
        xlabel="x", ylabel="step", title="TDVP: |<x|ψ(step)>|²", colorbar_title="probability")
    p5 = heatmap(basis_axis, steps_axis, abs_diff;
        xlabel="x", ylabel="step", title="Difference in |<x|ψ>|", colorbar_title="abs diff")
    p6 = heatmap(basis_axis, steps_axis, prob_diff;
        xlabel="x", ylabel="step", title="Difference in |<x|ψ>|²", colorbar_title="abs diff")

    display(plot(p1, p2; layout=(1, 2), size=(1200, 400)))
    display(plot(p3, p4; layout=(1, 2), size=(1200, 400)))
    display(plot(p5, p6; layout=(1, 2), size=(1200, 400)))

    if plot_initial_overlap
        initial_index = argmax(tdvp_data.probabilities[1, :])
        p7 = plot(steps_axis, mpo_data.abs_overlaps[:, initial_index];
            xlabel="step", ylabel="|<x₀|ψ(step)>|", label="MPO",
            title="Overlap with dominant initial basis state")
        plot!(p7, steps_axis, tdvp_data.abs_overlaps[:, initial_index]; label="TDVP")
        display(p7)
    end

    p8 = plot(steps_axis, mpo_data.norms;
        xlabel="step", ylabel="<ψ|ψ>", label="MPO", title="Norm comparison")
    plot!(p8, steps_axis, tdvp_data.norms; label="TDVP")
    display(p8)

    p9 = plot(steps_axis, state_overlap_abs;
        xlabel="step", ylabel="|<ψ_TDVP|ψ_MPO>|", label="|overlap|", title="State agreement")
    display(p9)

    p10 = plot(steps_axis, state_phase_distance;
        xlabel="step", ylabel="phase-aligned distance", label="distance",
        title="Phase-aligned state distance")
    display(p10)

    return (
        mpo_states             = mpo_states,
        tdvp_states            = tdvp_states,
        mpo_overlaps           = mpo_data.overlaps,
        tdvp_overlaps          = tdvp_data.overlaps,
        mpo_abs_overlaps       = mpo_data.abs_overlaps,
        tdvp_abs_overlaps      = tdvp_data.abs_overlaps,
        mpo_probabilities      = mpo_data.probabilities,
        tdvp_probabilities     = tdvp_data.probabilities,
        abs_overlap_difference = abs_diff,
        probability_difference = prob_diff,
        mpo_norms              = mpo_data.norms,
        tdvp_norms             = tdvp_data.norms,
        state_overlaps         = state_overlaps,
        state_overlap_abs      = state_overlap_abs,
        state_phase_distance   = state_phase_distance,
    )
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
    bond_current_x_trajectory(states, j, tx, L, sites; dt) -> Vector{ComplexF64}

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
    central_x_bond(L; Nx) -> Int

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


# ── TBHamiltonian overloads (apply -im internally) ───────────────────────────

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

function compare_propagator_and_tdvp_heatmaps(U_mpo::MPO, H::TBHamiltonian,
                                               psi0::MPS, nsteps::Int; kwargs...)
    return compare_propagator_and_tdvp_heatmaps(U_mpo, -im * H.mpo, psi0,
                                                  H.L, H.sites, nsteps; kwargs...)
end
