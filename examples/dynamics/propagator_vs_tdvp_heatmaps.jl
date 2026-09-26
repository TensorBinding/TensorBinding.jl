# propagator_vs_tdvp_heatmaps.jl
#
# Side-by-side comparison of MPO-propagator and TDVP time evolution: heatmaps of
# |<x|psi(t)>| and |<x|psi(t)>|^2 for both methods, their difference, and the
# norm / overlap / phase-aligned-distance agreement curves.
#
# compare_propagator_and_tdvp_heatmaps used to live in src/solvers/Timeev_tk.jl.
# It was moved here because it calls heatmap/plot/display and Plots is not a
# TensorBinding dependency. The numerical helpers it calls (evolve_with_propagator,
# evolve_with_tdvp, compute_basis_overlaps, phase_aligned_distance) are still in
# the package.
#
# Plots must be available in the active or a stacked environment; it is not in
# the TensorBinding Project.toml. Usage:
#
#   include("propagator_vs_tdvp_heatmaps.jl")
#   L    = 6
#   dt   = 0.05
#   H    = TensorBinding.get_Hamiltonian("chain_1d", 1.0; L=L, scale=4.5)
#   U    = TensorBinding.build_tdvp_propagator_mpo(H, dt)
#   psi0 = TensorBinding.binary_to_MPS(2^L ÷ 2, L, H.sites)
#   res  = compare_propagator_and_tdvp_heatmaps(U, H, psi0, 40; dt=dt)

using Plots
using TensorBinding
using TensorBinding: TBHamiltonian, evolve_with_propagator, evolve_with_tdvp,
                     compute_basis_overlaps, phase_aligned_distance


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


# ── TBHamiltonian overload (applies -im internally) ───────────────────────────

function compare_propagator_and_tdvp_heatmaps(U_mpo::MPO, H::TBHamiltonian,
                                               psi0::MPS, nsteps::Int; kwargs...)
    return compare_propagator_and_tdvp_heatmaps(U_mpo, -im * H.mpo, psi0,
                                                  H.L, H.sites, nsteps; kwargs...)
end
