# TensorBinding reorganisation — todo

Outcome of the code-organisation review of 2026-09-23 (four area sweeps over
`lattice/` + `core/Hamiltonian.jl`, `solvers/`, `physics/`, `gpu/`, plus repo-wide
metrics). Nothing here is implemented yet. Tiers are ordered so that each one can be
done and merged on its own with the test suite as the guard; Tier 1 changes no
behaviour, Tier 2 changes internals only, Tier 3 is user-visible.

Line numbers refer to the working tree on that date and will drift.

## Bugs (fix first, independently of the reorganisation)

- [ ] `core/TBSystem.jl` `_build_haldane` calls `haldane_hoppingf`, which is not defined
      anywhere; `get_Hamiltonian("haldane", …)` throws. Restore the function or drop the model.
- [ ] `physics/RPA_tk.jl:935` `get_bubble_mpo_haydock` calls `_build_heff` with 3 arguments;
      the definition (l.454) takes 4.
- [ ] `physics/RPA_tk.jl` ~1065 and ~1249: `get_rpa_susceptibility_wynn` and
      `get_magnon_susceptibility_wynn` assign `nq` inside `if chi_partial === nothing` inside
      the frequency loop, so the second frequency hits an undefined variable. Hoist `nq`.
- [ ] `physics/Topology_tk.jl` `get_C` accepts `Lambda` (ASCII alias) and never uses it; the
      GPU twin honours it.
- [ ] `core/TBSystem.jl` 17-argument `TBHamiltonian` compatibility constructor silently drops
      `Lx`, `interaction_mpo`, `fock_mpo` and `position_space`; SCF (l.364, 485, 1136),
      RPA (l.1141) and NH (l.99) copy Hamiltonians through it. Replace with a keyword copy
      constructor (see Tier 2) and delete the positional ones.
- [ ] `solvers/KPM_tk.jl` `get_density_quantics` uses an undefined global `sites`. Delete.
- [ ] `solvers/Timeev_tk.jl` `compare_propagator_and_tdvp_heatmaps` calls `heatmap`/`plot`/
      `display` although Plots is not a dependency. Move to `examples/`.
- [ ] `gpu/GPU_tk.jl:871` second `_sample_state_amplitudes_gpu` call drops `pointavg`.
- [ ] `gpu/GPU_tk.jl:267` `_onehot_gpu` only accepts `T<:Complex`; the advertised
      `type=Float32/Float64` paths fail.
- [ ] `physics/Purification_tk.jl:20` header example uses `method=:KPM`; code accepts `:kpm`.
- [ ] `README.md:22` claims CUDA is an installed dependency; `Project.toml` has none.
- [ ] `core/Utils.jl` `_exciton_block_groups` is reachable only through a branch that
      `get_exciton_ldos_spatial_gpu` rejects earlier (`reduce=:block`). Delete both.

## Tier 1 — mechanical, no behaviour change

### Split the three grab-bag files
- [ ] `solvers/KPM_tk.jl` (2067 lines) → `solvers/kpm/kernels.jl` (`_kpm_kernel`,
      `_dos_weight_matrix`, HODC helpers, `_kpm_weight_matrix` from QFT), `recursion.jl`
      (`KPM_Tn`, `KPM_Tn_mps`, `_run_kpm_mps!`), `cached.jl` (`get_ldos`, `get_ldos_spectrum`,
      `*_from_Tn`, `*_from_mun`, Green's functions), `ldos.jl` (`get_ldos_online`,
      `get_ldos_spatial`, split into `_ldos_spatial_mps`/`_ldos_spatial_mpo`), `dos.jl`
      (`get_dos_stochastic`, `get_dos_trace`), `exciton.jl` (l.1648–1984).
- [ ] `physics/RPA_tk.jl` (2148 lines) → `physics/rpa/Bubble.jl`, `Cheb2D.jl`, `Dyson.jl`;
      MPO kron/interleave plumbing (l.10–287) → `core/Utils.jl`; Haydock recursion →
      `solvers/Krylov_tk.jl`; `get_spect_k` → QFT conjugation file; delete l.288–377.
- [ ] `physics/QFT_tk.jl` (1643 lines) → `Conjugation.jl` (l.96–205), `Bands.jl`
      (l.638–948, 1223–1370), `KPath.jl` (l.426–635); exciton spectra (l.206–281, 949–1220)
      → exciton folder; aux projection (l.1373–1505) → `core/AuxDOF.jl`.
- [ ] `physics/NH_tk.jl` → `NH_model.jl` (struct, `hermitize`, `add_nh_*`) and `NH_KPM.jl`.
- [ ] `lattice/2Dlattice_tk.jl` (1615 lines) → `Masks2D.jl`, `Hopping2D.jl`, `Presets.jl`
      (QTCI `H*` builders incl. the 1D `HUniform`/`HSSH`/`HAAH`), `Sublattice.jl`
      (kagome/lieb/dice/honeycomb), `Geometry.jl`; `MODEL_REGISTRY`/`build_hamiltonian` →
      `core/ModelRegistry.jl`.
- [ ] `gpu/GPU_tk.jl` (3647 lines) → `device.jl`, `primitives.jl`, `kpm.jl`, `bands.jl`,
      `topology.jl`, `purification.jl`, `scf.jl`, `exciton.jl`, `nh.jl`, `timeev.jl`;
      the conductivity-only Tucker/QFT/Hadamard block (~300 lines) → its example.

### Move misplaced helpers next to their callers
- [ ] One `core/AuxDOF.jl` owning spin/Nambu indices and op tables, `prepend_spin`/`prepend_nambu`,
      Symbol overloads of `prepend_op`/`postpend_op` (from `Supercond_tk.jl`), `project_aux`,
      `aux_site`, `_autoenable_proj` (from QFT), `_aux_setup`, `_ldos_make_psi0` (from KPM),
      and the four `add_spin!`/`add_zeeman!`/`add_superconductivity!`/`add_soc!` mutators
      (from TBSystem). Include it right after TBSystem.
- [ ] `_estimate_spectral_bounds` → `solvers/DMRG_tk.jl`; include DMRG before KPM.
- [ ] `_eval_diag_mps` → `core/Utils.jl` beside `eval_mps`; `mpsexciton` → Utils beside the
      other product-state builders.
- [ ] `qtt_mpo`, `compose_power`, `_row_break/_row_select/_col_select/_row_checker_mpo`,
      `_site_projector_mpo`, `sigma_d/sigma_u` ops, layer prepend helpers → `core/Utils.jl`
      (or `lattice/Masks2D.jl` for the masks).
- [ ] BdG/pairing builders in `SCF_tk.jl` (l.298–528) → AuxDOF / Supercond.
- [ ] `_project_spin_sector` (RPA) → AuxDOF as `project_sector(H, :spin, σ)`.
- [ ] All geometry (`*_positions`, `_*_geometry`, `lattice_positions`, `_resolve_2d_geometry`,
      junction geometry, `geometry_uc` closures) → `lattice/Geometry.jl` with one `(Lx, Ly)`
      signature.
- [ ] `_reconstruct_ldos_moment_columns` (GPU) → `solvers/kpm/kernels.jl`; move its test out of
      `test/gpu_mps_ldos.jl`.

### Delete dead and legacy code
- [ ] Confirmed unreferenced everywhere (incl. notebooks and generated docs):
      `build_cyclic_shift_mpo`, `_geom_n_sub`, `_nsublat`, `nsitelegs`, `_tb_spatial_groups_gpu`,
      `get_nh_state_trajectory_gpu`, the `Delta_*` one-liners in SCF.
- [ ] Unreferenced in src/test/tracked examples: `projop_2DSL`, `projop_1DSL`, `sample_diag`,
      `project_spin`, `get_density_quantics`, `_get_exciton_ldos_cached` + exciton
      `KPM_Tn(H, N, X)`, `ldos_exc_KPM_Tn`, `get_mus_raw`, `compute_dos_ldos_hodc`,
      `kinetic_1d_nn_custom`, `qtci_matrix_to_MPO`, `quasicrystal_modulation_30deg`,
      `circular_mod`, `interchain_hopping_*` (2nd_plus/minus, triangle, honeycomb) with their
      skeleton/template helpers, `postpend_layer_projector/hopping`, `sdf_interval`,
      `mps_kron`, `merge_mps_to_mpo`, `convert_mpo`, `_swap_mpo`, `apply_interleave_swaps`,
      `get_Tnlists`, `get_bublle_expanded_from_Tn`, `build_bubble_mpo`,
      `get_bubble_mpo_haydock`, `hopping_mpo_exciton`, `get_valley_projectors`,
      `fock_exchange_builder`, `initial_guess_trivial_*_1D`, `nh_imag_onsite_mpo`,
      `add_nh_imag_onsite!`, `add_nh_loss!`, `nh_reconstruct_spectral_mpo`,
      `nh_spectral_function_allsite_mpo`, `spin_hamiltonian`, `bdg_hamiltonian` (re-inlined
      in TBSystem), `_onehot_gpu_f32`, `nh_spectrum_grid_gpu`. Check each once more before
      deleting; `examples/nontracked/APSOS/Modified_GPU_funcs.jl` carries forks of some.
- [ ] Commented-out legacy: `QFT_tk.jl:1511–1643` (old `get_bands`, `get_spect_k*`),
      `Purification_tk.jl:95–96`, unreachable code after early `return` in
      `2Dlattice_tk.jl` (`generate_kin_u/d` l.33–63, six kinetic builders l.388–543).
- [ ] Six positional "backward-compatible" `TBHamiltonian` constructors (TBSystem l.98–116,
      190–214) once Tier 2 keyword constructor exists.
- [ ] Unconditional `println` in library code (~70 in src): `Hamiltonian.jl` 85–123,
      `KPM_tk.jl` 14/30/31, `QFT_tk.jl` 1453–1470, `Topology_tk.jl` 499–539,
      `TBSystem.jl` 1175, RPA legacy pipeline; switch to `@info … maxlog=1` or `verbose` gates.

### Make the structure legible
- [ ] Explicit `export` list (today only ITensors names are exported) so public vs private is visible.
- [ ] One banner style (`# ====` vs `# ───` vs none); numbered sections that match contents
      (2Dlattice runs 8, 8b, 8c, 8d, 8f; SCF header lists 8 sections, file has 9).
- [ ] Rewrite the load-order comment in `TensorBinding.jl` as a real dependency graph; fix the
      include order where a solver depends on a physics file (KPM ↔ QFT, TBSystem → Supercond,
      Krylov → RPA, Bilayer → Twisted, SCF/RPA/Topology → Purification).
- [ ] File names: drop the `_tk` suffix; rename `2Dlattice_tk.jl`; fix header comments that
      cite files that do not exist (`utils.jl`, `2D_lattice.jl`, `twoparticle_tk.jl`, `krylov_tk.jl`).
- [ ] Re-save `2Dlattice_tk.jl` as UTF-8 and restore the mojibake symbols (√, ·, ≠ appear as
      `-`/`_`, e.g. `b=(1+-)/2` for the golden ratio).
- [ ] Docstrings vs signatures: `get_ldos_spatial` omits 9 kwargs; `get_ldos_from_mun` omits
      `eta`/`m_order`; Bilayer/Twisted claim `(MPO, sites)` returns but return `TBHamiltonian`;
      Flake/TBSystem examples pass `Lx=16`/`32` where `Lx` is a qubit count; `get_Hamiltonian`
      table lists 8 of 21 names; QFT table of contents (l.76–92) wrong in five places;
      Topology header lists `berry_curvature_integrand`, which does not exist.
- [ ] Tests: lattice builders, RPA, SCF, NH, Topology have no tests; add smoke tests before
      splitting so the moves are guarded.

## Tier 2 — shared kernels (internal behaviour only)

- [ ] `_scaled_hamiltonian(H; cutoff)` = `(1/scale)·(H − center·physical_projector(H))`,
      replacing ~20 inline copies (some use `MPO(sites,"Id")` and mishandle projected spaces:
      `KPM_tk.jl` 1799, 1917, 1675; `QPI_tk.jl` 155).
- [ ] `chebyshev_foreach(f!, H̃, T₀; maxdim, cutoff)` working for MPO and MPS on any device,
      replacing ~22 hand-written three-term loops (6 KPM, 14 GPU, QFT, QPI) and 5 NH partial
      recurrences; one truncation policy.
- [ ] `_kpm_energy_grid(H, ωs; kernel, …) -> (ω_r, W, denom, valid)` replacing 14 copies of the
      rescale/weights/valid block and 7 hand-written `π²·N·√(1−ω²)` normalisations.
- [ ] `_chebyshev_sum(Tn, coeffs; …)` replacing 6 weighted-sum copies; HODC variants become a
      coefficient choice.
- [ ] One Jackson kernel (`_kpm_kernel`) with a `normalize` keyword; delete `_jackson_kernel`
      (RPA) and `nh_jackson_weights` (NH).
- [ ] `AuxProjection` struct (or `aux...` kwargs forwarded to `_aux_setup`) replacing the
      8-keyword block copied into ~10 signatures; one `_project_aux_sectors` replacing the
      nambu→spin→layer→sublattice chain written 4× (KPM, QFT, GPU ×2) and the 4 sector
      projectors (`project_aux`, `_project_aux_block`, `_project_spin_sector`, `contract_nh_block`).
- [ ] `probe_state(H, x, σ…)` replacing the psi0 selection duplicated 3× in KPM.
- [ ] Keyword `TBHamiltonian(; L, N, sites, mpo, …)` plus `similar(H; mpo=, sites=, …)` copy
      constructor; delete the six positional overloads.
- [ ] One model registry entry per model (builder → `TBHamiltonian`, dim, params, geometry,
      scale) replacing `get_Hamiltonian`'s if-chain + `build_hamiltonian` + `_build_preset` +
      `_build_sublattice` + `_preset_geometry` + `_estimate_scale`; `_param(params, :t, default)`
      replacing the parsing ternaries; remove drifted `kw_defaults` from the registry.
- [ ] One `masked_shift_hopping(Lx, Ly, sites, hop, q; src_mask)` replacing six near-identical
      2D kinetic builders; retire `generate_kin_u/d` in favour of `shift_mpo`.
- [ ] `_sublattice_bond` + `_sublattice_setup` replacing ~12 repeated bond blocks in
      kagome/lieb/honeycomb/dice; `_basis_positions` replacing 4 identical position loops;
      `sum_mpos(terms; cutoff)`.
- [ ] `get_density` as the only projector dispatcher (delete `_get_density_matrix` in RPA and
      `_get_projector` in Topology); `_purified_pair` for the ρ± blocks in Purification.
- [ ] RPA: `_cheb2d_setup` + `_tucker_bases` (5 copied prologues, 2 Tucker blocks); one Wynn
      driver (3 copies); magnon functions as `mode=:magnetic`.
- [ ] Timeev: `_rk4_step(rhs, …)` (2 copies), one `evolve_rk4_dm_*`, one trajectory loop;
      remove the double normalisation after `tdvp(normalize=true)`.
- [ ] GPU: thin wrappers over CPU kernels with a `to_device` hook (stochastic DOS, McWeeny/SP2,
      Chern operator assembly, NH kernels, `_eval_block_mps`, `extract_diagonal_to_mps`,
      `mps_to_diagonal_mpo`, `density_profile_from_dm`); one `_to_gpu(x, T)`; one
      `_resolve_gpu_type` with a single warning threshold; `_gpu_log`.
- [ ] Decide the one remaining sampling divergence: `get_ldos_spatial_mps_gpu` automatic plan
      (balanced `fld` bins, `unique(round.(range))` samples) vs `spatial_sampling_plan` 1D branch.

## Tier 3 — API consistency (user-visible)

- [ ] `Ncheb` everywhere, positional (today `N`, `Ncheb`, `Nchebychev`, NH `n` meaning 2n).
- [ ] `cutoff` for SVD truncation; `tci_tol` / `krylov_tol` / `scf_tol` for the others
      (`tol` currently means four things).
- [ ] `boundary` only (drop `bc`, `cyclic` aliases); `maxdim` defaults from one
      `const KPM_DEFAULTS`; document the loose `tol=1e-8, maxdim=15` that `get_Hamiltonian`
      hands to every builder.
- [ ] `dtype` only (drop `type`); one `verbose::Int` level (drop `printinfo`).
- [ ] Method symbols in one case (`:kpm`, not `:KPM`); `fermi` vs `ϵF`; `Λ` vs `Lambda`;
      `omega` vs `ω_phys_vals`; exciton momenta `Q_*` only, one indexing convention.
- [ ] Return NamedTuples instead of kwarg-dependent shapes (`get_bands` Matrix/NamedTuple,
      `get_ldos_spatial_mps_gpu` four shapes, `get_ldos` MPS/MPO/Real/nothing, `thouless_pump`,
      `nh_spectrum_grid`, the four SCF drivers); an `SCFResult` struct.
- [ ] Split `mode` into `output=:operator|:diagonal` and `algorithm=:mpo|:mps`.
- [ ] Naming: `chern_marker`/`winding_marker` (keep `get_C`/`get_W` as deprecated aliases),
      `<model>_hamiltonian` everywhere, lowercase `_mpo` (`hopping2MPO` → `hopping_mpo`),
      `exciton_mpo` for `Exciton_Hamiltonian`, fix `get_bublle_expanded_from_Tn`.
- [ ] Replace hidden mutable caches (`_tn_cache`, `_tn_mps_cache`, `_density_cache`,
      `_ensure_scale!` side effects, solvers mutating user Hamiltonians) with an explicit
      `KPMExpansion` object passed to the reconstruction functions.
- [ ] CUDA as a package extension (`[weakdeps] CUDA`, `ext/TensorBindingCUDAExt/`), replacing
      the `Base.loaded_modules` UUID lookup; fix the README dependency statement.
