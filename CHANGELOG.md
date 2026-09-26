# Changelog

All notable changes to TensorBinding.jl are listed here. Versions follow Julia's
reading of semantic versioning: before 1.0, a change in the minor number (0.1 → 0.2)
is breaking, and a change in the last number (0.1.0 → 0.1.1) adds features or fixes
bugs without breaking the API.

## [Unreleased]

Work towards v0.2.0: the code reorganisation tracked in `docs/dev/REORGANISATION_TODO.md`.

### Removed

Functions with no caller in the package, its examples or its tests. They were not exported,
but could be called as `TensorBinding.name`:
- `build_cyclic_shift_mpo`, `kinetic_1d_nn_custom`, `quasicrystal_modulation_30deg`,
  `circular_mod` and a few internal helpers (`nsitelegs`, `_nsublat`, `_geom_n_sub`, …);
- the legacy lattice builders `interchain_hopping_square_2nd_plus`,
  `interchain_hopping_square_2nd_minus`, `interchain_hopping_triangle`,
  `interchain_hopping_honeycomb` with their masks (`skeleton`, `odd_template`,
  `even_template`, `odd_skeleton`, `even_skeleton`), and `postpend_layer_projector`,
  `postpend_layer_hopping`;
- the legacy sublattice projectors `projop_1DSL`, `projop_2DSL`, `sample_diag` and
  `project_spin` (use `project_aux`);
- the cached exciton path `KPM_Tn(H, Ncheb, X::Int)` and `ldos_exc_KPM_Tn` (use
  `get_exciton_ldos_spatial`), `get_mus_raw`, `compute_dos_ldos_hodc`;
- the old RPA pipeline `get_Tnlists`, `get_bublle_expanded_from_Tn`, `build_bubble_mpo`
  (use `get_bubble_mpo`) with `mps_kron`, `merge_mps_to_mpo`, `convert_mpo`,
  `apply_interleave_swaps`; `hopping_mpo_exciton`;
- `add_nh_loss!(H, f)` (use `add_loss!(H, f)`, same `−i` default),
  `add_nh_imag_onsite!(H, f; prefactor)` (use `add_loss!(H, f; coefficient=prefactor)`;
  note that `add_nh_imag_onsite!` defaulted to `+i`), `nh_imag_onsite_mpo(H, f; prefactor)`
  (use `prefactor * loss_profile_mpo(H, f)`), `nh_reconstruct_spectral_mpo` and
  `nh_spectral_function_allsite_mpo` (use `nh_spectral_function`);
- the GPU aliases `get_nh_state_trajectory_gpu` (use `get_state_amplitude_trajectory_gpu`)
  and `nh_spectrum_grid_gpu` (use `get_nh_dos_grid_gpu`);
- `nh_jackson_weights(N)`: the NH KPM now takes its Jackson weights from the shared KPM
  kernel, `TensorBinding._kpm_kernel(N + 1, :jackson)[1:N]`, which gives the same values
  bit for bit.

The positional `TBHamiltonian` constructors with 13, 14, 15, 16, 17 and 20 arguments (the
"backward-compatible" forms, which filled in the fields added since) are removed. Use the
keyword constructor (see **Added**) instead:
`TBHamiltonian(L, N, sites, mpo, geom, scale, 0.0, nothing, nothing, nothing, nothing, 0,
nothing)` becomes `TBHamiltonian(; L, N, sites, mpo, geometry=geom, scale)`. To change
fields of an existing Hamiltonian, use `TBHamiltonian(H; field=value, ...)`. The
constructor that takes all 21 fields in order, caches included, remains.

### Added

- An export list: `using TensorBinding` now brings 58 main entry points into scope
  (`get_Hamiltonian`, the `add_*!` mutators, `KPM_Tn`, `get_ldos_spatial`, `get_bands`,
  `get_scf`, the main `*_gpu` functions, …), listed by area under "Public API" on the
  documentation home page. Only names specific to TensorBinding are exported; the rest of
  the API, including every type, is still called as `TensorBinding.name`. Qualified calls
  and `using TensorBinding: name` imports work as before. A script that defines its own
  top-level function or variable with one of these names now shadows the export; on Julia
  1.11 and earlier that definition is an error if the script used the exported name first.
  A session that loads the source with
  `include("src/TensorBinding.jl"); using .TensorBinding` and runs that setup again (or
  also loads the installed package) now gets ambiguous names: bare calls fail with
  `UndefVarError`, qualified `TensorBinding.name` calls still work. Restart the session
  before re-running the setup, or use `] dev` and Revise with `using TensorBinding`.
- `estimate_scale(geometry, params; L, method)`: KPM scale estimates for any
  `get_Hamiltonian` model, by the dense spectrum of the same model at a small size
  (`:small`), the padded row-sum bound of its terms (`:geometry`) or DMRG (`:dmrg`).
  `get_Hamiltonian(...; scale=:small | :geometry | :dmrg)` builds with that estimate.
- `interval_sampling_plan`: the automatic probe plan of `get_ldos_spatial_mps_gpu`, now a
  planner of its own in `core/Utils.jl` (same groups as before).
- `TBHamiltonian(; L, N, sites, mpo, geometry, geometry_uc, scale, center, spin_s, nambu_s,
  layer_s, sublattice_s, aux_side, Lx, position_space, interaction_mpo, fock_mpo)`: build a
  Hamiltonian from its fields by name. Only `L`, `N`, `sites` and `mpo` are required; the
  other fields default to "not set" (`scale = center = 0.0`, `aux_side = :pre`, the binary
  position space, `nothing` otherwise) and the caches start empty. Every lattice, registry
  and position-space builder now constructs through it; the Hamiltonians they return are
  unchanged.
- `chebyshev_foreach(f!, H̃, T₀, N; maxdim, cutoff, ...)`: the Chebyshev recursion
  `T_{n+1} = 2H̃T_n − T_{n−1}` on MPOs (operator series) or MPS (Chebyshev vectors), with
  CPU or GPU tensors, calling `f!(n, T_n)` at every order; keywords say where `cutoff` and
  `maxdim` truncate (the product, the sum, an extra `truncate!`). The KPM, band-structure,
  QPI, non-Hermitian and GPU solvers now all run their recursions through it, each with
  its former truncation, so their results are unchanged bit for bit.

### Changed

- Unconditional progress prints in library code ("MPS COMPUTED!", "C1 done", …) are
  `@debug` messages (`JULIA_DEBUG=TensorBinding` shows them). The DMRG estimate of the
  spectral bounds, which sets an automatic KPM scale, is reported as an `@info` message.
  The spinless s-wave → p-wave notice of `add_superconductivity!` is an `@info` message.
- `get_Hamiltonian` looks every geometry up in one model registry (`MODELS` in
  `core/ModelRegistry.jl`); `MODEL_REGISTRY` and `build_hamiltonian` are unchanged.
- **GPU precision warning**: the GPU entry points that warn about a tight `cutoff` share
  one helper and one wording, and warn only for a 32-bit element type (`ComplexF32`,
  `Float32`). Each keeps its threshold: `cutoff < 1e-6`, or `1e-4` for
  `get_nh_dos_grid_gpu` and `get_nh_dos_points_gpu`, `1e-5` for
  `scf_magnetic_hubbard_gpu`. What changes: `get_ldos_spatial_gpu` and
  `get_exciton_ldos_spatial_gpu` also warn for `Float32`; the non-Hermitian DOS no longer
  repeat the warning for every point; `scf_magnetic_hubbard_gpu` no longer warns for a
  64-bit type. No value changes.
- The GPU functions now run the CPU kernels on GPU tensors wherever those kernels apply
  unchanged (stochastic DOS, McWeeny/SP2 purification, Chern-marker assembly, diagonal
  extraction and embedding, block evaluation, the non-Hermitian block contraction and
  probes); their results and printed progress are unchanged. The internal upload helpers
  `_to_gpu_mpo`/`_to_gpu_mps` and `_ensure_gpu_mpo`/`_ensure_gpu_mps` are replaced by
  `TensorBinding._to_gpu(x, T)` and `_ensure_gpu(x, T; caller)`; a script that called the
  one-argument `_to_gpu_mps(ψ)` (ComplexF32) calls `TensorBinding._to_gpu(ψ, ComplexF32)`.

### Changed results

- **Default KPM scales** of `get_Hamiltonian` (no `scale` keyword) are now
  `max(former default, estimate_scale(...; method=:auto))` for `"chain_1d"` and the preset
  models except `"chernhex"`. The Hamiltonians themselves are unchanged. Exactly these
  defaults move:
  - `"qc2dsquare"`: always, `6|t|` → `10.56|t|` (padded row-sum bound); `6|t|` was below the
    spectral radius from `L ≈ 12`.
  - `"chern8"`: when `|V·t2| > 1/11` (with the default `t2 = 0.2`: `|V| > 0.455`), to
    `1.1(4|t| + 16|t·V·t2|)`; e.g. `V = t = 1`: `6` → `7.92`.
  - `"aah"`: wherever `1.1 ×` the spectral radius of the chain (built at `L ≤ 10`) exceeds
    `1.2(|t| + |V|)`, i.e. for weak potentials: at `t = 1`, `V = 0.5` goes from `1.8` to
    `2.11` at `L = 3` and `2.24` at `L ≥ 10`, `V = 0.8` from `2.16` to `2.30` at `L ≥ 10`;
    `V = 1` and `V = 2` are unchanged.
  - `"uniform"`: when the on-site `v` exceeds about `0.27|t|` (`2.5|t|` < `1.1(2|t| + |v|)`).
  - `"ssh"`: when `|d|` exceeds about `1.14|t|` (bonds `t ± d` beyond the `2.5|t|` window).
  - Never: `"chain_1d"`, `"square_2d"`, `"hex_2d"`, `"triangular_2d"`,
    `"triangular_bravais"`, whose former default already bounds `1.1 ×` the row sum;
    `"haldane"`, `"chernhex"`, the multi-atom lattices, `"custom"` and the projected spaces
    keep their builders' defaults.
  Pass `scale=` explicitly to reproduce an old result.
- **`get_qpi`** on a projected position space raises a clear `ArgumentError` (QPI needs a
  binary register). It used to return maps that included the unphysical register states.
- **Projected position spaces** (Fibonacci, metallic-mean, k-bonacci): every KPM path that
  accepts one now shifts the spectrum and starts its Chebyshev recursion with
  `physical_projector(H)`, as most of them already did. (The Green's-function recursion of the RPA
  bubble on the doubled 2L-site register still uses that register's identity: there is no
  projected space for it.) The last ones that used the
  identity of the whole register:
  - the RPA bubbles with `P_method=:kpm` (`get_bubble_mpo`, `get_bubble_mpo_haydock`,
    the cheb2d bubbles and the susceptibilities built on them): the density matrix is now
    zero on the unphysical register states, where it used to hold spurious weight. The
    physical block is the same up to truncation (without truncation the two agree to
    ~1e-12; with a binding `maxdim` it moves within the truncation error, e.g. ~20 % at
    `maxdim = 30` on an L = 4 Fibonacci chain). The cheb2d bubbles move by ~1e-13.
  - `get_qpi` now throws on a projected space, in the diagonal-LDOS accumulation (a known
    issue with the leg order of the projector). It used to return maps that included the
    unphysical register states. It was never valid there: the impurity address and the
    Fourier transform are binary.
  Binary position spaces are unaffected, bit for bit.

## [0.1.1] — unreleased

This release keeps the v0.1 API. Every entry under **Changed results** is a bug fix
that moves numbers; reproduce an old result by pinning v0.1.0 or by passing the
keyword named in the entry.

### Added

- **Projected position spaces for one-dimensional quasicrystals** (the construction of
  arXiv:2609.06040), available through `get_Hamiltonian`:
  - `"fibonacci"`: Zeckendorf-projected Fibonacci chain, with conumber ordering and the
    Fibonacci LDOS sampling planner (`fibonacci_ldos_sampling_plan`);
  - `"metallic_mean"`: the metallic-mean family `A → AᵐB, B → A` (keyword `m`);
  - `"kbonacci"`: the k-bonacci family `aᵢ → a₁aᵢ₊₁, aₖ → a₁` (keyword `k`; `k = 3` is
    Tribonacci).
  CPU KPM DOS/LDOS and `get_ldos_spatial_mps_gpu` support these spaces; other functions
  raise an `ArgumentError` on them.
- `TBHamiltonian(H; field=value, ...)`: copy a Hamiltonian, replacing the named fields.
- `return_maxlinkdim` keyword for `get_exciton_ldos_spatial` and its GPU version.
- `hopping2MPO` keywords `nrandominitpivot` and `nsearchglobalpivot` (defaults unchanged).
- `build_tdvp_propagator_mpo` keywords `expand_basis` and `cache_columns`.
- Regression tests for every fix below, and golden tests that pin the output of every
  sampling planner.

### Changed results

- **The KPM density matrix is the occupied-state projector.** `get_density_from_Tn`
  had the signs of its Chebyshev coefficients reversed and returned θ(H − ϵF), the projector
  onto the EMPTY states (at half filling with a symmetric spectrum both have trace N/2, which
  hid it). This affects `get_density(H; method=:kpm)`, local Chern markers computed with
  `method=:KPM` (`get_C`, `get_C_gpu`), whose sign was opposite to `:mcweeny` (winding
  markers, `get_W`, are unchanged under P → 1 − P and were not affected), RPA bubbles with
  `P_method=:kpm`, and SCF runs that use the KPM density.
  Purification-based paths (`:mcweeny`, `:sp2`, the defaults) were not affected.
- **Haldane models are now the textbook, C₃-symmetric model.** `get_Hamiltonian("haldane")`
  and `"chernhex"` gave the vertical next-nearest-neighbour bonds the wrong phase, so the
  Dirac masses were `−M ± √3·t2·sin φ` instead of `−M ± 3√3·t2·sin φ` and the topological
  phase ended at `|M| = √3|t2 sin φ|`. Only those vertical entries change; results of
  `"haldane"` at `φ = 0` or `π` are unchanged. The sign of the Chern number in the
  topological phase is unchanged for both presets: `"haldane"` uses the same `φ`
  convention as a model built with `add_hopping_2D!` from the C₃ phase table (as in the
  manuscript scripts), and `"chernhex"` keeps its opposite sign (it equals `"haldane"` at
  `φ = −π/2`). `"haldane"` now checks that `rs` has the `honeycomb_positions` layout.
- **The default `"chernhex"` domain wall** keeps its mass profile `ms + 3.3√3·t2` on the
  right half, which now sits 10 % above the textbook critical mass `3√3·t2` (before: far
  into the trivial phase), so the trivial side's gap is smaller. For `ms < −0.3√3·t2` both
  halves are Chern insulators with opposite Chern numbers.
- **The default `"chernhex"` scale** is `max(6|t|, 1.1·(3|t| + 6|t2| + max|Ms|))`, which
  bounds the spectrum (before: `6|t|`, too small for larger `t2` or `ms`). It is unchanged
  where `6|t|` is the larger; parameters passed through `mparams` now count, which can
  also lower it (e.g. `t = 0.5`).
- **`get_Hamiltonian("haldane")`** builds a deterministic MPO that is checked against the
  model (before: it depended on the global random number generator, often missed part of
  the matrix at `L ≥ 7`, and sometimes threw "maxsamplevalue is zero!"). Its default scale
  is `1.1·(3 + 6|t2| + |M|)`, which always bounds the spectrum.
- **`build_tdvp_propagator_mpo`** now matches `exp(−iH·dt)` to about `1e-4` (before: about
  `0.1` off, with hops across several qubits missing). New defaults
  `use_diagonal_pivots=false`, `reverse_step=true`, `expand_basis=true`; the old default
  path threw an error, and the old output is reproducible with
  `reverse_step=false, expand_basis=false`. TDVP runs at most once per sampled column, so
  default builds are 5–13× faster than with `use_diagonal_pivots=true`.
- **`get_bands_gpu`** in grid mode on models with an auxiliary index (sublattice, spin,
  Nambu, layer) plans the k-grid on the correct number of position qubits, as the CPU
  `get_bands` does (before: half the k-range, and wrong values with `sublattice=true`).
- **`get_bubble_mpo_cheb2d`/`_tucker`** keep every level of qudit position registers
  (metallic mean); before, they silently truncated each site to two levels.
- **Layered builders** (`bilayer_hamiltonian`, `multilayer_hamiltonian`,
  `twisted_*_hamiltonian`) set `H.Lx`, so block, grid and box LDOS maps and
  `get_valley_operator` are correct on them (before: wrong neighbourhoods or errors on
  rectangular systems).
- **Copies of Hamiltonians inside SCF, NH, RPA and exciton code** keep `Lx`,
  `interaction_mpo`, `fock_mpo` and the position space. As a consequence,
  `scf_meanfield`/`scf_magnetic_hubbard` on a projected position space raise an error
  instead of silently running unprojected.
- **`spatial_sampling_plan`** raises an error for 1D requests it cannot fill: `num_x`
  larger than the window, an empty window, or `num_avg < 1`. Before, they returned empty
  groups, and `get_ldos_spatial` then crashed (`mode=:mpo`) or returned NaN columns
  (`mode=:mps`). This is the common case of setting `x_start`/`x_end` without `num_x`;
  pass `num_x=0` to sample every site in the window. No working call changes: the golden
  tests confirm every other plan is identical.

### Fixed

- `get_Hamiltonian("haldane")` threw `UndefVarError` (its hopping function had been deleted).
- `hopping2MPO` passed QTCI initial pivots as a keyword and threw whenever initial
  positions were given (this broke `build_tdvp_propagator_mpo`'s default).
- `evolve_with_propagator` called the wrong `truncate!` and always threw.
- RPA: `get_bubble_mpo_haydock` called `_build_heff` with too few arguments; the Wynn
  drivers failed from the second frequency on; `get_rpa_susceptibility(mode=:magnetic)`
  mixed spinful and spin-projected sites; the cheb2d MPO bubbles failed on spinful,
  sublattice, layer and Nambu registers (the diagonal variants now reject them with a
  clear message).
- `get_C` ignored its `Lambda` keyword.
- GPU: `get_state_amplitude_trajectory_gpu` dropped `pointavg` after the first sample;
  `get_ldos_spatial_gpu` failed for `type=Float32/Float64`; documented `*_gpu` helpers now
  raise the "load CUDA.jl" error when CUDA is missing.
- `add_hopping!` on a layered Hamiltonian without a geometry now says what to call instead.
- `get_bands_gpu` with a real `type` raises a clear `ArgumentError` (the quantics Fourier
  transform is complex); before, it failed with a `MethodError`.
- Documentation: the Purification example used `method=:KPM` (the code accepts `:kpm`);
  the README claimed CUDA is a dependency (it is optional: load it with `using CUDA`);
  McWeeny purification converges quadratically; `get_C`/`get_C_gpu` signatures list
  `Lambda` and `sequential`; the `bdg_hamiltonian` example is a valid call.

### Removed

- The unused HDF5 dependency.
- `get_density_quantics`, which could not run (it used an undefined variable).
- `compare_propagator_and_tdvp_heatmaps`, which needed Plots; it now lives in
  `examples/dynamics/propagator_vs_tdvp_heatmaps.jl`.
- The unreachable internal helper `_exciton_block_groups`.

## [0.1.0] — 2026-07-01

First registered release.
