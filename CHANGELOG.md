# Changelog

All notable changes to TensorBinding.jl are listed here. Versions follow Julia's
reading of semantic versioning: before 1.0, a change in the minor number (0.1 → 0.2)
is breaking, and a change in the last number (0.1.0 → 0.1.1) adds features or fixes
bugs without breaking the API.

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
