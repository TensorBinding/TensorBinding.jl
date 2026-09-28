# Changelog

All notable changes to TensorBinding.jl are listed here. Versions follow Julia's
reading of semantic versioning: before 1.0, a change in the minor number (0.1 → 0.2)
is breaking, and a change in the last number (0.1.0 → 0.1.1) adds features or fixes
bugs without breaking the API.

## [Unreleased]

Work towards v0.2.0: the code reorganisation tracked in `docs/dev/REORGANISATION_TODO.md`,
and fixes for the bugs it found. A fix that moves numbers is listed under **Changed results**.

### Removed

- The `Arpack` dependency, which the package never used (a script that relies on it being
  installed alongside TensorBinding adds it to its own environment).

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
  bit for bit;
- `_jackson_kernel(N)` (RPA): the low-rank cheb2d bubbles take the textbook Jackson kernel
  `_kpm_kernel(N + 1, :jackson)[1:N] ./ (N + 1)` instead (see **Changed results**).

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
- `hopping2MPO(f, N, sites; check=false)`: with `check=true` the QTCI result is compared
  with `f` on a fixed sample of entries (a spread of rows; column offsets 0, ±1, ±2, ±3,
  ±2^k, ±(2^k ± 1)) and, if it is wrong or QTCI threw "maxsamplevalue is zero!", rebuilt
  from the nonzero sampled entries as pivots, deterministically; a second failure is an
  error. A build that passes is returned unchanged, and the check draws nothing from the
  RNG. `get_Hamiltonian("custom", f)` and `add_hopping!(H, f)` with a two-argument `f`
  turn it on; their new keyword `check=false` turns it off.

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
  projected space for it.) The last ones that used the identity of the whole register are
  the RPA bubbles with `P_method=:kpm` (`get_bubble_mpo`, `get_bubble_mpo_haydock`, the
  cheb2d bubbles and the susceptibilities built on them): the density matrix is now zero on
  the unphysical register states, where it used to hold spurious weight. The physical block
  is the same up to truncation (without truncation the two agree to ~1e-12; with a binding
  `maxdim` it moves within the truncation error, e.g. ~20 % at `maxdim = 30` on an L = 4
  Fibonacci chain). The cheb2d bubbles move by ~1e-13. (`get_qpi` stops earlier, with the
  `ArgumentError` above.) Binary position spaces are unaffected, bit for bit.
- **`get_ldos(H, ω; mode=:mps, psi0)`** with a probe that is not normalised: `KPM_Tn_mps`
  caches the Chebyshev vectors of `ψ₀/‖ψ₀‖`, but the moments were taken with `ψ₀` itself,
  so the LDOS grew linearly with `‖ψ₀‖`. It is now the LDOS of `ψ₀/‖ψ₀‖`. Normalised probes
  (`probe_state`, `physical_site_state`, `mpsexciton`) are unaffected.
- **k-space sampling** (`kspace_sampling_plan`, behind `get_bands` and `get_bands_gpu`
  without `k_groups`):
  - `ilinspace(xmin, xmax, 1)` returns `[xmin]`; it returned `[0]`, outside the window
    when `xmin > 0`. A 1D band plot with `num_x = 1` and `xmin > 0` moves accordingly.
  - The 2D diagonal cut now places its `num_x` points along the whole cut from
    `(xmin, ymin)` to `(xmax, ymax)`. It took the first `num_x` points of a
    full-resolution grid, so `num_x = 4` on a 16 × 16 zone sampled `k = 0, 1, 2, 3`
    along the diagonal. On registers with twice as many y as x labels (odd `L`) the cut
    follows the physical diagonal `kx = ky` (it followed `ky = kx / 2`). A window
    narrower than the zone (`xmin > 0` or `xmax < 2^Lx − 1`) failed an assertion; it
    now samples the cut inside the window. Unchanged: square zones sampled at full
    resolution (`num_x ≥ 2^Lx`, the default up to `Lx = 5`) and every `kpath_2d` path.
- **`fix_sites`** on an MPO whose tensors store the ket leg first, `(s, s')`: it took the
  second leg as the ket and returned the transpose. It now maps legs by prime level, so
  only such MPOs change (the physical projector of a projected position space is one);
  QTCI tensor trains and MPOs stored `(s', s)` are mapped as before.
- **cheb2d bubbles are 4× larger.** `chebyshev2d_gf_coeffs` divided the 2D cosine
  transform by `(2N)²` instead of `N²`, so its expansion reproduced `f/4`. This scales
  `get_bubble_mpo_cheb2d(_tucker)` and `get_bubble_diag_cheb2d(_svd, _tucker)`: with
  `kernel=:none` and no binding `coeff_tol`, by exactly 4. `coeff_tol` is an absolute
  threshold on the coefficients, so a given value now keeps more terms.
- **Jackson kernel of the low-rank cheb2d bubbles** (`kernel=:jackson`, the default of the
  Tucker and SVD variants): the textbook kernel for the `Ncheb + 1` moments (`g₀ = 1`),
  from the shared `_kpm_kernel`. The RPA-only `_jackson_kernel` had `(N − m)` for
  `(N − m + 1)`, so `g₀ = N/(N + 1)`; the bubbles move by up to `(N + 1)²/N²` on top of the
  factor 4 (e.g. ×4.16 at `Ncheb = 50`).
- **cheb2d bubbles on a complex `H₁`**: the Hadamard product takes the transpose of the
  `H₁` factors, so the bubble has the Lindhard form `(P_a)ᵀ ⊙ P_b` of `get_bubble_mpo`
  and conserves particles; it was `P_a ⊙ P_b` (44 % off on a complex L = 2 chain, row
  sums 0.35‖Π‖). Real `H₁` results are unchanged, bit for bit. The cheb2d bubbles carry
  the opposite sign to `get_bubble_mpo`; their docstrings now say so (which sign `Π₀`
  should have is open).
- **RPA bubbles with `P_method=:purification` use `ϵF`.** It was never passed, so every
  purification ran at `ϵF = 0`. McWeeny now starts from the level `H.center + ϵF`, the
  convention of `mcweeny_purify` (the `:kpm` density is `θ(ϵF − H)`: the same level when
  `H.center = 0`). `ϵF = 0` is unchanged, bit for bit. A cached `H._density_cache` is
  still returned whatever `ϵF`, as in `get_density`: clear it when scanning `ϵF`.
- **`haydock_cf`** (and `get_bubble_mpo_haydock`, `haydock_resolve_mpo`) measures with the
  Frobenius product `Tr[A†B]`, contracted exactly. It took `Tr[conj(A)·B]` of a truncated
  product, which is `Tr[A†B]` only for symmetric `A`: wrong for complex `H` and for seeds
  whose Krylov vectors are not symmetric (a projector seed stopped after one step).
  Real symmetric cases move only at the truncation level.
- **`wynn_epsilon`** returns the limit of an exactly converged sequence (a constant or
  geometric one) instead of the `1e30` sentinel; a singular table (an arithmetic
  sequence) keeps the sentinel. The Wynn RPA drivers' pinned results are unchanged.
- **Local Chern markers are real.** `get_C`, `get_C_gpu`, `get_valley_C` and
  `get_C_op_MPO_from_P` evaluate ⟨α|C|α⟩ for C = 2πi(QXPYQ − PXQYP), which is not
  Hermitian: its traceless anti-Hermitian part gave every site an imaginary part of up to
  O(0.1) (±0.16i on a trivial Semenoff honeycomb, whose marker is 0). They now return the
  diagonal of its Hermitian part, the real part, which is the mean of the Bianco–Resta
  P- and Q-forms. Real parts are unchanged, bit for bit; the values stay `ComplexF64`.
- **SP2 purification** (`sp2_purify`, `get_density(method=:sp2)`, the SP2 paths of the
  topology markers, `get_C_gpu`, the SCF drivers and the RPA bubbles) stopped only at `tol`
  or `maxiters`. With the default cutoff its truncation floor (1e-5 to 1e-3) lies above
  the default `tol` = 1e-5, and once truncation pushes eigenvalues out of [0, 1] the
  iteration doubles them each step, to NaN or to a matrix far from a projector. It now
  stops as soon as Tr ρ² > Tr ρ and returns the iterate with the smallest residual (an
  error if that residual is still ≥ 0.1: an invalid guess). Runs where the stop never
  fires return what they did before.
- **Default SP2 electron count**: half the number of states, `prod(dim, H.sites) ÷ 2`,
  instead of `H.N ÷ 2` (half the unit cells: quarter filling on sublattice, layer, spin and
  BdG models), in `sp2_purify(H)`, `get_density`, `_get_projector` (the topology markers),
  `get_C_gpu` and the RPA bubbles with `purify_method=:sp2`. Models with only position
  sites are unchanged. (The SCF drivers keep their documented `Nel = H0.N ÷ 2` defaults.)
- **`add_superconductivity!`** keeps a KPM scale: `|center| + scale + 1.1‖Δ̂‖` (center 0;
  ‖Δ̂‖ = |Δ| for s-wave, 2|Δ| for p-wave) when `H` had a scale and `Δ` is a number. Its
  update `scale + 1.1|Δ|` stood before the cache invalidation, which reset the scale to 0,
  so every BdG model was left to the DMRG estimate; a function `Δ` or an unset scale still
  are.
- **`get_scf(H, U, :magnetic)`** without `scale` uses `scf_magnetic_hubbard`'s default
  `H0.scale`: `get_scf` passed `scale=nothing` and so estimated every mean-field
  Hamiltonian by DMRG. The other channels are unchanged.
- **`_get_projector(:KPM)`** (the KPM paths of `get_C`, `get_W`, `get_thouless_pump`)
  rebuilds a cached Chebyshev list shorter than `Nchebychev` (it used the short one) and
  expands with `cutoff` (it used 1e-8 whatever `cutoff`).
- **Exciton Hamiltonian.** `exciton_hamiltonian` / `Exciton_Hamiltonian` put `H_c` on the
  hole (even) sites and `−H_v` on the electron (odd) sites, both transposed (see
  `interleave_mpo` below). The MPO is now `(H_c + V) ⊗ I − I ⊗ (H_v − V) + U` on
  `[H_c.sites[1], H_v.sites[1], …]`, as documented. The old operator was the new one with
  the electron and hole registers exchanged (and complex-conjugated for complex `H_c`,
  `H_v`), so:
  - unchanged in exact arithmetic, for any `H_c`, `H_v`, `on_site` and `Ufunc`: the
    spectrum, `get_dos_trace`, and every contact-probe (`|X, X⟩`) result
    (`get_exciton_ldos_spatial`, `get_exciton_ldos`, their GPU twins, the Chebyshev
    convergence check), and `get_exciton_bands` for real models;
  - under truncation the MPO is a different network, so truncated runs agree to the
    truncation error; for the same bipartite nearest-neighbour model on both carriers,
    confined only through `on_site` (the usual set-up), the two are related by a local
    diagonal unitary and agree to rounding;
  - probes with `x_e ≠ x_h` see electron and hole exchanged: the separation LDOS ρ(d, R)
    of `get_exciton_ldos_separation` is the former ρ(−d, R + d), `exciton_radius2` resolves
    the hole position as documented, and the random probes of `get_dos_stochastic` and
    `get_exciton_continuum` give other realizations of the same expectation (with
    `k_list`, k ↔ Q − k); for complex, time-reversal-breaking `H_c` or `H_v` the momentum
    axis is reflected, Q → −Q.
- **`interleave_mpo`** embedded the transpose of every operator whose tensors are stored
  `(s', s)` (most builders): it mapped the site legs in storage order onto `(p, p')`. It now
  reads them by prime level. `conjugate_by_qft_exciton` loses the `swapprime` that undid
  the transpose, and `get_green_krylov` embeds `(z − H)ᵀ` explicitly (both unchanged);
  `swap_every_other_legs` reads the legs by prime level too (its docstring now says it
  swaps the odd sites). **`get_bubble_mpo`, `get_bubble_mpo_haydock`, `get_magnon_bubble`**
  and the susceptibilities built on them returned Π₀ᵀ for complex (time-reversal-breaking)
  Hamiltonians and now return the Lindhard Π₀, like the cheb2d bubbles: the k-resolved
  Wynn χ(q) becomes χ(−q) there. Real models move at the truncation level, and by ~1e-5
  from the slight asymmetry of the purified density at the default tolerance.
  `rpa_from_bubble_diag` solves `(I − Π₀V)x = diag(Π₀)` instead of its transpose (it
  differs for complex Π₀ and for a V that does not commute with Π₀).
- **Non-Hermitian scale**: `scale = 0.0` in the `NonHermitianHamiltonian` methods means
  "not given", like `nothing`: a scale stored on `NH.hermitized` is used, as everywhere
  else in the package (it was re-estimated).
- **Density caches answer only their own method.** `get_density`, `_get_projector` and the
  RPA purification returned any `H._density_cache` whatever method had computed it (a
  McWeeny matrix answered `method=:kpm`). Each stored density is now recorded with its
  method; one set by hand still answers every method. An unknown `method` of
  `get_density` is an error even with a cache.
- **Default KPM scales of the 2D multi-atom lattices** (`get_Hamiltonian` without `scale`)
  are `max(builder formula, estimate_scale(...; method=:small))`, like the presets:
  `"lieb"` 2.5|t| → 2.73|t| at L = 3 and 3.10|t| from 16 × 16 cells (its bulk radius is
  2√2|t|); `"dice"` from 8 × 8 cells (≈ 4.63|t| for large systems); `"honeycomb_nnn"` once
  |t2| is large (t2 = 0.3: radius 4.76 > 4.55). `"kagome"` and `"honeycomb"` never
  change, and direct builder calls (`lieb_hamiltonian(...)`) keep their own formula.
- **`"chern8"`** takes `t2 = 0.2t`, the default of `HChern8`; the registry passed an
  absolute `t2 = 0.2`. Only `t ≠ 1` without an explicit `t2` changes.
- **`ref_sites`** of `get_Hamiltonian` is honoured by `"chain_1d"`, `"haldane"`,
  `"custom"`, `"ssh_sublattice"` and the multi-atom lattices, which ignored it: the `L`
  position qubits of the result are `ref_sites` (a multi-atom lattice keeps its own
  sublattice index). A `ref_sites` of the wrong length or dimension is an
  `ArgumentError`.
- **`get_Hamiltonian("custom", f)`** and **`add_hopping!(H, f(i, j))`** could return a
  wrong MPO for a sparse `f` (a nearest-neighbour chain, a single bond), depending on the
  global RNG: QTCI missed whole bond classes from its default pivots. They now run the
  sampled self-check of `hopping2MPO` (see **Added**); builds that were right are
  unchanged, bit for bit.
- **Honeycomb builders** (`honeycomb_sublattice_hamiltonian`, `honeycomb_nnn_hamiltonian`,
  `"honeycomb"`, `"honeycomb_nnn"`) at `Lx = 2, Ly = 1` with `|t| = 1` carried spurious
  entries of 1e-5: ITensors' density-matrix MPO sum projects on the eigenvectors of a
  nearly degenerate Hermitian eigenproblem, which LAPACK returned non-orthonormal. The
  sublattice builders now check their sum against the exact direct sum of the terms and
  fall back to its SVD truncation; every other size and lattice is unchanged, bit for bit.
- **`sdf_convex_polygon`** with counter-clockwise vertices (the documented order) used the
  outward edge normals and was negative everywhere, so its masks suppressed the whole
  lattice. The orientation is now read from the signed area: positive inside for either
  order; clockwise input is unchanged, collinear vertices are an error.
- **`intrachain_hopping`** (legacy) was not Hermitian even for real `t`: its backward hop
  put the row break on the wrong side, dropping the bond from `ix = Nx − 2` to `Nx − 1`
  and adding a wrap to the previous row's end.
- The `"lieb"` `geometry_uc` (unit-cell positions) is square; it used the triangular
  Bravais vectors of the other multi-atom lattices.

### Fixed

- `get_ldos(mode=:diag)`, `get_ldos_spectrum` and `extract_diagonal_to_mps` on a projected
  position space (Fibonacci, metallic-mean, k-bonacci): the Chebyshev term `T_0`, the
  physical projector, stores its legs as `(s, s')`, so its diagonal came out on the primed
  site indices and the first sum of diagonals threw. Diagonals are now extracted on the
  unprimed index whatever the storage order.
- An empty group in `x_groups`, `X_groups` or `Q_groups` (`spatial_sampling_plan`,
  `get_ldos_spatial`, `get_exciton_ldos_spatial`, `get_exciton_bands`, their GPU twins)
  is a clear error ("every group in x_groups needs at least one position") instead of a
  `BoundsError`.
- `mps_to_diagonal_mpo` accepts a one-site MPS (its GPU twin already did).
- Exciton detection: a one-particle model whose auxiliary indices bring it to `2L` sites
  (a spinful `L = 1` chain) was taken for an exciton register by `get_dos_stochastic`
  (and its GPU twin) and printed as one. Every exciton check now also requires no
  auxiliary index; the exciton entry points reject such a model with their
  "not an exciton Hamiltonian" error.
- `get_ldos_spatial` and `get_ldos_spatial_gpu` with `grid`, a window, `box_half` or
  `reduce=:block` on a T-junction Hamiltonian raise an error: its positions are chains
  drawn in 2D, not a row-major grid, and the maps silently split the register at
  `L ÷ 2`.
- `kernel=:hodc` in a function without `eta`/`m_order` keywords (`get_ldos_spatial`,
  `get_bands`, `get_qpi`, …) still raises "Unknown KPM kernel", now saying which
  functions take it.
- `haydock_cf` threw a `DomainError` for complex Hermitian seeds (e.g. `Y ⊗ I`).
- RPA bubbles with `P_method=:purification, purify_method=:sp2` and `ϵF ≠ 0` raise an
  `ArgumentError`: SP2 fixes the filling at `Nel = H.N ÷ 2`, and ignored `ϵF` silently.
- RPA `P_method=:kpm` and the Green's-function recursion of `get_bubble_mpo` printed
  "Computed T_n …" with `verbose=false`.
- The `get_magnon_bubble` docstring had the spin-flip energy reversed (the code computes
  `ω − (ε↓ − ε↑)`).
- Hermitian for complex parameters: the intra-cell bond of the honeycomb builders, the AA
  interlayer coupling of `bilayer_hamiltonian` / `multilayer_hamiltonian` (its backward
  hop now takes `conj(t_inter)`), the interlayer coupling of `twisted_bilayer_hamiltonian` /
  `twisted_multilayer_hamiltonian` (its backward hop was the transpose, compressed as
  `Float64`: a complex `t_inter` threw), and `interchain_hopping_square` with a complex
  profile.
  Real parameters are unchanged, bit for bit.
- `kagome_hamiltonian`, `lieb_hamiltonian` and `dice_hamiltonian` accept complex
  amplitudes (an `InexactError` before), with `⟨A|H|B⟩ = t_AB` on every bond.
- `mask_hamiltonian` works on sublattice Hamiltonians (kagome, Lieb, honeycomb, dice): each
  atom is masked at its own position (a `DimensionMismatch` before).
- A QTCI field that vanishes identically no longer throws "maxsamplevalue is zero!": its
  term is left out in `H2DChernhex` (e.g. `uniformsemenoff=true, ms=0`), `HUniform(v=0)`,
  `HAAH(V=0)` and `HChern8` without modulation, and `add_onsite!(H, 0)` leaves `H.mpo`
  as it is.
- `add_hopping_2D!` and `get_shell_disps` raise an error on layouts that are not a Bravais
  lattice in the cell index (`"triangular_2d"` with `Ly ≥ 2`, `"hex_2d"`, the layered
  `lattice=:triangular` / `:honeycomb`): their neighbour shells differ between even and odd
  rows, and one shift per displacement put half the bonds in the wrong place.
  `"triangular_bravais"`, `"square_2d"` and the sublattice lattices are unchanged.
- Docstrings: the dice bands reach ±3√2 t (not ±3t) and the Lieb bands ±2√2 t; the
  positional `cyclic=true` default of `build_shift_mpo`, never reachable (a two-argument
  call takes the keyword method, `cyclic=false`), is gone.
- `add_zeeman!` and `add_soc!` failed on every Hamiltonian with a layer or sublattice index
  (bilayers, honeycomb, kagome, …: the term lacked those sites, and the MPO sum threw), and
  `add_soc!` on a BdG model; `add_superconductivity!` did the same, and lifted the pairing
  to the spin on the side of the Nambu index rather than of the spin (a postpended spin
  with `position=:pre` threw). The terms are now built on the site layout of `H.sites`:
  the operator on the spin, τ_z on a Nambu site, the identity on every other auxiliary
  site. Chains are unchanged, bit for bit.
- `scf_magnetic_hubbard_gpu` with a real `type` (`Float64`) switched to complex arithmetic
  after the first Hartree step (its ComplexF32 Hartree deltas); it now stays real.
- `rms_error` and the GPU SCF residual used ITensors' deprecated index matching
  (`inner(ψ', ψ)`, "will error in ITensors v0.4"); same values.
- `nh_spectrum_grid(mode=:diag)` dropped the imaginary part of `Z_spatial`; it is
  `ComplexF64` like `Z` (the real part is unchanged).
- `hermitize(NH)` rebuilt the Hermitian dilation with the default convention and scale,
  so an `:H_minus_z` wrapper silently flipped the sign of its upper block;
  `NonHermitianHamiltonian` records `convention` and `scale` (two new fields, the
  five-argument constructor still works) and `hermitize(NH)` keeps them.
  `hermitized_hamiltonian` reports the side of its block index as `aux_side` (always
  `:pre` before).
- `rk4_step_dm_nh_gpu` and `get_nh_density_trajectory_gpu` rounded `dt/2`, `dt`, `dt/6` to
  Float32 in ComplexF64 runs (~1e-8 per step); ComplexF32 runs are unchanged, bit for bit.
- Docstrings: the `Exciton_Hamiltonian` example called `x -> -U` attractive (the contact
  term is `−Ufunc`, so a positive `Ufunc` attracts); `tol_quantics` and
  `maxbonddim_quantics` apply to `on_site` only.
- A spin index added with `add_spin!(H; position=:post)` (alone or with a postpended Nambu
  index) was always projected from the first site: `get_bands` crashed Julia with a
  segfault, `get_ldos_spatial(mode=:mpo)` threw inside ITensors, and the GPU twins did
  the same. Every auxiliary index is now projected from the side `H.sites` puts it on
  (`aux_site`); the low-level `get_bands` takes the spin's side as the new keyword
  `spin_side` (default `:pre`). `project_aux` and its GPU twin check that the index is the
  site of the end tensor they contract, and throw a clear error instead of returning a
  corrupt MPO (also for a wrong `sublat_side` in the low-level `get_bands`).
- `aux_site(H, :spin)` on BdG models with spin, and `aux_site(H, :layer)` on spinful
  layered models, refused the index as "interior", so every spectral method threw on
  those models; the side is now that of the block of auxiliary sites holding the index.
- `get_bands(H, …)` and `get_bands_gpu` with the default `num_x` failed on 1D systems with
  fewer than 60 momenta (L < 6): the default 60 is clamped to the momenta of the window;
  an explicit `num_x` behaves as before (the keyword's type is now
  `Union{Nothing, Int}`, `nothing` meaning the default).
- The low-level `get_bands` with `nambu_s`, `layer_s` or `sublat_s` given but its
  projection flag off transformed the auxiliary site as one more momentum bit (a
  meaningless result); it raises an `ArgumentError`. The `TBHamiltonian` methods always
  switch those flags on.
- `_project_spin_sector` (the RPA and SCF spin channels) on a spin index inside the
  auxiliary block dropped sites from the MPO: the Nambu site for `[pos…, spin, nambu]`,
  every position site for `[nambu, spin, pos…]`.
- A projection flag without its index (`nambu_proj=true` on a model without Nambu index,
  `spin_proj=true` in `get_ldos_spatial(mode=:mpo)` without spin) threw a `TypeError`; the
  error now names the DOF. `project_aux(W, nothing, σ)` no longer blames `sublat_proj`
  whatever the DOF.

- Example notebooks: `examples/spectral/aux_ldos_examples.ipynb` called
  `TensorBinding.plot_ldos_2d`, which the package does not define (plotting is not part of
  it); the notebook now defines the helper itself. `examples/manybody/excitons.ipynb` used
  an undefined `H_exc_band` for the momentum-resolved spectrum; it is now built in that
  cell, as the confined exciton without its confinement potential.

## [0.1.1] — 2026-09-26

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
