# TensorBinding reorganisation — todo

Outcome of the code-organisation review of 2026-09-23 (four area sweeps over
`lattice/` + `core/Hamiltonian.jl`, `solvers/`, `physics/`, `gpu/`, plus repo-wide
metrics). Items are ticked as they land, with their commits. Tiers are ordered so that
each one can be done and merged on its own with the test suite as the guard; Tier 1
changes no behaviour, Tier 2 changes internals only, Tier 3 is user-visible.

Line numbers refer to the working tree on that date and will drift.

## Bugs (fix first, independently of the reorganisation)

- [x] `core/TBSystem.jl` `_build_haldane` calls `haldane_hoppingf`, which is not defined
      anywhere; `get_Hamiltonian("haldane", …)` throws. Restore the function or drop the model.
      *Fixed in b579377.*
- [x] `physics/RPA_tk.jl:935` `get_bubble_mpo_haydock` calls `_build_heff` with 3 arguments;
      the definition (l.454) takes 4.
      *Fixed in cfe41cf.*
- [x] `physics/RPA_tk.jl` ~1065 and ~1249: `get_rpa_susceptibility_wynn` and
      `get_magnon_susceptibility_wynn` assign `nq` inside `if chi_partial === nothing` inside
      the frequency loop, so the second frequency hits an undefined variable. Hoist `nq`.
      *Fixed in cfe41cf.*
- [x] `physics/Topology_tk.jl` `get_C` accepts `Lambda` (ASCII alias) and never uses it; the
      GPU twin honours it.
      *Fixed in b3bf907.*
- [x] `core/TBSystem.jl` 17-argument `TBHamiltonian` compatibility constructor silently drops
      `Lx`, `interaction_mpo`, `fock_mpo` and `position_space`; SCF (l.364, 485, 1136),
      RPA (l.1141) and NH (l.99) copy Hamiltonians through it. Replace with a keyword copy
      constructor (see Tier 2) and delete the positional ones.
      *Fixed in c87275a, bc4c9b4, cfe41cf.*
- [x] `solvers/KPM_tk.jl` `get_density_quantics` uses an undefined global `sites`. Delete.
      *Fixed in f85750b.*
- [x] `solvers/Timeev_tk.jl` `compare_propagator_and_tdvp_heatmaps` calls `heatmap`/`plot`/
      `display` although Plots is not a dependency. Move to `examples/`.
      *Fixed in f85750b.*
- [x] `gpu/GPU_tk.jl:871` second `_sample_state_amplitudes_gpu` call drops `pointavg`.
      *Fixed in 552ff3f.*
- [x] `gpu/GPU_tk.jl:267` `_onehot_gpu` only accepts `T<:Complex`; the advertised
      `type=Float32/Float64` paths fail.
      *Fixed in 552ff3f.*
- [x] `physics/Purification_tk.jl:20` header example uses `method=:KPM`; code accepts `:kpm`.
      *Fixed in 4738800.*
- [x] `README.md:22` claims CUDA is an installed dependency; `Project.toml` has none.
      *Fixed in 4738800.*
- [x] `core/Utils.jl` `_exciton_block_groups` is reachable only through a branch that
      `get_exciton_ldos_spatial_gpu` rejects earlier (`reduce=:block`). Delete both.
      *Fixed in 552ff3f.*

### Found while fixing the list above (2026-09-25/26)

- [x] `core/Utils.jl` `spatial_sampling_plan` 1D `:point`: `num_x` larger than the window,
      an empty window or `num_avg < 1` returned empty groups (NaN or crashing LDOS columns).
      *Now an error (7c7fe96); every other plan pinned by `test/sampling_golden.jl` (15d66e3).*
- [x] `core/Hamiltonian.jl` `hopping2MPO` passed QTCI initial pivots as a keyword and threw.
      *Fixed in 8ba4742.*
- [x] `solvers/Timeev_tk.jl` `evolve_with_propagator` called the `TBHamiltonian`-only
      `truncate!`. *Fixed in a9aadc2.*
- [x] `physics/RPA_tk.jl` `get_rpa_susceptibility(mode=:magnetic)` mixed spinful and
      spin-projected sites. *Fixed in ce8d1d1.*
- [x] RPA cheb2d bubbles on spinful/sublattice/layer/Nambu registers failed; the MPO variants
      silently truncated qudit (metallic-mean) sites. *Fixed in e5a0d5f, f8daa7f.*
- [x] `_build_haldane`: default scale below the spectral radius for large `t2` (1b4657d); QTCI
      build non-deterministic and often wrong (structural pivots + check, e0285c3).
- [x] Layered builders never set `Lx` (block/grid/box LDOS maps wrong). *Fixed in 4cbfa01;
      clearer `add_hopping!` error on geometry-less layered systems in b453021.*
- [x] GPU: helpers reaching CUDA without `_check_gpu`, stale header, `get_C`/`get_C_gpu`
      signatures (7030de2); `get_bands_gpu` k-grid on aux models (581c7d7); docstrings
      (1d36ff4, 4ce2c2e).
- [x] `build_tdvp_propagator_mpo`: default path threw, then cost O(N) TDVP runs and was 0.1 off
      `exp(-iH dt)`. *Fixed in ee1fb6d..ddb4c71 (no diagonal seeding, reverse_step=true,
      basis expansion, one TDVP run per column).*
- [x] Decision taken: every Haldane construction is the textbook C3-symmetric model.
      `haldane` and `chernhex` had the wrong phase on the vertical NNN bonds (transition at
      √3·t2 instead of 3√3·t2). *Fixed in c1470e3, 2d529b6, test 9277c03.* The manuscript's
      `build_APSOS_hamiltonian` and `get_valley_operator` were already textbook; the manuscript
      model and `haldane` share the φ convention, `chernhex` has the opposite Chern sign.
- [x] Default `chern8` / `qc2dsquare` scales (6|t|) can be below the spectral radius
      (`qc2dsquare` from L≈12). Decided: fix through the Tier 2 scale maker (below).
      *Fixed by the scale maker (tier2/registry): max(6|t|, 1.1 × row-sum bound).*
- [ ] `get_Hamiltonian("custom", f)` and other `hopping2MPO` callers without pivots share the
      QTCI weakness that broke the Haldane build (wrong MPO for sparse `f`, seed-dependent).
- [ ] `H2DChernhex` (and `add_onsite!(H, 0.0)`) throw "maxsamplevalue is zero!" when a QTCI
      field is identically zero (e.g. `uniformsemenoff=true, ms=0`).
- [x] `get_ldos_spatial(_gpu)` grid/block/box maps on 1D systems drawn in 2D (T-junction) use
      `Lx = H.L ÷ 2` silently; `tjunction_lattice_hamiltonian` has no meaningful `Lx`.
      *They raise an error on T-junctions (`_is_tjunction`, the "TJunction" branch index),
      87f6562. Haldane and custom 2D models keep the `L ÷ 2` fallback, the split of
      `honeycomb_positions`' default layout.*
- [ ] `scf_magnetic_hubbard_gpu` with a real `type` silently switches to complex arithmetic
      after the first Hartree step.
- [x] `ilinspace(xmin, xmax, 1)` returns `[0]` even when `xmin > 0`; 2D `kspace_sampling_plan`
      asserts whenever `xmin > 0` or `xmax < 2^Lx - 1` (both pinned by the golden test).
      *Fixed in 87f6562: `[xmin]`, and the 2D plan places `min(num_x, window)` points.*
- [ ] `get_C`/`get_C_gpu` cannot detect `Λ` and `Lambda` passed together; fold into Tier 3.
- [x] `examples/manybody/excitons.ipynb` cell 5 uses an undefined `H_exc_band`.
      *The cell builds it: the chain exciton of `H_exc` without its confinement potential.*

### Found by the Tier 1 characterization sweep (2026-09-26; pinned as-is, not fixed)

The golden tests pin today's behaviour, these bugs included; fixing one means regenerating
the affected golden cases in the same commit.

**Silently wrong results**
- [x] `get_density_from_Tn` expands θ(x − μ), the projector onto the EMPTY states (verified
      against exact diagonalisation: ‖ρ − θ(H − ϵF)‖ = 0.004). Affects `get_density(:kpm)`,
      `_get_projector(:KPM)` (Chern/winding markers flip sign vs `:mcweeny`) and RPA
      `P_method=:kpm`. The manuscript scripts use `:mcweeny` and are unaffected.
      *Fixed in v0.1.1 (release-0.1.1, 5ef5b2d) and ported to `solvers/kpm/cached.jl` in the
      merge of release-0.1.1 into Anouar: c₀ = 1 − acos(μ)/π, cₙ = −2 sin(n acos μ)/(nπ);
      test/bugfix_density.jl. Regenerated golden cases: kpm 5, scftopo 11, rpa 42, gpu 1 (the
      merge commit lists them). The winding marker σ_z(PxQ + QxP) is symmetric under P ↔ Q,
      so `get_W(:KPM)` did not change; only the Chern marker changes sign (the pinned
      Hofstadter case, whose marker is ~0, moved by 4e-11).*
- [x] `chebyshev2d_gf_coeffs` is 4× too small (divides by (2N)²); all cheb2d bubbles inherit it.
      *Fixed in ead1d66 (divides by N²). Nothing compensated it: bugfix_rpacheb2d's dense
      reference used the same coefficients.*
- [ ] `exciton_hamiltonian`/`Exciton_Hamiltonian` put `H_c` on the hole sites and `−H_v` on the
      electron sites (`interleave_mpo(..., 0)` targets even sites).
- [x] 2D `kspace_sampling_plan` pairs `xcenters[i]` with `ycenters[i]`: a 2D k-grid samples only
      the kx = ky diagonal.
      *By design: its docstring calls it the legacy diagonal cut (`kpath_2d` gives paths; a
      full 2D grid would change the output shape, Tier 3). The bug inside it is fixed
      (87f6562): it took the first `num_x` points of a full-resolution grid, so
      `num_x = 4` on a 16 × 16 zone sampled k = 0…3; the points now span the cut, which on
      odd-`L` registers follows the physical kx = ky line.*
- [x] `_estimate_scale("aah")` = 1.2(|t|+|V|) is below the AAH spectral radius (→ 2|t|+|V|).
      *The default is now max(that, 1.1 × dense radius at L ≤ 10) (tier2/registry); the
      formula itself stays in `_estimate_scale` (golden-pinned).*
- [ ] `honeycomb_sublattice_hamiltonian`/`honeycomb_nnn_hamiltonian` (and the `"honeycomb"`,
      `"honeycomb_nnn"` presets) are ~1e-6 off after compression at cutoff 1e-8 (spurious entries).
- [ ] `get_C`/`get_C_gpu` on multi-atom unit cells return O(0.1) imaginary local markers.
- [ ] Not Hermitian for complex parameters: honeycomb sublattice intra-cell term, AA-stacked
      bilayer `t_inter`, legacy `intrachain_hopping` / `interchain_hopping_*`.
- [ ] `add_hopping_2D!` shells are wrong on non-Bravais layouts (`triangular_2d`, brick `hex_2d`).
- [ ] `sdf_convex_polygon` has its sign flipped (polygon masks are inverted).
- [x] RPA: `ϵF` never reaches the purification density (always half filling); `haydock_cf` uses
      tr(conj(A)B) instead of tr(A†B).
      *Fixed in ead1d66: McWeeny starts from `H.center + ϵF` (the `mcweeny_purify` level), SP2
      refuses `ϵF ≠ 0`; `haydock_cf` takes the exact Frobenius product `inner(A, B)`.*
- [ ] SP2: diverges to NaN near convergence; default `Nel = H.N ÷ 2` counts unit cells, not
      states (quarter filling on sublattice/spin/BdG models).
- [x] `get_ldos(mode=:mps)` scales with `norm(psi0)` for unnormalised probes.
      *Fixed in 87f6562: the moments take `ψ₀/‖ψ₀‖`, like the cache `KPM_Tn_mps` builds.*
- [ ] NH: `nh_spectrum_grid(mode=:diag)` drops the imaginary part of `Z_spatial`; rebuilding via
      `hermitize(NH)` forgets the convention and scale; `hermitized_hamiltonian` reports
      `aux_side=:pre` and the parent's `L`/`N`.
- [ ] `rk4_step_dm_nh_gpu` casts `dt/2`, `dt`, `dt/6` to Float32 even for ComplexF64 (~1e-8 error).
- [ ] `fix_sites` transposes MPOs stored ket-first; `interleave_mpo` embeds `transpose(op)`
      (see the memory note on interleave_mpo).
      *`fix_sites` fixed in 87f6562 (legs by prime level, `_mpo_site_pair`); the exciton QFT
      conjugation is unchanged by it. `interleave_mpo` still open.*

**Crashes and unhelpful errors**
- [ ] SEGFAULT: `get_bands` on a postpended spin (`add_spin!(...; position=:post)`) or sublattice
      index — `project_aux` hard-codes `side=:pre` and never checks the index is on the tensor.
      Same `:pre` hard-coding in `get_ldos_spatial(mode=:mpo)`.
- [ ] `get_bands(H)` default `num_x=60` fails for 1D systems with L < 6; low-level `get_bands`
      with `sublat_s` but `sublat_proj=false` silently transforms the sublattice index.
- [ ] `aux_site(H, :spin)` errors on every BdG+spin model.
- [x] `get_ldos(:diag)` and `get_ldos_spectrum` throw on Fibonacci (projector leg order).
      *Fixed in 87f6562: `extract_diagonal_to_mps` takes the unprimed leg whatever the order.*
- [x] Empty-group checks in `get_exciton_ldos_spatial`/`get_exciton_bands` are unreachable
      (BoundsError first); `mps_to_diagonal_mpo` fails on a 1-site MPS.
      *Fixed in 87f6562: `spatial_sampling_plan` rejects empty groups (the callers' dead
      checks are gone); `mps_to_diagonal_mpo` takes a one-site MPS.*
- [ ] `mask_hamiltonian` fails on sublattice Hamiltonians; kagome/Lieb/dice reject complex `t`.
- [x] `haydock_cf` throws DomainError for complex Hermitian seeds.
      *Fixed in ead1d66 (same change).*

**Minor / API**
- [ ] Method symbols: `_get_projector`, `get_C`, `get_W`, `get_thouless_pump` accept only `:KPM`,
      `get_density`/`get_scf` only `:kpm` (Tier 3).
- [ ] `add_superconductivity!`'s scale update is dead (`_invalidate_cache!` resets it);
      `get_scf` passes `scale=nothing`, overriding `scf_magnetic_hubbard`'s default.
- [ ] `rms_error`/`_rms_error_gpu` and several `inner` calls rely on ITensors' deprecated index
      matching ("will error in ITensors v0.4").
- [x] `wynn_epsilon` returns the 1e30 sentinel for exactly converged sequences.
      *Fixed in ead1d66: 1/(∞ − ∞) is taken as 0; singular tables keep the sentinel.*
- [ ] `build_shift_mpo(sites, q)` positional `cyclic=true` default is unreachable.
- [ ] `MODEL_REGISTRY["chern8"]` uses absolute `t2=0.2` (HChern8 defaults to 0.2t); Lieb's
      `geometry_uc` uses a triangular basis; `get_Hamiltonian` silently ignores `ref_sites` for
      several geometries.
- [ ] `_nh_resolve_scale`: `scale=0.0` and `scale=nothing` mean different things.
- [x] `scf_magnetic_hubbard_gpu` warns about ComplexF32 even with ComplexF64.
      *Fixed with the shared GPU warning (tier2/gpuwrap): its extra `cutoff < 1e-5` warning for
      every type is deleted; it now warns for a 32-bit type below 1e-5, its old 32-bit range.*
- [x] `get_dos_stochastic` detects excitons by `length(H.sites) == 2H.L` (misfires at L = 1).
      *Fixed in 87f6562: `_is_exciton_register` (2L sites and no auxiliary index) behind every
      exciton check; any aux index could bring a one-particle model to 2L sites.*
- [x] `_kpm_weight_matrix` rejects `:hodc` while `_dos_weight_matrix` accepts it.
      *By design: the functions behind `_kpm_weight_matrix` have no `eta`/`m_order` keywords
      and document the four damping kernels. The error now says which functions take
      `:hodc` (87f6562).*
- [ ] Docstrings: `get_rpa_susceptibility_wynn` (π), exciton interaction sign convention,
      `project_aux` error message names sublattice for every aux index.
      *The π was already right (4cf039f: "no 1/π factor", as the code). The
      `get_magnon_bubble` docstring had the spin-flip energy reversed (fixed in ead1d66).*

### Found by the Tier 2 scale maker (2026-09-26; not fixed, decision needed)

- [ ] `"lieb"` default scale 2.5|t| is below the bulk spectral radius 2√2|t| (2.82 at
      Lx = Ly = 4); `"honeycomb_nnn"` 3.5(|t| + |t2|) is below 3|t| + 6|t2| once
      |t2| > 0.2|t| (t2 = 0.3, Lx = 5, Ly = 4: radius 4.76 > 4.55). The multi-atom lattices
      keep their builder defaults because the max rule would also move the pinned
      `lieb_L3_*` golden cases, whose radius (2.48) is below 2.5 but above 2.5/1.1.
      `scale=:small` / `:geometry` give a bounding scale today.
- [ ] `dice_hamiltonian` docstring says the bands reach ±3t; they reach ±3√2 t (the
      4.5|t| default still bounds them).

### Found by the Tier 2 KPM kernels (2026-09-26; not fixed, decision needed)

- [ ] `_jackson_kernel(N)` (rpa/cheb2d.jl, the `kernel=:jackson` option of the SVD/Tucker
      cheb2d bubbles) has `(N − m)` where the Jackson kernel for N moments has `(N − m + 1)`:
      it is the textbook g_m minus `cos(πm/(N+1))/(N+1)`, so g_0 = N/(N+1) instead of 1
      (max deviation 1/(N+1): 0.1 at N = 9, 0.0066 at N = 151). Fixing it moves the pinned
      `jackson_kernel_*` and low-rank cheb2d golden cases.
      *Fixed in ead1d66: `_kpm_kernel(N + 1, :jackson)[1:N] ./ (N + 1)`; `_jackson_kernel` deleted.*
- [x] `get_qpi` accepts projected position spaces but is binary-only (the impurity sits
      at the binary address `x0 − 1`, the QFT is over the binary register). With
      `physical_projector` as T₀ a Fibonacci call now throws in the diagonal accumulation
      (the projector leg-order bug listed under "Crashes"); before, it returned maps that
      included the unphysical register states. A `_require_binary_position_space` guard
      would give a clear error.
      *Done in 492fac4 (Tier 2): `get_qpi` raises that `ArgumentError`.*
- [ ] RPA bubbles on projected spaces: the density (`P_method=:kpm`) now has an empty
      unphysical block, but `_build_heff`, the numerator and the 2L-site Green's function
      (`KPM_Tn(Heff, …, sites_combined)`) still use ambient identities. On an L = 4
      Fibonacci chain the physical block of `get_bubble_mpo` is the same before and after
      the switch without truncation (3e-12); at `maxdim = 30` both are ~30 % off that
      converged value and differ from each other by ~19 %, so these bubbles need a
      convergence check in `maxdim` on projected spaces.
      *Not a correctness bug (2026-09-28 check): H, ρ and the identity are block-diagonal in
      physical ⊕ unphysical, so the collapsed physical block is exact (0.0 difference on an
      L = 4 Fibonacci chain without truncation). It is an accuracy cost: 92 % of ‖G·N‖ sits
      outside P⊗P and is discarded, but uses bond dimension. The fix, bit for bit on binary
      spaces (`physical_projector` = identity there): projectors in `_build_heff`, in the
      bubble numerators and seeds, and T₀ = P₁⊗P₂ for the 2L-site KPM, then a `maxdim`
      convergence study. Left open as an improvement.*

### Found by the Tier 2 aux and density kernels (2026-09-26; not fixed, decision needed)

- [ ] `_project_spin_sector` on a spin site inside the MPO (`add_spin!(H; position=:post)`
      followed by `add_superconductivity!(H, Δ; position=:post)`, sites `[pos…, spin, nambu]`)
      drops every site after the spin site from the MPO, while the returned `sites` keep
      the Nambu index; `_project_aux_block` keeps them. Kept as one explicit line in
      `_project_spin_sector`.
- [ ] The density helpers differ from `get_density` in more than their method symbols:
      `_get_projector(:KPM)` expands any cached Chebyshev list, also one shorter than
      `Nchebychev`, and with cutoff 1e-8 whatever its `cutoff`; its `:sp2` runs 40
      iterations where `get_density` runs 30; RPA `P_method=:kpm` builds a fresh uncached
      list on every call and prints "Computed T_n …" (the raw `KPM_Tn`'s `verbose=true`
      default) even with `verbose=false`; `get_density` checks the density cache before the
      method (a cached McWeeny matrix answers `method=:kpm`), the helpers for purification
      only. All kept, as keyword choices of the shared dispatcher `_density_matrix`.
- [ ] The projection chain takes the Nambu and spin sectors as `1:2`, the probe loops the
      Nambu sectors as `1:dim(nambu index)` and the spin sectors as `1:2`: the same for every
      Nambu index the package builds (dimension 2); kept as they were.

### Found by the export-list checks (2026-09-26; not fixed)

- [x] `examples/spectral/aux_ldos_examples.ipynb` calls `TensorBinding.plot_ldos_2d`, which the
      package does not define (the notebook's stored output already shows the UndefVarError).
      *The notebook defines the helper in its second cell (a copy of the untracked
      plotting helper's); the stale error output is cleared.*
- [x] `Arpack` is a declared dependency (Project.toml `[deps]` and `[compat]`) that `src/` never
      uses; dropping it would remove a dependency (a Project.toml change, fine in any release).
      *Dropped from `[deps]` and `[compat]`, and from the packages `test/exports.jl` checks.*

### Found by the bug pass (2026-09-28)

- [ ] Sign: for real H the cheb2d bubbles are −1 × `get_bubble_mpo` (their D_mn carries the
      numerator P₁⊗I − I⊗P₂, `get_bubble_mpo` has I⊗P₂ − P₁⊗I). Both are documented as Π₀
      and feed the same Dyson/Wynn drivers, whose (I − Π₀V)χ = Π₀ is the Stoner form of
      `get_bubble_mpo`'s sign; cheb2d's sign is the physical retarded response, which the
      untracked Ward-conductivity script relies on. Decision for the author: which sign
      Π₀ has, and whether the cheb2d bubbles flip.
- [ ] cheb2d bubbles on complex H take the Hadamard product P_a ⊙ P_b where the Lindhard
      bubble has P_aᵀ ⊙ P_b: 44 % off on a complex L = 2 chain, and ‖Σ_j Π_ij‖ = 0.34 for
      ‖Π‖ = 0.62 (particle number not conserved).
- [ ] `rpa_from_bubble_diag` does not return χ: its output is rank one,
      x[(i,j)] = [(Aᵀ)⁻¹ diag Π₀]_j with A = I − Π₀V (behind `get_rpa_susceptibility` and
      `get_magnon_susceptibility`).
- [ ] `get_green_krylov` (`get_bubble_mpo(GF_method=:krylov)`) is ~48 % off on the real
      2L-site Heff whatever the sweeps; exact on L-site H and on complex Heff.

## Tier 1 — mechanical, no behaviour change

### Split the three grab-bag files
- [x] `solvers/KPM_tk.jl` (2067 lines) → `solvers/kpm/kernels.jl` (`_kpm_kernel`,
      `_dos_weight_matrix`, HODC helpers, `_kpm_weight_matrix` from QFT), `recursion.jl`
      (`KPM_Tn`, `KPM_Tn_mps`, `_run_kpm_mps!`), `cached.jl` (`get_ldos`, `get_ldos_spectrum`,
      `*_from_Tn`, `*_from_mun`, Green's functions), `ldos.jl` (`get_ldos_online`,
      `get_ldos_spatial`, split into `_ldos_spatial_mps`/`_ldos_spatial_mpo`), `dos.jl`
      (`get_dos_stochastic`, `get_dos_trace`), `exciton.jl` (l.1648–1984).
      *Split into `solvers/kpm/` (2acd149); `_kpm_weight_matrix` and `_reconstruct_ldos_moment_columns` moved into `kpm/kernels.jl` (c630c9c). `get_ldos_spatial` is not split into `_ldos_spatial_mps`/`_mpo` (not a pure move; Tier 2).*
- [x] `physics/RPA_tk.jl` (2148 lines) → `physics/rpa/Bubble.jl`, `Cheb2D.jl`, `Dyson.jl`;
      MPO kron/interleave plumbing (l.10–287) → `core/Utils.jl`; Haydock recursion →
      `solvers/Krylov_tk.jl`; `get_spect_k` → QFT conjugation file; delete l.288–377.
      *Split into `physics/rpa/` (a9bacc0); plumbing → `core/MPOTools.jl` (6ccf897), Haydock → `solvers/Krylov.jl` and `get_spect_k` → `qft/conjugation.jl` (c630c9c).*
- [x] `physics/QFT_tk.jl` (1643 lines) → `Conjugation.jl` (l.96–205), `Bands.jl`
      (l.638–948, 1223–1370), `KPath.jl` (l.426–635); exciton spectra (l.206–281, 949–1220)
      → exciton folder; aux projection (l.1373–1505) → `core/AuxDOF.jl`.
      *Split into `physics/qft/` (c5b4353); aux projection → `core/AuxDOF.jl` (e36e505).*
- [x] `physics/NH_tk.jl` → `NH_model.jl` (struct, `hermitize`, `add_nh_*`) and `NH_KPM.jl`.
      *Split into `physics/nh/model.jl` and `nh/kpm.jl` (9407bf0).*
- [x] `lattice/2Dlattice_tk.jl` (1615 lines) → `Masks2D.jl`, `Hopping2D.jl`, `Presets.jl`
      (QTCI `H*` builders incl. the 1D `HUniform`/`HSSH`/`HAAH`), `Sublattice.jl`
      (kagome/lieb/dice/honeycomb), `Geometry.jl`; `MODEL_REGISTRY`/`build_hamiltonian` →
      `core/ModelRegistry.jl`.
      *Split into `lattice/{masks2d,hopping2d,presets,sublattice}.jl` (a4dabbd); geometry → `lattice/geometry.jl`, registry → `core/ModelRegistry.jl` (a853263).*
- [x] `gpu/GPU_tk.jl` (3647 lines) → `device.jl`, `primitives.jl`, `kpm.jl`, `bands.jl`,
      `topology.jl`, `purification.jl`, `scf.jl`, `exciton.jl`, `nh.jl`, `timeev.jl`;
      the conductivity-only Tucker/QFT/Hadamard block (~300 lines) → its example.
      *Split into eleven `gpu/*.jl` files (8ba22a5); the conductivity block stays as `gpu/conductivity.jl` (an untracked script uses it).*

### Move misplaced helpers next to their callers
- [x] One `core/AuxDOF.jl` owning spin/Nambu indices and op tables, `prepend_spin`/`prepend_nambu`,
      Symbol overloads of `prepend_op`/`postpend_op` (from `Supercond_tk.jl`), `project_aux`,
      `aux_site`, `_autoenable_proj` (from QFT), `_aux_setup`, `_ldos_make_psi0` (from KPM),
      and the four `add_spin!`/`add_zeeman!`/`add_superconductivity!`/`add_soc!` mutators
      (from TBSystem). Include it right after TBSystem. (moved in tier1/move-auxdof)
- [x] `_estimate_spectral_bounds` → `solvers/DMRG_tk.jl`; include DMRG before KPM. (moved in tier1/move-solvers)
- [x] `_eval_diag_mps` → `core/Utils.jl` beside `eval_mps`; `mpsexciton` → Utils beside the
      other product-state builders. (moved in tier1/move-utils)
- [x] `qtt_mpo`, `compose_power`, `_row_break/_row_select/_col_select/_row_checker_mpo`,
      `_site_projector_mpo`, `sigma_d/sigma_u` ops, layer prepend helpers → `core/Utils.jl`
      (or `lattice/Masks2D.jl` for the masks). (moved in tier1/move-utils)
- [x] BdG/pairing builders in `SCF_tk.jl` (l.298–528) → AuxDOF / Supercond.
      *Reviewed (e36e505): only the generic `_project_aux_block` moved to AuxDOF; the BdG/pairing builders use SCF state (µ, Hartree terms, `_split_spin_channels`) and stay in `physics/SCF.jl`.*
- [x] `_project_spin_sector` (RPA) → AuxDOF as `project_sector(H, :spin, σ)`.
      (moved in tier1/move-auxdof; name kept: the rename is left to Tier 2's `_project_aux_sectors`)
- [x] All geometry (`*_positions`, `_*_geometry`, `lattice_positions`, `_resolve_2d_geometry`,
      junction geometry, `geometry_uc` closures) → `lattice/Geometry.jl` with one `(Lx, Ly)`
      signature. (moved in tier1/move-geometry) Signatures unchanged (the one `(Lx, Ly)`
      signature is Tier 3); the `geometry_uc` closures stay inline in their builders.
- [x] `_reconstruct_ldos_moment_columns` (GPU) → `solvers/kpm/kernels.jl`; move its test out of
      `test/gpu_mps_ldos.jl`. (moved in tier1/move-solvers)

### Delete dead and legacy code
- [x] Confirmed unreferenced everywhere (incl. notebooks and generated docs):
      `build_cyclic_shift_mpo`, `_geom_n_sub`, `_nsublat`, `nsitelegs`, `_tb_spatial_groups_gpu`,
      `get_nh_state_trajectory_gpu`, the `Delta_*` one-liners in SCF.
      *Deleted in 9dbee85, fb8b2d8, 28f611f, d1b680c. The `Delta_*` "one-liners" are formulas
      in the `scf_*` docstrings, not functions; nothing to delete.*
- [x] Unreferenced in src/test/tracked examples: `projop_2DSL`, `projop_1DSL`, `sample_diag`,
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
      *Deleted in 9dbee85 (core), fb8b2d8 (lattice), c91ff65 (KPM), 44a79c2 (QFT), 28f611f (RPA),
      fbc5a94 (TwoParticle), 6d79888 (NH), d1b680c (GPU); `get_density_quantics` was already
      gone (f85750b). Kept, because they are used: `sdf_interval` (QPI_tk.jl apodization
      window), `qtci_matrix_to_MPO` (test/bugfix_pivots.jl), `get_bubble_mpo_haydock` (fixed
      in cfe41cf; test/bugfix_rpa.jl, golden_rpa), `fock_exchange_builder`
      (examples/manybody/scf_examples.ipynb, golden_scftopo). Not decided yet: the four open
      items below. Modified_GPU_funcs.jl only defines its own copies of
      `_onehot_gpu_f32`/`nh_spectrum_grid_gpu` and is included nowhere.*
- [x] Decide on `get_valley_projectors` (Topology_tk.jl): no library code calls it, but
      golden_scftopo pins it without a skip-on-delete rule, so deleting it means deleting its
      scftopo case in the same commit.
      *Decided 2026-09-26: keep (may be useful later).*
- [x] Decide on `initial_guess_trivial_up_1D` / `initial_guess_trivial_down_1D` (SCF_tk.jl):
      unused by the library, pinned by golden_scftopo (same situation).
      *Decided 2026-09-26: keep (may be useful later).*
- [x] Decide on `spin_hamiltonian` (Supercond_tk.jl): unused by the library, pinned by
      golden_scftopo (same situation).
      *Decided 2026-09-26: keep (may be useful later).*
- [x] Decide on `bdg_hamiltonian` (Supercond_tk.jl): unused by the library (TBSystem builds
      the BdG MPO inline), pinned by golden_scftopo (same situation); its docstring example
      was fixed for v0.1.1 and the `pairingNNN`/`pairing2MPO` docstrings point to it.
      *Decided 2026-09-26: keep (may be useful later).*
- [x] Commented-out legacy: `QFT_tk.jl:1511–1643` (old `get_bands`, `get_spect_k*`),
      `Purification_tk.jl:95–96`, unreachable code after early `return` in
      `2Dlattice_tk.jl` (`generate_kin_u/d` l.33–63, six kinetic builders l.388–543).
      *Removed in 3291292.*
- [x] Six positional "backward-compatible" `TBHamiltonian` constructors (TBSystem l.98–116,
      190–214) once Tier 2 keyword constructor exists.
      *Deleted in tier2/ctor, with the keyword constructor (Tier 2 below).*
- [x] Unconditional `println` in library code (~70 in src): `Hamiltonian.jl` 85–123,
      `KPM_tk.jl` 14/30/31, `QFT_tk.jl` 1453–1470, `Topology_tk.jl` 499–539,
      `TBSystem.jl` 1175, RPA legacy pipeline; switch to `@info … maxlog=1` or `verbose` gates.
      *Done except QFT: f6a29c4 made the progress chatter in Hamiltonian.jl and
      Topology_tk.jl (no flag there) `@debug` and the spinless s-wave → p-wave notice in
      TBSystem.jl `@info`; in KPM_tk.jl the "estimating…" line is gone and the DMRG estimate
      of the spectral bounds is one `@info` record (56b3787: it is the only sign that an
      automatic scale was chosen); the RPA legacy prints went with the pipeline (28f611f).
      Every other `println` in src is behind `verbose`/`printinfo`, is the point of its
      function (`get_shell_disps`, `check_tdvp_vs_U_mpo`) or is a `show` method.
      Still open: the `QFT_tk.jl` `_autoenable_proj` "Info: … auto-enabling …" lines go to
      stdout, and test/golden_qft.jl pins them in the captured `stdout` of 20 cases, so
      switching them to `@info` means regenerating those fields in the same commit.*
      *Progress prints → `@debug` (f6a29c4), DMRG scale report → `@info` (56b3787), `_autoenable_proj` → `@info` (4cf039f).*

### Make the structure legible
- [x] Explicit `export` list (today only ITensors names are exported) so public vs private is visible.
      (done in tier1/exports) 58 entry points in 12 groups next to the six ITensors names, listed
      under "Public API" in `docs/src/index.md`. Only names specific to TensorBinding are
      exported: generic names (`truncate!`, `hermitize`, `get_matrix`, `get_density`,
      `add_loss!`, …), the lattice `*_hamiltonian` builders, types and constants, and the
      names Tier 3 renames (`get_C`, `get_W`, `get_C_gpu`, `get_valley_C`, `hopping2MPO`,
      `Exciton_Hamiltonian`, the magnon functions) stay qualified. So does `get_ldos`: an
      untracked plotting helper (examples/nontracked/plotting_helpers) that notebooks include
      next to the package defines a different top-level `get_ldos`. `test/exports.jl` checks
      the list against the exports of the dependencies and standard libraries, and against
      the docs list. Left for the user: the example notebooks and manuscript scripts load the
      source with `include(...); using .TensorBinding`, and re-running that setup in one
      session makes the exported names ambiguous (see "Public API"). They call everything
      qualified, so nothing breaks today; moving them to `using TensorBinding` would remove
      the trap.
- [x] One banner style (`# ====` vs `# ───` vs none); numbered sections that match contents
      (2Dlattice runs 8, 8b, 8c, 8d, 8f; SCF header lists 8 sections, file has 9).
      *One `# ====` banner style with matching section numbers (7582f6c, 7f2c568, aef25fc, c172d19, 4cf039f).*
- [x] Rewrite the load-order comment in `TensorBinding.jl` as a real dependency graph.
      (done in tier1/rename) One entry per included file: what it holds and the files it calls
      into, derived from the code (every package-defined name each file uses, plus ITensors op
      names and the builders `build_hamiltonian` looks up by Symbol); `*` marks a call into a
      file included later.
- [x] Fix the include order where a file calls into a later one. Of the original list, KPM ↔ QFT
      and Krylov → RPA are gone (the shared helpers moved to `solvers/kpm/kernels.jl` and
      `solvers/Krylov.jl`) and TBSystem → Supercond is now AuxDOF → Supercond. The map in
      `TensorBinding.jl` shows what is left: Utils → Fibonacci; TBSystem → position_spaces/,
      geometry, ModelRegistry, sublattice, NNNeighbor (the `get_Hamiltonian` builders); AuxDOF →
      hopping2d, Supercond; position_spaces/ → geometry; Bilayer → Twisted; SCF → Purification,
      Supercond; rpa/bubble, Topology → Purification; rpa/cheb2d, rpa/dyson → qft/conjugation;
      qft/bands → qft/kpath. Several are cycles (TBSystem ↔ the lattice builders), so not every
      one can be fixed by reordering.
      *Reordered: geometry before the position spaces, Twisted before Bilayer, Purification
      and Supercond before SCF, the qft/ files before rpa/, kpath before bands. What is left
      is cyclic and cannot be fixed by ordering: Utils → Fibonacci, TBSystem ↔ the
      `get_Hamiltonian` builders, AuxDOF ↔ hopping2d/Supercond (marked with `*` in the map).*
- [x] File names: drop the `_tk` suffix; rename `2Dlattice_tk.jl`; fix header comments that
      cite files that do not exist (`utils.jl`, `2D_lattice.jl`, `twoparticle_tk.jl`, `krylov_tk.jl`).
      (renamed in 9f59782, comments in the next commit; tier1/rename) `lattice/{NNNeighbor,Flake,
      Bilayer,Twisted,TJunction}.jl`, `solvers/{DMRG,Krylov,Timeev}.jl`, `physics/{SCF,Topology,
      Purification,TwoParticle,QPI,Supercond}.jl`; `2Dlattice_tk.jl` had already been split into
      `lattice/{masks2d,hopping2d,presets,sublattice,geometry}.jl` and `core/ModelRegistry.jl`.
      Notes that record where code came from now say "the former …_tk.jl". Left as they are: the
      `"NH_tk model-building helpers …"` error message in `physics/nh/model.jl` (a string that
      test/data/nh_golden.jl pins, not a comment) and the old file names elsewhere in this checklist.
- [x] Re-save `2Dlattice_tk.jl` as UTF-8 and restore the mojibake symbols (√, ·, ≠ appear as
      `-`/`_`, e.g. `b=(1+-)/2` for the golden ratio).
      *Done in the lattice split (a4dabbd).*
- [x] Docstrings vs signatures: `get_ldos_spatial` omits 9 kwargs; `get_ldos_from_mun` omits
      `eta`/`m_order`; Bilayer/Twisted claim `(MPO, sites)` returns but return `TBHamiltonian`;
      Flake/TBSystem examples pass `Lx=16`/`32` where `Lx` is a qubit count; `get_Hamiltonian`
      table lists 8 of 21 names; QFT table of contents (l.76–92) wrong in five places;
      Topology header lists `berry_curvature_integrand`, which does not exist.
      *Docstring signatures checked against the code in every area (7582f6c, 7f2c568, aef25fc, c172d19, 4cf039f, 20c085c).*
- [x] Tests: lattice builders, RPA, SCF, NH, Topology have no tests; add smoke tests before
      splitting so the moves are guarded.
      *Golden characterization tests for eight areas (12acc5a..78a15a8, f3ebab3; gaps closed in d7bff9d, ac73a8e, e4a5682).*

## Tier 2 — shared kernels (internal behaviour only)

- [x] `_scaled_hamiltonian(H; cutoff)` = `(1/scale)·(H − center·physical_projector(H))`,
      replacing ~20 inline copies (some use `MPO(sites,"Id")` and mishandle projected spaces:
      `KPM_tk.jl` 1799, 1917, 1675; `QPI_tk.jl` 155).
      *`solvers/kpm/recursion.jl` §1 (tier2/kpmkernels): a raw-MPO method
      `(H_mpo, scale, center, identity; cutoff)` and a `TBHamiltonian` method (identity =
      `physical_projector(H)` unless passed), both `(1 / scale) * +(H, (-center)·I; cutoff)`,
      the only form in use; no `maxdim` option (no site truncates the shift by bond
      dimension). 20 call sites (KPM_Tn(_mps), ldos, dos, exciton, qft/bands,
      exciton_spectra, QPI, gpu/kpm ×4, gpu/bands, gpu/exciton ×2); binary outputs bit for
      bit unchanged. Switched from `MPO(H.sites, "Id")` to `physical_projector`: the CPU and
      GPU exciton LDOS, `get_exciton_bands/continuum`, `get_qpi`, `get_bands_gpu`,
      `get_ldos_spatial_gpu`, `get_dos_stochastic_gpu`, and (as `identity_mpo` of the raw
      `KPM_Tn`) the RPA `_get_density_matrix(:kpm)` and `_cheb2d_setup`. Reachable with a
      projected space: only `get_qpi` and the RPA bubbles with `P_method=:kpm` (changelog);
      the others are behind `_require_binary_position_space` or need a 2L-site exciton
      register, which only binary spaces build. Left on the ambient identity: the raw-MPO
      `get_bands` and `KPM_Tn_gpu` (no position space to ask) and the NH recursions
      (`A = Hh.mpo / scale`, no centre, a division: a different formula; hermitized
      Hamiltonians are binary-only).*
- [x] `chebyshev_foreach(f!, H̃, T₀; maxdim, cutoff)` working for MPO and MPS on any device,
      replacing ~22 hand-written three-term loops (6 KPM, 14 GPU, QFT, QPI) and 5 NH partial
      recurrences; one truncation policy.
      *`solvers/kpm/recursion.jl` §2 (tier2/cheb): `chebyshev_foreach(f!, H̃, T₀, N; maxdim,
      cutoff, T1, apply_trunc, add_trunc, post_trunc, two, negone)` calls `f!(n, T_n)` for
      n = 0…N−1 (T₀ and T₁ always, as the loops did); each later term is one
      `_chebyshev_step`. Not one truncation policy, which would move results: the loops
      differ in which of `cutoff`/`maxdim` the product `apply(H̃, T)`, the sum and an extra
      `truncate!` receive (five combinations), in the factor (`2`, `2.0`, GPU-typed `T(2)`)
      and in `-T` vs `T(-1) * T`. Each is a keyword (the docstring tables them per caller),
      so every output is bit for bit unchanged: checked old vs new, tensor by tensor, on
      51 cases at small `maxdim` (CPU, and GPU in ComplexF64/ComplexF32/Float64;
      `get_qpi` with the RNG seeded, since the scale of its impurity Hamiltonian is a
      DMRG estimate from a random start and differs run to run). Routed: `KPM_Tn`,
      `KPM_Tn_mps`, `_run_kpm_mps!`, `get_dos_trace`, `get_ldos_spatial(:mpo)` (the other
      solvers/kpm sites already called `_run_kpm_mps!`), `get_bands`, `get_qpi`,
      `KPM_Tn_gpu`, `get_ldos_spatial_gpu`, `get_ldos_spatial_mps_gpu`, `get_bands_gpu`;
      the GPU exciton LDOS and stochastic DOS loops were copies of `_run_kpm_mps!` and now
      call it with GPU tensors. NH: the T_k(A) of the five CPU (`nh_kpm_partials`,
      `_nh_kpm_mps_ldos`, `_nh_scalar_online`, `_nh_diag_online`, `_nh_stochastic_online`)
      and three GPU recurrences run on `chebyshev_foreach`; the partial recurrence P_k is
      not a Chebyshev recursion and rides in `f!` in the old order (after T_k; before
      T_{k+1} in `nh_kpm_partials`), its step shared as `_nh_partial_step` by six of them
      (`nh_kpm_partials` writes `apply(2S, T)` and a second sum, the GPU stochastic trace
      skips the S product at odd k by parity: both keep their own). Left as a loop:
      `get_exciton_cheb_convergence_gpu`, whose two recursions run in lockstep and are
      compared order by order; each of its steps is `_chebyshev_step`. The GPU loops keep
      their `_gpu_gc!()` after each step; since the recursion drops T_{n−1} before calling
      `f!`, a GC inside the callback (the LDOS and bands accumulators have one) can
      already reclaim it.*
- [x] `_kpm_energy_grid(H, ωs; kernel, …) -> (ω_r, W, denom, valid)` replacing 14 copies of the
      rescale/weights/valid block and 7 hand-written `π²·N·√(1−ω²)` normalisations.
      *`solvers/kpm/kernels.jl` §6 (tier2/kpmkernels): `_kpm_energy_grid(H, Ncheb, ωs; …)`
      and `(Ncheb, ω_r; …)` for energies already rescaled, with `_rescaled_energies(H, ωs)`;
      `allow_hodc=true` is `_dos_weight_matrix`, the default keeps the convolution kernels
      and their `:hodc` error. 17 call sites (ldos ×2, dos ×2, exciton ×2,
      `get_ldos_diag_from_Tn`, low-level `get_bands`, exciton_spectra ×2, QPI ×2, gpu ×5)
      and 8 normalisations now `denom[iω]` (same expression, bit for bit).
      `get_ldos_from_mun` (one scalar E) keeps its own.*
- [x] `_chebyshev_sum(Tn, coeffs; …)` replacing 6 weighted-sum copies; HODC variants become a
      coefficient choice.
      *`solvers/kpm/cached.jl` §3 (tier2/kpmkernels): each coefficient is a number or a
      tuple of factors applied left to right, so `2 * T * g * k` stays `((T·2)·g)·k`;
      `A = +(A, term; maxdim)` then `truncate!(A; cutoff)` as before. Used by
      `get_density_from_Tn` (coefficients kept: the θ(x − μ) bug, fixed since by the
      v0.1.1 merge, was then still pending),
      `get_Green_retarded_from_Tn`, `get_ldos_w_from_Tn`, both `_hodc` variants (their
      weight vectors) and `_weighted_mpo_sum` (rpa/cheb2d.jl, after dropping |w| < tol).
      The per-energy diagonal accumulators (`get_ldos_diag_from_Tn`, QPI, cheb2d
      `_accumulate_scaled!`) and the NH reconstructions (no truncation, first term
      unweighted) keep their loops; `_weighted_mpo_sum_gpu` (conductivity only) too.*
- [x] One Jackson kernel (`_kpm_kernel`) with a `normalize` keyword; delete `_jackson_kernel`
      (RPA) and `nh_jackson_weights` (NH).
      *Partly done in tier2/kpmkernels: `nh_jackson_weights(N)` is bit for bit `_kpm_kernel(N + 1,
      :jackson)[1:N]` (checked element by element for N = 1…4000) and is deleted; its eight
      callers (nh/kpm.jl ×5, gpu/nh.jl ×3) call `_kpm_kernel`, and the golden case keeps its
      record through a local definition. Not done: `_jackson_kernel` stays, because it is
      not `_kpm_kernel` under any normalisation (see "Found by the Tier 2 KPM kernels"), and
      no `normalize` keyword was added, since no caller would use it without changing values.*
      *Completed in ead1d66: `_jackson_kernel` was the off-by-one kernel; the cheb2d bubbles
      now take `_kpm_kernel(N + 1, :jackson)[1:N] ./ (N + 1)` and it is deleted.*
- [x] `AuxProjection` struct (or `aux...` kwargs forwarded to `_aux_setup`) replacing the
      8-keyword block copied into ~10 signatures; one `_project_aux_sectors` replacing the
      nambu→spin→layer→sublattice chain written 4× (KPM, QFT, GPU ×2) and the 4 sector
      projectors (`project_aux`, `_project_aux_block`, `_project_spin_sector`, `contract_nh_block`).
      *`core/AuxDOF.jl` §8–9 (tier2/auxproj). A struct, because the chain needs a flag, a
      selector, an Index and a side per DOF and the low-level `get_bands` supplies its
      indices instead of detecting them: `AuxProjection(nambu, spin, layer, sublat)` of
      `AuxDOFProjection(on, sector, index, side)`, built by `_aux_projection(H; <the eight
      keywords>, autoenable)` (`_autoenable_proj` + `aux_site` detection) in the seven
      TBHamiltonian bodies, and from the explicit keywords in the low-level `get_bands`;
      the public keywords are unchanged. `_project_aux_sectors(T, aux; project, spin_index,
      sublattice)` is the chain of `get_bands`, `get_ldos_spatial(:mpo)`, `get_bands_gpu`
      and `get_ldos_spatial_gpu` (the GPU passes `project=_project_aux_gpu`), in the old
      accumulation order (the spin step's 2D comprehension included). Projectors: two
      kernels, `_project_end_site` (one-hot pair, keeps real operators real) behind
      `project_aux` and `contract_nh_block`, and `_block_projector` + `_absorb_aux_site`
      (ComplexF64, any site) behind `_project_aux_block` and `_project_spin_sector`; they
      stay apart because merging would change element types, and the wrappers keep how
      the site is found, their checks and messages. `_aux_setup` is now a view of the
      struct (golden-pinned); `get_ldos_spatial_mps_gpu` keeps its rejection test (building
      the struct would run the index detection first). Outputs bit for bit unchanged.*
- [x] `probe_state(H, x, σ…)` replacing the psi0 selection duplicated 3× in KPM.
      *`core/AuxDOF.jl` §10 (tier2/auxproj): `probe_state(H, x)` (position probe, or |x, x⟩ on
      an exciton register) and `probe_state(H, x, σ)` with `σ` from `_probe_sectors(aux)`
      (`_ldos_make_psi0`), in `get_ldos_online`, `get_ldos_spatial` (`:mps` and the `:mpo`
      probe dictionary) and both stochastic DOS.*
- [x] Keyword `TBHamiltonian(; L, N, sites, mpo, …)` plus `similar(H; mpo=, sites=, …)` copy
      constructor; delete the six positional overloads.
      *tier2/ctor: the 19 positional calls in `src` (chain, Haldane, custom, the preset
      registry builder, the six sublattice builders, bilayer/multilayer/twisted, T-junction,
      the three projected spaces) pass the same values by keyword; the `H.Lx = Lx` and
      `H.position_space = …` lines after them moved into the call. Every builder's
      Hamiltonian is field-by-field identical (MPO tensors bitwise). The copy constructor is
      the existing `TBHamiltonian(H; field=value, …)` (v0.1.1), not a `similar` method; the
      21-field positional constructor stays. The untracked scripts
      `examples/nontracked/{exciton_benchmarking,Exciton_Resub}/scripts*/exciton_hio.jl` and
      `examples/nontracked/Fibonacci_LDOS/fibonacci_hamiltonian_io.jl` call the 13-argument
      form and need the keyword form.*
- [x] One model registry entry per model (builder → `TBHamiltonian`, dim, params, geometry,
      scale) replacing `get_Hamiltonian`'s if-chain + `build_hamiltonian` + `_build_preset` +
      `_build_sublattice` + `_preset_geometry` + `_estimate_scale`; `_param(params, :t, default)`
      replacing the parsing ternaries; remove drifted `kw_defaults` from the registry.
      *`MODELS` (one `ModelEntry` per geometry) in `core/ModelRegistry.jl` (tier2/registry);
      `MODEL_REGISTRY` is now its preset view and `_preset_geometry`/`_estimate_scale` read
      the entries. Not done: removing the drifted `kw_defaults`, which set the pinned MPOs
      (e.g. `qc2dsquare` tol 1e-9); the quirks kept are listed in the file header.*
- [x] Universal scale maker `estimate_scale` in the model registry (decided 2026-09-26):
      preset default = max(today's formula, estimate); `:small` = exact dense spectrum of
      the same preset at a small size, padded, for presets whose terms do not depend on
      the system size; `:geometry` = row-sum bound from the builder's terms for the
      size-scaled `chern8` and `qc2dsquare` (their defaults change; changelog); `:dmrg` =
      today's `scale=0` path for modified Hamiltonians; `scale=:small|:geometry|:dmrg`
      selects a method explicitly. Supersedes the held commit b6b9fba.
      *Done in tier2/registry for `chain_1d` and the MODEL_REGISTRY presets (chernhex keeps
      its analytic bound, the multi-atom lattices their builder defaults: see "Found by
      the Tier 2 scale maker"); test/scale_maker.jl.*

- [x] One `masked_shift_hopping(Lx, Ly, sites, hop, q; src_mask)` replacing six near-identical
      2D kinetic builders; retire `generate_kin_u/d` in favour of `shift_mpo`.
      *af55259: `masked_shift_hopping` (lattice/hopping2d.jl) is the body of the seven NNN
      builders (`kineticintra2DNNN` & co. keep their names and assertions); `src_mask` names
      the mask (`:xplus`, `(:xplus, :even)`, `(:xplus, :checker)`, … or an MPO). The SSH
      builder and `_bernal_interlayer_mpo` call `shift_mpo`. `generate_kin_u/d` stay as
      public wrappers (documented; golden_lattice calls them by name), still used by
      `add_hopping_2D!`, where their `num_site` assertion is the only thing that rejects a
      projected (Fibonacci) position space, and by `add_soc!` (core/AuxDOF.jl).*
- [x] `_sublattice_bond` + `_sublattice_setup` replacing ~12 repeated bond blocks in
      kagome/lieb/honeycomb/dice; `_basis_positions` replacing 4 identical position loops;
      `sum_mpos(terms; cutoff)`.
      *af55259: all 18 bond blocks of the kagome/Lieb/honeycomb/honeycomb-NNN/dice/SSH
      builders; `_basis_positions` (Bravais vectors + basis) for the four sublattice position
      tables and `_closure_positions` for the four preset ones; `sum_mpos` (core/MPOTools.jl,
      a left fold of `+(a, b; cutoff)`) in the sublattice builders, four presets, the
      T-junction lattice and the multilayer/twisted layer sums, in their old order. Outputs
      are bit for bit those of 0a8e5cb (2018 calls, tensor by tensor). One error type moved:
      `interlayer_mpo(:honeycomb, :Bernal, Lx, Ly, sites)` with `Lx + Ly == 1` and a `sites`
      vector of the wrong length throws DimensionMismatch instead of AssertionError.*
- [x] `get_density` as the only projector dispatcher (delete `_get_density_matrix` in RPA and
      `_get_projector` in Topology); `_purified_pair` for the ρ± blocks in Purification.
      *`physics/Purification.jl` §1 and §5 (tier2/auxproj): `get_density` keeps its position-space
      and cache checks and calls `_density_matrix(H, method; …, Tn, store)`, the one dispatch
      over `:mcweeny`/`:sp2`/`:kpm`. `_get_density_matrix` and `_get_projector` are not
      deleted (golden-pinned by name, with their error texts) but are translation layers over
      it: each keeps its accepted symbols (`:purification` + `purify_method`, `:KPM`), its
      cache rule, prints and defaults (listed under "Found by the Tier 2 aux and density
      kernels"), and the pending bugs stay (θ(x − μ) coefficients, fixed since by the v0.1.1
      merge; RPA purification without ϵF). `_purified_pair(guess, a₊, a₋; …)` purifies the two initial guesses of `sign_mpo`,
      `get_ldos_drho` and `get_dos_drho`. Outputs, caches and prints bit for bit unchanged.
      Not covered: `get_C_gpu`'s own GPU McWeeny/SP2 loops (the GPU-wrapper item below;
      since tier2/gpuwrap they are `mcweeny_purify`'s and `sp2_purify`'s loops).*
- [x] RPA: `_cheb2d_setup` + `_tucker_bases` (5 copied prologues, 2 Tucker blocks); one Wynn
      driver (3 copies); magnon functions as `mode=:magnetic`.
      *`physics/rpa/cheb2d.jl` §2: `_cheb2d_setup`, the plain (m,n) sweep `_cheb2d_pair_sweep!`,
      `_tucker_bases`, `_tucker_components`, `_tucker_hadamard`, `_tucker_accumulate`;
      `physics/rpa/dyson.jl`: `_rpa_wynn_series` behind `rpa_wynn_from_bubbles`,
      `get_rpa_susceptibility_wynn` and `get_magnon_susceptibility_wynn` (tier2/rpa). Outputs and
      verbose lines unchanged. The magnon functions stay public: folding them into
      `mode=:magnetic` changes the API, so it moved to Tier 3.*
- [x] Timeev: `_rk4_step(rhs, …)` (2 copies), one `evolve_rk4_dm_*`, one trajectory loop;
      remove the double normalisation after `tdvp(normalize=true)`.
      *Section 1 of `solvers/Timeev.jl` (tier2/timeev): `_rk4_step` (also behind
      `rk4_step_dm_nh_gpu`, which passes its ComplexF32 coefficients and `maxdim` for the
      MPO sums), `_evolve_rk4_dm`, `_trajectory` (all five `evolve_*` loops) and `_tdvp_step`
      (every `tdvp` call, GPU included); outputs bit for bit unchanged. The second
      normalisation is gone from `tdvp_evolve`, `evolve_with_tdvp(_timedep)` and
      `get_state_amplitude_trajectory_gpu`: `tdvp` already ends each half-sweep with
      `normalize!`, so only `tdvp_evolve` moved, by ≤ 4.5e-16. The two GPU sampled-trajectory
      loops keep their own loops (they sample on the fly instead of storing states).*
- [x] GPU: thin wrappers over CPU kernels with a `to_device` hook (stochastic DOS, McWeeny/SP2,
      Chern operator assembly, NH kernels, `_eval_block_mps`, `extract_diagonal_to_mps`,
      `mps_to_diagonal_mpo`, `density_profile_from_dm`); one `_to_gpu(x, T)`; one
      `_resolve_gpu_type` with a single warning helper; `_gpu_log`.
      *tier2/gpuwrap. The hook is `to_device(x, T)`: a shared kernel moves every tensor it
      builds itself (one-hot and summing vectors, deltas, identities, probe states, position
      operators) with it; the CPU default `_on_host` (core/Utils.jl) returns `x`, the GPU
      wrappers pass `_to_gpu`. Kernels, each behind its CPU function and its GPU twin:
      `_extract_diagonal` (`extract_diagonal_to_mps(_gpu)`); `_mps_to_diagonal`
      (`mps_to_diagonal_mpo`, `_mps_to_diagonal_mpo_gpu`: `delta_type` keeps the GPU's
      ComplexF32 deltas, `one_site` its one-site MPS, which the CPU still rejects with the
      golden-pinned BoundsError); `_eval_block_mps` (also the GPU point, 1D-block and
      all-sites evaluators: six copies gone, `value` gives the complex amplitudes);
      `_dos_stochastic` (solvers/kpm/dos.jl: sampling and normalisation of both stochastic
      DOS; the GPU's `continuum_only`, progress lines and GPU memory release are keywords);
      `_mcweeny_iterate`, `_sp2_iterate`, `_linear_density_guess` (physics/Purification.jl:
      `mcweeny_purify`, `sp2_purify`, `purification_initial_guess`, and on GPU `get_C_gpu`,
      `_mcweeny_purify_gpu`, `_mcweeny_purify_mpo_gpu`, `_purification_initial_guess_gpu`;
      the GPU truncates squares and updates with `cutoff` only and sums the SP2 expansion
      with both: keywords `trunc`, `add_trunc`); `_chern_marker` (physics/Topology.jl:
      `get_C_op_MPO_from_P` and `get_C_gpu`; keywords for the GPU's truncations of Q, C1–C4
      and the flat operator, `Ck = +(Ck, -ck)` for both since `-1.0 * ck` would promote a
      ComplexF32 site); `_project_end_site` (`project_aux`, `contract_nh_block`,
      `_contract_nh_block_gpu`); `_nh_product_probe` (the ket/bra MPS of both stochastic NH
      traces). `_to_gpu(x, T)` (ITensor, MPO, MPS) replaces `_to_gpu_mpo`/`_to_gpu_mps`
      (six methods; the one-argument ones meant ComplexF32), `_mpo_to_f32`, `_onehot_gpu`
      and every `cu` call (`_project_aux_gpu`'s one-hot projector now keeps the tensor's
      element type, which is exact; the block evaluators' 0/1 vectors keep `cu`'s 32-bit cast
      through `_to_gpu_vec`, so a mixed-precision MPS sums as before); `_ensure_gpu(x, T;
      caller)` the four `_ensure_gpu_mp*`. `_resolve_gpu_type` = `_gpu_type` +
      `_warn_gpu_cutoff` (32-bit type with cutoff < `below`, `_resolve_gpu_type`'s old text;
      `below` = 1e-6, or each entry point's old threshold: 1e-4 for the two stochastic NH
      entry points, 1e-5 for `scf_magnetic_hubbard_gpu`), now also behind
      `get_bands_gpu`, `get_ldos_spatial_gpu` (which still warns just before its recursion),
      `get_exciton_ldos_spatial_gpu` and the two stochastic NH entry points (changelog).
      `_gpu_log(msg; indent)` prints the "[gpu] " progress lines, text unchanged.
      Checked old vs new on 175 cases (43 CPU, 132 GPU in ComplexF64/ComplexF32/Float64/Float32,
      small `maxdim` so truncation binds): every value bit for bit, tensor by tensor, and
      every printed line; only the intended warnings differ.
      Left, and why: `density_profile_from_dm_gpu`'s `:complement` branch (it truncates
      1 − diag with `maxdim`/`cutoff`, uploads the constant profile as ComplexF32 and
      defaults `sites` to the diagonal's; the CPU subtracts without truncation); the NH
      recurrences `_nh_diag_trace_scalar_online_gpu`, `_nh_diag_trace_online_gpu`,
      `_nh_stochastic_online_gpu` (the CPU ones trace through `nh_ones_mps` without
      truncating the diagonal, do not re-truncate P_k, weight with Float64 instead of
      ComplexF64 factors and draw their probes from the global RNG in another order: a
      wrapper would move the GPU results; they already share `chebyshev_foreach` and
      `_nh_partial_step`); `_eval_diag_mps_gpu` (LSB-first; `_eval_diag_mps` uses `setelt`,
      which has no element type to match); `_project_aux_gpu` (a dense |σ⟩⟨σ| projector, and
      it accepts a one-site MPO, where `_project_end_site` has no neighbour to absorb into);
      the LDOS/bands accumulations, trajectories, the exciton LDOS and convergence check and
      the conductivity helpers (GPU code with no CPU kernel of the same operations). The
      NH diagonal-trace entry points still have no precision warning (none was added).
      `_contract_nh_block_gpu` keeps `_onehot_gpu`'s range check and ErrorException for an
      invalid `block_row`/`block_col`. After review: the per-caller thresholds, the 32-bit
      evaluator vectors and the range check restore the old warnings, mixed-precision sums
      and error.*
- [x] Move the `get_ldos_spatial_mps_gpu` automatic plan into `core/Utils.jl` without
      changing its output (decided 2026-09-25); a balanced tiler may come later as an opt-in
      keyword with today's behaviour as the default.
      *`interval_sampling_plan` (tier2/registry); the sampling golden calls it directly.*

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
- [ ] RPA: `get_magnon_susceptibility(_wynn)` as deprecated aliases of
      `get_rpa_susceptibility(_wynn)(…; mode=:magnetic)`, which already compute the same channel
      (today the magnon functions forward any `kwargs...` to `get_bubble_mpo` and have their own
      error messages); `get_magnon_bubble` likewise. Moved from Tier 2's RPA item.
- [ ] Naming: `chern_marker`/`winding_marker` (keep `get_C`/`get_W` as deprecated aliases;
      `get_C_gpu` and the valley-resolved `get_valley_C` are renamed with them),
      `<model>_hamiltonian` everywhere, lowercase `_mpo` (`hopping2MPO` → `hopping_mpo`),
      `exciton_mpo` for `Exciton_Hamiltonian`, fix `get_bublle_expanded_from_Tn`.
      Export the new names that are specific enough (add them to "Public API" in
      `docs/src/index.md` too; `test/exports.jl` checks both).
- [ ] Replace hidden mutable caches (`_tn_cache`, `_tn_mps_cache`, `_density_cache`,
      `_ensure_scale!` side effects, solvers mutating user Hamiltonians) with an explicit
      `KPMExpansion` object passed to the reconstruction functions.
- [ ] CUDA as a package extension (`[weakdeps] CUDA`, `ext/TensorBindingCUDAExt/`), replacing
      the `Base.loaded_modules` UUID lookup; fix the README dependency statement.
