```@meta
CurrentModule = TensorBinding
```

# Position Spaces

## Fibonacci quasicrystal

Fibonacci chains use a projected Zeckendorf position space: `H.N = F_(L+2)`
physical positions are embedded in an ambient `2^L` qubit register. The physical
identity is therefore `physical_projector(H)`, not the ambient identity.

```julia
H = TensorBinding.fibonacci_hamiltonian(
    8; A=1.0, B=2.0,
    model=:hopping,
    boundary=:periodic,
)

# Equivalent generic constructor
H = TensorBinding.get_Hamiltonian(
    "fibonacci", (A=1.0, B=2.0);
    L=8, model=:hopping, boundary=:periodic,
)
```

`model=:onsite` uses `A` and `B` as onsite energies with uniform hopping `t`.
`model=:hopping` uses them as bond amplitudes with uniform `onsite`. Periodic
boundaries close the physical Fibonacci approximant; they do not wrap at the
last ambient binary state.

CPU KPM construction, cached/online LDOS, spatial LDOS, stochastic DOS,
deterministic trace DOS, and KPM density construction are projector-aware.
Binary-only modifiers, QFT/bands, GPU, purification, topology, SCF, exciton,
and non-Hermitian APIs currently reject projected position spaces explicitly.

```julia
energies = range(-4, 4; length=400)

dos = TensorBinding.get_dos_trace(H, 200, energies)
ldos = TensorBinding.get_ldos_spatial(
    H, 200, energies;
    ordering=:conumber,
    conumber_orientation=:standard,
    conumber_centered=true,
    conumber_alignment=:atomic,
)
conumbers = TensorBinding.site_axis(H; ordering=:conumber)
```

The two multipliers are reflections because
`F_(n-1) = F_n - F_(n-2)` modulo `F_n`; changing orientation cannot repair a
wrong cyclic phase. The raw modular formula alone does not choose where the periodic perpendicular-
space interval is cut. The default `alignment=:atomic` chooses that cut to give
three contiguous blocks of sizes `F_L | F_(L-1) | F_L`: molecular, atomic
(`AA`), molecular. `centered=true` then labels this ordered axis around zero;
it does not rotate it again. Set `alignment=:raw` to inspect unshifted residues,
or use `orientation=:reversed` for the reflected perpendicular-space direction.

Pointwise helpers avoid allocating a full permutation for very large systems:

```julia
c = TensorBinding.fibonacci_conumber(43, site)
site_again = TensorBinding.fibonacci_site_from_conumber(43, c)
kind = TensorBinding.fibonacci_site_environment(43, site)
depth = TensorBinding.fibonacci_atomic_depth(43, site)
```

Successive atomic deflations map `L -> L-3`. To zoom without losing the induced
phason/origin, retain the original conumber coordinates and slice the nested
window instead of assigning the selected sites fresh indices `1:F_(L'+2)`:

```julia
zoom = TensorBinding.fibonacci_rg_partition(43; depth=13)
@assert zoom.effective_L == 4
@assert zoom.window_count == 8
@assert (zoom.molecular_count, zoom.atomic_count, zoom.molecular_count) == (3, 2, 3)

# Probe those original physical sites in their inherited conumber order.
zoom_sites = [
    TensorBinding.fibonacci_site_from_conumber(43, c; centered=false)
    for c in zoom.window_ranks
]
ldos_zoom = TensorBinding.get_ldos_spatial(
    H, 4000, energies;
    ordering=:physical,
    x_groups=[[site] for site in zoom_sites],
)
# Plot ldos_zoom against zoom.window_axis; do not conumber it a second time.
```

A periodic hopping ring with odd `H.N` is not bipartite and is therefore not
required to have exact `E -> -E` chiral symmetry.

```@autodocs
Modules = [TensorBinding]
Pages   = ["position_spaces/Fibonacci.jl"]
```

## Metallic-mean quasicrystals

The metallic-mean chain with parameter `m` is the fixed point of `A -> A^m B`,
`B -> A` (`m = 1` Fibonacci, `m = 2` silver mean, `m = 3` bronze mean). Sites
are labelled in the numeration system with basis `q_0 = 1`, `q_1 = m + 1`,
`q_(l+1) = m q_l + q_(l-1)`, digits in `0:m`, and the rule that a digit `m` must
be followed by `0`. `L` digits enumerate `H.N = q_L` physical sites inside an
ambient `(m+1)^L` register of `Qudit` sites of dimension `m + 1`. The letter at
site `n` is `B` exactly when the least significant digit of `n` is `m`, so the
word and the validity indicator are both bond-dimension-2 automaton MPS and the
Hamiltonian MPO `P (V + T K + h.c.) P` is exact at any `L`.

```julia
H = TensorBinding.metallic_mean_hamiltonian(
    2, 8; A=1.0, B=2.0, model=:hopping, boundary=:periodic,
)

# Equivalent generic constructor (m is required)
H = TensorBinding.get_Hamiltonian(
    "metallic_mean", (A=1.0, B=2.0); L=8, m=2,
)
```

The projector-aware CPU KPM entry points work exactly as for Fibonacci
(`KPM_Tn`, `get_ldos_online`, `get_ldos_spatial` with `ordering=:physical`,
`get_dos_stochastic`, `get_dos_trace`), as does `get_ldos_spatial_mps_gpu`.
Conumbering and the inherited-conumber sampling plans are currently
Fibonacci-only, so `ordering=:conumber` throws for metallic means.
`m = 1` reproduces the Fibonacci chain of `fibonacci_hamiltonian` on
dimension-2 `Qudit` sites.

```@autodocs
Modules = [TensorBinding]
Pages   = ["position_spaces/MetallicMean.jl"]
```

## k-bonacci quasicrystals

The k-bonacci chain on the alphabet `a_1, …, a_k` (written `A, B, C, …`) is the
fixed point of `a_i -> a_1 a_(i+1)` for `i < k` and `a_k -> a_1` (`k = 2`
Fibonacci, `k = 3` Tribonacci `A -> AB, B -> AC, C -> A`, `k = 4` Tetranacci).
Sites are labelled by binary strings with no `k` consecutive ones, read with the
weights `w_l = 2^l` for `l < k` and `w_l = w_(l-1) + … + w_(l-k)` otherwise
(Zeckendorf for `k = 2`, the Tribonacci numbers `T_(l+3)` for `k = 3`). `L`
digits enumerate `H.N = w_L` physical sites inside the ambient `2^L` `Qubit`
register, exactly like the Fibonacci chain. The letter at site `n` is `a_(r+1)`
where `r` is the number of trailing ones of `n`, so the word and the validity
indicator are `k`-state automaton MPS (bond dimension `k`) and the Hamiltonian
MPO `P (V + T K + h.c.) P` is exact at any `L`. The decrement `K` clears the
least significant one and rewrites the tail with the pattern `1^(k-1) 0`.

```julia
# Tribonacci hopping chain with t_A/t_B = t_B/t_C = 0.8 and t_C = 1
H = TensorBinding.kbonacci_hamiltonian(
    3, 10; values=(0.64, 0.8, 1.0), model=:hopping, boundary=:periodic,
)

# Equivalent generic constructor (k is required); letter keys or values=(…)
H = TensorBinding.get_Hamiltonian(
    "kbonacci", (A=0.64, B=0.8, C=1.0); L=10, k=3,
)
```

The projector-aware CPU KPM entry points and `get_ldos_spatial_mps_gpu` work as
for Fibonacci with `ordering=:physical`; conumbering is Fibonacci-only, so
`ordering=:conumber` throws. `k = 2` reproduces `fibonacci_hamiltonian` exactly,
on the same `Qubit` sites.

```@autodocs
Modules = [TensorBinding]
Pages   = ["position_spaces/KBonacci.jl"]
```
