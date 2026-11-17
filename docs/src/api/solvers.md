```@meta
CurrentModule = TensorBinding
```

# Solvers

## Kernel Polynomial Method (KPM)

### Conventions

- **Rescaled energies.** The expansion runs on `H̃ = (H − H.center)/H.scale`, whose
  spectrum lies in (−1, 1); a physical energy `ω` enters as `ω̃ = (ω − H.center)/H.scale`.
- **Normalisation.** With the convolution kernels (`:jackson`, `:lorentz`, `:fejer`,
  `:dirichlet`) every Hermitian KPM spectrum (LDOS, DOS, `A(k, ω)`, exciton spectra, QPI)
  is divided by `π²·Ncheb·√(1 − ω̃²)`. It is the spectral density per unit rescaled energy
  times `1/π`: the LDOS of one state integrates to `1/π` over `ω̃`. Multiply by
  `π/H.scale` for the density per unit physical energy. `kernel = :hodc`, where offered,
  carries the δ-function normalisation itself (one state integrates to 1; multiply by
  `1/H.scale`), so switching between `:hodc` and a convolution kernel rescales the output
  by `π`. The reconstruction is spelled out in [`get_ldos_from_mun`](@ref).
- **Jackson moments.** `kernel = :jackson` with `Ncheb` moments uses the textbook Jackson
  kernel for `Ncheb − 1` moments, whose weight on the last moment is zero, so its
  resolution is that of `Ncheb − 1` moments. The non-Hermitian KPM and the low-rank
  `cheb2d` bubbles use the kernel for all `N` moments.

```@autodocs
Modules = [TensorBinding]
Pages   = ["solvers/kpm/kernels.jl", "solvers/kpm/recursion.jl", "solvers/kpm/cached.jl",
           "solvers/kpm/ldos.jl", "solvers/kpm/dos.jl", "solvers/kpm/exciton.jl"]
```

## Krylov Green's Function

```@autodocs
Modules = [TensorBinding]
Pages   = ["solvers/Krylov.jl"]
```

## Correction Vectors

The resolvent of one probe state, `(ω + iη − H)⁻¹|b⟩`, by a DMRG-like linear solve:
site and exciton LDOS at a resolution set by η instead of a number of moments, and the
energy and weight of an isolated level. When to use which route:

- **One site, one probe, a few frequencies, small η:** correction vectors. They cost
  far less than the full Green's function and resolve isolated peaks and gaps that KPM
  needs thousands of moments for. Check `converged` (the residual).
- **Many frequencies, a continuum, or a spectrum over a whole window:** KPM, whose one
  recursion serves every ω and η.
- **The whole G(ω) as an MPO:** `get_green_krylov`, cheap when G has a low rank (clean
  and impurity lattices, ω outside the band).

The correction-vector LDOS is per unit physical energy (one state integrates to 1);
with the default KPM kernel, `get_ldos_online(…) · π/H.scale` is in the same units
(`1/H.scale` with `kernel = :hodc`, which is normalised to 1).

```@autodocs
Modules = [TensorBinding]
Pages   = ["solvers/CorrectionVector.jl"]
```

## DMRG

```@autodocs
Modules = [TensorBinding]
Pages   = ["solvers/DMRG.jl"]
```

## Time Evolution

```@autodocs
Modules = [TensorBinding]
Pages   = ["solvers/Timeev.jl"]
```
