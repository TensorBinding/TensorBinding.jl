# DMRG_tk.jl — Variational ground state and spectral DMRG utilities
#
# Two physical functions:
#   dmrg_gs       — ground state of H (standard DMRG energy minimisation)
#   dmrg_spectral — ground state of K(ω,η) = (H−ωI)² + η²I
#
# LDoS connection:
#   Minimising ⟨ψ|K|ψ⟩ forces |ψ(ω)⟩ to concentrate on whichever eigenstate
#   of H lies closest to ω.  The local weight |⟨i|ψ(ω)⟩|² then approximates
#   the LDoS at position i and energy ω (broadened by η).  Sweeping ω
#   reconstructs the full site-resolved spectral function.
#
# Helper:
#   build_K        — assemble K as an MPO (useful standalone)
#   local_weight   — |⟨i|ψ⟩|²  (LDoS proxy at a single site)


# ─────────────────────────────────────────────────────────────────
# 1.  Ground-state DMRG
# ─────────────────────────────────────────────────────────────────

"""
    dmrg_gs(H_mpo, sites; nsweeps, maxdim, cutoff, noise, linkdim_init) -> (E, ψ)

Find the ground state and energy of `H_mpo` using DMRG.

# Keyword arguments
- `linkdim_init` : bond dimension of the random initial MPS
- `nsweeps`      : total number of DMRG sweeps
- `maxdim`       : max bond dimension per sweep (scalar or vector)
- `cutoff`       : SVD truncation cutoff
- `noise`        : perturbative noise per sweep (scalar or vector; aids convergence)

Returns `(E, ψ)`.
"""
function dmrg_gs(H_mpo::MPO, sites;
                 linkdim_init::Int = 10,
                 nsweeps::Int      = 10,
                 maxdim            = [10, 20, 50, 100, 200],
                 cutoff::Real      = 1e-8,
                 noise             = [1e-6, 1e-7, 1e-8, 0.0],
                 kwargs...)
    ψ0 = random_mps(sites; linkdims = linkdim_init)
    E, ψ = dmrg(H_mpo, ψ0;
                nsweeps = nsweeps,
                maxdim  = maxdim,
                cutoff  = cutoff,
                noise   = noise,
                kwargs...)
    return E, ψ
end


# ─────────────────────────────────────────────────────────────────
# 2.  Resolvent-squared MPO  K(ω,η) = (H−ωI)² + η²I
# ─────────────────────────────────────────────────────────────────

"""
    build_K(H_mpo, sites, ω, η; maxdim_K, cutoff_K) -> MPO

Build the resolvent-squared MPO

    K(ω,η) = (H − ω I)² + η² I

The ground state energy of K satisfies `E_K ≥ η²`, with `E_K → η²` when ω
coincides with an eigenvalue of H.  The ground state concentrates on the
eigenstate of H nearest to ω.

`maxdim_K` and `cutoff_K` control truncation of the intermediate MPO product;
larger `maxdim_K` gives a more accurate K at the cost of DMRG wall time.
"""
function build_K(H_mpo::MPO, sites, ω::Real, η::Real;
                 maxdim_K::Int  = 200,
                 cutoff_K::Real = 1e-8)
    I_mpo   = MPO(sites, "Id")
    H_shift = +(H_mpo, (-ω) * I_mpo; cutoff = cutoff_K)
    H_sq    = apply(H_shift, H_shift; cutoff = cutoff_K, maxdim = maxdim_K)
    return +(H_sq, (η^2) * I_mpo; cutoff = cutoff_K)
end


# ─────────────────────────────────────────────────────────────────
# 3.  Spectral DMRG  (minimise K)
# ─────────────────────────────────────────────────────────────────

"""
    dmrg_spectral(H_mpo, sites, ω, η; ...) -> (E_K, ψ)

Find the ground state of `K(ω,η) = (H−ωI)² + η²I` using DMRG.

The ground state `|ψ(ω)⟩` concentrates on the eigenstate of H closest to ω.
Use `local_weight(ψ, i, L, sites)` to read off the LDoS proxy `|⟨i|ψ(ω)⟩|²`.

# Keyword arguments
- `ψ0`                 : warm-start MPS (random if `nothing`)
- `maxdim_K`, `cutoff_K` : truncation for building K (see `build_K`)
- `linkdim_init`       : bond dimension of the random initial MPS (if ψ0=nothing)
- `nsweeps`, `maxdim`, `cutoff`, `noise` : DMRG sweep parameters

Returns `(E_K, ψ)` where `E_K ≈ η²` at spectral peaks.
"""
function dmrg_spectral(H_mpo::MPO, sites, ω::Real, η::Real;
                       ψ0                = nothing,
                       maxdim_K::Int     = 200,
                       cutoff_K::Real    = 1e-8,
                       linkdim_init::Int = 10,
                       nsweeps::Int      = 10,
                       maxdim            = [10, 20, 50, 100, 200],
                       cutoff::Real      = 1e-8,
                       noise             = [1e-6, 1e-7, 1e-8, 0.0],
                       kwargs...)
    K    = build_K(H_mpo, sites, ω, η; maxdim_K = maxdim_K, cutoff_K = cutoff_K)
    init = isnothing(ψ0) ? random_mps(sites; linkdims = linkdim_init) : ψ0
    E_K, ψ = dmrg(K, init;
                  nsweeps = nsweeps,
                  maxdim  = maxdim,
                  cutoff  = cutoff,
                  noise   = noise,
                  kwargs...)
    return E_K, ψ
end


# ─────────────────────────────────────────────────────────────────
# 4.  LDoS proxy: |⟨i|ψ⟩|²
# ─────────────────────────────────────────────────────────────────

"""
    local_weight(ψ, i, L, sites) -> Float64

Return `|⟨i|ψ⟩|²` where `|i⟩` is the position-basis state for 0-based
integer `i` (big-endian quantics encoding over `L` qubit `sites`).

When `ψ = ψ(ω)` is the DMRG ground state of `K(ω,η)`, this gives the
**LDoS proxy** at site `i` and energy `ω`:

    ρ(ω, i) ≈ |⟨i|ψ(ω)⟩|²

which peaks at eigenvalues of H that have support on site i.

For a BdG or spin-extended system pass the full `ext_sites` and set
`L = length(ext_sites)`.
"""
function local_weight(ψ::MPS, i::Integer, L::Integer, sites)
    ket = binary_to_MPS(i, L, sites)
    return abs2(inner(ket, ψ))
end
