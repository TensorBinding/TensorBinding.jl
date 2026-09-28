# Twisted.jl — twisted multilayer tight-binding Hamiltonians, returned as a
# TBHamiltonian with a layer index: each layer is rigidly rotated, and the layers
# are coupled by an exponentially decaying interlayer hopping learned with QTCI
# (hopping2MPO). Also holds monolayer_hamiltonian, the NN single-layer MPO that
# lattice/Bilayer.jl stacks as well.
#
# Entry points: twisted_multilayer_hamiltonian, twisted_bilayer_hamiltonian,
#   monolayer_hamiltonian.
#
# Depends on: core/Utils.jl (fix_sites, prepend_layer_projector,
# prepend_layer_hopping), core/MPOTools.jl (sum_mpos), core/Hamiltonian.jl
# (hopping2MPO), core/TBSystem.jl (TBHamiltonian), lattice/geometry.jl
# (lattice_positions) and core/ModelRegistry.jl (build_hamiltonian).
#
# Encoding: the first site is a dim-n_layers "Layer" index; the remaining
# L = Lx+Ly sites are position qubits (quantics binary, row-major).

# ============================================================
# 1. Single-layer MPO builder
# ============================================================

"""
    monolayer_hamiltonian(lattice, Lx, Ly, sites; t=1.0, cutoff=1e-8) -> MPO

NN tight-binding Hamiltonian MPO on `L = Lx+Ly` qubit `sites` for
`lattice ∈ {:square, :triangular, :honeycomb}` with uniform hopping `t`.
Delegates to `build_hamiltonian` and replaces internal site indices with `sites`.
"""
function monolayer_hamiltonian(lattice::Symbol, Lx::Int, Ly::Int, sites;
                               t::Number = 1.0, cutoff::Real = 1e-8)
    model = lattice === :square     ? "square_2d"     :
            lattice === :triangular ? "triangular_2d" :
            lattice === :honeycomb  ? "hex_2d"        :
            error("Unknown lattice :$lattice. Choose :square, :triangular, or :honeycomb.")
    mpo = build_hamiltonian(model, Lx, Ly;
                            mparam_dict=Dict{Symbol,Any}(:t => t, :cutoff => cutoff))
    return fix_sites(mpo, sites)
end


# ============================================================
# 2. Twisted multilayer Hamiltonian
# ============================================================

"""
    twisted_multilayer_hamiltonian(lattice, Lx, Ly, angles_deg;
        t_intra=1.0, t_inter=0.3, α_decay=1/16.0,
        tol=1e-6, cutoff=1e-8, maxdim=200) -> TBHamiltonian

Build a twisted multilayer tight-binding Hamiltonian as a `TBHamiltonian`.

**Site encoding** (`L+1` sites, `L = Lx+Ly`):
  - Site 1      : layer index (dim = `n_layers`)  ← replaces the bilayer qubit
  - Sites 2…L+1 : `L` position qubits (quantics binary, row-major)

Each layer k is rigidly rotated to angle `angles_deg[k]` (degrees).  The
interlayer coupling is exponentially decaying:
  `V_{kl}(i,j) = t_inter * exp(−α_decay * |r_k[i] − r_l[j]|)`.

The assembled Hamiltonian is:
  `H = Σ_k P_k ⊗ H_mono  +  Σ_{k<l} (|k⟩⟨l| ⊗ V_{kl} + |l⟩⟨k| ⊗ V_{lk})`

which is Hermitian for real V_{kl} since V_{lk}(i,j) = V_{kl}(j,i) = V_{kl}^T.

**Arguments**
- `lattice`    : `:square`, `:triangular`, or `:honeycomb`
- `Lx`, `Ly`  : `2^Lx × 2^Ly` sites per layer
- `angles_deg` : twist angles (°) for each layer; `length` = `n_layers`

**Keyword arguments**
- `t_intra`  : intra-layer NN hopping amplitude (same for all layers)
- `t_inter`  : interlayer hopping amplitude
- `α_decay`  : exponential decay constant for interlayer hopping
- `tol`      : QTCI tolerance passed to `hopping2MPO`
- `cutoff`   : MPO truncation cutoff used throughout assembly
- `maxdim`   : maximum bond dimension of the final MPO

Returns a `TBHamiltonian` with `H.sites = [layer_s; pos_sites]`, the layer index in
`H.layer_s` (`H.aux_side = :pre`), `H.Lx = Lx` and `H.scale = 0.0`, so the spectral
bounds are estimated on first use; no geometry is set.
"""
function twisted_multilayer_hamiltonian(
    lattice::Symbol, Lx::Int, Ly::Int,
    angles_deg::AbstractVector{<:Real};
    t_intra::Number = 1.0,
    t_inter::Number = 0.3,
    α_decay::Real   = 1 / 16.0,
    tol::Real       = 1e-6,
    cutoff::Real    = 1e-8,
    maxdim::Int     = 200,
)
    n_layers = length(angles_deg)
    n_layers ≥ 2 || error("Need at least 2 layers; got $n_layers.")
    L = Lx + Ly;  N = 2^L

    # Site spaces: one dim-n_layers layer site + L position qubits
    layer_s   = Index(n_layers, "Layer")
    pos_sites = siteinds("Qubit", L)
    ext_sites = [layer_s; pos_sites]

    # Real-space positions per layer (rotated by the respective angle)
    positions = [lattice_positions(lattice, Lx, Ly; angle_deg = θ)
                 for θ in angles_deg]

    # ── Intralayer: Σ_k P_k ⊗ H_mono  ────────────────────────────
    # All layers share the same geometry; twist only affects interlayer terms.
    H_mono   = monolayer_hamiltonian(lattice, Lx, Ly, pos_sites;
                                     t = t_intra, cutoff = cutoff)
    H_intra  = sum_mpos((prepend_layer_projector(H_mono, layer_s, k) for k in 1:n_layers);
                        cutoff=cutoff)

    # ── Interlayer: Σ_{k<l} (|k⟩⟨l|⊗V_{kl} + |l⟩⟨k|⊗V_{lk})  ──
    H_inter = nothing
    for k in 1:n_layers, l in (k+1):n_layers
        rsk = positions[k];  rsl = positions[l]
        V_kl(i, j) = t_inter * exp(-α_decay * norm(rsk[Int(i), :] - rsl[Int(j), :]))
        V_lk(i, j) = conj(V_kl(j, i))   # = V_{kl}† (the transpose for a real t_inter)

        # A complex t_inter needs complex tensors (it threw an InexactError, and V_lk was
        # the transpose, not the adjoint); a real one keeps the Float64 build.
        T_V     = t_inter isa Real ? Float64 : ComplexF64
        Vkl_mpo = hopping2MPO(V_kl, N, pos_sites; tol=tol, type=T_V)
        Vlk_mpo = hopping2MPO(V_lk, N, pos_sites; tol=tol, type=T_V)

        term = +(prepend_layer_hopping(Vkl_mpo, layer_s, k, l),
                 prepend_layer_hopping(Vlk_mpo, layer_s, l, k); cutoff=cutoff)
        H_inter = H_inter === nothing ? term : +(H_inter, term; cutoff=cutoff)
    end

    # ── Assembly ──────────────────────────────────────────────────
    H_total = +(H_intra, H_inter; cutoff=cutoff)
    ITensorMPS.truncate!(H_total; maxdim=maxdim, cutoff=cutoff)
    # scale left at 0.0 → lazy DMRG estimation on first KPM_Tn call
    return TBHamiltonian(; L, N=2^L, sites=ext_sites, mpo=H_total, layer_s, Lx)
end


"""
    twisted_bilayer_hamiltonian(lattice, Lx, Ly, θ_deg;
        t_intra=1.0, t_inter=0.3, α_decay=1/16.0,
        tol=1e-6, cutoff=1e-8, maxdim=200) -> TBHamiltonian

Convenience wrapper: two layers, layer 1 at 0° and layer 2 at `θ_deg`.
Equivalent to `twisted_multilayer_hamiltonian(lattice, Lx, Ly, [0.0, θ_deg]; …)`.
"""
function twisted_bilayer_hamiltonian(
    lattice::Symbol, Lx::Int, Ly::Int, θ_deg::Real;
    t_intra::Number = 1.0,
    t_inter::Number = 0.3,
    α_decay::Real   = 1 / 16.0,
    tol::Real       = 1e-6,
    cutoff::Real    = 1e-8,
    maxdim::Int     = 200,
)
    return twisted_multilayer_hamiltonian(
        lattice, Lx, Ly, [0.0, Float64(θ_deg)];
        t_intra=t_intra, t_inter=t_inter, α_decay=α_decay,
        tol=tol, cutoff=cutoff, maxdim=maxdim,
    )
end
