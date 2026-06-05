# SCF_tk.jl -- self-consistent mean-field loops
#
# This module keeps the SCF driver generic: the physical channel is encoded in
# a user-provided Hartree/Fock/Pairing builder
#
#     hartree_builder(density_mps, sites) -> MPO
#
# so CDW, magnetic mean-field, superconducting pairing, etc. can share the same
# density -> mean-field -> density iteration skeleton.
#
# Sections
# --------
#   1. Density / profile extraction
#   2. Interaction-to-mean-field (one-shot MPO builders)
#   3. Builder factories (closures for SCF loops)
#   4. Initial state constructors
#   5. BdG / pairing helpers
#   6. SCF loop drivers
#   7. Internal utilities
#   8. Entry points


# ============================================================
# 1. Density / profile extraction
# ============================================================

"""
    density_profile_from_dm(density_mpo, sites; mode=:direct) -> MPS

Extract a density profile MPS from the diagonal of a density-matrix MPO.
`mode=:complement` returns `1 - diag(D)`.
"""
function density_profile_from_dm(density_mpo::MPO, sites; mode::Symbol=:direct)
    diag_mps = extract_diagonal_to_mps(density_mpo)
    if mode === :direct
        return diag_mps
    elseif mode === :complement
        return constant_mps(collect(sites), 1.0) - diag_mps
    else
        error("Unsupported density extraction mode :$mode. Use :direct or :complement.")
    end
end

function _subtract_background(density_mps::MPS, sites, background;
                                  maxdim::Int, cutoff::Real)
    background === nothing && return density_mps
    bg = background isa MPS ? background :
         background isa Number ? constant_mps(collect(sites), background) :
         error("Unsupported SCF background type $(typeof(background)); use nothing, Number, or MPS.")
    return +(density_mps, -bg; maxdim=maxdim, cutoff=cutoff)
end


# ============================================================
# 2. Interaction-to-mean-field (one-shot MPO builders)
# ============================================================

"""
    hartree_mpo_from_density(density_mps, interaction_op, sites;
                                 background=nothing, maxdim=100, cutoff=1e-8)

Apply an interaction kernel MPO to a density profile and convert the resulting
coefficient MPS into a diagonal Hartree MPO.
"""
function hartree_mpo_from_density(density_mps::MPS, interaction_op::MPO, sites;
                                      background = nothing,
                                      maxdim::Int = 100,
                                      cutoff::Real = 1e-8)
    rho = _subtract_background(density_mps, sites, background;
                                   maxdim=maxdim, cutoff=cutoff)
    coeff_mps = apply(interaction_op, rho; maxdim=maxdim, cutoff=cutoff)
    return mps_to_diagonal_mpo(coeff_mps, sites)
end

function _local_hartree_from_density(rho::MPS, sites, U::Number, background::Real;
                                         maxdim::Int, cutoff::Real)
    bg = constant_mps(collect(sites), background)
    coeff = +(rho, -bg; maxdim=maxdim, cutoff=cutoff)
    return mps_to_diagonal_mpo(U * coeff, sites)
end

"""
    fock_mpo_from_density(density_mpo, interaction_op, sites;
                              sign=-1, maxdim=100, cutoff=1e-8) -> MPO

Build a Fock/exchange MPO from a density matrix and a two-body interaction
kernel by element-wise multiplication:

```text
F[i, j] = sign * V[i, j] * rho[i, j]
```

For the usual exchange term use the default `sign=-1`.
"""
function fock_mpo_from_density(density_mpo::MPO, interaction_op::MPO, sites;
                                   sign::Number = -1,
                                   maxdim::Int = 100,
                                   cutoff::Real = 1e-8)
    fock = hadamard_mpo(interaction_op, density_mpo, sites;
                            maxdim=maxdim, cutoff=cutoff)
    return sign == 1 ? fock : sign * fock
end


# ============================================================
# 3. Builder factories (closures for SCF loops)
# ============================================================

function fock_exchange_builder(interaction_op::MPO;
                          sign::Number = -1,
                          maxdim::Int = 100,
                          cutoff::Real = 1e-8)
    return function (density_mpo::MPO, sites)
        return fock_mpo_from_density(density_mpo, interaction_op, sites;
                                         sign=sign,
                                         maxdim=maxdim,
                                         cutoff=cutoff)
    end
end

"""
    cdw_hartree_builder(U; background=0.5, maxdim=100, cutoff=1e-8)

Return a simple spinless CDW Hartree builder:

```text
V_H[n] = U * (rho[n] - background)
```

This is intentionally local and minimal; pass a custom `hartree_builder` to
`meanfield` for long-range interactions or other channels.
"""
function cdw_hartree_builder(U::Number;
                                 background::Real = 0.5,
                                 maxdim::Int = 100,
                                 cutoff::Real = 1e-8)
    return function (density_mps::MPS, sites)
        bg = constant_mps(collect(sites), background)
        coeff = +(density_mps, -bg; maxdim=maxdim, cutoff=cutoff)
        return mps_to_diagonal_mpo(U * coeff, sites)
    end
end

"""
    dense_hartree_builder(V, L, sites; background=nothing, kwargs...)

Return a Hartree builder backed by a dense interaction kernel. `V` can be an
existing interaction MPO or a function `V(i, j)` on 0-indexed coordinates.
"""
function dense_hartree_builder(V, L::Int, sites;
                                   background = nothing,
                                   type=Float64,
                                   tol::Real=1e-8,
                                   maxdim::Int=100,
                                   cutoff::Real=1e-8,
                                   kwargs...)
    interaction_op = V isa MPO ? V :
                     get_mpo(L, sites, V; type=type, tol=tol, kwargs...)
    return function (density_mps::MPS, density_sites)
        return hartree_mpo_from_density(density_mps, interaction_op, density_sites;
                                            background=background,
                                            maxdim=maxdim,
                                            cutoff=cutoff)
    end
end

function _weight_to_mpo(L::Int, sites, weight;
                                  type=Float64,
                                  tol::Real=1e-8)
    if weight isa MPO
        return weight
    elseif weight isa Number
        return weight * MPO(sites, "Id")
    elseif weight isa Function
        return get_diagonal_mpo(L, sites,
                                x -> weight(round(Int, x) - 1);
                                type=type, tol=tol)
    else
        error("Unsupported distance-interaction weight $(typeof(weight)); use Number, Function, or MPO.")
    end
end

function _pair_term(term)
    if term isa Pair
        return Int(term.first), term.second
    elseif term isa Tuple && length(term) == 2
        return Int(term[1]), term[2]
    else
        error("Pair-distance terms must be `distance => weight` or `(distance, weight)`.")
    end
end

"""
    pair_distance_interaction_mpo(L, sites, distance, weight=1; kwargs...)
    pair_distance_interaction_mpo(L, sites, terms; kwargs...)

Build an interaction MPO for arbitrary fixed-distance site pairs. For a bond
weight `w(i)` on the pair `(i, i+d)`, the operator contains
`row=i+d,col=i` with weight `w(i)` and `row=i,col=i+d` with weight `w(i)`.
The shifts are non-cyclic, so boundary-wrapping bonds are absent.
"""
function pair_distance_interaction_mpo(L::Int, sites, distance::Integer, weight=1;
                                           type=Float64,
                                           tol::Real=1e-8,
                                           maxdim::Int=100,
                                           cutoff::Real=1e-8)
    N = 1 << L
    d = mod(Int(distance), N)
    (d == 0 || d >= N) && error("distance must be in 1:$(N - 1), got $distance.")

    D = _weight_to_mpo(L, sites, weight; type=type, tol=tol)
    K_fwd, K_bwd = shift_pair_mpos(sites, d; cyclic=false)

    term_fwd = apply(K_fwd, D; maxdim=maxdim, cutoff=cutoff)
    term_bwd = apply(D, K_bwd; maxdim=maxdim, cutoff=cutoff)
    op = +(term_fwd, term_bwd; maxdim=maxdim, cutoff=cutoff)
    ITensorMPS.truncate!(op; maxdim=maxdim, cutoff=cutoff)
    return op
end

function pair_distance_interaction_mpo(L::Int, sites, terms::AbstractVector;
                                           type=Float64,
                                           tol::Real=1e-8,
                                           maxdim::Int=100,
                                           cutoff::Real=1e-8)
    isempty(terms) && error("At least one pair-distance interaction term is required.")
    acc = nothing
    for term in terms
        d, weight = _pair_term(term)
        op = pair_distance_interaction_mpo(L, sites, d, weight;
                                               type=type, tol=tol,
                                               maxdim=maxdim, cutoff=cutoff)
        acc = acc === nothing ? op : +(acc, op; maxdim=maxdim, cutoff=cutoff)
    end
    ITensorMPS.truncate!(acc; maxdim=maxdim, cutoff=cutoff)
    return acc
end

"""
    pair_distance_hartree_builder(L, sites, distance, weight=1; kwargs...)
    pair_distance_hartree_builder(L, sites, terms; kwargs...)

Return a Hartree builder for sparse pair interactions generated from
non-cyclic shift MPOs.
"""
function pair_distance_hartree_builder(L::Int, sites, distance_or_terms, weight=1;
                                           background = nothing,
                                           type=Float64,
                                           tol::Real=1e-8,
                                           maxdim::Int=100,
                                           cutoff::Real=1e-8)
    interaction_op = distance_or_terms isa AbstractVector ?
                     pair_distance_interaction_mpo(L, sites, distance_or_terms;
                                                       type=type, tol=tol,
                                                       maxdim=maxdim, cutoff=cutoff) :
                     pair_distance_interaction_mpo(L, sites, distance_or_terms, weight;
                                                       type=type, tol=tol,
                                                       maxdim=maxdim, cutoff=cutoff)
    return function (density_mps::MPS, density_sites)
        return hartree_mpo_from_density(density_mps, interaction_op, density_sites;
                                            background=background,
                                            maxdim=maxdim,
                                            cutoff=cutoff)
    end
end


# ============================================================
# 4. Initial state constructors
# ============================================================

"""
    staggered_magnetic_initial(H; amplitude=0.05, background=0.5)
        -> (rho_up, rho_dn)

Build a simple antiferromagnetic initial guess for a Hubbard mean-field loop.
If `H` is spinful, the spin core is first projected out so the returned
profiles live on the same spinless position/sublattice sites as each spin block.
"""
function staggered_magnetic_initial(H::TBHamiltonian;
                                        amplitude::Real = 0.05,
                                        background::Real = 0.5)
    H_up, _ = _split_spin_channels(H)
    rho_up = get_mps(H_up.L, H_up.sites,
                    n -> background + amplitude * (-1)^n;
                    type=Float64)
    rho_dn = get_mps(H_up.L, H_up.sites,
                    n -> background - amplitude * (-1)^n;
                    type=Float64)
    return rho_up, rho_dn
end


# ============================================================
# 5. BdG / pairing helpers
# ============================================================

function _project_aux_block(mpo::MPO, aux_s::Index, row::Int, col::Int; tag::String="")
    aux_pos = findfirst(n -> any(i -> i == aux_s || (!isempty(tag) && hastags(i, tag)),
                                 siteinds(mpo, n)),
                        1:length(mpo))
    aux_pos === nothing && error("_project_aux_block: auxiliary index not found")

    proj = ITensor(ComplexF64, aux_s', aux_s)
    proj[aux_s' => row, aux_s => col] = 1.0

    tensors = ITensor[mpo[i] for i in eachindex(mpo)]
    contracted = tensors[aux_pos] * proj

    if length(tensors) == 1
        return MPO(ITensor[])
    elseif aux_pos == 1
        tensors[2] = contracted * tensors[2]
        return MPO(tensors[2:end])
    else
        tensors[aux_pos - 1] = tensors[aux_pos - 1] * contracted
        return MPO(vcat(tensors[1:aux_pos - 1], tensors[aux_pos + 1:end]))
    end
end

function _pairing_profile_mps(delta, L::Int, sites;
                                  type=ComplexF64,
                                  tol::Real=1e-8)
    if delta isa MPS
        return delta
    elseif delta isa Number
        return constant_mps(collect(sites), delta)
    elseif delta isa Function
        return get_mps(L, sites, delta; type=type, tol=tol)
    else
        error("Unsupported pairing profile $(typeof(delta)); use Number, Function, or MPS.")
    end
end

function _bdg_from_pairing(H0::TBHamiltonian, delta_mps::MPS;
                               mu::Real=0.0,
                               hartree_up::Union{Nothing,MPO}=nothing,
                               hartree_dn::Union{Nothing,MPO}=nothing,
                               scale::Union{Nothing,Real}=nothing,
                               center::Real=0.0,
                               maxdim::Int=100,
                               cutoff::Real=1e-8)
    H0_work = H0
    if H0_work.spin_s === nothing
        H0_work = deepcopy(H0)
        add_spin!(H0_work; cutoff=cutoff, maxdim=maxdim)
    end

    H_up, H_dn = _split_spin_channels(H0_work)
    sites = H_up.sites
    Id = MPO(sites, "Id")
    Hkin_up = +(H_up.mpo, (-mu) * Id; maxdim=maxdim, cutoff=cutoff)
    Hkin_dn = +(H_dn.mpo, (-mu) * Id; maxdim=maxdim, cutoff=cutoff)
    hartree_up === nothing || (Hkin_up = +(Hkin_up, hartree_up; maxdim=maxdim, cutoff=cutoff))
    hartree_dn === nothing || (Hkin_dn = +(Hkin_dn, hartree_dn; maxdim=maxdim, cutoff=cutoff))

    H_pair = mps_to_diagonal_mpo(delta_mps, sites)
    spin_s = H0_work.spin_s
    nambu_s = nambu_index()
    Hbdg_mpo = bdg_spin_hamiltonian(Hkin_up, Hkin_dn, H_pair, spin_s, nambu_s;
                                    cutoff=cutoff)
    ITensorMPS.truncate!(Hbdg_mpo; maxdim=maxdim, cutoff=cutoff)

    return TBHamiltonian(
        H_up.L, H_up.N, [nambu_s; spin_s; sites], Hbdg_mpo,
        H_up.geometry, H_up.geometry_uc,
        Float64(something(scale, H0.scale == 0.0 ? 0.0 : H0.scale)),
        Float64(center),
        spin_s, nambu_s, H0_work.layer_s, H0_work.sublattice_s,
        :pre,
        nothing, nothing, 0, nothing
    )
end

"""
    swave_anomalous_profile(density_mpo, Hbdg; maxdim=100, cutoff=1e-8) -> MPS

Extract the local singlet anomalous profile from a spinful s-wave BdG density
matrix. The returned MPS is the diagonal of the Nambu particle-hole block
projected onto the spin-singlet tensor `(i sigma_y)`.
"""
function swave_anomalous_profile(density_mpo::MPO, Hbdg::TBHamiltonian;
                                     maxdim::Int=100,
                                     cutoff::Real=1e-8)
    Hbdg.nambu_s === nothing && error("swave_anomalous_profile requires a BdG Hamiltonian.")
    Hbdg.spin_s === nothing && error("swave_anomalous_profile requires a spinful BdG Hamiltonian.")

    ph = _project_aux_block(density_mpo, Hbdg.nambu_s, 1, 2; tag="Nambu")
    f_ud = _project_aux_block(ph, Hbdg.spin_s, 1, 2; tag="Spin")
    f_du = _project_aux_block(ph, Hbdg.spin_s, 2, 1; tag="Spin")
    singlet = +(0.5 * f_ud, -0.5 * f_du; maxdim=maxdim, cutoff=cutoff)
    ITensorMPS.truncate!(singlet; maxdim=maxdim, cutoff=cutoff)
    return extract_diagonal_to_mps(singlet)
end

"""
    swave_normal_profiles(density_mpo, Hbdg; mode=:particle,
                               maxdim=100, cutoff=1e-8)
        -> (rho_up, rho_dn)

Extract spin-resolved local normal densities from a spinful BdG density
matrix. `mode=:particle` uses the particle-particle block directly;
`mode=:hole_complement` uses `1 - diag(hole-hole block)`.
"""
function swave_normal_profiles(density_mpo::MPO, Hbdg::TBHamiltonian;
                                   mode::Symbol=:particle,
                                   maxdim::Int=100,
                                   cutoff::Real=1e-8)
    Hbdg.nambu_s === nothing && error("swave_normal_profiles requires a BdG Hamiltonian.")
    Hbdg.spin_s === nothing && error("swave_normal_profiles requires a spinful BdG Hamiltonian.")

    if mode === :particle
        block = _project_aux_block(density_mpo, Hbdg.nambu_s, 1, 1; tag="Nambu")
        rho_mode = :direct
    elseif mode === :hole_complement
        block = _project_aux_block(density_mpo, Hbdg.nambu_s, 2, 2; tag="Nambu")
        rho_mode = :complement
    else
        error("Unsupported s-wave normal density mode :$mode. Use :particle or :hole_complement.")
    end

    rho_up_mpo = _project_aux_block(block, Hbdg.spin_s, 1, 1; tag="Spin")
    rho_dn_mpo = _project_aux_block(block, Hbdg.spin_s, 2, 2; tag="Spin")
    ITensorMPS.truncate!(rho_up_mpo; maxdim=maxdim, cutoff=cutoff)
    ITensorMPS.truncate!(rho_dn_mpo; maxdim=maxdim, cutoff=cutoff)
    return density_profile_from_dm(rho_up_mpo, Hbdg.sites[3:end]; mode=rho_mode),
           density_profile_from_dm(rho_dn_mpo, Hbdg.sites[3:end]; mode=rho_mode)
end

function _pwave_bond_mpo(delta_mps::MPS, sites, distance::Integer;
                             maxdim::Int=100,
                             cutoff::Real=1e-8)
    N = prod(dim(s) for s in sites)
    0 < distance < N || error("p-wave bond distance must satisfy 0 < distance < N.")

    D = mps_to_diagonal_mpo(delta_mps, sites)
    K_fwd, K_bwd = shift_pair_mpos(sites, distance; cyclic=false)

    upper = apply(D, K_bwd; maxdim=maxdim, cutoff=cutoff)
    lower = apply(K_fwd, D; maxdim=maxdim, cutoff=cutoff)
    pair = +(upper, -lower; maxdim=maxdim, cutoff=cutoff)
    ITensorMPS.truncate!(pair; maxdim=maxdim, cutoff=cutoff)
    return pair
end

function _triplet_equalspin_bdg(H0::TBHamiltonian, delta_up::MPS, delta_dn::MPS;
                                    distance::Integer=1,
                                    mu::Real=0.0,
                                    scale::Union{Nothing,Real}=nothing,
                                    center::Real=0.0,
                                    maxdim::Int=100,
                                    cutoff::Real=1e-8)
    H0_work = H0
    if H0_work.spin_s === nothing
        H0_work = deepcopy(H0)
        add_spin!(H0_work; cutoff=cutoff, maxdim=maxdim)
    end

    H_up, H_dn = _split_spin_channels(H0_work)
    sites = H_up.sites
    Id = MPO(sites, "Id")
    Hkin_up = +(H_up.mpo, (-mu) * Id; maxdim=maxdim, cutoff=cutoff)
    Hkin_dn = +(H_dn.mpo, (-mu) * Id; maxdim=maxdim, cutoff=cutoff)

    P_up = _pwave_bond_mpo(delta_up, sites, distance;
                               maxdim=maxdim, cutoff=cutoff)
    P_dn = _pwave_bond_mpo(delta_dn, sites, distance;
                               maxdim=maxdim, cutoff=cutoff)
    P_up_adj = swapprime(dag(P_up), 0, 1)
    P_dn_adj = swapprime(dag(P_dn), 0, 1)

    spin_s = H0_work.spin_s
    nambu_s = nambu_index()
    H = +(prepend_nambu(prepend_spin(Hkin_up, spin_s, :Pup), nambu_s, :tz),
          prepend_nambu(prepend_spin(Hkin_dn, spin_s, :Pdn), nambu_s, :tz);
          cutoff=cutoff)
    H_tp = +(prepend_spin(P_up, spin_s, :Pup),
             prepend_spin(P_dn, spin_s, :Pdn); cutoff=cutoff)
    H_tm = +(prepend_spin(P_up_adj, spin_s, :Pup),
             prepend_spin(P_dn_adj, spin_s, :Pdn); cutoff=cutoff)
    H = +(+(H, prepend_nambu(H_tp, nambu_s, :tp); cutoff=cutoff),
            prepend_nambu(H_tm, nambu_s, :tm); cutoff=cutoff)
    ITensorMPS.truncate!(H; maxdim=maxdim, cutoff=cutoff)

    return TBHamiltonian(
        H_up.L, H_up.N, [nambu_s; spin_s; sites], H,
        H_up.geometry, H_up.geometry_uc,
        Float64(something(scale, H0.scale == 0.0 ? 0.0 : H0.scale)),
        Float64(center),
        spin_s, nambu_s, H0_work.layer_s, H0_work.sublattice_s,
        :pre,
        nothing, nothing, 0, nothing
    )
end

function _pwave_bond_profile(anom_mpo::MPO, sites, distance::Integer;
                                 maxdim::Int=100,
                                 cutoff::Real=1e-8)
    K_fwd = shift_mpo(sites, distance; cyclic=false)
    bond_diag = apply(anom_mpo, K_fwd; maxdim=maxdim, cutoff=cutoff)
    ITensorMPS.truncate!(bond_diag; maxdim=maxdim, cutoff=cutoff)
    return extract_diagonal_to_mps(bond_diag)
end

"""
    pwave_equalspin_anomalous_profiles(density_mpo, Hbdg; distance=1)
        -> (F_up, F_dn)

Extract nearest-neighbor equal-spin triplet anomalous bond profiles from a
spinful BdG density matrix. The profile value at `i` corresponds to the bond
`(i, i + distance)` in the same non-cyclic convention as `build_shift_mpo`.
"""
function pwave_equalspin_anomalous_profiles(density_mpo::MPO, Hbdg::TBHamiltonian;
                                                distance::Integer=1,
                                                maxdim::Int=100,
                                                cutoff::Real=1e-8)
    Hbdg.nambu_s === nothing && error("pwave_equalspin_anomalous_profiles requires a BdG Hamiltonian.")
    Hbdg.spin_s === nothing && error("pwave_equalspin_anomalous_profiles requires a spinful BdG Hamiltonian.")

    ph = _project_aux_block(density_mpo, Hbdg.nambu_s, 1, 2; tag="Nambu")
    f_up_mpo = _project_aux_block(ph, Hbdg.spin_s, 1, 1; tag="Spin")
    f_dn_mpo = _project_aux_block(ph, Hbdg.spin_s, 2, 2; tag="Spin")
    sites = Hbdg.sites[3:end]
    return _pwave_bond_profile(f_up_mpo, sites, distance;
                                   maxdim=maxdim, cutoff=cutoff),
           _pwave_bond_profile(f_dn_mpo, sites, distance;
                                   maxdim=maxdim, cutoff=cutoff)
end


# ============================================================
# 6. SCF loop drivers
# ============================================================

"""
    meanfield(H0, hartree_builder; kwargs...) -> NamedTuple

Generic self-consistent mean-field loop.

Workflow per iteration:
1. Build `H_MF = H0 + H_Hartree`.
2. Compute the density matrix by `density_method = :kpm | :mcweeny | :sp2`.
3. Extract a density profile MPS.
4. Compute RMS error, mix density, rebuild Hartree term.

Returns a named tuple with density, Hartree term, final Hamiltonian, and history.
"""
function scf_meanfield(H0::TBHamiltonian, hartree_builder;
                       initial_density::Union{Nothing,MPS}=nothing,
                       initial_hartree::Union{Nothing,MPO}=nothing,
                       fock_builder = nothing,
                       initial_fock::Union{Nothing,MPO}=nothing,
                       density_method::Symbol = :sp2,
                       density_mode::Symbol = :direct,
                       Nel::Int = H0.N ÷ 2,
                       fermi::Real = 0.0,
                       Ncheb::Int = 100,
                       scale::Union{Nothing,Real} = nothing,
                       spectral_bounds::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
                       purification_scale_padding::Real = 1.05,
                       max_scf_iter::Int = 30,
                       scf_tol::Real = 1e-6,
                       mix::Real = 0.4,
                       maxdim::Int = 100,
                       cutoff::Real = 1e-8,
                       purif_maxiter::Int = 40,
                       purif_tol::Real = 1e-6,
                       stop_on_increase::Bool = false,
                       verbose::Bool = true)
    sites = H0.sites
    density_mps = initial_density === nothing ?
        constant_mps(collect(sites), float(Nel) / float(H0.N)) :
        initial_density
    hartree_mpo = initial_hartree === nothing ?
        hartree_builder(density_mps, sites) :
        initial_hartree
    fock_mpo = initial_fock === nothing ? nothing : initial_fock

    history = NamedTuple[]
    best_error = Inf
    best_state = nothing

    for iter in 1:max_scf_iter
        Hmf_mpo = +(H0.mpo, hartree_mpo; maxdim=maxdim, cutoff=cutoff)
        fock_mpo === nothing ||
            (Hmf_mpo = +(Hmf_mpo, fock_mpo; maxdim=maxdim, cutoff=cutoff))
        Hmf = _copy_with_mpo(H0, Hmf_mpo;
                                 scale=something(scale, 0.0), center=0.0)
        _set_purification_scale!(Hmf, density_method;
                                    scale=scale,
                                    spectral_bounds=spectral_bounds,
                                    padding=purification_scale_padding)

        density_mpo = get_density(Hmf;
                                  method=density_method,
                                  ϵF=fermi,
                                  Ncheb=Ncheb,
                                  maxdim=maxdim,
                                  cutoff=Float64(cutoff),
                                  Nel=Nel,
                                  maxiters=purif_maxiter,
                                  tol=Float64(purif_tol),
                                  verbose=false)
        density_new = density_profile_from_dm(density_mpo, sites; mode=density_mode)
        err = rms_error(density_new, density_mps)
        particle_err = abs(real(tr(density_mpo)) - float(Nel))

        push!(history, (iter=iter, rms_error=err, particle_error=particle_err,
                        maxlinkdim_H=ITensorMPS.maxlinkdim(Hmf_mpo),
                        maxlinkdim_density=ITensorMPS.maxlinkdim(density_mpo)))

        verbose && println("SCF iter=$iter rms=$err particle_err=$particle_err")

        if err < best_error
            best_error = err
            best_state = (density_mpo=density_mpo, density_mps=density_new,
                          hartree_mpo=hartree_mpo, fock_mpo=fock_mpo,
                          ham=Hmf)
        elseif stop_on_increase
            verbose && println("SCF residual increased; returning best previous state.")
            return (converged=false, stopped_by_residual_increase=true,
                    iterations=iter, history=history, rms_error=best_error,
                    best_state...)
        end

        mixed_density = +(mix * density_new, (1.0 - mix) * density_mps;
                          maxdim=maxdim, cutoff=cutoff)

        if err < scf_tol
            return (converged=true, stopped_by_residual_increase=false,
                    iterations=iter, history=history, rms_error=err,
                    density_mpo=density_mpo, density_mps=mixed_density,
                    hartree_mpo=hartree_mpo, fock_mpo=fock_mpo, ham=Hmf)
        end

        density_mps = mixed_density
        hartree_mpo = hartree_builder(density_mps, sites)
        fock_builder === nothing ||
            (fock_mpo = fock_builder(density_mpo, sites))
    end

    return (converged=false, stopped_by_residual_increase=false,
            iterations=max_scf_iter, history=history, rms_error=best_error,
            best_state...)
end

"""
    scf_magnetic_hubbard(H0, U; kwargs...) -> NamedTuple

Two-channel collinear magnetic mean-field loop for the on-site Hubbard model.
If `H0` is spinful, the spin-up and spin-down one-body blocks are obtained by
projecting out the spin core, matching the magnetic RPA convention. If `H0` is
spinless, the previous two-copy behavior is retained.

```text
H_up = H0_up + U * diag(n_down - background)
H_dn = H0_dn + U * diag(n_up   - background)
```
"""
function scf_magnetic_hubbard(H0::TBHamiltonian, U::Union{Number, MPO};
                              initial_up::Union{Nothing,MPS}=nothing,
                              initial_dn::Union{Nothing,MPS}=nothing,
                              background::Real = 0.5,
                              density_method::Symbol = :sp2,
                              Nel_up::Int = H0.N ÷ 2,
                              Nel_dn::Int = H0.N ÷ 2,
                              fermi::Real = 0.0,
                              Ncheb::Int = 100,
                              scale::Union{Nothing,Real} = H0.scale == 0.0 ? nothing : H0.scale,
                              purification_scale_padding::Real = 1.05,
                              max_scf_iter::Int = 30,
                              scf_tol::Real = 1e-6,
                              mix::Real = 0.4,
                              maxdim::Int = 100,
                              cutoff::Real = 1e-8,
                              purif_maxiter::Int = 40,
                              purif_tol::Real = 1e-6,
                              verbose::Bool = true)
    H0_up, H0_dn = _split_spin_channels(H0)
    sites = H0_up.sites
    if initial_up === nothing || initial_dn === nothing
        rho_up, rho_dn = staggered_magnetic_initial(H0; background=background)
        initial_up === nothing || (rho_up = initial_up)
        initial_dn === nothing || (rho_dn = initial_dn)
    else
        rho_up, rho_dn = initial_up, initial_dn
    end

    history = NamedTuple[]
    density_up_mpo = nothing
    density_dn_mpo = nothing
    Hup = H0_up
    Hdn = H0_dn
    err = Inf

    for iter in 1:max_scf_iter
        V_up = U isa MPO ?
            hartree_mpo_from_density(rho_dn, U, sites;
                                     background=background, maxdim=maxdim, cutoff=cutoff) :
            _local_hartree_from_density(rho_dn, sites, U, background;
                                        maxdim=maxdim, cutoff=cutoff)
        V_dn = U isa MPO ?
            hartree_mpo_from_density(rho_up, U, sites;
                                     background=background, maxdim=maxdim, cutoff=cutoff) :
            _local_hartree_from_density(rho_up, sites, U, background;
                                        maxdim=maxdim, cutoff=cutoff)

        Hup = _copy_with_mpo(H0_up, +(H0_up.mpo, V_up; maxdim=maxdim, cutoff=cutoff);
                                 scale=something(scale, 0.0), center=0.0)
        Hdn = _copy_with_mpo(H0_dn, +(H0_dn.mpo, V_dn; maxdim=maxdim, cutoff=cutoff);
                                 scale=something(scale, 0.0), center=0.0)
        _set_purification_scale!(Hup, density_method;
                                    scale=scale,
                                    padding=purification_scale_padding)
        _set_purification_scale!(Hdn, density_method;
                                    scale=scale,
                                    padding=purification_scale_padding)

        density_up_mpo = get_density(Hup;
                                     method=density_method,
                                    ϵF=fermi,
                                     Ncheb=Ncheb,
                                     maxdim=maxdim,
                                     cutoff=Float64(cutoff),
                                     Nel=Nel_up,
                                     maxiters=purif_maxiter,
                                     tol=Float64(purif_tol),
                                     verbose=false)
        density_dn_mpo = get_density(Hdn;
                                     method=density_method,
                                     ϵF=fermi,
                                     Ncheb=Ncheb,
                                     maxdim=maxdim,
                                     cutoff=Float64(cutoff),
                                     Nel=Nel_dn,
                                     maxiters=purif_maxiter,
                                     tol=Float64(purif_tol),
                                     verbose=false)

        rho_up_new = density_profile_from_dm(density_up_mpo, sites)
        rho_dn_new = density_profile_from_dm(density_dn_mpo, sites)

        err_up = rms_error(rho_up_new, rho_up)
        err_dn = rms_error(rho_dn_new, rho_dn)
        err = sqrt((err_up^2 + err_dn^2) / 2)
        particle_err = abs(real(tr(density_up_mpo)) - float(Nel_up)) +
                       abs(real(tr(density_dn_mpo)) - float(Nel_dn))

        push!(history, (iter=iter, rms_error=err, rms_up=err_up, rms_dn=err_dn,
                        particle_error=particle_err))
        verbose && println("magnetic SCF iter=$iter rms=$err particle_err=$particle_err")

        rho_up_mixed = +(mix * rho_up_new, (1.0 - mix) * rho_up;
                         maxdim=maxdim, cutoff=cutoff)
        rho_dn_mixed = +(mix * rho_dn_new, (1.0 - mix) * rho_dn;
                         maxdim=maxdim, cutoff=cutoff)

        rho_up, rho_dn = rho_up_mixed, rho_dn_mixed
        err < scf_tol && return (
            converged=true,
            iterations=iter,
            rms_error=err,
            rho_up=rho_up,
            rho_dn=rho_dn,
            density_up_mpo=density_up_mpo,
            density_dn_mpo=density_dn_mpo,
            H_up=Hup,
            H_dn=Hdn,
            history=history,
        )
    end

    return (
        converged=false,
        iterations=max_scf_iter,
        rms_error=err,
        rho_up=rho_up,
        rho_dn=rho_dn,
        density_up_mpo=density_up_mpo,
        density_dn_mpo=density_dn_mpo,
        H_up=Hup,
        H_dn=Hdn,
        history=history,
    )
end

"""
    scf_swave_superconducting(H0, g; kwargs...) -> NamedTuple

Spin-singlet s-wave pairing SCF. By default this keeps the normal Hamiltonian
fixed and updates only the pairing channel. Set `include_hartree=true` to also
self-consistently update spin-resolved onsite Hartree potentials,

```text
Delta_new(i) = pairing_sign * g * F_singlet(i)
V_up(i)      = hartree_sign * hartree_coupling * (rho_dn(i) - background)
V_dn(i)      = hartree_sign * hartree_coupling * (rho_up(i) - background)
```

where `F_singlet` is extracted from the Nambu off-diagonal block of the BdG
density matrix. If `H0` is spinless, a spin degree of freedom is added to a
working copy before building BdG Hamiltonians.
"""
function scf_swave_superconducting(H0::TBHamiltonian, g::Number;
                                   initial_delta = 0.1,
                                   initial_up::Union{Nothing,MPS}=nothing,
                                   initial_dn::Union{Nothing,MPS}=nothing,
                                   pairing_sign::Real = -1.0,
                                   include_hartree::Bool = false,
                                   hartree_coupling = g,
                                   hartree_sign::Real = -1.0,
                                   background::Real = 0.5,
                                   mu::Real = 0.0,
                                   density_method::Symbol = :mcweeny,
                                   normal_density_mode::Symbol = :particle,
                                   Ncheb::Int = 100,
                                   scale::Union{Nothing,Real} = nothing,
                                   purification_scale_padding::Real = 1.05,
                                   max_scf_iter::Int = 30,
                                   scf_tol::Real = 1e-6,
                                   mix::Real = 0.4,
                                   maxdim::Int = 100,
                                   cutoff::Real = 1e-8,
                                   purif_maxiter::Int = 40,
                                   purif_tol::Real = 1e-6,
                                   verbose::Bool = true)
    H_base_up, _ = _split_spin_channels(H0.spin_s === nothing ? (Htmp = deepcopy(H0); add_spin!(Htmp; cutoff=cutoff, maxdim=maxdim); Htmp) : H0)
    sites = H_base_up.sites
    delta = _pairing_profile_mps(initial_delta, H_base_up.L, sites;
                                     type=ComplexF64, tol=cutoff)
    rho_up = initial_up === nothing ? constant_mps(collect(sites), background) : initial_up
    rho_dn = initial_dn === nothing ? constant_mps(collect(sites), background) : initial_dn

    history = NamedTuple[]
    density_mpo = nothing
    anomalous_mps = nothing
    rho_up_new = nothing
    rho_dn_new = nothing
    hartree_up_mpo = nothing
    hartree_dn_mpo = nothing
    Hbdg = nothing
    err = Inf

    for iter in 1:max_scf_iter
        if include_hartree
            hartree_up_mpo = _local_hartree_from_density(
                rho_dn, sites, hartree_sign * hartree_coupling, background;
                maxdim=maxdim, cutoff=cutoff)
            hartree_dn_mpo = _local_hartree_from_density(
                rho_up, sites, hartree_sign * hartree_coupling, background;
                maxdim=maxdim, cutoff=cutoff)
        end

        Hbdg = _bdg_from_pairing(H0, delta;
                                     mu=mu,
                                     hartree_up=hartree_up_mpo,
                                     hartree_dn=hartree_dn_mpo,
                                     scale=scale,
                                     maxdim=maxdim,
                                     cutoff=cutoff)
        _set_purification_scale!(Hbdg, density_method;
                                    scale=scale,
                                    padding=purification_scale_padding)
        full_dim = prod(dim(s) for s in Hbdg.sites)
        Nel_bdg = div(full_dim, 2)

        density_mpo = get_density(Hbdg;
                                  method=density_method,
                                  ϵF=0.0,
                                  Ncheb=Ncheb,
                                  maxdim=maxdim,
                                  cutoff=Float64(cutoff),
                                  Nel=Nel_bdg,
                                  maxiters=purif_maxiter,
                                  tol=Float64(purif_tol),
                                  verbose=false)
        anomalous_mps = swave_anomalous_profile(density_mpo, Hbdg;
                                                    maxdim=maxdim,
                                                    cutoff=cutoff)
        delta_new = (pairing_sign * g) * anomalous_mps
        err_delta = rms_error(delta_new, delta)

        err_up = 0.0
        err_dn = 0.0
        if include_hartree
            rho_up_new, rho_dn_new = swave_normal_profiles(density_mpo, Hbdg;
                                                               mode=normal_density_mode,
                                                               maxdim=maxdim,
                                                               cutoff=cutoff)
            err_up = rms_error(rho_up_new, rho_up)
            err_dn = rms_error(rho_dn_new, rho_dn)
        end
        err = include_hartree ? sqrt((err_delta^2 + err_up^2 + err_dn^2) / 3) : err_delta

        push!(history, (iter=iter, rms_error=err, rms_delta=err_delta,
                        rms_up=err_up, rms_dn=err_dn,
                        maxlinkdim_H=ITensorMPS.maxlinkdim(Hbdg.mpo),
                        maxlinkdim_density=ITensorMPS.maxlinkdim(density_mpo),
                        maxlinkdim_delta=ITensorMPS.maxlinkdim(delta_new)))
        verbose && println("s-wave SCF iter=$iter rms=$err")

        delta_mixed = +(mix * delta_new, (1.0 - mix) * delta;
                        maxdim=maxdim, cutoff=cutoff)
        delta = delta_mixed
        if include_hartree
            rho_up = +(mix * rho_up_new, (1.0 - mix) * rho_up;
                       maxdim=maxdim, cutoff=cutoff)
            rho_dn = +(mix * rho_dn_new, (1.0 - mix) * rho_dn;
                       maxdim=maxdim, cutoff=cutoff)
        end

        err < scf_tol && return (
            converged=true,
            iterations=iter,
            rms_error=err,
            delta_mps=delta,
            anomalous_mps=anomalous_mps,
            rho_up_mps=rho_up,
            rho_dn_mps=rho_dn,
            hartree_up_mpo=hartree_up_mpo,
            hartree_dn_mpo=hartree_dn_mpo,
            density_mpo=density_mpo,
            ham=Hbdg,
            history=history,
        )
    end

    return (
        converged=false,
        iterations=max_scf_iter,
        rms_error=err,
        delta_mps=delta,
        anomalous_mps=anomalous_mps,
        rho_up_mps=rho_up,
        rho_dn_mps=rho_dn,
        hartree_up_mpo=hartree_up_mpo,
        hartree_dn_mpo=hartree_dn_mpo,
        density_mpo=density_mpo,
        ham=Hbdg,
        history=history,
    )
end

"""
    scf_swave_hubbard(H0, U; kwargs...) -> NamedTuple

Attractive on-site Hubbard s-wave SCF. This is a convenience wrapper around
`swave_superconducting` with the Hartree channel enabled. Positive `U`
means attraction by default:

```text
Delta_i = -U F_i
V_up    = -U (rho_dn - background)
V_dn    = -U (rho_up - background)
```
"""
function scf_swave_hubbard(H0::TBHamiltonian, U::Number;
                           pairing_sign::Real=-1.0,
                           hartree_coupling=U,
                           hartree_sign::Real=-1.0,
                           kwargs...)
    return scf_swave_superconducting(H0, U;
                                     include_hartree=true,
                                     pairing_sign=pairing_sign,
                                     hartree_coupling=hartree_coupling,
                                     hartree_sign=hartree_sign,
                                     kwargs...)
end

"""
    scf_pwave_equalspin(H0, V; kwargs...) -> NamedTuple

Minimal spinful equal-spin p-wave triplet SCF. The pairing fields live on
nearest-neighbor bonds by default:

```text
Delta_up(i) = pairing_sign * V * F_up(i, i + distance)
Delta_dn(i) = pairing_sign * V * F_dn(i, i + distance)
```

`eta_down=-1` gives a helical initial seed, while `eta_down=1` starts the two
equal-spin channels with the same sign.
"""
function scf_pwave_equalspin(H0::TBHamiltonian, V::Number;
                             initial_up = 0.1,
                             initial_dn = nothing,
                             distance::Integer = 1,
                             eta_down::Real = -1.0,
                             pairing_sign::Real = -1.0,
                             mu::Real = 0.0,
                             density_method::Symbol = :mcweeny,
                             Ncheb::Int = 100,
                             scale::Union{Nothing,Real} = nothing,
                             purification_scale_padding::Real = 1.05,
                             max_scf_iter::Int = 30,
                             scf_tol::Real = 1e-6,
                             mix::Real = 0.4,
                             maxdim::Int = 100,
                             cutoff::Real = 1e-8,
                             purif_maxiter::Int = 40,
                             purif_tol::Real = 1e-6,
                             verbose::Bool = true)
    H_base_up, _ = _split_spin_channels(H0.spin_s === nothing ? (Htmp = deepcopy(H0); add_spin!(Htmp; cutoff=cutoff, maxdim=maxdim); Htmp) : H0)
    sites = H_base_up.sites
    delta_up = _pairing_profile_mps(initial_up, H_base_up.L, sites;
                                        type=ComplexF64, tol=cutoff)
    delta_dn_seed = initial_dn === nothing ? (n -> eta_down * eval_mps(delta_up, n)) : initial_dn
    delta_dn = _pairing_profile_mps(delta_dn_seed, H_base_up.L, sites;
                                        type=ComplexF64, tol=cutoff)

    history = NamedTuple[]
    density_mpo = nothing
    F_up = nothing
    F_dn = nothing
    Hbdg = nothing
    err = Inf

    for iter in 1:max_scf_iter
        Hbdg = _triplet_equalspin_bdg(H0, delta_up, delta_dn;
                                          distance=distance,
                                          mu=mu,
                                          scale=scale,
                                          maxdim=maxdim,
                                          cutoff=cutoff)
        _set_purification_scale!(Hbdg, density_method;
                                    scale=scale,
                                    padding=purification_scale_padding)
        full_dim = prod(dim(s) for s in Hbdg.sites)
        Nel_bdg = div(full_dim, 2)
        density_mpo = get_density(Hbdg;
                                  method=density_method,
                                  ϵF=0.0,
                                  Ncheb=Ncheb,
                                  maxdim=maxdim,
                                  cutoff=Float64(cutoff),
                                  Nel=Nel_bdg,
                                  maxiters=purif_maxiter,
                                  tol=Float64(purif_tol),
                                  verbose=false)
        F_up, F_dn = pwave_equalspin_anomalous_profiles(density_mpo, Hbdg;
                                                            distance=distance,
                                                            maxdim=maxdim,
                                                            cutoff=cutoff)
        delta_up_new = (pairing_sign * V) * F_up
        delta_dn_new = (pairing_sign * V) * F_dn
        err_up = rms_error(delta_up_new, delta_up)
        err_dn = rms_error(delta_dn_new, delta_dn)
        err = sqrt((err_up^2 + err_dn^2) / 2)

        push!(history, (iter=iter, rms_error=err, rms_up=err_up, rms_dn=err_dn,
                        maxlinkdim_H=ITensorMPS.maxlinkdim(Hbdg.mpo),
                        maxlinkdim_density=ITensorMPS.maxlinkdim(density_mpo),
                        maxlinkdim_delta_up=ITensorMPS.maxlinkdim(delta_up_new),
                        maxlinkdim_delta_dn=ITensorMPS.maxlinkdim(delta_dn_new)))
        verbose && println("p-wave equal-spin SCF iter=$iter rms=$err")

        delta_up = +(mix * delta_up_new, (1.0 - mix) * delta_up;
                     maxdim=maxdim, cutoff=cutoff)
        delta_dn = +(mix * delta_dn_new, (1.0 - mix) * delta_dn;
                     maxdim=maxdim, cutoff=cutoff)

        err < scf_tol && return (
            converged=true,
            iterations=iter,
            rms_error=err,
            delta_up_mps=delta_up,
            delta_dn_mps=delta_dn,
            anomalous_up_mps=F_up,
            anomalous_dn_mps=F_dn,
            density_mpo=density_mpo,
            ham=Hbdg,
            history=history,
        )
    end

    return (
        converged=false,
        iterations=max_scf_iter,
        rms_error=err,
        delta_up_mps=delta_up,
        delta_dn_mps=delta_dn,
        anomalous_up_mps=F_up,
        anomalous_dn_mps=F_dn,
        density_mpo=density_mpo,
        ham=Hbdg,
        history=history,
    )
end


# ============================================================
# 7. Internal utilities
# ============================================================

function _canonical_channel(channel::Symbol)
    s = lowercase(String(channel))
    s in ("cdw", "charge", "hartree") && return :cdw
    s in ("magnetic", "magnetism", "spin", "hubbard") && return :magnetic
    s in ("swave", "s_wave", "s-wave", "superconducting", "superconductivity") && return :swave
    s in ("pwave", "p_wave", "p-wave", "triplet") && return :pwave
    error("Unknown SCF channel :$channel. Supported: :CDW, :magnetic, :swave, :pwave.")
end

function _canonical_density_method(method::Symbol)
    s = lowercase(String(method))
    s in ("purification", "sp2") && return :sp2
    s in ("mcweeny", "mcpurify") && return :mcweeny
    s in ("kpm", "chebyshev") && return :kpm
    return method
end

_is_purification(method::Symbol) = method === :sp2 || method === :mcweeny

function _set_purification_scale!(H::TBHamiltonian, method::Symbol;
                                     scale::Union{Nothing,Real}=nothing,
                                     spectral_bounds::Union{Nothing,Tuple{<:Real,<:Real}}=nothing,
                                     padding::Real=1.05)
    _is_purification(method) || return H
    padding >= 1.0 || error("purification_scale_padding must be >= 1.0, got $padding.")

    if spectral_bounds !== nothing
        emin, emax = spectral_bounds
        emax > emin || error("spectral_bounds must satisfy emax > emin, got $spectral_bounds.")
        H.center = Float64((emax + emin) / 2)
        H.scale  = Float64((emax - emin) / 2 * padding)
    elseif scale !== nothing
        H.scale = Float64(scale * padding)
    else
        _ensure_scale!(H)
        H.scale *= Float64(padding)
    end
    return H
end

function _copy_with_mpo(H0::TBHamiltonian, mpo::MPO; scale=0.0, center=0.0)
    return TBHamiltonian(H0.L, H0.N, H0.sites, mpo,
                         H0.geometry, H0.geometry_uc,
                         Float64(scale), Float64(center),
                         H0.spin_s, H0.nambu_s, H0.layer_s, H0.sublattice_s,
                         H0.aux_side,
                          nothing, nothing, 0, nothing)
end

function _split_spin_channels(H0::TBHamiltonian)
    H0.spin_s === nothing && return H0, H0
    return _project_spin_sector(H0, 1), _project_spin_sector(H0, 2)
end

function _make_cdw_builder(H0::TBHamiltonian, U;
                          interaction::Symbol,
                          background,
                          distance::Integer,
                          maxdim::Int,
                          cutoff::Real,
                          tol::Real,
                          kwargs...)
    mode = lowercase(String(interaction))
    if mode in ("local", "onsite", "on_site")
        U isa Number || error("get_scf(..., :CDW; interaction=:local) requires numeric U.")
        return cdw_hartree_builder(U; background=background,
                                       maxdim=maxdim, cutoff=cutoff)
    elseif mode in ("dense", "longrange", "long_range")
        return dense_hartree_builder(U, H0.L, H0.sites;
                                         background=background,
                                         maxdim=maxdim,
                                         cutoff=cutoff,
                                         tol=tol,
                                         kwargs...)
    elseif mode in ("distance", "pair", "pairs")
        if U isa AbstractVector
            return pair_distance_hartree_builder(H0.L, H0.sites, U;
                                                     background=background,
                                                     maxdim=maxdim,
                                                     cutoff=cutoff,
                                                     tol=tol)
        else
            return pair_distance_hartree_builder(H0.L, H0.sites, distance, U;
                                                     background=background,
                                                     maxdim=maxdim,
                                                     cutoff=cutoff,
                                                     tol=tol)
        end
    end
    error("Unknown CDW interaction :$interaction. Use :local, :dense, or :distance.")
end


# ============================================================
# 8. Entry points
# ============================================================

"""
    get_scf(H0::TBHamiltonian, channel::Symbol; interaction=:dense, kwargs...) -> NamedTuple

Like `get_scf(H0, U, channel)` but reads the interaction from `H0.interaction_mpo`
(stored via [`add_interaction!`](@ref)).  Supported channels: `:cdw`, `:magnetic`.

For `:cdw` the default `interaction=:dense` passes the stored MPO directly to
`dense_hartree_builder`.  For `:magnetic` the stored MPO is handled by
`magnetic_hubbard` which dispatches on `U::Union{Number,MPO}`.

Calling with `:swave` or `:pwave` raises an informative error — those channels
require an explicit coupling constant.
"""
function get_scf(H0::TBHamiltonian, channel::Symbol;
                 interaction::Symbol = :dense,
                 kwargs...)
    ch = _canonical_channel(channel)
    ch === :swave &&
        error("get_scf(H0, :swave) requires an explicit coupling: use get_scf(H0, g, :swave).")
    ch === :pwave &&
        error("get_scf(H0, :pwave) requires an explicit coupling: use get_scf(H0, V, :pwave).")
    H0.interaction_mpo !== nothing ||
        error("get_scf(H0, :$channel) requires an interaction stored via add_interaction!(H, V).")
    return get_scf(H0, H0.interaction_mpo, channel; interaction=interaction, kwargs...)
end

"""
    get_scf(H0, U, channel; kwargs...) -> NamedTuple

Convenience front-end for the self-consistent mean-field solvers.

Examples:

```julia
res = get_scf(H0, U, :CDW; tol=1e-5, maxiters=30, mixing=0.4)
Hmf = res.ham
```

Supported channels are `:CDW`, `:magnetic`, `:swave`, and `:pwave`.
The wrapper keeps the lower-level SCF routines available while providing a
single `get_*` style entry point for notebooks.
"""
function get_scf(H0::TBHamiltonian, U, channel::Symbol;
                 interaction::Symbol = :local,
                 background::Real = 0.5,
                 distance::Integer = 1,
                 initial_density::Union{Nothing,MPS}=nothing,
                 initial_hartree::Union{Nothing,MPO}=nothing,
                 density_mode::Symbol = :direct,
                 density_method::Symbol = :sp2,
                 method::Union{Nothing,Symbol} = nothing,
                 Nel::Int = H0.N ÷ 2,
                 fermi::Real = 0.0,
                 Ncheb::Int = 100,
                 scale::Union{Nothing,Real} = nothing,
                 spectral_bounds::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
                 purification_scale_padding::Real = 1.05,
                 tol::Real = 1e-6,
                 maxiters::Int = 30,
                 mixing::Real = 0.4,
                 maxdim::Int = 100,
                 cutoff::Real = 1e-8,
                 purif_maxiter::Int = 40,
                 purif_tol::Real = 1e-6,
                 stop_on_increase::Bool = false,
                 verbose::Bool = true,
                 builder_kwargs...)
    ch = _canonical_channel(channel)
    dmethod = _canonical_density_method(method === nothing ? density_method : method)

    if ch === :cdw
        builder = _make_cdw_builder(H0, U;
                                   interaction=interaction,
                                   background=background,
                                   distance=distance,
                                   maxdim=maxdim,
                                   cutoff=cutoff,
                                   tol=tol,
                                   builder_kwargs...)
        return scf_meanfield(H0, builder;
                             initial_density=initial_density,
                             initial_hartree=initial_hartree,
                             density_method=dmethod,
                             density_mode=density_mode,
                             Nel=Nel,
                             fermi=fermi,
                             Ncheb=Ncheb,
                             scale=scale,
                             spectral_bounds=spectral_bounds,
                             purification_scale_padding=purification_scale_padding,
                             max_scf_iter=maxiters,
                             scf_tol=tol,
                             mix=mixing,
                             maxdim=maxdim,
                             cutoff=cutoff,
                             purif_maxiter=purif_maxiter,
                             purif_tol=purif_tol,
                             stop_on_increase=stop_on_increase,
                             verbose=verbose)
    elseif ch === :magnetic
        return scf_magnetic_hubbard(H0, U;
                                    background=background,
                                    density_method=dmethod,
                                    fermi=fermi,
                                    Ncheb=Ncheb,
                                    scale=scale,
                                    purification_scale_padding=purification_scale_padding,
                                    max_scf_iter=maxiters,
                                    scf_tol=tol,
                                    mix=mixing,
                                    maxdim=maxdim,
                                    cutoff=cutoff,
                                    purif_maxiter=purif_maxiter,
                                    purif_tol=purif_tol,
                                    verbose=verbose,
                                    builder_kwargs...)
    elseif ch === :swave
        U isa Number || error("get_scf(..., :swave) requires numeric U.")
        return scf_swave_hubbard(H0, U;
                                 density_method=dmethod,
                                 Ncheb=Ncheb,
                                 scale=scale,
                                 purification_scale_padding=purification_scale_padding,
                                 max_scf_iter=maxiters,
                                 scf_tol=tol,
                                 mix=mixing,
                                 maxdim=maxdim,
                                 cutoff=cutoff,
                                 purif_maxiter=purif_maxiter,
                                 purif_tol=purif_tol,
                                 verbose=verbose,
                                 builder_kwargs...)
    elseif ch === :pwave
        U isa Number || error("get_scf(..., :pwave) requires numeric U.")
        return scf_pwave_equalspin(H0, U;
                                   density_method=dmethod,
                                   Ncheb=Ncheb,
                                   scale=scale,
                                   purification_scale_padding=purification_scale_padding,
                                   max_scf_iter=maxiters,
                                   scf_tol=tol,
                                   mix=mixing,
                                   maxdim=maxdim,
                                   cutoff=cutoff,
                                   purif_maxiter=purif_maxiter,
                                   purif_tol=purif_tol,
                                   verbose=verbose,
                                   builder_kwargs...)
    end
end




# ============================================================
# 9. Antiferromagnetic / Néel initial-guess density matrices
#    Used as seeds for mean-field SCF on interacting models.
#    Return (density_MPO, density_MPS).
# ============================================================

"""
    initial_guess_trivial_up_1D(L, sites) -> (MPO, MPS)

Diagonal density MPO with occupation `x % 2` on site `x` (spin-up Néel seed for 1D).
"""
function initial_guess_trivial_up_1D(L, sites)
    xvals = range(0, 2^L - 1; length=2^L)
    qtt   = QuanticsTCI.quanticscrossinterpolate(Float64, x -> Float64(Int(x) % 2), xvals;
                maxbonddim=10, tolerance=1e-8)[1]
    mps   = MPS(TCI.tensortrain(qtt.tci); sites)
    mpo   = outer(mps', mps)
    for i in 1:L; mpo.data[i] = Quantics._asdiagonal(mps.data[i], sites[i]); end
    return mpo, mps
end


"""
    initial_guess_trivial_down_1D(L, sites) -> (MPO, MPS)

Diagonal density MPO with occupation `(x+1) % 2` on site `x` (spin-down Néel seed for 1D).
"""
function initial_guess_trivial_down_1D(L, sites)
    xvals = range(0, 2^L - 1; length=2^L)
    qtt   = QuanticsTCI.quanticscrossinterpolate(Float64, x -> Float64((Int(x)+1) % 2), xvals;
                maxbonddim=10, tolerance=1e-8)[1]
    mps   = MPS(TCI.tensortrain(qtt.tci); sites)
    mpo   = outer(mps', mps)
    for i in 1:L; mpo.data[i] = Quantics._asdiagonal(mps.data[i], sites[i]); end
    return mpo, mps
end


"""
    initial_guess_Neel_up(Lx, Ly, sites) -> (MPO, MPS)

Checkerboard spin-up density seed for 2D Hubbard: occupation 1 where `(ix+iy)` is even.
"""
function initial_guess_Neel_up(Lx, Ly, sites)
    Nx    = 2^Lx
    L     = Lx + Ly
    N     = Nx * 2^Ly
    xvals = 0:N-1
    f     = i -> isodd(i%Nx + i÷Nx) ? 0.0 : 1.0
    qtt   = QuanticsTCI.quanticscrossinterpolate(Float64, f, xvals; maxbonddim=10, tolerance=1e-8)[1]
    mps   = MPS(TCI.tensortrain(qtt.tci); sites)
    mpo   = outer(mps', mps)
    for i in 1:L; mpo.data[i] = Quantics._asdiagonal(mps.data[i], sites[i]); end
    return mpo, mps
end


"""
    initial_guess_Neel_dn(Lx, Ly, sites) -> (MPO, MPS)

Checkerboard spin-down density seed for 2D Hubbard: occupation 1 where `(ix+iy)` is odd.
"""
function initial_guess_Neel_dn(Lx, Ly, sites)
    Nx    = 2^Lx
    L     = Lx + Ly
    N     = Nx * 2^Ly
    xvals = 0:N-1
    f     = i -> isodd(i%Nx + i÷Nx) ? 1.0 : 0.0
    qtt   = QuanticsTCI.quanticscrossinterpolate(Float64, f, xvals; maxbonddim=10, tolerance=1e-8)[1]
    mps   = MPS(TCI.tensortrain(qtt.tci); sites)
    mpo   = outer(mps', mps)
    for i in 1:L; mpo.data[i] = Quantics._asdiagonal(mps.data[i], sites[i]); end
    return mpo, mps
end
