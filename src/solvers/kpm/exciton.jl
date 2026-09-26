# solvers/kpm/exciton.jl — CPU exciton KPM on the 2L-site electron–hole chain
#
# Contents: the bound-pair LDOS ⟨X,X|δ(E − H)|X,X⟩ at sampled positions
# (get_exciton_ldos_spatial) and its one-position wrapper (get_exciton_ldos); the
# separation-resolved LDOS ⟨R+d,R|δ(E − H)|R+d,R⟩ (get_exciton_ldos_separation) and
# the energy-resolved exciton radius built from it (exciton_radius2). Every probe
# runs an online MPS recursion; nothing is cached on H.
#
# Entry points: get_exciton_ldos_spatial, get_exciton_ldos,
#   get_exciton_ldos_separation, exciton_radius2
# Depends on: core/Utils.jl (mpsexciton, spatial_sampling_plan), core/TBSystem.jl
#   (TBHamiltonian), solvers/DMRG.jl (_ensure_scale!), solvers/kpm/kernels.jl
#   (_kpm_energy_grid), solvers/kpm/recursion.jl (_scaled_hamiltonian,
#   _run_kpm_mps!).
#
# Split from the former solvers/KPM_tk.jl in Tier 1.

# ============================================================
# 1. Exciton LDOS (MPS-based only — no MPO Chebyshev for the 2L-site chain)
# ============================================================

"""
    get_exciton_ldos_spatial(H::TBHamiltonian, Ncheb::Int, omega_phys_vals;
                             X_list=nothing, X_groups=nothing, x_groups=nothing,
                             num_x=H.N, num_avg=1, x_start=1, x_end=H.N,
                             kernel=:jackson, lambda=4.0, eta=0.0, m_order=4,
                             maxdim=100, cutoff=1e-8, verbose=false,
                             printinfo=false, return_maxlinkdim=false)
        -> Matrix{Float64}   # (result, linkdims) with return_maxlinkdim=true

CPU spatial exciton LDOS. For each bound exciton position `X` (electron = hole =
`X`, 1-indexed in `1:H.N`) this runs an online MPS Chebyshev recursion from
`|X,X>` and accumulates all requested energies in one pass. No Chebyshev cache is
stored on `H`.

Rows are energies, columns are positions/groups. `X_list` selects positions
directly. `X_groups` (or alias `x_groups`) averages several bound-pair probes into
one output column. If no explicit positions are provided, `num_x` coarse groups
are generated over `x_start:x_end`, with `num_avg` subpositions per group.

`kernel=:hodc` uses the HODC reconstruction (`eta`, `m_order`; `eta=0` means
`1/(Ncheb+1)`); otherwise the standard KPM kernels are available (`:jackson`,
`:lorentz` with `lambda`, `:fejer`, `:dirichlet`). `maxdim` and `cutoff` truncate
each MPS recursion step; `verbose` or `printinfo` prints progress every 5 columns.

`return_maxlinkdim=true` returns `(result, linkdims)` instead of just `result`,
where `linkdims::Vector{Int}` is the reached MPS bond dimension per output column
(the χ the Chebyshev recursion hit under the given `maxdim`/`cutoff`). Mirrors the
GPU entry point; useful for cutoff/tolerance studies where χ is the observable.
"""
function get_exciton_ldos_spatial(H::TBHamiltonian, Ncheb::Int, omega_phys_vals;
                                  X_list           = nothing,
                                  X_groups         = nothing,
                                  x_groups         = nothing,
                                  num_x::Int       = H.N,
                                  num_avg::Int     = 1,
                                  x_start::Int     = 1,
                                  x_end::Int       = H.N,
                                  kernel::Symbol   = :jackson,
                                  lambda::Real     = 4.0,
                                  eta::Real        = 0.0,
                                  m_order::Int     = 4,
                                  maxdim::Int      = 100,
                                  cutoff::Real     = 1e-8,
                                  verbose::Bool    = false,
                                  printinfo::Bool  = false,
                                  return_maxlinkdim::Bool = false)
    _ensure_scale!(H)
    length(H.sites) == 2 * H.L ||
        error("get_exciton_ldos_spatial: H is not an exciton Hamiltonian (expected length(H.sites) == 2*H.L).")

    X_groups !== nothing && x_groups !== nothing &&
        error("get_exciton_ldos_spatial: pass only one of X_groups or x_groups.")
    X_list !== nothing && (X_groups !== nothing || x_groups !== nothing) &&
        error("get_exciton_ldos_spatial: pass either X_list or grouped positions, not both.")

    group_arg = X_groups !== nothing ? X_groups : x_groups
    groups = if group_arg !== nothing
        spatial_sampling_plan(H.L; x_groups=group_arg).groups
    elseif X_list !== nothing
        [[Int(x)] for x in X_list]
    else
        num_x > 0 || error("get_exciton_ldos_spatial: num_x must be positive.")
        num_avg > 0 || error("get_exciton_ldos_spatial: num_avg must be positive.")
        1 <= x_start <= x_end <= H.N ||
            error("get_exciton_ldos_spatial: expected 1 <= x_start <= x_end <= H.N.")
        window = x_end - x_start + 1
        num_x <= window ||
            error("get_exciton_ldos_spatial: num_x=$num_x exceeds sampling window length $window.")
        # 1D point layout of the shared planner (core/Utils.jl): stride
        # window ÷ num_x with num_avg sub-probes per coarse cell.
        spatial_sampling_plan(H.L; num_x, num_avg, x_start, x_end).groups
    end

    isempty(groups) && error("get_exciton_ldos_spatial: no spatial groups were selected.")
    for grp in groups
        isempty(grp) && error("get_exciton_ldos_spatial: empty spatial group.")
        all(x -> 1 <= x <= H.N, grp) ||
            error("get_exciton_ldos_spatial: all positions must lie in 1:H.N.")
    end

    Ham_n = _scaled_hamiltonian(H; cutoff=cutoff)

    omega_vals, W, denom, valid = _kpm_energy_grid(H, Ncheb, omega_phys_vals;
                                                   kernel=kernel, lambda=lambda,
                                                   eta=eta, m_order=m_order,
                                                   allow_hodc=true)
    Nomega     = length(omega_vals)

    nX     = length(groups)
    Xs     = first.(groups)
    result = zeros(Float64, Nomega, nX)
    linkdims = zeros(Int, nX)   # reached MPS bond dim per output column (see return_maxlinkdim)

    for (j, group) in enumerate(groups)
        last_linkdim = 0
        accum_group  = zeros(Float64, Nomega)

        for X in group
            psi0 = mpsexciton(X, H.sites)
            last_linkdim = _run_kpm_mps!(Ham_n, psi0, Ncheb, W, valid, accum_group;
                                         weight=1.0 / length(group),
                                         cutoff=cutoff, maxdim=maxdim)
        end

        for iomega in 1:Nomega
            valid[iomega] || continue
            result[iomega, j] = accum_group[iomega] / denom[iomega]
        end
        linkdims[j] = last_linkdim

        (verbose || printinfo) && (j % 5 == 0 || j == nX) &&
            println("  exciton ldos $j/$nX (X=$(Xs[j]), n_avg=$(length(group)))  maxlinkdim=$last_linkdim")
    end

    return return_maxlinkdim ? (result, linkdims) : result
end

"""
    get_exciton_ldos(H::TBHamiltonian, X::Int, omega_phys::Real; Ncheb=200,
                     kernel=:jackson, lambda=4.0, eta=0.0, m_order=4, maxdim=40,
                     cutoff=1e-8, verbose=false) -> Float64
    get_exciton_ldos(H::TBHamiltonian, X::Int, omega_phys_vals; <same keywords>)
        -> Vector{Float64}

Bound-pair exciton LDOS at the single position `X` (1-indexed in `1:H.N`): a
one-column call of `get_exciton_ldos_spatial` with `X_list=[X]`. Note that the
Chebyshev order is the keyword `Ncheb` here, and that `maxdim` defaults to `40`
(`get_exciton_ldos_spatial`: `100`).
"""
function get_exciton_ldos(H::TBHamiltonian, X::Int, omega_phys::Real;
                          Ncheb::Int     = 200,
                          kernel::Symbol = :jackson,
                          lambda::Real   = 4.0,
                          eta::Real      = 0.0,
                          m_order::Int   = 4,
                          maxdim::Int    = 40,
                          cutoff::Real   = 1e-8,
                          verbose::Bool  = false)
    ldos = get_exciton_ldos_spatial(H, Ncheb, [omega_phys];
                                    X_list=[X], kernel=kernel,
                                    lambda=lambda, eta=eta,
                                    m_order=m_order, maxdim=maxdim,
                                    cutoff=cutoff, verbose=verbose)
    return ldos[1, 1]
end

function get_exciton_ldos(H::TBHamiltonian, X::Int, omega_phys_vals;
                          Ncheb::Int     = 200,
                          kernel::Symbol = :jackson,
                          lambda::Real   = 4.0,
                          eta::Real      = 0.0,
                          m_order::Int   = 4,
                          maxdim::Int    = 40,
                          cutoff::Real   = 1e-8,
                          verbose::Bool  = false)
    ldos = get_exciton_ldos_spatial(H, Ncheb, omega_phys_vals;
                                    X_list=[X], kernel=kernel,
                                    lambda=lambda, eta=eta,
                                    m_order=m_order, maxdim=maxdim,
                                    cutoff=cutoff, verbose=verbose)
    return vec(ldos[:, 1])
end


# ============================================================
# 2. Separation-resolved exciton LDOS and exciton radius
# ============================================================

"""
    get_exciton_ldos_separation(H, Ncheb, omega_phys_vals; d_list, R_list=1:H.N,
                                boundary=:open, kernel=:jackson, lambda=4.0, eta=0.0,
                                m_order=4, maxdim=100, cutoff=1e-8, verbose=false,
                                printinfo=false) -> Array{Float64,3}

Relative-separation-resolved exciton LDOS

    rho(d, R, E) = <R+d, R| delta(E - H) |R+d, R>

generalising `get_exciton_ldos_spatial` (the `d = 0` slice) to electron-hole probes
with electron at `R+d` and hole at `R` (both 1-indexed in `1:H.N`).

For each `(d, R)` pair the probe `|R+d, R>` is a single product state
(`mpsexciton(R+d, R, H.sites)`) — no superposition / carry construction is needed,
unlike the momentum-space probes `mpsexcitonQ`/`mpsexcitonQTrace`. An online MPS
Chebyshev recursion accumulates all energies in one pass, exactly as in
`get_exciton_ldos_spatial`.

`boundary=:open` (default): pairs with `R+d` outside `1:H.N` are left as `NaN`.
`boundary=:periodic`: `R+d` is wrapped modulo `H.N`.

Returns an `(Nomega, length(d_list), length(R_list))` array. Cost is
`length(d_list) * length(R_list)` Chebyshev recursions, so keep these (and `Ncheb`)
small for a first pass.
"""
function get_exciton_ldos_separation(H::TBHamiltonian, Ncheb::Int, omega_phys_vals;
                                     d_list,
                                     R_list           = 1:H.N,
                                     boundary::Symbol = :open,
                                     kernel::Symbol   = :jackson,
                                     lambda::Real     = 4.0,
                                     eta::Real        = 0.0,
                                     m_order::Int     = 4,
                                     maxdim::Int      = 100,
                                     cutoff::Real     = 1e-8,
                                     verbose::Bool    = false,
                                     printinfo::Bool  = false)
    _ensure_scale!(H)
    length(H.sites) == 2 * H.L ||
        error("get_exciton_ldos_separation: H is not an exciton Hamiltonian (expected length(H.sites) == 2*H.L).")
    boundary in (:open, :periodic) ||
        error("get_exciton_ldos_separation: boundary must be :open or :periodic, got $boundary.")

    Ham_n = _scaled_hamiltonian(H; cutoff=cutoff)

    omega_vals, W, denom, valid = _kpm_energy_grid(H, Ncheb, omega_phys_vals;
                                                   kernel=kernel, lambda=lambda,
                                                   eta=eta, m_order=m_order,
                                                   allow_hodc=true)
    Nomega     = length(omega_vals)

    ds = collect(Int, d_list)
    Rs = collect(Int, R_list)
    result = fill(NaN, Nomega, length(ds), length(Rs))

    for (jR, R) in enumerate(Rs), (jd, d) in enumerate(ds)
        xe = R + d
        if boundary == :periodic
            xe = mod(xe - 1, H.N) + 1
        elseif !(1 <= xe <= H.N)
            continue   # leave as NaN: electron position falls outside the chain
        end

        psi0  = mpsexciton(xe, R, H.sites)
        accum = zeros(Float64, Nomega)
        last_linkdim = _run_kpm_mps!(Ham_n, psi0, Ncheb, W, valid, accum;
                                     cutoff=cutoff, maxdim=maxdim)

        for iomega in 1:Nomega
            valid[iomega] || continue
            result[iomega, jd, jR] = accum[iomega] / denom[iomega]
        end

        (verbose || printinfo) &&
            println("  exciton separation d=$d, R=$R (x_e=$xe)  maxlinkdim=$last_linkdim")
    end

    return result
end


"""
    exciton_radius2(rho, d_list) -> Matrix{Float64}

Energy-resolved exciton radius

    xi^2(R, E) = sum_d d^2 * P(d | R, E),   P(d | R, E) = rho(d,R,E) / sum_d' rho(d',R,E)

from a `rho` array as returned by `get_exciton_ldos_separation`
(`size(rho) == (Nomega, length(d_list), n_R)`). Returns an `(Nomega, n_R)` matrix.
Entries where any `rho[:, :, R]` along `d` is `NaN` (e.g. `R+d` outside the chain
under `boundary=:open`), or where the normalisation `sum_d rho(d,R,E)` is zero, are
returned as `NaN`.
"""
function exciton_radius2(rho::AbstractArray{<:Real,3}, d_list)
    Nomega, nd, nR = size(rho)
    nd == length(d_list) || error("exciton_radius2: size(rho,2) must match length(d_list).")
    d2 = Float64.(collect(d_list)) .^ 2

    xi2 = fill(NaN, Nomega, nR)
    for iR in 1:nR, iomega in 1:Nomega
        slice = @view rho[iomega, :, iR]
        any(isnan, slice) && continue
        norm = sum(slice)
        norm == 0 && continue
        xi2[iomega, iR] = sum(d2 .* slice) / norm
    end
    return xi2
end
