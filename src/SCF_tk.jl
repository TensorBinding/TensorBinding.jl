# SCF_tk.jl -- self-consistent mean-field loops
#
# This module keeps the SCF driver generic: the physical channel is encoded in
# a user-provided Hartree/Fock/Pairing builder
#
#     hartree_builder(density_mps, sites) -> MPO
#
# so CDW, magnetic mean-field, superconducting pairing, etc. can share the same
# density -> mean-field -> density iteration skeleton.

"""
    scf_constant_mps(sites, value) -> MPS

Rank-1 MPS whose amplitude is `value` on every computational-basis state.
"""
function scf_constant_mps(sites::Vector{<:Index}, value::Number)
    N = length(sites)
    N == 0 && return MPS(ITensor[])
    links = [Index(1, "Link,l=$i") for i in 1:N-1]
    tensors = Vector{ITensor}(undef, N)
    for i in 1:N
        inds_i = if N == 1
            (sites[i],)
        elseif i == 1
            (sites[i], links[i])
        elseif i == N
            (links[i-1], sites[i])
        else
            (links[i-1], sites[i], links[i])
        end
        T = ITensor(ComplexF64, inds_i...)
        local_value = i == 1 ? ComplexF64(value) : 1.0 + 0.0im
        for v in 1:dim(sites[i])
            if N == 1
                T[sites[i] => v] = local_value
            elseif i == 1
                T[sites[i] => v, links[i] => 1] = local_value
            elseif i == N
                T[links[i-1] => 1, sites[i] => v] = local_value
            else
                T[links[i-1] => 1, sites[i] => v, links[i] => 1] = local_value
            end
        end
        tensors[i] = T
    end
    return MPS(tensors)
end

"""
    scf_profile_mps(L, sites, f; type=Float64) -> MPS

Compress a scalar profile `f(n)` on `n = 0, ..., 2^L-1` into an MPS.
"""
function scf_profile_mps(L::Int, sites, f; type=Float64, tol::Real=1e-8)
    xvals = range(1, 2^L; length=2^L)
    qtt, _, _ = quanticscrossinterpolate(type, i -> f(round(Int, i) - 1), xvals;
                                         tolerance=tol)
    tt = TensorCrossInterpolation.tensortrain(qtt.tci)
    return MPS(tt; sites=sites)
end

"""
    scf_eval_profile_mps(A, n) -> Real

Evaluate a profile MPS created by `scf_profile_mps` at the 0-indexed basis
coordinate `n`. This uses the same big-endian convention as `binary_to_MPS`
and `get_diagonal_mpo`; do not use `_eval_diag_mps` here, since that helper is
LSB-first for the QFT momentum convention.
"""
function scf_eval_profile_mps(A::MPS, n::Int)
    sites = siteinds(A)
    psi = binary_to_MPS(n, length(sites), sites)
    return real(inner(psi, A))
end

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
        return scf_constant_mps(collect(sites), 1.0) - diag_mps
    else
        error("Unsupported density extraction mode :$mode. Use :direct or :complement.")
    end
end

"""
    scf_rms_error(a, b) -> Float64

RMS distance between two density-profile MPS objects.
"""
function scf_rms_error(a::MPS, b::MPS)
    diff = a - b
    n = prod(dim(s) for s in siteinds(a))
    return sqrt(abs(real(inner(diff', diff))) / n)
end

"""
    scf_hartree_mpo_from_density(density_mps, interaction_op, sites;
                                 background=nothing, maxdim=100, cutoff=1e-8)

Apply an interaction kernel MPO to a density profile and convert the resulting
coefficient MPS into a diagonal Hartree MPO.
"""
function scf_hartree_mpo_from_density(density_mps::MPS, interaction_op::MPO, sites;
                                      background = nothing,
                                      maxdim::Int = 100,
                                      cutoff::Real = 1e-8)
    rho = background === nothing ? density_mps :
          +(density_mps, -background; maxdim=maxdim, cutoff=cutoff)
    coeff_mps = apply(interaction_op, rho; maxdim=maxdim, cutoff=cutoff)
    return mps_to_diagonal_mpo(coeff_mps, sites)
end

"""
    scf_cdw_hartree_builder(U; background=0.5, maxdim=100, cutoff=1e-8)

Return a simple spinless CDW Hartree builder:

```text
V_H[n] = U * (rho[n] - background)
```

This is intentionally local and minimal; pass a custom `hartree_builder` to
`scf_meanfield` for long-range interactions or other channels.
"""
function scf_cdw_hartree_builder(U::Number;
                                 background::Real = 0.5,
                                 maxdim::Int = 100,
                                 cutoff::Real = 1e-8)
    return function (density_mps::MPS, sites)
        bg = scf_constant_mps(collect(sites), background)
        coeff = +(density_mps, -bg; maxdim=maxdim, cutoff=cutoff)
        return mps_to_diagonal_mpo(U * coeff, sites)
    end
end

function _scf_copy_with_mpo(H0::TBHamiltonian, mpo::MPO; scale=0.0, center=0.0)
    return TBHamiltonian(H0.L, H0.N, H0.sites, mpo,
                         H0.geometry, H0.geometry_uc,
                         Float64(scale), Float64(center),
                         H0.spin_s, H0.nambu_s, H0.layer_s, H0.sublattice_s,
                         H0.aux_side,
                         nothing, nothing, 0, nothing)
end

"""
    scf_meanfield(H0, hartree_builder; kwargs...) -> NamedTuple

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
                       density_method::Symbol = :sp2,
                       density_mode::Symbol = :direct,
                       Nel::Int = H0.N ÷ 2,
                       fermi::Real = 0.0,
                       Ncheb::Int = 100,
                       scale::Union{Nothing,Real} = nothing,
                       spectral_bounds::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
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
        scf_constant_mps(collect(sites), float(Nel) / float(H0.N)) :
        initial_density
    hartree_mpo = initial_hartree === nothing ?
        hartree_builder(density_mps, sites) :
        initial_hartree

    history = NamedTuple[]
    best_error = Inf
    best_state = nothing

    for iter in 1:max_scf_iter
        Hmf_mpo = +(H0.mpo, hartree_mpo; maxdim=maxdim, cutoff=cutoff)
        Hmf = _scf_copy_with_mpo(H0, Hmf_mpo;
                                 scale=something(scale, 0.0), center=0.0)

        if spectral_bounds !== nothing && (density_method === :sp2 || density_method === :mcweeny)
            emin, emax = spectral_bounds
            sc = max(abs(emin), abs(emax), eps(Float64))
            Hmf.scale = Float64(sc)
            Hmf.center = 0.0
        end

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
        err = scf_rms_error(density_new, density_mps)
        particle_err = abs(real(tr(density_mpo)) - float(Nel))

        push!(history, (iter=iter, rms_error=err, particle_error=particle_err,
                        maxlinkdim_H=ITensorMPS.maxlinkdim(Hmf_mpo),
                        maxlinkdim_density=ITensorMPS.maxlinkdim(density_mpo)))

        verbose && println("SCF iter=$iter rms=$err particle_err=$particle_err")

        if err < best_error
            best_error = err
            best_state = (density_mpo=density_mpo, density_mps=density_new,
                          hartree_mpo=hartree_mpo, ham=Hmf)
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
                    hartree_mpo=hartree_mpo, ham=Hmf)
        end

        density_mps = mixed_density
        hartree_mpo = hartree_builder(density_mps, sites)
    end

    return (converged=false, stopped_by_residual_increase=false,
            iterations=max_scf_iter, history=history, rms_error=best_error,
            best_state...)
end

"""
    scf_staggered_magnetic_initial(H; amplitude=0.05, background=0.5)
        -> (rho_up, rho_dn)

Build a simple antiferromagnetic initial guess for a spinful Hubbard mean-field
loop represented as two spinless density profiles.
"""
function scf_staggered_magnetic_initial(H::TBHamiltonian;
                                        amplitude::Real = 0.05,
                                        background::Real = 0.5)
    rho_up = scf_profile_mps(H.L, H.sites,
                             n -> background + amplitude * (-1)^n;
                             type=Float64)
    rho_dn = scf_profile_mps(H.L, H.sites,
                             n -> background - amplitude * (-1)^n;
                             type=Float64)
    return rho_up, rho_dn
end

function _scf_local_hartree_from_density(rho::MPS, sites, U::Number, background::Real;
                                         maxdim::Int, cutoff::Real)
    bg = scf_constant_mps(collect(sites), background)
    coeff = +(rho, -bg; maxdim=maxdim, cutoff=cutoff)
    return mps_to_diagonal_mpo(U * coeff, sites)
end

"""
    scf_magnetic_hubbard(H0, U; kwargs...) -> NamedTuple

Two-channel collinear magnetic mean-field loop for the on-site Hubbard model:

```text
H_up = H0 + U * diag(n_down - background)
H_dn = H0 + U * diag(n_up   - background)
```

The spin channels are represented by two spinless `TBHamiltonian`s. This keeps
the first implementation simple and makes the AF/CDW distinction explicit.
"""
function scf_magnetic_hubbard(H0::TBHamiltonian, U::Number;
                              initial_up::Union{Nothing,MPS}=nothing,
                              initial_dn::Union{Nothing,MPS}=nothing,
                              background::Real = 0.5,
                              density_method::Symbol = :sp2,
                              Nel_up::Int = H0.N ÷ 2,
                              Nel_dn::Int = H0.N ÷ 2,
                              fermi::Real = 0.0,
                              Ncheb::Int = 100,
                              scale::Union{Nothing,Real} = H0.scale == 0.0 ? nothing : H0.scale,
                              max_scf_iter::Int = 30,
                              scf_tol::Real = 1e-6,
                              mix::Real = 0.4,
                              maxdim::Int = 100,
                              cutoff::Real = 1e-8,
                              purif_maxiter::Int = 40,
                              purif_tol::Real = 1e-6,
                              verbose::Bool = true)
    sites = H0.sites
    if initial_up === nothing || initial_dn === nothing
        rho_up, rho_dn = scf_staggered_magnetic_initial(H0; background=background)
        initial_up === nothing || (rho_up = initial_up)
        initial_dn === nothing || (rho_dn = initial_dn)
    else
        rho_up, rho_dn = initial_up, initial_dn
    end

    history = NamedTuple[]
    density_up_mpo = nothing
    density_dn_mpo = nothing
    Hup = H0
    Hdn = H0
    err = Inf

    for iter in 1:max_scf_iter
        V_up = _scf_local_hartree_from_density(rho_dn, sites, U, background;
                                               maxdim=maxdim, cutoff=cutoff)
        V_dn = _scf_local_hartree_from_density(rho_up, sites, U, background;
                                               maxdim=maxdim, cutoff=cutoff)

        Hup = _scf_copy_with_mpo(H0, +(H0.mpo, V_up; maxdim=maxdim, cutoff=cutoff);
                                 scale=something(scale, 0.0), center=0.0)
        Hdn = _scf_copy_with_mpo(H0, +(H0.mpo, V_dn; maxdim=maxdim, cutoff=cutoff);
                                 scale=something(scale, 0.0), center=0.0)

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

        err_up = scf_rms_error(rho_up_new, rho_up)
        err_dn = scf_rms_error(rho_dn_new, rho_dn)
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
