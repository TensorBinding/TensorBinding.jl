using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Logging, Test
using TensorBinding: get_Hamiltonian, add_onsite!, add_spin!, exciton_hamiltonian,
                     probe_state, mpsexciton, get_correction_vector, get_ldos_cv,
                     get_exciton_ldos_cv, get_ldos_online, refine_bound_state

# Correction vectors x = (ω + iη − H)⁻¹|b⟩ against dense resolvents: a site of a chain, the
# contact probe of an exciton, the residual check, and the Rayleigh-quotient refinement
# of an isolated level.
@testset "correction vectors: get_correction_vector, get_ldos_cv, refine_bound_state" begin
    dmat(M, s) = (D = prod(dim.(s)); reshape(Array(prod(M), prime.(s)..., s...), D, D))
    dvec(ψ, s) = vec(Array(prod(ψ), s...))
    gref(Hd, bv, z) = dot(bv, (z * I - Hd) \ bv)

    # A 64-site chain with an attractive impurity (a bound level below the band).
    H = get_Hamiltonian("chain_1d", 1.0; L=6)
    v = zeros(64); v[20] = -2.0
    add_onsite!(H, v; tol=1e-14)
    s, Hd = H.sites, dmat(H.mpo, H.sites)
    b  = probe_state(H, 20); bv = dvec(b, s)

    # one frequency, GMRES and MINRES, TBHamiltonian and raw MPO
    for (ω, η) in ((0.3, 0.05), (-2.8, 1e-3))
        for solver in (:gmres, :minres)
            r = get_correction_vector(H, b, ω; eta=η, solver)
            @test r.converged && r.residual < 1e-4      # rounding floor ~1e-8·‖x‖/‖b‖
            @test r.G ≈ gref(Hd, bv, ω + im * η) rtol=1e-6
        end
    end
    @test get_correction_vector(H.mpo, b, 0.3; eta=0.05).G ≈ gref(Hd, bv, 0.3 + 0.05im) rtol=1e-6
    # continuation in η reaches the same answer
    r = get_correction_vector(H, b, -2.8; eta=1e-3, eta_path=(0.1, 0.01))
    @test r.G ≈ gref(Hd, bv, -2.8 + 1e-3im) rtol=1e-6

    # the LDOS scan: −Im G/π per unit physical energy, one entry per frequency
    ωs = range(-3.0, 3.0; length=7)
    res = get_ldos_cv(H, 20, ωs; eta=0.05)
    @test all(res.converged)
    @test res.ldos ≈ [-imag(gref(Hd, bv, ω + 0.05im)) / π for ω in ωs] rtol=1e-6
    @test res.maxlinkdims isa Vector{Int} && length(res.maxlinkdims) == 7
    # a single frequency returns scalars
    one = get_ldos_cv(H, 20, 0.3; eta=0.05)
    @test one.ldos isa Float64 && one.converged isa Bool && one.maxlinkdim isa Int
    @test one.ldos ≈ -imag(gref(Hd, bv, 0.3 + 0.05im)) / π rtol=1e-6
    # the scan warm-starts each frequency itself: x0 is refused, not silently reused
    @test_throws ArgumentError get_ldos_cv(H, 20, ωs; eta=0.05, x0=b)
    # any collection of frequencies
    @test get_ldos_cv(H, 20, (0.3, 0.5); eta=0.05).ldos[1] ≈ one.ldos rtol=1e-8

    # a forced failure (maxdim 2) is flagged, and the scan warns
    bad = get_correction_vector(H, b, 0.3; eta=0.05, maxdim=2, seed_maxdim=2, rounds=1)
    @test !bad.converged && bad.residual > 0.02
    @test_logs (:warn, r"not converged") match_mode=:any get_ldos_cv(
        H, 20, [0.3]; eta=0.05, maxdim=2, seed_maxdim=2, rounds=1)

    # the isolated impurity level: energy and probe weight against dense eigenstates
    F = eigen(Hermitian(Hd))
    k = argmin(F.values)
    rb = refine_bound_state(H, b, -2.7)
    @test rb.energy ≈ F.values[k] atol=1e-8
    @test rb.weight ≈ abs2(dot(F.vectors[:, k], bv)) rtol=1e-6
    @test rb.converged && rb.energy_step < 1e-5
    @test rb.weight ≈ rb.history[end-1].weight rtol=1e-3   # stationary on a level
    # between the levels of this small chain (spacing ~0.1) the quotient crawls from step
    # to step, and `converged` says so although the last linear solve is converged
    rc = refine_bound_state(H, b, 0.3)
    @test !rc.converged && rc.residual < 0.02
    @test rc.energy_step > 1e-3
    # In a real continuum (2^16 sites, level spacing 1e-4) the energy step is about the
    # last η whatever ω0, and shows nothing: E − ω = η·Re G/Im G. The weight does: it is
    # proportional to η there, and stationary on a level.
    H16  = get_Hamiltonian("chain_1d", 1.0; L=16)
    rc16 = refine_bound_state(H16, probe_state(H16, 2^15 + 3), 0.3;
                              eta_steps=(0.05, 0.01, 1e-3))
    @test rc16.residual < 0.02 && rc16.energy_step < 0.01
    @test rc16.weight / rc16.history[end-1].weight ≈ 0.1 rtol=0.05
    @test !rc16.converged
    # a single step, or two equal broadenings at the end, cannot show a level
    @test !refine_bound_state(H, b, F.values[k]; eta_steps=(1e-3,)).converged
    @test !refine_bound_state(H, b, -2.7; eta_steps=(0.01, 1e-3, 1e-3)).converged
    # the broadenings are eta_steps: a stray `eta` would override every step
    @test_throws ArgumentError refine_bound_state(H, b, -2.7; eta=1e-2)
    @test_throws ArgumentError refine_bound_state(H, b, -2.7; eta_steps=())

    # Units: per unit physical energy, one state integrates to 1; the KPM LDOS times
    # π/H.scale is in the same units (16-site chain, both sum rules on one grid).
    H4 = get_Hamiltonian("chain_1d", 1.0; L=4)
    ωg = range(-6.0, 6.0; length=121)
    dω = step(ωg)
    cv = get_ldos_cv(H4, 5, ωg; eta=0.3)
    @test all(cv.converged)
    @test sum(cv.ldos) * dω ≈ 1 atol=0.06              # Lorentzian tails beyond ±6
    kpm = get_ldos_online(H4, 60, 5, ωg)
    @test sum(kpm) * dω * π / H4.scale ≈ 1 atol=0.02

    # Auxiliary sites: on a spinful chain the default sums the two spin sectors, and a
    # projection selects one.
    Hs = get_Hamiltonian("chain_1d", 1.0; L=4)
    add_spin!(Hs)
    ref = get_ldos_cv(H4, 5, 0.4; eta=0.2).ldos
    @test get_ldos_cv(Hs, 5, 0.4; eta=0.2).ldos ≈ 2ref rtol=1e-6
    @test get_ldos_cv(Hs, 5, 0.4; eta=0.2, spin_proj=true, proj_s=1).ldos ≈ ref rtol=1e-6

    # Exciton: contact probe |X, X⟩ on a 16-site chain (4 qubits per particle, 8
    # interleaved), attractive contact U = 2: a bound exciton below the continuum.
    He = exciton_hamiltonian("chain_1d", 1.0, x -> 2.0; L=4)
    se = He.sites
    Hed = ITensors.@disable_warn_order dmat(He.mpo, se)
    X  = 7
    be = mpsexciton(X, se); bev = dvec(be, se)
    ωe = range(-6.0, 1.0; length=6)
    re = get_exciton_ldos_cv(He, X, ωe; eta=0.05)
    @test all(re.converged)
    @test re.ldos ≈ [-imag(gref(Hed, bev, ω + 0.05im)) / π for ω in ωe] rtol=1e-6
    @test get_exciton_ldos_cv(He, X, ωe[2]; eta=0.05).ldos ≈ re.ldos[2] rtol=1e-6
    # get_ldos_cv on an exciton Hamiltonian uses the same contact probe
    @test get_ldos_cv(He, X, ωe; eta=0.05).ldos ≈ re.ldos rtol=1e-8
    @test_throws ArgumentError get_exciton_ldos_cv(H, 3, [0.0])

    # argument checks
    @test_throws ArgumentError get_correction_vector(H, b, 0.3; eta=0.0)
    @test_throws ArgumentError get_correction_vector(H, b, 0.3; solver=:cg)
    Hn = +(H.mpo, 0.3im * MPO(s, ["Z", "Id", "Id", "Id", "Id", "Id"]); cutoff=1e-14)
    @test_throws ArgumentError get_correction_vector(Hn, b, 0.3; solver=:minres)
end
