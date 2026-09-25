using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test
using TensorBinding: evolve_with_propagator, binary_to_MPS

# evolve_with_propagator called the bare name `truncate!` on an MPS. Inside the
# module that resolves to TensorBinding.truncate! (TBHamiltonian method only), so
# every call threw a MethodError. The propagator here is the exact e^{-iH dt}
# as an MPO, and the reference is the same evolution done densely.
@testset "evolve_with_propagator truncates with ITensorMPS.truncate!" begin
    L, dt, nsteps = 3, 0.1, 4
    N = 2^L
    H = TensorBinding.get_Hamiltonian("chain_1d", 1.0; L=L)
    s = H.sites

    U = MPO(exp(-im * dt * contract(H.mpo), prime.(s), s), s)

    basis = [binary_to_MPS(i - 1, L, s) for i in 1:N]
    amps(psi) = [inner(b, psi) for b in basis]
    Hm = [inner(basis[i]', H.mpo, basis[j]) for i in 1:N, j in 1:N]
    Ue = exp(-im * dt * Hm)

    psi0 = binary_to_MPS(N ÷ 2, L, s)
    v0 = amps(psi0)

    for normalize_each_step in (true, false)
        states = evolve_with_propagator(U, psi0, nsteps;
                                        normalize_each_step = normalize_each_step,
                                        cutoff = 1e-12)
        @test length(states) == nsteps + 1
        @test norm(amps(states[1]) - v0) < 1e-12
        for k in 0:nsteps
            @test abs(norm(states[k + 1]) - 1) < 1e-8
            @test norm(amps(states[k + 1]) - Ue^k * v0) < 1e-8
        end
    end
end
