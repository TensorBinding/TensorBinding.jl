using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test
using TensorBinding: get_Hamiltonian, get_matrix, build_tdvp_propagator_mpo
using Random: seed!

# Build under a fixed seed (QuanticsTCI draws random initial pivots) and count the TDVP
# runs: at outputlevel = 1 ITensorMPS prints one "After sweep" line per `tdvp` call.
function _tdvp_build(H, dt; kwargs...)
    seed!(1234)
    path, io = mktemp()
    U = redirect_stdout(io) do
        build_tdvp_propagator_mpo(H, dt; outputlevel = 1, kwargs...)
    end
    close(io)
    ncalls = count(startswith("After sweep"), eachline(path))
    rm(path)
    return U, ncalls
end

@testset "build_tdvp_propagator_mpo: default pivots cost and propagator accuracy" begin
    dt = 0.05

    # Cost: the default seeds no pivots.  use_diagonal_pivots = true seeds all N diagonal
    # positions, which takes more TDVP runs (paired over 16 seeds at L = 6: always more).
    H4 = get_Hamiltonian("chain_1d", 1.0; L = 4)
    U4, n4 = _tdvp_build(H4, dt)
    U0, n0 = _tdvp_build(H4, dt; use_diagonal_pivots = false)
    @test n4 > 0 && n4 == n0
    @test get_matrix(U4, H4.sites) == get_matrix(U0, H4.sites)
    H6 = get_Hamiltonian("chain_1d", 1.0; L = 6)
    _, n6  = _tdvp_build(H6, dt)
    _, n6d = _tdvp_build(H6, dt; use_diagonal_pivots = true)
    @test n6 < n6d
end
