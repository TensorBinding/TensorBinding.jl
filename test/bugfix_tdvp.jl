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

    # Cost: the default seeds no pivots and runs TDVP once per sampled column j, keeping
    # U|j> for the other elements of that column, so at most N runs.  Before, every sampled
    # element was a TDVP run (L = 8: ~3600 seeded, ~700 unseeded).  use_diagonal_pivots =
    # true makes TCI sample every column: exactly N runs.
    H4 = get_Hamiltonian("chain_1d", 1.0; L = 4)
    U4, n4 = _tdvp_build(H4, dt)
    U0, n0 = _tdvp_build(H4, dt; use_diagonal_pivots = false)
    @test 0 < n4 <= H4.N && n4 == n0
    @test get_matrix(U4, H4.sites) == get_matrix(U0, H4.sites)
    H8 = get_Hamiltonian("chain_1d", 1.0; L = 8)
    _, n8  = _tdvp_build(H8, dt)
    _, n8d = _tdvp_build(H8, dt; use_diagonal_pivots = true)
    @test n8 < n8d == H8.N

    # reverse_step = false counted terms of H twice: the hop 1 <-> 2 (two qubits flip) came
    # out 1.5x too large (|dU| = 0.025 at L = 4).  The default is now true; false warns.
    Uex4 = exp(-im * dt * get_matrix(H4.mpo, H4.sites))
    M4   = get_matrix(U4, H4.sites)
    @test abs(M4[2, 3] - Uex4[2, 3]) < 1e-4
    @test abs(M4[3, 2] - Uex4[3, 2]) < 1e-4
    H3 = get_Hamiltonian("chain_1d", 1.0; L = 3)
    @test_logs (:warn, r"reverse_step=false") match_mode=:any build_tdvp_propagator_mpo(H3, dt; reverse_step = false)

    # TDVP cannot leave the tangent space of a bond-dimension-1 basis state, so the middle
    # hop N/2-1 <-> N/2 (011 <-> 100, all qubits flip) was dropped (|dU| = dt).  With the
    # Krylov expansion every element matches exp(-iH dt) to TDVP accuracy: each sampled
    # column is good to ~sqrt(cutoff) = 1e-4 (measured Frobenius 7e-5 / 8e-5 at L = 3 / 4,
    # against 0.10 / 0.16 before).
    U3, _ = _tdvp_build(H3, dt)
    for (H, U) in ((H3, U3), (H4, U4))
        Uex = exp(-im * dt * get_matrix(H.mpo, H.sites))
        M   = get_matrix(U, H.sites)
        c   = H.N ÷ 2
        @test abs(M[c, c + 1] - Uex[c, c + 1]) < 1e-4
        @test abs(M[c + 1, c] - Uex[c + 1, c]) < 1e-4
        @test norm(M - Uex) < 1e-3
    end

    # cache_columns = false re-runs TDVP for every sampled element; the samples, and so the
    # MPO, are the same.
    Unc, nnc = _tdvp_build(H4, dt; cache_columns = false)
    @test nnc > n4
    @test get_matrix(Unc, H4.sites) == get_matrix(U4, H4.sites)
end
