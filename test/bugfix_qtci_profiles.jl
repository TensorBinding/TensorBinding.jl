using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test, Random
using TensorBinding: get_mps, get_mpo, get_diagonal_mpo, extract_diagonal_to_mps, add_onsite!,
                     get_Hamiltonian

# get_mps / get_mpo / get_diagonal_mpo run QTCI from a few random pivots and checked nothing:
# at L = 10 a 4-site box on a 0.1 background was lost in 3 of 5 seeds, and a profile that
# vanishes almost everywhere threw "maxsamplevalue is zero!". A vector is now compressed
# by TT-SVD (exact up to tol), and `check=true` verifies a function's QTCI build (on the
# whole grid up to 2^20 points) and replaces a wrong one.

const _TBq = TensorBinding
_diag(M::MPO, sites) = _TBq._mps_dense_vector(extract_diagonal_to_mps(M), sites)

@testset "QTCI profiles: vectors by TT-SVD, opt-in check for functions" begin
    L = 10; N = 2^L; s = siteinds("Qubit", L)
    c = round(Int, 0.68N)                                   # an off-centre feature
    box(x)    = 0.1 + (c <= x <= c + 3 ? 1.0 : 0.0)          # 1-indexed x, as get_diagonal_mpo
    single(x) = x == c ? 1.0 : 0.0
    vbox = [box(x) for x in 1:N]

    @testset "vectors" begin
        @test maximum(abs.(_diag(get_diagonal_mpo(L, s, vbox), s) .- vbox)) < 1e-10
        vs = [single(x) for x in 1:N]
        @test maximum(abs.(_diag(get_diagonal_mpo(L, s, vs), s) .- vs)) < 1e-10
        vc = [cis(2π * 3x / N) + (x == c ? 0.5im : 0) for x in 1:N]      # complex
        @test maximum(abs.(_diag(get_diagonal_mpo(L, s, vc), s) .- vc)) < 1e-10
        @test maximum(abs.(_TBq._mps_dense_vector(get_mps(L, s, vbox), s) .- vbox)) < 1e-10
        @test maximum(abs.(_diag(get_mpo(L, s, vbox), s) .- vbox)) < 1e-10   # v[n+1] = f(n)
        s1 = siteinds("Qubit", 1)
        @test _TBq._mps_dense_vector(get_mps(1, s1, [2.0, -1.0]), s1) ≈ [2.0, -1.0]
        @test_throws DimensionMismatch get_diagonal_mpo(L, s, vbox[1:end-1])
        @test maxlinkdim(get_mps(L, s, vbox)) <= 4                         # a box is low rank
        # abstractly typed vectors
        s3 = siteinds("Qubit", 3)
        @test _TBq._mps_dense_vector(get_mps(3, s3, Any[1.0, 2, 3, 4, 5, 6, 7, 8]), s3) ≈ 1.0:8.0
        @test _TBq._mps_dense_vector(get_mps(3, s3, Real[1, 2.5, 3, 4, 5, 6, 7, 8]), s3) ≈ [1, 2.5, 3, 4, 5, 6, 7, 8]
        # L = 16: plain-array TT-SVD and readout, so ITensors prints no order-16 contraction warning
        L16 = 16; s16 = siteinds("Qubit", L16)
        v16 = [x == 40_000 ? 1.0 : 0.0 for x in 1:2^L16]
        path, io = mktemp()
        m16 = redirect_stdout(() -> get_mps(L16, s16, v16), io)
        d16 = redirect_stdout(() -> _TBq._mps_dense_vector(m16, s16), io)
        close(io)
        @test isempty(read(path, String)); rm(path)
        @test maximum(abs.(d16 .- v16)) < 1e-10
    end

    @testset "check=true on a function" begin
        # the seeds of the 2026-11-12 check, in which the unchecked build lost the box
        # (3 of 5) or threw on the single site (5 of 5)
        for seed in 1:5
            Random.seed!(seed)
            checked = _diag(get_diagonal_mpo(L, s, box; check=true), s)
            @test maximum(abs.(checked .- vbox)) < 1e-6
            Random.seed!(seed)
            @test maximum(abs.(_diag(get_diagonal_mpo(L, s, single; check=true), s) .-
                               [single(x) for x in 1:N])) < 1e-6
        end

        # A build that passes the check is returned as it is.
        sm(x) = cos(2π * x / N)
        Random.seed!(3); a = _diag(get_diagonal_mpo(L, s, sm), s)
        Random.seed!(3); b = _diag(get_diagonal_mpo(L, s, sm; check=true), s)
        @test a == b
        # An identically zero profile gives the zero MPS instead of throwing.
        @test all(iszero, _diag(get_diagonal_mpo(L, s, x -> 0.0; check=true), s))
        @test_throws ArgumentError get_mpo(3, siteinds("Qubit", 3), (i, j) -> 1.0; check=true)
    end

    @testset "add_onsite! with a vector or check=true" begin
        H  = get_Hamiltonian("chain_1d", 1.0; L=L)
        d0 = _diag(H.mpo, H.sites)
        Hv = deepcopy(H); add_onsite!(Hv, vbox)                 # v[n+1] at 0-indexed n
        @test maximum(abs.(_diag(Hv.mpo, Hv.sites) .- d0 .- vbox)) < 1e-8
        Hc = deepcopy(H); Random.seed!(1)
        add_onsite!(Hc, n -> box(n + 1); check=true)
        @test maximum(abs.(_diag(Hc.mpo, Hc.sites) .- d0 .- vbox)) < 1e-6

        # add_zeeman! takes a vector through get_diagonal_mpo: h[i]·S_z, S_z = σ_z/2
        Hz = get_Hamiltonian("chain_1d", 1.0; L=3); TensorBinding.add_spin!(Hz)
        dz = real(diag(_TBq._mpo_dense_matrix(Hz.mpo)))
        hv = [0.1x for x in 1:8]
        TensorBinding.add_zeeman!(Hz, hv)                    # sites [spin, pos…]: ↑ block first
        @test real(diag(_TBq._mpo_dense_matrix(Hz.mpo))) ≈ dz .+ [hv ./ 2; -hv ./ 2] atol=1e-10
    end

    # Above 2^20 points the check samples (ends, carry boundaries 2^k - 2 … 2^k + 1, a
    # golden-ratio stride) and rebuilds QTCI with the missed points as pivots.
    @testset "check=true above the full-grid limit" begin
        Lb = 21; Nb = 2^Lb; sb = siteinds("Qubit", Lb)
        fb(n) = 0.1 + (2^20 - 2 <= n <= 2^20 + 1 ? 1.0 : 0.0)    # 0-indexed, at a carry boundary
        Random.seed!(2)
        m = get_mps(Lb, sb, fb; check=true)
        for n in (0, 2^20 - 3, 2^20 - 2, 2^20, 2^20 + 1, 2^20 + 2, Nb - 1)
            @test abs(_TBq._mps_entry(m, sb, n) - fb(n)) < 1e-6
        end
    end
end
