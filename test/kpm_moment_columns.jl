# CPU test of the KPM moment-column LDOS reconstruction
# (_reconstruct_ldos_moment_columns, solvers/kpm/kernels.jl). Moved verbatim
# from test/gpu_mps_ldos.jl, which still uses the helper on GPU moments.

using Test
using TensorBinding

@testset "KPM moment-column reconstruction (CPU)" begin
    @testset "moment-column reconstruction" begin
        moments = [1.0 2.0; 0.5 -1.0; -0.25 0.75]
        weights = [1.0 2.0 3.0; 0.5 -1.0 4.0; 2.0 0.25 -2.0]
        denom = [2.0, 4.0, 0.0]
        valid = [true, true, false]
        reconstructed = TensorBinding._reconstruct_ldos_moment_columns(
            moments, weights, denom, valid,
        )
        expected = transpose(weights) * moments
        expected[1, :] ./= denom[1]
        expected[2, :] ./= denom[2]
        expected[3, :] .= 0.0
        @test reconstructed == expected
        @test size(reconstructed) == (size(weights, 2), size(moments, 2))
        @test_throws DimensionMismatch TensorBinding._reconstruct_ldos_moment_columns(
            moments[1:2, :], weights, denom, valid,
        )
        @test_throws DimensionMismatch TensorBinding._reconstruct_ldos_moment_columns(
            moments, weights, denom[1:2], valid,
        )
        @test_throws DimensionMismatch TensorBinding._reconstruct_ldos_moment_columns(
            moments, weights, denom, valid[1:2],
        )
    end
end
