using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test
using TensorBinding: get_Hamiltonian, honeycomb_positions, haldane_hoppingf

@testset "Haldane default scale bounds the spectrum" begin
    # The default scale was 4(1 + |t2| + |M|), below the Gershgorin bound 3 + 6|t2| + |M|
    # once |t2| > 0.5 + 1.5|M|: at phi = 0 or π the spectrum then left [-scale, scale]
    # (L = 7, t2 = 1, M = 0, phi = 0: radius 8.49 > 8).
    L  = 7
    rs = honeycomb_positions(L)
    # With M = 0 the QTCI's initial pivots can all land on zero entries ("maxsamplevalue is
    # zero!"), an intermittent failure separate from the scale; retry the build on it.
    build(p; kw...) = for _ in 1:10
        try
            return get_Hamiltonian("haldane", p; L=L, rs=rs, kw...)
        catch e
            e isa ErrorException && occursin("maxsamplevalue", e.msg) || rethrow()
        end
    end
    for (t2, M) in ((1.0, 0.0), (1.5, 0.0), (2.0, 0.5), (0.2, 0.3)), phi in (0.0, π/2, π)
        p = (t2=t2, phi=phi, M=M)
        H = build(p)
        A = ComplexF64[haldane_hoppingf(rs[i, :], rs[j, :], i, j; p...)
                       for i in 1:H.N, j in 1:H.N]
        @test H.scale > maximum(abs, eigvals(Hermitian(A)))
    end
    # An explicit scale is still passed through unchanged
    @test build((t2=1.0, phi=0.0, M=0.0); scale=2.5).scale == 2.5
end
