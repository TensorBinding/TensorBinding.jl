using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test
using TensorBinding: get_Hamiltonian, chern_marker

# chern_marker's quenching period is the ASCII keyword `Lambda`. Up to v0.1.1 the
# function was get_C, whose `Lambda` was at first accepted and then ignored (the marker
# was always built with Λ=10), and later silently won over `Λ` when both were given.
# `Λ` is now a deprecated alias (test/deprecations.jl), and passing both is an
# ArgumentError. Uniform Haldane-type Chern insulator on a 4×4 hexagonal patch.
@testset "chern_marker's Lambda keyword" begin
    Lx, Ly = 2, 2
    Nx     = 2^Lx
    H  = get_Hamiltonian("chernhex", (t=1.0, t2=0.3, ms=0.1,
                                      uniformhaldane=true, uniformsemenoff=true);
                         L=Lx + Ly, Lx=Lx, Ly=Ly)
    xf = (i, _) -> Float64(mod(i, Nx))
    yf = (i, _) -> Float64(div(i, Nx))
    kw = (method=:mcweeny, maxdim=64, cutoff=1e-10)

    marker(C) = [C(α) for α in 1:H.N]
    C_Lambda = marker(chern_marker(H, xf, yf; Lambda=2.5, kw...))
    C_def    = marker(chern_marker(H, xf, yf; kw...))
    C_10     = marker(chern_marker(H, xf, yf; Lambda=10, kw...))

    @test !(C_Lambda ≈ C_def)
    @test C_10 ≈ C_def                       # the default period is 10
    @test_throws ArgumentError chern_marker(H, xf, yf; Λ=10, Lambda=2.5, kw...)
end
