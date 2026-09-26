using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test
using TensorBinding: get_Hamiltonian, get_C

# get_C accepts the ASCII keyword `Lambda` as an alias for the quenching period
# `Λ`; it used to be accepted and then ignored (the marker was always built with
# Λ=10).  Uniform Haldane-type Chern insulator on a 4×4 hexagonal patch.
@testset "get_C honours the Lambda alias" begin
    Lx, Ly = 2, 2
    Nx     = 2^Lx
    H  = get_Hamiltonian("chernhex", (t=1.0, t2=0.3, ms=0.1,
                                      uniformhaldane=true, uniformsemenoff=true);
                         L=Lx + Ly, Lx=Lx, Ly=Ly)
    xf = (i, _) -> Float64(mod(i, Nx))
    yf = (i, _) -> Float64(div(i, Nx))
    kw = (method=:mcweeny, maxdim=64, cutoff=1e-10)

    marker(C) = [C(α) for α in 1:H.N]
    C_Λ      = marker(get_C(H, xf, yf; Λ=2.5, kw...))
    C_Lambda = marker(get_C(H, xf, yf; Lambda=2.5, kw...))
    C_def    = marker(get_C(H, xf, yf; kw...))

    @test C_Lambda ≈ C_Λ
    @test !(C_Lambda ≈ C_def)
    # Lambda takes precedence over Λ when both are given (as in get_C_gpu)
    @test marker(get_C(H, xf, yf; Λ=10, Lambda=2.5, kw...)) ≈ C_Λ
end
