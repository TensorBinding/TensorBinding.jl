using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test
using TensorBinding: get_Hamiltonian, honeycomb_positions, haldane_hoppingf, hopping2MPO,
                     _haldane_pivots, _check_haldane_mpo, bilayer_hamiltonian,
                     twisted_bilayer_hamiltonian, add_hopping!, add_hopping_2D!
using Random

# Dense matrix of a Qubit MPO (site 1 = most significant bit), as get_matrix but fast
function dense_tb3(mpo, sites)
    mid = length(mpo) ÷ 2   # contract two halves: no intermediate above order 2L
    Tl, Tr = ITensor(1.0), ITensor(1.0)
    for k in 1:mid; Tl *= mpo[k]; end
    for k in mid+1:length(mpo); Tr *= mpo[k]; end
    N = 2^length(sites)
    return reshape(Array(Tl * Tr, prime.(reverse(sites))..., reverse(sites)...), N, N)
end

@testset "Deterministic, exact Haldane MPO; layered add_hopping! error" begin
    # Haldane: the QTCI started from the all-ones pivot plus 5 random ones, so the build
    # depended on the global RNG, threw "maxsamplevalue is zero!" for M = 0 and often
    # missed a bond class (relative error 0.15-0.5 at L = 7, with a small TCI error).
    L  = 7
    rs = honeycomb_positions(L)
    for p in ((t2=0.6, phi=0.0, M=0.0), (t2=0.2, phi=0.7, M=0.3))
        A  = ComplexF64[haldane_hoppingf(rs[i, :], rs[j, :], i, j; p...) for i in 1:2^L, j in 1:2^L]
        Ds = map(1:5) do seed
            Random.seed!(seed)
            H = get_Hamiltonian("haldane", p; L=L, rs=rs)
            @test rand() == (Random.seed!(seed); rand())      # global RNG left untouched
            dense_tb3(H.mpo, H.sites)
        end
        @test all(D -> norm(D - A) / norm(A) < 1e-10, Ds)
        @test all(==(Ds[1]), Ds)                               # same MPO for every seed
    end
    # The spot check rejects a wrong MPO (here the Haldane MPO of other parameters)
    p, q = (t2=0.2, phi=0.7, M=0.3), (t2=0.2, phi=0.7, M=0.0)
    Hq = get_Hamiltonian("haldane", q; L=L, rs=rs)
    f(i, j) = haldane_hoppingf(rs[i, :], rs[j, :], i, j; p...)
    @test_throws ErrorException _check_haldane_mpo(Hq.mpo, Hq.sites, f, rs, _haldane_pivots(rs))
    # hopping2MPO: the new QTCI keywords default to QuanticsTCI's own values
    g(i, j) = abs(i - j) == 1 ? -1.0 : (i == j ? 0.1 * i : 0.0)
    s3 = siteinds("Qubit", 3)
    M1 = (Random.seed!(3); dense_tb3(hopping2MPO(g, 8, s3), s3))
    M2 = (Random.seed!(3); dense_tb3(hopping2MPO(g, 8, s3; nrandominitpivot=5, nsearchglobalpivot=5), s3))
    @test M1 == M2

    # Layered builders without a geometry: since they set Lx, add_hopping! reached
    # add_hopping_2D!, whose error asked for lattice=/geometry=, which add_hopping! lacks
    Random.seed!(2)                       # twisted builder: hopping2MPO random pivots
    for Hl in (bilayer_hamiltonian(:square, 2, 1), twisted_bilayer_hamiltonian(:square, 2, 1, 5.0))
        err = try add_hopping!(Hl, 0.1); nothing catch e; e end
        @test err isa ErrorException &&
              occursin("add_hopping! cannot be used on a layered Hamiltonian", err.msg)
        @test add_hopping_2D!(Hl, 0.1; Lx=Hl.Lx, Ly=Hl.L - Hl.Lx, lattice=:square) === Hl
    end
    # Layered sublattice builders carry a geometry, so add_hopping! still works there
    Hb = bilayer_hamiltonian(:honeycomb, 2, 1; sublattice=true)
    @test add_hopping!(Hb, 0.1) === Hb
end
