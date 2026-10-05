using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Random, Test
using TensorBinding: get_Hamiltonian, build_hamiltonian, MODEL_REGISTRY, add_hopping_2D!,
                     get_shell_disps, mask_hamiltonian, hopping2MPO

# Lattice-builder fixes of the fourth audit round (docs/dev/REORGANISATION_TODO.md,
# "Found by the Tier 1 characterization sweep" and "Found by the Tier 2 scale maker").
# Every check compares with a dense ground truth built here from the bond lists.

const L4 = TensorBinding
l4_dense(M::MPO) = L4._mpo_dense_matrix(M)
l4_dense(H::L4.TBHamiltonian) = l4_dense(H.mpo)
l4_herm(M) = maximum(abs, M - M')

# Dense sublattice model on 2^Lx × 2^Ly open cells: `intra[a, b]` inside each cell and, for
# every bond (a, b, dx, dy, amp), ⟨a, cell + (dx, dy)|H|b, cell⟩ = amp plus its conjugate.
function l4_sublattice(Lx, Ly, nsub, intra, bonds)
    Nx, Ny = 2^Lx, 2^Ly
    H = zeros(ComplexF64, nsub * Nx * Ny, nsub * Nx * Ny)
    at(ix, iy, s) = nsub * (ix + iy * Nx) + s
    for ix in 0:Nx-1, iy in 0:Ny-1
        H[at(ix, iy, 1):at(ix, iy, nsub), at(ix, iy, 1):at(ix, iy, nsub)] .+= intra
        for (a, b, dx, dy, amp) in bonds
            (0 <= ix + dx < Nx && 0 <= iy + dy < Ny) || continue
            H[at(ix + dx, iy + dy, a), at(ix, iy, b)] += amp
            H[at(ix, iy, b), at(ix + dx, iy + dy, a)] += conj(amp)
        end
    end
    return H
end
l4_honeycomb(Lx, Ly, t, t2=0.0) = l4_sublattice(Lx, Ly, 2, ComplexF64[0 t; conj(t) 0],
    [(1, 2, 1, 0, t), (1, 2, 0, 1, t),
     ((s, s, dx, dy, t2) for s in 1:2 for (dx, dy) in ((1, 0), (0, 1), (1, -1)))...])
l4_kagome(Lx, Ly, a, c, b) = l4_sublattice(Lx, Ly, 3,
    ComplexF64[0 a c; conj(a) 0 b; conj(c) conj(b) 0],
    [(1, 2, 1, 0, a), (1, 3, 0, 1, c), (2, 3, -1, 1, b)])
l4_lieb(Lx, Ly, a, c) = l4_sublattice(Lx, Ly, 3, ComplexF64[0 a c; conj(a) 0 0; conj(c) 0 0],
    [(1, 2, 1, 0, a), (1, 3, 0, 1, c)])
l4_dice(Lx, Ly, a, c) = l4_sublattice(Lx, Ly, 3, ComplexF64[0 a 0; conj(a) 0 0; 0 0 0],
    [(1, 2, 1, 0, a), (1, 3, 1, 0, c), (1, 2, 0, 1, a), (1, 3, 0, 1, c), (1, 3, 1, 1, c)])

# Open square grid of 2^Lx × 2^Ly sites, bonds (dx, dy) with amplitude `amp(n)` at the
# source n (0-based) of the +(dx, dy) hop, plus the conjugate back.
function l4_grid(Lx, Ly, bonds)
    Nx, Ny = 2^Lx, 2^Ly
    H = zeros(ComplexF64, Nx * Ny, Nx * Ny)
    for ix in 0:Nx-1, iy in 0:Ny-1, (dx, dy, amp) in bonds
        (0 <= ix + dx < Nx && 0 <= iy + dy < Ny) || continue
        n, m = ix + iy * Nx, ix + dx + (iy + dy) * Nx
        H[m + 1, n + 1] += amp(n)
        H[n + 1, m + 1] += conj(amp(n))
    end
    return H
end

@testset "1. honeycomb builders match the dense model (no spurious entries)" begin
    # ITensors' density-matrix `+` projected on non-orthonormal LAPACK eigenvectors of a
    # degenerate spectrum: 1e-5 spurious entries at Lx = 2, Ly = 1, t = 1, any cutoff.
    for (Lx, Ly) in ((2, 1), (1, 2), (2, 2))
        @test l4_dense(L4.honeycomb_sublattice_hamiltonian(Lx, Ly, 1.0)) ≈
              l4_honeycomb(Lx, Ly, 1.0) atol=1e-12
        @test l4_dense(L4.honeycomb_nnn_hamiltonian(Lx, Ly, 1.0, 0.1)) ≈
              l4_honeycomb(Lx, Ly, 1.0, 0.1) atol=1e-12
    end
    @test l4_dense(get_Hamiltonian("honeycomb", 1.0; L=3, Lx=2, Ly=1)) ≈
          l4_honeycomb(2, 1, 1.0) atol=1e-12
    # the checked sum keeps a result that is right, and a truncation that removes weight
    Hk = L4.kagome_hamiltonian(2, 1; maxdim=4)
    @test maximum(ITensorMPS.linkdims(Hk.mpo)) <= 4
end

@testset "2. Hermitian for complex parameters" begin
    t = 0.7 + 0.3im
    @test l4_dense(L4.honeycomb_sublattice_hamiltonian(2, 1, t)) ≈ l4_honeycomb(2, 1, t) atol=1e-12
    @test l4_dense(L4.honeycomb_nnn_hamiltonian(1, 2, t, 0.1im)) ≈
          l4_honeycomb(1, 2, t, 0.1im) atol=1e-12
    # AA stacking: |1⟩⟨2| ⊗ t_inter Id + |2⟩⟨1| ⊗ conj(t_inter) Id
    Hb = L4.bilayer_hamiltonian(:square, 1, 1; t_inter=0.3im)
    Mb = l4_dense(Hb)
    @test l4_herm(Mb) < 1e-12
    @test Mb[1:4, 5:8] ≈ 0.3im * I(4) atol=1e-12          # layer index = most significant
    @test l4_herm(l4_dense(L4.multilayer_hamiltonian(:square, 1, 1, 3; t_inter=0.2 - 0.1im))) < 1e-12
    # legacy row / column hoppings (2^2 × 2^1 grid)
    s  = siteinds("Qubit", 3)
    w  = L4.get_diagonal_mpo(3, s, x -> 1.0 + 0.1im * x; type=ComplexF64)   # site x ↦ w(x)
    wn = n -> 1.0 + 0.1im * (n + 1)
    # intrachain: t K·B·W + h.c., the weight at the source; the default was not Hermitian
    # even for real t (backward row break on the wrong side: a row wrap-around bond)
    @test l4_dense(L4.intrachain_hopping(4, 8, s)) ≈ l4_grid(2, 1, [(1, 0, n -> 1.0)]) atol=1e-12
    @test l4_dense(L4.intrachain_hopping(4, 8, s; hopping=w, t=0.5 + 0.2im)) ≈
          l4_grid(2, 1, [(1, 0, n -> (0.5 + 0.2im) * wn(n))]) atol=1e-12
    # interchain: t W·K + h.c., the weight at the destination
    @test l4_dense(L4.interchain_hopping_square(4, 8, s; hopping=w, t=0.7im)) ≈
          l4_grid(2, 1, [(0, 1, n -> 0.7im * wn(n + 4))]) atol=1e-12
end

@testset "3. add_hopping_2D! rejects non-Bravais layouts" begin
    for (g, kw) in (("triangular_2d", (L=4, Lx=2)), ("hex_2d", (L=4, Lx=2)), ("hex_2d", (L=3, Lx=1)))
        H = get_Hamiltonian(g, 1.0; kw...)
        Lx, Ly = kw.Lx, kw.L - kw.Lx
        err = try add_hopping_2D!(H, 0.1; Lx, Ly); nothing catch e; e end
        @test err isa ErrorException && occursin("not a Bravais lattice", err.msg)
        @test_throws ErrorException redirect_stdout(() -> get_shell_disps(H, 1; Lx, Ly), devnull)
    end
    # Bravais layouts: the nn = 1 shell of triangular_bravais is the model's own bond set
    H  = get_Hamiltonian("triangular_bravais", 1.0; L=4, Lx=2)
    M0 = l4_dense(H)
    @test l4_dense(add_hopping_2D!(H, 0.1; Lx=2, Ly=2)) ≈ 1.1 * M0 atol=1e-8
    # two rows of triangular_2d are a strip of a Bravais lattice: still accepted
    H2 = get_Hamiltonian("triangular_2d", 1.0; L=3, Lx=2)
    M2 = l4_dense(H2)
    @test l4_dense(add_hopping_2D!(H2, 0.1; Lx=2, Ly=1)) ≈ 1.1 * M2 atol=1e-8
end

@testset "4. sdf_convex_polygon is positive inside" begin
    ccw = L4.sdf_convex_polygon([(0.0, 0.0), (3.0, 0.0), (0.0, 3.0)])
    cw  = L4.sdf_convex_polygon([(0.0, 0.0), (0.0, 3.0), (3.0, 0.0)])
    for (x, y) in ((0.5, 0.5), (1.0, 0.2), (5.0, 5.0), (-1.0, 0.5), (1.0, -2.0))
        @test ccw(x, y) ≈ cw(x, y)
    end
    @test ccw(0.5, 0.5) ≈ 0.5                      # distance to the nearest edge
    @test ccw(1.0, 0.2) ≈ 0.2
    @test ccw(5.0, 5.0) < 0 && ccw(-1.0, 0.5) ≈ -1.0
    @test sign(ccw(0.5, 0.5)) == sign(L4.sdf_disk(0.5, 0.5, 0.2)(0.5, 0.5)) == 1
    @test_throws ErrorException L4.sdf_convex_polygon([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)])
    # a flake cut with the documented (CCW) order keeps the inside
    H  = get_Hamiltonian("square_2d", 1.0; L=4, Lx=2)
    Hm = mask_hamiltonian(H, L4.sdf_convex_polygon([(-0.5, -0.5), (2.0, -0.5), (-0.5, 2.0)]);
                          sigma=0.05)
    M = l4_dense(Hm)
    @test abs(M[1, 2]) > 0.99          # (0,0)–(1,0) inside
    @test abs(M[3, 4]) < 1e-6          # (2,0)–(3,0) outside
end

@testset "5. mask_hamiltonian on sublattice lattices; complex kagome/Lieb/dice" begin
    H  = get_Hamiltonian("kagome", 1.0; L=2)                  # 2 × 2 cells, 12 atoms
    sdf = L4.sdf_disk(0.5, 0.5, 0.8)
    Hm = mask_hamiltonian(H, sdf; sigma=0.3)
    m  = [1 / (1 + exp(-sdf(H.geometry(i)...) / 0.3)) for i in 1:12]
    @test l4_dense(Hm) ≈ Diagonal(m) * l4_dense(H) * Diagonal(m) atol=1e-7
    @test Hm.sublattice_s == H.sublattice_s
    a, b, c = 0.5im, 0.8 + 0.1im, -0.6
    @test l4_dense(L4.kagome_hamiltonian(2, 1; t_AB=a, t_AC=b, t_BC=c)) ≈
          l4_kagome(2, 1, a, b, c) atol=1e-12
    @test l4_dense(L4.lieb_hamiltonian(2, 1; t_AB=a, t_AC=b)) ≈ l4_lieb(2, 1, a, b) atol=1e-12
    @test l4_dense(L4.dice_hamiltonian(2, 1; t_AB=a, t_AC=b)) ≈ l4_dice(2, 1, a, b) atol=1e-12
    @test l4_herm(l4_dense(get_Hamiltonian("kagome", 0.3 + 0.4im; L=2))) < 1e-12
end

@testset "6. build_shift_mpo has no unreachable positional default" begin
    s = siteinds("Qubit", 3)
    @test l4_dense(L4.build_shift_mpo(s, 2)) == l4_dense(L4.build_shift_mpo(s, 2, false))
    @test l4_dense(L4.build_shift_mpo(s, 2; cyclic=true)) == l4_dense(L4.build_shift_mpo(s, 2, true))
    @test !hasmethod(L4.build_shift_mpo, Tuple{Vector{Index{Int}}, Float64})
end

@testset "7. identically zero QTCI fields leave their term out" begin
    # uniform Semenoff mass: ms = 0 threw "maxsamplevalue is zero!"
    kw = (; uniformhaldane=true, uniformsemenoff=true)
    Random.seed!(1); M0 = l4_dense(L4.H2DChernhex(2, 1, 1.0, 0.2, 0.0; kw...))
    Random.seed!(1); M3 = l4_dense(L4.H2DChernhex(2, 1, 1.0, 0.2, 0.3; kw...))
    semenoff = [isodd(n % 4 + n ÷ 4) ? 0.3 : -0.3 for n in 0:7]
    @test M3 - M0 ≈ Diagonal(semenoff) atol=1e-8
    Mnn = l4_dense(L4.H2DChernhex(2, 1, 1.0, 0.0, 0.0))         # no NNN, no mass
    @test Mnn ≈ l4_dense(L4.HUniform2Dhex(2, 1, 1.0)) atol=1e-8
    @test get_Hamiltonian("chernhex", (t=1.0, t2=0.2, ms=0.0);
                          L=3, Lx=2, mparams="uniformsemenoff=true") isa L4.TBHamiltonian
    # add_onsite!(H, 0): H.mpo unchanged, caches invalidated as for any term
    H = get_Hamiltonian("chain_1d", 1.0; L=3)
    M = l4_dense(H)
    @test L4.add_onsite!(H, 0.0) === H && l4_dense(H) == M && H.scale == 0.0
    Hk = get_Hamiltonian("kagome", 1.0; L=2)
    Mk = l4_dense(Hk)
    @test l4_dense(L4.add_onsite!(Hk, 0; sublat=2)) == Mk
    @test_throws ErrorException L4.add_onsite!(get_Hamiltonian("chain_1d", 1.0; L=3), 0.0; sublat=1)
    Hl = L4.bilayer_hamiltonian(:square, 1, 1)
    Ml = l4_dense(Hl)
    @test l4_dense(L4.add_onsite!(Hl, 0.0; layer=1)) == Ml
    # the other presets with a vanishing field
    chain = l4_grid(3, 0, [(1, 0, n -> 1.0)])
    @test l4_dense(L4.HUniform(3, 1.0; v=0.0)) ≈ chain atol=1e-8
    @test l4_dense(L4.HAAH(3, 0.0, 0.3, 1.0)) ≈ chain atol=1e-8
    @test l4_dense(L4.HChern8(2, 1, 0.0, 1.0)) ≈
          l4_grid(2, 1, [(1, 0, n -> 1.0), (0, 1, n -> (-1)^((n % 4) + 1))]) atol=1e-8
end

@testset "8. custom QTCI hoppings: sampled self-check with a structural rebuild" begin
    chain(i, j) = abs(i - j) == 1 ? -1.0 : 0.0
    grid(i, j) = (a = Int(i) - 1; b = Int(j) - 1;
                  abs(a % 8 - b % 8) + abs(a ÷ 8 - b ÷ 8) == 1 ? -1.0 : 0.0)
    for (f, L) in ((chain, 8), (chain, 10), (grid, 6)), seed in (1, 3)
        # (chain, L = 8, seed = 1) threw "maxsamplevalue is zero!"; the others were wrong
        Random.seed!(seed)
        H = get_Hamiltonian("custom", f; L, scale=4.5)
        N = 2^L
        @test l4_dense(H) ≈ [f(i, j) for i in 1:N, j in 1:N] atol=1e-10
    end
    H = get_Hamiltonian("chain_1d", 1.0; L=6)
    L4.add_hopping!(H, (i, j) -> abs(i - j) == 2 ? 0.15 : 0.0)
    @test l4_dense(H) ≈ [abs(i - j) == 1 ? 1.0 : abs(i - j) == 2 ? 0.15 : 0.0
                         for i in 1:64, j in 1:64] atol=1e-8
    # a build that passes the check is returned as it is
    f(i, j) = abs(i - j) == 1 ? -1.0 : (i == j ? 0.1 * i : 0.0)
    s = siteinds("Qubit", 3)
    Random.seed!(2); A = l4_dense(hopping2MPO(f, 8, s))
    Random.seed!(2); B = l4_dense(hopping2MPO(f, 8, s; check=true))
    @test A == B
    # a vanishing f is a clear error; check=false keeps QTCI's own
    zero_f(i, j) = 0.0
    err = try hopping2MPO(zero_f, 8, s; check=true); nothing catch e; e end
    @test err isa ErrorException && occursin("vanishes at all", err.msg)
    @test_throws ErrorException hopping2MPO(zero_f, 8, s)
end

@testset "9. dice bands reach ±3√2 t" begin
    E = eigvals(Hermitian(l4_dice(3, 3, 1.0, 1.0)))
    @test 4.1 < maximum(abs, E) <= 3sqrt(2) + 1e-12
    @test count(e -> abs(e) < 1e-10, E) >= 64            # the flat band (64 cells)
end

@testset "10. chern8 t2 default, Lieb cell positions, ref_sites" begin
    # "chern8" leaves t2 to HChern8, whose default is 0.2t
    @test !haskey(MODEL_REGISTRY["chern8"][4], :t2)
    Random.seed!(4); A = l4_dense(build_hamiltonian("chern8", 2, 1; mparams="V=0.5, t=2.0"))
    Random.seed!(4); B = l4_dense(L4.HChern8(2, 1, 0.5, 2.0))
    Random.seed!(4); C = l4_dense(L4.HChern8(2, 1, 0.5, 2.0; t2=0.2))
    @test A ≈ B atol=1e-12
    @test !(A ≈ C)
    # Lieb: square Bravais cells
    Hl = get_Hamiltonian("lieb", 1.0; L=3, Lx=2, Ly=1)
    @test [Hl.geometry_uc(3n + s) for n in (0, 1, 4) for s in (1, 3)] ==
          [[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    @test Hl.geometry_uc(13) == Hl.geometry(13)           # atom A of cell (0, 1)
    Hk = get_Hamiltonian("kagome", 1.0; L=3, Lx=2, Ly=1)
    @test Hk.geometry_uc(13) ≈ [0.5, sqrt(3) / 2]         # triangular for the others
    # ref_sites is honoured (position qubits) or rejected, never ignored
    ref = siteinds("Qubit", 3)
    for (g, p, kw) in (("chain_1d", 1.0, (;)), ("custom", (i, j) -> abs(i - j) == 1 ? 1.0 : 0.0, (scale=2.5,)),
                       ("kagome", 1.0, (Lx=2, Ly=1)), ("honeycomb_nnn", (t=1.0, t2=0.1), (Lx=1, Ly=2)),
                       ("ssh_sublattice", (t=1.0, d=0.2), (;)))
        H0 = get_Hamiltonian(g, p; L=3, kw...)
        H  = get_Hamiltonian(g, p; L=3, ref_sites=ref, kw...)
        @test L4._pos_sites(H) == ref
        @test all(hasind(H.mpo[k], H.sites[k]) for k in eachindex(H.sites))
        @test l4_dense(H) ≈ l4_dense(H0) atol=1e-10
    end
    rs = L4.honeycomb_positions(3)
    Hh = get_Hamiltonian("haldane", (t2=0.2, phi=π / 2, M=0.1); L=3, rs=rs, ref_sites=ref)
    @test Hh.sites == ref
    @test_throws ArgumentError get_Hamiltonian("kagome", 1.0; L=3, ref_sites=siteinds("Qubit", 2))
    @test_throws ArgumentError get_Hamiltonian("chain_1d", 1.0; L=3,
                                               ref_sites=[Index(3, "x") for _ in 1:3])
end

@testset "11. multi-atom default scales bound the spectrum" begin
    # default = max(builder formula, :small estimate); the formulas 2.5|t| (Lieb) and
    # 3.5(|t| + |t2|) (honeycomb_nnn) were below the spectral radius
    radius(H) = maximum(abs, eigvals(Hermitian(l4_dense(H))))
    for (g, p, kw, formula) in (("lieb", 1.0, (L=3, Lx=2, Ly=1), 2.5),
                                ("lieb", 1.0, (L=6, Lx=3, Ly=3), 2.5),
                                ("dice", 1.0, (L=6, Lx=3, Ly=3), 4.5),
                                ("honeycomb_nnn", (t=1.0, t2=0.3), (L=9, Lx=5, Ly=4), 3.5 * 1.3))
        H = get_Hamiltonian(g, p; kw...)
        ρ = radius(H)
        @test H.scale > ρ
        @test H.scale ≈ max(formula, 1.1 * ρ) rtol=1e-8      # all diagonalised as they are
    end
    # the formula stays where it already reaches 1.1 × the row-sum bound
    @test get_Hamiltonian("kagome", 1.0; L=4, Lx=2).scale == 4.5
    @test get_Hamiltonian("honeycomb", 0.8; L=4, Lx=2).scale == 3.5 * 0.8
    @test get_Hamiltonian("honeycomb_nnn", (t=1.0, t2=0.05); L=4, Lx=2).scale == 3.5 * 1.0 + 3.5 * 0.05
    @test get_Hamiltonian("lieb", 1.0; L=4, Lx=2, scale=2.0).scale == 2.0
end
