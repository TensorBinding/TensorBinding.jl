using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test
using TensorBinding: get_Hamiltonian, honeycomb_positions, square_positions, haldane_hoppingf,
                     H2DChernhex, matrix_checker

# Textbook Haldane model: H = Σ_ij H_ij c†_i c_j with H_ij = t2 exp(i φ ν_ij) between next-nearest
# neighbours, ν_ij = sign((d1 × d2)_z) for the path i → k → j through their common nearest
# neighbour k (d1 = r_k - r_i, d2 = r_j - r_k). It is C3-symmetric, with Dirac masses
# -M ± 3√3 t2 sin φ. The "haldane" and "chernhex" presets had ν flipped on the vertical
# next-nearest bonds only, which gave Dirac masses -M ± √3 t2 sin φ.

htb_cross(a, b) = a[1] * b[2] - a[2] * b[1]

# Dense ⟨i|H|j⟩ of an MPO, site 1 = most significant (checked against matrix_checker below).
function htb_dense(mpo, sites)
    T = ITensor(1.0)
    for k in eachindex(mpo); T *= mpo[k]; end
    N = prod(dim, sites)
    return reshape(Array(T, prime.(reverse(sites))..., reverse(sites)...), N, N)
end
htb_sites(mpo) = [only(filter(i -> plev(i) == 0, siteinds(mpo, k))) for k in eachindex(mpo)]
htb_dense(mpo) = htb_dense(mpo, htb_sites(mpo))

# Unit-distance graph of the positions, and the textbook ν of every pair with a common neighbour.
function htb_nbrs(rs)
    N = size(rs, 1)
    return [[j for j in 1:N if j != i && isapprox(norm(rs[j, :] .- rs[i, :]), 1; atol=1e-8)]
            for i in 1:N]
end
function htb_nu(rs, nbrs)
    ν = Dict{Tuple{Int,Int},Int}()
    for i in eachindex(nbrs), k in nbrs[i], j in nbrs[k]
        j == i && continue
        v = Int(sign(htb_cross(rs[k, :] .- rs[i, :], rs[j, :] .- rs[k, :])))
        @assert get!(ν, (i, j), v) == v
    end
    return ν
end

# Sublattice of a honeycomb_positions site: 1 (A) at x ∈ 1.5ℤ, 2 (B) at x ∈ 1.5ℤ + 1.
htb_sub(r) = isapprox(mod(r[1], 1.5), 1.0; atol=1e-6) ? 2 : 1

# Bloch Hamiltonian from the rows of one interior site per sublattice, in the periodic gauge
# (basis τ_A = 0, τ_B = (1, 0)), so that H(k + G) = H(k).
function htb_bloch(amp, rs, rows)
    τ = ([0.0, 0.0], [1.0, 0.0])
    terms = Tuple{Int,Int,ComplexF64,Vector{Float64}}[]
    for s in 1:2, j in axes(rs, 1)
        a = amp(rows[s], j)
        abs(a) > 1e-12 || continue
        s2 = htb_sub(rs[j, :])
        push!(terms, (s, s2, a, rs[j, :] .- rs[rows[s], :] .- (τ[s2] .- τ[s])))
    end
    return k -> begin
        Hk = zeros(ComplexF64, 2, 2)
        for (s, s2, a, R) in terms; Hk[s, s2] += a * cis(dot(k, R)); end
        Hk
    end
end
function htb_rows(rs, nbrs)
    full(i) = length(nbrs[i]) == 3 &&
              count(j -> isapprox(norm(rs[j, :] .- rs[i, :]), √3; atol=1e-6), axes(rs, 1)) == 6
    return [first(i for i in axes(rs, 1) if htb_sub(rs[i, :]) == s && full(i)) for s in 1:2]
end

# Dirac points of this layout (nearest-neighbour vectors (1, 0), (-1/2, ±√3/2)) and the
# reciprocal basis of a1 = (3/2, √3/2), a2 = (0, √3) (htb_cross(a1, a2) > 0).
htb_K() = (Float64[0, 4π / (3√3)], Float64[0, -4π / (3√3)])
htb_G() = (B = 2π * inv([1.5 √3/2; 0.0 √3]); (B[:, 1], B[:, 2]))
htb_masses(Hk) = sort([real(Hk(K)[1, 1] - Hk(K)[2, 2]) / 2 for K in htb_K()])

# Fukui-Hatsugai-Suzuki Chern number of the lower band
function htb_chern(Hk; n=30)
    b1, b2 = htb_G()
    u = [eigen(Hermitian(Hk((m - 1) / n .* b1 .+ (l - 1) / n .* b2))).vectors[:, 1]
         for m in 1:n, l in 1:n]
    U(a, b) = (z = dot(a, b); z / abs(z))
    F = 0.0
    for m in 1:n, l in 1:n
        mp, lp = mod1(m + 1, n), mod1(l + 1, n)
        F += angle(U(u[m, l], u[mp, l]) * U(u[mp, l], u[mp, lp]) *
                   U(u[mp, lp], u[m, lp]) * U(u[m, lp], u[m, l]))
    end
    return round(Int, F / 2π)
end

@testset "haldane preset is the textbook Haldane model" begin
    rs   = honeycomb_positions(6; Lx=3)            # 8 × 8 sites
    nbrs = htb_nbrs(rs)
    ν    = htb_nu(rs, nbrs)
    # every pair at distance √3 is a next-nearest pair with a common neighbour in the patch
    @test Set(keys(ν)) == Set((i, j) for i in axes(rs, 1), j in axes(rs, 1)
                                  if isapprox(norm(rs[i, :] .- rs[j, :]), √3; atol=1e-6))
    vertical(i, j) = isapprox(rs[i, 1], rs[j, 1]; atol=1e-8)
    @test count(vertical(i, j) for (i, j) in keys(ν)) > 0

    # 1. haldane_hoppingf bond by bond, vertical and diagonal next-nearest bonds alike
    for (t2, phi, M) in ((0.2, 0.7, 0.3), (0.1, π/2, 0.0), (0.3, -1.2, -0.4))
        f(i, j) = haldane_hoppingf(rs[i, :], rs[j, :], i, j; t2=t2, phi=phi, M=M)
        @test all(f(i, j) ≈ t2 * cis(phi * v) for ((i, j), v) in ν if vertical(i, j))
        @test all(f(i, j) ≈ t2 * cis(phi * v) for ((i, j), v) in ν if !vertical(i, j))
        @test all(f(i, j) == -1 for i in axes(rs, 1) for j in nbrs[i])
        @test all(f(i, i) ≈ (htb_sub(rs[i, :]) == 1 ? -M : M) for i in axes(rs, 1))
    end

    # 2. the MPO built by get_Hamiltonian("haldane") is the textbook matrix, entry by entry
    t2, phi, M = 0.2, 0.7, 0.3
    H = get_Hamiltonian("haldane", (t2=t2, phi=phi, M=M); L=6, rs=rs)
    D = htb_dense(H.mpo, H.sites)
    E = zeros(ComplexF64, H.N, H.N)
    for i in axes(rs, 1)
        E[i, i] = htb_sub(rs[i, :]) == 1 ? -M : M
        for j in nbrs[i]; E[i, j] = -1; end
    end
    for ((i, j), v) in ν; E[i, j] = t2 * cis(phi * v); end
    @test norm(D - E) < 1e-8 * norm(E)

    # 3. Dirac masses -M ± 3√3 t2 sin φ from the Bloch Hamiltonian of haldane_hoppingf
    rows = htb_rows(rs, nbrs)
    b1   = htb_G()[1]
    for (t2, phi, M) in ((0.1, π/2, 0.0), (0.2, 0.7, 0.3), (0.15, -1.1, -0.2))
        Hk = htb_bloch((i, j) -> haldane_hoppingf(rs[i, :], rs[j, :], i, j; t2=t2, phi=phi, M=M),
                       rs, rows)
        @test all(abs(Hk(K)[1, 2]) < 1e-10 for K in htb_K())      # the NN term vanishes at K, K'
        @test htb_masses(Hk) ≈ sort([-M + 3√3 * t2 * sin(phi), -M - 3√3 * t2 * sin(phi)])
        @test Hk(b1 ./ 7) ≈ Hk(b1 ./ 7)'                           # Hermitian
        @test Hk(htb_K()[1] .+ b1) ≈ Hk(htb_K()[1])                # periodic gauge
    end

    # 4. topological for |M| < 3√3 |t2 sin φ| (was √3 |t2 sin φ|); Chern sign as before the fix
    t2, phi = 0.1, π/2
    Mc = 3√3 * t2
    C(M, phi) = htb_chern(htb_bloch((i, j) -> haldane_hoppingf(rs[i, :], rs[j, :], i, j;
                                                              t2=t2, phi=phi, M=M), rs, rows))
    @test C(0.0, phi) == -1                 # the sign the old model had
    @test C(0.8Mc, phi) == -1 && C(-0.8Mc, phi) == -1
    @test C(1.2Mc, phi) == 0 && C(-1.2Mc, phi) == 0
    @test C(0.0, -phi) == 1
end

@testset "haldane preset refuses layouts its sign rules do not fit" begin
    p  = (t2=0.2, phi=0.7, M=0.3)
    rs = honeycomb_positions(4; Lx=2)
    for bad in (rs .+ [0.25 0.0],                          # sublattice rule reads x
                hcat(-rs[:, 2], rs[:, 1]),                 # rotated by 90°
                square_positions(4; Lx=2))                 # not a honeycomb
        err = try get_Hamiltonian("haldane", p; L=4, rs=bad); nothing catch e; e end
        @test err isa ArgumentError && occursin("honeycomb_positions", sprint(showerror, err))
    end
    err = try get_Hamiltonian("haldane", p; L=4, rs=honeycomb_positions(3; Lx=1)); nothing catch e; e end
    @test err isa ArgumentError                            # fewer rows than 2^L sites
    # a translation by a lattice vector keeps every site on the lattice and gives the same model
    H0 = get_Hamiltonian("haldane", p; L=4, rs=rs)
    H1 = get_Hamiltonian("haldane", p; L=4, rs=rs .+ [3.0 √3])
    @test htb_dense(H1.mpo, H1.sites) ≈ htb_dense(H0.mpo, H0.sites)
end

@testset "chernhex preset is the textbook Haldane model at φ = -π/2" begin
    Lx, Ly = 2, 3
    t, t2, ms = 1.0, 0.1, 0.05
    mpo = H2DChernhex(Lx, Ly, t, t2, ms; uniformhaldane=true, uniformsemenoff=true)
    D   = htb_dense(mpo)
    N   = size(D, 1)
    # ⟨i|H|j⟩ convention: htb_dense agrees with matrix_checker (inner(e_i, apply(H, e_j)), as
    # in get_matrix) on the row and column of an interior site, i.e. its NN and NNN both ways
    s, i0 = htb_sites(mpo), 14
    @test D[i0, :] ≈ [matrix_checker(mpo, s, i0 - 1, j - 1) for j in 1:N]
    @test D[:, i0] ≈ [matrix_checker(mpo, s, j - 1, i0 - 1) for j in 1:N]
    @test D ≈ D'

    # Index map from the nearest-neighbour structure: the entries of modulus t form exactly the
    # unit-distance graph of honeycomb_positions with row i ↔ index i. A planar honeycomb patch
    # has one embedding up to rotations and reflections, which change ν at most by a global sign.
    rs   = honeycomb_positions(Lx + Ly; Lx=Lx)
    nbrs = htb_nbrs(rs)
    nnD  = Set((i, j) for i in 1:N, j in 1:N if i != j && isapprox(abs(D[i, j]), t; atol=1e-8))
    @test nnD == Set((i, j) for i in 1:N for j in nbrs[i])
    @test all(D[i, j] ≈ t for (i, j) in nnD)
    ν = htb_nu(rs, nbrs)
    # the next-nearest entries sit exactly on the pairs with a common neighbour ...
    @test Set((i, j) for i in 1:N, j in 1:N if i != j && abs(D[i, j]) > 1e-8 && (i, j) ∉ nnD) ==
          Set(keys(ν))
    # ... with the textbook ν on every bond, times the builder's global convention -1 (φ = -π/2)
    vertical(i, j) = isapprox(rs[i, 1], rs[j, 1]; atol=1e-8)
    @test all(D[i, j] ≈ -im * t2 * v for ((i, j), v) in ν if vertical(i, j))
    @test all(D[i, j] ≈ -im * t2 * v for ((i, j), v) in ν if !vertical(i, j))
    @test all(D[i, i] ≈ (htb_sub(rs[i, :]) == 1 ? -ms : ms) for i in 1:N)

    # the "haldane" preset at φ = -π/2, M = ms, up to the gauge c → -c on sublattice B (t = 1)
    A = ComplexF64[haldane_hoppingf(rs[i, :], rs[j, :], i, j; t2=t2, phi=-π/2, M=ms)
                   for i in 1:N, j in 1:N]
    G = Diagonal([htb_sub(rs[i, :]) == 1 ? 1.0 : -1.0 for i in 1:N])
    @test D ≈ G * A * G

    # Dirac masses -ms ± 3√3 t2 and the transition at |ms| = 3√3 |t2| (was √3 |t2|)
    rows = htb_rows(rs, nbrs)
    @test htb_masses(htb_bloch((i, j) -> D[i, j], rs, rows)) ≈
          sort([-ms + 3√3 * t2, -ms - 3√3 * t2])
    Mc = 3√3 * t2
    C(m) = (Dm = htb_dense(H2DChernhex(Lx, Ly, t, t2, m; uniformhaldane=true, uniformsemenoff=true));
            htb_chern(htb_bloch((i, j) -> Dm[i, j], rs, rows)))
    @test C(ms) == 1        # the sign the old model had; opposite to "haldane" at φ = +π/2
    @test C(0.8Mc) == 1
    @test C(1.2Mc) == 0
end

@testset "chernhex default scale bounds the spectrum" begin
    Lx, Ly = 2, 3
    cases = ((0.3, 0.1, true, true), (1.0, 0.5, true, true), (1.5, -2.0, true, true),
             (0.3, 0.0, false, false), (1.5, -2.0, false, false), (1.0, 0.5, true, false),
             (0.8, 0.3, false, true))
    for (t2, ms, uh, us) in cases
        H = get_Hamiltonian("chernhex", (t=1.0, t2=t2, ms=ms, uniformhaldane=uh, uniformsemenoff=us);
                            L=Lx + Ly, Lx=Lx, Ly=Ly)
        ρ = maximum(abs, eigvals(Hermitian(htb_dense(H.mpo))))
        @test H.scale > ρ
        @test H.scale >= 6.0                                   # never below the old 6|t|
    end
    # parameters that reach the builder through `mparams` count too
    H = get_Hamiltonian("chernhex", (t2=0.1, ms=0.1); L=Lx + Ly, Lx=Lx, Ly=Ly, mparams="t=5.0")
    @test H.scale > maximum(abs, eigvals(Hermitian(htb_dense(H.mpo))))
    # an explicit scale is passed through unchanged
    @test get_Hamiltonian("chernhex", (t=1.0, t2=1.5, ms=-2.0); L=Lx + Ly, Lx=Lx, Ly=Ly,
                          scale=2.5).scale == 2.5
end
