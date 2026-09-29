using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Random, Test
using TensorBinding: get_Hamiltonian, estimate_scale, honeycomb_positions

# The KPM scale maker (core/ModelRegistry.jl): estimate_scale with its :small, :geometry
# and :dmrg methods, get_Hamiltonian(...; scale=method), and the default scale
# max(today's formula, estimate_scale(:auto)) of "chain_1d", the preset models and the 2D
# multi-atom lattices.

const SM = TensorBinding

sm_dense(H) = SM._mpo_dense_matrix(H.mpo)
sm_radius(M) = maximum(abs, eigvals(Hermitian((M + M') / 2)))
sm_rowsum(M) = maximum(sum(abs, M; dims=2))

# Every registry geometry with a row-sum rule, at small sizes (all diagonalised directly).
const SM_CASES = [
    ("chain_1d", 1.0, (L=4,)), ("chain_1d", -0.7, (L=3, boundary=:periodic)),
    ("uniform", 1.0, (L=4,)), ("uniform", (t=1.0, v=0.8), (L=4,)),
    ("ssh", (t=1.0, d=0.3), (L=4,)), ("ssh", (t=0.5, d=1.5), (L=4,)),
    ("ssh", (t=1.0, d=0.4, nn=2), (L=4,)),
    ("aah", (V=0.5, phi=0.2, t=1.0), (L=4,)), ("aah", (V=2.0, phi=0.0, t=1.0), (L=5,)),
    ("aah", Dict(:V => 0.3, :phi => 0.1, :t => 1.0), (L=4,)),
    ("square_2d", 1.0, (L=4, Lx=2)), ("square_2d", 0.8, (L=5, Lx=3)),
    ("hex_2d", 1.0, (L=4, Lx=2)), ("triangular_2d", 1.0, (L=4, Lx=2)),
    ("triangular_bravais", 0.9, (L=4, Lx=2)),
    ("chern8", (V=0.4, t=1.0), (L=4, Lx=2)), ("chern8", (V=1.0, t=1.0, t2=1.0), (L=4, Lx=2)),
    ("chernhex", (t=1.0, t2=0.3, ms=0.1), (L=4, Lx=2)),
    ("qc2dsquare", 1.0, (L=4, Lx=2)),
    ("kagome", 1.0, (L=2, Lx=1, Ly=1)), ("lieb", 1.0, (L=3, Lx=2, Ly=1)),
    ("dice", 0.8, (L=2, Lx=1, Ly=1)), ("honeycomb", 1.0, (L=3, Lx=2, Ly=1)),
    ("honeycomb_nnn", (t=1.0, t2=0.3), (L=3, Lx=2, Ly=1)),
    ("ssh_sublattice", (t=1.0, d=0.3), (L=3,)),
]

# The builders' own formulas: kept by the :formula entries, and the `f` of
# max(f, :small) for the multi-atom lattices.
sm_formula(g, p) =
    g == "kagome" || g == "dice" ? 4.5 * SM._abs_t(p) :
    g == "lieb"                  ? 2.5 * SM._abs_t(p) :
    g == "honeycomb"             ? 3.5 * SM._abs_t(p) :
    g == "honeycomb_nnn"         ? 3.5 * abs(p.t) + 3.5 * abs(p.t2) :
    g == "ssh_sublattice"        ? (abs(p.t + p.d) + abs(p.t - p.d)) * 1.1 :
    g == "chernhex"              ? SM._chernhex_scale(p, "") : error("no formula for $g")

@testset "KPM scale maker" begin
    Random.seed!(11)
    for (g, p, kw) in SM_CASES
        entry = SM.MODELS[g]
        H = get_Hamiltonian(g, p; kw...)
        M = sm_dense(H)
        ρ = sm_radius(M)
        # the row-sum rule bounds every row of the builder's matrix (the 1e-5 margin
        # covered the ~1e-6 compression noise of the honeycomb builders, gone since 11a4e3c)
        @test entry.rowsum(p, kw) >= sm_rowsum(M) * (1 - 1e-5)
        # the default scale bounds the spectrum
        @test H.scale > ρ
        if entry.default_rule === :max
            # the multi-atom lattices' former default is their builder's formula
            legacy = entry.kind === :sublattice ? sm_formula(g, p) : SM._estimate_scale(g, p)
            est = estimate_scale(g, p; kw..., method=:auto)
            @test isapprox(H.scale, max(legacy, est); rtol=1e-8)
            @test H.scale >= legacy
        else
            @test H.scale == sm_formula(g, p)
        end
        # explicit methods
        @test get_Hamiltonian(g, p; kw..., scale=:small).scale ≈ 1.1 * ρ rtol=1e-6
        @test estimate_scale(g, p; kw..., method=:small) ≈ 1.1 * ρ rtol=1e-6
        sg = get_Hamiltonian(g, p; kw..., scale=:geometry).scale
        @test sg == 1.1 * entry.rowsum(p, kw) == estimate_scale(g, p; kw..., method=:geometry)
        @test sg > ρ
        @test get_Hamiltonian(g, p; kw..., scale=:geometry).center == 0.0
    end

    # :auto per model; the size-scaled presets take the padded row-sum bound
    @test get_Hamiltonian("chern8", (V=0.4, t=1.0); L=4, Lx=2).scale == 6.0   # bound 5.81 < 6
    @test get_Hamiltonian("chern8", (V=1.0, t=1.0); L=4, Lx=2).scale ≈ 1.1 * (4 + 16 * 0.2)
    @test get_Hamiltonian("qc2dsquare", 1.0; L=4, Lx=2).scale ≈ 10.56
    @test get_Hamiltonian("qc2dsquare", 2.0; L=4, Lx=2, mparams="").scale ≈ 21.12
    # the formula is kept where it already reaches 1.1 × the row-sum bound
    @test get_Hamiltonian("square_2d", 1.0; L=12, Lx=6).scale == 4.4
    @test get_Hamiltonian("chain_1d", 2.0; L=12).scale == 5.0
    # "aah": 1.2(|t| + |V|) = 1.8 is below the spectral radius at V = 0.5
    Ha = get_Hamiltonian("aah", (V=0.5, phi=0.2, t=1.0); L=4)
    @test Ha.scale > 1.8 && Ha.scale ≈ 1.1 * sm_radius(sm_dense(Ha))

    # :small above the dense limit: built at L = 10 (1D), cached per (params, sizes)
    @test SM._small_sizes(SM.MODELS["aah"], (12,)) == (10,)
    @test SM._small_sizes(SM.MODELS["ssh_sublattice"], (12,)) == (9,)
    @test SM._small_sizes(SM.MODELS["square_2d"], (6, 3)) == (5, 3)
    @test SM._small_sizes(SM.MODELS["kagome"], (5, 5)) == (4, 4)
    @test SM._small_sizes(SM.MODELS["honeycomb"], (6, 6)) == (4, 5)
    p = (V=0.37, phi=0.21, t=1.0)
    n0 = length(SM._SMALL_SCALE_CACHE)
    Random.seed!(3)
    s12 = estimate_scale("aah", p; L=12)
    @test rand() == (Random.seed!(3); rand())                  # caller's RNG untouched
    @test length(SM._SMALL_SCALE_CACHE) == n0 + 1
    @test estimate_scale("aah", p; L=11) === s12               # same small build, cached
    @test length(SM._SMALL_SCALE_CACHE) == n0 + 1
    Hs = get_Hamiltonian("aah", p; L=10)                       # L = 10 is diagonalised as is
    @test s12 ≈ 1.1 * sm_radius(sm_dense(Hs)) rtol=1e-6
    # the default rule draws nothing from the RNGs: same stream as an explicit scale
    for L in (4, 12)
        Random.seed!(5); get_Hamiltonian("aah", p; L, scale=3.3); a = rand()
        Random.seed!(5); H12 = get_Hamiltonian("aah", p; L);    b = rand()
        @test a == b
        # (at L = 4 the default diagonalises H itself, estimate_scale a build of its own)
        @test H12.scale ≈ max(SM._estimate_scale("aah", p), estimate_scale("aah", p; L)) rtol=1e-6
    end

    # a numeric scale passes through unchanged
    @test get_Hamiltonian("aah", p; L=4, scale=3.3).scale === 3.3
    @test get_Hamiltonian("kagome", 1.0; L=2, Lx=1, Ly=1, scale=7).scale === 7.0

    # :dmrg: DMRG spectral bounds of the full H, with its centre
    Hd = get_Hamiltonian("square_2d", 1.0; L=4, Lx=2, scale=:dmrg)
    E = eigvals(Hermitian(sm_dense(Hd)))
    @test Hd.center - Hd.scale <= E[1] + 1e-6 && Hd.center + Hd.scale >= E[end] - 1e-6
    @test estimate_scale("chain_1d", 1.0; L=4, method=:dmrg) ≈ 1.1 * maximum(abs, eigvals(
        Hermitian(sm_dense(get_Hamiltonian("chain_1d", 1.0; L=4))))) rtol=1e-4
    Hf = get_Hamiltonian("fibonacci", (A=1.0, B=2.0); L=4, scale=:dmrg)
    @test Hf.scale > 0 && Hf.position_space isa SM.FibonacciPositionSpace

    # analytic defaults: "haldane" is its padded row-sum bound, "custom" supports :dmrg
    rs = honeycomb_positions(4)
    ph = (t2=0.3, phi=0.4, M=0.2)
    Hh = get_Hamiltonian("haldane", ph; L=4, rs=rs)
    @test get_Hamiltonian("haldane", ph; L=4, rs=rs, scale=:geometry).scale == Hh.scale
    @test get_Hamiltonian("haldane", ph; L=4, rs=rs, scale=:small).scale ≈
          1.1 * sm_radius(sm_dense(Hh)) rtol=1e-8
    f(i, j) = abs(i - j) == 1 ? -1.0 : (i == j ? 0.1 * i : 0.0)
    @test get_Hamiltonian("custom", f; L=3, scale=:dmrg).scale > 0

    # unsupported methods are errors, raised before anything is built
    @test_throws ArgumentError get_Hamiltonian("fibonacci", (A=1.0, B=2.0); L=4, scale=:small)
    @test_throws ArgumentError get_Hamiltonian("fibonacci", (A=1.0, B=2.0); L=4, scale=:geometry)
    @test_throws ArgumentError get_Hamiltonian("custom", f; L=3, scale=:geometry)
    @test_throws ArgumentError get_Hamiltonian("haldane", ph; L=11, rs=rs, scale=:small)
    @test_throws ArgumentError get_Hamiltonian("square_2d", 1.0; L=4, scale=:exact)
    @test_throws ArgumentError estimate_scale("aah", p; L=4, method=:exact)
    @test_throws ErrorException estimate_scale("no_such_geometry", 1.0; L=4)
    @test_throws ErrorException estimate_scale("aah", (V=0.5, t=1.0); L=4, method=:geometry)
end
