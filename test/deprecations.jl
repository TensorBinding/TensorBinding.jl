using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test
using TensorBinding: get_Hamiltonian, add_onsite!, chern_marker, winding_marker,
                     valley_chern_marker, chern_marker_gpu, get_thouless_pump

# The 0.1 names and keywords renamed in 0.2 keep working through 0.2.x and go in 0.3:
# get_C, get_W, get_valley_C and get_C_gpu (chern_marker, winding_marker,
# valley_chern_marker, chern_marker_gpu), the keywords Nchebychev (Ncheb) and Λ (Lambda),
# and the value method=:KPM (:kpm). Each warns once per session with a `@warn` (visible
# in notebooks, unlike `Base.depwarn`) and gives the result of the new spelling; passing
# an old and a new keyword together is an ArgumentError. Under --depwarn=error the old
# spellings throw instead (a run with that flag shows that no internal caller still uses
# one), and this file checks the throw.

const TBD = TensorBinding

"Calls `old()`, expecting the deprecation warning `pattern`, and returns its result;
under --depwarn=error checks that it throws and returns `nothing`."
function deprecated_call(old, pattern)
    if Base.JLOptions().depwarn == 2
        @test_throws ErrorException old()
        return nothing
    end
    return @test_logs (:warn, pattern) match_mode = :any old()
end

"Dense matrix of an MPO over `sites` (site 1 fastest, rows = primed indices)."
dense_dep(M, s) = (D = prod(dim.(s)); reshape(Array(prod(M), prime.(s)..., s...), D, D))

function chernhex_dep()
    return get_Hamiltonian("chernhex", (t = 1.0, t2 = 0.3, ms = 0.1, uniformhaldane = true,
                                        uniformsemenoff = true); L = 4, Lx = 2, Ly = 2)
end
ssh_dep() = get_Hamiltonian("ssh_sublattice", (t = 1.0, d = -0.3); L = 3)
function semenoff_dep()
    H = get_Hamiltonian("honeycomb", 1.0; L = 2, Lx = 1, Ly = 1)
    add_onsite!(H, 0.4; sublat = 1)
    add_onsite!(H, -0.4; sublat = 2)
    H.scale = 3.5
    return H
end
gapped_dep() = (H = get_Hamiltonian("chain_1d", 1.0; L = 3, scale = 2.5);
                add_onsite!(H, n -> 0.4 * (-1)^n); H.scale = 2.8; H)

const XF_DEP = (i, _) -> Float64(mod(i, 4))
const YF_DEP = (i, _) -> Float64(div(i, 4))
marker_dep(C, n) = [C(a) for a in 1:n]

@testset "Deprecated names: one warning, the new function's result" begin
    kw = (; method = :mcweeny, maxdim = 64, cutoff = 1e-10)
    n  = chernhex_dep().N

    old = deprecated_call(() -> marker_dep(TBD.get_C(chernhex_dep(), XF_DEP, YF_DEP; kw...), n),
                          r"get_C is deprecated, use chern_marker")
    old === nothing ||
        @test old ≈ marker_dep(chern_marker(chernhex_dep(), XF_DEP, YF_DEP; kw...), n) rtol = 1e-12

    m = ssh_dep().N
    old = deprecated_call(() -> marker_dep(TBD.get_W(ssh_dep(); method = :mcweeny, maxdim = 30), m),
                          r"get_W is deprecated, use winding_marker")
    old === nothing ||
        @test old ≈ marker_dep(winding_marker(ssh_dep(); method = :mcweeny, maxdim = 30), m) rtol = 1e-12

    v = semenoff_dep().N
    old = deprecated_call(() -> marker_dep(TBD.get_valley_C(semenoff_dep(); valley = :K, maxdim = 40), v),
                          r"get_valley_C is deprecated, use valley_chern_marker")
    old === nothing ||
        @test old ≈ marker_dep(valley_chern_marker(semenoff_dep(); valley = :K, maxdim = 40), v) rtol = 1e-12

    # get_C_gpu forwards everything to chern_marker_gpu; passing both period keywords
    # makes the new function throw before any GPU work, with or without CUDA.
    if Base.JLOptions().depwarn == 2
        @test_throws ErrorException TBD.get_C_gpu(chernhex_dep(); Λ = 1, Lambda = 2)
    else
        @test_logs (:warn, r"get_C_gpu is deprecated, use chern_marker_gpu") match_mode = :any begin
            @test_throws ArgumentError TBD.get_C_gpu(chernhex_dep(); Λ = 1, Lambda = 2)
        end
    end
    @test_throws ArgumentError chern_marker_gpu(chernhex_dep(); Λ = 1, Lambda = 2)
end

@testset "Deprecated keywords and method=:KPM: one warning, the same result" begin
    kw = (; method = :mcweeny, maxdim = 64, cutoff = 1e-10)
    n  = chernhex_dep().N

    # Λ → Lambda
    old = deprecated_call(() -> marker_dep(chern_marker(chernhex_dep(), XF_DEP, YF_DEP; Λ = 2.5, kw...), n),
                          r"chern_marker: the keyword `Λ` is deprecated, use `Lambda`")
    old === nothing ||
        @test old ≈ marker_dep(chern_marker(chernhex_dep(), XF_DEP, YF_DEP; Lambda = 2.5, kw...), n) rtol = 1e-12
    @test_throws ArgumentError chern_marker(chernhex_dep(), XF_DEP, YF_DEP; Λ = 2.5, Lambda = 2.5, kw...)

    s = siteinds("Qubit", 3)
    xp = (i, N) -> Float64(i + 1)
    old = deprecated_call(() -> TBD.get_pump_xop(3, s, xp; quenched = true, Λ = 5.0),
                          r"get_pump_xop: the keyword `Λ` is deprecated, use `Lambda`")
    old === nothing ||
        @test dense_dep(old, s) == dense_dep(TBD.get_pump_xop(3, s, xp; quenched = true, Lambda = 5.0), s)

    H = semenoff_dep()
    P = TBD.mcweeny_purify(H; maxdim = 40)
    geom = H.geometry_uc
    xf = (i, _) -> geom(i + 1)[1]
    yf = (i, _) -> geom(i + 1)[2]
    old = deprecated_call(() -> marker_dep(TBD.get_C_op_MPO_from_P(P, H.L, H.sites, xf, yf; maxdim = 40, Λ = 3), H.N),
                          r"get_C_op_MPO_from_P: the keyword `Λ` is deprecated, use `Lambda`")
    old === nothing ||
        @test old ≈ marker_dep(TBD.get_C_op_MPO_from_P(P, H.L, H.sites, xf, yf; maxdim = 40, Lambda = 3), H.N) rtol = 1e-12

    # Nchebychev → Ncheb and method=:KPM → :kpm
    m = ssh_dep().N
    old = deprecated_call(() -> marker_dep(winding_marker(ssh_dep(); method = :kpm, Nchebychev = 40, maxdim = 30), m),
                          r"winding_marker: the keyword `Nchebychev` is deprecated, use `Ncheb`")
    new = marker_dep(winding_marker(ssh_dep(); method = :kpm, Ncheb = 40, maxdim = 30), m)
    old === nothing || @test old ≈ new rtol = 1e-12
    @test_throws ArgumentError winding_marker(ssh_dep(); Nchebychev = 40, Ncheb = 40)
    old = deprecated_call(() -> marker_dep(winding_marker(ssh_dep(); method = :KPM, Ncheb = 40, maxdim = 30), m),
                          r"method=:KPM is deprecated, use method=:kpm")
    old === nothing || @test old ≈ new rtol = 1e-12

    Hold = gapped_dep()
    Pold = deprecated_call(() -> TBD._get_projector(Hold; method = :KPM, Ncheb = 30, maxdim = 30),
                           r"method=:KPM is deprecated")
    Hn = gapped_dep()
    Pold === nothing ||
        @test dense_dep(Pold, Hold.sites) ≈ dense_dep(TBD._get_projector(Hn; method = :kpm, Ncheb = 30, maxdim = 30), Hn.sites)

    base = get_Hamiltonian("chain_1d", 1.0; L = 3, scale = 2.5)
    pump(t) = (H = deepcopy(base); add_onsite!(H, k -> 0.5 * cos(2π * (t + k / 3))); H.scale = 3.0; H)
    old = deprecated_call(() -> get_thouless_pump(pump, 3, 1.0, xp; maxdim = 30, quenched = true, Λ = 5.0),
                          r"get_thouless_pump: the keyword `Λ` is deprecated, use `Lambda`")
    old === nothing ||
        @test old ≈ get_thouless_pump(pump, 3, 1.0, xp; maxdim = 30, quenched = true, Lambda = 5.0) rtol = 1e-12
end
