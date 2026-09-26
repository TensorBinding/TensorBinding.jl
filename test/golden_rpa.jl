using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test, Random
using TensorBinding: get_Hamiltonian, add_spin!, add_zeeman!, TBHamiltonian, replace_sites,
                     mpo_kron, interleave_mpo, interleave_mpo_tb, swap_every_other_legs,
                     collapse_mpo_pairs, rpa_from_bubble_diag, get_bubble_mpo,
                     get_rpa_susceptibility, wynn_epsilon, rpa_wynn_from_bubbles, haydock_cf,
                     eval_haydock_cf, haydock_resolve_mpo, get_bubble_mpo_haydock, get_spect_k,
                     get_rpa_susceptibility_wynn, get_magnon_bubble, get_magnon_susceptibility,
                     get_magnon_susceptibility_wynn, chebyshev2d_gf_coeffs, get_bubble_mpo_cheb2d,
                     get_bubble_mpo_cheb2d_tucker, get_bubble_diag_cheb2d,
                     get_bubble_diag_cheb2d_svd, get_bubble_diag_cheb2d_tucker

# Characterization ("golden") tests for src/physics/RPA_tk.jl.
#
# These tests pin what the RPA code computes *today*, bugs included, so that the
# Tier 1 reorganisation (docs/dev/REORGANISATION_TODO.md: split RPA_tk.jl into
# physics/rpa/{Bubble,Cheb2D,Dyson}.jl, move the MPO kron/interleave plumbing to
# core/Utils.jl, the Haydock recursion to solvers/Krylov.jl and get_spect_k
# to the QFT conjugation file) cannot silently change an output. The expected
# values live in `test/data/rpa_golden.jl`, written by
# `test/data/generate_rpa_golden.jl` (see its header for how to rerun it).
#
# A failure here means an output changed. If the change is a regression, fix
# the code. If it is intentional, regenerate the data file in the same commit
# as the behaviour change and record it in the changelog.
#
# Every case is (name, fn, setup, seed, args, kwargs): `setup` names an input
# builder below (Hamiltonians and MPOs cannot be written as literals), `fn`
# selects the call in `run_case`, `args`/`kwargs` are the plain values passed on.
# Both RNGs a case can touch are seeded with `seed` right before the inputs are
# built: the global RNG (random_mps in the DMRG spectral-bound estimate, QTCI
# pivots) and ITensors' Index-id RNG (so that fresh indices are reproducible).
# Hand-built MPOs are filled from a closed formula (`det_mpo`), not from an RNG.
#
# Comparison rules:
#   * every field recorded in the golden data must still be returned;
#   * floating-point scalars and arrays: same size and element type, and every
#     entry isapprox(rtol=RTOL, atol=ATOL) on its own -- entry by entry, not
#     norm-wise, so that a small entry next to a large one (the 1e30 that
#     wynn_epsilon returns once a sequence has converged exactly, say) is pinned
#     to its own size; NaN/Inf entries must sit at the same positions with the
#     same value;
#   * everything else (Int, Bool, Symbol, String, types, `nothing`): `==` and the
#     same type;
#   * a case recorded as throwing must still throw that exception type with the
#     same first MESSAGE_PREFIX_CHARS characters of its message;
#   * the number of cases per function must equal EXPECTED_CASE_COUNTS.
#
# MPO outputs are stored as dense matrices (rows = primed/bra indices, columns =
# unprimed/ket indices, site 1 fastest), MPS outputs as dense vectors, together
# with length, site dimensions, element type and, where the result is meant to
# live on known indices, whether it does (`on_ref_sites`). Bond dimensions are
# not pinned.
#
# The drivers that hand `verbose` on to get_bubble_mpo (VERBOSE_FORWARDING_FNS)
# also record `bubble_progress_printed`, whether get_bubble_mpo's progress lines
# ("Polarization bubble: ...") reached standard output: `verbose` changes no
# number, so this is the value a dropped `verbose` changes. Their other
# forwarded keywords are pinned by cases in which each one changes the output.
#
# The runner below is shared with the generator, which includes this file with
# `RPA_GOLDEN_GENERATOR` defined so that only the module is loaded.

module RPAGoldenRunner

using Test, Random, LinearAlgebra
using ITensors, ITensorMPS
using TensorBinding
const TB = TensorBinding

const RTOL = 1e-10
const ATOL = 1e-12
const MESSAGE_PREFIX_CHARS = 60

# Number of cases per function in test/data/rpa_golden.jl. Update it by hand, in
# the same commit, when cases are added or removed on purpose (the generator
# prints the new counts and warns when they differ from these).
const EXPECTED_CASE_COUNTS = Dict{Symbol,Int}(
    :_build_heff                        => 1,
    :_cheb2d_out_sites                  => 2,
    :_cheb2d_require_position_sites     => 3,
    :_get_density_matrix                => 8,
    :_jackson_kernel                    => 3,
    :_project_spin_sector               => 4,
    :_rpa_pair_sites                    => 2,
    :_weighted_mpo_sum                  => 3,
    :chebyshev2d_gf_coeffs              => 2,
    :collapse_mpo_pairs                 => 2,
    :eval_haydock_cf                    => 3,
    :get_bubble_diag_cheb2d             => 6,
    :get_bubble_diag_cheb2d_svd         => 7,
    :get_bubble_diag_cheb2d_tucker      => 7,
    :get_bubble_mpo                     => 14,
    :get_bubble_mpo_cheb2d              => 6,
    :get_bubble_mpo_cheb2d_tucker       => 7,
    :get_bubble_mpo_haydock             => 3,
    :get_magnon_bubble                  => 4,
    :get_magnon_susceptibility          => 6,
    :get_magnon_susceptibility_wynn     => 6,
    :get_rpa_susceptibility             => 9,
    :get_rpa_susceptibility_wynn        => 9,
    :get_spect_k                        => 3,
    :haydock_cf                         => 4,
    :haydock_resolve_mpo                => 2,
    :interleave_mpo                     => 4,
    :interleave_mpo_tb                  => 3,
    :mpo_kron                           => 1,
    :rpa_from_bubble_diag               => 3,
    :rpa_wynn_from_bubbles              => 3,
    :swap_every_other_legs              => 1,
    :wynn_epsilon                       => 7,
)

# ── Deterministic inputs ────────────────────────────────────────────────────────

"""
    det_mpo(sites; χ=2, phase=0.0, ketfirst=false, realpart=false) -> MPO

A dense, non-Hermitian MPO on `sites` with bond dimension `χ`, filled entry by
entry with `cos(0.37k + phase) + im*sin(0.91k + 2phase)` (k = 1, 2, …). Each site
tensor stores its indices as (s', s, links…), or (s, s', links…) with
`ketfirst=true`: some helpers read `siteinds(M, n)` in storage order.
"""
function det_mpo(sites; χ::Int=2, phase::Real=0.0, ketfirst::Bool=false, realpart::Bool=false)
    L     = length(sites)
    links = [Index(χ, "Link,l=$n") for n in 1:L-1]
    k     = 0
    tensors = ITensor[]
    for n in 1:L
        s     = sites[n]
        sinds = ketfirst ? (s, prime(s)) : (prime(s), s)
        linds = Index[]
        n > 1 && push!(linds, links[n-1])
        n < L && push!(linds, links[n])
        dims  = (dim.(sinds)..., dim.(linds)...)
        vals  = Vector{ComplexF64}(undef, prod(dims))
        for j in eachindex(vals)
            k += 1
            vals[j] = cos(0.37k + phase) + im * sin(0.91k + 2phase)
        end
        A = reshape(realpart ? real.(vals) : vals, dims)
        push!(tensors, ITensor(A, sinds..., linds...))
    end
    return MPO(tensors)
end

sub3(n) = Index(3, "Site,Sub,n=$n")
interleaved(a, b) = reduce(vcat, [[x, y] for (x, y) in zip(a, b)])
position_sites(H) = filter(i -> !hastags(i, "Spin"), H.sites)
Vid(sites, u=0.5) = u * MPO(sites, "Id")

# Plain real chain; `H.scale` fixed so that no DMRG bound estimate is needed.
chain(L; scale=2.2) = TB.get_Hamiltonian("chain_1d", 1.0; L, scale)

# Chain + 0.3·Y on qubit 2 + 0.2·Z on qubit 3: Hermitian, complex, not symmetric
# (so a transposed embedding shows up), with a nonzero spectral center.
function chain_cplx(L=3)
    H = chain(L)
    s = H.sites
    ops_y = ["Id" for _ in 1:L]; ops_y[2] = "Y"
    ops_z = ["Id" for _ in 1:L]; ops_z[min(3, L)] = "Z"
    H.mpo = +(H.mpo, 0.3 * MPO(ComplexF64, s, ops_y), 0.2 * MPO(s, ops_z); cutoff=1e-14)
    H.scale, H.center = 2.8, 0.1
    return H
end

# A second Hamiltonian on the same sites as `H1` (the H1 !== H2 branches).
function partner(H1)
    s   = H1.sites
    ops = ["Id" for _ in 1:H1.L]; ops[1] = "Z"
    kin = TB.replace_sites(chain(H1.L).mpo, s)
    return TB.TBHamiltonian(H1; mpo=+(kin, 0.25 * MPO(s, ops); cutoff=1e-14),
                         scale=2.6, center=-0.05)
end

function spinful(L; h=0.4, direction=:z, position=:pre, scale=2.8)
    H = chain(L)
    position === :post && TB.add_spin!(H; position=:post)
    TB.add_zeeman!(H, h; direction)
    H.scale, H.center = scale, 0.0
    return H
end

"""
    build_setup(setup) -> NamedTuple

The non-literal inputs of a case. Called right after the RNGs are seeded.
"""
function build_setup(setup::Symbol)
    setup === :none && return (;)
    # ── plumbing helpers ──
    if setup === :kron
        sA = siteinds("Qubit", 2); sB = [sub3(1)]
        return (; A=det_mpo(sA; phase=0.1), B=det_mpo(sB; phase=0.7), ref=[sA; sB])
    elseif setup === :ileave2
        return (; op=det_mpo(siteinds("Qubit", 2)), phys=siteinds("Qubit", 4))
    elseif setup === :ileave2_ketfirst
        return (; op=det_mpo(siteinds("Qubit", 2); ketfirst=true), phys=siteinds("Qubit", 4))
    elseif setup === :ileave1
        return (; op=det_mpo(siteinds("Qubit", 1); phase=0.2), phys=siteinds("Qubit", 2))
    elseif setup === :ileave_tb_het
        sA = [siteind("Qubit", 1), sub3(2)]; sB = sim.(sA)
        return (; op=det_mpo(sA; phase=0.3), sA, sB, ref=interleaved(sA, sB))
    elseif setup === :ileave_tb_q
        sA = siteinds("Qubit", 2); sB = sim.(sA)
        return (; op=det_mpo(sA; phase=0.4), sA, sB, ref=interleaved(sA, sB))
    elseif setup === :swap4
        return (; M=det_mpo(siteinds("Qubit", 4); phase=0.5),
                newsites=[Index(2, "New,n=$i") for i in 1:4])
    elseif setup === :collapse4
        return (; M=det_mpo(siteinds("Qubit", 4); phase=0.6),
                out=[Index(2, "Out,n=$i") for i in 1:2])
    elseif setup === :collapse_het
        q = siteinds("Qubit", 2)
        return (; M=det_mpo([q[1], q[2], sub3(3), sub3(4)]; phase=0.7),
                out=[Index(2, "Out,n=1"), Index(3, "Out,n=2")])
    elseif setup === :pair_q
        return (; out=siteinds("Qubit", 2))
    elseif setup === :pair_het
        return (; out=[siteinds("Qubit", 2); sub3(3)])
    elseif setup === :dyson_q
        out = siteinds("Qubit", 2)
        return (; Π=0.3 * det_mpo(out; phase=0.8), V=Vid(out), out)
    elseif setup === :dyson_het
        out = [siteind("Qubit", 1), sub3(2)]
        return (; Π=0.3 * det_mpo(out; phase=0.9), V=Vid(out), out)
    elseif setup === :hay_chain2
        H = chain(2); s = H.sites
        seed = +(MPO(s, ["Z", "Id"]), 0.5 * MPO(s, ["X", "X"]); cutoff=1e-14)
        return (; H1=H, seed)
    elseif setup === :hay_chain2_id
        H = chain(2)
        return (; H1=H, seed=MPO(H.sites, "Id"))
    elseif setup === :hay_chain2_imag
        # Hermitian, purely imaginary seed Y ⊗ I (Frobenius norm 2).
        H = chain(2)
        return (; H1=H, seed=MPO(ComplexF64, H.sites, ["Y", "Id"]))
    elseif setup === :spect_H
        return (; W=chain_cplx(3).mpo)
    elseif setup === :spect_det
        return (; W=det_mpo(siteinds("Qubit", 3); phase=1.1))
    elseif setup === :wynn_bubbles
        q = siteinds("Qubit", 3)
        return (; Πs=[0.3 * det_mpo(q; phase=0.0), 0.3 * det_mpo(q; phase=0.5)], V=Vid(q))
    elseif setup === :wsum
        s = siteinds("Qubit", 2)
        return (; mpos=[det_mpo(s; phase=p) for p in (0.0, 0.3, 0.6)], ref=s)
    elseif setup === :wsum_real
        s = siteinds("Qubit", 2)
        return (; mpos=[det_mpo(s; phase=p, realpart=true) for p in (0.0, 0.3, 0.6)], ref=s)
    # ── Hamiltonians ──
    elseif setup === :chain2_cplx
        H1 = chain_cplx(2)
        return (; H1, H2=partner(H1))
    elseif setup === :chain3
        H = chain(3)
        return (; H1=H, H2=H, V=Vid(H.sites))
    elseif setup === :chain3_cplx
        H = chain_cplx(3); s = H.sites
        V = +(Vid(s), 0.2 * MPO(s, ["Z", "Id", "Id"]); cutoff=1e-14)
        return (; H1=H, H2=H, V)
    elseif setup === :chain3_pair
        H1 = chain_cplx(3)
        return (; H1, H2=partner(H1), V=Vid(H1.sites))
    elseif setup === :chain3_cached
        # A density matrix already cached on H: the :purification path must return it.
        H = chain_cplx(3)
        H._density_cache = +(Vid(H.sites), 0.1 * MPO(H.sites, ["Z", "X", "Id"]); cutoff=1e-14)
        return (; H1=H, H2=H, cache=H._density_cache)
    elseif setup === :chain3_vs_chain2
        return (; H1=chain(3), H2=chain(2))
    elseif setup === :chain2
        # The smallest inputs, for the cases that run a driver with its default keywords.
        H = chain(2)
        return (; H1=H, H2=H, V=Vid(H.sites))
    elseif setup === :spin2_z
        H = spinful(2)
        return (; H1=H, H2=H, V=Vid(position_sites(H)))
    elseif setup === :spin3_z
        H = spinful(3)
        return (; H1=H, H2=H, V=Vid(position_sites(H)))
    elseif setup === :spin3_z_post
        H = spinful(3; position=:post)
        return (; H1=H, H2=H, V=Vid(position_sites(H)))
    elseif setup === :spin2_y
        H = spinful(2; h=0.3, direction=:y, scale=2.6)
        return (; H1=H, H2=H, V=Vid(position_sites(H)))
    elseif setup === :spin2_vs_chain2
        return (; H1=spinful(2; h=0.3, direction=:y, scale=2.6), H2=chain(2))
    elseif setup === :kagome2
        H = TB.get_Hamiltonian("kagome", 1.0; L=2, Lx=1, Ly=1)
        H.scale, H.center = 4.5, 0.0
        return (; H1=H, H2=H, V=Vid(H.sites))
    end
    error("RPAGoldenRunner: unknown setup :$setup")
end

# ── Records of outputs ─────────────────────────────────────────────────────────

_eltype(M) = mapreduce(eltype, promote_type, M)

function _site_pair(M::MPO, n::Int)
    s = siteinds(M, n)
    ket = only(filter(i -> plev(i) == 0, s))
    bra = only(filter(i -> plev(i) == 1, s))
    return bra, ket
end

# Site tensor n of an MPS/MPO as a plain array (phys..., left links, right links),
# reshaped to (dims of phys..., χ_left, χ_right). The dense contraction is done in
# plain Julia below rather than with prod(M): contracting the whole chain in
# ITensors compiles NDTensors kernels for every tensor rank met (4 to 12 here),
# about 10 s of extra compilation in a cold run of this file.
function _site_array(M, n::Int, phys::Vector{<:Index})
    L      = length(M)
    lefts  = n > 1 ? collect(commoninds(M[n-1], M[n])) : Index[]
    rights = n < L ? collect(commoninds(M[n], M[n+1])) : Index[]
    known  = Index[phys; lefts; rights]
    length(inds(M[n])) == length(known) && all(i -> i in known, inds(M[n])) ||
        error("RPAGoldenRunner: site $n has indices other than its site and link indices")
    W = Array(M[n], known...)
    return reshape(W, dim.(phys)..., prod(dim, lefts; init=1), prod(dim, rights; init=1))
end

# Dense matrix of an MPO: rows = bra, columns = ket, site 1 fastest.
function _dense_mpo(M::MPO)
    A = ones(Float64, 1, 1, 1)                      # (D_bra, D_ket, χ)
    for n in 1:length(M)
        bra, ket = _site_pair(M, n)
        W = _site_array(M, n, [bra, ket])           # (d_bra, d_ket, χl, χr)
        Db, Dk, χ = size(A)
        χ == size(W, 3) || error("RPAGoldenRunner: link dimensions do not match at site $n")
        B = zeros(promote_type(eltype(A), eltype(W)), Db, size(W, 1), Dk, size(W, 2), size(W, 4))
        for r in axes(W, 4), l in axes(W, 3), k in axes(W, 2), b in axes(W, 1)
            w = W[b, k, l, r]
            iszero(w) && continue
            @views B[:, b, :, k, r] .+= A[:, :, l] .* w
        end
        A = reshape(B, Db * size(W, 1), Dk * size(W, 2), size(W, 4))
    end
    return A[:, :, 1]
end

# Dense vector of an MPS, site 1 fastest.
function _dense_mps(ψ::MPS)
    A = ones(Float64, 1, 1)                         # (D, χ)
    for n in 1:length(ψ)
        W = _site_array(ψ, n, [only(siteinds(ψ, n))])   # (d, χl, χr)
        D, χ = size(A)
        χ == size(W, 2) || error("RPAGoldenRunner: link dimensions do not match at site $n")
        B = zeros(promote_type(eltype(A), eltype(W)), D, size(W, 1), size(W, 3))
        for r in axes(W, 3), l in axes(W, 2), a in axes(W, 1)
            @views B[:, a, r] .+= A[:, l] .* W[a, l, r]
        end
        A = reshape(B, D * size(W, 1), size(W, 3))
    end
    return A[:, 1]
end

"""
    mpo_record(M; ref=nothing) -> NamedTuple

Length, site dimensions, element type and dense matrix of `M` (rows = bra,
columns = ket, site 1 fastest); with `ref`, also whether the ket indices of `M`
are exactly `ref`.
"""
function mpo_record(M::MPO; ref=nothing)
    kets = [last(_site_pair(M, n)) for n in 1:length(M)]
    rec  = (; length=length(M), site_dims=dim.(kets), eltype=_eltype(M), dense=_dense_mpo(M))
    return ref === nothing ? rec : merge(rec, (; on_ref_sites=kets == collect(ref)))
end

function mps_record(ψ::MPS; ref=nothing)
    s   = [only(siteinds(ψ, n)) for n in 1:length(ψ)]
    rec = (; length=length(ψ), site_dims=dim.(s), eltype=_eltype(ψ), dense=_dense_mps(ψ))
    return ref === nothing ? rec : merge(rec, (; on_ref_sites=s == collect(ref)))
end

function list_record(xs::AbstractVector; ref=nothing)
    recs = [x isa MPO ? mpo_record(x; ref) : mps_record(x; ref) for x in xs]
    return (; count=length(xs), types=Symbol[Symbol(nameof(typeof(x))) for x in xs],
            site_dims=[r.site_dims for r in recs], eltypes=DataType[r.eltype for r in recs],
            dense=[r.dense for r in recs],
            on_ref_sites=ref === nothing ? Bool[] : Bool[r.on_ref_sites for r in recs])
end

index_record(v) = (; dims=dim.(v), tags=[replace(string(tags(i)), "\"" => "") for i in v],
                   plevs=plev.(v))

wynn_record((cp, cw)) = (; chi_partial=cp, chi_wynn=cw)

# ── One call per case ──────────────────────────────────────────────────────────

# The drivers that forward `verbose` to get_bubble_mpo, and the prefix of the
# progress lines get_bubble_mpo prints with verbose=true.
const VERBOSE_FORWARDING_FNS = (:get_rpa_susceptibility, :get_rpa_susceptibility_wynn,
                                :get_magnon_bubble, :get_magnon_susceptibility,
                                :get_magnon_susceptibility_wynn)
const BUBBLE_PROGRESS = "Polarization bubble:"

"""
    run_case(fn, setup, seed, args, kwargs) -> NamedTuple

Seed both RNGs, build the inputs named by `setup`, make the call named by `fn`
and return a record of its output. Standard output (progress prints of the
KPM/DMRG helpers) is discarded; for the drivers in VERBOSE_FORWARDING_FNS it is
first searched for get_bubble_mpo's progress lines (`bubble_progress_printed`).
"""
function run_case(fn::Symbol, setup::Symbol, seed::Int,
                  @nospecialize(args::Tuple), @nospecialize(kwargs::NamedTuple))
    Random.seed!(seed)
    Random.seed!(ITensors.index_id_rng(), seed)
    fn in VERBOSE_FORWARDING_FNS || return redirect_stdout(devnull) do
        _run_case(fn, build_setup(setup), args, kwargs)
    end
    path, io = mktemp()
    try
        rec = redirect_stdout(io) do
            r = _run_case(fn, build_setup(setup), args, kwargs)
            flush(stdout)
            r
        end
        close(io)
        printed = occursin(BUBBLE_PROGRESS, read(path, String))
        return merge(rec, (; bubble_progress_printed=printed))
    finally
        isopen(io) && close(io)
        rm(path; force=true)
    end
end

function _run_case(fn::Symbol, @nospecialize(S::NamedTuple),
                   @nospecialize(args::Tuple), @nospecialize(kw::NamedTuple))
    # ---- MPO plumbing ----
    if fn === :mpo_kron
        return mpo_record(TB.mpo_kron(S.A, S.B); ref=S.ref)
    elseif fn === :interleave_mpo                      # args = (n,)
        return mpo_record(TB.interleave_mpo(S.op, S.phys, args[1]); ref=S.phys)
    elseif fn === :interleave_mpo_tb                   # args = (which,)
        return mpo_record(TB.interleave_mpo_tb(S.op, S.sA, S.sB, args[1]); ref=S.ref)
    elseif fn === :swap_every_other_legs
        return mpo_record(TB.swap_every_other_legs(S.M, S.newsites); ref=S.newsites)
    elseif fn === :collapse_mpo_pairs
        return mpo_record(TB.collapse_mpo_pairs(S.M, S.out); ref=S.out)
    elseif fn === :_rpa_pair_sites
        return index_record(TB._rpa_pair_sites(S.out))
    elseif fn === :_build_heff
        sites2 = sim.(S.H2.sites)
        Heff = TB._build_heff(S.H1.mpo, TB.replace_sites(S.H2.mpo, sites2), S.H1.sites, sites2)
        return mpo_record(Heff; ref=interleaved(S.H1.sites, sites2))
    elseif fn === :_get_density_matrix
        # kw holds _get_density_matrix's positional arguments by name
        P = TB._get_density_matrix(S.H1, kw.ϵF, kw.P_method, kw.Ncheb, kw.maxdim, kw.cutoff,
                                   kw.purify_method, kw.purify_maxdim, kw.purify_maxiters,
                                   kw.purify_tol, kw.verbose)
        cache = get(S, :cache, nothing)
        return merge(mpo_record(P; ref=S.H1.sites),
                     (; returned_old_cache=cache !== nothing && P === cache,
                        cached_on_H=S.H1._density_cache === P))
    elseif fn === :rpa_from_bubble_diag
        χ = TB.rpa_from_bubble_diag(S.Π, S.V, TB._rpa_pair_sites(S.out), S.out; kw...)
        return mps_record(χ)
    # ---- Wynn ε and Haydock recursion ----
    elseif fn === :wynn_epsilon
        return (; value=TB.wynn_epsilon(args[1]))
    elseif fn === :eval_haydock_cf
        return (; value=TB.eval_haydock_cf(args...))
    elseif fn === :haydock_cf                          # args = (N_steps,)
        a, b, basis, norm0 = TB.haydock_cf(S.H1.mpo, S.seed, args[1]; kw...)
        return (; a, b, norm0, nbasis=length(basis),
                basis=[mpo_record(B; ref=S.H1.sites).dense for B in basis])
    elseif fn === :haydock_resolve_mpo                 # args = (N_steps, nkeep, z)
        a, b, basis, _ = TB.haydock_cf(S.H1.mpo, S.seed, args[1]; maxdim=20, cutoff=1e-10)
        n = args[2]
        return mpo_record(TB.haydock_resolve_mpo(a[1:n], b[1:n], basis[1:n], args[3]; kw...);
                          ref=S.H1.sites)
    elseif fn === :get_bubble_mpo_haydock              # args = (ωlist,)
        return list_record(TB.get_bubble_mpo_haydock(S.H1, S.H2, args[1]; kw...); ref=S.H1.sites)
    elseif fn === :get_spect_k
        return (; value=TB.get_spect_k(S.W; kw...))
    # ---- bubbles and Dyson solves ----
    elseif fn === :get_bubble_mpo                      # args = (ω,)
        return mpo_record(TB.get_bubble_mpo(S.H1, S.H2, args[1]; kw...); ref=S.H1.sites)
    elseif fn === :get_rpa_susceptibility              # args = (ω,)
        return mps_record(TB.get_rpa_susceptibility(S.H1, S.V, args[1]; kw...))
    elseif fn === :get_magnon_bubble                   # args = (ω,)
        return mpo_record(TB.get_magnon_bubble(S.H1, args[1]; kw...); ref=position_sites(S.H1))
    elseif fn === :get_magnon_susceptibility           # args = (ω,)
        return mps_record(TB.get_magnon_susceptibility(S.H1, S.V, args[1]; kw...))
    elseif fn === :rpa_wynn_from_bubbles
        return wynn_record(TB.rpa_wynn_from_bubbles(S.Πs, S.V; kw...))
    elseif fn === :get_rpa_susceptibility_wynn         # args = (ωlist,)
        return wynn_record(TB.get_rpa_susceptibility_wynn(S.H1, S.V, args[1]; kw...))
    elseif fn === :get_magnon_susceptibility_wynn      # args = (ωlist,)
        return wynn_record(TB.get_magnon_susceptibility_wynn(S.H1, S.V, args[1]; kw...))
    elseif fn === :_project_spin_sector                # args = (sector,)
        Hp = TB._project_spin_sector(S.H1, args[1])
        return merge(mpo_record(Hp.mpo; ref=position_sites(S.H1)),
                     (; L=Hp.L, N=Hp.N, nsites=length(Hp.sites),
                        sites_are_position=Hp.sites == position_sites(S.H1),
                        scale=Hp.scale, center=Hp.center, spin_s_is_nothing=Hp.spin_s === nothing,
                        aux_side=Hp.aux_side, Lx=Hp.Lx,
                        caches_empty=Hp._density_cache === nothing && Hp._tn_cache === nothing))
    # ---- double Chebyshev ----
    elseif fn === :chebyshev2d_gf_coeffs
        return (; value=TB.chebyshev2d_gf_coeffs(args...))
    elseif fn === :_jackson_kernel
        return (; value=TB._jackson_kernel(args[1]))
    elseif fn === :_weighted_mpo_sum                   # args = (weights,)
        r = TB._weighted_mpo_sum(args[1], S.mpos; kw...)
        return r === nothing ? (; is_nothing=true) :
               merge(mpo_record(r; ref=S.ref), (; is_nothing=false))
    elseif fn === :_cheb2d_out_sites                   # args = (fname,)
        out = TB._cheb2d_out_sites(S.H1, S.H2, args[1])
        return merge(index_record(out), (; fresh=!any(in(S.H1.sites), out)))
    elseif fn === :_cheb2d_require_position_sites      # args = (fname,)
        return (; value=TB._cheb2d_require_position_sites(S.H1, S.H2, args[1]))
    elseif fn in (:get_bubble_mpo_cheb2d, :get_bubble_mpo_cheb2d_tucker,
                  :get_bubble_diag_cheb2d, :get_bubble_diag_cheb2d_svd,
                  :get_bubble_diag_cheb2d_tucker)      # args = (ωlist,)
        f = getfield(TB, fn)
        return list_record(f(S.H1, S.H2, args[1]; kw...); ref=S.H1.sites)
    end
    error("RPAGoldenRunner: unknown fn :$fn")
end

# ── Comparison ────────────────────────────────────────────────────────────────

"""
    approx_equal(actual, expected) -> Bool

Floating-point numbers and arrays: same type/size/element type and isapprox
with RTOL/ATOL, entry by entry for arrays, non-finite entries equal and in
place. Containers of other things: element by element. Anything else: `isequal`
and the same type.
"""
approx_equal(@nospecialize(a), @nospecialize(b)) = typeof(a) == typeof(b) && isequal(a, b)
approx_equal(a::Integer, b::Integer) = typeof(a) == typeof(b) && a == b
function approx_equal(a::Number, b::Number)
    typeof(a) == typeof(b) || return false
    isfinite(a) && isfinite(b) || return isequal(a, b)
    return isapprox(a, b; rtol=RTOL, atol=ATOL)
end
# Entry by entry: a norm-wise isapprox would let every entry much smaller than the
# largest one (next to a 1e30 Wynn sentinel, all of them) drift unchecked.
entry_matches(x::Number, y::Number) =
    isfinite(x) && isfinite(y) ? isapprox(x, y; rtol=RTOL, atol=ATOL) : isequal(x, y)
function approx_equal(a::AbstractArray{<:Number}, b::AbstractArray{<:Number})
    size(a) == size(b) && eltype(a) == eltype(b) || return false
    return all(i -> entry_matches(a[i], b[i]), eachindex(a, b))
end
function approx_equal(a::AbstractArray, b::AbstractArray)
    size(a) == size(b) || return false
    return all(approx_equal(x, y) for (x, y) in zip(a, b))
end
approx_equal(a::Tuple, b::Tuple) =
    length(a) == length(b) && all(approx_equal(x, y) for (x, y) in zip(a, b))
approx_equal(a::NamedTuple, b::NamedTuple) =
    keys(a) == keys(b) && all(approx_equal(a[k], b[k]) for k in keys(b))

function _detail(@nospecialize(a), @nospecialize(b))
    if a isa AbstractArray && b isa AbstractArray
        size(a) == size(b) || return "size $(size(a)) != expected $(size(b))"
        if !(a isa AbstractArray{<:Number} && b isa AbstractArray{<:Number})
            j = findfirst(k -> !approx_equal(a[k], b[k]), eachindex(a, b))
            return j === nothing ? "entries match" : "entry $j: " * _detail(a[j], b[j])
        end
        eltype(a) == eltype(b) || return "eltype $(eltype(a)) != expected $(eltype(b))"
        bad = findall(i -> !entry_matches(a[i], b[i]), eachindex(a, b))
        isempty(bad) && return "entries match"
        i = bad[argmax([abs(a[j] - b[j]) for j in bad])]
        return "$(length(bad)) of $(length(a)) entries differ; worst at $(Tuple(CartesianIndices(a)[i])): " *
               "got $(a[i]), expected $(b[i])"
    end
    return "got $(repr(a)), expected $(repr(b))"
end

function field_matches(case_name::AbstractString, field::Symbol,
                       @nospecialize(actual::NamedTuple), @nospecialize(expected))
    if !haskey(actual, field)
        @error "RPA output changed: field missing" case = case_name field = field
        return false
    end
    value = getfield(actual, field)
    approx_equal(value, expected) && return true
    @error "RPA output changed" case = case_name field = field detail = _detail(value, expected)
    return false
end

function message_prefix(err)
    hasfield(typeof(err), :msg) || return nothing
    msg = getfield(err, :msg)
    msg isa AbstractString || return nothing
    return String(first(first(split(msg, '\n')), MESSAGE_PREFIX_CHARS))
end

function check_case(@nospecialize(case))
    actual, err = try
        (run_case(case.fn, case.setup, case.seed, case.args, case.kwargs), nothing)
    catch e
        (nothing, e)
    end
    if case.throws !== nothing
        if err === nothing
            @error "RPA output changed: case no longer throws" case = case.name expected = case.throws
            @test false
        elseif !(err isa case.throws)
            @error "RPA output changed: different exception" case = case.name expected = case.throws got = typeof(err)
            @test false
        elseif message_prefix(err) != case.expected.message_prefix
            @error "RPA output changed: different error message" case = case.name expected = case.expected.message_prefix got = message_prefix(err)
            @test false
        else
            @test true
        end
        return
    end
    if err !== nothing
        @error "RPA output changed: case now throws" case = case.name exception = err
        @test err === nothing
        return
    end
    for field in keys(case.expected)
        @test field_matches(case.name, field, actual, case.expected[field])
    end
end

case_counts(cases) = Dict{Symbol,Int}(f => count(c -> c.fn === f, cases)
                                      for f in unique(c.fn for c in cases))

function run_tests(cases)
    @testset "RPA outputs are pinned" begin
        @test allunique(case.name for case in cases)
        counts = case_counts(cases)
        counts == EXPECTED_CASE_COUNTS ||
            @error "Golden case counts differ from EXPECTED_CASE_COUNTS" got = counts expected = EXPECTED_CASE_COUNTS
        @test counts == EXPECTED_CASE_COUNTS
        for fn in unique(case.fn for case in cases)
            @testset "$fn" begin
                for case in cases
                    case.fn === fn && check_case(case)
                end
            end
        end
    end
end

end # module RPAGoldenRunner

if !isdefined(@__MODULE__, :RPA_GOLDEN_GENERATOR)
    RPAGoldenRunner.run_tests(include(joinpath(@__DIR__, "data", "rpa_golden.jl")))
end
