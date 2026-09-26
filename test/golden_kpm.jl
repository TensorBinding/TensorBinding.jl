using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test, Random
using TensorBinding: KPM_Tn, KPM_Tn_mps, get_ldos, get_ldos_spectrum, get_ldos_online,
                     get_ldos_spatial, get_dos_stochastic, get_dos_trace

# Characterization ("golden") tests for the KPM solver, src/solvers/KPM_tk.jl.
#
# These tests pin what the KPM functions compute *today*, suspected bugs
# included, so that the Tier 1 split of KPM_tk.jl (docs/dev/REORGANISATION_TODO.md)
# cannot silently change an output. The expected values live in
# test/data/kpm_golden.jl, written by test/data/generate_kpm_golden.jl (see its
# header for how to rerun it).
#
# A failure here means an output changed. If that is a regression, fix the code.
# If it is intentional, regenerate the data file in the same commit as the
# behaviour change and review the diff case by case.
#
# Every case below builds its inputs from a deep copy of one of a few tiny
# models (built once, each right after its own `Random.seed!`), reseeds the
# global RNG with CASE_SEED right before it runs (QTCI and DMRG draw from it),
# and returns a NamedTuple of plain values: numbers, arrays, `nothing`, symbols.
# MPS and MPO outputs are stored densely (see `dense`): an MPS as a vector over
# its own sites, an MPO as the matrix <row|M|col>, site 1 the most significant
# digit in both.
#
# Comparison rules (`mismatch`):
#   * floating-point and complex values compare with
#     isapprox(rtol=RTOL, atol=ATOL); NaN matches NaN;
#   * integers, symbols, strings, `nothing` and Bool compare with `==`;
#   * the type of every value and the element type and shape of every array
#     must match exactly, and a NamedTuple must have the same field names in the
#     same order;
#   * a case recorded as throwing must still throw an exception of that type
#     whose message starts with the recorded prefix (MESSAGE_PREFIX_CHARS
#     characters of its first line);
#   * the data file must hold one entry for every case in CASES, except that
#     the entry of a skipped case (below) may be missing, and no other entry.
#
# Cases whose `requires` returns false are skipped (and counted): they pin code
# that REORGANISATION_TODO.md lists for deletion ("Delete dead and legacy code":
# `_get_exciton_ldos_cached` + the exciton `KPM_Tn(H, N, X)` method,
# `ldos_exc_KPM_Tn`, `get_mus_raw`, `compute_dos_ldos_hodc`, all deleted in
# c91ff65), so that deleting it does not need a data regeneration. Their data
# can be dropped at leisure: the generator leaves a skipped case out, and the
# test accepts the data with or without its entry.
#
# Runtime: about 2 minutes in a fresh process, nearly all of it first-call
# compilation (real and complex MPO/MPS Chebyshev loops, QTCI in the exciton
# builder, the Fibonacci builder, DMRG); the cases themselves run in ~16 s.
#
# The runner below is shared with the generator, which includes this file with
# `KPM_GOLDEN_GENERATOR` defined so that only the module is loaded.

module KPMGoldenRunner

using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test, Random
using TensorBinding: KPM_Tn, KPM_Tn_mps, get_ldos, get_ldos_spectrum, get_ldos_online,
                     get_ldos_spatial, get_dos_stochastic, get_dos_trace,
                     get_ldos_from_mun, get_ldos_hodc_from_mun, compute_hodc_params,
                     get_hodc_weights, get_hodc_gf_weights, get_density_from_Tn,
                     get_Green_retarded_from_Tn, get_Green_retarded_from_Tn_hodc,
                     get_ldos_w_from_Tn, get_ldos_w_from_Tn_hodc, get_ldos_diag_from_Tn,
                     get_exciton_ldos_spatial, get_exciton_ldos,
                     get_exciton_ldos_separation, exciton_radius2,
                     get_Hamiltonian, exciton_hamiltonian, physical_site_state,
                     mpsexciton, add_zeeman!
const TB = TensorBinding

const RTOL = 1e-10
const ATOL = 1e-12
const MODEL_SEED = 4242
const CASE_SEED  = 20260925
const MESSAGE_PREFIX_CHARS = 60

# Number of cases in CASES, skipped ones included (test/data/kpm_golden.jl holds
# this many entries, or fewer by the skipped cases it was generated without).
# Update it by hand, in the same commit, when cases are added or removed on purpose.
const EXPECTED_CASE_COUNT = 261

# ── Dense views of tensor-network outputs ────────────────────────────────────
# Built by a left-to-right sweep over plain arrays of each tensor, so that only
# rank <= 4 ITensor -> Array conversions are compiled (contracting a whole MPO
# into one ITensor compiles a new contraction for every tensor rank).
_linkind_or_nothing(x, k) = (1 <= k < length(x)) ? linkind(x, k) : nothing

function _site_array(T::ITensor, left, mids, right)
    idx = Index[i for i in (left, mids..., right) if i !== nothing]
    A = Array(T, idx...)
    dl = left === nothing ? 1 : dim(left)
    dr = right === nothing ? 1 : dim(right)
    return reshape(A, dl, dim.(mids)..., dr)
end

"""
    dense(x)

Plain-array view of a KPM output: an `MPS` becomes the vector of its amplitudes
over its own site indices, an `MPO` the matrix `<row|M|col>` (row = primed
indices); site 1 is the most significant digit in both. Vectors of MPS/MPO (or
`nothing`) are mapped entrywise; other values pass through unchanged.
"""
function dense(psi::MPS)
    s = siteinds(psi)
    acc = ones(Float64, 1, 1)                     # acc[state prefix, right link]
    for k in eachindex(psi)
        A = _site_array(psi[k], _linkind_or_nothing(psi, k - 1), (s[k],),
                        _linkind_or_nothing(psi, k))
        dl, ds, dr = size(A)
        new = zeros(promote_type(eltype(acc), eltype(A)), ds, size(acc, 1), dr)
        for r in 1:dr, n in axes(acc, 1), sk in 1:ds, l in 1:dl
            new[sk, n, r] += acc[n, l] * A[l, sk, r]
        end
        acc = reshape(new, ds * size(acc, 1), dr)  # new state index = old * ds + sk
    end
    return vec(acc)
end

function dense(M::MPO)
    s = [only(filter(i -> plev(i) == 0, siteinds(M, j))) for j in eachindex(M)]
    acc = ones(Float64, 1, 1, 1)                  # acc[row prefix, col prefix, link]
    for k in eachindex(M)
        A = _site_array(M[k], _linkind_or_nothing(M, k - 1), (prime(s[k]), s[k]),
                        _linkind_or_nothing(M, k))
        dl, ds, _, dr = size(A)
        nr, nc = size(acc, 1), size(acc, 2)
        new = zeros(promote_type(eltype(acc), eltype(A)), ds, nr, ds, nc, dr)
        for r in 1:dr, c in 1:nc, sq in 1:ds, n in 1:nr, sp in 1:ds, l in 1:dl
            new[sp, n, sq, c, r] += acc[n, c, l] * A[l, sp, sq, r]
        end
        acc = reshape(new, ds * nr, ds * nc, dr)
    end
    return acc[:, :, 1]
end

dense(v::AbstractVector{<:Union{Nothing,MPS,MPO}}) = [dense(x) for x in v]
dense(x) = x

idx_info(::Nothing) = nothing
idx_info(i::Index)  = (dim = dim(i), tags = string(tags(i)), plev = plev(i))

tr_all(Tn) = [tr(T) for T in Tn]
moments(psi, list) = [inner(psi, phi) for phi in list]

# ── Models (built once; every case works on a deep copy) ─────────────────────
const MODELS = Dict{Symbol,Any}()

# The fixtures avoid QTCI-built presets on purpose: compiling QTCI and the 2D
# preset builders cost more than all KPM cases together. Instead a step
# potential is added from product MPOs (it breaks the chain's reflection
# symmetry, so a site-order bug cannot hide), and the two 2D fixtures reuse a
# 1D register with a 2D geometry attached: get_ldos_spatial only reads the
# dimension of H.geometry(1) and H.Lx, so they run exactly the 2D sampling code.
function _step_potential(H, weights)
    terms = [w * MPO(H.sites, [j == k ? "Proj1" : "Id" for j in eachindex(H.sites)])
             for (k, w) in weights]
    H.mpo = +(H.mpo, terms...; cutoff=1e-12)
    return H
end

function build_models!()
    isempty(MODELS) || return MODELS
    redirect_stdout(devnull) do
        # 1D chain, L=4: potential +0.3 on the right half (MSB) and -0.2 on sites
        # with bit 1 of x-1 set; off-centre KPM window.
        Random.seed!(MODEL_SEED)
        H = get_Hamiltonian("chain_1d", 1.0; L=4)
        _step_potential(H, (1 => 0.3, 3 => -0.2))
        H.scale, H.center = 2.9, 0.1
        MODELS[:chain4] = H

        # 1D chain, L=3, t=0.9 (Tn-list reconstructions: 8x8 dense MPOs).
        Random.seed!(MODEL_SEED + 1)
        H = get_Hamiltonian("chain_1d", 0.9; L=3)
        _step_potential(H, (2 => 0.25,))
        H.scale, H.center = 2.8, -0.05
        MODELS[:chain3] = H

        # Spinful chain, L=3: spin index prepended by add_zeeman! (Sz field 0.4),
        # so the two spin sectors differ.
        Random.seed!(MODEL_SEED + 2)
        H = get_Hamiltonian("chain_1d", 1.0; L=3)
        add_zeeman!(H, 0.4)
        H.scale, H.center = 3.0, -0.05
        MODELS[:spin3] = H

        # SSH chain with a postpended sublattice index (complex MPO), L=3.
        Random.seed!(MODEL_SEED + 3)
        MODELS[:ssh3] = get_Hamiltonian("ssh_sublattice", (t=1.0, d=0.3); L=3)

        # The same SSH register read as a 2x4 grid of two-atom unit cells
        # (Lx=1, Ly=2): 2D sampling with a sublattice index.
        H = deepcopy(MODELS[:ssh3])
        H.geometry = i -> Float64[((i - 1) ÷ 2) % 2, ((i - 1) ÷ 2) ÷ 2]
        H.Lx = 1
        MODELS[:ssh3_2d] = H

        # The chain4 register read as a 4x4 grid (Lx=2, Ly=2): 2D sampling
        # without an auxiliary index.
        H = deepcopy(MODELS[:chain4])
        H.geometry = i -> Float64[(i - 1) % 4, (i - 1) ÷ 4]
        H.Lx = 2
        MODELS[:snake4] = H

        # Fibonacci projected position space, L=4 (8 physical sites in 16).
        Random.seed!(MODEL_SEED + 6)
        MODELS[:fib4] = get_Hamiltonian("fibonacci", (A=1.0, B=2.0); L=4)

        # Exciton on two L=3 chains (t_c=1, t_v=0.8), contact U=-1.5, 6 sites.
        Random.seed!(MODEL_SEED + 7)
        Hc = get_Hamiltonian("chain_1d", 1.0; L=3)
        Hv = get_Hamiltonian("chain_1d", 0.8; L=3)
        H = exciton_hamiltonian(Hc, Hv, x -> -1.5; scale=5.0)
        H.center = -0.2
        MODELS[:exc3] = H
    end
    return MODELS
end

model(name::Symbol) = deepcopy(MODELS[name])

# Energies per model (physical units); each list has points outside the
# rescaled support |(ω - center)/scale| < 1 on both sides where possible.
const W_CHAIN4 = [-3.2, -2.1, -0.75, 0.0, 0.4, 1.3, 2.5, 3.05]
const W_CHAIN3 = [-3.0, -1.4, 0.0, 0.6, 2.2]
const W_SPIN3  = [-3.1, -1.5, 0.0, 0.7, 2.2, 3.0]
const W_SSH3   = [-2.3, -1.2, -0.3, 0.0, 0.6, 1.5]
const W_FIB4  = [-4.3, -3.0, -1.2, 0.0, 1.5, 3.5]
const W_EXC3   = [-5.3, -3.9, -3.0, -1.5, 0.0, 1.0, 2.5, 4.9]

# Rescaled energies and synthetic moments for the pure reconstruction helpers.
const E_GRID  = [-1.2, -1.0, -0.999, -0.6, 0.0, 0.45, 0.999, 1.0]
const MUN     = [cos(0.4 * n) * exp(-0.05 * n) for n in 0:11]
const MUN_C   = MUN .+ 0.3im .* sin.(0:11)

# ── Case table ───────────────────────────────────────────────────────────────
struct Case
    name::String
    f::Function
    requires::Function
end
const CASES = Case[]
always() = true
case!(f::Function, name::AbstractString; requires::Function = always) =
    push!(CASES, Case(String(name), f, requires))

has_exciton_KPM_Tn() = hasmethod(TB.KPM_Tn, Tuple{TB.TBHamiltonian, Int, Int})
has(sym::Symbol) = () -> isdefined(TB, sym)
# The cached helper calls the exciton KPM_Tn method; both go in the same deletion.
has_cached_exciton() = isdefined(TB, :_get_exciton_ldos_cached) && has_exciton_KPM_Tn()

# ════════════════════════════════════════════════════════════════════════════
# 1. Kernels and weight matrices
# ════════════════════════════════════════════════════════════════════════════
for N in (7, 16), k in (:jackson, :lorentz, :fejer, :dirichlet)
    case!("kernel/$k/N$N") do
        (; w = TB._kpm_kernel(N, k))
    end
end
case!("kernel/lorentz/N12/lambda2.5") do
    (; w = TB._kpm_kernel(12, :lorentz; lambda=2.5))
end
case!("kernel/jackson/N1") do
    (; w = TB._kpm_kernel(1, :jackson))
end
case!("kernel/unknown") do
    TB._kpm_kernel(8, :bogus)
end

for k in (:jackson, :lorentz, :fejer, :dirichlet)
    case!("weights/kpm/$k") do
        (; W = TB._kpm_weight_matrix(9, E_GRID; kernel=k))
    end
end
case!("weights/kpm/lorentz/lambda1.5") do
    (; W = TB._kpm_weight_matrix(9, E_GRID; kernel=:lorentz, lambda=1.5))
end
case!("weights/dos/jackson") do
    W, denom = TB._dos_weight_matrix(9, E_GRID)
    (; W, denom)
end
case!("weights/dos/lorentz/lambda3") do
    W, denom = TB._dos_weight_matrix(9, E_GRID; kernel=:lorentz, lambda=3.0)
    (; W, denom)
end
case!("weights/dos/fejer") do
    W, denom = TB._dos_weight_matrix(9, E_GRID; kernel=:fejer)
    (; W, denom)
end
case!("weights/dos/hodc/eta0/m4") do
    W, denom = TB._dos_weight_matrix(9, E_GRID; kernel=:hodc)
    (; W, denom)
end
case!("weights/dos/hodc/eta0.05/m6") do
    W, denom = TB._dos_weight_matrix(9, E_GRID; kernel=:hodc, eta=0.05, m_order=6)
    (; W, denom)
end

# HODC helpers
for m in (4, 6)
    case!("hodc/params/m$m") do
        zl, wl = compute_hodc_params(m)
        (; zl = collect(zl), wl)
    end
end
case!("hodc/params/default") do
    zl, wl = compute_hodc_params()
    (; zl = collect(zl), wl)
end
case!("hodc/weights/y0.3/N10/eta0.05/m4") do
    zl, wl = compute_hodc_params(4)
    (; nu = get_hodc_weights(0.3, 10, 0.05, zl, wl))
end
case!("hodc/weights/y-0.7/N7/eta0.1/m6") do
    zl, wl = compute_hodc_params(6)
    (; nu = get_hodc_weights(-0.7, 7, 0.1, zl, wl))
end
case!("hodc/gf_weights/y0.3/N10/eta0.05/m4") do
    zl, wl = compute_hodc_params(4)
    (; c = get_hodc_gf_weights(0.3, 10, 0.05, zl, wl))
end
case!("hodc/gf_weights/y-0.9/N7/eta0.02/m6") do
    zl, wl = compute_hodc_params(6)
    (; c = get_hodc_gf_weights(-0.9, 7, 0.02, zl, wl))
end

# Reconstruction from moments
for k in (:jackson, :lorentz, :fejer, :dirichlet, :hodc)
    case!("from_mun/$k") do
        (; A = [get_ldos_from_mun(MUN, 12, E; kernel=k) for E in E_GRID])
    end
end
case!("from_mun/lorentz/lambda2") do
    (; A = [get_ldos_from_mun(MUN, 12, E; kernel=:lorentz, lambda=2.0) for E in E_GRID])
end
case!("from_mun/hodc/eta0.08/m6") do
    (; A = [get_ldos_from_mun(MUN, 12, E; kernel=:hodc, eta=0.08, m_order=6) for E in E_GRID])
end
case!("from_mun/jackson/complex_moments") do
    (; A = [get_ldos_from_mun(MUN_C, 12, E) for E in E_GRID])
end
case!("from_mun/jackson/N8_of_12") do
    (; A = [get_ldos_from_mun(MUN, 8, E) for E in E_GRID])
end
case!("from_mun/unknown_kernel") do
    get_ldos_from_mun(MUN, 12, 0.2; kernel=:bogus)
end
case!("hodc_from_mun/default") do
    (; A = [get_ldos_hodc_from_mun(MUN, 12, E) for E in E_GRID])
end
case!("hodc_from_mun/eta0.1/m6/complex") do
    (; A = [get_ldos_hodc_from_mun(MUN_C, 12, E; eta=0.1, m_order=6) for E in E_GRID])
end

# ════════════════════════════════════════════════════════════════════════════
# 2. Chebyshev lists: KPM_Tn / KPM_Tn_mps
# ════════════════════════════════════════════════════════════════════════════
case!("KPM_Tn/raw_mpo/chain3") do
    H = model(:chain3)
    Tn, sc, c = KPM_Tn(H.mpo, 10, H.sites; scale=H.scale, center=H.center,
                       maxdim=40, cutoff=1e-10)
    (; n = length(Tn), scale = sc, center = c, T = dense(Tn))
end
case!("KPM_Tn/raw_mpo/chain3/maxdim3_quiet") do
    H = model(:chain3)
    Tn, sc, c = KPM_Tn(H.mpo, 8, H.sites; scale=H.scale, center=H.center,
                       maxdim=3, verbose=false)
    (; n = length(Tn), T_last = dense(Tn[end]), traces = tr_all(Tn))
end
case!("KPM_Tn/raw_mpo/chain3/dmrg_bounds") do
    H = model(:chain3)
    Tn, sc, c = KPM_Tn(H.mpo, 4, H.sites; dmrg_nsweeps=2)
    (; n = length(Tn), scale = sc, center = c, traces = tr_all(Tn))
end
case!("KPM_Tn/tb/chain3/ensure_scale_dmrg") do
    H = model(:chain3)
    H.scale, H.center = 0.0, 0.0
    Tn, sc, c = KPM_Tn(H, 4; dmrg_nsweeps=2)
    (; scale = sc, center = c, H_scale = H.scale, H_center = H.center,
       traces = tr_all(Tn))
end
case!("KPM_Tn/mpo/chain4") do
    H = model(:chain4)
    Tn, sc, c = KPM_Tn(H, 12)
    (; n = length(Tn), scale = sc, center = c, traces = tr_all(Tn),
       T1 = dense(Tn[2]), T6 = dense(Tn[7]), T12 = dense(Tn[13]),
       cache_Ncheb = H._tn_Ncheb, cache_len = length(H._tn_cache),
       cache_is_result = H._tn_cache === Tn, mps_cache = H._tn_mps_cache === nothing)
end
case!("KPM_Tn/mpo/chain4/maxdim4_cutoff1e-6") do
    H = model(:chain4)
    Tn, _, _ = KPM_Tn(H, 12; maxdim=4, cutoff=1e-6)
    (; traces = tr_all(Tn), T12 = dense(Tn[13]))
end
case!("KPM_Tn/mps/chain4") do
    H = model(:chain4)
    psi0 = physical_site_state(H, 6)
    Tn, sc, c = KPM_Tn(H, 12; mode=:mps, psi0)
    (; n = length(Tn), scale = sc, center = c, phi = dense(Tn),
       cache_Ncheb = H._tn_Ncheb, mpo_cache = H._tn_cache === nothing,
       cache_is_result = H._tn_mps_cache === Tn)
end
case!("KPM_Tn/mps/chain4/unnormalised_psi0") do
    H = model(:chain4)
    psi0 = +(physical_site_state(H, 3), physical_site_state(H, 9); cutoff=1e-14)
    Tn, _, _ = KPM_Tn(H, 12; mode=:mps, psi0, maxdim=3)
    (; phi = dense(Tn))
end
case!("KPM_Tn/mps/missing_psi0") do
    KPM_Tn(model(:chain4), 4; mode=:mps)
end
case!("KPM_Tn/unknown_mode") do
    KPM_Tn(model(:chain4), 4; mode=:bogus)
end
case!("KPM_Tn/mpo/spin3") do
    H = model(:spin3)
    Tn, _, _ = KPM_Tn(H, 8)
    (; traces = tr_all(Tn), T8 = dense(Tn[9]))
end
case!("KPM_Tn/mpo/ssh3") do
    H = model(:ssh3)
    Tn, _, _ = KPM_Tn(H, 8)
    (; traces = tr_all(Tn), T3 = dense(Tn[4]))
end
case!("KPM_Tn/mpo/fib4") do
    H = model(:fib4)
    Tn, sc, c = KPM_Tn(H, 10)
    (; scale = sc, center = c, traces = tr_all(Tn),
       T0 = dense(Tn[1]), T1 = dense(Tn[2]), T10 = dense(Tn[11]))
end
case!("KPM_Tn/mps/fib4") do
    H = model(:fib4)
    Tn, _, _ = KPM_Tn(H, 10; mode=:mps, psi0=physical_site_state(H, 5))
    (; phi = dense(Tn))
end
case!("KPM_Tn_mps/raw/chain3") do
    H = model(:chain3)
    psi0 = physical_site_state(H, 2)
    Tn, sc, c = KPM_Tn_mps(H.mpo, 10, psi0, H.sites; scale=H.scale, center=H.center)
    (; n = length(Tn), scale = sc, center = c, phi = dense(Tn))
end
case!("KPM_Tn_mps/tb/chain3/maxdim2") do
    H = model(:chain3)
    psi0 = physical_site_state(H, 7)
    Tn, sc, c = KPM_Tn_mps(H, 10, psi0; maxdim=2)
    (; phi = dense(Tn), cache_Ncheb = H._tn_Ncheb, cache_is_result = H._tn_mps_cache === Tn)
end

# ════════════════════════════════════════════════════════════════════════════
# 3. Cached reconstructions: get_ldos / get_ldos_spectrum / *_from_Tn
# ════════════════════════════════════════════════════════════════════════════
function chain4_mpo_cached(Ncheb = 12)
    H = model(:chain4)
    KPM_Tn(H, Ncheb)
    return H
end
function chain4_mps_cached(psi0_fn; Ncheb = 12)
    H = model(:chain4)
    psi0 = psi0_fn(H)
    KPM_Tn(H, Ncheb; mode=:mps, psi0)
    return H, psi0
end
site6(H) = physical_site_state(H, 6)
sup39(H) = +(physical_site_state(H, 3), physical_site_state(H, 9); cutoff=1e-14)

case!("get_ldos/diag/chain4/jackson") do
    H = chain4_mpo_cached()
    (; A = [dense(get_ldos(H, w)) for w in W_CHAIN4])
end
case!("get_ldos/diag/chain4/lorentz_lambda3") do
    H = chain4_mpo_cached()
    (; A = [dense(get_ldos(H, w; kernel=:lorentz, lambda=3.0)) for w in W_CHAIN4])
end
case!("get_ldos/diag/chain4/fejer_maxdim2") do
    H = chain4_mpo_cached()
    (; A = [dense(get_ldos(H, w; kernel=:fejer, maxdim=2, cutoff=1e-6)) for w in W_CHAIN4])
end
case!("get_ldos/mpo/chain4/jackson") do
    H = chain4_mpo_cached()
    (; A = [dense(get_ldos(H, w; mode=:mpo)) for w in (-0.75, 0.4)])
end
case!("get_ldos/mpo/chain4/dirichlet") do
    H = chain4_mpo_cached()
    (; A = dense(get_ldos(H, 1.3; mode=:mpo, kernel=:dirichlet)))
end
case!("get_ldos/mpo/chain4/hodc_default_eta") do
    H = chain4_mpo_cached()
    zl, wl = compute_hodc_params(4)
    (; A = dense(get_ldos(H, 0.4; mode=:mpo, kernel=:hodc, zl, wl)))
end
case!("get_ldos/mpo/chain4/hodc_eta0.1_m6") do
    H = chain4_mpo_cached()
    zl, wl = compute_hodc_params(6)
    (; A = dense(get_ldos(H, -2.1; mode=:mpo, kernel=:hodc, zl, wl, eta=0.1)))
end
case!("get_ldos/mpo/chain4/hodc_missing_zl") do
    get_ldos(chain4_mpo_cached(), 0.4; mode=:mpo, kernel=:hodc)
end
case!("get_ldos/mps/chain4/jackson") do
    H, psi0 = chain4_mps_cached(site6)
    (; A = [get_ldos(H, w; mode=:mps, psi0) for w in W_CHAIN4])
end
case!("get_ldos/mps/chain4/lorentz_lambda2") do
    H, psi0 = chain4_mps_cached(site6)
    (; A = [get_ldos(H, w; mode=:mps, psi0, kernel=:lorentz, lambda=2.0) for w in W_CHAIN4])
end
case!("get_ldos/mps/chain4/hodc_default_eta") do
    H, psi0 = chain4_mps_cached(site6)
    (; A = [get_ldos(H, w; mode=:mps, psi0, kernel=:hodc) for w in W_CHAIN4])
end
case!("get_ldos/mps/chain4/hodc_eta0.05_m6") do
    H, psi0 = chain4_mps_cached(site6)
    (; A = [get_ldos(H, w; mode=:mps, psi0, kernel=:hodc, eta=0.05, m_order=6) for w in W_CHAIN4])
end
# Pinned as it is today: KPM_Tn(mode=:mps) normalises psi0 but get_ldos(mode=:mps)
# takes moments against the caller's (here unnormalised, norm sqrt(2)) psi0, so
# the result scales with norm(psi0) rather than 1 or norm(psi0)^2.
case!("get_ldos/mps/chain4/unnormalised_psi0") do
    H, psi0 = chain4_mps_cached(sup39)
    (; A = [get_ldos(H, w; mode=:mps, psi0) for w in W_CHAIN4])
end
case!("get_ldos/diag/no_mpo_cache") do
    get_ldos(model(:chain4), 0.0)
end
case!("get_ldos/mpo/no_mpo_cache") do
    get_ldos(model(:chain4), 0.0; mode=:mpo)
end
case!("get_ldos/mps/no_mps_cache") do
    get_ldos(chain4_mpo_cached(4), 0.0; mode=:mps, psi0=site6(model(:chain4)))
end
case!("get_ldos/mps/missing_psi0") do
    H, _ = chain4_mps_cached(site6; Ncheb=4)
    get_ldos(H, 0.0; mode=:mps)
end
case!("get_ldos/unknown_mode") do
    get_ldos(chain4_mpo_cached(4), 0.0; mode=:bogus)
end
# Pinned as it is today (it throws): the Fibonacci projector T_0 stores its site
# legs as (s, s') while T_n>=1 store (s', s), so extract_diagonal_to_mps puts the
# T_0 diagonal on primed indices and the first `+` of diagonals fails.
case!("get_ldos/diag/fib4") do
    H = model(:fib4)
    KPM_Tn(H, 10)
    (; A = [dense(get_ldos(H, w)) for w in W_FIB4])
end
case!("get_ldos/mpo/fib4") do
    H = model(:fib4)
    KPM_Tn(H, 10)
    (; A = dense(get_ldos(H, -1.2; mode=:mpo)))
end
case!("get_ldos/mps/fib4") do
    H = model(:fib4)
    psi0 = physical_site_state(H, 5)
    KPM_Tn(H, 10; mode=:mps, psi0)
    (; A = [get_ldos(H, w; mode=:mps, psi0) for w in W_FIB4])
end

case!("get_ldos_spectrum/chain4/jackson") do
    H = chain4_mpo_cached()
    (; A = dense(get_ldos_spectrum(H, W_CHAIN4)))
end
case!("get_ldos_spectrum/chain4/lorentz_lambda2") do
    H = chain4_mpo_cached()
    (; A = dense(get_ldos_spectrum(H, W_CHAIN4; kernel=:lorentz, lambda=2.0)))
end
case!("get_ldos_spectrum/chain4/dirichlet_maxdim2") do
    H = chain4_mpo_cached()
    (; A = dense(get_ldos_spectrum(H, W_CHAIN4; kernel=:dirichlet, maxdim=2, cutoff=1e-6)))
end
case!("get_ldos_spectrum/chain4/range_input") do
    H = chain4_mpo_cached(8)
    (; A = dense(get_ldos_spectrum(H, range(-2.0, 2.0; length=3))))
end
case!("get_ldos_spectrum/spin3") do
    H = model(:spin3)
    KPM_Tn(H, 8)
    (; A = dense(get_ldos_spectrum(H, W_SPIN3)))
end
case!("get_ldos_spectrum/ssh3") do
    H = model(:ssh3)
    KPM_Tn(H, 8)
    (; A = dense(get_ldos_spectrum(H, W_SSH3)))
end
# Throws today, for the reason given at "get_ldos/diag/fib4".
case!("get_ldos_spectrum/fib4") do
    H = model(:fib4)
    KPM_Tn(H, 10)
    (; A = dense(get_ldos_spectrum(H, W_FIB4)))
end
case!("get_ldos_spectrum/no_cache") do
    get_ldos_spectrum(model(:chain4), W_CHAIN4)
end

# Direct reconstructions from a chain3 Chebyshev list (rescaled energies).
function chain3_Tn(Ncheb = 10)
    H = model(:chain3)
    Tn, _, _ = KPM_Tn(H, Ncheb)
    return Tn
end

case!("get_ldos_diag_from_Tn/chain3/N8") do
    Tn = chain3_Tn()
    (; A = dense(get_ldos_diag_from_Tn(Tn, 8, [-1.1, -0.5, 0.2, 0.9, 1.0])))
end
case!("get_ldos_diag_from_Tn/chain3/lorentz_maxdim2") do
    Tn = chain3_Tn()
    (; A = dense(get_ldos_diag_from_Tn(Tn, 10, [-0.5, 0.2]; kernel=:lorentz, lambda=2.0,
                                       maxdim=2, cutoff=1e-6)))
end
case!("get_ldos_w_from_Tn/chain3/jackson") do
    (; A = dense(get_ldos_w_from_Tn(chain3_Tn(), 10, 0.2)))
end
case!("get_ldos_w_from_Tn/chain3/lorentz_maxdim3") do
    (; A = dense(get_ldos_w_from_Tn(chain3_Tn(), 10, -0.6; kernel=:lorentz, lambda=2.0,
                                    maxdim=3, cutoff=1e-6)))
end
case!("get_ldos_w_from_Tn/chain3/fejer_N7") do
    (; A = dense(get_ldos_w_from_Tn(chain3_Tn(), 7, 0.5; kernel=:fejer)))
end
case!("get_ldos_w_from_Tn/chain3/hodc") do
    zl, wl = compute_hodc_params(4)
    (; A = dense(get_ldos_w_from_Tn(chain3_Tn(), 10, 0.2; kernel=:hodc, zl, wl, eta=0.05)))
end
case!("get_ldos_w_from_Tn/hodc_missing_zl") do
    get_ldos_w_from_Tn(chain3_Tn(4), 4, 0.2; kernel=:hodc)
end
case!("get_ldos_w_from_Tn_hodc/chain3") do
    zl, wl = compute_hodc_params(6)
    (; A = dense(get_ldos_w_from_Tn_hodc(chain3_Tn(), 10, -0.3, zl, wl)))
end
case!("density/chain3/jackson/fermi0") do
    (; rho = dense(get_density_from_Tn(chain3_Tn(), 10)))
end
case!("density/chain3/jackson/fermi0.3") do
    (; rho = dense(get_density_from_Tn(chain3_Tn(), 10; fermi=0.3)))
end
case!("density/chain3/lorentz/fermi-0.4") do
    (; rho = dense(get_density_from_Tn(chain3_Tn(), 10; fermi=-0.4, kernel=:lorentz, lambda=3.0)))
end
case!("density/chain3/maxdim2_N6") do
    (; rho = dense(get_density_from_Tn(chain3_Tn(), 6; fermi=0.1, maxdim=2, cutoff=1e-6)))
end
case!("density/chain4/fermi0") do
    H = model(:chain4)
    Tn, _, _ = KPM_Tn(H, 12)
    (; rho = dense(get_density_from_Tn(Tn, 12)))
end
case!("green/chain3/jackson/w0.2_eta0.01") do
    (; G = dense(get_Green_retarded_from_Tn(chain3_Tn(), 10, 0.2)))
end
case!("green/chain3/fejer/w-0.5_eta0.05") do
    (; G = dense(get_Green_retarded_from_Tn(chain3_Tn(), 10, -0.5; η=0.05, kernel=:fejer)))
end
case!("green/chain3/lorentz/w0.7_maxdim3") do
    (; G = dense(get_Green_retarded_from_Tn(chain3_Tn(), 10, 0.7; kernel=:lorentz, lambda=2.0,
                                            maxdim=3, cutoff=1e-6)))
end
case!("green/chain3/hodc/w0.2") do
    zl, wl = compute_hodc_params(4)
    (; G = dense(get_Green_retarded_from_Tn(chain3_Tn(), 10, 0.2; η=0.05, kernel=:hodc, zl, wl)))
end
case!("green/hodc_missing_zl") do
    get_Green_retarded_from_Tn(chain3_Tn(4), 4, 0.2; kernel=:hodc)
end
case!("green_hodc/chain3/direct") do
    zl, wl = compute_hodc_params(6)
    (; G = dense(get_Green_retarded_from_Tn_hodc(chain3_Tn(), 10, -0.4, zl, wl; eta=0.08)))
end

# ════════════════════════════════════════════════════════════════════════════
# 4. Online LDOS at one site: get_ldos_online
# ════════════════════════════════════════════════════════════════════════════
case!("online/chain4/X6/jackson") do
    (; A = get_ldos_online(model(:chain4), 12, 6, W_CHAIN4))
end
case!("online/chain4/X1/lorentz_lambda2") do
    (; A = get_ldos_online(model(:chain4), 12, 1, W_CHAIN4; kernel=:lorentz, lambda=2.0))
end
case!("online/chain4/X16/fejer_maxdim2") do
    (; A = get_ldos_online(model(:chain4), 12, 16, W_CHAIN4; kernel=:fejer, maxdim=2, cutoff=1e-6))
end
case!("online/chain4/range_input") do
    (; A = get_ldos_online(model(:chain4), 8, 9, range(-2.0, 2.0; length=4)))
end
case!("online/spin3/auto_spin_sum") do
    (; A = get_ldos_online(model(:spin3), 10, 3, W_SPIN3))
end
for s in (1, 2)
    case!("online/spin3/proj_s$s") do
        (; A = get_ldos_online(model(:spin3), 10, 3, W_SPIN3; proj_s=s))
    end
end
case!("online/ssh3/auto_sublat_sum") do
    (; A = get_ldos_online(model(:ssh3), 10, 4, W_SSH3))
end
case!("online/ssh3/proj_sl2") do
    (; A = get_ldos_online(model(:ssh3), 10, 4, W_SSH3; proj_sl=2))
end
case!("online/ssh3/X8_proj_sl1") do
    (; A = get_ldos_online(model(:ssh3), 10, 8, W_SSH3; proj_sl=1))
end
case!("online/fib4/X5") do
    (; A = get_ldos_online(model(:fib4), 10, 5, W_FIB4))
end
case!("online/exc3/X4") do
    (; A = get_ldos_online(model(:exc3), 10, 4, W_EXC3))
end

# ════════════════════════════════════════════════════════════════════════════
# 5. Spatial LDOS: get_ldos_spatial
# ════════════════════════════════════════════════════════════════════════════
spatial(name, Ncheb, ws; kw...) = (; A = get_ldos_spatial(model(name), Ncheb, ws; kw...))

# chain4: 1D, no aux index
for mode in (:mpo, :mps)
    case!("spatial/chain4/$mode/full") do
        spatial(:chain4, 12, W_CHAIN4; mode)
    end
    case!("spatial/chain4/$mode/num_x4_num_avg2") do
        spatial(:chain4, 12, W_CHAIN4; mode, num_x=4, num_avg=2)
    end
    case!("spatial/chain4/$mode/x_groups_nested") do
        spatial(:chain4, 12, W_CHAIN4; mode, x_groups=[[1, 2], [5], [9, 10, 11]])
    end
end
case!("spatial/chain4/mpo/window_x3-10_num_x4") do
    spatial(:chain4, 12, W_CHAIN4; num_x=4, x_start=3, x_end=10)
end
case!("spatial/chain4/mpo/x_groups_flat") do
    spatial(:chain4, 12, W_CHAIN4; x_groups=[3, 7, 16])
end
case!("spatial/chain4/mpo/lorentz_maxdim4") do
    spatial(:chain4, 12, W_CHAIN4; kernel=:lorentz, lambda=2.0, maxdim=4, cutoff=1e-6, num_x=8)
end
case!("spatial/chain4/mps/fejer_maxdim2") do
    spatial(:chain4, 12, W_CHAIN4; mode=:mps, kernel=:fejer, maxdim=2, cutoff=1e-6, num_x=8)
end
case!("spatial/chain4/grid_on_1d") do
    spatial(:chain4, 8, W_CHAIN4; grid=true)
end
case!("spatial/chain4/unknown_mode") do
    spatial(:chain4, 8, W_CHAIN4; mode=:bogus)
end
case!("spatial/chain4/bad_ordering") do
    spatial(:chain4, 8, W_CHAIN4; ordering=:bogus)
end
case!("spatial/chain4/conumber_on_binary") do
    spatial(:chain4, 8, W_CHAIN4; ordering=:conumber)
end
case!("spatial/chain4/num_x_exceeds_window") do
    spatial(:chain4, 8, W_CHAIN4; num_x=6, x_start=1, x_end=4)
end

# spin3: spin index prepended, auto-enabled spin projection
for mode in (:mpo, :mps)
    case!("spatial/spin3/$mode/auto_spin_sum") do
        spatial(:spin3, 10, W_SPIN3; mode)
    end
end
case!("spatial/spin3/mpo/proj_s2") do
    spatial(:spin3, 10, W_SPIN3; proj_s=2)
end
case!("spatial/spin3/mps/proj_s1_num_x4") do
    spatial(:spin3, 10, W_SPIN3; mode=:mps, proj_s=1, num_x=4)
end

# ssh3: 1D with sublattice index (postpended)
for mode in (:mpo, :mps)
    case!("spatial/ssh3/$mode/auto_resolve") do
        spatial(:ssh3, 10, W_SSH3; mode)
    end
    case!("spatial/ssh3/$mode/num_x4_auto_average") do
        spatial(:ssh3, 10, W_SSH3; mode, num_x=4)
    end
end
case!("spatial/ssh3/mpo/num_x4_force_resolve") do
    spatial(:ssh3, 10, W_SSH3; num_x=4, sublattice=:resolve)
end
case!("spatial/ssh3/mpo/full_force_average") do
    spatial(:ssh3, 10, W_SSH3; sublattice=:average)
end
case!("spatial/ssh3/mpo/proj_sl2") do
    spatial(:ssh3, 10, W_SSH3; proj_sl=2)
end
case!("spatial/ssh3/mps/proj_sl1_num_x4") do
    spatial(:ssh3, 10, W_SSH3; mode=:mps, proj_sl=1, num_x=4)
end
case!("spatial/ssh3/mpo/x_groups") do
    spatial(:ssh3, 10, W_SSH3; x_groups=[[2, 3], [7]])
end

# ssh3_2d: 2x4 grid of two-atom unit cells (Lx=1, Ly=2), sublattice index
case!("spatial/ssh3_2d/mpo/full_auto_resolve") do
    spatial(:ssh3_2d, 10, W_SSH3)
end
for mode in (:mpo, :mps)
    case!("spatial/ssh3_2d/$mode/grid2x2_average") do
        spatial(:ssh3_2d, 10, W_SSH3; mode, grid=true, num_x=2, num_y=2)
    end
    case!("spatial/ssh3_2d/$mode/window_zoom_resolve") do
        spatial(:ssh3_2d, 10, W_SSH3; mode, grid=true, xwin=(0, 1), ywin=(1, 2))
    end
end
case!("spatial/ssh3_2d/mpo/grid_box_half1") do
    spatial(:ssh3_2d, 10, W_SSH3; grid=true, num_x=2, num_y=4, box_half=1)
end
case!("spatial/ssh3_2d/mpo/grid2x2_num_avg2") do
    spatial(:ssh3_2d, 10, W_SSH3; grid=true, num_x=2, num_y=2, num_avg=2)
end
case!("spatial/ssh3_2d/mpo/block2x2") do
    spatial(:ssh3_2d, 10, W_SSH3; reduce=:block, num_x=2, num_y=2)
end
case!("spatial/ssh3_2d/mpo/block2x2_resolve") do
    spatial(:ssh3_2d, 10, W_SSH3; reduce=:block, num_x=2, num_y=2, sublattice=:resolve)
end
case!("spatial/ssh3_2d/mpo/grid2x2_proj_sl2") do
    spatial(:ssh3_2d, 10, W_SSH3; grid=true, num_x=2, num_y=2, proj_sl=2)
end
case!("spatial/ssh3_2d/block_in_mps_mode") do
    spatial(:ssh3_2d, 10, W_SSH3; mode=:mps, reduce=:block, num_x=2, num_y=2)
end

# snake4: 4x4 grid (Lx=2, Ly=2), no auxiliary index
case!("spatial/snake4/mpo/grid2x2") do
    spatial(:snake4, 10, W_CHAIN4; grid=true, num_x=2, num_y=2)
end
for mode in (:mpo, :mps)
    case!("spatial/snake4/$mode/grid_box_half1") do
        spatial(:snake4, 10, W_CHAIN4; mode, grid=true, num_x=2, num_y=2, box_half=1)
    end
end
case!("spatial/snake4/mpo/linear_box_half1") do
    spatial(:snake4, 10, W_CHAIN4; num_x=4, box_half=1)
end
case!("spatial/snake4/mpo/block2x4") do
    spatial(:snake4, 10, W_CHAIN4; reduce=:block, num_x=2, num_y=4)
end
case!("spatial/snake4/mpo/block1x1") do
    spatial(:snake4, 10, W_CHAIN4; reduce=:block, num_x=1, num_y=1)
end
case!("spatial/snake4/block_not_power_of_two") do
    spatial(:snake4, 10, W_CHAIN4; reduce=:block, num_x=3, num_y=2)
end

# fib4: projected position space. In mode=:mpo the T_0 diagonal sits on primed
# indices (see "get_ldos/diag/fib4"); `inner` still matches them up, with
# ITensors' once-per-session deprecation warning.
for mode in (:mpo, :mps)
    case!("spatial/fib4/$mode/physical") do
        spatial(:fib4, 10, W_FIB4; mode)
    end
end
case!("spatial/fib4/mps/conumber") do
    spatial(:fib4, 10, W_FIB4; mode=:mps, ordering=:conumber)
end
case!("spatial/fib4/mpo/conumber_raw_reversed_uncentered") do
    spatial(:fib4, 10, W_FIB4; ordering=:conumber, conumber_alignment=:raw,
            conumber_orientation=:reversed, conumber_centered=false, conumber_origin=2)
end
case!("spatial/fib4/conumber_coarse") do
    spatial(:fib4, 10, W_FIB4; ordering=:conumber, num_x=4)
end

# exc3: exciton register (mpsexciton probes)
for mode in (:mpo, :mps)
    case!("spatial/exc3/$mode/num_x4") do
        spatial(:exc3, 10, W_EXC3; mode, num_x=4)
    end
end

# ════════════════════════════════════════════════════════════════════════════
# 6. DOS: get_dos_stochastic / get_dos_trace
# ════════════════════════════════════════════════════════════════════════════
stoch(name, Ncheb, ws; kw...) = (; D = get_dos_stochastic(model(name), Ncheb, ws; kw...))

case!("dos_stochastic/chain4/default_seed") do
    stoch(:chain4, 12, W_CHAIN4; N_sample=8)
end
case!("dos_stochastic/chain4/normalize") do
    stoch(:chain4, 12, W_CHAIN4; N_sample=8, normalize=true)
end
case!("dos_stochastic/chain4/seed7_lorentz_lambda2") do
    stoch(:chain4, 12, W_CHAIN4; N_sample=8, seed=7, kernel=:lorentz, lambda=2.0)
end
case!("dos_stochastic/chain4/seed_nothing_global_rng") do
    stoch(:chain4, 12, W_CHAIN4; N_sample=8, seed=nothing)
end
case!("dos_stochastic/chain4/hodc_default_eta") do
    stoch(:chain4, 12, W_CHAIN4; N_sample=8, kernel=:hodc)
end
case!("dos_stochastic/chain4/hodc_eta0.05_m6") do
    stoch(:chain4, 12, W_CHAIN4; N_sample=8, kernel=:hodc, eta=0.05, m_order=6)
end
case!("dos_stochastic/chain4/sample_weighting") do
    stoch(:chain4, 12, W_CHAIN4; N_sample=8, dos_weighting=:sample, normalize=true)
end
case!("dos_stochastic/chain4/fejer_maxdim2") do
    stoch(:chain4, 12, W_CHAIN4; N_sample=8, kernel=:fejer, maxdim=2, cutoff=1e-6)
end
case!("dos_stochastic/chain4/bad_weighting") do
    stoch(:chain4, 12, W_CHAIN4; N_sample=2, dos_weighting=:bogus)
end
case!("dos_stochastic/spin3/full_space") do
    stoch(:spin3, 10, W_SPIN3; N_sample=8)
end
case!("dos_stochastic/spin3/spin_proj_all") do
    stoch(:spin3, 10, W_SPIN3; N_sample=6, spin_proj=true)
end
case!("dos_stochastic/spin3/proj_s1_normalize") do
    stoch(:spin3, 10, W_SPIN3; N_sample=6, spin_proj=true, proj_s=1, normalize=true)
end
case!("dos_stochastic/spin3/spin_proj_sample_weighting") do
    stoch(:spin3, 10, W_SPIN3; N_sample=6, spin_proj=true, dos_weighting=:sample)
end
case!("dos_stochastic/ssh3/full_space_normalize") do
    stoch(:ssh3, 10, W_SSH3; N_sample=8, normalize=true)
end
case!("dos_stochastic/ssh3/sublat_proj_sl1") do
    stoch(:ssh3, 10, W_SSH3; N_sample=6, sublat_proj=true, proj_sl=1)
end
case!("dos_stochastic/ssh3/sublat_proj_all_hodc") do
    stoch(:ssh3, 10, W_SSH3; N_sample=6, sublat_proj=true, kernel=:hodc)
end
case!("dos_stochastic/fib4/projected") do
    stoch(:fib4, 10, W_FIB4; N_sample=8)
end
case!("dos_stochastic/fib4/projected_normalize") do
    stoch(:fib4, 10, W_FIB4; N_sample=8, normalize=true, seed=3)
end
case!("dos_stochastic/exc3/uniform") do
    stoch(:exc3, 10, W_EXC3; N_sample=6)
end
case!("dos_stochastic/exc3/N_bound4") do
    stoch(:exc3, 10, W_EXC3; N_sample=6, N_bound=4)
end
case!("dos_stochastic/exc3/N_bound4_normalize") do
    stoch(:exc3, 10, W_EXC3; N_sample=6, N_bound=4, normalize=true)
end
case!("dos_stochastic/exc3/N_bound4_sample_weighting") do
    stoch(:exc3, 10, W_EXC3; N_sample=6, N_bound=4, dos_weighting=:sample)
end
case!("dos_stochastic/chain4/N_bound_ignored_non_exciton") do
    stoch(:chain4, 12, W_CHAIN4; N_sample=8, N_bound=3)
end

trace_dos(name, Ncheb, ws; kw...) = (; D = get_dos_trace(model(name), Ncheb, ws; kw...))

case!("dos_trace/chain4/jackson") do
    trace_dos(:chain4, 12, W_CHAIN4)
end
case!("dos_trace/chain4/normalize") do
    trace_dos(:chain4, 12, W_CHAIN4; normalize=true)
end
case!("dos_trace/chain4/hodc_default_eta") do
    trace_dos(:chain4, 12, W_CHAIN4; kernel=:hodc)
end
case!("dos_trace/chain4/hodc_eta0.05_m6") do
    trace_dos(:chain4, 12, W_CHAIN4; kernel=:hodc, eta=0.05, m_order=6)
end
case!("dos_trace/chain4/lorentz_maxdim3") do
    trace_dos(:chain4, 12, W_CHAIN4; kernel=:lorentz, lambda=2.0, maxdim=3, cutoff=1e-6)
end
case!("dos_trace/chain4/dirichlet_Ncheb2") do
    trace_dos(:chain4, 2, W_CHAIN4; kernel=:dirichlet)
end
case!("dos_trace/spin3") do
    trace_dos(:spin3, 10, W_SPIN3)
end
case!("dos_trace/ssh3") do
    trace_dos(:ssh3, 10, W_SSH3)
end
case!("dos_trace/ssh3/normalize") do
    trace_dos(:ssh3, 10, W_SSH3; normalize=true)
end
case!("dos_trace/fib4") do
    trace_dos(:fib4, 10, W_FIB4)
end
case!("dos_trace/fib4/normalize") do
    trace_dos(:fib4, 10, W_FIB4; normalize=true)
end
case!("dos_trace/exc3") do
    trace_dos(:exc3, 10, W_EXC3)
end
case!("dos_trace/Ncheb1") do
    trace_dos(:chain4, 1, W_CHAIN4)
end

# ════════════════════════════════════════════════════════════════════════════
# 7. Shared internal helpers of KPM_tk.jl
# ════════════════════════════════════════════════════════════════════════════
function aux_record(H, args...)
    a = TB._aux_setup(H, args...)
    (; nambu_s = idx_info(a.nambu_s_det), nambu_side = a.nambu_side_det,
       spin_s = idx_info(a.spin_s_det), layer_s = idx_info(a.layer_s_det),
       layer_side = a.layer_side_det, sublat_s = idx_info(a.sublat_s_det),
       sublat_side = a.sublat_side_det, nambu_range = a.nambu_range,
       spin_range = a.spin_range, layer_range = a.layer_range, sl_range = a.sl_range,
       any_aux_proj = a.any_aux_proj)
end
case!("aux_setup/chain4/none") do
    aux_record(model(:chain4), false, nothing, false, nothing, false, nothing, false, nothing)
end
case!("aux_setup/chain4/flags_without_indices") do
    aux_record(model(:chain4), true, nothing, true, 2, true, nothing, true, 1)
end
case!("aux_setup/spin3/spin_all") do
    aux_record(model(:spin3), false, nothing, true, nothing, false, nothing, false, nothing)
end
case!("aux_setup/spin3/spin_s2") do
    aux_record(model(:spin3), false, nothing, true, 2, false, nothing, false, nothing)
end
case!("aux_setup/spin3/spin_not_projected") do
    aux_record(model(:spin3), false, nothing, false, 2, false, nothing, false, nothing)
end
case!("aux_setup/ssh3/sublat_all") do
    aux_record(model(:ssh3), false, nothing, false, nothing, false, nothing, true, nothing)
end
case!("aux_setup/ssh3/sublat_sl2") do
    aux_record(model(:ssh3), false, nothing, false, nothing, false, nothing, true, 2)
end
case!("make_psi0/spin3") do
    H = model(:spin3)
    (; psi = [dense(TB._ldos_make_psi0(H, x, 1, s, 1, 1)) for x in (1, 4, 8) for s in (1, 2)])
end
case!("make_psi0/ssh3") do
    H = model(:ssh3)
    (; psi = [dense(TB._ldos_make_psi0(H, x, 1, 1, 1, s)) for x in (2, 7) for s in (1, 2)])
end
case!("make_psi0/ssh3/absent_dofs_ignored") do
    H = model(:ssh3)
    (; psi = [dense(TB._ldos_make_psi0(H, x, 2, 2, 2, s)) for x in (1, 8) for s in (1, 2)])
end
case!("run_kpm_mps/chain4/weight0.5") do
    H = model(:chain4)
    Ham_n = (1 / H.scale) * +(H.mpo, (-H.center) * TB.physical_projector(H); cutoff=1e-8)
    E = (W_CHAIN4 .- H.center) ./ H.scale
    W = TB._kpm_weight_matrix(12, E)
    valid = [abs(e) < 1 for e in E]
    accum = fill(0.25, length(E))
    chi = TB._run_kpm_mps!(Ham_n, physical_site_state(H, 11), 12, W, valid, accum;
                           weight=0.5, maxdim=3, cutoff=1e-6)
    (; chi, accum)
end
case!("ensure_scale/noop_when_set") do
    H = model(:chain4)
    TB._ensure_scale!(H)
    (; scale = H.scale, center = H.center)
end

# ════════════════════════════════════════════════════════════════════════════
# 8. Exciton KPM (KPM_tk.jl l.~1640-1990)
# ════════════════════════════════════════════════════════════════════════════
exc_spatial(Ncheb, ws; kw...) = get_exciton_ldos_spatial(model(:exc3), Ncheb, ws; kw...)

case!("exciton/KPM_Tn_X/X3"; requires=has_exciton_KPM_Tn) do
    H = model(:exc3)
    Tn, sc, c = KPM_Tn(H, 10, 3)
    (; n = length(Tn), scale = sc, center = c,
       mu = moments(mpsexciton(3, H.sites), Tn), norms = [norm(t) for t in Tn],
       phi1 = dense(Tn[2]), phi10 = dense(Tn[end]),
       cache_Ncheb = H._tn_Ncheb, cache_is_result = H._tn_mps_cache === Tn)
end
case!("exciton/cached_ldos/miss_then_hit"; requires=has_cached_exciton) do
    H = model(:exc3)
    (; A = [TB._get_exciton_ldos_cached(H, 3, w; Ncheb=10) for w in W_EXC3])
end
# Pinned as it is today: the cache hit checks only Ncheb, not the X the cache was
# built for, so this returns moments <5,5|T_n|3,3> (values of both signs).
case!("exciton/cached_ldos/hit_other_X"; requires=has_cached_exciton) do
    H = model(:exc3)
    KPM_Tn(H, 10, 3)
    (; A = [TB._get_exciton_ldos_cached(H, 5, w; Ncheb=10) for w in W_EXC3])
end
case!("exciton/cached_ldos/hodc_Ncheb_mismatch"; requires=has_cached_exciton) do
    H = model(:exc3)
    KPM_Tn(H, 6, 3)
    (; A = [TB._get_exciton_ldos_cached(H, 3, w; Ncheb=10, kernel=:hodc) for w in W_EXC3])
end

case!("exciton/spatial/default") do
    (; A = exc_spatial(10, W_EXC3))
end
case!("exciton/spatial/X_list") do
    (; A = exc_spatial(10, W_EXC3; X_list=[8, 2, 5]))
end
case!("exciton/spatial/X_groups") do
    (; A = exc_spatial(10, W_EXC3; X_groups=[[1, 2], [4], [6, 7, 8]]))
end
case!("exciton/spatial/x_groups_alias_flat") do
    (; A = exc_spatial(10, W_EXC3; x_groups=[3, 6]))
end
case!("exciton/spatial/num_x4_num_avg2") do
    (; A = exc_spatial(10, W_EXC3; num_x=4, num_avg=2))
end
case!("exciton/spatial/window_x3-8_num_x3") do
    (; A = exc_spatial(10, W_EXC3; num_x=3, x_start=3, x_end=8))
end
case!("exciton/spatial/hodc_default_eta") do
    (; A = exc_spatial(10, W_EXC3; X_list=[4], kernel=:hodc))
end
case!("exciton/spatial/hodc_eta0.05_m6") do
    (; A = exc_spatial(10, W_EXC3; X_list=[4], kernel=:hodc, eta=0.05, m_order=6))
end
case!("exciton/spatial/lorentz_lambda2") do
    (; A = exc_spatial(10, W_EXC3; X_list=[4, 5], kernel=:lorentz, lambda=2.0))
end
case!("exciton/spatial/return_maxlinkdim") do
    A, chi = exc_spatial(10, W_EXC3; X_groups=[[2, 3], [7]], return_maxlinkdim=true)
    (; A, chi)
end
case!("exciton/spatial/maxdim2_return_maxlinkdim") do
    A, chi = exc_spatial(10, W_EXC3; X_list=[4], maxdim=2, cutoff=1e-6, return_maxlinkdim=true)
    (; A, chi)
end
case!("exciton/spatial/not_exciton") do
    get_exciton_ldos_spatial(model(:chain4), 6, W_CHAIN4)
end
case!("exciton/spatial/both_group_keywords") do
    exc_spatial(6, W_EXC3; X_groups=[[1]], x_groups=[[2]])
end
case!("exciton/spatial/list_and_groups") do
    exc_spatial(6, W_EXC3; X_list=[1], x_groups=[[2]])
end
case!("exciton/spatial/num_x_exceeds_window") do
    exc_spatial(6, W_EXC3; num_x=10)
end
case!("exciton/spatial/bad_window") do
    exc_spatial(6, W_EXC3; x_start=5, x_end=3)
end
case!("exciton/spatial/num_x0") do
    exc_spatial(6, W_EXC3; num_x=0)
end
case!("exciton/spatial/num_avg0") do
    exc_spatial(6, W_EXC3; num_avg=0)
end
case!("exciton/spatial/X_out_of_range") do
    exc_spatial(6, W_EXC3; X_list=[9])
end
# Pinned as it is today: spatial_sampling_plan takes first() of every group, so an
# empty group raises a BoundsError before the "empty spatial group" check below it.
case!("exciton/spatial/empty_group") do
    exc_spatial(6, W_EXC3; X_groups=[[1], Int[]])
end
case!("exciton/spatial/empty_list") do
    exc_spatial(6, W_EXC3; X_list=Int[])
end
case!("exciton/ldos/scalar") do
    (; A = get_exciton_ldos(model(:exc3), 4, -3.9; Ncheb=10))
end
case!("exciton/ldos/scalar_outside_support") do
    (; A = get_exciton_ldos(model(:exc3), 4, 4.9; Ncheb=10))
end
case!("exciton/ldos/vector") do
    (; A = get_exciton_ldos(model(:exc3), 4, W_EXC3; Ncheb=10))
end
case!("exciton/ldos/vector_hodc_maxdim3") do
    (; A = get_exciton_ldos(model(:exc3), 6, W_EXC3; Ncheb=10, kernel=:hodc, maxdim=3,
                            cutoff=1e-6))
end
case!("exciton/ldos/vector_lorentz") do
    (; A = get_exciton_ldos(model(:exc3), 6, W_EXC3; Ncheb=10, kernel=:lorentz, lambda=2.0))
end

sep(Ncheb, ws; kw...) = get_exciton_ldos_separation(model(:exc3), Ncheb, ws; kw...)
case!("exciton/separation/open") do
    (; rho = sep(8, W_EXC3; d_list=-1:1, R_list=[1, 4, 8]))
end
case!("exciton/separation/periodic") do
    (; rho = sep(8, W_EXC3; d_list=[0, 2, -3], R_list=[1, 5, 8], boundary=:periodic))
end
case!("exciton/separation/default_R_list_hodc") do
    (; rho = sep(6, W_EXC3; d_list=[1], kernel=:hodc, eta=0.05))
end
case!("exciton/separation/lorentz_maxdim2") do
    (; rho = sep(8, W_EXC3; d_list=[0, 1], R_list=2:3, kernel=:lorentz, lambda=2.0,
                 maxdim=2, cutoff=1e-6))
end
case!("exciton/separation/bad_boundary") do
    sep(6, W_EXC3; d_list=[0], boundary=:bogus)
end
case!("exciton/separation/not_exciton") do
    get_exciton_ldos_separation(model(:chain4), 6, W_CHAIN4; d_list=[0])
end
case!("exciton/radius2/open") do
    d_list = -1:1
    rho = sep(8, W_EXC3; d_list, R_list=[1, 4, 8])
    (; xi2 = exciton_radius2(rho, d_list))
end
case!("exciton/radius2/periodic") do
    d_list = [0, 2, -3]
    rho = sep(8, W_EXC3; d_list, R_list=[1, 5, 8], boundary=:periodic)
    (; xi2 = exciton_radius2(rho, d_list))
end
case!("exciton/radius2/synthetic") do
    rho = zeros(3, 2, 2)
    rho[:, :, 1] = [1.0 2.0; 0.0 0.0; 0.5 NaN]
    rho[:, :, 2] = [0.25 0.75; 3.0 1.0; -1.0 2.0]
    (; xi2 = exciton_radius2(rho, [0, 2]))
end
case!("exciton/radius2/d_list_mismatch") do
    exciton_radius2(zeros(2, 3, 1), [0, 1])
end
# Throws today: the exciton MPO stores its site legs as (s, s'), so
# `getindex.(siteinds(H), 2)` returns the primed indices and the probe MPS no
# longer matches apply(H, ...) at the first `+`.
case!("exciton/ldos_exc_KPM_Tn/X3"; requires=has(:ldos_exc_KPM_Tn)) do
    H = model(:exc3)
    Ham_n = (1 / H.scale) * +(H.mpo, (-H.center) * MPO(H.sites, "Id"); cutoff=1e-8)
    (; mu = TB.ldos_exc_KPM_Tn(Ham_n, 8, 3))
end
case!("legacy/get_mus_raw/chain3"; requires=has(:get_mus_raw)) do
    (; mu = TB.get_mus_raw(chain3_Tn(6)))
end
case!("legacy/compute_dos_ldos_hodc/chain3"; requires=has(:compute_dos_ldos_hodc)) do
    Tn = chain3_Tn(8)
    dos, ldos = TB.compute_dos_ldos_hodc(8, Tn[1:8], 5; eta=0.05, m_order=4, maxdim=20)
    (; dos, n_ldos = length(ldos), ldos_first = dense(ldos[1]), ldos_last = dense(ldos[end]),
       ldos_traces = tr_all(ldos))
end
case!("legacy/compute_dos_ldos_hodc/wrong_length"; requires=has(:compute_dos_ldos_hodc)) do
    TB.compute_dos_ldos_hodc(8, chain3_Tn(8), 3)
end

# ════════════════════════════════════════════════════════════════════════════
# Running and comparing
# ════════════════════════════════════════════════════════════════════════════
"""
    message_prefix(err) -> Union{String,Nothing}

The first MESSAGE_PREFIX_CHARS characters of the first line of `err.msg`, or
`nothing` for an exception without a string message.
"""
function message_prefix(err)
    hasfield(typeof(err), :msg) || return nothing
    msg = getfield(err, :msg)
    msg isa AbstractString || return nothing
    return String(first(first(split(msg, '\n')), MESSAGE_PREFIX_CHARS))
end

"""
    run_case(case) -> (; value, err)

Run one case on freshly seeded global RNG with stdout silenced (the Chebyshev
loops and the aux auto-enable print progress lines). Exactly one of `value`
and `err` is `nothing`.
"""
function run_case(case::Case)
    build_models!()
    Random.seed!(CASE_SEED)
    try
        value = redirect_stdout(devnull) do
            case.f()
        end
        return (; value, err = nothing)
    catch err
        return (; value = nothing, err)
    end
end

_close(a, e) = (isnan(a) && isnan(e)) || isapprox(a, e; rtol=RTOL, atol=ATOL)
_equal(a, e) = a == e || (a isa Number && isnan(a) && isnan(e))

_numeq(a::AbstractFloat, e::AbstractFloat, exact) = exact ? _equal(a, e) : _close(a, e)
_numeq(a::Complex, e::Complex, exact) =
    exact ? (_equal(real(a), real(e)) && _equal(imag(a), imag(e))) : _close(a, e)
_numeq(a, e, exact) = a == e

"""
    mismatch(actual, expected; path="", exact=false) -> Union{Nothing,String}

`nothing` when `actual` matches `expected` under the rules in the file header
(with `exact=true`, floating-point values must be `==`, NaN matching NaN);
otherwise a description of the first difference.
"""
function mismatch(@nospecialize(actual), @nospecialize(expected); path::String = "", exact::Bool = false)
    if expected isa NamedTuple
        actual isa NamedTuple || return "$path: expected a NamedTuple, got $(typeof(actual))"
        keys(actual) == keys(expected) ||
            return "$path: fields $(keys(actual)) != expected $(keys(expected))"
        for k in keys(expected)
            m = mismatch(actual[k], expected[k]; path = "$path.$k", exact)
            m === nothing || return m
        end
        return nothing
    elseif expected isa Tuple
        actual isa Tuple && length(actual) == length(expected) ||
            return "$path: expected a $(length(expected))-tuple, got $(typeof(actual))"
        for i in eachindex(expected)
            m = mismatch(actual[i], expected[i]; path = "$path[$i]", exact)
            m === nothing || return m
        end
        return nothing
    elseif expected isa AbstractArray
        actual isa AbstractArray || return "$path: expected an array, got $(typeof(actual))"
        eltype(actual) == eltype(expected) ||
            return "$path: element type $(eltype(actual)) != expected $(eltype(expected))"
        size(actual) == size(expected) ||
            return "$path: size $(size(actual)) != expected $(size(expected))"
        if eltype(expected) <: Number
            i = findfirst(k -> !_numeq(actual[k], expected[k], exact), eachindex(expected))
            i === nothing && return nothing
            I = CartesianIndices(expected)[i]
            return "$path[$(join(Tuple(I), ","))]: got $(repr(actual[i])), expected $(repr(expected[i]))"
        end
        for (i, I) in zip(eachindex(expected), CartesianIndices(expected))
            m = mismatch(actual[i], expected[i]; path = "$path[$(join(Tuple(I), ","))]", exact)
            m === nothing || return m
        end
        return nothing
    else
        typeof(actual) == typeof(expected) ||
            return "$path: type $(typeof(actual)) != expected $(typeof(expected))"
        _numeq(actual, expected, exact) && return nothing
        return "$path: got $(repr(actual)), expected $(repr(expected))"
    end
end

function check_case(case::Case, @nospecialize(entry))
    got = run_case(case)
    if entry.throws !== nothing
        if got.err === nothing
            @error "KPM output changed: case no longer throws" case = case.name expected = entry.throws
            return false
        end
        if !(got.err isa entry.throws)
            @error "KPM output changed: different exception" case = case.name expected = entry.throws got = typeof(got.err)
            return false
        end
        prefix = message_prefix(got.err)
        prefix == entry.expected.message_prefix && return true
        @error "KPM output changed: different error message" case = case.name expected = entry.expected.message_prefix got = prefix
        return false
    end
    if got.err !== nothing
        @error "KPM output changed: case now throws" case = case.name exception = (got.err, nothing)
        return false
    end
    m = mismatch(got.value, entry.expected)
    m === nothing && return true
    @error "KPM output changed" case = case.name detail = m
    return false
end

function run_tests(entries)
    @testset "KPM outputs are pinned" begin
        names = [c.name for c in CASES]
        @test allunique(names)
        @test length(CASES) == EXPECTED_CASE_COUNT
        @test allunique(e.name for e in entries)
        by_name = Dict(e.name => e for e in entries)
        # A skipped case may keep its entry (data generated while its function
        # existed) or not (data regenerated since); every other case needs one.
        @test Set(keys(by_name)) ==
              Set(c.name for c in CASES if c.requires() || haskey(by_name, c.name))
        skipped = String[]
        for case in CASES
            if !case.requires()
                push!(skipped, case.name)
                continue
            end
            haskey(by_name, case.name) || continue
            @test check_case(case, by_name[case.name])
        end
        isempty(skipped) ||
            @info "golden_kpm: skipped $(length(skipped)) case(s) whose function was deleted" skipped
    end
end

end # module KPMGoldenRunner

if !isdefined(@__MODULE__, :KPM_GOLDEN_GENERATOR)
    KPMGoldenRunner.run_tests(include(joinpath(@__DIR__, "data", "kpm_golden.jl")))
end
