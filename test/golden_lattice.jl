using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test, Random
using TensorBinding: get_Hamiltonian, build_hamiltonian, MODEL_REGISTRY, lattice_positions,
                     bilayer_hamiltonian, multilayer_hamiltonian, twisted_bilayer_hamiltonian,
                     twisted_multilayer_hamiltonian, add_hopping_2D!, get_shell_disps,
                     mask_hamiltonian, add_tjunction!, tjunction_hamiltonian,
                     tjunction_lattice_hamiltonian

# Characterization ("golden") tests for the lattice builders.
#
# These tests pin what the lattice code computes *today*, bugs included, so that
# the Tier 1 reorganisation (docs/dev/REORGANISATION_TODO.md: split the former
# lattice/2Dlattice_tk.jl, move the geometry helpers and MODEL_REGISTRY, delete
# dead code, rename files) cannot silently change a Hamiltonian, a position
# table or a registry entry. The expected values live in
# `test/data/lattice_golden.jl`, written by `test/data/generate_lattice_golden.jl`
# (see its header for how to rerun it).
#
# Covered: every builder in src/lattice/ (masks, legacy and NNN kinetic MPOs,
# H* presets, kagome/lieb/honeycomb/honeycomb_nnn/dice/ssh sublattice
# Hamiltonians and positions, bilayer, multilayer and twisted stacks, flake
# SDFs and mask_hamiltonian, T-junctions, add_hopping_2D! with every amplitude
# form, and the geometry closures, *_positions and _preset_geometry of
# lattice/geometry.jl), MODEL_REGISTRY and build_hamiltonian
# (core/ModelRegistry.jl) and, from core/TBSystem.jl, every get_Hamiltonian
# geometry name plus _estimate_scale and central_index. Left out on purpose
# (other work is changing them): get_Hamiltonian("haldane"), "chernhex"
# (H2DChernhex), haldane_hoppingf/chirality and the default scale of "chern8"
# and "qc2dsquare" (those cases pass an explicit `scale=`).
#
# An MPO is pinned through its dense matrix (row = output/primed index, column
# = input index, first MPO site most significant), its site index dims and
# tags, and its bond dimensions. A TBHamiltonian adds its scalar fields, its
# auxiliary indices and its geometry / geometry_uc closures evaluated at every
# atom.
#
# A failure here means a lattice output changed. If the change is a
# regression, fix the code. If it is intentional, regenerate the data file in
# the same commit as the behaviour change and record it in the changelog.
#
# Comparison rules:
#   * floating-point values and arrays: same element type and shape, and
#     elementwise isapprox(rtol=1e-10, atol=1e-12) (Tier 1 should be exact;
#     the tolerance only absorbs reordered arithmetic). Large matrices are
#     stored sparsely: entries below 1e-14 in magnitude (100x below atol) are
#     stored as zero;
#   * everything else (integers, bond dimensions, tags, symbols, strings,
#     tuples, `nothing`) must be equal and of the same type;
#   * every field recorded in the golden data must still be returned; fields
#     added later are allowed;
#   * a case recorded as throwing must still throw an exception of that type
#     with the recorded `message_prefix` (first MESSAGE_PREFIX_CHARS characters
#     of the first line of its message);
#   * the number of cases per builder, skipped cases (next rule) not counted,
#     must equal EXPECTED_CASE_COUNTS below;
#   * a case that names a function which no longer exists is skipped, not
#     failed, only when that function is on the Tier 1 deletion list
#     (DELETABLE_FUNCTIONS: exactly the lattice-file functions of the
#     checklist's "Delete dead and legacy code" section). A missing function
#     that is not on that list fails the case. The generator leaves skipped
#     cases out of regenerated data; the test accepts the data with or without
#     them.
#
# Every case reseeds the global RNG and the ITensors index-id RNG with its own
# seed before it runs (QTCI draws random pivots from the global RNG).
#
# The runner below is shared with the generator, which includes this file with
# `LATTICE_GOLDEN_GENERATOR` defined so that only the module is loaded.

module LatticeGoldenRunner

using Test
using Random
using LinearAlgebra
using ITensors
using ITensorMPS
using TensorBinding
const TB = TensorBinding

const RTOL = 1e-10
const ATOL = 1e-12
# Sparse storage drops entries with magnitude below DROP_BELOW (100x below ATOL,
# so a dropped entry cannot decide a comparison).
const DROP_BELOW = 1e-14
# Matrices with more than DENSE_MAX_LENGTH entries are stored sparsely.
const DENSE_MAX_LENGTH = 36
const MESSAGE_PREFIX_CHARS = 60

# Number of cases per builder in test/data/lattice_golden.jl, not counting the
# skipped cases of deleted functions (the data may hold them or not; see
# `skipped`). Update it by hand, in the same commit, when cases are added or
# removed on purpose (the generator prints the new counts and warns when they
# differ from these). The deletions of fb8b2d8 took all of :geom_counts and
# :legacy_mask, 6 :legacy_hopping cases and 2 :layer_ops cases.
const EXPECTED_CASE_COUNTS = Dict{Symbol,Int}(
    :add_hopping_2D           => 26,
    :add_tjunction            => 6,
    :bilayer                  => 8,
    :build_hamiltonian        => 18,
    :central_index            => 3,
    :estimate_scale           => 16,
    :geom_positions           => 4,
    :geometry_closure         => 9,
    :get_hamiltonian          => 45,
    :interlayer_mpo           => 5,
    :kinetic2d                => 12,
    :layer_ops                => 2,
    :lattice_positions        => 6,
    :legacy_hopping           => 4,
    :mask                     => 13,
    :mask_hamiltonian         => 6,
    :model_registry           => 1,
    :monolayer                => 5,
    :multilayer               => 5,
    :parse_param_string       => 7,
    :positions                => 8,
    :preset1d                 => 7,
    :preset2d                 => 9,
    :preset_geometry          => 11,
    :resolve_layer_selection  => 5,
    :sdf                      => 11,
    :shell_disps              => 14,
    :shift_mpo_nnn            => 9,
    :shift_primitive          => 3,
    :sublattice_hamiltonian   => 15,
    :sublattice_positions     => 8,
    :tjunction_hamiltonian    => 5,
    :tjunction_lattice        => 3,
    :tjunction_parts          => 5,
    :twisted                  => 5,
)

# Exactly the functions that the Tier 1 checklist (docs/dev/REORGANISATION_TODO.md,
# "Delete dead and legacy code") names for deletion and that live in the lattice
# files (src/lattice/*.jl), and nothing else. A case that calls one of them is
# skipped once the function is gone; while it exists it must still match. A
# case whose function is missing and is NOT listed here fails (check_case):
# deleting a function the checklist does not name, e.g. one still in use, is a
# behaviour change. Change this set only together with that checklist section.
# All of them were deleted in fb8b2d8 except `sdf_interval`, which QPI.jl uses.
const DELETABLE_FUNCTIONS = Set{Symbol}([
    # the former lattice/2Dlattice_tk.jl: `interchain_hopping_*` (2nd_plus/minus, triangle,
    # honeycomb) "with their skeleton/template helpers", `_geom_n_sub`, `_nsublat`
    :interchain_hopping_square_2nd_plus, :interchain_hopping_square_2nd_minus,
    :interchain_hopping_triangle, :interchain_hopping_honeycomb,
    :skeleton, :odd_template, :even_template, :odd_skeleton, :even_skeleton,
    :_geom_n_sub, :_nsublat,
    # Twisted.jl: `postpend_layer_projector/hopping`
    :postpend_layer_projector, :postpend_layer_hopping,
    # Flake.jl: `sdf_interval`
    :sdf_interval,
])

# ── Reproducible state ─────────────────────────────────────────────────────────
function seed_all!(seed::Integer)
    Random.seed!(seed)
    isdefined(ITensors, :index_id_rng) && Random.seed!(ITensors.index_id_rng(), seed)
    return nothing
end

# Silence library println/@warn output while a case runs (a message is not an
# output this file pins).
quiet(f) = redirect_stdout(devnull) do
    Base.CoreLogging.with_logger(f, Base.CoreLogging.NullLogger())
end

# ── MPO / TBHamiltonian summaries ──────────────────────────────────────────────
"""
    mpo_site_inds(M) -> Vector{Index}

The unprimed site indices of `M` in MPO order (the indices of each tensor that
are not shared with a neighbour).
"""
function mpo_site_inds(M::MPO)
    n = length(M)
    out = Index[]
    for k in 1:n
        nb = ITensor[]
        k > 1 && push!(nb, M[k - 1])
        k < n && push!(nb, M[k + 1])
        own = isempty(nb) ? collect(inds(M[k])) : collect(uniqueinds(M[k], nb...))
        append!(out, filter(i -> plev(i) == 0, own))
    end
    return out
end

"""
    dense_matrix(M) -> Matrix

Dense matrix of `M`: row index = primed (output) sites, column index = unprimed
(input) sites, the first MPO site being the most significant digit.

The site tensors are converted to plain arrays (at most four indices each) and
contracted with ordinary matrix products, so that no high-order ITensor
permutation has to be compiled for every system size.
"""
function dense_matrix(M::MPO)
    n = length(M)
    s = mpo_site_inds(M)
    links = [commoninds(M[k], M[k + 1]) for k in 1:n - 1]
    (length(s) == n && all(l -> length(l) == 1, links)) || return _dense_matrix_generic(M, s)
    T = mapreduce(eltype, promote_type, M)
    R = ones(T, 1, 1, 1)                         # (rows so far, columns so far, right link)
    for k in 1:n
        order = Index[]
        k > 1 && push!(order, only(links[k - 1]))
        push!(order, prime(s[k]), s[k])
        k < n && push!(order, only(links[k]))
        d = dim(s[k])
        a = k > 1 ? dim(only(links[k - 1])) : 1
        b = k < n ? dim(only(links[k])) : 1
        A = reshape(convert(Array{T}, Array(M[k], order...)), a, d * d * b)
        nr, nc, _ = size(R)
        P = reshape(reshape(R, nr * nc, a) * A, nr, nc, d, d, b)
        # new row = (output digit of site k, old row) with the old row more significant
        R = reshape(permutedims(P, (3, 1, 4, 2, 5)), d * nr, d * nc, b)
    end
    return R[:, :, 1]
end

function _dense_matrix_generic(M::MPO, s)
    T = M[1]
    for k in 2:length(M)
        T *= M[k]
    end
    A = Array(T, reverse([prime(i) for i in s])..., reverse(s)...)
    D = prod(dim.(s))
    return reshape(A, D, D)
end

function linkdims_of(M::MPO)
    return [prod(dim.(commoninds(M[k], M[k + 1])); init=1) for k in 1:length(M) - 1]
end

site_desc(s) = [(dim(i), string(tags(i))) for i in s]
idx_desc(::Nothing) = nothing
idx_desc(i::Index) = (dim(i), string(tags(i)))

mpo_summary(M::MPO) = (dense = dense_matrix(M), mpo_sites = site_desc(mpo_site_inds(M)),
                       linkdims = linkdims_of(M))

n_atoms(H) = H.N * (H.sublattice_s === nothing ? 1 : dim(H.sublattice_s))

function geom_eval(g, n::Int)
    g === nothing && return nothing
    rows = try
        [Float64.(collect(g(i))) for i in 1:n]
    catch err
        return "throws $(typeof(err))"
    end
    return permutedims(reduce(hcat, rows))
end

function tb_summary(H)
    return (; mpo_summary(H.mpo)...,
            L = H.L, N = H.N, Lx = H.Lx, scale = H.scale, center = H.center,
            aux_side = H.aux_side,
            sites = site_desc(H.sites),
            sites_match_mpo = collect(H.sites) == mpo_site_inds(H.mpo),
            spin_s = idx_desc(H.spin_s), nambu_s = idx_desc(H.nambu_s),
            layer_s = idx_desc(H.layer_s), sublattice_s = idx_desc(H.sublattice_s),
            position_space = string(nameof(typeof(H.position_space))),
            geometry = geom_eval(H.geometry, n_atoms(H)),
            geometry_uc = geom_eval(H.geometry_uc, n_atoms(H)),
            caches_empty = H._tn_cache === nothing && H._tn_mps_cache === nothing &&
                           H._tn_Ncheb == 0 && H._density_cache === nothing,
            no_interaction = H.interaction_mpo === nothing && H.fock_mpo === nothing)
end

# ── Inputs that cannot be stored as literals ───────────────────────────────────
qubits(n) = siteinds("Qubit", n)

"""
    test_hopping_mpo(sites; complex=true) -> MPO

Exact, position-dependent diagonal hopping profile `1 + 0.2 Z_1 - 0.07 Z_2 + c Z_L`
with `c = 0.1 + 0.05im` (or `0.1`), written directly as a bond-dimension-2 MPO
(no QTCI, no OpSum): `W_k = [I  c_k Z_k; 0  I]` between the boundary vectors
`(1, 0)` and `(0, 1)`, the constant riding on the first site.
"""
function test_hopping_mpo(sites; complex::Bool=true)
    L = length(sites)
    L >= 2 || error("test_hopping_mpo needs at least two sites")
    T = complex ? ComplexF64 : Float64
    coef = zeros(T, L)
    coef[1] += 0.2
    L >= 3 && (coef[2] += -0.07)
    coef[L] += complex ? 0.1 + 0.05im : 0.1
    links = [Index(2, "Link,l=$k") for k in 1:L - 1]
    M = MPO(L)
    for k in 1:L
        s = sites[k]
        W = k == 1 ? ITensor(T, s', s, links[1]) :
            k == L ? ITensor(T, links[L - 1], s', s) :
                     ITensor(T, links[k - 1], s', s, links[k])
        for σ in 1:2
            z = σ == 1 ? 1.0 : -1.0
            if k == 1
                W[s' => σ, s => σ, links[1] => 1] = 1.0
                W[s' => σ, s => σ, links[1] => 2] = 1.0 + coef[1] * z
            elseif k == L
                W[links[L - 1] => 1, s' => σ, s => σ] = coef[L] * z
                W[links[L - 1] => 2, s' => σ, s => σ] = 1.0
            else
                W[links[k - 1] => 1, s' => σ, s => σ, links[k] => 1] = 1.0
                W[links[k - 1] => 1, s' => σ, s => σ, links[k] => 2] = coef[k] * z
                W[links[k - 1] => 2, s' => σ, s => σ, links[k] => 2] = 1.0
            end
        end
        M[k] = W
    end
    return M
end

hopping_input(kind::Symbol, sites) =
    kind === :id      ? MPO(sites, "Id") :
    kind === :test    ? test_hopping_mpo(sites) :
    kind === :testre  ? test_hopping_mpo(sites; complex=false) :
    error("unknown hopping input :$kind")

# Amplitude functions for add_hopping_2D! and get_Hamiltonian("custom").
amp_dir(dx, dy, fs, ts)             = 0.1 + 0.02dx - 0.03dy + 0.01im * (ts - fs)
amp_pos1d(n)                        = 0.1 + 0.01n
amp_pos2d(ix, iy)                   = 0.1 + 0.02ix + 0.03im * iy
amp_pos1d_dir(n, dx, dy, fs, ts)    = (0.1 + 0.01n) * (1 + 0.1dx + 0.05im * dy)
amp_pos2d_dir(ix, iy, dx, dy, fs, ts) = 0.1 + 0.01ix * dx + 0.02im * iy * dy + 0.005 * (fs + ts)
amp_bad(a, b, c)                    = 0.0
custom_nn(i, j)                     = abs(i - j) == 1 ? -1.0 : (i == j ? 0.1 * i : 0.0)
custom_cplx(i, j)                   = abs(i - j) == 1 ? -1.0 + 0.2im * sign(j - i) : 0.0

const AMPLITUDES = Dict{Symbol,Function}(
    :amp_dir => amp_dir, :amp_pos1d => amp_pos1d, :amp_pos2d => amp_pos2d,
    :amp_pos1d_dir => amp_pos1d_dir, :amp_pos2d_dir => amp_pos2d_dir,
    :amp_bad => amp_bad, :custom_nn => custom_nn, :custom_cplx => custom_cplx)

amplitude(f) = f isa Symbol ? AMPLITUDES[f] : f

"""
    make_sdf(spec)

`spec = (kind, args...)`: a primitive `(:disk, cx, cy, r)`, `(:rect, cx, cy, w, h)`,
`(:halfplane, nx, ny, d)`, `(:annulus, cx, cy, r_in, r_out)`,
`(:polygon, vertices)`, `(:interval, lo, hi)`, or a combination
`(:union | :intersect | :subtract, spec_f, spec_g)`.
"""
function make_sdf(spec::Tuple)
    kind = spec[1]
    kind === :disk      && return TB.sdf_disk(spec[2:end]...)
    kind === :rect      && return TB.sdf_rect(spec[2:end]...)
    kind === :halfplane && return TB.sdf_halfplane(spec[2:end]...)
    kind === :annulus   && return TB.sdf_annulus(spec[2:end]...)
    kind === :polygon   && return TB.sdf_convex_polygon(spec[2])
    kind === :interval  && return TB.sdf_interval(spec[2:end]...)
    kind === :union     && return TB.sdf_union(make_sdf(spec[2]), make_sdf(spec[3]))
    kind === :intersect && return TB.sdf_intersect(make_sdf(spec[2]), make_sdf(spec[3]))
    kind === :subtract  && return TB.sdf_subtract(make_sdf(spec[2]), make_sdf(spec[3]))
    error("unknown sdf kind :$kind")
end

"""
    build_base(b) -> TBHamiltonian

A Hamiltonian that a case starts from: `b.via` is `:get_hamiltonian`
(`b.geometry`, `b.params`, `b.kw`), `:bilayer` (`b.lattice`, `b.Lx`, `b.Ly`,
`b.kw`) or `:no_geometry` (the base `b.of` with `geometry = nothing`).
"""
function build_base(b::NamedTuple)
    b.via === :get_hamiltonian && return TB.get_Hamiltonian(b.geometry, b.params; b.kw...)
    b.via === :bilayer         && return TB.bilayer_hamiltonian(b.lattice, b.Lx, b.Ly; b.kw...)
    b.via === :no_geometry     && return TB.TBHamiltonian(build_base(b.of); geometry=nothing)
    error("unknown base :$(b.via)")
end

# add_hopping_2D! geometry keyword built in the runner (a coordinate matrix or a
# closure over it), independent of the package's position helpers.
function square_rows(Lx, Ly)
    Nx = 2^Lx
    return Float64[j == 1 ? (i - 1) % Nx : (i - 1) ÷ Nx for i in 1:2^(Lx + Ly), j in 1:2]
end

function hopping_kwargs(kw::NamedTuple)
    haskey(kw, :geometry) || return kw
    g = kw.geometry
    rows = g.rows === :square ? square_rows(g.Lx, g.Ly) : error("unknown geometry rows :$(g.rows)")
    rows = rows[1:g.nrows, :]
    geom = g.kind === :matrix ? rows : let m = rows; i -> m[i, :]; end
    return merge(kw, (; geometry = geom))
end

# ── Case dispatch ──────────────────────────────────────────────────────────────
# One small function per builder (a single function with every branch took
# ~18 s to compile on its first call). Each takes the case's `spec` and returns
# every output as a record of comparable fields.

function case_shift_primitive(@nospecialize(spec))
    s = qubits(spec.L)
    return mpo_summary(getfield(TB, spec.fn)(s, spec.num_site))
end

function case_mask(@nospecialize(spec))
    s = qubits(spec.Lx + spec.Ly)
    fn = getfield(TB, spec.fn)
    M = spec.fn === :_row_checker_mpo ? fn(spec.Lx, spec.Ly, s) : fn(spec.Lx, spec.Ly, s; spec.kw...)
    return mpo_summary(M)
end

function case_legacy(@nospecialize(spec))
    s = qubits(spec.Lx + spec.Ly)
    kw = haskey(spec, :hopping) ? merge(spec.kw, (; hopping = hopping_input(spec.hopping, s))) : spec.kw
    return mpo_summary(getfield(TB, spec.fn)(2^spec.Lx, 2^(spec.Lx + spec.Ly), s; kw...))
end

function case_kinetic2d(@nospecialize(spec))
    s = qubits(spec.Lx + spec.Ly)
    fn = getfield(TB, spec.fn)
    hop = hopping_input(spec.hopping, s)
    M = spec.nn === nothing ? fn(spec.Lx, spec.Ly, s, hop; spec.kw...) :
                              fn(spec.Lx, spec.Ly, s, hop, spec.nn; spec.kw...)
    return mpo_summary(M)
end

case_preset(@nospecialize(spec)) = mpo_summary(getfield(TB, spec.fn)(spec.args...; spec.kw...))

case_sublattice_positions(@nospecialize(spec)) = (; value = getfield(TB, spec.fn)(spec.Lx, spec.Ly))

case_geom_positions(@nospecialize(spec)) =
    (; value = TB._geom_positions(Val(spec.lattice), spec.Lx, spec.Ly))

case_tb_builder(@nospecialize(spec)) = tb_summary(getfield(TB, spec.fn)(spec.args...; spec.kw...))

function case_parse_param_string(@nospecialize(spec))
    d = TB._parse_param_string(spec.s)
    entries = Tuple((k, d[k] isa AbstractString ? String(d[k]) : d[k], string(typeof(d[k])))
                    for k in sort!(collect(keys(d))))
    return (; entries)
end

function case_model_registry(@nospecialize(spec))
    keys_kept = filter(k -> !(k in spec.skip), sort!(collect(keys(TB.MODEL_REGISTRY))))
    entries = Tuple((k, TB.MODEL_REGISTRY[k]...) for k in keys_kept)
    return (; keys = Tuple(keys_kept), entries, value_type = string(valtype(TB.MODEL_REGISTRY)))
end

case_build_hamiltonian(@nospecialize(spec)) =
    mpo_summary(TB.build_hamiltonian(spec.model, spec.dims...; spec.kw...))

function case_get_hamiltonian(@nospecialize(spec))
    kw = spec.kw
    ref = nothing
    if get(spec, :ref_sites, false)
        ref = qubits(kw.L)
        kw = merge(kw, (; ref_sites = ref))
    end
    if haskey(spec, :custom_geometry)
        g = spec.custom_geometry
        m = square_rows(g.Lx, g.Ly)
        kw = merge(kw, (; geometry = g.kind === :matrix ? m : let m = m; i -> m[i, :]; end))
    end
    H = TB.get_Hamiltonian(spec.geometry, amplitude(spec.params); kw...)
    out = tb_summary(H)
    ref === nothing && return out
    return (; out..., sites_are_ref = collect(H.sites) == ref)
end

function case_geometry_closure(@nospecialize(spec))
    g = spec.fn === :_chain_geometry ? TB._chain_geometry() : getfield(TB, spec.fn)(spec.Nx)
    return (; value = geom_eval(g, spec.n))
end

function case_preset_geometry(@nospecialize(spec))
    g = TB._preset_geometry(spec.geometry, spec.Nx)
    return (; is_nothing = g === nothing, value = geom_eval(g, spec.n))
end

case_estimate_scale(@nospecialize(spec)) = (; value = TB._estimate_scale(spec.geometry, spec.params))

case_positions(@nospecialize(spec)) = (; value = getfield(TB, spec.fn)(spec.L; spec.kw...))

case_lattice_positions(@nospecialize(spec)) =
    (; value = TB.lattice_positions(spec.lattice, spec.Lx, spec.Ly; spec.kw...))

case_geom_counts(@nospecialize(spec)) = (; value = getfield(TB, spec.fn)(Val(spec.lattice)))

function case_central_index(@nospecialize(spec))
    spec.kind === :closure &&
        return (; value = TB.central_index(getfield(TB, spec.fn)(spec.Nx), spec.N))
    return (; value = TB.central_index(build_base(spec.base)))
end

function case_monolayer(@nospecialize(spec))
    s = qubits(spec.Lx + spec.Ly)
    M = TB.monolayer_hamiltonian(spec.lattice, spec.Lx, spec.Ly, s; spec.kw...)
    return (; mpo_summary(M)..., sites_are_input = mpo_site_inds(M) == s)
end

function case_layer_ops(@nospecialize(spec))
    s = qubits(2)
    layer = Index(spec.nlayers, "Layer")
    return mpo_summary(getfield(TB, spec.fn)(test_hopping_mpo(s), layer, spec.levels...))
end

function case_interlayer_mpo(@nospecialize(spec))
    s = qubits(spec.Lx + spec.Ly)
    return mpo_summary(TB.interlayer_mpo(spec.lattice, spec.stacking, spec.Lx, spec.Ly, s; spec.kw...))
end

case_bilayer(@nospecialize(spec)) =
    tb_summary(TB.bilayer_hamiltonian(spec.lattice, spec.Lx, spec.Ly; spec.kw...))

case_multilayer(@nospecialize(spec)) =
    tb_summary(TB.multilayer_hamiltonian(spec.lattice, spec.Lx, spec.Ly, spec.n_layers; spec.kw...))

function case_sdf(@nospecialize(spec))
    f = make_sdf(spec.sdf)
    vals = spec.dim == 1 ? Float64[f(x) for x in spec.xs] :
                           Float64[f(x, y) for x in spec.xs, y in spec.ys]
    return (; value = vals)
end

function case_mask_hamiltonian(@nospecialize(spec))
    H = build_base(spec.base)
    Hm = TB.mask_hamiltonian(H, make_sdf(spec.sdf); spec.kw...)
    return (; tb_summary(Hm)..., input_untouched = H.mpo !== Hm.mpo)
end

function case_tjunction_parts(@nospecialize(spec))
    spec.fn === :tjunction_index && return (; value = idx_desc(TB.tjunction_index()))
    spec.fn === :tjunction_positions &&
        return (; value = TB.tjunction_positions(spec.N, spec.junction_site))
    return mpo_summary(TB._site_projector_mpo(spec.L, qubits(spec.L), spec.n))
end

function case_add_tjunction(@nospecialize(spec))
    H = build_base(spec.base)
    out = TB.add_tjunction!(H, spec.t_j; spec.kw...)
    return (; tb_summary(H)..., returns_input = out === H)
end

case_tjunction_hamiltonian(@nospecialize(spec)) =
    tb_summary(TB.tjunction_hamiltonian(spec.args...; spec.kw...))

case_tjunction_lattice(@nospecialize(spec)) =
    tb_summary(TB.tjunction_lattice_hamiltonian(spec.args...; spec.kw...))

function case_shift_mpo_nnn(@nospecialize(spec))
    s = qubits(spec.Lx + spec.Ly)
    N = 2^(spec.Lx + spec.Ly)
    ku, kd = TB.generate_kin_u(s, N), TB.generate_kin_d(s, N)
    Id = MPO(s, "Id")
    brk = TB._row_break_mpo(spec.Lx, spec.Ly, s; which=:xplus)
    M = TB._shift_mpo(spec.dx, spec.dy, ku, kd, Id, brk, 2^spec.Lx; spec.kw...)
    return (; mpo_summary(M)..., is_identity_object = M === Id)
end

case_shell_disps(@nospecialize(spec)) =
    (; value = TB.get_shell_disps(build_base(spec.base), spec.nn; Lx=spec.Lx, Ly=spec.Ly))

function case_add_hopping_2D(@nospecialize(spec))
    H = build_base(spec.base)
    out = TB.add_hopping_2D!(H, amplitude(spec.f); hopping_kwargs(spec.kw)...)
    return (; tb_summary(H)..., returns_input = out === H)
end

case_resolve_layer_selection(@nospecialize(spec)) =
    (; value = TB._resolve_layer_selection(Index(spec.nlayers, "Layer"), spec.layer))

const BUILDERS = Dict{Symbol,Function}(
    :shift_primitive => case_shift_primitive, :mask => case_mask,
    :legacy_hopping => case_legacy, :legacy_mask => case_legacy,
    :kinetic2d => case_kinetic2d, :preset1d => case_preset, :preset2d => case_preset,
    :sublattice_positions => case_sublattice_positions, :geom_positions => case_geom_positions,
    :sublattice_hamiltonian => case_tb_builder, :twisted => case_tb_builder,
    :parse_param_string => case_parse_param_string, :model_registry => case_model_registry,
    :build_hamiltonian => case_build_hamiltonian, :get_hamiltonian => case_get_hamiltonian,
    :geometry_closure => case_geometry_closure, :preset_geometry => case_preset_geometry,
    :estimate_scale => case_estimate_scale, :positions => case_positions,
    :lattice_positions => case_lattice_positions, :geom_counts => case_geom_counts,
    :central_index => case_central_index, :monolayer => case_monolayer,
    :layer_ops => case_layer_ops, :interlayer_mpo => case_interlayer_mpo,
    :bilayer => case_bilayer, :multilayer => case_multilayer, :sdf => case_sdf,
    :mask_hamiltonian => case_mask_hamiltonian, :tjunction_parts => case_tjunction_parts,
    :add_tjunction => case_add_tjunction, :tjunction_hamiltonian => case_tjunction_hamiltonian,
    :tjunction_lattice => case_tjunction_lattice, :shift_mpo_nnn => case_shift_mpo_nnn,
    :shell_disps => case_shell_disps, :add_hopping_2D => case_add_hopping_2D,
    :resolve_layer_selection => case_resolve_layer_selection,
)

"""
    run_case(builder, spec) -> NamedTuple

Build the inputs described by `spec`, call the lattice function(s) of
`builder` (see BUILDERS) and return every output as a record of comparable
fields.
"""
function run_case(builder::Symbol, @nospecialize(spec::NamedTuple))
    haskey(BUILDERS, builder) || error("LatticeGoldenRunner: unknown builder :$builder")
    return BUILDERS[builder](spec)
end

# The TensorBinding function behind each make_sdf kind.
const SDF_FUNCTIONS = Dict{Symbol,Symbol}(
    :disk => :sdf_disk, :rect => :sdf_rect, :halfplane => :sdf_halfplane,
    :annulus => :sdf_annulus, :polygon => :sdf_convex_polygon, :interval => :sdf_interval,
    :union => :sdf_union, :intersect => :sdf_intersect, :subtract => :sdf_subtract)

"""
    case_functions(spec) -> Vector{Symbol}

The TensorBinding functions a case names in its spec: `spec.fn` and the SDF
builders of `spec.sdf`. (Functions a builder calls without naming them in the
spec are reached directly; if one is gone, the case throws and fails.)
"""
function case_functions(@nospecialize(spec::NamedTuple))
    names = Symbol[]
    haskey(spec, :fn) && spec.fn isa Symbol && push!(names, spec.fn)
    haskey(spec, :sdf) && _sdf_functions!(names, spec.sdf)
    return unique!(names)
end

function _sdf_functions!(names, spec::Tuple)
    spec[1] isa Symbol && haskey(SDF_FUNCTIONS, spec[1]) && push!(names, SDF_FUNCTIONS[spec[1]])
    for x in spec
        x isa Tuple && !isempty(x) && x[1] isa Symbol && _sdf_functions!(names, x)
    end
    return names
end

# Whether TensorBinding still defines `f`: the one place the runner asks.
function_defined(f::Symbol) = isdefined(TB, f)

"""
    missing_functions(spec) -> Vector{Symbol}

The functions of `case_functions(spec)` that TensorBinding no longer defines.
"""
missing_functions(@nospecialize(spec::NamedTuple)) = filter(f -> !function_defined(f), case_functions(spec))

"""
    skipped(spec) -> Bool

Whether a case is skipped: some function it names is gone, and every such
function is on DELETABLE_FUNCTIONS. The generator leaves these cases out.
"""
function skipped(@nospecialize(spec::NamedTuple))
    gone = missing_functions(spec)
    return !isempty(gone) && all(in(DELETABLE_FUNCTIONS), gone)
end

"""
    evaluate(builder, spec, seed) -> (result, error)

Run one case from a freshly seeded RNG with library output silenced.
"""
function evaluate(builder::Symbol, @nospecialize(spec::NamedTuple), seed::Integer)
    seed_all!(seed)
    try
        return quiet(() -> run_case(builder, spec)), nothing
    catch err
        return nothing, err
    end
end

# ── Storage encoding ───────────────────────────────────────────────────────────
"""
    encode(x)

Storage form of a result: matrices with more than DENSE_MAX_LENGTH numeric
entries become a sparse record (entries below DROP_BELOW dropped); records and
tuples are encoded field by field; everything else is kept as is.
"""
function encode(x)
    if x isa NamedTuple
        return NamedTuple{keys(x)}(map(encode, values(x)))
    elseif x isa AbstractMatrix{<:Number} && !(eltype(x) <: Integer) && length(x) > DENSE_MAX_LENGTH
        idx = findall(v -> abs(v) >= DROP_BELOW, vec(x))
        return (; __sparse__ = true, size = size(x), eltype = eltype(x),
                  idx = collect(Int, idx), vals = collect(eltype(x), vec(x)[idx]))
    else
        return x
    end
end

is_sparse(x) = x isa NamedTuple && haskey(x, :__sparse__)

function decode(x)
    if is_sparse(x)
        A = zeros(x.eltype, x.size...)
        A[x.idx] .= x.vals
        return A
    elseif x isa NamedTuple
        return NamedTuple{keys(x)}(map(decode, values(x)))
    else
        return x
    end
end

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

# ── Comparison ─────────────────────────────────────────────────────────────────
_is_float_like(T) = T <: Number && !(T <: Integer)

"""
    mismatch(actual, expected) -> Union{Nothing,String}

`nothing` when `actual` matches the (decoded) golden value under the rules in
the file header, otherwise a description of the first difference.
"""
function mismatch(@nospecialize(actual), @nospecialize(expected))
    if expected isa NamedTuple
        actual isa NamedTuple || return "expected a record, got $(typeof(actual))"
        for k in keys(expected)
            haskey(actual, k) || return "field `$k` missing"
            d = mismatch(actual[k], expected[k])
            d === nothing || return "$k: $d"
        end
        return nothing
    elseif expected isa Tuple
        (actual isa Tuple && length(actual) == length(expected)) ||
            return "expected $(repr(expected)), got $(repr(actual))"
        for i in eachindex(expected)
            d = mismatch(actual[i], expected[i])
            d === nothing || return "[$i]: $d"
        end
        return nothing
    elseif expected isa AbstractArray
        actual isa AbstractArray || return "expected an array, got $(typeof(actual))"
        size(actual) == size(expected) || return "size $(size(actual)) != expected $(size(expected))"
        if eltype(expected) <: Number
            eltype(actual) == eltype(expected) ||
                return "element type $(eltype(actual)) != expected $(eltype(expected))"
            if _is_float_like(eltype(expected))
                bad = findfirst(i -> !isapprox(actual[i], expected[i]; rtol=RTOL, atol=ATOL),
                                eachindex(expected))
                bad === nothing && return nothing
                maxdiff = maximum(abs.(actual .- expected))
                return "entry $(Tuple(CartesianIndices(expected)[bad])): got $(actual[bad]), " *
                       "expected $(expected[bad]) (max abs difference $maxdiff)"
            end
            actual == expected && return nothing
            i = findfirst(k -> actual[k] != expected[k], eachindex(expected))
            return "entry $i: got $(repr(actual[i])), expected $(repr(expected[i]))"
        end
        for i in eachindex(expected)
            d = mismatch(actual[i], expected[i])
            d === nothing || return "[$i]: $d"
        end
        return nothing
    elseif expected isa Number && _is_float_like(typeof(expected))
        typeof(actual) == typeof(expected) || return "type $(typeof(actual)) != expected $(typeof(expected))"
        isapprox(actual, expected; rtol=RTOL, atol=ATOL) && return nothing
        return "got $(repr(actual)), expected $(repr(expected))"
    else
        (typeof(actual) == typeof(expected) && isequal(actual, expected)) && return nothing
        return "got $(repr(actual)) ::$(typeof(actual)), expected $(repr(expected)) ::$(typeof(expected))"
    end
end

function check_case(@nospecialize(case))
    gone = missing_functions(case.spec)
    if !isempty(gone)
        unlisted = filter(f -> !(f in DELETABLE_FUNCTIONS), gone)
        if isempty(unlisted)
            @info "Lattice golden case skipped: $(join(gone, ", ")) deleted (Tier 1 deletion list)" case = case.name
            @test_skip isempty(gone)
        else
            @error "Lattice function removed but not on the Tier 1 deletion list (DELETABLE_FUNCTIONS)" case = case.name functions = unlisted
            @test isempty(unlisted)
        end
        return
    end
    actual, err = evaluate(case.builder, case.spec, case.seed)
    if case.throws !== nothing
        if err === nothing
            @error "Lattice output changed: case no longer throws" case = case.name expected = case.throws
            @test false
        elseif !(err isa case.throws)
            @error "Lattice output changed: different exception" case = case.name expected = case.throws got = typeof(err)
            @test false
        else
            got = message_prefix(err)
            expected = case.expected.message_prefix
            got == expected ||
                @error "Lattice output changed: different error message" case = case.name expected got
            @test got == expected
        end
        return
    end
    if err !== nothing
        @error "Lattice output changed: case now throws" case = case.name exception = err
        @test err === nothing
        return
    end
    expected = decode(case.expected)
    for field in keys(expected)
        d = mismatch(haskey(actual, field) ? actual[field] : missing, expected[field])
        d === nothing || @error "Lattice output changed" case = case.name field detail = d
        @test d === nothing
    end
end

# Cases per builder, skipped cases (see `skipped`) not counted.
function case_counts(cases)
    kept = [c for c in cases if !skipped(c.spec)]
    return Dict{Symbol,Int}(b => count(c -> c.builder === b, kept) for b in unique(c.builder for c in kept))
end

function run_tests(cases)
    @testset "Lattice outputs are pinned" begin
        @test allunique(case.name for case in cases)
        counts = case_counts(cases)
        counts == EXPECTED_CASE_COUNTS ||
            @error "Golden case counts differ from EXPECTED_CASE_COUNTS" got = counts expected = EXPECTED_CASE_COUNTS
        @test counts == EXPECTED_CASE_COUNTS
        # Every deletable function that still exists must be one that some case
        # calls (no stale entries); a deleted one may have lost its cases already.
        named = reduce(union!, (case_functions(case.spec) for case in cases); init = Set{Symbol}())
        unused = setdiff(filter(function_defined, DELETABLE_FUNCTIONS), named)
        isempty(unused) || @error "DELETABLE_FUNCTIONS names functions no case calls" unused
        @test isempty(unused)
        for builder in unique(case.builder for case in cases)
            @testset "$builder" begin
                for case in cases
                    case.builder === builder && check_case(case)
                end
            end
        end
    end
end

end # module LatticeGoldenRunner

if !isdefined(@__MODULE__, :LATTICE_GOLDEN_GENERATOR)
    LatticeGoldenRunner.run_tests(include(joinpath(@__DIR__, "data", "lattice_golden.jl")))
end
