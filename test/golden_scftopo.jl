using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test, Random
using TensorBinding: get_Hamiltonian, add_onsite!, add_hopping!, add_spin!, add_zeeman!,
                     add_soc!, add_interaction!, add_superconductivity!, get_scf, get_density,
                     KPM_Tn, chern_marker, winding_marker

# Characterization ("golden") tests for the SCF, superconductivity, purification and
# topology code and for the TBSystem mutators:
#
#   src/physics/SCF.jl            get_scf (every channel), the SCF drivers, BdG
#                                 builders, mean-field and profile helpers
#   src/physics/Supercond.jl      pairing MPOs and spin/BdG assemblers
#   src/physics/Purification.jl   McWeeny/SP2, get_density (:mcweeny/:sp2/:kpm),
#                                 sign_mpo, finite-difference DOS/LDOS
#   src/physics/Topology.jl       projector, winding and Chern markers, valley
#                                 operator/projectors/Chern, Thouless pump
#   src/core/TBSystem.jl          add_onsite!, add_hopping!, add_interaction!
#   src/core/AuxDOF.jl            spin/Nambu indices and operator tables, prepend/
#                                 postpend wrappers, add_spin!, add_zeeman!,
#                                 add_soc!, add_superconductivity!
#
# They pin what the code computes *today*, bugs included, so that the Tier 1 moves of
# docs/dev/REORGANISATION_TODO.md (splitting files, moving helpers, deleting dead
# code, renaming files) cannot change an output unnoticed. The expected values live
# in test/data/scftopo_golden.jl, written by test/data/generate_scftopo_golden.jl
# (see its header for when and how to regenerate).
#
# A failure means an output changed. If that is a regression, fix the code. If it is
# intentional, regenerate the data file in the same commit and review its diff.
#
# Every case builds its inputs from scratch on tiny systems (L = 2..4) with explicit
# spectral scales, so no DMRG scale estimate runs. The global RNG is re-seeded before
# each case from the case name (QTCI draws random pivots from it); stdout and log
# records are silenced while a case runs (several routines println unconditionally,
# and up to v0.1.1 rms_error triggered an ITensors deprecation warning). Warnings are
# not pinned.
#
# Runtime (2026-09, workstation, one thread, other Julia jobs running): ~3 min cold.
# Nearly all of it is first-call compilation shared with any test of this package:
# the first 3-site chain compiles ~27 s of ITensors MPO code, the first QTCI call
# ~15 s and the first ComplexF64 QTCI ~12 s; no single case computes for over a second.
#
# Comparison rules (ScftopoGolden.matches):
#   * Float64 / ComplexF64 scalars and arrays: same type and shape, and
#     isapprox(rtol=1e-10, atol=1e-12) (norm-wise for arrays);
#   * Int, Bool, Symbol, String, nothing: same type and equal;
#   * NamedTuples: the same field names, compared field by field; tuples and other
#     vectors element by element;
#   * a case (or an `attempt`ed call inside a case) that threw must still throw the
#     same exception type with the same first line of its message (first
#     MESSAGE_PREFIX_CHARS characters, Index ids masked);
#   * the set of case names in the data must equal the set of cases defined here.
#
# MPOs and MPSs are compared through their dense matrix / vector (site 1 = most
# significant digit, columns = unprimed legs). Matrices with more than FULL_MAX
# entries are stored as a fingerprint (size, diagonal, first row and column, sum,
# weighted sum, Frobenius norm) to keep the data file small.

module ScftopoGolden

using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test, Random
using TensorBinding: get_Hamiltonian, add_onsite!, add_hopping!, add_spin!, add_zeeman!,
                     add_soc!, add_interaction!, add_superconductivity!, get_scf, get_density,
                     KPM_Tn, chern_marker, winding_marker
const TB = TensorBinding

const RTOL = 1e-10
const ATOL = 1e-12
const MESSAGE_PREFIX_CHARS = 60
const FULL_MAX = 256          # dense matrices up to 16x16 are stored in full

# ═════════════════════════════════════════════════════════════════════════════
# Dense views of MPOs / MPSs and summaries of the package's objects
# ═════════════════════════════════════════════════════════════════════════════

"Unprimed site index of every tensor of `mpo`, in chain order."
mpo_sites(mpo::MPO) = [only(filter(i -> plev(i) == 0, siteinds(mpo, k))) for k in eachindex(mpo)]

"Dense matrix of `mpo` in the order of `sites` (site 1 = most significant digit)."
function dense(mpo::MPO, sites=mpo_sites(mpo))
    T = ITensor(1.0)
    for k in eachindex(mpo)
        T *= mpo[k]
    end
    D = prod(dim, sites)
    return reshape(Array(T, prime.(reverse(sites))..., reverse(sites)...), D, D)
end

"Dense vector of `psi` in the order of its site indices."
function densev(psi::MPS)
    sites = siteinds(psi)
    T = ITensor(1.0)
    for k in eachindex(psi)
        T *= psi[k]
    end
    return reshape(Array(T, reverse(sites)...), prod(dim, sites))
end

"The matrix itself when small, a fingerprint otherwise."
function mat(M::AbstractMatrix)
    length(M) <= FULL_MAX && return Matrix(M)
    n, m = size(M)
    w = [cos(0.7 * i + 1.3 * j) for i in 1:n, j in 1:m]
    return (; fingerprint = true, size = size(M), diag = diag(M), row1 = M[1, :],
            col1 = M[:, 1], total = sum(M), weighted = sum(w .* M), fro = norm(M))
end

mpofp(mpo::MPO) = mat(dense(mpo))
mpofp(mpo::MPO, sites) = mat(dense(mpo, sites))

_nsub(H) = H.sublattice_s === nothing ? 1 : dim(H.sublattice_s)
_auxpos(H, s) = s === nothing ? nothing : findfirst(==(s), H.sites)

"Every field of a TBHamiltonian that the mutators and SCF builders touch."
function hsum(H)
    sites_match = issetequal(H.sites, mpo_sites(H.mpo))
    natoms = H.N * _nsub(H)
    geom(g) = g === nothing ? nothing : attempt(() -> [Vector{Float64}(g(i)) for i in 1:natoms])
    return (; mpo = sites_match ? mpofp(H.mpo, H.sites) : mpofp(H.mpo),
            sites_match, dims = [dim(s) for s in H.sites],
            tags = [string(tags(s)) for s in H.sites],
            L = H.L, N = H.N, Lx = H.Lx, scale = H.scale, center = H.center,
            aux_side = H.aux_side,
            spin_pos = _auxpos(H, H.spin_s), nambu_pos = _auxpos(H, H.nambu_s),
            layer_pos = _auxpos(H, H.layer_s), sublattice_pos = _auxpos(H, H.sublattice_s),
            geometry = geom(H.geometry), geometry_uc = geom(H.geometry_uc),
            interaction = H.interaction_mpo === nothing ? nothing : mpofp(H.interaction_mpo),
            fock = H.fock_mpo === nothing ? nothing : mpofp(H.fock_mpo),
            tn_Ncheb = H._tn_Ncheb, has_tn_cache = H._tn_cache !== nothing,
            has_density_cache = H._density_cache !== nothing,
            position_space = string(nameof(typeof(H.position_space))))
end

"Recursively replace MPOs, MPSs and Hamiltonians by their dense summaries."
summ(x::MPS) = densev(x)
summ(x::MPO) = mpofp(x)
summ(x::TB.TBHamiltonian) = hsum(x)
summ(x::NamedTuple) = map(summ, x)
summ(x::Tuple) = map(summ, x)
summ(x::AbstractVector{<:MPO}) = [summ(m) for m in x]
summ(x::AbstractVector{<:MPS}) = [summ(m) for m in x]
summ(x) = x

"Density-matrix functionals: dense summary, trace and idempotency residual."
function dmsum(rho::MPO)
    M = dense(rho)
    return (; rho = mat(M), trace = tr(M), idempotency = norm(M * M - M))
end

# ═════════════════════════════════════════════════════════════════════════════
# Exceptions
# ═════════════════════════════════════════════════════════════════════════════

"First line of `err.msg`, Index ids and addresses masked, cut to MESSAGE_PREFIX_CHARS."
function message_prefix(err)
    hasfield(typeof(err), :msg) || return nothing
    msg = getfield(err, :msg)
    msg isa AbstractString || return nothing
    line = first(split(msg, '\n'))
    line = replace(line, r"id=\d+" => "id=#", r"0x[0-9a-fA-F]+" => "0x#")
    return String(first(line, MESSAGE_PREFIX_CHARS))
end

"`f()`, or a record of the exception it throws."
function attempt(f)
    try
        return f()
    catch err
        return (; throws = string(nameof(typeof(err))), message_prefix = message_prefix(err))
    end
end

# ═════════════════════════════════════════════════════════════════════════════
# Cases
# ═════════════════════════════════════════════════════════════════════════════

const CASES = Tuple{String,Function}[]
case!(f::Function, name::String) = (push!(CASES, (name, f)); nothing)

"Seed of a case, derived from its name only (independent of the case order)."
case_seed(name) = foldl((h, c) -> (31 * h + Int(c)) % 2_147_483_647, codeunits(name); init = 17)

"Run one case: seed the global RNG, silence stdout and log records, record the value or
the exception."
function run_case(name::String, f::Function)
    Random.seed!(case_seed(name))
    try
        value = redirect_stdout(devnull) do
            Base.CoreLogging.with_logger(Base.CoreLogging.NullLogger()) do
                summ(f())
            end
        end
        return (; throws = nothing, expected = value)
    catch err
        return (; throws = string(nameof(typeof(err))),
                expected = (; message_prefix = message_prefix(err)))
    end
end

# ── Small Hamiltonians (fresh indices every call; explicit scales, no DMRG) ─────
chain(L = 3; t = 1.0, scale = 2.5) = get_Hamiltonian("chain_1d", t; L = L, scale = scale)
spinful_chain(L = 2; scale = 2.5) = (H = chain(L; scale = scale); add_spin!(H); H.scale = scale; H)
honeycomb() = get_Hamiltonian("honeycomb", 1.0; L = 2, Lx = 1, Ly = 1)
kagome() = get_Hamiltonian("kagome", 1.0; L = 2, Lx = 1, Ly = 1)
ssh(; L = 3, d = -0.3) = get_Hamiltonian("ssh_sublattice", (t = 1.0, d = d); L = L)
# Tiny Chern model without the lattice presets (which are pinned elsewhere and cost
# ~30 s of compilation): Hofstadter hopping on an open 4x4 square patch, flux 1/4 per
# plaquette, site n = ix + 4 iy. Spectrum in [-2.65, 2.65], gap [-0.2, 0.2] at half
# filling. The "custom" QTCI build reproduces the matrix to 1e-15 for any seed.
function hofstadter_hop(i, j)
    a, b = round(Int, i) - 1, round(Int, j) - 1
    ax, ay, bx, by = a % 4, a ÷ 4, b % 4, b ÷ 4
    ay == by && abs(ax - bx) == 1 && return ComplexF64(-1.0)
    ax == bx && by == ay + 1 && return -cis(-π / 2 * ax)
    ax == bx && by == ay - 1 && return -cis(π / 2 * ax)
    return 0.0im
end
const HOFSTADTER_RS = hcat(Float64[n % 4 for n in 0:15], Float64[n ÷ 4 for n in 0:15])
hofstadter(; scale = 3.0) = get_Hamiltonian("custom", hofstadter_hop; L = 4, scale = scale,
                                            geometry = HOFSTADTER_RS)
function semenoff_honeycomb(; M = 0.4, scale = 3.5)
    H = honeycomb()
    add_onsite!(H, M; sublat = 1)
    add_onsite!(H, -M; sublat = 2)
    H.scale = scale           # add_onsite! resets the scale to 0 (DMRG sentinel)
    return H
end
"Two layers built by hand (the bilayer builders are lattice code, pinned elsewhere and
slow to compile): layer 1 carries H, layer 2 carries H/2, no interlayer term; the layer
qubit comes first, as in bilayer_hamiltonian."
function layered(H)
    ls = siteinds("Qubit", 1)[1]
    mpo = +(TB.prepend_layer_projector(H.mpo, ls, 1), TB.prepend_layer_projector(0.5 * H.mpo, ls, 2);
            cutoff = 1e-12)
    return TB.TBHamiltonian(H; sites = [ls; H.sites], mpo = mpo, layer_s = ls, aux_side = :pre, scale = 0.0)
end
XSQ(Nx) = (i, _) -> Float64(mod(i, Nx))    # square-lattice coordinates of 0-indexed site i
YSQ(Nx) = (i, _) -> Float64(div(i, Nx))

# ─────────────────────────────────────────────────────────────────────────────
# TBSystem mutators
# ─────────────────────────────────────────────────────────────────────────────

case!("tbsystem/add_onsite!/scalar") do
    H = chain(); add_onsite!(H, 0.3); H
end
case!("tbsystem/add_onsite!/f1d") do
    H = chain(); add_onsite!(H, n -> 0.1 * n^2 - 0.2); H
end
case!("tbsystem/add_onsite!/f1d_maxdim1") do
    H = chain(); add_onsite!(H, n -> 0.1 * n^2 - 0.2; maxdim = 1); H
end
case!("tbsystem/add_onsite!/f2d_hofstadter") do
    H = hofstadter(); add_onsite!(H, (ix, iy) -> 0.1 * ix - 0.2 * iy + 0.05 * ix * iy; Lx = 2); H
end
case!("tbsystem/add_onsite!/f2d_honeycomb_sublat2") do
    H = honeycomb(); add_onsite!(H, (ix, iy) -> 0.1 * ix - 0.2 * iy; Lx = 1, sublat = 2); H
end
case!("tbsystem/add_onsite!/f2d_without_Lx") do
    H = chain(); add_onsite!(H, (ix, iy) -> 0.1 * ix); H
end
case!("tbsystem/add_onsite!/honeycomb_semenoff") do
    semenoff_honeycomb()
end
case!("tbsystem/add_onsite!/honeycomb_all_sublattices") do
    H = honeycomb(); add_onsite!(H, n -> 0.1 * n + 0.05); H
end
case!("tbsystem/add_onsite!/sublat_without_sublattice") do
    H = chain(); add_onsite!(H, 0.1; sublat = 1); H
end
case!("tbsystem/add_onsite!/layer_without_layers") do
    H = chain(); add_onsite!(H, 0.1; layer = 1); H
end
case!("tbsystem/add_onsite!/layered_chain_layer1") do
    H = layered(chain(2)); add_onsite!(H, 0.2; layer = 1); H
end
case!("tbsystem/add_onsite!/layered_chain_all_layers") do
    H = layered(chain(2)); add_onsite!(H, n -> 0.1 * n); H
end
case!("tbsystem/add_onsite!/layered_chain_bad_layer") do
    H = layered(chain(2)); add_onsite!(H, 0.2; layer = 3); H
end
case!("tbsystem/add_onsite!/layered_honeycomb_layer2_sublat1") do
    H = layered(honeycomb()); add_onsite!(H, 0.3; layer = 2, sublat = 1); H
end
case!("tbsystem/add_onsite!/layered_spinful") do
    H = layered(spinful_chain()); add_onsite!(H, 0.2; layer = 1); H
end
case!("tbsystem/add_onsite!/unsupported_signature") do
    H = chain(); add_onsite!(H, (a, b, c) -> 0.0); H
end
case!("tbsystem/add_onsite!/spinful") do
    H = spinful_chain(); add_onsite!(H, 0.1); H
end

case!("tbsystem/add_hopping!/scalar_nn1") do
    H = chain(); add_hopping!(H, 0.3); H
end
case!("tbsystem/add_hopping!/scalar_nn2") do
    H = chain(); add_hopping!(H, 0.2; nn = 2); H
end
case!("tbsystem/add_hopping!/scalar_nn3_periodic") do
    H = chain(); add_hopping!(H, 0.2; nn = 3, boundary = :periodic); H
end
case!("tbsystem/add_hopping!/bc_keyword_periodic") do
    H = chain(); add_hopping!(H, 0.2; bc = "periodic"); H
end
case!("tbsystem/add_hopping!/bad_boundary") do
    H = chain(); add_hopping!(H, 0.2; boundary = :twisted); H
end
case!("tbsystem/add_hopping!/complex_scalar") do
    H = chain(); add_hopping!(H, 0.2 + 0.1im); H
end
case!("tbsystem/add_hopping!/site_amplitude_f1") do
    H = chain(); add_hopping!(H, i -> 0.1 * i); H
end
case!("tbsystem/add_hopping!/matrix_f2_qtci") do
    H = chain(); add_hopping!(H, (i, j) -> abs(i - j) == 2 ? 0.15 : 0.0); H
end
case!("tbsystem/add_hopping!/ssh_intra_sublattice") do
    H = ssh(); add_hopping!(H, -0.1; sublat = 1, nn = 1); H
end
case!("tbsystem/add_hopping!/ssh_inter_sublattice_nn1") do
    H = ssh(); add_hopping!(H, 0.2; sublat_from = 1, sublat_to = 2, nn = 1); H
end
case!("tbsystem/add_hopping!/ssh_inter_sublattice_nn0") do
    H = ssh(); add_hopping!(H, 0.2im; sublat_from = 2, sublat_to = 1, nn = 0); H
end
case!("tbsystem/add_hopping!/ssh_inter_sublattice_nn2_periodic") do
    H = ssh(); add_hopping!(H, 0.1; sublat_from = 1, sublat_to = 2, nn = 2, boundary = :periodic); H
end
case!("tbsystem/add_hopping!/sublat_from_without_to") do
    H = ssh(); add_hopping!(H, 0.2; sublat_from = 1); H
end
case!("tbsystem/add_hopping!/inter_sublattice_function") do
    H = ssh(); add_hopping!(H, i -> 0.1; sublat_from = 1, sublat_to = 2); H
end
case!("tbsystem/add_hopping!/sublat_on_plain_chain") do
    H = chain(); add_hopping!(H, 0.1; sublat = 1); H
end
case!("tbsystem/add_hopping!/sublattice_without_keywords") do
    H = ssh(); add_hopping!(H, 0.1); H
end
case!("tbsystem/add_hopping!/after_add_spin") do
    H = spinful_chain(); add_hopping!(H, 0.1); H
end
case!("tbsystem/add_hopping!/sublat_after_add_spin") do
    H = ssh(); add_spin!(H); add_hopping!(H, 0.1; sublat = 1); H
end
case!("tbsystem/add_hopping!/honeycomb_2d_nn2") do
    H = honeycomb(); add_hopping!(H, 0.05; nn = 2); H
end
case!("tbsystem/add_hopping!/2d_sublat_keyword") do
    H = honeycomb(); add_hopping!(H, 0.1; sublat = 1); H
end
case!("tbsystem/add_hopping!/layered_2d_without_geometry") do
    H = TB.TBHamiltonian(layered(chain(2)); Lx = 1, geometry = nothing); add_hopping!(H, 0.1); H
end

case!("tbsystem/add_spin!/pre") do
    H = chain(2); add_spin!(H); H
end
case!("tbsystem/add_spin!/post") do
    H = chain(2); add_spin!(H; position = :post); H
end
case!("tbsystem/add_spin!/twice_is_noop") do
    H = chain(2); add_spin!(H); mpo = H.mpo; s = H.spin_s
    add_spin!(H; position = :post)
    (; same_mpo = H.mpo === mpo, same_spin = H.spin_s === s, H = hsum(H))
end
case!("tbsystem/add_spin!/ssh_sublattice") do
    H = ssh(; L = 2); add_spin!(H); H
end

case!("tbsystem/add_zeeman!/z_scalar_autospin") do
    H = chain(2); add_zeeman!(H, 0.3); H
end
case!("tbsystem/add_zeeman!/x_scalar") do
    H = chain(2); add_zeeman!(H, 0.3; direction = :x); H
end
case!("tbsystem/add_zeeman!/y_scalar") do
    H = chain(2); add_zeeman!(H, 0.3; direction = :y); H
end
case!("tbsystem/add_zeeman!/z_function") do
    H = chain(2); add_zeeman!(H, i -> 0.1 * i); H
end
case!("tbsystem/add_zeeman!/post_position") do
    H = chain(2); add_zeeman!(H, 0.2; position = :post); H
end
case!("tbsystem/add_zeeman!/after_bdg") do
    H = spinful_chain(); add_superconductivity!(H, 0.2); add_zeeman!(H, 0.1); H
end
case!("tbsystem/add_zeeman!/bad_direction") do
    H = chain(2); add_zeeman!(H, 0.3; direction = :w); H
end

case!("tbsystem/add_soc!/rashba") do
    H = chain(2); add_soc!(H, 0.1); H
end
case!("tbsystem/add_soc!/rashba_post") do
    H = chain(2); add_soc!(H, 0.1; position = :post); H
end
case!("tbsystem/add_soc!/ising_scalar") do
    H = chain(2); add_soc!(H, 0.2; type = :ising); H
end
case!("tbsystem/add_soc!/ising_function") do
    H = chain(2); add_soc!(H, i -> 0.05 * i; type = :ising); H
end
case!("tbsystem/add_soc!/custom_scalar_x") do
    H = chain(2); add_soc!(H, 0.1; type = :custom, direction = :x); H
end
case!("tbsystem/add_soc!/custom_f1_y") do
    H = chain(2); add_soc!(H, i -> 0.1 * i; type = :custom, direction = :y); H
end
case!("tbsystem/add_soc!/custom_f2_z_qtci") do
    H = chain(2)
    add_soc!(H, (i, j) -> i == j ? 0.1 : (abs(i - j) == 1 ? 0.05 : 0.0); type = :custom); H
end
case!("tbsystem/add_soc!/rashba_vector") do
    H = chain(2); add_soc!(H, [0.1]); H
end
case!("tbsystem/add_soc!/custom_bad_direction") do
    H = chain(2); add_soc!(H, 0.1; type = :custom, direction = :w); H
end
case!("tbsystem/add_soc!/custom_bad_lambda") do
    H = chain(2); add_soc!(H, "0.1"; type = :custom); H
end
case!("tbsystem/add_soc!/unknown_type") do
    H = chain(2); add_soc!(H, 0.1; type = :dresselhaus); H
end

case!("tbsystem/add_interaction!/mpo") do
    H = chain(); V = 0.5 * MPO(H.sites, "Id"); add_interaction!(H, V)
    (; same = H.interaction_mpo === V, H = hsum(H))
end
case!("tbsystem/add_interaction!/number") do
    H = chain(); add_interaction!(H, 2.0); H
end
case!("tbsystem/add_interaction!/f1") do
    H = chain(); add_interaction!(H, n -> 0.1 * n); H
end
case!("tbsystem/add_interaction!/f2_qtci") do
    H = chain(); add_interaction!(H, (i, j) -> 1.0 / (1 + abs(i - j))); H
end
case!("tbsystem/add_interaction!/fock_channel") do
    H = chain(); add_interaction!(H, 0.7; channel = :fock); H
end
case!("tbsystem/add_interaction!/exchange_and_default_channels") do
    H = chain(); add_interaction!(H, 0.7; channel = :Exchange); add_interaction!(H, 0.4; channel = :default); H
end
case!("tbsystem/add_interaction!/spinful_number") do
    H = spinful_chain(); add_interaction!(H, 1.0); H
end
case!("tbsystem/add_interaction!/bad_type") do
    H = chain(); add_interaction!(H, "V"); H
end
case!("tbsystem/add_interaction!/bad_channel") do
    H = chain(); add_interaction!(H, 1.0; channel = :pairing); H
end

case!("tbsystem/add_superconductivity!/spinless_swave_redirects_to_pwave") do
    H = chain(); add_superconductivity!(H, 0.2); H
end
case!("tbsystem/add_superconductivity!/spinless_pwave") do
    H = chain(2); add_superconductivity!(H, 0.2; type = :pwave); H
end
case!("tbsystem/add_superconductivity!/spinless_pwave_post") do
    H = chain(2); add_superconductivity!(H, 0.2; type = :pwave, position = :post); H
end
case!("tbsystem/add_superconductivity!/spinless_swave_function") do
    H = chain(2); add_superconductivity!(H, i -> 0.1); H
end
case!("tbsystem/add_superconductivity!/spinful_swave_number") do
    H = spinful_chain(); add_superconductivity!(H, 0.3); H
end
case!("tbsystem/add_superconductivity!/spinful_swave_function") do
    H = spinful_chain(); add_superconductivity!(H, i -> 0.1 * i); H
end
case!("tbsystem/add_superconductivity!/spinful_post_swave") do
    H = chain(2); add_spin!(H; position = :post); add_superconductivity!(H, 0.3); H
end
case!("tbsystem/add_superconductivity!/custom_qtci") do
    H = chain(2)
    add_superconductivity!(H, (i, j) -> i < j ? 0.1 : (i > j ? -0.1 : 0.0); type = :custom); H
end
case!("tbsystem/add_superconductivity!/errors") do
    (; spinful_pwave = attempt(() -> add_superconductivity!(spinful_chain(), 0.1; type = :pwave)),
       twice = attempt(() -> (H = chain(2); add_superconductivity!(H, 0.1); add_superconductivity!(H, 0.1))),
       unknown_type = attempt(() -> add_superconductivity!(spinful_chain(), 0.1; type = :dwave)),
       swave_bad_delta = attempt(() -> add_superconductivity!(spinful_chain(), [0.1])),
       pwave_function = attempt(() -> add_superconductivity!(chain(2), i -> 0.1; type = :pwave)),
       custom_number = attempt(() -> add_superconductivity!(chain(2), 0.1; type = :custom)))
end

# ─────────────────────────────────────────────────────────────────────────────
# Supercond.jl
# ─────────────────────────────────────────────────────────────────────────────

case!("supercond/indices_and_op_tables") do
    s, n = TB.spin_index(), TB.nambu_index()
    table(d) = [(k, d[k]) for k in sort!(collect(keys(d)))]
    (; spin = (dim(s), string(tags(s))), nambu = (dim(n), string(tags(n))),
       spin_ops = table(TB._SPIN_OPS), nambu_ops = table(TB._NAMBU_OPS))
end
# base operator on 2 qubits with distinct entries, so every placement is visible
base2() = (s = siteinds("Qubit", 2); (s, TB.get_diagonal_mpo(2, s, x -> 0.5 * x) + TB.kinetic_1d_nn(2, s)))
case!("supercond/prepend_op_spin_symbols") do
    s, B = base2(); sp = TB.spin_index()
    [(op, dense(TB.prepend_op(B, sp, op), [sp; s])) for op in sort!(collect(keys(TB._SPIN_OPS)))]
end
case!("supercond/postpend_op_nambu_symbols") do
    s, B = base2(); nb = TB.nambu_index()
    [(op, dense(TB.postpend_op(B, nb, op), [s; nb])) for op in sort!(collect(keys(TB._NAMBU_OPS)))]
end
case!("supercond/spin_nambu_wrappers") do
    s, B = base2(); sp = TB.spin_index(); nb = TB.nambu_index()
    M = ComplexF64[0.3 0.1im; -0.1im 0.7]
    (; prepend_spin_sym = dense(TB.prepend_spin(B, sp, :Sy), [sp; s]),
       prepend_spin_mat = dense(TB.prepend_spin(B, sp, M), [sp; s]),
       postpend_spin_sym = dense(TB.postpend_spin(B, sp, :Sp), [s; sp]),
       postpend_spin_mat = dense(TB.postpend_spin(B, sp, M), [s; sp]),
       prepend_nambu_sym = dense(TB.prepend_nambu(B, nb, :tp), [nb; s]),
       prepend_nambu_mat = dense(TB.prepend_nambu(B, nb, M), [nb; s]),
       postpend_nambu_sym = dense(TB.postpend_nambu(B, nb, :ty), [s; nb]),
       postpend_nambu_mat = dense(TB.postpend_nambu(B, nb, M), [s; nb]),
       nambu_then_spin = dense(TB.prepend_nambu(TB.prepend_spin(B, sp, :iSy), nb, :tm), [nb; sp; s]))
end
case!("supercond/op_errors") do
    s, B = base2(); plain = Index(2, "Orbital")
    (; unknown_spin = attempt(() -> TB.prepend_op(B, TB.spin_index(), :Sq)),
       unknown_nambu = attempt(() -> TB.prepend_op(B, TB.nambu_index(), :tq)),
       untagged = attempt(() -> TB.prepend_op(B, plain, :Id)),
       post_unknown_spin = attempt(() -> TB.postpend_op(B, TB.spin_index(), :Sq)),
       post_unknown_nambu = attempt(() -> TB.postpend_op(B, TB.nambu_index(), :tq)),
       post_untagged = attempt(() -> TB.postpend_op(B, plain, :Id)))
end
case!("supercond/pairingNNN") do
    s = siteinds("Qubit", 3)
    D = TB.get_diagonal_mpo(3, s, x -> 0.1 * x)
    (; uniform_nn1 = dense(TB.pairingNNN(3, s, 0.2 * MPO(s, "Id"), 1), s),
       uniform_nn2 = dense(TB.pairingNNN(3, s, 0.2 * MPO(s, "Id"), 2), s),
       profile_nn1 = dense(TB.pairingNNN(3, s, D, 1), s),
       nn0 = attempt(() -> TB.pairingNNN(3, s, D, 0)))
end
case!("supercond/pairing2MPO_qtci") do
    s = siteinds("Qubit", 2)
    dense(TB.pairing2MPO((i, j) -> i < j ? 0.1 * (j - i) : (i > j ? -0.1 * (i - j) : 0.0), 4, s), s)
end
case!("supercond/spin_hamiltonian") do
    s = siteinds("Qubit", 2); sp = TB.spin_index()
    Hu = TB.kinetic_1d_nn(2, s); Hd = 0.5 * TB.kinetic_1d_nn(2, s)
    Z = TB.get_diagonal_mpo(2, s, x -> 0.1 * x)
    (; plain = dense(TB.spin_hamiltonian(Hu, Hd, sp), [sp; s]),
       zeeman = dense(TB.spin_hamiltonian(Hu, Hd, sp; H_Zeeman = Z), [sp; s]))
end
case!("supercond/bdg_hamiltonian") do
    s = siteinds("Qubit", 2); nb = TB.nambu_index()
    Hk = TB.kinetic_1d_nn(2, s) + (-0.2) * MPO(s, "Id")
    P = TB.pairingNNN(2, s, (0.3 + 0.1im) * MPO(s, "Id"), 1)
    dense(TB.bdg_hamiltonian(Hk, P, nb), [nb; s])
end
case!("supercond/bdg_spin_hamiltonian") do
    s = siteinds("Qubit", 2); sp = TB.spin_index(); nb = TB.nambu_index()
    Hu = TB.kinetic_1d_nn(2, s); Hd = TB.kinetic_1d_nn(2, s) + 0.1 * MPO(s, "Id")
    P = TB.get_diagonal_mpo(2, s, x -> 0.2 + 0.05 * x)
    soc = TB.get_diagonal_mpo(2, s, x -> 0.03 * x)
    (; plain = dense(TB.bdg_spin_hamiltonian(Hu, Hd, P, sp, nb), [nb; sp; s]),
       soc = dense(TB.bdg_spin_hamiltonian(Hu, Hd, P, sp, nb; H_soc = soc), [nb; sp; s]))
end

# ─────────────────────────────────────────────────────────────────────────────
# Purification.jl
# ─────────────────────────────────────────────────────────────────────────────

rho0(H = chain()) = TB.purification_initial_guess(H.mpo, 2.5, H.sites)
case!("purification/helpers") do
    H = chain(); r0 = rho0(H)
    r2 = TB._mpo_sq(r0; maxdim = 20, cutoff = 1e-10)
    (; rho0 = dense(r0, H.sites), rho0_sq = dense(r2, H.sites),
       idempotency_error = TB._idempotency_error(r0, r2),
       guess_maxdim1 = dense(TB.purification_initial_guess(H.mpo, 3.0, H.sites; maxdim = 1), H.sites))
end
case!("purification/mcweeny_mpo") do
    dmsum(TB.mcweeny_purify(rho0(); maxiters = 8, maxdim = 20, cutoff = 1e-10, tol = 1e-8))
end
case!("purification/mcweeny_mpo_defaults") do
    dmsum(TB.mcweeny_purify(rho0()))
end
case!("purification/sp2_mpo_Nel4") do
    dmsum(TB.sp2_purify(rho0(), 4; maxiters = 12, maxdim = 20))
end
case!("purification/sp2_mpo_Nel3_defaults") do
    dmsum(TB.sp2_purify(rho0(), 3))
end
case!("purification/initial_guess_tb_center") do
    H = chain(); H.center = 0.1
    dense(TB.purification_initial_guess(H; ϵF = 0.2), H.sites)
end
case!("purification/mcweeny_tb") do
    H = chain(); add_onsite!(H, n -> 0.05 * n); H.scale = 2.8
    r = TB.mcweeny_purify(H; ϵF = 0.3, maxiters = 10, maxdim = 30)
    (; dm = dmsum(r), cached = H._density_cache === r)
end
case!("purification/sp2_tb") do
    H = chain(); add_onsite!(H, n -> 0.05 * n); H.scale = 2.8
    r = TB.sp2_purify(H; Nel = 3, maxdim = 30)
    (; dm = dmsum(r), cached = H._density_cache === r)
end
case!("purification/get_density/mcweeny") do
    H = chain()
    dmsum(get_density(H; method = :mcweeny, ϵF = -0.5, maxiters = 12, maxdim = 30, tol = 1e-6))
end
case!("purification/get_density/sp2") do
    H = chain()
    dmsum(get_density(H; method = :sp2, Nel = 5, maxiters = 20, maxdim = 30))
end
case!("purification/get_density/kpm_center") do
    H = chain(); H.center = 0.1
    r = get_density(H; method = :kpm, ϵF = 0.2, Ncheb = 30, maxdim = 30)
    (; dm = dmsum(r), tn_Ncheb = H._tn_Ncheb, cached = H._density_cache === r)
end
case!("purification/get_density/kpm_lorentz") do
    dmsum(get_density(chain(); method = :kpm, Ncheb = 24, kernel = :lorentz, lambda = 3.0, maxdim = 30))
end
case!("purification/get_density/kpm_other_kernels") do
    k(kernel) = attempt(() -> dmsum(get_density(chain(); method = :kpm, Ncheb = 16, kernel = kernel, maxdim = 30)))
    (; fejer = k(:fejer), dirichlet = k(:dirichlet), hodc = k(:hodc))
end
case!("purification/get_density/kpm_reuses_longer_tn_cache") do
    H = chain(); KPM_Tn(H, 36; maxdim = 30)
    (; dm = dmsum(get_density(H; method = :kpm, Ncheb = 24, maxdim = 30)), tn_Ncheb = H._tn_Ncheb)
end
case!("purification/get_density/kpm_rebuilds_shorter_tn_cache") do
    H = chain(); KPM_Tn(H, 12; maxdim = 30)
    (; dm = dmsum(get_density(H; method = :kpm, Ncheb = 24, maxdim = 30)), tn_Ncheb = H._tn_Ncheb)
end
# A density set by hand answers every method (an unknown method is an error even with a
# cache: "unknown_method" below).
case!("purification/get_density/returns_cache") do
    H = chain(); sentinel = 0.25 * MPO(H.sites, "Id"); H._density_cache = sentinel
    (; kpm = get_density(H; method = :kpm) === sentinel,
       sp2 = get_density(H; method = :sp2) === sentinel)
end
case!("purification/get_density/unknown_method") do
    get_density(chain(); method = :nonsense)
end
case!("purification/sign_mpo") do
    H = chain()
    A = (1 / 2.5) * H.mpo + 0.05 * MPO(H.sites, "Id")
    S = TB.sign_mpo(A, H.sites; maxdim = 30, maxiters = 12)
    M = dense(S, H.sites)
    (; S = M, square_minus_id = norm(M * M - I))
end
case!("purification/sign_mpo_scale") do
    H = chain()
    dense(TB.sign_mpo(H.mpo, H.sites; scale = 2.5, maxdim = 30, maxiters = 6, tol = 1e-3), H.sites)
end
case!("purification/ldos_drho_mpo") do
    dense(TB.get_ldos_drho(chain(), 0.35; dmu = 0.1, maxiters = 10, maxdim = 30))
end
case!("purification/ldos_drho_mps") do
    densev(TB.get_ldos_drho(chain(), 0.35; mode = :mps, dmu = 0.1, maxiters = 10, maxdim = 30))
end
case!("purification/ldos_drho_vector") do
    [densev(m) for m in TB.get_ldos_drho(chain(), [-1.0, 0.35]; mode = :mps, maxiters = 10, maxdim = 30)]
end
case!("purification/ldos_drho_bad_mode") do
    TB.get_ldos_drho(chain(), 0.0; mode = :diag)
end
case!("purification/dos_drho") do
    H = chain()
    (; scalar = TB.get_dos_drho(H, 0.35; dmu = 0.1, maxiters = 10, maxdim = 30),
       vector = TB.get_dos_drho(H, [-1.0, 0.0, 1.0]; maxiters = 10, maxdim = 30))
end

# ─────────────────────────────────────────────────────────────────────────────
# Topology.jl
# ─────────────────────────────────────────────────────────────────────────────

gapped_chain() = (H = chain(); add_onsite!(H, n -> 0.4 * (-1)^n); H.scale = 2.8; H)
case!("topology/_get_projector/KPM_fresh") do
    H = gapped_chain()
    (; P = dense(TB._get_projector(H; method = :kpm, fermi = 0.3, Ncheb = 30, maxdim = 30)),
       cached_tn = H._tn_cache !== nothing, tn_Ncheb = H._tn_Ncheb)
end
case!("topology/_get_projector/KPM_rebuilds_short_tn_cache") do
    H = gapped_chain(); KPM_Tn(H, 20; maxdim = 30)
    dense(TB._get_projector(H; method = :kpm, Ncheb = 300, maxdim = 30))
end
case!("topology/_get_projector/mcweeny") do
    H = gapped_chain()
    (; P = dense(TB._get_projector(H; method = :mcweeny, fermi = -0.2, maxdim = 30)),
       cached = H._density_cache !== nothing)
end
case!("topology/_get_projector/cache_short_circuits") do
    H = gapped_chain(); sentinel = 0.5 * MPO(H.sites, "Id"); H._density_cache = sentinel
    (; mcweeny = TB._get_projector(H; method = :mcweeny) === sentinel,
       sp2 = TB._get_projector(H; method = :sp2) === sentinel)
end
case!("topology/_get_projector/sp2_Nel") do
    # up to v0.1.1 SP2 stalled just above tol, then diverged (to NaN for some fillings)
    (; Nel3 = attempt(() -> dense(TB._get_projector(gapped_chain(); method = :sp2, Nel = 3.0, maxdim = 30))),
       default = attempt(() -> dense(TB._get_projector(gapped_chain(); method = :sp2, maxdim = 30))))
end
case!("topology/_get_projector/errors") do
    # `:kpm` was an error here up to v0.1.1 (it is the canonical spelling since 0.2)
    (; unknown = attempt(() -> TB._get_projector(gapped_chain(); method = :exact)))
end

winding(W, H) = [W(uc) for uc in 1:H.N]
case!("topology/get_W/KPM_quenched") do
    H = ssh(); winding(winding_marker(H; method = :kpm, Ncheb = 40, maxdim = 30), H)
end
case!("topology/get_W/KPM_flat") do
    H = ssh(); winding(winding_marker(H; method = :kpm, Ncheb = 40, maxdim = 30, quenched = false), H)
end
case!("topology/get_W/mcweeny_explicit_xfunc") do
    H = ssh(); winding(winding_marker(H, (i, _) -> Float64(i ÷ 2); method = :mcweeny, maxdim = 30), H)
end
case!("topology/get_W/mcweeny_l_Lambda") do
    H = ssh(); winding(winding_marker(H; method = :mcweeny, l = 2, Lambda = 4, maxdim = 30), H)
end
case!("topology/get_W/sp2_half_filling") do       # SP2 diverged to NaN here up to v0.1.1
    H = ssh(); winding(winding_marker(H; method = :sp2, Nel = 8, maxdim = 30), H)
end
case!("topology/get_W/sp2_default_Nel") do     # default Nel: half the states (H.N ÷ 2 up to v0.1.1)
    H = ssh(); winding(winding_marker(H; method = :sp2, maxdim = 30), H)
end
case!("topology/get_W/trivial_mcweeny_flat") do
    H = ssh(; d = 0.3); winding(winding_marker(H; method = :mcweeny, quenched = false, maxdim = 30), H)
end
case!("topology/get_W/errors") do
    Hng = ssh(); Hng.geometry = nothing; Hng.geometry_uc = nothing
    (; no_sublattice = attempt(() -> winding_marker(chain())),
       three_sublattices = attempt(() -> winding_marker(kagome())),
       no_geometry = attempt(() -> winding_marker(Hng; method = :mcweeny)))
end

case!("topology/position_operators") do
    s = siteinds("Qubit", 2); xf = XSQ(2); yf = YSQ(2)
    (; sinx = dense(TB.get_sinx_op(2, s, 2, 3.0, xf), s), cosx = dense(TB.get_cosx_op(2, s, 2, 3.0, xf), s),
       siny = dense(TB.get_siny_op(2, s, 2, 3.0, yf), s), cosy = dense(TB.get_cosy_op(2, s, 2, 3.0, yf), s))
end

chern_P() = (H = hofstadter(); (H, TB.mcweeny_purify(H; maxdim = 40)))
marker(C, n) = [C(a) for a in 1:n]
case!("topology/get_C_op_MPO_from_P/plain_quenched") do
    H, P = chern_P()
    marker(TB.get_C_op_MPO_from_P(P, H.L, H.sites, XSQ(4), YSQ(4); maxdim = 40), H.N)
end
case!("topology/get_C_op_MPO_from_P/plain_flat") do
    H, P = chern_P()
    marker(TB.get_C_op_MPO_from_P(P, H.L, H.sites, XSQ(4), YSQ(4); maxdim = 40, quenched = false), H.N)
end
case!("topology/get_C_op_MPO_from_P/plain_sequential_l1") do
    H, P = chern_P()
    marker(TB.get_C_op_MPO_from_P(P, H.L, H.sites, XSQ(4), YSQ(4); maxdim = 40, l = 1, Lambda = 3,
                                  sequential = true), H.N)
end
case!("topology/get_C_op_MPO_from_P/pk_mpo") do
    H, P = chern_P(); pk = 0.5 * MPO(H.sites, "Id") + 0.5 * P
    (; quenched = marker(TB.get_C_op_MPO_from_P(P, H.L, H.sites, XSQ(4), YSQ(4); maxdim = 40, pk_mpo = pk), H.N),
       flat = marker(TB.get_C_op_MPO_from_P(P, H.L, H.sites, XSQ(4), YSQ(4); maxdim = 40, pk_mpo = pk,
                                            quenched = false), H.N),
       sequential = marker(TB.get_C_op_MPO_from_P(P, H.L, H.sites, XSQ(4), YSQ(4); maxdim = 40, pk_mpo = pk,
                                                  sequential = true), H.N))
end
case!("topology/get_C/hofstadter_KPM") do
    H = hofstadter(); marker(chern_marker(H, XSQ(4), YSQ(4); method = :kpm, Ncheb = 40, maxdim = 40), H.N)
end
case!("topology/get_C/hofstadter_mcweeny") do
    H = hofstadter(); marker(chern_marker(H, XSQ(4), YSQ(4); method = :mcweeny, maxdim = 40), H.N)
end
case!("topology/get_C/hofstadter_sp2_flat") do
    H = hofstadter(); marker(chern_marker(H, XSQ(4), YSQ(4); method = :sp2, Nel = 8, maxdim = 40, quenched = false), H.N)
end
case!("topology/get_C/hofstadter_mcweeny_sequential_Lambda") do
    # passing the deprecated Λ next to Lambda is an error since 0.2 (Lambda won up to v0.1.1)
    H = hofstadter()
    (; Lambda = marker(chern_marker(H, XSQ(4), YSQ(4); method = :mcweeny, maxdim = 40, Lambda = 2.5,
                                    sequential = true), H.N),
       both = attempt(() -> chern_marker(H, XSQ(4), YSQ(4); method = :mcweeny, maxdim = 40, Λ = 7,
                                         Lambda = 2.5, sequential = true)))
end
case!("topology/get_C/hofstadter_auto_geometry_staggered") do
    H = hofstadter(); add_onsite!(H, (ix, iy) -> 0.3 * (-1)^(ix + iy); Lx = 2); H.scale = 3.2
    marker(chern_marker(H; method = :mcweeny, maxdim = 40), H.N)
end
case!("topology/get_C/honeycomb_semenoff_auto_geometry_uc") do
    H = semenoff_honeycomb(); marker(chern_marker(H; method = :mcweeny, maxdim = 40), H.N)
end
case!("topology/get_C/errors") do
    # `:kpm` was an error here up to v0.1.1 (it is the canonical spelling since 0.2)
    (; no_geometry = attempt(() -> chern_marker(TB.TBHamiltonian(hofstadter(); geometry = nothing); method = :mcweeny)))
end

case!("topology/valley/operator_and_projectors") do
    H = honeycomb(); V = TB.get_valley_operator(H; maxdim = 40)
    PK, PKp = TB.get_valley_projectors(V, H.sites; maxdim = 40)
    (; V = dense(V, H.sites), PK = dense(PK, H.sites), PK_prime = dense(PKp, H.sites))
end
case!("topology/valley/operator_errors") do
    (; no_Lx = attempt(() -> TB.get_valley_operator(ssh())),
       no_sublattice = attempt(() -> TB.get_valley_operator(TB.TBHamiltonian(hofstadter(); Lx = 2))),
       three_sublattices = attempt(() -> TB.get_valley_operator(kagome())))
end
case!("topology/valley/C_K") do
    H = semenoff_honeycomb()
    (; quenched = marker(TB.valley_chern_marker(H; valley = :K, maxdim = 40), H.N),
       sequential = marker(TB.valley_chern_marker(semenoff_honeycomb(); valley = :K, maxdim = 40, sequential = true), H.N))
end
case!("topology/valley/C_K_prime_nosign_flat") do
    H = semenoff_honeycomb()
    marker(TB.valley_chern_marker(H; valley = :K_prime, use_sign = false, quenched = false, maxdim = 40), H.N)
end
case!("topology/valley/C_bad_valley") do
    TB.valley_chern_marker(semenoff_honeycomb(); valley = :Gamma)
end

xpump(i, N) = Float64(i + 1)
pump_H(base, t) = (H = deepcopy(base); add_onsite!(H, n -> 0.5 * cos(2π * (t + n / 3))); H.scale = 3.0; H)
case!("topology/pump/xop") do
    s = siteinds("Qubit", 3)
    (; flat = dense(TB.get_pump_xop(3, s, xpump), s),
       quenched = dense(TB.get_pump_xop(3, s, xpump; quenched = true), s),
       quenched_L5 = dense(TB.get_pump_xop(3, s, xpump; quenched = true, Lambda = 5.0), s))
end
case!("topology/pump/thouless_pump") do
    base = chain()
    Ps = [TB.mcweeny_purify(pump_H(base, t); maxdim = 30) for t in (0.0, 1 / 3, 2 / 3)]
    x = TB.get_pump_xop(3, base.sites, xpump)
    C, traj = TB.thouless_pump(Ps, 1 / 3, x, base.sites; r_center = 4, maxdim = 30, return_trajectory = true)
    (; C = C, trajectory = traj,
       C_only = TB.thouless_pump(Ps, 1 / 3, x, base.sites; r_center = 3, maxdim = 30))
end
case!("topology/pump/get_thouless_pump") do
    base = chain()
    (; flat = TB.get_thouless_pump(t -> pump_H(base, t), 3, 1.0, xpump; maxdim = 30),
       quenched_r2 = TB.get_thouless_pump(t -> pump_H(base, t), 3, 1.0, xpump; quenched = true,
                                          Lambda = 5.0, r_center = 2, maxdim = 30),
       sp2 = attempt(() -> TB.get_thouless_pump(t -> pump_H(base, t), 3, 1.0, xpump; P_method = :sp2,
                                                maxdim = 30)))
end

# ─────────────────────────────────────────────────────────────────────────────
# SCF.jl — helpers
# ─────────────────────────────────────────────────────────────────────────────

profile(H = chain()) = TB.extract_diagonal_to_mps(rho0(H))       # diag of (I - H/2.5)/2 + a tilt
tilted(H = chain()) = TB.get_mps(H.L, H.sites, n -> 0.5 + 0.05 * n - 0.01 * n^2)
case!("scf/density_profile_from_dm") do
    H = chain(); r = rho0(H)
    (; direct = densev(TB.density_profile_from_dm(r, H.sites)),
       complement = densev(TB.density_profile_from_dm(r, H.sites; mode = :complement)),
       bad = attempt(() -> TB.density_profile_from_dm(r, H.sites; mode = :sum)))
end
case!("scf/_subtract_background") do
    H = chain(); rho = tilted(H); kw = (maxdim = 20, cutoff = 1e-10)
    (; none = densev(TB._subtract_background(rho, H.sites, nothing; kw...)),
       number = densev(TB._subtract_background(rho, H.sites, 0.45; kw...)),
       mps = densev(TB._subtract_background(rho, H.sites, TB.constant_mps(collect(H.sites), 0.2); kw...)),
       bad = attempt(() -> TB._subtract_background(rho, H.sites, "0.5"; kw...)))
end
case!("scf/hartree_and_local_hartree") do
    H = chain(); rho = tilted(H); V = TB.pair_distance_interaction_mpo(3, H.sites, 1, 0.7)
    (; hartree = dense(TB.hartree_mpo_from_density(rho, V, H.sites), H.sites),
       hartree_bg = dense(TB.hartree_mpo_from_density(rho, V, H.sites; background = 0.5, maxdim = 4), H.sites),
       local_hartree = dense(TB._local_hartree_from_density(rho, H.sites, 1.5, 0.5; maxdim = 20, cutoff = 1e-10), H.sites))
end
case!("scf/fock") do
    H = chain(); r = rho0(H); V = TB.pair_distance_interaction_mpo(3, H.sites, 1, 0.7) + 0.3 * MPO(H.sites, "Id")
    (; minus = dense(TB.fock_mpo_from_density(r, V, H.sites), H.sites),
       plus = dense(TB.fock_mpo_from_density(r, V, H.sites; sign = 1), H.sites),
       half = dense(TB.fock_mpo_from_density(r, V, H.sites; sign = 0.5), H.sites),
       builder = dense(TB.fock_exchange_builder(V; sign = -2)(r, H.sites), H.sites))
end
case!("scf/hartree_builders") do
    H = chain(); rho = tilted(H); V = TB.pair_distance_interaction_mpo(3, H.sites, 2, 0.4)
    (; cdw = dense(TB.cdw_hartree_builder(1.2)(rho, H.sites), H.sites),
       cdw_bg = dense(TB.cdw_hartree_builder(1.2; background = 0.3)(rho, H.sites), H.sites),
       dense_mpo = dense(TB.dense_hartree_builder(V, 3, H.sites; background = 0.5)(rho, H.sites), H.sites),
       dense_qtci = dense(TB.dense_hartree_builder((i, j) -> 0.5 * exp(-abs(i - j)), 3, H.sites)(rho, H.sites), H.sites),
       pair_scalar = dense(TB.pair_distance_hartree_builder(3, H.sites, 1, 0.8; background = 0.5)(rho, H.sites), H.sites),
       pair_terms = dense(TB.pair_distance_hartree_builder(3, H.sites, [1 => 0.8, (3, 0.2)])(rho, H.sites), H.sites))
end
case!("scf/_weight_to_mpo_and_pair_term") do
    s = siteinds("Qubit", 3); M = 0.3 * MPO(s, "Id")
    (; mpo_same = TB._weight_to_mpo(3, s, M) === M,
       number = dense(TB._weight_to_mpo(3, s, 0.4), s),
       func = dense(TB._weight_to_mpo(3, s, i -> 0.1 * i), s),
       bad = attempt(() -> TB._weight_to_mpo(3, s, "w")),
       pair = TB._pair_term(2 => 0.5), tuple = TB._pair_term((3.0, 0.25)),
       bad_term = attempt(() -> TB._pair_term([1, 2])))
end
case!("scf/pair_distance_interaction_mpo") do
    s = siteinds("Qubit", 3)
    (; d1 = dense(TB.pair_distance_interaction_mpo(3, s, 1), s),
       d2_function = dense(TB.pair_distance_interaction_mpo(3, s, 2, i -> 0.1 * (i + 1)), s),
       negative_distance = dense(TB.pair_distance_interaction_mpo(3, s, -1, 0.5), s),
       terms = dense(TB.pair_distance_interaction_mpo(3, s, [1 => 0.5, (2, 0.25)]), s),
       distance_zero = attempt(() -> TB.pair_distance_interaction_mpo(3, s, 0)),
       distance_N = attempt(() -> TB.pair_distance_interaction_mpo(3, s, 8)),
       empty_terms = attempt(() -> TB.pair_distance_interaction_mpo(3, s, Any[])),
       bad_term = attempt(() -> TB.pair_distance_interaction_mpo(3, s, [1])))
end
case!("scf/staggered_magnetic_initial") do
    (; spinless = map(densev, TB.staggered_magnetic_initial(chain())),
       custom = map(densev, TB.staggered_magnetic_initial(chain(); amplitude = 0.2, background = 0.4)),
       spinful = map(densev, TB.staggered_magnetic_initial(spinful_chain())))
end
case!("scf/_pairing_profile_mps") do
    H = chain(2); m = TB.constant_mps(collect(H.sites), 0.3)
    (; mps_same = TB._pairing_profile_mps(m, 2, H.sites) === m,
       number = densev(TB._pairing_profile_mps(0.2, 2, H.sites)),
       func = densev(TB._pairing_profile_mps(n -> 0.1 + 0.05im * n, 2, H.sites)),
       func_real = densev(TB._pairing_profile_mps(n -> 0.1 * n, 2, H.sites; type = Float64)),
       bad = attempt(() -> TB._pairing_profile_mps("d", 2, H.sites)))
end

bdg_H0() = spinful_chain(2; scale = 2.5)
delta_mps(H0, v = 0.3) = TB.constant_mps(collect(TB._pos_sites(H0)), v)
case!("scf/_bdg_from_pairing/spinless_input_mu") do
    H0 = chain(2); d = TB._pairing_profile_mps(n -> 0.2 + 0.1 * n, 2, H0.sites)
    (; H0_untouched = H0.spin_s === nothing, Hbdg = hsum(TB._bdg_from_pairing(H0, d; mu = 0.1)))
end
case!("scf/_bdg_from_pairing/spinful_hartree_scale_center") do
    H0 = bdg_H0(); ps = TB._pos_sites(H0)
    hu = TB._local_hartree_from_density(TB.constant_mps(collect(ps), 0.6), ps, 1.0, 0.5; maxdim = 20, cutoff = 1e-10)
    hd = TB._local_hartree_from_density(TB.constant_mps(collect(ps), 0.4), ps, 1.0, 0.5; maxdim = 20, cutoff = 1e-10)
    TB._bdg_from_pairing(H0, delta_mps(H0); hartree_up = hu, hartree_dn = hd, scale = 3.0, center = 0.1)
end
case!("scf/_bdg_from_pairing/zero_scale_stays_zero") do
    H0 = bdg_H0(); H0.scale = 0.0
    TB._bdg_from_pairing(H0, delta_mps(H0)).scale
end
case!("scf/_project_aux_block") do
    H0 = bdg_H0(); Hb = TB._bdg_from_pairing(H0, delta_mps(H0); scale = 3.0)
    ph = TB._project_aux_block(Hb.mpo, Hb.nambu_s, 1, 2; tag = "Nambu")
    (; ph = dense(ph), ph_ud = dense(TB._project_aux_block(ph, Hb.spin_s, 1, 2; tag = "Spin")),
       hh = dense(TB._project_aux_block(Hb.mpo, Hb.nambu_s, 2, 2)),
       foreign_index_by_tag = attempt(() -> dense(TB._project_aux_block(Hb.mpo, TB.nambu_index(), 2, 1;
                                                                        tag = "Nambu"))),
       not_found = attempt(() -> TB._project_aux_block(Hb.mpo, TB.spin_index(), 1, 1)))
end
case!("scf/swave_profiles") do
    H0 = bdg_H0()
    Hb = TB._bdg_from_pairing(H0, delta_mps(H0); mu = 0.2, scale = 3.0)
    r = get_density(Hb; method = :mcweeny, maxiters = 15, maxdim = 30)
    (; anomalous = densev(TB.swave_anomalous_profile(r, Hb)),
       particle = map(densev, TB.swave_normal_profiles(r, Hb)),
       hole_complement = map(densev, TB.swave_normal_profiles(r, Hb; mode = :hole_complement)),
       bad_mode = attempt(() -> TB.swave_normal_profiles(r, Hb; mode = :total)),
       not_bdg = attempt(() -> TB.swave_anomalous_profile(r, H0)),
       not_bdg_normal = attempt(() -> TB.swave_normal_profiles(r, H0)),
       spinless_bdg = attempt(() -> TB.swave_anomalous_profile(r, TB.TBHamiltonian(Hb; spin_s = nothing))))
end
case!("scf/pwave_profiles") do
    H0 = bdg_H0(); ps = collect(TB._pos_sites(H0))
    du = TB._pairing_profile_mps(n -> 0.2 + 0.05 * n, 2, ps); dd = (-1.0) * du
    Hb = TB._triplet_equalspin_bdg(H0, du, dd; distance = 1, mu = 0.1, scale = 3.0)
    r = get_density(Hb; method = :mcweeny, maxiters = 15, maxdim = 30)
    (; bond = dense(TB._pwave_bond_mpo(du, ps, 1), ps), bond_d2 = dense(TB._pwave_bond_mpo(du, ps, 2), ps),
       bond_d0 = attempt(() -> TB._pwave_bond_mpo(du, ps, 0)),
       Hbdg = hsum(Hb), anomalous = map(densev, TB.pwave_equalspin_anomalous_profiles(r, Hb)),
       anomalous_d2 = map(densev, TB.pwave_equalspin_anomalous_profiles(r, Hb; distance = 2)),
       bond_profile = densev(TB._pwave_bond_profile(TB._pwave_bond_mpo(du, ps, 1), ps, 1)),
       not_bdg = attempt(() -> TB.pwave_equalspin_anomalous_profiles(r, H0)),
       spinless_input = let H1 = chain(2), d1 = TB._pairing_profile_mps(0.2, 2, H1.sites)
           hsum(TB._triplet_equalspin_bdg(H1, d1, (-1.0) * d1; distance = 1))
       end)
end
case!("scf/channel_and_method_names") do
    chans = [:CDW, :charge, :Hartree, :magnetic, :magnetism, :spin, :Hubbard, :swave, :s_wave,
             Symbol("s-wave"), :superconducting, :superconductivity, :pwave, :p_wave, Symbol("p-wave"), :triplet]
    meths = [:purification, :SP2, :mcweeny, :McPurify, :kpm, :Chebyshev, :exact]
    (; channels = [(c, TB._canonical_channel(c)) for c in chans],
       bad_channel = attempt(() -> TB._canonical_channel(:dwave)),
       methods = [(m, TB._canonical_density_method(m)) for m in meths],
       is_purification = [(m, TB._is_purification(m)) for m in (:sp2, :mcweeny, :kpm, :SP2)])
end
case!("scf/_set_purification_scale!") do
    mk() = (H = chain(); H.scale = 2.0; H.center = 0.3; H)
    sc(H) = (H.scale, H.center)
    (; kpm = sc(TB._set_purification_scale!(mk(), :kpm; scale = 5.0)),
       bounds = sc(TB._set_purification_scale!(mk(), :sp2; spectral_bounds = (-2.0, 3.0), padding = 1.1)),
       bounds_over_scale = sc(TB._set_purification_scale!(mk(), :mcweeny; scale = 7.0, spectral_bounds = (-1, 1))),
       scale = sc(TB._set_purification_scale!(mk(), :mcweeny; scale = 2.0)),
       existing = sc(TB._set_purification_scale!(mk(), :sp2; padding = 1.2)),
       padding_below_1 = attempt(() -> TB._set_purification_scale!(mk(), :sp2; padding = 0.9)),
       bad_bounds = attempt(() -> TB._set_purification_scale!(mk(), :sp2; spectral_bounds = (1.0, 1.0))),
       kpm_bad_padding = sc(TB._set_purification_scale!(mk(), :kpm; padding = 0.5)))
end
case!("scf/_copy_with_mpo_and_split_spin_channels") do
    H = chain(); add_interaction!(H, 0.5)
    Hc = TB._copy_with_mpo(H, 2.0 * H.mpo; scale = 1.5, center = 0.2)
    a, b = TB._split_spin_channels(H)
    Hs = spinful_chain(); add_zeeman!(Hs, 0.3); Hs.scale = 2.5
    u, d = TB._split_spin_channels(Hs)
    (; copy = hsum(Hc), shares_interaction = Hc.interaction_mpo === H.interaction_mpo,
       spinless_same = a === H && b === H, up = hsum(u), dn = hsum(d))
end
case!("scf/_make_cdw_builder") do
    H = chain(); rho = tilted(H)
    b(U, mode; kw...) = attempt(() -> dense(TB._make_cdw_builder(H, U; interaction = mode, background = 0.5,
                                   distance = 2, maxdim = 30, cutoff = 1e-10, tol = 1e-8, kw...)(rho, H.sites), H.sites))
    V = TB.pair_distance_interaction_mpo(3, H.sites, 1, 0.5)
    (; local_ = b(1.0, :local), onsite = b(1.0, :onsite), on_site = b(1.0, :On_Site),
       dense_ = b(V, :dense), longrange = b(V, :longrange), long_range = b(V, :long_range),
       distance = b(0.7, :distance), pair = b(0.7, :pair), pairs_vector = b([1 => 0.3, 2 => 0.1], :pairs),
       local_mpo = b(V, :local), unknown = b(1.0, :yukawa))
end
case!("scf/initial_guesses") do
    s3 = siteinds("Qubit", 3); s2 = siteinds("Qubit", 2)
    g(t) = (; mpo = dense(t[1]), mps = densev(t[2]))
    (; trivial_up = g(TB.initial_guess_trivial_up_1D(3, s3)),
       trivial_down = g(TB.initial_guess_trivial_down_1D(3, s3)),
       neel_up = g(TB.initial_guess_Neel_up(1, 1, s2)),
       neel_dn = g(TB.initial_guess_Neel_dn(1, 1, s2)))
end

# ─────────────────────────────────────────────────────────────────────────────
# SCF.jl — drivers through get_scf and directly (1–3 iterations each)
# ─────────────────────────────────────────────────────────────────────────────

const SCFKW = (maxiters = 2, purif_maxiter = 15, purif_tol = 1e-6, maxdim = 30, cutoff = 1e-10,
               mixing = 0.4, verbose = false)

case!("scf/get_scf/cdw_local_sp2") do
    get_scf(chain(), 1.0, :cdw; scale = 3.0, SCFKW...)
end
case!("scf/get_scf/cdw_local_mcweeny_complement") do
    get_scf(chain(), 1.0, :CDW; method = :mcweeny, density_mode = :complement, background = 0.4,
            scale = 3.0, SCFKW...)
end
case!("scf/get_scf/cdw_local_kpm_chebyshev") do
    get_scf(chain(), 1.0, :charge; density_method = :chebyshev, Ncheb = 30, fermi = 0.1, scale = 3.0, SCFKW...)
end
case!("scf/get_scf/cdw_dense_qtci_purification") do
    get_scf(chain(), (i, j) -> 0.5 * exp(-abs(i - j)), :hartree; interaction = :dense,
            method = :purification, scale = 3.0, SCFKW...)
end
case!("scf/get_scf/cdw_distance_scalar") do
    get_scf(chain(), 0.8, :cdw; interaction = :distance, distance = 1, scale = 3.0, SCFKW...)
end
case!("scf/get_scf/cdw_distance_terms") do
    get_scf(chain(), [1 => 0.8, 2 => 0.3], :cdw; interaction = :pairs, scale = 3.0, SCFKW...)
end
case!("scf/get_scf/cdw_spectral_bounds_Nel3") do
    get_scf(chain(), 1.0, :cdw; spectral_bounds = (-2.5, 2.8), purification_scale_padding = 1.1, Nel = 3,
            SCFKW...)
end
case!("scf/get_scf/cdw_stop_on_increase") do
    # stops at iteration 2 (rms 0.0086 -> 0.0116) and returns the iteration-1 state
    get_scf(chain(), 4.0, :cdw; scale = 5.0, stop_on_increase = true, SCFKW..., maxiters = 3, mixing = 1.0)
end
case!("scf/get_scf/cdw_converges_first_iteration") do
    get_scf(chain(), 1.0, :cdw; scale = 3.0, SCFKW..., tol = 1e3)
end
case!("scf/get_scf/cdw_initial_density_and_hartree") do
    H = chain()
    get_scf(H, 1.0, :cdw; scale = 3.0, initial_density = TB.constant_mps(collect(H.sites), 0.3),
            initial_hartree = 0.1 * TB.get_diagonal_mpo(3, H.sites, x -> x), SCFKW...)
end
case!("scf/scf_meanfield/fock_builder") do
    H = chain(); V = TB.pair_distance_interaction_mpo(3, H.sites, 1, 0.5)
    TB.scf_meanfield(H, TB.cdw_hartree_builder(1.0); fock_builder = TB.fock_exchange_builder(V),
                     initial_fock = 0.05 * MPO(H.sites, "Id"), density_method = :mcweeny, scale = 3.0,
                     max_scf_iter = 2, purif_maxiter = 15, maxdim = 30, cutoff = 1e-10, verbose = false)
end
case!("scf/scf_meanfield/defaults_kpm") do
    TB.scf_meanfield(chain(), TB.cdw_hartree_builder(0.5); density_method = :kpm, Ncheb = 24, scale = 3.0,
                     max_scf_iter = 1, maxdim = 30, verbose = false)
end
case!("scf/get_scf/cdw_stored_interaction") do
    H = chain(); add_interaction!(H, n -> 0.8 + 0.1 * n)
    get_scf(H, :cdw; scale = 3.0, SCFKW...)
end

case!("scf/get_scf/magnetic_spinless_mcweeny") do
    get_scf(chain(), 2.0, :magnetic; method = :mcweeny, scale = 4.0, SCFKW...)
end
case!("scf/get_scf/magnetic_spinful_sp2") do
    H = chain(); add_spin!(H); H.scale = 2.5
    get_scf(H, 2.0, :hubbard; density_method = :sp2, scale = 4.0, SCFKW...)
end
case!("scf/get_scf/magnetic_kpm_fermi") do
    get_scf(chain(), 1.5, :spin; method = :kpm, Ncheb = 30, fermi = 0.2, scale = 4.0, SCFKW...)
end
case!("scf/get_scf/magnetic_initial_and_Nel") do
    H = chain()
    get_scf(H, 2.0, :magnetism; scale = 4.0, initial_up = TB.constant_mps(collect(H.sites), 0.7),
            Nel_up = 5, Nel_dn = 3, background = 0.4, SCFKW...)
end
case!("scf/get_scf/magnetic_converges_first_iteration") do
    get_scf(chain(), 2.0, :magnetic; scale = 4.0, SCFKW..., tol = 1e3)
end
case!("scf/get_scf/magnetic_stored_mpo_interaction") do
    H = chain(); add_interaction!(H, TB.pair_distance_interaction_mpo(3, H.sites, 1, 0.6) + 1.5 * MPO(H.sites, "Id"))
    get_scf(H, :magnetic; scale = 4.0, SCFKW...)
end
case!("scf/scf_magnetic_hubbard/direct_default_scale") do
    H = chain(); H.scale = 4.0
    TB.scf_magnetic_hubbard(H, 2.0; max_scf_iter = 1, purif_maxiter = 15, maxdim = 30, verbose = false)
end

case!("scf/get_scf/swave_hubbard") do
    get_scf(chain(2), 1.5, :swave; method = :mcweeny, scale = 3.5, SCFKW...)
end
case!("scf/get_scf/swave_hole_complement_mu_seed") do
    get_scf(bdg_H0(), 1.5, :superconducting; method = :mcweeny, scale = 3.5, mu = 0.2,
            normal_density_mode = :hole_complement, initial_delta = n -> 0.1 + 0.05 * n, SCFKW...)
end
case!("scf/scf_swave_superconducting/pairing_only_sp2") do
    TB.scf_swave_superconducting(bdg_H0(), 1.0; density_method = :sp2, scale = 3.5, max_scf_iter = 2,
                                 purif_maxiter = 15, maxdim = 30, cutoff = 1e-10, verbose = false)
end
case!("scf/scf_swave_superconducting/converges_first_iteration") do
    TB.scf_swave_superconducting(bdg_H0(), 1.0; scale = 3.5, max_scf_iter = 2, scf_tol = 1e3,
                                 purif_maxiter = 15, maxdim = 30, verbose = false)
end
case!("scf/scf_swave_hubbard/direct") do
    TB.scf_swave_hubbard(chain(2), 1.2; hartree_coupling = 0.6, scale = 3.5, max_scf_iter = 1,
                         purif_maxiter = 15, maxdim = 30, verbose = false)
end

case!("scf/get_scf/pwave") do
    get_scf(chain(2), 1.0, :pwave; method = :mcweeny, scale = 3.5, SCFKW...)
end
case!("scf/get_scf/pwave_eta_initial_dn") do
    get_scf(bdg_H0(), 1.0, :triplet; scale = 3.5, eta_down = 1.0, initial_up = n -> 0.1 + 0.02 * n,
            initial_dn = 0.05, mu = 0.1, SCFKW...)
end
case!("scf/scf_pwave_equalspin/converges_first_iteration") do
    TB.scf_pwave_equalspin(bdg_H0(), 1.0; scale = 3.5, max_scf_iter = 2, scf_tol = 1e3,
                           purif_maxiter = 15, maxdim = 30, verbose = false)
end

case!("scf/get_scf/errors") do
    H = chain(); Hi = chain(); add_interaction!(Hi, 1.0)
    (; stored_swave = attempt(() -> get_scf(Hi, :swave)),
       stored_pwave = attempt(() -> get_scf(Hi, :pwave)),
       no_interaction = attempt(() -> get_scf(H, :cdw)),
       unknown_channel = attempt(() -> get_scf(H, 1.0, :dwave)),
       swave_mpo = attempt(() -> get_scf(H, MPO(H.sites, "Id"), :swave)),
       pwave_mpo = attempt(() -> get_scf(H, MPO(H.sites, "Id"), :pwave)),
       cdw_local_mpo = attempt(() -> get_scf(H, MPO(H.sites, "Id"), :cdw)),
       cdw_unknown_interaction = attempt(() -> get_scf(H, 1.0, :cdw; interaction = :yukawa)),
       unknown_density_method = attempt(() -> get_scf(chain(), 1.0, :cdw; method = :exact, scale = 3.0,
                                                       SCFKW...)))
end

# ═════════════════════════════════════════════════════════════════════════════
# Comparison
# ═════════════════════════════════════════════════════════════════════════════

const Num = Union{AbstractFloat,Complex{<:AbstractFloat}}

function _report(path, msg)
    @error "Scftopo output changed" path msg
    return false
end

"`true` when `a` matches the golden value `e` (see the rules in the header)."
function matches(path::String, @nospecialize(a), @nospecialize(e))
    if e isa Num
        typeof(a) == typeof(e) || return _report(path, "type $(typeof(a)) != expected $(typeof(e))")
        (isnan(e) ? isnan(a) : isapprox(a, e; rtol = RTOL, atol = ATOL)) ||
            return _report(path, "got $(repr(a)), expected $(repr(e))")
        return true
    elseif e isa AbstractArray && eltype(e) <: Num
        typeof(a) == typeof(e) || return _report(path, "type $(typeof(a)) != expected $(typeof(e))")
        size(a) == size(e) || return _report(path, "size $(size(a)) != expected $(size(e))")
        nan = isnan.(e)
        isnan.(a) == nan || return _report(path, "NaN pattern differs")
        ok = any(nan) ? isapprox(a[.!nan], e[.!nan]; rtol = RTOL, atol = ATOL) :
                        isapprox(a, e; rtol = RTOL, atol = ATOL)
        ok && return true
        i = argmax(abs.(ifelse.(nan, 0, a .- e)))
        return _report(path, "max deviation at $(Tuple(CartesianIndices(e)[i])): got $(repr(a[i])), " *
                             "expected $(repr(e[i])); norm(diff) = $(norm(ifelse.(nan, 0, a .- e)))")
    elseif e isa NamedTuple
        a isa NamedTuple || return _report(path, "got $(typeof(a)), expected a NamedTuple")
        keys(a) == keys(e) || return _report(path, "fields $(keys(a)) != expected $(keys(e))")
        return all([matches("$path.$k", a[k], e[k]) for k in keys(e)])
    elseif e isa Tuple
        a isa Tuple && length(a) == length(e) ||
            return _report(path, "got $(repr(a)), expected a $(length(e))-tuple")
        return all([matches("$path[$i]", a[i], e[i]) for i in eachindex(e)])
    elseif e isa AbstractArray
        typeof(a) == typeof(e) || return _report(path, "type $(typeof(a)) != expected $(typeof(e))")
        size(a) == size(e) || return _report(path, "size $(size(a)) != expected $(size(e))")
        return all([matches("$path[$i]", a[i], e[i]) for i in eachindex(e)])
    else
        typeof(a) == typeof(e) && isequal(a, e) && return true
        return _report(path, "got $(repr(a)), expected $(repr(e))")
    end
end

"Exact equality with identical leaf types (the generator's round-trip check)."
function identical(@nospecialize(a), @nospecialize(b))
    if b isa NamedTuple
        return a isa NamedTuple && keys(a) == keys(b) && all(identical(a[k], b[k]) for k in keys(b))
    elseif b isa Tuple
        return a isa Tuple && length(a) == length(b) && all(identical(a[i], b[i]) for i in eachindex(b))
    elseif b isa AbstractArray
        return typeof(a) == typeof(b) && size(a) == size(b) && all(identical(a[i], b[i]) for i in eachindex(b))
    else
        return typeof(a) == typeof(b) && isequal(a, b)
    end
end

function check_case(golden, got)
    if golden.throws !== nothing
        got.throws == golden.throws ||
            return _report(golden.name, "expected a $(golden.throws), got " *
                                        (got.throws === nothing ? "a value" : got.throws))
        got.expected.message_prefix == golden.expected.message_prefix && return true
        return _report(golden.name, "error message $(repr(got.expected.message_prefix)) != " *
                                    "expected $(repr(golden.expected.message_prefix))")
    end
    got.throws === nothing ||
        return _report(golden.name, "now throws $(got.throws): $(got.expected.message_prefix)")
    return matches(golden.name, got.expected, golden.expected)
end

function run_tests(golden)
    @testset "Scftopo outputs are pinned" begin
        code_names = [c[1] for c in CASES]
        data_names = [g.name for g in golden]
        @test allunique(code_names)
        @test allunique(data_names)
        missing_data = setdiff(code_names, data_names)
        stale_data = setdiff(data_names, code_names)
        isempty(missing_data) || @error "Cases without golden data (regenerate)" missing_data
        isempty(stale_data) || @error "Golden data without a case" stale_data
        @test isempty(missing_data)
        @test isempty(stale_data)
        byname = Dict(CASES)
        for g in golden
            haskey(byname, g.name) || continue
            @test check_case(g, run_case(g.name, byname[g.name]))
        end
    end
end

end # module ScftopoGolden

if !isdefined(@__MODULE__, :SCFTOPO_GOLDEN_GENERATOR)
    ScftopoGolden.run_tests(include(joinpath(@__DIR__, "data", "scftopo_golden.jl")))
end
