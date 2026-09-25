# utils.jl - shared infrastructure used across TensorBinding
#
# Functions here are pure plumbing: binary <-> MPS conversions,
# site-index manipulation, diagonal MPO construction, and debug
# helpers.  No physics lives here.

# ============================================================
# Operator extensions (defined once to avoid duplicate definitions)
# ============================================================

ITensors.op(::OpName"sigma_plus", ::SiteType"Qubit") =
    [0 1
     0 0]

ITensors.op(::OpName"sigma_minus", ::SiteType"Qubit") =
    [0 0
     1 0]

# ============================================================
# Binary / index utilities
# ============================================================


# ---------------------------------------------------------------------
# Shift MPO:  (Q f)(x) = f(x + q)  on a binary-encoded chain
# ---------------------------------------------------------------------

function build_shift_mpo(sites, q,cyclic=true)
    N      = length(sites)
    q_bits = [(q >> (N - i)) & 1 for i in 1:N]
    links  = [Index(2, "Link,l$n") for n in 0:N+1]
    mpo    = MPO(sites)

    for n in N:-1:1
        s     = sites[n]
        l_in  = links[n+1]
        l_out = links[n]
        T     = ITensor(s', s, l_in, l_out)
        qn    = q_bits[n]

        for cin in 0:1, s_val in 0:1
            total   = s_val + qn + cin
            res_val = total % 2
            cout    = div(total, 2)
            T[s' => (res_val + 1), s => (s_val + 1),
              l_in => (cin == 1 ? 1 : 2), l_out => (cout == 1 ? 1 : 2)] = 1.0
        end
        mpo[n] = T
    end

    mpo[N] *= onehot(links[N+1] => 2)
    if cyclic
        mpo[1] *= (onehot(links[1] => 1) + onehot(links[1] => 2)) # cyclic
    else
        mpo[1] *= onehot(links[1] => 2)
    end

    return mpo
end

build_shift_mpo(sites, q::Integer; cyclic::Bool=false) =
    build_shift_mpo(sites, q, cyclic)

build_cyclic_shift_mpo(sites, q::Integer) = build_shift_mpo(sites, q, true)

shift_adjoint_mpo(K::MPO) = swapprime(dag(K), 0, 1)

function shift_mpo(sites, q::Integer; cyclic::Bool=false)
    q >= 0 && return build_shift_mpo(sites, q, cyclic)
    K = build_shift_mpo(sites, -q, cyclic)
    return shift_adjoint_mpo(K)
end

function shift_pair_mpos(sites, q::Integer; cyclic::Bool=false)
    K = shift_mpo(sites, q; cyclic=cyclic)
    return K, shift_adjoint_mpo(K)
end

function shift_hopping_mpo(hopping::MPO, sites, q::Integer;
                           cyclic::Bool=false,
                           maxdim::Int=typemax(Int),
                           cutoff::Real=1e-12,
                           apply_kwargs=NamedTuple())
    K, Kdag = shift_pair_mpos(sites, q; cyclic=cyclic)
    apkw = isempty(apply_kwargs) ?
        (maxdim == typemax(Int) ? (; cutoff=cutoff) : (; cutoff=cutoff, maxdim=maxdim)) :
        apply_kwargs
    return +(apply(hopping, K; apkw...),
             apply(Kdag, dag(hopping); apkw...);
             cutoff=cutoff,
             (maxdim == typemax(Int) ? NamedTuple() : (; maxdim=maxdim))...)
end


"""
    to_binary_vector(n, L) -> Vector{String}

Convert non-negative integer `n` to a length-`L` vector of `"0"`/`"1"`
strings (big-endian), suitable as state labels for `MPS(sites, state)`.

# Example
```julia
to_binary_vector(5, 4)   # -> ["0", "1", "0", "1"]
```
"""
function to_binary_vector(n::Integer, L::Integer)
    return map(string, collect(lpad(string(n; base=2), L, '0')))
end


"""
    binary_to_MPS(n, L, sites) -> MPS

Return the computational-basis state |n> as an `L`-site MPS, where
`n` is encoded in big-endian binary across the `L` qubit sites.

# Example
```julia
sites = siteinds("Qubit", 4)
psi   = binary_to_MPS(5, 4, sites)   # |0101>
```
"""
function binary_to_MPS(n::Integer, L::Integer, sites)
    return MPS(sites, to_binary_vector(n, L))
end

# ============================================================
# MPO / MPS site-index manipulation
# ============================================================

"""
    fix_sites(mpo, sites) -> MPO

Replace the site indices of `mpo` (typically built from a TCI
tensor train whose indices do not match the system's physical sites)
with `sites`.  Modifies `mpo` in-place and returns it.
"""
function fix_sites(mpo, sites)
    oldsites      = getindex.(siteinds(mpo), 2)   # unprimed (ket)
    oldsitesprime = getindex.(siteinds(mpo), 1)   # primed   (bra)
    for i in eachindex(mpo)
        mpo[i] = replaceind(mpo[i], oldsites[i]      => sites[i])
        mpo[i] = replaceind(mpo[i], oldsitesprime[i] => sites[i]')
    end
    return mpo
end


"""
    custom_mpo(mps, new_sites) -> MPO

Convert a `2N`-site MPS produced by QTCI on an interleaved 2D
quantics grid into an `N`-site MPO by contracting each pair of
tensors `(2i-1, 2i)` and mapping old site indices to `new_sites[i]`.

The first index of each pair becomes the bra (primed) site and the
second becomes the ket (unprimed) site, consistent with ITensors
MPO conventions.
"""
function custom_mpo(mps, new_sites)
    N     = length(mps)
    new_N = div(N, 2)
    @assert new_N == length(new_sites) "MPS has $N sites but new_sites has $(length(new_sites)) sites; expected $new_N."
    new_mpo = MPO(new_N)
    for i in 1:new_N
        A          = mps[2i - 1]
        B          = mps[2i]
        combined_T = A * B
        old_s1     = siteind(mps, 2i - 1)   # -> bra (primed)
        old_s2     = siteind(mps, 2i)       # -> ket (unprimed)
        new_mpo[i] = replaceinds(combined_T,
                                 [old_s1, old_s2] => [new_sites[i]', new_sites[i]])
    end
    return new_mpo
end


"""
    fused_mpo(mps, new_sites) -> MPO

Convert an `N`-site MPS with dim-4 physical indices produced by QTCI on a
`:fused` 2D quantics grid into an `N`-site MPO.  Each dim-4 physical index
encodes one (bra-bit, ket-bit) pair; a combiner splits it into
`new_sites[i]'` (bra) and `new_sites[i]` (ket).
"""
function fused_mpo(mps, new_sites)
    N = length(mps)
    @assert N == length(new_sites) "MPS has $N sites but new_sites has $(length(new_sites)) sites."
    new_mpo = MPO(N)
    for i in 1:N
        T     = mps[i]
        old_s = siteind(mps, i)           # dim-4 fused index
        comb  = combiner(new_sites[i]', new_sites[i])
        c_idx = combinedind(comb)
        new_mpo[i] = replaceind(T, old_s, c_idx) * dag(comb)
    end
    return new_mpo
end


"""
    custom_mps(qtt, sites) -> MPS

Replace the site indices of an MPS obtained from a 1D TCI tensor
train with the physical `sites` of the target system.
"""
function custom_mps(qtt, sites)
    old_mps = ITensors.MPS(qtt)
    N       = length(old_mps)
    new_mps = MPS(N)
    for i in 1:N
        old_s   = siteind(old_mps, i)
        new_mps[i] = replaceinds(old_mps[i], [old_s] => [sites[i]])
    end
    return new_mps
end


"""
    mps2mpo(L, sites, density_mps) -> MPO

Convert a diagonal MPS (a function sampled on computational-basis
states) into a diagonal MPO by calling `Quantics._asdiagonal` on
each site tensor.
"""
function mps2mpo(L, sites, density_mps)
    density_mpo = outer(density_mps', density_mps)
    for i in 1:L
        density_mpo.data[i] = Quantics._asdiagonal(density_mps.data[i], sites[i])
    end
    return density_mpo
end


@inline function _bra_ket(sij)
    @assert length(sij) == 2 "MPO site tensor should have exactly 2 site legs"
    return plev(sij[1]) == 1 ? (sij[1], sij[2]) : (sij[2], sij[1])
end

"""
    replace_sites(MPOin, newsites) -> MPO

Replace the physical (bra + ket) indices of each site in `MPOin` with the
corresponding index from `newsites`, preserving prime levels.
"""
function replace_sites(MPOin::MPO, newsites)
    L = length(MPOin)
    @assert length(newsites) == L
    indsMPO = siteinds(MPOin)
    T = MPO(L)
    for n in 1:L
        bra_old, ket_old = _bra_ket(indsMPO[n])
        T[n] = MPOin[n] *
               delta(bra_old, prime(newsites[n])) *
               delta(ket_old, newsites[n])
    end
    return T
end

"""
    hadamard_mpo(A, B, out_sites; maxdim=100, cutoff=1e-8) -> MPO

Site-wise Hadamard product of two MPOs: `C[i,j] = A[i,j] * B[i,j]`.
`out_sites` may be any index set, including A/B's own sites; fresh working
indices are created internally and the result is remapped to `out_sites`.
"""
function hadamard_mpo(A::MPO, B::MPO, out_sites;
                      maxdim::Int = 100,
                      cutoff::Real = 1e-8)
    L = length(A)
    @assert length(B) == L && length(out_sites) == L
    sindsA = siteinds(A)
    sindsB = siteinds(B)
    fresh_sites = [sim(out_sites[n]) for n in 1:L]

    tens = Vector{ITensor}(undef, L)
    for n in 1:L
        bra_A, ket_A = _bra_ket(sindsA[n])
        bra_B, ket_B = _bra_ket(sindsB[n])
        bra_out = prime(fresh_sites[n])
        ket_out = fresh_sites[n]
        bra_B_f = sim(bra_B)
        ket_B_f = sim(ket_B)
        B_n = replaceinds(B[n], [bra_B, ket_B], [bra_B_f, ket_B_f])
        W = A[n] * B_n
        W = W * delta(bra_A, bra_B_f, bra_out)
        W = W * delta(ket_A, ket_B_f, ket_out)
        tens[n] = W
    end

    if L > 1
        Cs = Vector{ITensor}(undef, L - 1)
        for b in 1:L-1
            lA = only(commoninds(A[b], A[b+1]))
            lB = only(commoninds(B[b], B[b+1]))
            Cs[b] = combiner(lA, lB; tags="Link,l=$b")
        end
        tens[1] = tens[1] * Cs[1]
        for n in 2:L-1
            tens[n] = tens[n] * Cs[n-1] * Cs[n]
        end
        tens[L] = tens[L] * Cs[L-1]
    end

    mpo = replace_sites(MPO(tens), out_sites)
    ITensorMPS.truncate!(mpo; maxdim=maxdim, cutoff=cutoff)
    return mpo
end

"""
    eval_mps(A, n) -> Real

Evaluate a profile MPS at the 0-indexed basis coordinate `n` using the
big-endian convention of `binary_to_MPS`. Do not use `_eval_diag_mps` for
this purpose — that helper uses LSB-first ordering for the QFT momentum
convention.
"""
function eval_mps(A::MPS, n::Int)
    sites = siteinds(A)
    psi = binary_to_MPS(n, length(sites), sites)
    return real(inner(psi, A))
end

# Block-integrated MPS element (reduce=:block): the sum of `A` over one coarse
# block, obtained by tracing out the within-block position bits (contracted with
# [1,1]) and pinning the kept top a/b block bits to the coarse pixel (ixp, iyp).
# Big-endian site order [iy_MSB..iy_LSB, ix_MSB..ix_LSB]: sites 1..Ly carry iy,
# Ly+1..L carry ix. See [`spatial_sampling_plan`](@ref) `reduce=:block`.
function _eval_block_mps(A::MPS, ixp::Int, iyp::Int,
                         a::Int, b::Int, Lx::Int, Ly::Int)
    s   = siteinds(A)
    ElT = eltype(A[1])
    L   = Lx + Ly
    acc = ITensor(one(ElT))
    for i in 1:L
        v_arr = zeros(ElT, dim(s[i]))
        if i <= b                       # keep: iy block bit (b - i)
            v_arr[((iyp >> (b - i)) & 1) + 1] = one(real(ElT))
        elseif i <= Ly                  # sum: iy within-block bit
            v_arr .= one(real(ElT))
        elseif i <= Ly + a              # keep: ix block bit (a - (i - Ly))
            v_arr[((ixp >> (a - (i - Ly))) & 1) + 1] = one(real(ElT))
        else                            # sum: ix within-block bit
            v_arr .= one(real(ElT))
        end
        acc *= A[i] * ITensor(v_arr, s[i])
    end
    return real(scalar(acc))
end

"""
    spatial_sampling_plan(L; Lx, grid, reduce, n_sub, num_x, num_y, num_avg,
                          x_start, x_end, xwin, ywin, x_groups, box_half, sublattice)
        -> (; centers, groups, resolve_sublattice, n_sub, stride_x, stride_y,
             grid, reduce, a, b)

Geometry-aware real-space sampling plan shared by every spatial sampler
([`eval_mps_spatial`](@ref), `get_ldos_spatial`, `get_ldos_spatial_gpu`,
`get_scf_magnetization_gpu`). It decides **where** to sample, **how** each output
pixel reduces the cells under it (`reduce`), and — for multi-atom unit cells —
whether to **resolve** or **average** the sublattice.

# The three sampling procedures (`reduce`)

A spatial map of a `2^Lx × 2^Ly`-unit-cell system at a coarse output resolution
can reduce the cells beneath each pixel in three qualitatively different ways.
The right choice depends on whether the quantity is *smooth on the large scale*
(e.g. a Chern marker, an SCF density envelope) or a *thin feature on a flat
background* (e.g. in-gap edge/domain-wall LDOS, width ξ ≪ system size).

1. **`:point` (default) — point / box sampling.**
   Lay out `num_x[×num_y]` sample positions and read the profile *at* each one.
   With `box_half > 0` each pixel is the **mean** over a `(2·box_half+1)²`
   neighbourhood (smoothing). Cost ∝ (number of pixels) × (box cells).

   *Aliasing caveat.* The pixels probe only the cells they land on (± `box_half`).
   On a grid coarser than a feature's width this **misses** thin features that
   fall between pixels: a domain-wall LDOS channel of width ξ sampled at stride
   `s ≫ ξ` is caught only on the rare pixel within `box_half` of it. Making the
   box *tile* the plane (`box_half ≈ s/2`) closes the gaps but then evaluates
   essentially every cell — i.e. full-resolution cost. Use `:point` for smooth
   quantities or for a fully-resolved zoom (`grid=true` + a small window).

2. **`:block` — block integration (gap-free coarse-graining).**
   Partition the system into `num_x × num_y` equal blocks (`num_x = 2^a`,
   `num_y = 2^b`, powers of two) and report, per pixel, the **sum** over its
   whole block. This is computed by *tracing out the low-order position bits*
   (contracting the within-block bits of the profile MPS with `[1,1]` and keeping
   the `a + b` high-order block bits) — a partial contraction, **not** a per-cell
   sweep, so the cost is independent of block size and scales to `Lx, Ly ≈ 14+`.

   Because every cell belongs to exactly one block, a thin feature **cannot fall
   between pixels** — whichever blocks it threads light up, on an otherwise dark
   (gapped) background. This is the tool for imaging edge / domain-wall networks
   on a heavily downsampled map. Block centres are reported in `centers`; the
   per-axis block widths are `stride_x = 2^(Lx-a)`, `stride_y = 2^(Ly-b)`.
   If `Lx` is omitted, the same mode becomes a 1D partition into `num_x = 2^a`
   contiguous intervals with `stride_x = 2^(L-a)`.

The fields `reduce`, `a`, `b` echo the chosen mode back to the caller; for
`:point` they are `(:point, 0, 0)`.

# Sublattice resolve vs average

For a multi-atom unit cell (`n_sub > 1`) the plan also decides whether to
**resolve** the sublattice (one output column per atom) or **average** it (one
value per unit cell, atoms traced out), via `sublattice`:

- `:auto` (default) — **resolve** only at the atomic scale: consecutive samples
  are adjacent unit cells (`:point` with `stride == 1` and `box_half == 0`).
  Otherwise (coarse grid, `box_half > 0`, or any `:block` map) **average**, since
  the intra-cell sublattice is below the sampling resolution.
- `:resolve` / `:average` force the choice. `n_sub == 1` is always `false`.

# Layout (`:point` mode)

`groups`/`centers` are 1-indexed unit-cell indices with `n = ix + iy·2^Lx`.

- `grid=false` (default) — centers on a **1D linear** sweep of the row-major index
  (`x_start`/`x_end`, `num_x` points, `num_avg` sub-probes per block). Stride
  `dx = window ÷ num_x`. `Lx` is used only for the optional `box_half` neighbourhood.
- `grid=true` (needs `Lx`) — centers on a **2D xy grid** of `num_x × num_y`
  points over the unit-cell window `xwin=(ix0,ix1)`, `ywin=(iy0,iy1)` (0-indexed;
  default full system). Per-axis strides `Nx_win÷num_x`, `Ny_win÷num_y`.

`x_groups` overrides the `:point` layout entirely; the stride is then unknown, so
`:auto` resolves (treats it as atomic) unless `box_half > 0`.
"""
function spatial_sampling_plan(L::Int;
                               Lx::Union{Nothing,Int}      = nothing,
                               grid::Bool                  = false,
                               reduce::Symbol              = :point,
                               n_sub::Int                  = 1,
                               num_x::Int                  = 0,
                               num_y::Union{Nothing,Int}   = nothing,
                               num_avg::Int                = 1,
                               x_start::Int                = 1,
                               x_end::Int                  = 2^L,
                               xwin                        = nothing,
                               ywin                        = nothing,
                               x_groups                    = nothing,
                               box_half::Int               = 0,
                               sublattice::Symbol          = :auto)
    sublattice in (:auto, :resolve, :average) ||
        error("spatial_sampling_plan: sublattice must be :auto, :resolve, or :average.")
    reduce in (:point, :block) ||
        error("spatial_sampling_plan: reduce must be :point or :block.")
    grid && Lx === nothing &&
        error("spatial_sampling_plan: grid=true requires Lx (the x-qubit count).")

    # ── :block — coarse-grain by tracing out the within-block position bits ────
    if reduce === :block
        if Lx === nothing
            num_x > 0 ||
                error("spatial_sampling_plan: 1D reduce=:block requires num_x > 0 (a power of two).")
            a = round(Int, log2(num_x))
            2^a == num_x ||
                error("spatial_sampling_plan: 1D reduce=:block needs num_x a power of two (got $num_x).")
            (0 <= a <= L) ||
                error("spatial_sampling_plan: 1D reduce=:block needs 1 <= num_x <= 2^L=$(2^L).")
            W = 2^(L - a)
            centers = Int[ixp * W + W ÷ 2 + 1 for ixp in 0:(num_x - 1)]
            groups = [[c] for c in centers]   # nominal; block eval does not use these
            return (; centers, groups, resolve_sublattice=false, n_sub=max(n_sub, 1),
                    stride_x=W, stride_y=1, grid=false, reduce=:block, a, b=0)
        end
        Ly = L - Lx
        num_x > 0 ||
            error("spatial_sampling_plan: reduce=:block requires num_x > 0 (a power of two).")
        nyv = num_y === nothing ? num_x : num_y
        nyv > 0 ||
            error("spatial_sampling_plan: reduce=:block requires num_y > 0 (a power of two).")
        a = round(Int, log2(num_x))
        b = round(Int, log2(nyv))
        2^a == num_x ||
            error("spatial_sampling_plan: reduce=:block needs num_x a power of two (got $num_x).")
        2^b == nyv ||
            error("spatial_sampling_plan: reduce=:block needs num_y a power of two (got $nyv).")
        (0 <= a <= Lx) ||
            error("spatial_sampling_plan: reduce=:block needs 1 <= num_x <= 2^Lx=$(2^Lx).")
        (0 <= b <= Ly) ||
            error("spatial_sampling_plan: reduce=:block needs 1 <= num_y <= 2^Ly=$(2^Ly).")
        Nx = 2^Lx
        Wx = 2^(Lx - a)          # block width in x (unit cells)
        Wy = 2^(Ly - b)          # block width in y
        # Block centre cell, row-major over coarse pixels (ixp fastest):
        #   col = ixp + iyp*num_x + 1
        centers = Int[(ixp * Wx + Wx ÷ 2) + (iyp * Wy + Wy ÷ 2) * Nx + 1
                      for iyp in 0:(2^b - 1) for ixp in 0:(2^a - 1)]
        groups = [[c] for c in centers]   # nominal; block eval does not use these
        resolve = n_sub > 1 && sublattice === :resolve   # block is large-scale → average
        return (; centers, groups, resolve_sublattice=resolve, n_sub=max(n_sub, 1),
                stride_x=Wx, stride_y=Wy, grid=true, reduce=:block, a, b)
    end

    stride_x = 1
    stride_y = 1
    stride_known = true

    local centers::Vector{Int}
    local groups::Vector{Vector{Int}}

    if x_groups !== nothing
        groups = x_groups isa AbstractVector{<:AbstractVector} ?
                 [collect(Int, g) for g in x_groups] : [[Int(x)] for x in x_groups]
        centers = Int[first(g) for g in groups]
        stride_known = false   # caller-supplied positions: stride is not defined
    elseif grid
        Nx = 2^Lx
        Ny = 2^(L - Lx)
        ix0, ix1 = xwin === nothing ? (0, Nx - 1) : (Int(xwin[1]), Int(xwin[2]))
        iy0, iy1 = ywin === nothing ? (0, Ny - 1) : (Int(ywin[1]), Int(ywin[2]))
        Nx_win = ix1 - ix0 + 1
        Ny_win = iy1 - iy0 + 1
        nx = num_x <= 0 ? Nx_win : min(num_x, Nx_win)
        ny = num_y === nothing ? (num_x <= 0 ? Ny_win : min(nx, Ny_win)) :
             (num_y <= 0 ? Ny_win : min(num_y, Ny_win))
        stride_x = Nx_win ÷ nx
        stride_y = Ny_win ÷ ny
        xcenters = nx <= 1 ? [ix0] : round.(Int, range(ix0, ix1; length=nx))
        ycenters = ny <= 1 ? [iy0] : round.(Int, range(iy0, iy1; length=ny))
        centers = Int[ix + iy * Nx + 1 for iy in ycenters for ix in xcenters]
        groups  = [[c] for c in centers]
        if num_avg > 1
            # Spread `num_avg × num_avg` sub-samples ACROSS each coarse block
            # (spacing stride÷num_avg), then average. On a coarse grid this is the
            # correct block-average: it washes out fast on-site modulation (period ≪
            # stride) while preserving the slow structure. Contrast `box_half`, a
            # *contiguous* neighbourhood that spans only ±box_half cells — far less
            # than one coarse stride — so it cannot average out sub-stride modulation
            # and leaves per-pixel aliasing/speckle on a coarse grid.
            sx = max(1, stride_x ÷ num_avg)
            sy = max(1, stride_y ÷ num_avg)
            groups = [
                let uc0 = c - 1, ix_c = uc0 % Nx, iy_c = uc0 ÷ Nx
                    unique([mod(ix_c + a * sx, Nx) + mod(iy_c + b * sy, Ny) * Nx + 1
                            for b in 0:num_avg-1 for a in 0:num_avg-1])
                end
                for c in centers
            ]
        end
    else
        window = x_end - x_start + 1
        nx     = num_x <= 0 ? window : num_x
        dx     = max(window ÷ nx, 1)
        stride_x = dx
        dx_sub = max(1, dx ÷ num_avg)
        centers = Int[x_start + (i - 1) * dx for i in 1:nx]
        groups  = [[ x_start + (i - 1) * dx + k * dx_sub
                     for k in 0:num_avg-1
                     if x_start + (i - 1) * dx + k * dx_sub <= x_end ]
                   for i in 1:nx]
    end

    # ── 2D box averaging (periodic wrap) ───────────────────────────────────────
    # Skipped when num_avg>1 on a grid already spread each pixel into a sub-grid
    # (the two averaging modes are mutually exclusive; num_avg takes precedence).
    if box_half > 0 && num_avg <= 1 && Lx !== nothing
        Nx = 2^Lx
        Ny = 2^(L - Lx)
        groups = [
            let uc0 = first(grp) - 1
                ix0 = uc0 % Nx
                iy0 = uc0 ÷ Nx
                unique([mod(ix0 + Δx, Nx) + mod(iy0 + Δy, Ny) * Nx + 1
                        for Δy in -box_half:box_half for Δx in -box_half:box_half])
            end
            for grp in groups
        ]
    end

    # ── Sublattice resolve / average decision ──────────────────────────────────
    resolve = if n_sub <= 1
        false
    elseif sublattice === :resolve
        true
    elseif sublattice === :average
        false
    else  # :auto
        box_half == 0 &&
            (stride_known ? (stride_x <= 1 && (grid ? stride_y <= 1 : true)) : true)
    end

    return (; centers, groups, resolve_sublattice=resolve, n_sub=max(n_sub, 1),
            stride_x, stride_y, grid, reduce=:point, a=0, b=0)
end

"""
    eval_mps_spatial(A::MPS; num_x, num_avg, x_start, x_end, x_groups,
                     box_half, Lx) -> (values, centers, groups)

Higher-level spatial sampler for a profile MPS such as an SCF occupation/density
profile (`res.rho_up`). It mirrors `get_ldos_spatial`'s `num_x` / `num_avg` /
`x_groups` / `box_half` sampling-and-averaging API, but evaluates the MPS
directly with [`eval_mps`](@ref) instead of running a KPM recursion — so it is
cheap enough to sweep a very large system by sampling a grid of positions and
averaging, rather than evaluating all `2^L` sites.

For each sampled group of (1-indexed) site coordinates the returned value is the
mean of `eval_mps(A, x-1)` over that group. With `box_half > 0` each sampled
position is expanded into a `(2·box_half+1)²` neighborhood on the 2D grid
(periodic wrap), exactly like `get_ldos_spatial`; this needs the 2D layout, taken
from `Lx` (defaults to `L÷2`, with `Ly = L - Lx`).

# Keyword arguments
- `num_x`    : number of sampled grid positions (default: all `2^L` sites).
- `num_avg`  : sub-positions averaged per grid point along the 1D index (stride).
- `x_start`, `x_end` : 1-indexed sampling window (default `1 … 2^L`).
- `x_groups` : explicit groups — a vector of site indices (one per group) or a
  vector of vectors (each averaged). Overrides `num_x`/`num_avg`/`x_start`/`x_end`.
- `box_half` : 2D neighborhood half-width for averaging (0 = no box averaging).
- `Lx`       : number of x qubits for the 2D layout (default `L÷2`).

# Returns
- `values`  : `Vector{Float64}`, the averaged MPS value per group.
- `centers` : `Vector{Int}`, the 1-indexed center site of each sampled group.
- `groups`  : `Vector{Vector{Int}}`, the site indices averaged over per group.

For a 2D map, the center `(ix, iy)` of group `g` is
`ix = (centers[g]-1) % 2^Lx`, `iy = (centers[g]-1) ÷ 2^Lx`.
"""
function eval_mps_spatial(A::MPS;
                          num_x::Int    = prod(dim(s) for s in siteinds(A)),
                          num_avg::Int  = 1,
                          x_start::Int  = 1,
                          x_end::Int    = prod(dim(s) for s in siteinds(A)),
                          x_groups      = nothing,
                          box_half::Int = 0,
                          Lx::Union{Nothing,Int} = nothing)
    sites = siteinds(A)
    L     = length(sites)

    # ── Build groups + grid centers via the shared geometry-aware planner ──────
    plan = spatial_sampling_plan(L;
        Lx       = (box_half > 0 && Lx === nothing) ? L ÷ 2 : Lx,
        num_x    = num_x, num_avg = num_avg,
        x_start  = x_start, x_end = x_end,
        x_groups = x_groups, box_half = box_half)
    centers = plan.centers
    groups  = plan.groups

    # ── Evaluate + average ─────────────────────────────────────────────────────
    values = Float64[ sum(eval_mps(A, x - 1) for x in grp) / length(grp)
                      for grp in groups ]
    return (values=values, centers=centers, groups=groups)
end

"""
    fibonacci_ldos_sampling_plan(L; depth=0, num_x=100, num_avg=1,
                                 orientation=:standard, alignment=:atomic,
                                 centered=true, origin=0)
    fibonacci_ldos_sampling_plan(H::TBHamiltonian; kwargs...)

Build a deterministic, bounded-size LDOS sampling plan for an `L`-qubit
Fibonacci approximant, the projected-space counterpart of
[`spatial_sampling_plan`](@ref). At `depth == 0` the plan covers the complete
`F_(L+2)` conumber interval. Each additional depth selects the nested atomic
renormalization window from [`fibonacci_rg_partition`](@ref), with
`effective_L == L - 3depth`.

The selected inherited interval is split into `min(num_x, window_count)`
contiguous integer intervals whose widths differ by at most one. Up to
`num_avg` equidistant conumbers (including both interval endpoints when there
is more than one sample) are chosen in each interval and mapped directly to
physical sites with [`fibonacci_site_from_conumber`](@ref). No full conumber
permutation or other `F_(L+2)`-element array is constructed; storage is
proportional to the requested output and sample counts.

Returned fields useful to an LDOS/Slurm/HDF5 workflow include:

- `groups`: physical-site vectors to pass as `x_groups` with
  `ordering=:physical`;
- `centers`: physical sites at the representative interval conumbers;
- `conumber_axis`: those representative conumbers in the original `L`
  coordinate system;
- `intervals`, `interval_first`, and `interval_last`: represented inherited
  conumber intervals;
- `sample_conumbers`: the inherited conumbers corresponding to `groups`;
- `sample_sites_flat`, `sample_conumbers_flat`, one-based `group_offsets`, and
  zero-based `group_offsets_zero`: flat representations convenient for Julia
  and Python/HDF5 consumers respectively (`flat[group_offsets[i]:
  group_offsets[i+1]-1]` reconstructs group `i` in Julia);
- `column_indices`: stable one-based output-column identifiers;
- `depth`, `effective_L`, the original `L` and `N`, requested/actual sampling
  counts, and all conumber conventions.

`intervals` and every field containing `conumber` use the requested `centered`
label convention. The corresponding `*_rank*` fields are always uncentered
ranks in `0:N-1`. Thus a zoom always retains its original-`L` coordinates;
the selected sites are never re-conumbered as an independent shorter chain.

When `num_avg` is larger than an interval, that interval is sampled at every
integer conumber and its `group_sizes` entry is smaller than `num_avg`.
`num_x` in the result is the actual number of output columns, while
`num_x_requested` records the input value.

Nested (`depth > 0`) windows are defined only for the canonical atomic phase,
so they require `alignment=:atomic` and `origin=0`. Reversing the orientation
is supported because it maps the canonical atomic interval onto itself.

The `TBHamiltonian` convenience method is defined in `position_spaces/Fibonacci.jl`.
"""
function fibonacci_ldos_sampling_plan(
    L::Integer;
    depth::Integer=0,
    num_x::Integer=100,
    num_avg::Integer=1,
    orientation::Symbol=:standard,
    alignment::Symbol=:atomic,
    centered::Bool=true,
    origin::Integer=0,
)
    num_x > 0 || throw(ArgumentError("num_x must be positive"))
    num_avg > 0 || throw(ArgumentError("num_avg must be positive"))
    orientation in (:standard, :reversed) ||
        throw(ArgumentError("orientation must be :standard or :reversed"))
    alignment in (:atomic, :raw) ||
        throw(ArgumentError("alignment must be :atomic or :raw"))
    if depth > 0 && (alignment !== :atomic || !iszero(origin))
        throw(ArgumentError(
            "depth > 0 requires alignment=:atomic and origin=0 so the selected " *
            "window remains the canonical nested atomic renormalization window",
        ))
    end

    # Work in uncentered ranks while partitioning. This keeps the RG embedding
    # independent of how callers choose to label the inherited conumber axis.
    partition = fibonacci_rg_partition(L; depth, centered=false)
    N = fibonacci_site_count(L)
    window_rank_first = first(partition.window_ranks)
    window_rank_last = last(partition.window_ranks)
    window_count = partition.window_count
    ncolumns = min(Int(num_x), window_count)

    # Tile the window exactly. Putting the remainder in the first intervals is
    # deterministic and makes every width either floor(W/n) or ceil(W/n).
    base_width, remainder = divrem(window_count, ncolumns)
    rank_intervals = Vector{UnitRange{Int}}(undef, ncolumns)
    cursor = window_rank_first
    for column in 1:ncolumns
        width = base_width + Int(column <= remainder)
        rank_intervals[column] = cursor:(cursor + width - 1)
        cursor += width
    end
    @assert cursor == window_rank_last + 1

    shift = centered ? fld(N, 2) : 0
    rank_to_conumber(rank::Int) = rank - shift
    to_axis_interval(interval::UnitRange{Int}) =
        rank_to_conumber(first(interval)):rank_to_conumber(last(interval))

    intervals = [to_axis_interval(interval) for interval in rank_intervals]
    interval_first = first.(intervals)
    interval_last = last.(intervals)
    interval_rank_first = first.(rank_intervals)
    interval_rank_last = last.(rank_intervals)

    # Integer samples are as uniformly spaced as possible. With one requested
    # sample use the lower integer midpoint; with two or more include endpoints.
    function equidistant_ranks(interval::UnitRange{Int})
        width = length(interval)
        count = min(Int(num_avg), width)
        lo = first(interval)
        count == 1 && return Int[lo + fld(width - 1, 2)]
        return Int[lo + fld(k * (width - 1), count - 1)
                   for k in 0:(count - 1)]
    end

    sample_ranks = [equidistant_ranks(interval) for interval in rank_intervals]
    sample_conumbers = [[rank_to_conumber(rank) for rank in ranks]
                         for ranks in sample_ranks]
    center_ranks = Int[first(interval) + fld(length(interval) - 1, 2)
                       for interval in rank_intervals]
    conumber_axis = rank_to_conumber.(center_ranks)

    site_from_rank(rank::Int) = fibonacci_site_from_conumber(
        L, rank_to_conumber(rank);
        orientation, alignment, centered, origin,
    )
    groups = [[site_from_rank(rank) for rank in ranks] for ranks in sample_ranks]
    centers = site_from_rank.(center_ranks)

    group_sizes = length.(groups)
    group_offsets = Vector{Int}(undef, ncolumns + 1)
    group_offsets[1] = 1
    for column in 1:ncolumns
        group_offsets[column + 1] = group_offsets[column] + group_sizes[column]
    end
    group_offsets_zero = group_offsets .- 1
    group_offsets_base = 1
    total_samples = group_offsets[end] - 1
    sample_sites_flat = Vector{Int}(undef, total_samples)
    sample_conumbers_flat = Vector{Int}(undef, total_samples)
    sample_ranks_flat = Vector{Int}(undef, total_samples)
    for column in 1:ncolumns
        destination = group_offsets[column]:(group_offsets[column + 1] - 1)
        sample_sites_flat[destination] = groups[column]
        sample_conumbers_flat[destination] = sample_conumbers[column]
        sample_ranks_flat[destination] = sample_ranks[column]
    end

    window_first = rank_to_conumber(window_rank_first)
    window_last = rank_to_conumber(window_rank_last)
    metadata = (;
        format="TensorBinding.fibonacci_ldos_sampling_plan",
        format_version=1,
        L=Int(L),
        N,
        depth=Int(depth),
        effective_L=partition.effective_L,
        window_count,
        window_first,
        window_last,
        window_rank_first,
        window_rank_last,
        num_x=ncolumns,
        num_x_requested=Int(num_x),
        num_avg=Int(num_avg),
        total_samples,
        group_offsets_base,
        orientation=String(orientation),
        alignment=String(alignment),
        centered,
        origin=Int(origin),
    )

    return (;
        groups,
        centers,
        conumber_axis,
        intervals,
        interval_first,
        interval_last,
        interval_rank_first,
        interval_rank_last,
        sample_conumbers,
        sample_ranks,
        sample_sites_flat,
        sample_conumbers_flat,
        sample_ranks_flat,
        group_offsets,
        group_offsets_zero,
        group_offsets_base,
        group_sizes,
        column_indices=collect(1:ncolumns),
        L=Int(L),
        N,
        depth=Int(depth),
        effective_L=partition.effective_L,
        window_count,
        window_first,
        window_last,
        window_rank_first,
        window_rank_last,
        num_x=ncolumns,
        num_x_requested=Int(num_x),
        num_avg=Int(num_avg),
        total_samples,
        orientation,
        alignment,
        centered,
        origin=Int(origin),
        metadata,
    )
end

"""
    ilinspace(xmin, xmax, num_x) -> Vector{Int}

Return `num_x` as almost evenly spaced integers in `[xmin, xmax]`, inclusive,
with a preference for the endpoints.  Used to build the k-point center
grid for band-structure sampling.
"""
function ilinspace(xmin, xmax, num_x::Int)
    xvals = xmin:xmax
    _N = length(xvals)
    @assert 1 ≤ num_x ≤ _N
    num_x == 1 && return [0]
    step = (_N - 1) ÷ (num_x - 1)
    return collect(xmin:step:(xmin+step*(num_x-1)))
end

"""
    kspace_sampling_plan(L_pos, D; num_x, num_y=10, num_avg=1,
                         xmin=0, xmax=nothing, ymin=0, ymax=nothing,
                         k_groups_override=nothing) -> (; k_groups, num_x)

Momentum-space sampling plan shared by `get_bands` and `get_bands_gpu`, the
k-space counterpart of [`spatial_sampling_plan`](@ref). Momenta are 0-indexed
QFT register labels `k in 0:2^L_pos-1`.

- `k_groups_override` (for example from `kpath_2d`) is passed through untouched
  and sets `num_x = length(k_groups)`.
- `D == 1`: `num_x` centres from [`ilinspace`](@ref) over `[xmin, xmax]`
  (`xmax` defaults to `2^L_pos - 1`); with `num_avg > 1` each centre is widened
  to `num_avg` equidistant offsets within half a step on either side, clamped
  to the register.
- `D == 2`: `Lx = L_pos ÷ 2`; the first `min(num_x, 2^Lx)` points of the
  `ilinspace` grids in `x` and `y` are zipped diagonally into row-major labels
  `(y << Lx) | x`, again with optional `num_avg` widening. This is the legacy
  diagonal cut through the 2D zone; for high-symmetry paths use `kpath_2d`.
"""
function kspace_sampling_plan(L_pos::Int, D::Int;
                              num_x::Int,
                              num_y::Int = 10,
                              num_avg::Int = 1,
                              xmin::Int = 0,
                              xmax = nothing,
                              ymin::Int = 0,
                              ymax = nothing,
                              k_groups_override = nothing)
    if !isnothing(k_groups_override)
        return (; k_groups = k_groups_override, num_x = length(k_groups_override))
    end
    N = 2^L_pos
    if D == 1
        _xmax     = xmax === nothing ? N - 1 : Int(xmax)
        xcenters  = ilinspace(xmin, _xmax, num_x)
        half_step = num_x > 1 ? (_xmax - xmin) / (2 * num_x) : 0
        offsets   = num_avg > 1 ? round.(Int, range(-half_step, half_step; length=num_avg)) : Int[0]
        k_groups  = [clamp.(xcenters[i] .+ offsets, 0, N - 1) for i in 1:num_x]
        return (; k_groups, num_x)
    elseif D == 2
        Lx     = div(L_pos, 2)
        Nx_loc = 2^Lx
        Ny_loc = 2^(L_pos - Lx)
        nx     = min(num_x, Nx_loc)   # can't have more output pts than grid positions
        _xmax  = xmax === nothing ? Nx_loc - 1 : Int(xmax)
        _ymax  = ymax === nothing ? Ny_loc - 1 : Int(ymax)
        xcenters    = ilinspace(xmin, _xmax, Nx_loc)
        ycenters    = ilinspace(ymin, _ymax, Ny_loc)
        half_step_x = nx > 1 ? (_xmax - xmin) / (2 * nx) : 0
        half_step_y = num_y > 1 ? (_ymax - ymin) / (2 * num_y) : 0
        x_offs = num_avg > 1 ? round.(Int, range(-half_step_x, half_step_x; length=num_avg)) : Int[0]
        y_offs = num_avg > 1 ? round.(Int, range(-half_step_y, half_step_y; length=num_avg)) : Int[0]
        k_groups = [
            begin
                xs = clamp.(xcenters[i] .+ x_offs, 0, Nx_loc - 1)
                ys = clamp.(ycenters[i] .+ y_offs, 0, Ny_loc - 1)
                [(y << Lx) | x for (x, y) in zip(xs, ys)]  # diagonal zip in 2D k-space
            end
            for i in 1:nx
        ]
        return (; k_groups, num_x = nx)
    else
        error("kspace_sampling_plan: D must be 1 or 2")
    end
end

"""
    rms_error(a, b) -> Float64

RMS distance between two MPS objects over all computational-basis states.
"""
function rms_error(a::MPS, b::MPS)
    diff = a - b
    n = prod(dim(s) for s in siteinds(a))
    return sqrt(abs(real(inner(diff', diff))) / n)
end

"""
    constant_mps(sites, value) -> MPS

Rank-1 MPS whose amplitude is `value` on every computational-basis state.
"""
function constant_mps(sites::Vector{<:Index}, value::Number)
    isempty(sites) && return MPS(ITensor[])
    return extract_diagonal_to_mps(value * MPO(sites, "Id"))
end

"""
    get_mps(L, sites, f; type=Float64, tol=1e-8) -> MPS

Compress a scalar profile `f(n)` on `n = 0, ..., 2^L-1` into an MPS via QTCI.
"""
function get_mps(L::Int, sites, f; type=Float64, tol::Real=1e-8)
    if L == 1
        s = first(sites)
        T = ITensor(type, s)
        T[s => 1] = type(f(0))
        T[s => 2] = type(f(1))
        return MPS([T])
    end
    xvals = range(1, 2^L; length=2^L)
    qtt, _, _ = quanticscrossinterpolate(type, i -> f(round(Int, i) - 1), xvals; tolerance=tol)
    tt = TensorCrossInterpolation.tensortrain(qtt.tci)
    return MPS(tt; sites=collect(sites))
end

"""
    get_mpo(L, sites, f; type=Float64, tol=1e-8, kwargs...) -> MPO

Compress a function into an MPO via QTCI using 0-indexed coordinates
`n = 0, ..., 2^L-1`. Dispatches on arity:

- **1-argument** `f(n)`: builds a diagonal MPO (on-site potential).
- **2-argument** `f(i, j)`: builds a full interaction MPO by 2D QTCI,
  pairing the interleaved quantics legs into bra/ket via `custom_mpo`.
"""
function get_mpo(L::Int, sites, f; type=Float64, tol::Real=1e-8, kwargs...)
    if applicable(f, 0, 0)
        N = 1 << L
        xvals = range(0, N - 1; length=N)
        kernel = (x, y) -> f(round(Int, x), round(Int, y))
        qtt, _, _ = quanticscrossinterpolate(type, kernel, [xvals, xvals];
                                             tolerance=tol, kwargs...)
        tt = TensorCrossInterpolation.tensortrain(qtt.tci)
        return custom_mpo(MPS(tt), collect(sites))
    else
        return mps2mpo(L, sites, get_mps(L, sites, f; type=type, tol=tol))
    end
end

"""
    get_diagonal_mpo(L, sites, f; type=Float64, tol=1e-8) -> MPO

Build a diagonal MPO with entry `f(x)` at 1-indexed site `x ∈ {1, ..., 2^L}`.
Wraps `get_mpo` with a 1→0 index shift for backward compatibility.

# Example
```julia
pot = get_diagonal_mpo(L, sites, x -> 0.01 * x)
```
"""
function get_diagonal_mpo(L, sites, f; type=Float64, tol::Real=1e-8)
    return get_mpo(L, sites, n -> f(n + 1); type=type, tol=tol)
end


"""
    extract_diagonal_to_mps(M) -> MPS

Extract the diagonal of an MPO `M` as an MPS by projecting each local bra/ket
pair onto equal physical values. This is shared by KPM trace/LDOS, SCF, RPA,
QFT, and purification routines.
"""
function extract_diagonal_to_mps(M::MPO)::MPS
    N = length(M)
    new_tensors = Vector{ITensor}(undef, N)
    for i in 1:N
        tensor = M[i]
        bra, ket = siteinds(M, i)
        diagonal_inds = uniqueinds(tensor, ket, bra)
        result = ITensor(diagonal_inds..., ket)
        for value in 1:dim(ket)
            slice = tensor * onehot(ket => value) * onehot(bra => value)
            result += slice * onehot(ket => value)
        end
        new_tensors[i] = result
    end
    return MPS(new_tensors)
end



# ---------------------------------------------------------------------
# MPS -> diagonal MPO conversion
# ---------------------------------------------------------------------

"""
    mps_to_diagonal_mpo(mps, sites) -> MPO

Convert an MPS into a diagonal MPO on `sites` by replacing each physical
index with a bra-ket pair tied by a 3-leg delta.  Used to convert the
output of a 2D QTCI (encoded as a flat MPS) into a diagonal MPO on the
interleaved (e.g. electron-hole) site space.
"""
function mps_to_diagonal_mpo(mps, sites)
    N          = length(mps)
    mpo_tensors = Vector{ITensor}(undef, N)
    for i in 1:N
        mps_t = mps[i]
        old_s = if i == 1
            uniqueind(mps_t, mps[i+1])
        elseif i == N
            uniqueind(mps_t, mps[i-1])
        else
            uniqueind(mps_t, mps[i-1], mps[i+1])
        end
        s              = sites[i]
        s_temp         = Index(dim(s), "temp")
        mpo_tensors[i] = replaceind(mps_t, old_s => s_temp) * delta(s_temp, s, s')
    end
    return MPO(mpo_tensors)
end

# ============================================================
# Auxiliary site prepend - unified prepend_op
# ============================================================

"""
    prepend_op(H_mpo, s, mat)       -> MPO   explicit matrix
    prepend_op(H_mpo, s, op::Symbol)-> MPO   named op (Spin / Nambu index)
    prepend_op(H_mpo, s, k, l)      -> MPO   sparse |k><l|  (Layer / any index)
    prepend_op(H_mpo, s, k)         -> MPO   projector |k><k|

Prepend a single-site operator on index `s` to `H_mpo`, extending it from
L sites to L+1 sites.  The returned MPO has site indices `[s; original...]`.

**Dispatch rules**
- Matrix form: `mat[i,j]` = <i|op|j> (1-indexed).  Element type is preserved.
- Symbol form: named operator looked up by the type of `s` (tag `"Spin"` or
  `"Nambu"`).  Defined in Supercond_tk.jl after the op dictionaries.
- Integer pair `(k, l)`: places a single 1 at row `k`, col `l` in a
  `dim(s) x dim(s)` zero matrix.  Covers layer hops and projectors for any
  dimension Layer index.
- Single integer `k`: shorthand for the projector `|k><k|`.
"""
function prepend_op(H_mpo::MPO, s::Index, mat::AbstractMatrix{T}) where T <: Number
    Lh    = length(H_mpo)
    bond0 = Index(1, "Link,l=0")
    Op    = ITensor(T, s', s, bond0)
    for j in axes(mat, 2), i in axes(mat, 1)
        iszero(mat[i, j]) || (Op[s' => i, s => j, bond0 => 1] = mat[i, j])
    end
    delta0 = ITensor(bond0);  delta0[bond0 => 1] = 1.0
    H1_ext = H_mpo[1] * delta0
    ext    = MPO(Lh + 1)
    ext[1] = Op
    ext[2] = H1_ext
    for k in 3:Lh+1
        ext[k] = H_mpo[k-1]
    end
    return ext
end
prepend_op(H::MPO, s::Index, mat::AbstractMatrix) =
    prepend_op(H, s, ComplexF64.(mat))

function prepend_op(H_mpo::MPO, s::Index, k::Int, l::Int)
    mat = zeros(Float64, dim(s), dim(s))
    mat[k, l] = 1.0
    return prepend_op(H_mpo, s, mat)
end
prepend_op(H_mpo::MPO, s::Index, k::Int) = prepend_op(H_mpo, s, k, k)


"""
    postpend_op(H_mpo, s, mat)        -> MPO   explicit matrix
    postpend_op(H_mpo, s, op::Symbol) -> MPO   named op (Spin / Nambu index)
    postpend_op(H_mpo, s, k, l)       -> MPO   sparse |k><l|  (Layer / any index)
    postpend_op(H_mpo, s, k)          -> MPO   projector |k><k|

Append a single-site operator on index `s` to the *end* of `H_mpo`, extending
it from L sites to L+1 sites.  The returned MPO has site indices `[original...; s]`.

Symmetric counterpart of `prepend_op`; dispatch rules are identical.
"""
function postpend_op(H_mpo::MPO, s::Index, mat::AbstractMatrix{T}) where T <: Number
    Lh       = length(H_mpo)
    bond_end = Index(1, "Link,l=$Lh")
    Op       = ITensor(T, s', s, bond_end)
    for j in axes(mat, 2), i in axes(mat, 1)
        iszero(mat[i, j]) || (Op[s' => i, s => j, bond_end => 1] = mat[i, j])
    end
    delta_end = ITensor(bond_end);  delta_end[bond_end => 1] = 1.0
    HLast_ext = H_mpo[Lh] * delta_end
    ext       = MPO(Lh + 1)
    for k in 1:Lh-1;  ext[k] = H_mpo[k];  end
    ext[Lh]   = HLast_ext
    ext[Lh+1] = Op
    return ext
end
postpend_op(H::MPO, s::Index, mat::AbstractMatrix) =
    postpend_op(H, s, ComplexF64.(mat))

function postpend_op(H_mpo::MPO, s::Index, k::Int, l::Int)
    mat = zeros(Float64, dim(s), dim(s))
    mat[k, l] = 1.0
    return postpend_op(H_mpo, s, mat)
end
postpend_op(H_mpo::MPO, s::Index, k::Int) = postpend_op(H_mpo, s, k, k)


# ============================================================
# Debug / validation utilities
# ============================================================

# Build a product-state MPS with an explicit 1-indexed value per site.
# Works for any site types (Qubit, Layer, Spin, Nambu, ...).
function _product_state_mps(sites::Vector{<:Index}, vals::Vector{Int})
    n = length(sites)
    links   = [Index(1, "Link,l=$i") for i in 1:n-1]
    tensors = Vector{ITensor}(undef, n)
    if n == 1
        t = ITensor(sites[1]);  t[sites[1] => vals[1]] = 1.0
        tensors[1] = t
    else
        t = ITensor(sites[1], links[1])
        t[sites[1] => vals[1], links[1] => 1] = 1.0
        tensors[1] = t
        for i in 2:n-1
            t = ITensor(links[i-1], sites[i], links[i])
            t[links[i-1] => 1, sites[i] => vals[i], links[i] => 1] = 1.0
            tensors[i] = t
        end
        t = ITensor(links[n-1], sites[n])
        t[links[n-1] => 1, sites[n] => vals[n]] = 1.0
        tensors[n] = t
    end
    return MPS(tensors)
end


# Build a product-state MPS for basis state k (0-indexed, big-endian across
# sites) without using string state names.  Works for any site types.
function _basis_state_mps(k::Int, sites::Vector{<:Index})
    n    = length(sites)
    dims = dim.(sites)
    vals = Vector{Int}(undef, n)
    rem  = k
    for i in n:-1:1          # peel off LSB first (big-endian storage)
        vals[i] = rem % dims[i] + 1   # 1-based ITensors convention
        rem      = div(rem, dims[i])
    end
    links   = [Index(1, "Link,l=$i") for i in 1:n-1]
    tensors = Vector{ITensor}(undef, n)
    if n == 1
        t = ITensor(sites[1]);  t[sites[1] => vals[1]] = 1.0
        tensors[1] = t
    else
        t = ITensor(sites[1], links[1])
        t[sites[1] => vals[1], links[1] => 1] = 1.0
        tensors[1] = t
        for i in 2:n-1
            t = ITensor(links[i-1], sites[i], links[i])
            t[links[i-1] => 1, sites[i] => vals[i], links[i] => 1] = 1.0
            tensors[i] = t
        end
        t = ITensor(links[n-1], sites[n])
        t[links[n-1] => 1, sites[n] => vals[n]] = 1.0
        tensors[n] = t
    end
    return MPS(tensors)
end


"""
    matrix_checker(mpo, sites, i, j) -> Number
    matrix_checker(mpo, L, sites, i, j) -> Number   (L ignored)

Return the matrix element <i|mpo|j>.  Works for any site types
(Qubit, Layer, Spin, Nambu, ...).  Intended for small-system validation.
"""
function matrix_checker(mpo, sites, i, j)
    psii = _basis_state_mps(Int(i), sites)
    psij = _basis_state_mps(Int(j), sites)
    return inner(psii, apply(mpo, psij))
end
matrix_checker(mpo, ::Int, sites, i, j) = matrix_checker(mpo, sites, i, j)


"""
    get_matrix(mpo, sites) -> Matrix{ComplexF64}
    get_matrix(mpo, L, sites) -> Matrix{ComplexF64}   (L ignored)

Return the full `D x D` dense matrix of `mpo`, where `D = prod(dim(s_i))`.
Works for any site types (Qubit, Layer, Spin, Nambu, ...).
Feasible only for small systems (D <= 512).
"""
function get_matrix(mpo, sites)
    sz  = prod(dim(s) for s in sites)
    mat = Matrix{ComplexF64}(undef, sz, sz)
    for i in 0:sz-1, j in 0:sz-1
        mat[i+1, j+1] = matrix_checker(mpo, sites, i, j)
    end
    return mat
end
get_matrix(mpo, ::Int, sites) = get_matrix(mpo, sites)

