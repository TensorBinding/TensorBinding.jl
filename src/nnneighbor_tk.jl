# nnneighbor_tk.jl — nth-nearest-neighbor hopping for 2D TBHamiltonians
#
# Public API:
#   add_hopping_2D!(H, f; Lx, Ly, nn=1, ...)  — f: scalar, f(Δix,Δiy,fs,ts),
#                                                f(n), f(ix,iy),
#                                                f(n,Δix,Δiy,fs,ts),
#                                                f(ix,iy,Δix,Δiy,fs,ts)
#   get_shell_disps(H, nn; Lx, Ly)             — inspect canonical displacements
#
# Internals:
#   _shift_mpo       — kinetic MPO for displacement (Δix, Δiy)
#   _nth_shell_disps — canonical displacement vectors for the nn-th shell

# ============================================================
# 1.  Low-level shift MPO
# ============================================================

"""
    _shift_mpo(Δix, Δiy, ku, kd, Id, brk_xp, Nx; apkw) -> MPO

Kinetic MPO that shifts the quantics unit-cell index by `Δix + Δiy·Nx`.

- x-component: composes single-step masked x-shifts (`apply(brk_xp, ku)` for
  forward, `apply(kd, brk_xp)` for backward) to suppress row wrap-arounds at
  every step, following the same pattern as the 2D lattice constructors.
- y-component: `compose_power(ku/kd, Nx·|Δiy|)` — pure y-shift, no break needed.
- x and y act on disjoint bit groups and commute; the result is `K_y ∘ K_x`.

`ku`, `kd`, `Id`, `brk_xp` must be pre-built from the same `pos_sites` for
efficiency when called in a loop over multiple displacements.
"""
function _shift_mpo(Δix::Int, Δiy::Int,
                    ku::MPO, kd::MPO, Id::MPO, brk_xp::MPO,
                    Nx::Int;
                    apkw = (; cutoff=1e-8, maxdim=100))
    K_x = if Δix > 0
        compose_power(apply(brk_xp, ku; apkw...), Δix; apply_kwargs=apkw)
    elseif Δix < 0
        compose_power(apply(kd, brk_xp; apkw...), -Δix; apply_kwargs=apkw)
    else
        Id
    end

    K_y = if Δiy > 0
        compose_power(ku, Nx * Δiy; apply_kwargs=apkw)
    elseif Δiy < 0
        compose_power(kd, Nx * (-Δiy); apply_kwargs=apkw)
    else
        Id
    end

    Δix == 0 && return K_y
    Δiy == 0 && return K_x
    return apply(K_y, K_x; apkw...)
end


# ============================================================
# 2.  Shell detection on a small reference patch
# ============================================================

"""
    _nth_shell_disps(H, n, Lx) -> Vector{NTuple{4,Int}}

Return the canonical set of displacement tuples `(Δix, Δiy, from_s, to_s)`
for the n-th nearest-neighbor shell of `H`.

Lattice vectors `a1`, `a2` and sublattice offsets are extracted from three
anchor cells (atom indices ≤ `n_sub·(Nx+1)`) and used to build a reference
patch of `(2n+3)²` unit cells analytically — no large atom-index queries,
no system-size constraint beyond `Lx, Ly ≥ 1`.

Only one member of each Hermitian-conjugate pair is returned, filtered by:

    Δix > 0  OR  (Δix=0 AND Δiy>0)  OR  (Δix=Δiy=0 AND from_s < to_s)
"""
function _nth_shell_disps(H::TBHamiltonian, n::Int, Lx::Int)
    @assert !isnothing(H.geometry) "H.geometry must be set for add_hopping_2D!"

    Nx    = 2^Lx
    n_sub = H.sublattice_s === nothing ? 1 : dim(H.sublattice_s)

    # Extract lattice vectors from three cheap anchor cells.
    # Atom indices used: 1..n_sub (cell 0,0), n_sub+1..2*n_sub (cell 1,0),
    # n_sub*Nx+1..n_sub*(Nx+1) (cell 0,1) — all ≤ n_sub*H.N for Lx,Ly ≥ 1.
    base   = [H.geometry(α)           for α in 1:n_sub]  # cell (0,0)
    xshift = [H.geometry(n_sub   + α) for α in 1:n_sub]  # cell (1,0)
    yshift = [H.geometry(n_sub*Nx + α) for α in 1:n_sub] # cell (0,1)
    a1 = xshift[1] - base[1]
    a2 = yshift[1] - base[1]

    Np   = 2n + 3
    ix_c = n + 1
    iy_c = n + 1

    # Build (2n+3)² reference patch positions analytically
    pos = Dict{NTuple{3,Int}, Vector{Float64}}()
    for ix in 0:Np-1, iy in 0:Np-1, α in 1:n_sub
        pos[(ix, iy, α)] = base[α] + ix * a1 + iy * a2
    end

    # Find the n-th shell distance
    all_dists = Float64[]
    for ix_t in 0:Np-1, iy_t in 0:Np-1, α in 1:n_sub, β in 1:n_sub
        push!(all_dists, norm(pos[(ix_t, iy_t, β)] - pos[(ix_c, iy_c, α)]))
    end
    unique_dists = sort(unique(round.(all_dists; digits=6)))
    filter!(d -> d > 1e-10, unique_dists)
    n ≤ length(unique_dists) ||
        error("Reference patch too small: cannot resolve shell n=$n (found only $(length(unique_dists)) shells).")
    d_shell = unique_dists[n]
    atol    = 1e-4

    # Collect canonical displacement tuples
    seen  = Set{NTuple{4,Int}}()
    disps = NTuple{4,Int}[]

    for ix_t in 0:Np-1, iy_t in 0:Np-1, from_s in 1:n_sub, to_s in 1:n_sub
        abs(norm(pos[(ix_t, iy_t, to_s)] - pos[(ix_c, iy_c, from_s)]) - d_shell) < atol ||
            continue
        Δix = ix_t - ix_c
        Δiy = iy_t - iy_c

        is_canonical = Δix > 0 ||
                       (Δix == 0 && Δiy > 0) ||
                       (Δix == 0 && Δiy == 0 && from_s < to_s)
        is_canonical || continue

        key = (Δix, Δiy, from_s, to_s)
        key ∈ seen && continue
        push!(seen, key);  push!(disps, key)
    end

    isempty(disps) && error("No displacements found for shell n=$n — check H.geometry.")
    return disps
end


# ============================================================
# 3.  Public API
# ============================================================

"""
    add_hopping_2D!(H, f; Lx, Ly, nn=1, layer=nothing,
                    lattice=nothing, geometry=nothing,
                    maxdim=50, tol=1e-8) -> H

Add the `nn`-th nearest-neighbor hopping to a 2D `TBHamiltonian`.

**Arguments**
- `f`: hopping amplitude.  Six forms are accepted:
  - **`Number`** — uniform amplitude for every bond in the shell.
  - **`f(Δix, Δiy, from_s, to_s)`** — direction-dependent scalar; called once per
    canonical displacement.  The Hermitian conjugate uses `conj(f(...))`.
    Call `get_shell_disps(H, nn; Lx, Ly)` to see which tuples will be passed.
  - **`f(n)`** — spatially varying amplitude as a function of the 0-based unit-cell
    index `n = ix + iy·Nx`.  The modulation MPO is built once via QTCI.
  - **`f(ix, iy)`** — same but with explicit 0-based 2D coordinates.
  - **`f(n, Δix, Δiy, from_s, to_s)`** — direction- and position-dependent; a
    separate modulation MPO is built per displacement via QTCI.
  - **`f(ix, iy, Δix, Δiy, from_s, to_s)`** — same but with explicit 2D coordinates.

  For spatially varying forms the modulation MPO `V` (built via `get_diagonal_mpo`)
  is applied at the **destination** of the forward hop: forward term uses
  `apply(V, K_fwd)`, the Hermitian conjugate uses `apply(K_bwd, dag(V))`.
- `Lx`, `Ly`: qubit factorisation of `H.L` into x-bits and y-bits (required).
- `layer`: for layered Hamiltonians, `nothing` applies the hopping to every
  layer, an integer targets one layer, and a collection targets those layers.
- `lattice` / `geometry`: optional geometry hints for layered Hamiltonians whose
  `H.geometry` is not set. `lattice` may be `:square`, `:triangular`, or
  `:honeycomb`; `geometry` may be either a function `i -> r_i` or a coordinate
  matrix with one row per position site, or per atom when `H.sublattice_s` is set.
- `nn`: neighbor shell index (1 = nearest, 2 = next-nearest, …).

**Compatibility**
Works for all 2D geometries registered in `get_Hamiltonian`:
- No explicit sublattice (`square_2d`, `triangular_2d`): hopping added directly
  as a position-space kinetic MPO.
- Explicit sublattice (`honeycomb`, `honeycomb_nnn`, `kagome`, `lieb`, `dice`):
  each displacement is augmented with the sublattice transition matrix via
  `postpend_op`.

**Complexity**
Shell detection uses a reference patch of `(2nn+3)²` unit cells — O(nn²) work,
independent of system size.  MPO construction uses `compose_power` (O(log N) per
displacement).  QTCI-based spatial modulation adds O(2^L) function evaluations.

```julia
H = get_Hamiltonian("honeycomb_nnn", (t=1.0, t2=0.0); L=8, Lx=4, Ly=4)
add_hopping_2D!(H, 0.1; Lx=4, Ly=4, nn=3)

# Spatially modulated (Gaussian envelope on unit-cell index):
Nx = 2^4
add_hopping_2D!(H, n -> 0.1 * exp(-((n % Nx - Nx÷2)^2)/50); Lx=4, Ly=4, nn=1)
```
"""
function add_hopping_2D!(H::TBHamiltonian, f;
                          Lx::Int,
                          Ly::Int,
                          nn::Int     = 1,
                          layer       = nothing,
                          lattice     = nothing,
                          geometry    = nothing,
                          maxdim::Int = 50,
                          tol::Real   = 1e-8)
    if H.layer_s !== nothing
        (H.spin_s === nothing && H.nambu_s === nothing) ||
            error("Layered add_hopping_2D! currently supports layer/position/sublattice Hamiltonians only.")

        pos_s  = _pos_sites(H)
        geom   = _resolve_2d_geometry(H, lattice, geometry, Lx, Ly)
        layers = _resolve_layer_selection(H.layer_s, layer)
        term_sites = H.sublattice_s === nothing ? pos_s : [pos_s; H.sublattice_s]
        zero_mpo = H.sublattice_s === nothing ?
            0.0 * MPO(pos_s, "Id") :
            0.0 * postpend_op(MPO(pos_s, "Id"), H.sublattice_s,
                               Matrix{Float64}(I, dim(H.sublattice_s), dim(H.sublattice_s)))

        H_layered_term = nothing
        for ell in layers
            H_pos = TBHamiltonian(H.L, H.N, term_sites, copy(zero_mpo),
                                  geom, 0.0, 0.0,
                                  nothing, nothing, nothing, H.sublattice_s, :post,
                                  nothing, nothing, 0, nothing)
            add_hopping_2D!(H_pos, f;
                            Lx=Lx, Ly=Ly, nn=nn,
                            maxdim=maxdim, tol=tol)
            term = prepend_layer_projector(H_pos.mpo, H.layer_s, ell)
            H_layered_term = H_layered_term === nothing ? term :
                +(H_layered_term, term; cutoff=tol)
        end

        H.mpo = +(H.mpo, H_layered_term; cutoff=tol, maxdim=maxdim)
        ITensorMPS.truncate!(H.mpo; cutoff=tol, maxdim=maxdim)
        _invalidate_cache!(H)
        return H
    end

    layer === nothing ||
        error("add_hopping_2D!: `layer` was provided but H.layer_s is not set.")

    if H.geometry === nothing && (lattice !== nothing || geometry !== nothing)
        H.geometry = _resolve_2d_geometry(H, lattice, geometry, Lx, Ly)
    end

    apkw   = (; cutoff=tol, maxdim=maxdim)
    n_sub  = H.sublattice_s === nothing ? 1 : dim(H.sublattice_s)
    pos_s  = _pos_sites(H)
    Nx     = 2^Lx
    L      = H.L

    # Build kinetic primitives once — reused for every displacement
    ku     = generate_kin_u(pos_s, H.N)
    kd     = generate_kin_d(pos_s, H.N)
    Id_pos = MPO(pos_s, "Id")
    brk_xp = _row_break_mpo(Lx, Ly, pos_s; which=:xplus)

    disps  = _nth_shell_disps(H, nn, Lx)

    Ny     = 2^Ly
    max_dx = maximum(abs(Δix) for (Δix, _, _, _) in disps)
    max_dy = maximum(abs(Δiy) for (_, Δiy, _, _) in disps)
    max_dx < Nx || error("nn=$nn shell requires Δix up to $max_dx but Nx=$Nx. Use Lx ≥ $(ceil(Int, log2(max_dx+1))).")
    max_dy < Ny || error("nn=$nn shell requires Δiy up to $max_dy but Ny=$Ny. Use Ly ≥ $(ceil(Int, log2(max_dy+1))).")

    # Detect callable signature (most-specific first to avoid false positives)
    fkind = if f isa Number
        :scalar
    elseif applicable(f, 0, 0, 0, 0, 1, 1)  # f(ix, iy, Δix, Δiy, fs, ts)
        :pos2d_dir
    elseif applicable(f, 0, 0, 0, 1, 1)     # f(n, Δix, Δiy, fs, ts)
        :pos1d_dir
    elseif applicable(f, 0, 0, 1, 1)        # f(Δix, Δiy, fs, ts)
        :dir
    elseif applicable(f, 0, 0)              # f(ix, iy)
        :pos2d
    elseif applicable(f, 0)                 # f(n)
        :pos1d
    else
        error("Cannot determine f signature for add_hopping_2D!.\n" *
              "Supported: Number, f(Δix,Δiy,fs,ts), f(n), f(ix,iy),\n" *
              "           f(n,Δix,Δiy,fs,ts), f(ix,iy,Δix,Δiy,fs,ts)")
    end

    # Build position-only modulation MPO once (shared across all displacements)
    # Note: get_diagonal_mpo evaluates f at Float64 grid points (from range(1,2^L;length=2^L));
    # convert to Int so user lambdas can safely use ix/iy as array indices.
    V_shared = if fkind === :pos1d
        get_diagonal_mpo(L, pos_s, i -> f(round(Int,i) - 1); type=ComplexF64)
    elseif fkind === :pos2d
        get_diagonal_mpo(L, pos_s, i -> (n = round(Int,i)-1; f(n % Nx, n ÷ Nx)); type=ComplexF64)
    else
        nothing
    end

    new_term = nothing
    for (Δix, Δiy, from_s, to_s) in disps
        # Scalar amplitude and per-displacement spatial modulation MPO
        amp, V = if fkind === :scalar
            f, nothing
        elseif fkind === :dir
            f(Δix, Δiy, from_s, to_s), nothing
        elseif fkind === :pos1d || fkind === :pos2d
            one(ComplexF64), V_shared
        elseif fkind === :pos1d_dir
            one(ComplexF64),
            get_diagonal_mpo(L, pos_s,
                i -> f(round(Int,i)-1, Δix, Δiy, from_s, to_s); type=ComplexF64)
        else  # :pos2d_dir
            one(ComplexF64),
            get_diagonal_mpo(L, pos_s,
                i -> (n = round(Int,i)-1; f(n % Nx, n ÷ Nx, Δix, Δiy, from_s, to_s));
                type=ComplexF64)
        end

        K_fwd = _shift_mpo( Δix,  Δiy, ku, kd, Id_pos, brk_xp, Nx; apkw=apkw)
        K_bwd = _shift_mpo(-Δix, -Δiy, ku, kd, Id_pos, brk_xp, Nx; apkw=apkw)

        # Apply spatial modulation: V evaluated at destination of forward hop
        K_fwd_m = V === nothing ? K_fwd : apply(V,     K_fwd; apkw...)
        K_bwd_m = V === nothing ? K_bwd : apply(K_bwd, dag(V); apkw...)

        fwd, bwd = if H.sublattice_s === nothing
            amp * K_fwd_m, conj(amp) * K_bwd_m
        else
            M_fwd = zeros(ComplexF64, n_sub, n_sub);  M_fwd[from_s, to_s  ] = 1
            M_bwd = zeros(ComplexF64, n_sub, n_sub);  M_bwd[to_s,   from_s] = 1
            amp       * postpend_op(K_fwd_m, H.sublattice_s, M_fwd),
            conj(amp) * postpend_op(K_bwd_m, H.sublattice_s, M_bwd)
        end

        pair     = +(fwd, bwd; cutoff=tol)
        new_term = new_term === nothing ? pair : +(new_term, pair; cutoff=tol)
    end

    H.mpo = +(H.mpo, new_term; cutoff=tol, maxdim=maxdim)
    ITensorMPS.truncate!(H.mpo; cutoff=tol, maxdim=maxdim)
    _invalidate_cache!(H)
    return H
end

function _resolve_layer_selection(layer_s::Index, layer)
    n_layers = dim(layer_s)
    layers = if layer === nothing
        collect(1:n_layers)
    elseif layer isa Integer
        [Int(layer)]
    else
        Int.(collect(layer))
    end
    all(l -> 1 <= l <= n_layers, layers) ||
        error("layer selection $layers is outside 1:$n_layers.")
    return layers
end

function _resolve_2d_geometry(H::TBHamiltonian, lattice, geometry, Lx::Int, Ly::Int)
    if geometry !== nothing
        if geometry isa AbstractMatrix
            n_geom = H.sublattice_s === nothing ? H.N : dim(H.sublattice_s) * H.N
            size(geometry, 1) == n_geom ||
                error("geometry matrix has $(size(geometry, 1)) rows, expected $n_geom.")
            return let m = Float64.(geometry)
                i -> m[i, :]
            end
        elseif geometry isa Function
            return geometry
        else
            error("geometry must be a function i -> r_i or an Nxd matrix.")
        end
    end

    H.geometry !== nothing && return H.geometry

    lattice !== nothing ||
        error("Layered add_hopping_2D! needs `lattice=:square/:triangular/:honeycomb` " *
              "or `geometry=...` because H.geometry is not set.")
    lat = lattice isa Symbol ? lattice : Symbol(lattice)
    rs = H.sublattice_s === nothing ? lattice_positions(lat, Lx, Ly) :
         lat === :honeycomb         ? honeycomb_sublattice_positions(Lx, Ly) :
         error("lattice=:$lat with H.sublattice_s is not supported by add_hopping_2D!.")
    n_geom = H.sublattice_s === nothing ? H.N : dim(H.sublattice_s) * H.N
    size(rs, 1) == n_geom ||
        error("geometry for :$lat returned $(size(rs, 1)) sites, expected $n_geom.")
    return let m = Float64.(rs)
        i -> m[i, :]
    end
end


# ============================================================
# 4.  Displacement inspector
# ============================================================

"""
    get_shell_disps(H, nn; Lx, Ly) -> Vector{NTuple{4,Int}}

Print and return the canonical displacement tuples `(Δix, Δiy, from_s, to_s)`
for the `nn`-th nearest-neighbor shell of `H`.

Use this before writing a direction-dependent amplitude function for
`add_hopping_2D!` to see exactly which tuples `f(Δix, Δiy, from_s, to_s)`
will be called with.  The Hermitian-conjugate bond `(-Δix, -Δiy, to_s, from_s)`
is handled automatically and is not listed here.

```julia
H = get_Hamiltonian("honeycomb", (t=1.0,); L=4, Lx=2, Ly=2)
get_shell_disps(H, 1; Lx=2, Ly=2)
# nn=1 shell: 3 canonical bond type(s)  [Nx=4, Ny=4]
#   #   Δix  Δiy  from_s → to_s
#   1     0    0       1 → 2
#   2     1    0       2 → 1
#   3     0    1       2 → 1

add_hopping_2D!(H, (Δix, Δiy, fs, ts) -> Δix == 0 && Δiy == 0 ? 1.2 : 0.9;
                Lx=2, Ly=2, nn=1)
```
"""
function get_shell_disps(H::TBHamiltonian, nn::Int; Lx::Int, Ly::Int)
    disps = _nth_shell_disps(H, nn, Lx)
    Nx    = 2^Lx
    Ny    = 2^Ly
    n_sub = H.sublattice_s === nothing ? 1 : dim(H.sublattice_s)

    println("nn=$nn shell: $(length(disps)) canonical bond type(s)  [Nx=$Nx, Ny=$Ny]")
    if n_sub > 1
        println("  #   Δix  Δiy  from_s → to_s")
    else
        println("  #   Δix  Δiy")
    end
    for (k, (Δix, Δiy, fs, ts)) in enumerate(disps)
        warn_x = abs(Δix) >= Nx ? "  ← |Δix| ≥ Nx, system too small" : ""
        warn_y = abs(Δiy) >= Ny ? "  ← |Δiy| ≥ Ny, system too small" : ""
        if n_sub > 1
            println("  $k  $(lpad(Δix,4)) $(lpad(Δiy,4))      $fs → $ts$warn_x$warn_y")
        else
            println("  $k  $(lpad(Δix,4)) $(lpad(Δiy,4))$warn_x$warn_y")
        end
    end
    return disps
end
