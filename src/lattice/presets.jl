# presets.jl — preset QTCI model Hamiltonians: the 1D HUniform/HSSH/HAAH chains and
# the 2D HUniform2D*, HChern8, H2DChernhex and HQC2Dsquare lattices. Each returns a
# bare MPO on Qubit sites it creates itself; core/ModelRegistry.jl builds them by
# name (build_hamiltonian("model_name", ...)) and get_Hamiltonian wraps the result
# in a TBHamiltonian.
#
# Entry points: HUniform, HSSH, HAAH, HUniform2Dsquare, HUniform2Dhex,
#   HUniform2Dtri, HUniform2Dtri_bravais, HChern8, H2DChernhex, HQC2Dsquare.
#
# Depends on: core/Utils.jl (qtt_mpo), core/MPOTools.jl (sum_mpos),
# core/Hamiltonian.jl (kineticNNN) and lattice/hopping2d.jl (the 2D kinetic
# builders). Each preset sums its terms left to right in the order written, every
# partial sum compressed at `cutoff`.
#
# Split from the former lattice/2Dlattice_tk.jl.

# ============================================================
# 1. 1D preset chains
# ============================================================

"""
    HUniform(L, t; v=1e-6, tol_quantics=1e-8, maxbonddim_quantics=10, nn=1) -> MPO

Uniform-hopping tight-binding chain on 2^L sites with an optional uniform onsite
potential `v`. `v = 0` leaves the on-site term out (QTCI cannot compress a function that
vanishes everywhere; this used to throw "maxsamplevalue is zero!").
"""
function HUniform(L::Integer, t;
                  v::Real                  = 1e-6,
                  tol_quantics::Real       = 1e-8,
                  maxbonddim_quantics::Int = 10,
                  nn::Integer              = 1)
    N     = 2^L
    sites = siteinds("Qubit", L)
    xvals = 0:N-1
    hops_MPO   = qtt_mpo(L, xvals, sites, _ -> t;  tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)
    iszero(v) && return kineticNNN(L, sites, hops_MPO, nn)
    onsite_MPO = qtt_mpo(L, xvals, sites, _ -> v;  tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)
    return +(kineticNNN(L, sites, hops_MPO, nn), onsite_MPO; cutoff=1e-8)
end


"""
    HSSH(L, t, d; tol_quantics=1e-8, maxbonddim_quantics=10, nn=1) -> MPO

SSH (Su-Schrieffer-Heeger) Hamiltonian: dimerized hopping `t±d` on alternating bonds.
"""
function HSSH(L::Integer, t, d;
              tol_quantics::Real       = 1e-8,
              maxbonddim_quantics::Int = 10,
              nn::Integer              = 1)
    N     = 2^L
    sites = siteinds("Qubit", L)
    xvals = 0:N-1
    hops_MPO = qtt_mpo(L, xvals, sites, x -> iseven(x) ? (t + d) : (t - d);
                       tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)
    return kineticNNN(L, sites, hops_MPO, nn)
end


"""
    HAAH(L, V, phi, t; b=(1+√5)/2, tol_quantics=1e-8, maxbonddim_quantics=50) -> MPO

Aubry–André–Harper quasicrystal:
    H = t * Σ c†_{i+1}c_i + V * cos(2π b i + φ) * n_i

`V = 0` leaves the on-site term out (the clean chain; QTCI cannot compress a zero field).
"""
function HAAH(L::Integer, V, phi, t;
              b::Real                  = (1 + sqrt(5)) / 2,
              tol_quantics::Real       = 1e-8,
              maxbonddim_quantics::Int = 50)
    N     = 2^L
    sites = siteinds("Qubit", L)
    xvals = 0:N-1
    hops_MPO   = qtt_mpo(L, xvals, sites, _ -> t;  tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)
    iszero(V) && return kineticNNN(L, sites, hops_MPO, 1)
    onsite_MPO = qtt_mpo(L, xvals, sites, x -> V * cos(2pi * b * x + phi);
                         tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)
    return +(kineticNNN(L, sites, hops_MPO, 1), onsite_MPO; cutoff=1e-8)
end


# ============================================================
# 2. 2D preset lattices
# ============================================================

"""
    HUniform2Dsquare(Lx, Ly, t; tol_quantics=1e-8, maxbonddim_quantics=10, cutoff=1e-10) -> MPO

Uniform tight-binding Hamiltonian on a `2^Lx × 2^Ly` square lattice (row-major encoding).
Intra-row: `kineticintra2DNNN(…, nn=1)`.  Inter-row: `kineticNNN(…, nn=Nx)`.
"""
function HUniform2Dsquare(Lx::Integer, Ly::Integer, t;
                          tol_quantics::Real       = 1e-8,
                          maxbonddim_quantics::Int = 10,
                          cutoff::Real             = 1e-10)
    Nx    = 2^Lx
    L     = Lx + Ly
    N     = Nx * 2^Ly
    sites = siteinds("Qubit", L)
    xvals = 0:N-1
    hops  = qtt_mpo(L, xvals, sites, _ -> t; tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)
    Hintra = kineticintra2DNNN(Lx, Ly, sites, hops, 1)
    Hinter = kineticNNN(L, sites, hops, Nx)
    return +(Hintra, Hinter; cutoff=cutoff)
end


"""
    HUniform2Dhex(Lx, Ly, t; tol_quantics=1e-8, maxbonddim_quantics=10, cutoff=1e-10) -> MPO

Uniform tight-binding Hamiltonian on a `2^Lx × 2^Ly` hexagonal lattice.
Intra-row uses `kineticintra2DNNhex` (checkerboard mask); inter-row uses `kineticNNN(…, Nx)`.
"""
function HUniform2Dhex(Lx::Integer, Ly::Integer, t;
                       tol_quantics::Real       = 1e-8,
                       maxbonddim_quantics::Int = 10,
                       cutoff::Real             = 1e-10)
    Nx    = 2^Lx
    L     = Lx + Ly
    N     = Nx * 2^Ly
    sites = siteinds("Qubit", L)
    xvals = 0:N-1
    hops  = qtt_mpo(L, xvals, sites, _ -> t; tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)
    Hintra = kineticintra2DNNhex(Lx, Ly, sites, hops, 1)
    Hinter = kineticNNN(L, sites, hops, Nx)
    return +(Hintra, Hinter; cutoff=cutoff)
end


"""
    HUniform2Dtri(Lx, Ly, t; tol_quantics=1e-8, maxbonddim_quantics=10, cutoff=1e-10) -> MPO

Uniform tight-binding Hamiltonian on a `2^Lx × 2^Ly` triangular lattice.
Three kinetic terms:
- `kineticintra2DNNN(…, 1)` — intra-row NN
- `kineticinterNNNtriSWNE(…, Nx+1)` — SW↗NE diagonal
- `kineticinterNNNtriSENW(…, Nx-1)` — SE↖NW diagonal
"""
function HUniform2Dtri(Lx::Integer, Ly::Integer, t;
                       tol_quantics::Real       = 1e-8,
                       maxbonddim_quantics::Int = 10,
                       cutoff::Real             = 1e-10)
    Nx    = 2^Lx
    L     = Lx + Ly
    N     = Nx * 2^Ly
    sites = siteinds("Qubit", L)
    xvals = 0:N-1
    hops       = qtt_mpo(L, xvals, sites, _ -> t; tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)
    HintraNN   = kineticintra2DNNN(       Lx, Ly, sites, hops,  1)
    HinterNN   = kineticNNN(              L,       sites, hops, Nx)
    HinterSWNE = kineticinterNNNtriSWNE(  Lx, Ly, sites, hops, Nx + 1)
    HinterSENW = kineticinterNNNtriSENW(  Lx, Ly, sites, hops, Nx - 1)
    return sum_mpos((HintraNN, HinterNN, HinterSWNE, HinterSENW); cutoff=cutoff)
end


"""
    HUniform2Dtri_bravais(Lx, Ly, t; tol_quantics=1e-8, maxbonddim_quantics=10, cutoff=1e-10) -> MPO

Uniform tight-binding Hamiltonian on a `2^Lx × 2^Ly` triangular lattice with proper
Bravais vectors a1=(1,0), a2=(1/2,√3/2).  Exactly three bond types per unit cell:
- (Δix=+1, Δiy= 0): intra-row x  via `kineticintra2DNNN`
- (Δix= 0, Δiy=+1): y-hop        via `kineticNNN(…, Nx)`
- (Δix=+1, Δiy=-1): Bravais diag via `kineticinterNNNtri_bravais_diag`
"""
function HUniform2Dtri_bravais(Lx::Integer, Ly::Integer, t;
                                tol_quantics::Real       = 1e-8,
                                maxbonddim_quantics::Int = 10,
                                cutoff::Real             = 1e-10)
    Nx    = 2^Lx
    L     = Lx + Ly
    N     = Nx * 2^Ly
    sites = siteinds("Qubit", L)
    xvals = 0:N-1
    hops  = qtt_mpo(L, xvals, sites, _ -> t;
                    tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)
    Hintra = kineticintra2DNNN(Lx, Ly, sites, hops, 1)
    Hy     = kineticNNN(L,      sites, hops, Nx)
    Hdiag  = kineticinterNNNtri_bravais_diag(Lx, Ly, sites, hops)
    return sum_mpos((Hintra, Hy, Hdiag); cutoff=cutoff)
end


"""
    HChern8(Lx, Ly, V, t; a=5/64*2^Lx, t2=0.2t, tol_quantics=1e-8,
            maxbonddim_quantics=10, cutoff=1e-10) -> MPO

8-fold "Chern mosaic" Hamiltonian: uniform intra/inter-row hoppings modulated by
a spatially varying 8-fold pattern using 4 rotated k-vectors. Without modulation
(`V t2 = 0`) the diagonal terms, whose QTCI field would vanish identically, are left out.
"""
function HChern8(Lx::Integer, Ly::Integer, V, t;
                 a::Real                  = 5/64 * 2^Lx,
                 t2::Real                 = 0.2 * t,
                 tol_quantics::Real       = 1e-8,
                 maxbonddim_quantics::Int = 10,
                 cutoff::Real             = 1e-10)
    Nx    = 2^Lx
    L     = Lx + Ly
    N     = Nx * 2^Ly
    sites = siteinds("Qubit", L)
    xvals = 0:N-1

    alt_hop_x(x) = (-1)^mod(x + 1, Nx) * t

    function func8fold(x, y)
        Ka1 = (2pi/a) .* [1.0, 0.0];  Kb1 = (2pi/a) .* [0.0, 1.0]
        theta = deg2rad(45.0);  Rt = [cos(theta) sin(theta); -sin(theta) cos(theta)]
        K = (Ka1, Kb1, Rt*Ka1, Rt*Kb1)
        return sum(1im * V * t2 * cos(dot(k, [x, y]))^2 for k in K)
    end

    wrap(f) = i -> f(i % Nx, div(i, Nx))

    w_alt = wrap((x,y) -> alt_hop_x(x))
    w1    = wrap((x,y) -> t)
    w2    = wrap((x,y) -> alt_hop_x(mod(x-1, Nx)) * func8fold(x+0.5, y+0.5))
    w3    = wrap((x,y) -> alt_hop_x(x)             * func8fold(x-0.5, y+0.5))

    # func8fold ∝ V t2: without modulation w2 = w3 = 0, which QTCI cannot compress
    modulated = !iszero(V * t2)

    hops_MPO  = qtt_mpo(L, xvals, sites, w_alt; tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)
    hops_MPO1 = qtt_mpo(L, xvals, sites, w1;    tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)
    hops_MPO2 = modulated ? qtt_mpo(L, xvals, sites, w2; tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics) : nothing
    hops_MPO3 = modulated ? qtt_mpo(L, xvals, sites, w3; tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics) : nothing

    HinterNN  = kineticNNN(          L,    sites, hops_MPO,  Nx)
    HintraNN  = kineticintra2DNNN(   Lx, Ly, sites, hops_MPO1, 1)
    modulated || return +(HinterNN, HintraNN; cutoff=cutoff)
    HinterSWNE = kineticinterNNNSWNE(Lx, Ly, sites, hops_MPO2, Nx+1)
    HinterSENW = kineticinterNNNSENW(Lx, Ly, sites, hops_MPO3, Nx-1)

    return sum_mpos((HinterNN, HinterSWNE, HinterSENW, HintraNN); cutoff=cutoff)
end


"""
    H2DChernhex(Lx, Ly, t, t2, ms; uniformhaldane=false, uniformsemenoff=false,
                tol_quantics=1e-8, maxbonddim_quantics=10, cutoff=1e-10) -> MPO

Haldane Chern insulator on the brick-wall honeycomb of a `2^Lx × 2^Ly` grid: basis index
`x + 2^Lx y`, bonds `(x, y)–(x, y±1)` and `(x, y)–(x+1, y)` for even `x + y`, drawn as in
[`honeycomb_positions`](@ref).
- NN hopping `t`.
- NNN hopping `⟨i|H|j⟩ = -i T2(x, y) ν_ij`: the textbook, C3-symmetric Haldane term
  `T2 exp(i φ ν_ij)` at `φ = -π/2`, with `ν_ij = sign((d1 × d2)_z)` for the path `i → k → j`
  through the common neighbour `k`. In grid steps, `(0, +2)`, `(+1, -1)` and `(-1, -1)`
  share one sign and the opposite three the other.
- Semenoff mass: `-Ms(x, y)` on even `x + y`, `+Ms(x, y)` on odd `x + y`.
- By default `T2 = t2`, `Ms = ms` for `x < Nx/2` and `T2 = -t2`, `Ms = ms + 3.3√3 t2`
  for `x ≥ Nx/2`, a domain wall at `x = Nx/2` (the right half is trivial for small `ms`,
  since the critical mass is `3√3 |t2|`). `uniformhaldane=true` and `uniformsemenoff=true`
  make the fields uniform.
- A field that vanishes identically (`t = 0`; `t2 = 0`; `ms = 0` with `uniformsemenoff`
  or `t2 = 0`) leaves its terms out, since QTCI cannot compress a zero function (this
  used to throw "maxsamplevalue is zero!"); with all three zero the result is the zero MPO.

With uniform fields and `t = 1` this is `get_Hamiltonian("haldane", (t2=t2, phi=-π/2,
M=ms))` on `honeycomb_positions(Lx + Ly; Lx=Lx)`, up to the gauge `c → -c` on odd `x + y`.
Its Dirac masses are `-ms ± 3√3 t2`, so it is a Chern insulator for `|ms| < 3√3 |t2|`.
At `t2 > 0` its Chern number is opposite to that of the `"haldane"` preset, and of the
manuscript's `build_APSOS_hamiltonian`, at `phi = π/2`. Earlier versions gave the vertical
NNN bonds `(0, ±2)` the wrong sign, which made the Dirac masses `-ms ± √3 t2`.
"""
function H2DChernhex(Lx::Integer, Ly::Integer, t, t2, ms;
                     uniformhaldane::Bool     = false,
                     uniformsemenoff::Bool    = false,
                     tol_quantics::Real       = 1e-8,
                     maxbonddim_quantics::Int = 10,
                     cutoff::Real             = 1e-10)
    Nx    = 2^Lx
    L     = Lx + Ly
    N     = Nx * 2^Ly
    sites = siteinds("Qubit", L)
    xvals = 0:N-1

    T2 = uniformhaldane ? ((_,_) -> t2) :
         ((x,_) -> x < div(Nx, 2) ? t2 : -t2)

    Ms = uniformsemenoff ? ((_,_) -> ms) :
         ((x,_) -> x < div(Nx, 2) ? ms : ms + 3.3*sqrt(3)*t2)

    alt_hop_xy(x,y) = isodd(x+y) ? -1im*T2(x,y) : 1im*T2(x,y)
    semenoff(x,y)   = isodd(x+y) ? Ms(x,y) : -Ms(x,y)

    wrap(f) = i -> f(i % Nx, div(i, Nx))

    # A field that vanishes identically (t = 0; t2 = 0; ms = 0 with a uniform mass or with
    # t2 = 0) leaves its terms out: QTCI cannot compress a zero function.
    zero_t, zero_t2 = iszero(t), iszero(t2)
    zero_ms = iszero(ms) && (uniformsemenoff || zero_t2)

    hops_MPO      = zero_t  ? nothing : qtt_mpo(L, xvals, sites, wrap((x,y) -> t);               tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)
    hops_MPOalter = zero_t2 ? nothing : qtt_mpo(L, xvals, sites, wrap((x,y) -> alt_hop_xy(x,y)); tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)
    on_site_MPO   = zero_ms ? nothing : qtt_mpo(L, xvals, sites, wrap((x,y) -> semenoff(x,y));   tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)

    terms = MPO[]
    zero_t || push!(terms,
        kineticintra2DNNhex( Lx, Ly, sites, hops_MPO,      1),    # Hintra
        kineticNNN(          L,       sites, hops_MPO,      Nx))  # Hinter
    # The vertical second neighbour (x, y+2) forms a C3 triple with (x±1, y-1), so it takes
    # the opposite sign to the diagonal ones (x±1, y+1) built from the same amplitudes.
    zero_t2 || push!(terms,
        -1 * kineticNNN(     L,       sites, hops_MPOalter, 2*Nx),
        kineticinterNNNSWNE( Lx, Ly, sites, hops_MPOalter, Nx+1),
        kineticinterNNNSENW( Lx, Ly, sites, hops_MPOalter, Nx-1))
    zero_ms || push!(terms, on_site_MPO)
    isempty(terms) && return zero(ComplexF64) * MPO(sites, "Id")

    return sum_mpos(terms; cutoff=cutoff)
end


"""
    HQC2Dsquare(Lx, Ly, t=1.0; tol_quantics=1e-8, maxbonddim_quantics=100, cutoff=1e-10) -> MPO

Quasicrystal-modulated square lattice.  The hopping amplitude at each bond is evaluated
at the bond midpoint using an 8-fold modulation with two competing wavevectors
`b1 = 5√5 a/2` and `b2 = √3 Nx a/16`.
"""
function HQC2Dsquare(Lx::Integer, Ly::Integer, t::Real = 1.0;
                     tol_quantics::Real       = 1e-8,
                     maxbonddim_quantics::Int = 100,
                     cutoff::Real             = 1e-10)
    Nx    = 2^Lx
    L     = Lx + Ly
    N     = Nx * 2^Ly
    sites = siteinds("Qubit", L)
    xvals = 0:N-1

    function func8fold(x, y)
        a  = 1
        b1 = 5*sqrt(5)*a/2
        b2 = sqrt(3)*(Nx*a/16)
        Ka1 = 2pi .* [1.0, 0.0]; Kb1 = 2pi .* [0.0, 1.0]
        tht = deg2rad(45.0); Rt = [cos(tht) sin(tht); -sin(tht) cos(tht)]
        K   = (Ka1, Kb1, Rt*Ka1, Rt*Kb1)
        xy  = [x - Nx/2, y - Nx/2]
        return t * (1 + 0.1 * sum(2.5*cos(dot(k,xy)/b1) + cos(dot(k,xy)/b2) for k in K))
    end

    intra = i -> func8fold(i%Nx + 0.5, div(i, Nx))
    inter = i -> func8fold(i%Nx,        div(i, Nx) + 0.5)

    hops_intra = qtt_mpo(L, xvals, sites, intra; tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)
    hops_inter = qtt_mpo(L, xvals, sites, inter; tol_quantics=tol_quantics, maxbonddim_quantics=maxbonddim_quantics)

    Hintra = kineticintra2DNNN(Lx, Ly, sites, hops_intra, 1)
    Hinter = kineticNNN(L,           sites, hops_inter, Nx)
    return +(Hinter, Hintra; cutoff=cutoff)
end
