# solvers/kpm/recursion.jl — Chebyshev recursions of the kernel polynomial method
#
# Contents: the Chebyshev argument H̃ = (H − center)/scale of every KPM solver of
# the package (_scaled_hamiltonian); the three-term recursion T_{n+1} = 2H̃T_n − T_{n−1}
# itself (chebyshev_foreach, one step _chebyshev_step), on MPOs or MPS, CPU or GPU,
# with each caller's truncation policy as keywords; the cached recursions that keep
# every order, as MPOs T_n(H̃) (KPM_Tn) or as MPS T_n(H̃)|ψ₀⟩ (KPM_Tn_mps), each with
# a raw-MPO method and a TBHamiltonian method that caches the list on H; and the
# online MPS recursion that accumulates weighted moments ⟨ψ₀|T_n(H̃)|ψ₀⟩ without
# storing the states (_run_kpm_mps!), shared by the LDOS, DOS and exciton solvers of
# solvers/kpm/, by physics/qft/exciton_spectra.jl and by two GPU solvers.
#
# Entry points: _scaled_hamiltonian, chebyshev_foreach, KPM_Tn, KPM_Tn_mps,
#   _run_kpm_mps!
# Depends on: core/TBSystem.jl (TBHamiltonian, physical_projector),
#   solvers/DMRG.jl (_estimate_spectral_bounds, _ensure_scale!).
#
# Split from the former solvers/KPM_tk.jl in Tier 1; _scaled_hamiltonian (Tier 2)
# replaced the rescaling written out in each solver, chebyshev_foreach (Tier 2) the
# three-term loop written out in each of them (KPM, QFT, QPI, NH, GPU).

# ============================================================
# 1. The rescaled Hamiltonian H̃ = (H − center)/scale
# ============================================================

"""
    _scaled_hamiltonian(H_mpo::MPO, scale, center, identity::MPO; cutoff) -> MPO
    _scaled_hamiltonian(H::TBHamiltonian; cutoff, identity=physical_projector(H)) -> MPO

The Chebyshev argument `H̃ = (H − center·identity) / scale`, computed as
`(1 / scale) * +(H_mpo, (-center) * identity; cutoff=cutoff)`: the shift is added
with `cutoff` as the only truncation, then the sum is multiplied by `1 / scale`.

The `TBHamiltonian` method uses `H.mpo`, `H.scale`, `H.center` and, as `identity`,
`physical_projector(H)`, the identity on the physical states: for a projected
position space (Fibonacci & co.) the ambient `MPO(H.sites, "Id")` would shift the
unphysical register states to `−center/scale` and give them Chebyshev weight. A
caller that also needs the projector as `T₀` builds it once and passes it as
`identity`. The raw-MPO method takes the identity explicitly (`MPO(sites, "Id")`
where no position space is known).
"""
_scaled_hamiltonian(H_mpo::MPO, scale::Real, center::Real, identity::MPO; cutoff::Real) =
    (1 / scale) * +(H_mpo, (-center) * identity; cutoff = cutoff)

_scaled_hamiltonian(H::TBHamiltonian; cutoff::Real,
                    identity::MPO = physical_projector(H)) =
    _scaled_hamiltonian(H.mpo, H.scale, H.center, identity; cutoff = cutoff)


# ============================================================
# 2. The three-term recursion: chebyshev_foreach
# ============================================================

"""
    chebyshev_foreach(f!, H̃::MPO, T₀, N::Integer; maxdim, cutoff, T1=nothing,
                      apply_trunc=(:cutoff, :maxdim), add_trunc=(:cutoff, :maxdim),
                      post_trunc=(), two=2, negone=nothing) -> T_last

Run the Chebyshev recursion

    T_{n+1} = 2 H̃ T_n − T_{n−1}

from `T₀` and `T₁`, calling `f!(n, T_n)` for the orders `n = 0, 1, …, N − 1`, in that
order, each as soon as `T_n` is built. With `T₀` an MPO the terms are operators (the
operator Chebyshev series `T_n(H̃)` for `T₀ = I` or `physical_projector(H)`); with `T₀`
an MPS they are the Chebyshev vectors `T_n(H̃)|ψ₀⟩`. The tensors stay on the device of
`H̃` and `T₀` (CPU or GPU) with their element types; nothing is transferred.

`T1` is `T₁`. By default it is `apply(H̃, T₀)` with the truncation of the step's
product; the operator series pass `T1 = H̃` (`H̃ · I`, no product). Orders 0 and 1 are
always visited, also for `N < 2`. Returns the last term built: `T_{N−1}`, or `T₁`
when `N ≤ 2`.

Each later term is one step (`_chebyshev_step`), in this order:

    T_{n+1} = +(two * apply(H̃, T_n; <apply_trunc>), -T_{n−1}; <add_trunc>)
    truncate!(T_{n+1}; <post_trunc>)          # skipped when post_trunc = ()

where `<…>` passes the parameters named in the tuple, out of `:cutoff` and `:maxdim`,
with the values of the `cutoff` and `maxdim` keywords; `negone * T_{n−1}` replaces
`-T_{n−1}` when `negone` is a number. The callers differ in exactly these choices,
and each keeps its own (so no result changes); "both" is `(:cutoff, :maxdim)`,
"typed" is `T(2)` and `T(-1)` for the GPU element type `T`:

| callers | `apply_trunc` | `add_trunc` | `post_trunc` | `two`, `negone` |
|---------|---------------|-------------|--------------|-----------------|
| `KPM_Tn`, `KPM_Tn_gpu`, `get_ldos_spatial(mode=:mpo)`, `get_bands`, `get_qpi` | `(:cutoff,)` | `(:maxdim,)` | `(:cutoff,)` | `2`, `nothing` |
| `get_dos_trace` | `(:cutoff,)` | both | both | `2`, `nothing` |
| `get_ldos_spatial_gpu`, `get_bands_gpu` | both | both | `(:cutoff,)` | typed |
| `KPM_Tn_mps`, `_run_kpm_mps!` (also behind the GPU exciton LDOS and stochastic DOS) | both | both | `()` | `2`, `nothing` |
| `get_ldos_spatial_mps_gpu`, the GPU NH stochastic trace | both | both | `()` | typed |
| the CPU NH KPM (`nh_kpm_partials`, `_nh_*_online`, `_nh_kpm_mps_ldos`) | both | both | `()` | `2.0`, `nothing` |
| the GPU NH diagonal traces | both | both | both | typed |

The partial recurrence of the NH KPM, `P_k = 2S·T_{k−1} + 2A·P_{k−1} − P_{k−2}`, is
not a Chebyshev recursion: it rides on the `T_n` of this one inside `f!`
(physics/nh/kpm.jl, gpu/nh.jl).

The recursion holds only the last two terms: it drops `T_{n−1}` before calling
`f!(n + 1, T_{n+1})`, so a callback that frees GPU memory reclaims it. `f!` may keep
`T_n` (the cached lists push it) but must not modify it.
"""
function chebyshev_foreach(f!, Htilde::MPO, T0::Union{MPO,MPS}, N::Integer;
                           maxdim, cutoff,
                           T1::Union{Nothing,MPO,MPS}    = nothing,
                           apply_trunc::Tuple            = (:cutoff, :maxdim),
                           add_trunc::Tuple              = (:cutoff, :maxdim),
                           post_trunc::Tuple             = (),
                           two::Number                   = 2,
                           negone::Union{Nothing,Number} = nothing)
    apply_kwargs = _trunc_kwargs(apply_trunc, maxdim, cutoff)
    add_kwargs   = _trunc_kwargs(add_trunc, maxdim, cutoff)
    post_kwargs  = _trunc_kwargs(post_trunc, maxdim, cutoff)
    f!(0, T0)
    Tnm1 = T0
    Tn   = T1 === nothing ? apply(Htilde, T0; apply_kwargs...) : T1
    f!(1, Tn)
    for n in 2:N-1
        Tnext = _chebyshev_step(Htilde, Tn, Tnm1; apply_kwargs, add_kwargs, post_kwargs,
                                two, negone)
        Tnm1 = Tn
        Tn   = Tnext
        f!(n, Tn)
    end
    return Tn
end

"""
    _chebyshev_step(H̃, T_n, T_nm1; apply_kwargs, add_kwargs, post_kwargs=(;), two=2,
                    negone=nothing) -> T_{n+1}

One Chebyshev step `T_{n+1} = +(two * apply(H̃, T_n; apply_kwargs...), -T_{n−1};
add_kwargs...)`, with `negone * T_{n−1}` for `-T_{n−1}` when `negone` is a number,
then `truncate!(T_{n+1}; post_kwargs...)` unless `post_kwargs` is empty. The step of
`chebyshev_foreach`; `get_exciton_cheb_convergence_gpu` calls it directly for its
two recursions run in lockstep.
"""
function _chebyshev_step(Htilde::MPO, Tn, Tnm1;
                         apply_kwargs::NamedTuple,
                         add_kwargs::NamedTuple,
                         post_kwargs::NamedTuple       = (;),
                         two::Number                   = 2,
                         negone::Union{Nothing,Number} = nothing)
    Tnext = +(two * apply(Htilde, Tn; apply_kwargs...),
              (negone === nothing ? -Tnm1 : negone * Tnm1); add_kwargs...)
    isempty(post_kwargs) || ITensorMPS.truncate!(Tnext; post_kwargs...)
    return Tnext
end

# The truncation keywords one stage of the step receives: the names listed, out of
# :cutoff and :maxdim, with the values of `cutoff` and `maxdim`.
function _trunc_kwargs(names::Tuple, maxdim, cutoff)
    vals = map(names) do s
        s === :cutoff ? cutoff :
        s === :maxdim ? maxdim :
        throw(ArgumentError("truncation parameters are :cutoff and :maxdim, got $(repr(s))"))
    end
    return NamedTuple{names}(vals)
end


# ============================================================
# 3. Cached Chebyshev MPO recursion: KPM_Tn
# ============================================================

"""
    KPM_Tn(H_mpo::MPO, N::Int, sites; scale=nothing, center=0.0, identity_mpo=nothing,
           maxdim=40, dmrg_nsweeps=5, dmrg_maxdim=[10, 20, 40], dmrg_linkdim=4,
           cutoff=1e-8, verbose=true) -> (Tn_list, scale, center)

Build the list of Chebyshev MPOs `T_n((H−center·I)/scale)` for `n = 0…N`.

## Scale / center arguments
- If `scale=nothing` (default): spectral bounds estimated automatically via
  `_estimate_spectral_bounds` (two short DMRG runs with `dmrg_nsweeps`,
  `dmrg_maxdim`, `dmrg_linkdim`); the estimate replaces `center` too.
- If `scale` is provided: used directly; `center` defaults to `0.0` but can be
  set explicitly for non-symmetric spectra.

## Other keywords
- `identity_mpo` : the MPO used as `I` (in the shift and as `T_0`). Default
  `nothing` = `MPO(sites, "Id")`; the `TBHamiltonian` method passes
  `physical_projector(H)`.
- `maxdim`, `cutoff` : truncation of each recursion step. Defaults `40`, `1e-8`.
- `verbose` : print the bond dimension every 5 orders and at the last. Default `true`.

## High-level overload
Pass a `TBHamiltonian` as the first argument to skip manual rescaling entirely:
    Tn, scale, center = KPM_Tn(H, Ncheb; maxdim=100)
`H.scale` and `H.center` are computed lazily on the first call and cached.

## Return value
Returns `(Tn_list, scale, center)`.  To convert a physical energy ω:
    ω_r = (ω − center) / scale  ∈ (−1, 1)
"""
function KPM_Tn(H_mpo::MPO, N::Int, sites;
                scale::Union{Real, Nothing} = nothing,
                center::Real       = 0.0,
                identity_mpo::Union{MPO,Nothing} = nothing,
                maxdim::Int        = 40,
                dmrg_nsweeps::Int  = 5,
                dmrg_maxdim        = [10, 20, 40],
                dmrg_linkdim::Int  = 4,
                cutoff::Real       = 1e-8,
                verbose::Bool    = true)

    # ── Spectral bounds ───────────────────────────────────────────────────
    if isnothing(scale)
        scale, center = _estimate_spectral_bounds(H_mpo, sites;
                             dmrg_nsweeps = dmrg_nsweeps,
                             dmrg_maxdim  = dmrg_maxdim,
                             dmrg_linkdim = dmrg_linkdim)
    end

    # ── Scaled Hamiltonian: (H − center·I) / scale ────────────────────────
    I_mpo   = isnothing(identity_mpo) ? MPO(sites, "Id") : copy(identity_mpo)
    Ham_n   = _scaled_hamiltonian(H_mpo, scale, center, I_mpo; cutoff = cutoff)

    # ── Chebyshev recursion T_0 = I,  T_1 = H_scaled,  T_k = 2H·T_{k-1} − T_{k-2}
    Tn_list = MPO[]
    chebyshev_foreach(Ham_n, I_mpo, N + 1; T1 = Ham_n, maxdim = maxdim, cutoff = cutoff,
                      apply_trunc = (:cutoff,), add_trunc = (:maxdim,),
                      post_trunc = (:cutoff,)) do n, T_n
        push!(Tn_list, T_n)
        # print info every 5 orders and at the end
        verbose && n >= 2 && ((n + 1) % 5 == 0 || n == N) &&
            println("Computed T_$n with maxlinkdim = ", ITensorMPS.maxlinkdim(T_n))
    end

    return Tn_list, scale, center
end


"""
    KPM_Tn(H::TBHamiltonian, Ncheb::Int; mode=:mpo, psi0=nothing,
           maxdim=40, cutoff=1e-8, dmrg_nsweeps=5, dmrg_maxdim=[10, 20, 40],
           dmrg_linkdim=4, verbose=false)
        -> (Tn_list, scale, center)

High-level Chebyshev expansion for a `TBHamiltonian`.

Lazily determines `H.scale` and `H.center` via DMRG if not already set (the
`dmrg_*` keywords, see `_ensure_scale!`), builds the rescaled Chebyshev list with
`physical_projector(H)` as the identity, caches the result on `H`, and returns
`(Tn_list, H.scale, H.center)`. `maxdim`, `cutoff` and `verbose` are passed to the
raw-MPO method (`verbose` defaults to `false` here).

**`mode` keyword**

| `mode` | What is cached | Used by |
|--------|---------------|---------|
| `:mpo` (default) | MPO list `{T_n(H̃)}` in `H._tn_cache` | `get_ldos` (`mode=:diag`, `:mpo`), `get_ldos_spectrum`, `get_density(…; method=:kpm)` |
| `:mps` | MPS list `{T_n(H̃)|ψ₀⟩}` in `H._tn_mps_cache` | `get_ldos(…; mode=:mps, psi0=…)` |

`mode=:mps` requires `psi0` (a reference MPS).  The MPS pathway is more
memory-efficient when a single reference state is sufficient.

**Overview of the four KPM pathways**

```
Pathway 1 — MPO × MPO cache  [legacy / rarely used]
  KPM_Tn(H, Ncheb; mode=:mpo)           # build and cache {T_n(H̃)} MPOs
  → get_ldos_spectrum(H, ωlist)          # all ω in one pass → Vector{MPS}
  → get_ldos(H, ω; mode=:diag)           # single ω → diagonal site-LDOS MPS
  → get_ldos(H, ω; mode=:mpo)            # single ω → full off-diagonal MPO

  ⚠ Bond dimension χ_T grows at each MPO × MPO step; memory scales as
  O(Ncheb × χ_T²).  Prefer Pathways 3 or 4 unless the cache is reused for
  multiple downstream calls.  Kept mainly for legacy compatibility.

Pathway 2 — MPS cache  [legacy / fixed reference state]
  KPM_Tn(H, Ncheb; mode=:mps, psi0=ψ₀)  # cache {T_n(H̃)|ψ₀⟩} MPS on H
  → get_ldos(H, ω; mode=:mps, psi0=ψ₀)  # μₙ = ⟨ψ₀|T_n(H̃)|ψ₀⟩ → scalar

  Propagates a single reference MPS and stores the full trajectory
  {|φ_n⟩ = T_n(H̃)|ψ₀⟩} for repeated re-use across many energy queries on the
  same state.  Kept for advanced workflows; the public LDOS helpers below avoid
  storing this cache.

Pathway 3 — Online MPO × MPO  [k-space and spatial spectral functions]
  get_bands(H, Ncheb, D, ωlist; …)       # k-resolved A(k,ω), QFT-conjugated
  get_ldos_spatial(H, Ncheb, ωlist; …)   # real-space LDOS heatmap (default mode)

  Runs the MPO × MPO Chebyshev recursion online with no prior KPM_Tn call;
  only 3 MPOs alive at a time (truncated after each step).  Preferred for
  computing band structures and spatial LDOS over many positions simultaneously.

Pathway 4 — Online MPO × MPS  [single-particle default, most memory-efficient]
  get_ldos_online(H, Ncheb, X, ωlist; …) # LDOS at one site, all ω
  get_ldos_spatial(H, Ncheb, ωlist; mode=:mps; …)  # per-position MPS recursion
  get_exciton_ldos_spatial(H, Ncheb, ωlist; …)      # bound-pair exciton LDOS
  get_exciton_ldos(H, X, ωlist; …)                  # one-position wrapper
  get_dos_stochastic(H, Ncheb, ωlist; …) # stochastic trace DOS

  Propagates MPS states rather than full MPOs: only 3 MPS alive per sample/site.
  For single-particle problems this is almost always the best choice —
  memory cost is O(χ_H × χ_ψ) instead of O(χ_T²).  get_dos_stochastic
  applies this with random initial states for a stochastic trace estimate.
```

All pathways share the same KPM kernel and normalization.
Aux-DOF projection keywords (`spin_proj`, `nambu_proj`, `layer_proj`,
`sublat_proj` and their sector selectors) are accepted by `get_bands`,
`get_ldos_spatial`, `get_ldos_online` and `get_dos_stochastic` (Pathways 3 and 4).
The cached Pathways 1 and 2 (`get_ldos`, `get_ldos_spectrum`) and the exciton
LDOS do not expose them.
"""
function KPM_Tn(H::TBHamiltonian, Ncheb::Int;
                mode::Symbol                  = :mpo,
                psi0::Union{MPS, Nothing}     = nothing,
                maxdim::Int                   = 40,
                cutoff::Real                  = 1e-8,
                dmrg_nsweeps::Int             = 5,
                dmrg_maxdim                   = [10, 20, 40],
                dmrg_linkdim::Int             = 4,
                verbose::Bool                 = false)
    _ensure_scale!(H; dmrg_nsweeps=dmrg_nsweeps,
                      dmrg_maxdim=dmrg_maxdim,
                      dmrg_linkdim=dmrg_linkdim)
    if mode == :mpo
        Tn, _, _ = KPM_Tn(H.mpo, Ncheb, H.sites;
                           scale    = H.scale,
                           center   = H.center,
                           identity_mpo = physical_projector(H),
                           maxdim   = maxdim,
                           cutoff   = cutoff,
                           verbose  = verbose)
        H._tn_cache = Tn
    elseif mode == :mps
        psi0 === nothing && error("KPM_Tn with mode=:mps requires the psi0 keyword argument")
        Tn, _, _ = KPM_Tn_mps(H.mpo, Ncheb, psi0, H.sites;
                               scale    = H.scale,
                               center   = H.center,
                               identity_mpo = physical_projector(H),
                               maxdim   = maxdim,
                               cutoff   = cutoff,
                               verbose  = verbose)
        H._tn_mps_cache = Tn
    else
        error("Unknown KPM mode: $mode. Choose :mpo or :mps")
    end
    H._tn_Ncheb = Ncheb
    return Tn, H.scale, H.center
end


# ============================================================
# 4. Cached Chebyshev MPS recursion: KPM_Tn_mps
# ============================================================

"""
    KPM_Tn_mps(H_mpo::MPO, N::Int, psi0::MPS, sites; scale=nothing, center=0.0,
               identity_mpo=nothing, maxdim=40, dmrg_nsweeps=5,
               dmrg_maxdim=[10, 20, 40], dmrg_linkdim=4, cutoff=1e-8, verbose=true)
    -> (Tn_mps_list, scale, center)
    KPM_Tn_mps(H::TBHamiltonian, N::Int, psi0::MPS; maxdim=40, cutoff=1e-8,
               dmrg_nsweeps=5, dmrg_maxdim=[10, 20, 40], dmrg_linkdim=4,
               verbose=false)
    -> (Tn_mps_list, H.scale, H.center)

MPS-based Chebyshev expansion. Instead of storing Chebyshev MPOs T_n(H) (as
`KPM_Tn` does), this builds the projected MPS states

    |φ_n⟩ = T_n((H − center·I)/scale) |ψ₀⟩,   n = 0 … N

via the three-term recurrence

    |φ₀⟩ = |ψ₀⟩,   |φ₁⟩ = H̃|ψ₀⟩,   |φ_k⟩ = 2H̃|φ_{k-1}⟩ − |φ_{k-2}⟩.

This is more memory-efficient than the full MPO version when only a single
reference state is needed (e.g. site-resolved LDoS). Moments and spectral
weights are then obtained as `inner(ref_mps, Tn_mps_list[n+1])`.

`psi0` is normalised internally. `scale`/`center` follow the same convention as
`KPM_Tn`: if `scale=nothing` the spectral bounds are estimated via DMRG.
`identity_mpo`, `maxdim` and `cutoff` are as in `KPM_Tn`; `verbose` prints the bond
dimension every 10 orders and at the last.
Returns `(Tn_mps_list, scale, center)` where `Tn_mps_list[n+1]` = |φ_n⟩.

The `TBHamiltonian` method fills `H.scale`/`H.center` on demand (`_ensure_scale!`),
uses `physical_projector(H)` as the identity, stores the list in `H._tn_mps_cache`
(with `H._tn_Ncheb = N`) and returns `(Tn_mps_list, H.scale, H.center)`.
"""
function KPM_Tn_mps(H_mpo::MPO, N::Int, psi0::MPS, sites;
                    scale::Union{Real, Nothing} = nothing,
                    center::Real       = 0.0,
                    identity_mpo::Union{MPO,Nothing} = nothing,
                    maxdim::Int        = 40,
                    dmrg_nsweeps::Int  = 5,
                    dmrg_maxdim        = [10, 20, 40],
                    dmrg_linkdim::Int  = 4,
                    cutoff::Real       = 1e-8,
                    verbose::Bool    = true)

    # ── Spectral bounds ───────────────────────────────────────────────────
    if isnothing(scale)
        scale, center = _estimate_spectral_bounds(H_mpo, sites;
                             dmrg_nsweeps = dmrg_nsweeps,
                             dmrg_maxdim  = dmrg_maxdim,
                             dmrg_linkdim = dmrg_linkdim)
    end

    # ── Scaled Hamiltonian: (H − center·I) / scale ────────────────────────
    I_mpo = isnothing(identity_mpo) ? MPO(sites, "Id") : copy(identity_mpo)
    Ham_n = _scaled_hamiltonian(H_mpo, scale, center, I_mpo; cutoff = cutoff)

    # ── Chebyshev recursion T_0 = |ψ₀⟩,  |T_1⟩ = H_scaled|ψ₀⟩,  |T_k⟩ = 2H_scaled|ψ_{k-1}⟩ − |ψ_{k-2}⟩
    psi0_n      = psi0 / norm(psi0)  # ensure normalisation
    Tn_mps_list = MPS[]
    chebyshev_foreach(Ham_n, psi0_n, N + 1; maxdim = maxdim, cutoff = cutoff) do n, T_n
        push!(Tn_mps_list, T_n)
        verbose && n >= 2 && ((n + 1) % 10 == 0 || n == N) &&
            println("Computed MPS T_$n with maxlinkdim = ", maxlinkdim(T_n))
    end

    return Tn_mps_list, scale, center
end

function KPM_Tn_mps(H::TBHamiltonian, N::Int, psi0::MPS;
                    maxdim::Int        = 40,
                    cutoff::Real       = 1e-8,
                    dmrg_nsweeps::Int  = 5,
                    dmrg_maxdim        = [10, 20, 40],
                    dmrg_linkdim::Int  = 4,
                    verbose::Bool    = false)
    _ensure_scale!(H; dmrg_nsweeps=dmrg_nsweeps,
                      dmrg_maxdim=dmrg_maxdim,
                      dmrg_linkdim=dmrg_linkdim)
    Tn_mps, _, _ = KPM_Tn_mps(H.mpo, N, psi0, H.sites;
                                scale     = H.scale,
                                center    = H.center,
                                identity_mpo = physical_projector(H),
                                maxdim    = maxdim,
                                cutoff    = cutoff,
                                verbose   = verbose)
    H._tn_mps_cache = Tn_mps
    H._tn_Ncheb     = N
    return Tn_mps, H.scale, H.center
end


# ============================================================
# 5. Online MPS Chebyshev recursion (shared by the LDOS, DOS and exciton solvers)
# ============================================================

"""
    _run_kpm_mps!(Ham_n, psi0, Ncheb, W, valid, accum;
                  weight=1.0, cutoff=1e-8, maxdim=100,
                  verbose=false, label="") -> Int

Online MPS Chebyshev KPM recursion.  Computes `μ_n = ⟨psi0|T_n(Ham_n)|psi0⟩`
for n = 1…Ncheb and accumulates `W[n,iω] × μ_n × weight` into `accum[iω]`
for each valid energy index.  Returns the `maxlinkdim` of the final state.
The recursion is `chebyshev_foreach` with its MPS defaults (`cutoff` and `maxdim`
on every product and sum). The GPU exciton LDOS and stochastic DOS call it with GPU
tensors (`Ham_n` and `psi0` uploaded, `cutoff` as a `Float64`).
"""
function _run_kpm_mps!(Ham_n::MPO, psi0::MPS, Ncheb::Int,
                        W::Matrix{Float64}, valid::Vector{Bool},
                        accum::Vector{Float64};
                        weight::Float64 = 1.0,
                        cutoff::Real    = 1e-8,
                        maxdim::Int     = 100,
                        verbose::Bool   = false,
                        label::String   = "")
    Nω = length(valid)
    function kpm_step!(phi, n)
        mu = real(inner(psi0, phi))
        for iω in 1:Nω
            valid[iω] || continue
            accum[iω] += W[n, iω] * mu * weight
        end
    end
    phi_last = chebyshev_foreach(Ham_n, psi0, Ncheb; maxdim=maxdim, cutoff=cutoff) do n, phi
        k = n + 1
        kpm_step!(phi, k)
        verbose && k >= 3 && (k % 10 == 0 || k == Ncheb) &&
            println(label, " step $k/$Ncheb  maxlinkdim=$(maxlinkdim(phi))")
    end
    return maxlinkdim(phi_last)
end
