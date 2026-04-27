# krylov_tk.jl — Green's function via vectorized linear solve
#
# Computes the retarded single-particle Green's function
#
#     G(ω) = (ω + iη − H)⁻¹
#
# by solving the linear system
#
#     [(ω + iη − H) ⊗ I] |G⟩⟩ = |I⟩⟩
#
# where |M⟩⟩ is the vectorized (MPS) representation of the matrix M on a 2L-site
# interleaved quantics chain (odd sites = row bits, even sites = column bits).
#
# Requires: interleave_mpo (RPA_tk.jl), custom_mpo (utils.jl).


"""
    _identity_vec_mps(L, sites2) -> MPS

Exact bond-2 MPS representing the identity matrix δ(i,j) in the interleaved
2L-site quantics encoding.  Odd sites carry row bits, even sites carry column
bits.  Each consecutive pair (row bit k, col bit k) contributes the state
|0⟩_row|0⟩_col + |1⟩_row|1⟩_col, giving bond dimension 2 within each pair and
bond dimension 1 between pairs.  No QTCI needed.
"""
function _identity_vec_mps(L::Int, sites2::Vector{<:Index})
    N = 2L
    @assert length(sites2) == N "sites2 must have length 2L = $(2L), got $(length(sites2))"
    tensors = Vector{ITensor}(undef, N)

    # Pre-create ALL bond indices before the loop so each Index object is
    # created exactly once and shared between the two tensors it connects.
    # Creating Index(...) inside the loop would give fresh unique IDs each
    # time, so adjacent pairs would never share an index and contractions
    # would accumulate dangling indices instead of canceling them.
    lnk_inner   = [Index(2, "Link,l=$(2k-1)") for k in 1:L]     # within pair k
    lnk_between = [Index(1, "Link,l=$(2k)")   for k in 1:L-1]    # between pairs k and k+1

    for k in 1:L
        s_row = sites2[2k - 1]
        s_col = sites2[2k]
        li    = lnk_inner[k]

        # Row tensor: encodes the row bit onto the inner bond
        if k == 1
            T_row = ITensor(ComplexF64, s_row, li)
            T_row[s_row => 1, li => 1] = 1.0
            T_row[s_row => 2, li => 2] = 1.0
        else
            lb = lnk_between[k - 1]
            T_row = ITensor(ComplexF64, lb, s_row, li)
            T_row[lb => 1, s_row => 1, li => 1] = 1.0
            T_row[lb => 1, s_row => 2, li => 2] = 1.0
        end

        # Col tensor: enforces col bit == row bit via the shared inner bond
        if k == L
            T_col = ITensor(ComplexF64, li, s_col)
            T_col[li => 1, s_col => 1] = 1.0
            T_col[li => 2, s_col => 2] = 1.0
        else
            rb = lnk_between[k]
            T_col = ITensor(ComplexF64, li, s_col, rb)
            T_col[li => 1, s_col => 1, rb => 1] = 1.0
            T_col[li => 2, s_col => 2, rb => 1] = 1.0
        end

        tensors[2k - 1] = T_row
        tensors[2k]     = T_col
    end

    return MPS(tensors)
end


"""
    get_green_krylov(H_mpo, sites, ω_phys; η, nsweeps, maxdim, cutoff,
                     ishermitian, tol, maxiter, krylovdim, verbose) -> MPO

Low-level: compute the retarded Green's function G(ω) = (ω + iη − H)⁻¹ for a
raw MPO `H_mpo` defined on `sites`, via the vectorized linear system

    [(ω + iη − H) ⊗ I] |G⟩⟩ = |I⟩⟩

See the `TBHamiltonian` overload for the full keyword-argument reference.
"""
function get_green_krylov(H_mpo::MPO, sites::Vector{<:Index}, ω_phys::Real;
                          η::Real           = 1e-2,
                          nsweeps::Int      = 12,
                          maxdim::Int       = 100,
                          cutoff::Real      = 1e-8,
                          ishermitian::Bool = false,
                          tol::Real         = 1e-10,
                          maxiter::Int      = 600,
                          krylovdim::Int    = 30,
                          verbose::Bool     = false)
    N      = length(sites)
    sites2 = siteinds("Qubit", 2N)

    z     = ComplexF64(ω_phys + im * η)
    ω_mpo = z * MPO(sites, "Id") - H_mpo
    Lop   = interleave_mpo(ω_mpo, sites2, 0)
    rhs   = _identity_vec_mps(N, sites2)
    x0    = deepcopy(rhs)

    verbose && println("Krylov GF: ω = $ω_phys + $(η)i  (N=$N, maxdim=$maxdim, nsweeps=$nsweeps)")

    sol = ITensorMPS.linsolve(Lop, rhs, x0;
                              nsweeps        = nsweeps,
                              maxdim         = maxdim,
                              cutoff         = cutoff,
                              updater_kwargs = (; ishermitian, tol, maxiter, krylovdim))
    return custom_mpo(sol, sites)
end


"""
    get_green_krylov(H::TBHamiltonian, ω_phys; η, nsweeps, maxdim, cutoff,
                     ishermitian, tol, maxiter, krylovdim, verbose) -> MPO

Compute the retarded Green's function

    G(ω) = (ω + iη − H)⁻¹

as an MPO by solving the vectorized linear system

    [(ω + iη − H) ⊗ I] |G⟩⟩ = |I⟩⟩

using `ITensorMPS.linsolve` (DMRG-like Krylov solver).  The Hamiltonian is used
unscaled — no KPM Chebyshev expansion required.

**Keyword arguments**
- `η`           : Lorentzian broadening. Default `1e-2`.
- `nsweeps`     : Number of DMRG sweeps for the linear solver. Default `12`.
- `maxdim`      : Maximum bond dimension of the solution MPS. Default `100`.
- `cutoff`      : SVD truncation cutoff. Default `1e-8`.
- `ishermitian` : Set `true` only when the shifted operator is Hermitian
                  (requires purely imaginary η = 0, not physical for GF). Default `false`.
- `tol`         : Krylov solver convergence tolerance. Default `1e-10`.
- `maxiter`     : Maximum Krylov iterations per site. Default `600`.
- `krylovdim`   : Krylov subspace dimension. Default `30`.
- `verbose`     : Print progress messages. Default `false`.

**Usage**
```julia
G = get_green_krylov(H, ω; η=0.05, nsweeps=20, maxdim=200)
dos  = -imag(tr(G)) / π                              # total DoS
ldos = real(inner(psi_i, apply(G, psi_i)))           # LDoS at site i
gij  = inner(psi_i, apply(G, psi_j))                 # off-diagonal element
```
"""
function get_green_krylov(H::TBHamiltonian, ω_phys::Real;
                          η::Real           = 1e-2,
                          nsweeps::Int      = 12,
                          maxdim::Int       = 100,
                          cutoff::Real      = 1e-8,
                          ishermitian::Bool = false,
                          tol::Real         = 1e-10,
                          maxiter::Int      = 600,
                          krylovdim::Int    = 30,
                          verbose::Bool     = false)
    return get_green_krylov(H.mpo, H.sites, ω_phys;
                            η, nsweeps, maxdim, cutoff, ishermitian,
                            tol, maxiter, krylovdim, verbose)
end
