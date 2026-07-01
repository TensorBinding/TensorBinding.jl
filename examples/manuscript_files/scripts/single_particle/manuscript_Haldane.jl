# APSOS_Haldane.jl — Haldane-Semenoff Hamiltonian builder for APSOS Triton scripts.
# Include after TensorBinding is loaded:
#
#   include("../../../src/TensorBinding.jl"); using .TensorBinding
#   include("APSOS_Haldane.jl")
#   H = build_APSOS_hamiltonian(Lx, Ly, t, t2, M, phi)

"""
    build_APSOS_hamiltonian(Lx, Ly, t, t2, M, phi; scale, maxdim, tol) -> TBHamiltonian

Honeycomb Hamiltonian with:
- NN hopping t
- NNN Haldane hopping with spatially modulated amplitude t2*cos(4π*ix/Nx)
  (period = Nx/2 unit cells along x; sign flip ≡ φ→-φ since φ=π/2)
- Uniform Semenoff mass ±M on sublattices A/B
"""
function build_APSOS_hamiltonian(Lx::Int, Ly::Int,
                                  t::Real, t2::Real, M::Real, phi::Real;
                                  scale::Real = 4.0,
                                  maxdim::Int = 200,
                                  tol::Real   = 1e-8)
    Nx = 2^Lx

    H = TensorBinding.get_Hamiltonian("honeycomb", (t=t,);
            L=Lx+Ly, Lx=Lx, Ly=Ly, scale=scale)

    haldane_phases = Dict(
        (1,0,1,1) => +1, (0,1,1,1) => -1, (1,-1,1,1) => -1,
        (1,0,2,2) => -1, (0,1,2,2) => +1, (1,-1,2,2) => +1)

    t2_profile(ix) = t2 * cos(4π * ix / Nx)

    TensorBinding.add_hopping_2D!(H,
        (ix, iy, dx, dy, fs, ts) ->
            t2_profile(ix) * exp(im * phi * get(haldane_phases, (dx,dy,fs,ts), 0));
        Lx=Lx, Ly=Ly, nn=2, maxdim=maxdim, tol=tol)

    TensorBinding.add_onsite!(H, +M; sublat=1)
    TensorBinding.add_onsite!(H, -M; sublat=2)

    return H
end
