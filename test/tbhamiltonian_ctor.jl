# Wrapped in a module so that its imports stay out of Main: the goldens pin error messages
# that print `typeof(H.position_space)`, which reads `TensorBinding.FibonacciPositionSpace`
# only while that name is not imported into Main.
module TBHamiltonianCtorTests

using TensorBinding, ITensors, ITensorMPS, Test
using TensorBinding: TBHamiltonian, BinaryPositionSpace, FibonacciPositionSpace,
                     get_Hamiltonian, kinetic_1d_nn

# The keyword constructor TBHamiltonian(; L, N, sites, mpo, …) replaced the positional
# "backward-compatible" overloads (13 to 17 and 20 arguments), which are gone.
@testset "TBHamiltonian keyword constructor" begin
    s   = siteinds("Qubit", 3)
    mpo = kinetic_1d_nn(3, s)

    # Only L, N, sites and mpo are required; the rest is "not set", caches empty.
    H = TBHamiltonian(; L=3, N=8, sites=s, mpo, scale=3)
    @test H.L == 3 && H.N == 8 && H.sites == s && H.mpo === mpo
    @test H.scale === 3.0 && H.center === 0.0
    @test H.aux_side === :pre && H.position_space isa BinaryPositionSpace
    @test all(isnothing, (H.geometry, H.geometry_uc, H.spin_s, H.nambu_s, H.layer_s,
                          H.sublattice_s, H.Lx, H.interaction_mpo, H.fock_mpo))
    @test H._tn_cache === nothing && H._tn_mps_cache === nothing &&
          H._tn_Ncheb == 0 && H._density_cache === nothing
    @test_throws UndefKeywordError TBHamiltonian(; L=3, N=8, sites=s)

    # Every other field can be set by name.
    sub   = Index(2, "Sublattice")
    g     = i -> [Float64(i)]
    space = FibonacciPositionSpace(MPO(s, "Id"))
    Hf = TBHamiltonian(; L=3, N=8, sites=[s; sub], mpo, geometry=g, geometry_uc=g,
                       scale=2.5, center=0.5, spin_s=sub, nambu_s=sub, layer_s=sub,
                       sublattice_s=sub, aux_side=:post, Lx=1, position_space=space,
                       interaction_mpo=mpo, fock_mpo=mpo)
    @test Hf.geometry === g && Hf.geometry_uc === g && Hf.scale === 2.5 && Hf.center === 0.5
    @test Hf.spin_s == Hf.nambu_s == Hf.layer_s == Hf.sublattice_s == sub
    @test Hf.aux_side === :post && Hf.Lx === 1 && Hf.position_space === space
    @test Hf.interaction_mpo === mpo && Hf.fock_mpo === mpo

    # A builder's Hamiltonian is its own keyword reconstruction.
    Hc = get_Hamiltonian("ssh_sublattice", (t=1.0, d=0.5); L=3)
    kw = (:L, :N, :sites, :mpo, :geometry, :geometry_uc, :scale, :center, :spin_s,
          :nambu_s, :layer_s, :sublattice_s, :aux_side, :Lx, :position_space,
          :interaction_mpo, :fock_mpo)
    Hk = TBHamiltonian(; (f => getfield(Hc, f) for f in kw)...)
    @test all(getfield(Hk, f) === getfield(Hc, f) for f in fieldnames(TBHamiltonian))

    # The positional overloads are gone; the all-fields constructor remains.
    @test_throws MethodError TBHamiltonian(3, 8, s, mpo, nothing, 2.0, 0.0,
                                           nothing, nothing, nothing, nothing, 0, nothing)
    @test_throws MethodError TBHamiltonian(3, 8, s, mpo, nothing, nothing, 2.0, 0.0,
                                           nothing, nothing, nothing, nothing, :pre,
                                           nothing, nothing, 0, nothing)
    Ha = TBHamiltonian((getfield(H, f) for f in fieldnames(TBHamiltonian))...)
    @test all(getfield(Ha, f) === getfield(H, f) for f in fieldnames(TBHamiltonian))
end

end # module TBHamiltonianCtorTests
