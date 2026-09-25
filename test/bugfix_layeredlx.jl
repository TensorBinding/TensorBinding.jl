using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test
using TensorBinding: get_Hamiltonian, bilayer_hamiltonian, multilayer_hamiltonian,
                     twisted_bilayer_hamiltonian, get_ldos_spatial, get_valley_operator,
                     prepend_layer_projector, fix_sites

# The layered builders left H.Lx = nothing, so 2D consumers fell back to
# Lx = H.L ÷ 2 (wrong for a rectangular 2^2 x 2^1 layer) or rejected H as 1D.
@testset "Layered builders set Lx" begin
    Lx, Ly = 2, 1
    @test bilayer_hamiltonian(:square, Lx, Ly).Lx == Lx
    @test multilayer_hamiltonian(:square, Lx, Ly, 3).Lx == Lx
    @test multilayer_hamiltonian(:honeycomb, Lx, Ly, 3; sublattice=true).Lx == Lx
    @test twisted_bilayer_hamiltonian(:square, Lx, Ly, 5.0).Lx == Lx

    Hb = bilayer_hamiltonian(:honeycomb, Lx, Ly; sublattice=true)
    @test Hb.Lx == Lx
    Hb.scale = 4.0
    ω  = [-0.5, 0.3]
    kw = (; layer_proj=true, proj_layer=1)
    # Full-resolution, sublattice-resolved layer-1 LDOS (1D sweep, does not use Lx)
    R = get_ldos_spatial(Hb, 20, ω; kw...)
    cell(ix, iy) = (c = ix + iy * 2^Lx; (R[:, 2c + 1] .+ R[:, 2c + 2]) ./ 2)
    # 2x2 block map of the 4x2 cell grid: each block sums two cells along x
    B = get_ldos_spatial(Hb, 20, ω; reduce=:block, num_x=2, num_y=2, kw...)
    @test B ≈ hcat([cell(2ixp, iyp) .+ cell(2ixp + 1, iyp)
                    for iyp in 0:1 for ixp in 0:1]...) atol=1e-10
    # Full 4x2 block map (was rejected: num_x <= 2^(H.L ÷ 2) = 2)
    B42 = get_ldos_spatial(Hb, 20, ω; reduce=:block, num_x=4, num_y=2, kw...)
    @test B42 ≈ hcat([cell(ix, iy) for iy in 0:1 for ix in 0:3]...) atol=1e-10

    # Valley operator of an AA bilayer = monolayer valley operator on each layer
    V  = get_valley_operator(Hb)
    Hm = get_Hamiltonian("honeycomb", 1.0; L=Lx + Ly, Lx=Lx, Ly=Ly)
    Vm = fix_sites(get_valley_operator(Hm), Hb.sites[2:end])
    V_ref = +(prepend_layer_projector(Vm, Hb.layer_s, 1),
              prepend_layer_projector(Vm, Hb.layer_s, 2); cutoff=1e-12)
    @test norm(V) > 0
    @test norm(V - V_ref) / norm(V_ref) < 1e-8
end
