using TensorBinding, ITensors, ITensorMPS, LinearAlgebra, Test
using TensorBinding: get_Hamiltonian, get_bands, get_ldos_spatial, get_dos_stochastic,
                     add_spin!, add_zeeman!, add_superconductivity!, project_aux, aux_site,
                     TBHamiltonian
const TBA = TensorBinding

# Regression tests for the auxiliary-DOF projection bugs of the 2026-09 characterization
# sweep (docs/dev/REORGANISATION_TODO.md, "Crashes and unhelpful errors", "Minor / API"
# and "Found by the Tier 2 aux and density kernels"). The references are dense: the
# KPM operator Σ_n W[n, ω] T_n(H̃) of the full dense matrix, cut into its aux-sector
# blocks, and (for the bands) each block read out by the package's QFT on
# position-only MPOs.

chain4(L) = get_Hamiltonian("chain_1d", 1.0; L=L)
dmat4(M::MPO) = TBA._mpo_dense_matrix(M)

# Exact MPO of the dense matrix A on `sites` (site 1 the most significant digit, as
# _mpo_dense_matrix reads it).
function dense_mpo4(A, sites)
    d = dim.(sites)
    T = ITensor(ComplexF64.(reshape(A, reverse(d)..., reverse(d)...)),
                reverse(prime.(sites))..., reverse(sites)...)
    return MPO(T, sites; cutoff=1e-15)
end

# The block of A (states on `sites`, site 1 most significant) whose digit at each
# aux site of `fix` (Index => sector) is that sector; the kept states stay in order,
# so the block lives on the other sites, in their order.
function sector_block4(A, sites, fix)
    d      = dim.(sites)
    stride = [prod(d[j+1:end]; init=1) for j in eachindex(d)]
    js     = [(findfirst(==(s), sites), σ) for (s, σ) in fix]
    keep   = [n for n in 0:prod(d)-1 if all((n ÷ stride[j]) % d[j] == σ - 1 for (j, σ) in js)]
    return A[keep .+ 1, keep .+ 1]
end

# Σ_n W[n, iω] T_{n-1}(H̃) for every energy, from the dense recursion on the full space.
function dense_kpm4(H, Ncheb, ωs)
    A = dmat4(H.mpo)
    _, W, denom, valid = TBA._kpm_energy_grid(H, Ncheb, ωs)
    Ht = (A - H.center * I) / H.scale
    Ts = [Matrix{ComplexF64}(I, size(A)), Ht]
    while length(Ts) < Ncheb
        push!(Ts, 2 * Ht * Ts[end] - Ts[end-1])
    end
    G = [sum(W[n, iω] * Ts[n] for n in 1:Ncheb) for iω in eachindex(ωs)]
    return G, denom, valid
end

# A(k, ω) summed over the sector blocks `sectors` (each a `fix` list), 1D k-groups.
function dense_bands4(H, Ncheb, ωs, sectors, k_groups)
    G, denom, valid = dense_kpm4(H, Ncheb, ωs)
    ps  = TBA._pos_sites(H)
    out = zeros(length(ωs), length(k_groups))
    for iω in eachindex(ωs), fix in sectors
        B = sector_block4(G[iω], H.sites, fix)
        d = TBA.extract_diagonal_to_mps(TBA.conjugate_by_qft(dense_mpo4(B, ps);
                                                            tol=1e-14, maxdim=1000))
        for (ik, xs) in enumerate(k_groups)
            out[iω, ik] += sum(TBA._eval_diag_mps(d, x) for x in xs) / length(xs)
        end
    end
    for iω in eachindex(ωs)
        valid[iω] && (out[iω, :] ./= denom[iω])
    end
    return out
end

# LDOS(ω, x) summed over the sector blocks `sectors`, x = 1…N in probe order.
function dense_ldos4(H, Ncheb, ωs, sectors)
    G, denom, valid = dense_kpm4(H, Ncheb, ωs)
    out = zeros(length(ωs), H.N)
    for iω in eachindex(ωs), fix in sectors
        out[iω, :] .+= real.(diag(sector_block4(G[iω], H.sites, fix)))
    end
    for iω in eachindex(ωs)
        valid[iω] && (out[iω, :] ./= denom[iω])
    end
    return out
end

errmsg4(f) = try f(); nothing catch e; sprint(showerror, e) end

const NC4 = 10
const W4  = [-2.2, -0.9, 0.0, 0.6, 1.7, 3.5]
const KG4 = TBA.kspace_sampling_plan(3, 1; num_x=8).k_groups
const KW4 = (maxdim=200, cutoff=1e-12)

# A Zeeman-split chain, the spin index prepended or postpended.
function zeeman_chain4(position)
    H = chain4(3)
    add_zeeman!(H, 0.4; position)
    H.scale, H.center = 3.0, -0.05
    return H
end

# BdG with spin: [nambu, spin, pos…] (:pre) or [pos…, spin, nambu] (:post).
function bdg_spin_chain4(position)
    H = chain4(3)
    add_spin!(H; position)
    add_zeeman!(H, 0.3)
    add_superconductivity!(H, 0.25)
    H.scale, H.center = 3.2, 0.0
    return H
end

@testset "Postpended spin: every projection uses the side of the index" begin
    # _aux_projection hard-coded side=:pre for the spin, so project_aux contracted the
    # first position tensor with the spin projector: get_bands segfaulted and
    # get_ldos_spatial(:mpo) threw deep inside ITensors.
    Hq = zeeman_chain4(:pre)
    Hp = zeeman_chain4(:post)
    s  = Hp.spin_s
    @test Hp.sites[end] == s && Hq.sites[1] == Hq.spin_s
    @test aux_site(Hp, :spin) == (s, :post)

    both = [[s => 1], [s => 2]]
    A = get_bands(Hp, NC4, 1, W4; num_x=8, KW4...)
    @test A ≈ dense_bands4(Hp, NC4, W4, both, KG4) atol=1e-8
    @test A ≈ get_bands(Hq, NC4, 1, W4; num_x=8, KW4...) atol=1e-8
    Adn = get_bands(Hp, NC4, 1, W4; num_x=8, proj_s=2, KW4...)
    @test Adn ≈ dense_bands4(Hp, NC4, W4, [[s => 2]], KG4) atol=1e-8
    # the low-level method takes the side as spin_side
    Alow = get_bands(Hp.mpo, Hp.scale, Hp.center, Hp.sites, NC4, 1,
                     TBA._rescaled_energies(Hp, W4);
                     spin_proj=true, spin_s_aux=s, spin_side=:post, num_x=8, KW4...)
    @test Alow ≈ A atol=1e-12

    ref = dense_ldos4(Hp, NC4, W4, both)
    for mode in (:mpo, :mps)
        @test get_ldos_spatial(Hp, NC4, W4; mode, KW4...) ≈ ref atol=1e-8
    end
    @test get_ldos_spatial(Hp, NC4, W4; proj_s=1, KW4...) ≈
          dense_ldos4(Hp, NC4, W4, [[s => 1]]) atol=1e-8
    # the stochastic DOS uses probe states, which were right already
    @test get_dos_stochastic(Hp, NC4, W4; N_sample=4, seed=3, spin_proj=true, KW4...) ≈
          get_dos_stochastic(Hq, NC4, W4; N_sample=4, seed=3, spin_proj=true, KW4...) atol=1e-8

    # A projection from the wrong end is an error that names the index, not a segfault.
    msg = errmsg4(() -> project_aux(Hp.mpo, s, 1; side=:pre))
    @test msg !== nothing && occursin("Spin", msg) && occursin("tensor 4 of 4", msg)
    msg = errmsg4(() -> get_bands(Hp.mpo, Hp.scale, Hp.center, Hp.sites, NC4, 1, [0.0];
                                  spin_proj=true, spin_s_aux=s, num_x=2))
    @test msg !== nothing && occursin("side=:pre", msg)
end

@testset "get_bands default num_x; aux indices must be projected" begin
    # num_x = 60 exceeded the 2^L momenta of a chain with L < 6 (ilinspace assertion).
    H = chain4(3); H.scale = 2.5
    A = get_bands(H, 8, 1, W4)
    @test size(A) == (length(W4), 8)
    @test A == get_bands(H, 8, 1, W4; num_x=8)
    @test size(get_bands(H, 8, 1, W4; xmin=2, xmax=5)) == (length(W4), 4)
    @test_throws AssertionError get_bands(H, 8, 1, W4; num_x=60)   # explicit: as before
    @test TBA._bands_num_x(nothing, 1, 6, 0, nothing, nothing) == 60
    @test TBA._bands_num_x(nothing, 1, 3, 1, 6, nothing) == 6
    @test TBA._bands_num_x(nothing, 2, 3, 0, nothing, nothing) == 60   # the 2D planner clamps
    @test TBA._bands_num_x(nothing, 1, 3, 0, nothing, [:G, :X]) == 60  # points per segment
    @test TBA._bands_num_x(9, 1, 3, 0, nothing, nothing) == 9

    # The low-level get_bands counted a given sublat_s (nambu_s, layer_s) out of the
    # position qubits but, with its flag off, left the site on the MPO, so the QFT took
    # it for one more momentum bit.
    Hs = get_Hamiltonian("ssh_sublattice", (t=1.0, d=0.3); L=3)
    Hs.scale = 2.8
    ωr = TBA._rescaled_energies(Hs, W4)
    low(; kw...) = get_bands(Hs.mpo, Hs.scale, Hs.center, Hs.sites, NC4, 1, ωr; num_x=4, kw...)
    msg = errmsg4(() -> low(sublat_s=Hs.sublattice_s))
    @test msg !== nothing && occursin("ArgumentError", msg) && occursin("sublat_proj", msg)
    @test low(sublat_s=Hs.sublattice_s, sublat_proj=true) ≈
          get_bands(Hs, NC4, 1, W4; num_x=4)
    Hb = chain4(3)
    add_superconductivity!(Hb, 0.3)
    Hb.scale = 3.0
    msg = errmsg4(() -> get_bands(Hb.mpo, Hb.scale, 0.0, Hb.sites, NC4, 1, [0.0];
                                  nambu_s=Hb.nambu_s, num_x=2))
    @test msg !== nothing && occursin("ArgumentError", msg) && occursin("nambu_proj", msg)
end

@testset "BdG with spin: aux_site and the projections on both sides" begin
    Hq = bdg_spin_chain4(:pre)
    Hp = bdg_spin_chain4(:post)
    # aux_site(H, :spin) refused the spin index as interior on every BdG+spin model.
    @test aux_site(Hq, :spin) == (Hq.spin_s, :pre) && aux_site(Hq, :nambu) == (Hq.nambu_s, :pre)
    @test aux_site(Hp, :spin) == (Hp.spin_s, :post) && aux_site(Hp, :nambu) == (Hp.nambu_s, :post)
    @test Hp.sites[end-1:end] == [Hp.spin_s, Hp.nambu_s]
    # an index between position sites is still refused
    Hs = zeeman_chain4(:pre)
    ps = TBA._pos_sites(Hs)
    Hi = TBHamiltonian(Hs; sites=[ps[1]; Hs.spin_s; ps[2:end]])
    msg = errmsg4(() -> aux_site(Hi, :spin))
    @test msg !== nothing && occursin("interior position 2", msg)

    s, ν = Hp.spin_s, Hp.nambu_s
    all4 = [[ν => a, s => b] for a in 1:2 for b in 1:2]
    A = get_bands(Hp, NC4, 1, W4; num_x=8, KW4...)
    @test A ≈ dense_bands4(Hp, NC4, W4, all4, KG4) atol=1e-8
    @test A ≈ get_bands(Hq, NC4, 1, W4; num_x=8, KW4...) atol=1e-8
    @test get_bands(Hp, NC4, 1, W4; num_x=8, proj_nambu=1, proj_s=2, KW4...) ≈
          dense_bands4(Hp, NC4, W4, [[ν => 1, s => 2]], KG4) atol=1e-8
    @test get_ldos_spatial(Hp, NC4, W4; KW4...) ≈ dense_ldos4(Hp, NC4, W4, all4) atol=1e-8

    # The same rule places a layer index behind a prepended spin ([spin, layer, pos…]),
    # which aux_site(H, :layer) also refused as interior (every spectral method threw).
    H0 = chain4(3); layer = siteinds("Qubit", 1)[1]; sl = [layer; H0.sites]
    M  = kron(Matrix(1.0I, 2, 2), dmat4(H0.mpo)) + kron([0.0 1.0; 1.0 0.0], Matrix(0.3I, 8, 8))
    Hl = TBHamiltonian(H0; sites=sl, mpo=dense_mpo4(M, sl), layer_s=layer)
    add_spin!(Hl)
    Hl.scale, Hl.center = 3.2, 0.0
    @test aux_site(Hl, :layer) == (layer, :pre) && Hl.sites[1:2] == [Hl.spin_s, layer]
    sl4 = [[Hl.spin_s => a, layer => b] for a in 1:2 for b in 1:2]
    @test get_ldos_spatial(Hl, NC4, W4; KW4...) ≈ dense_ldos4(Hl, NC4, W4, sl4) atol=1e-8
    @test get_bands(Hl, NC4, 1, W4; num_x=8, KW4...) ≈
          dense_bands4(Hl, NC4, W4, sl4, KG4) atol=1e-8
end

@testset "_project_spin_sector keeps the sites after an inner spin site" begin
    # With the spin inside the aux block ([pos…, spin, nambu], [nambu, spin, pos…])
    # every site after it was dropped from the MPO while `sites` kept them.
    for position in (:post, :pre)
        H = bdg_spin_chain4(position)
        A = dmat4(H.mpo)
        for σ in 1:2
            P = TBA._project_spin_sector(H, σ)
            @test P.spin_s === nothing && length(P.sites) == 4
            @test TBA._mpo_ket_sites(P.mpo) == P.sites
            @test dmat4(P.mpo) ≈ sector_block4(A, H.sites, [H.spin_s => σ]) atol=1e-12
        end
    end
    # spin alone (first or last site): unchanged
    for position in (:pre, :post)
        H = zeeman_chain4(position)
        P = TBA._project_spin_sector(H, 2)
        @test TBA._mpo_ket_sites(P.mpo) == P.sites == TBA._pos_sites(H)
        @test dmat4(P.mpo) ≈ sector_block4(dmat4(H.mpo), H.sites, [H.spin_s => 2]) atol=1e-12
    end
end

@testset "Missing aux indices: the errors name the DOF" begin
    # project_aux(W, nothing, σ) said "sublat_proj=true requires sublat_s" for every DOF.
    H = chain4(3); H.scale = 2.5
    msg = errmsg4(() -> project_aux(H.mpo, nothing, 1))
    @test msg !== nothing && !occursin("sublat_proj", msg) && occursin("Index is nothing", msg)
    # A projected DOF without an index threw a TypeError inside the projection chain.
    for (kw, name) in ((:nambu_proj, "Nambu"), (:layer_proj, "layer"), (:sublat_proj, "sublattice"))
        msg = errmsg4(() -> get_bands(H, 4, 1, [0.0]; num_x=2, (kw => true,)...))
        @test msg !== nothing && occursin("$kw=true", msg) && occursin("no $name index", msg)
    end
    msg = errmsg4(() -> get_ldos_spatial(H, 4, [0.0]; spin_proj=true, mode=:mpo))
    @test msg !== nothing && occursin("spin_proj=true", msg) && occursin("no spin index", msg)
end

@testset "GPU twins: postpended spin and BdG with spin" begin
    cuda_functional = false
    if Base.find_package("CUDA") !== nothing
        try
            @eval using CUDA
            cuda_functional = CUDA.functional()
        catch
        end
    end
    if !cuda_functional
        @test_skip "CUDA.jl not functional"
    else
        for H in (zeeman_chain4(:post),
                  bdg_spin_chain4(:post))
            cpu = get_bands(H, NC4, 1, W4; num_x=8, KW4...)
            gpu = TBA.get_bands_gpu(H, NC4, W4; num_x=8, type=ComplexF64, KW4...)
            @test size(gpu) == size(cpu) && isapprox(gpu, cpu; atol=1e-8)
            @test isapprox(TBA.get_ldos_spatial_gpu(H, NC4, W4; type=ComplexF64, KW4...),
                           get_ldos_spatial(H, NC4, W4; KW4...); atol=1e-8)
        end
        # the default num_x is clamped as on the CPU
        H = chain4(3); H.scale = 2.5
        @test size(TBA.get_bands_gpu(H, 6, W4; type=ComplexF64)) == (length(W4), 8)
        msg = errmsg4(() -> TBA._project_aux_gpu(TBA._to_gpu(H.mpo, ComplexF64),
                                                 H.sites[2], 1; side=:post))
        @test msg !== nothing && occursin("_project_aux_gpu", msg)
    end
end
