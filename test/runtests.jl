using TensorBinding
using ITensors
using Test

@testset "TensorBinding.jl" begin

    # shared setup — built once, reused across testsets
    H = get_Hamiltonian("chain_1d", 1.0; L=4)
    Tn, _, _ = KPM_Tn(H, 80)

    @testset "Construction" begin
        @test H.L == 4
        @test H.N == 16
        @test H.mpo isa MPO
        @test length(H.sites) == 4

        H_ssh = get_Hamiltonian("ssh_sublattice", (t=1.0, d=0.5); L=3)
        @test H_ssh.N == 8
        @test length(H_ssh.sites) == 4   # 3 position qubits + 1 sublattice index
    end

    @testset "Hermiticity" begin
        for geom in ("chain_1d", "ssh")
            Hg = get_Hamiltonian(geom, 1.0; L=4)
            Hadj = swapprime(dag(Hg.mpo), 0, 1)
            @test norm(Hg.mpo - Hadj) / norm(Hg.mpo) < 1e-10
        end
    end

    @testset "KPM — T₀ sum rule" begin
        # T_0(H̃) = identity, so tr(T_0) = N
        @test real(tr(Tn[1])) ≈ H.N atol=1e-6
    end

    @testset "Density matrix" begin
        ρ = get_density_from_Tn(Tn, 80; fermi=0.0, maxdim=40)
        @test real(tr(ρ)) ≈ H.N / 2 atol=0.5
    end

    @testset "Purification idempotency" begin
        ρ0 = get_density_from_Tn(Tn, 80; fermi=0.0, maxdim=40)
        ρ  = mcweeny_purify(ρ0; maxdim=60, tol=1e-4)
        ρ2 = apply(ρ, ρ; maxdim=60, cutoff=1e-8)
        @test norm(ρ2 - ρ) / norm(ρ) < 0.01
    end

    @testset "LDOS" begin
        ldos = get_ldos(H, 0.0)
        @test ldos isa MPS
        @test real(inner(ldos, ldos)) > 0
    end

    @testset "Band structure" begin
        ω_vals = range(-1.5, 1.5; length=5)
        Ak = get_bands(H, 80, 1, ω_vals; num_x=4)
        @test Ak isa Matrix
        @test size(Ak, 1) == 5
        @test size(Ak, 2) == 4
        @test all(≥(0), real.(Ak))
    end

    @testset "SSH winding number" begin
        L = 4
        H_top = get_Hamiltonian("ssh_sublattice", (t=1.0, d=-0.5); L=L)
        W_top = get_W(H_top; method=:KPM, Nchebychev=100, maxdim=40, l=L, Λ=10)
        @test real(W_top(H_top.N ÷ 2 + 1)) ≈ 1.0 atol=0.1

        H_triv = get_Hamiltonian("ssh_sublattice", (t=1.0, d=0.5); L=L)
        W_triv = get_W(H_triv; method=:KPM, Nchebychev=100, maxdim=40, l=L, Λ=10)
        @test real(W_triv(H_triv.N ÷ 2 + 1)) ≈ 0.0 atol=0.1
    end

    @testset "SCF magnetic Hubbard" begin
        H_scf = get_Hamiltonian("chain_1d", 1.0; L=3)
        add_spin!(H_scf)
        res = get_scf(H_scf, 2.0, :magnetic; Ncheb=30, maxdim=30, maxiters=2, verbose=false)
        @test res.H_up.mpo isa MPO
        @test res.H_dn.mpo isa MPO
    end

end
