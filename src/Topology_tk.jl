###### SSH MODEL

function get_x_op_SSH_quantics(L,sites)
    f(x) = -2^(L-2)+div(x + 1, 2)
    mpo = get_diagonal_mpo(L, sites, f)/2^L
    return mpo
end


function get_sz_quantics(L,sites)
    f(x) = (-1)^(x + 1)
    mpo = get_diagonal_mpo(L, sites, f)
    return mpo
end


function get_SSH_hamiltonian(L, sites, t1, t2)
    function f(x)
        x % 2 == 0 ? t2 : t1
    end
    mpo = get_diagonal_mpo(L, sites, f)
    Ham = kinetic_1d_nn_custom(L, sites, mpo)
    ITensorMPS.truncate!(Ham;cutoff=1e-8)
    return Ham
end



function get_C_op_MPO_SSH(L,sites, t1, t2; Nchebychev = 300, maxdim = 15)
    Ham = get_SSH_hamiltonian(L, sites, t1, t2)
    factor = 5
    H = Ham/factor
    Tnlist = KPM_Tn(H,Nchebychev,sites,maxdim = maxdim)
    Id_op = MPO(sites, "Id")
    P = get_density_from_Tn(Tnlist,Nchebychev,fermi=0.0,maxdim = maxdim)
    x_op = get_x_op_SSH_quantics(L,sites)
    sz = get_sz_quantics(L,sites)
    Q = Id_op - P
    
    T1 = apply(P, apply(x_op, Q))
    T2 = apply(Q, apply(x_op, P))
    C_op = apply(sz, T1+T2)
    return C_op
end


function W1D(L,sites, t1, t2, i; Nchebychev = 200, maxbonddim = 15)
    C_op = get_C_op_MPO_SSH(L,sites, t1, t2, Nchebychev = Nchebychev, maxdim = maxbonddim)
    
    function Cmarker(x)
        psi1 = binary_to_MPS(Int(2*x-1), L, sites)
        psi2 = binary_to_MPS(Int(2*x), L, sites)
        C1 = inner(psi1,apply(C_op,psi1))
        C2 = inner(psi2,apply(C_op,psi2))
        return (C1 + C2)
    end

    return Cmarker(i)*2^L
end



###### HALDANE MODEL

function chirality(r1, r2)
    # This assumes all NNN hoppings are via a triangle path on a 2D honeycomb
    δ = r2 .- r1
    θ = atan(δ[2], δ[1])  # angle from r1 to r2
    # Map angle into 0 to 2π
    θ = mod(θ, 2π)
    # Assign ν = ±1 depending on angular sector
    return if θ < π
        +1  # counter-clockwise
    else
        -1  # clockwise
    end
end

function haldane_hoppingf(r1, r2, i, j; t2 = 0.2, phi=pi/2, M=0.0)
    δ = r2 .- r1
    d = norm(δ)
    if isapprox(d, 0.0; atol=1e-3)
        return M*(-1)^i #factor of -1^i accounts for sublattice
    elseif isapprox(d, 1.0; atol=1e-8)
        return -1.0 #t1
    elseif isapprox(d, √3; atol=1e-3)
        ν = chirality(r1, r2)
        return t2*exp(-1im*phi*ν*(-1)^i) #factor of -1^i accounts for sublattice
    else
        return 0.0
    end
end


function get_pos_opND_quantics(L,sites,rs,d)
    f(x) = rs[Int(x),Int(d)]
    mpo = get_diagonal_mpo(L, sites, f)
    return mpo
end


function get_C_op_MPO_Haldane(L, sites, rs, t2, M)
    #truevals = [Haldaneij3(i, j) for i in 1:stepp:N, j in 1:stepp:N]
    function f(i,j)
            return haldane_hoppingf(rs[Int(i),:], rs[Int(j),:], Int(i), Int(j); t2=t2, M=M)
    end
    
    initial_positions = [] # Example position

    Ham = hopping2MPO(f, 2^L, sites; tol = 1e-8, initial_positions = initial_positions, type = ComplexF64)
    Ham = ITensorMPS.truncate!(Ham; maxdim = 15, cutoff=1e-8)
    println("Constructed Hamiltonian!")

    ef = 0.0
    factor = 10
    N1 = 300
    Tnlist = KPM_Tn(Ham/factor,N1,sites)
    Id_op = MPO(sites, "Id")

    println("Calculated KPM!")
    x_op = get_pos_opND_quantics(L,sites,rs,1)
    y_op = get_pos_opND_quantics(L,sites,rs,2)
    P = get_density_from_Tn(Tnlist,N1,ef)
    Q = Id_op - P
    T1 = apply(Q, apply(x_op, apply(P, apply(y_op, Q))))
    T1 = ITensorMPS.truncate!(T1; maxdim = 40, cutoff=1e-8)
    T2 = apply(P, apply(x_op, apply(Q, apply(y_op, P))))
    T2 = ITensorMPS.truncate!(T2; maxdim = 40, cutoff=1e-8)
    C_op = 2im*pi*(T1-T2)
    return C_op
end



function get_x_op_square(x_mid, L, sites, L_chain)
    f(x) = mod(x - 1, L_chain) - x_mid
    mpo = get_diagonal_mpo(L, sites, f)
    return mpo
end


function get_y_op_square(y_mid,L, sites, L_chain)
    f(x) =  div(x - 1, L_chain) - y_mid
    mpo = get_diagonal_mpo(L, sites, f)
    return mpo
end


function get_x_op_square_approx(x_mid, L, sites, L_chain,a)
    f(x) = mod(x - 1, L_chain) - x_mid
    g(x) = sin(f(x)/a)*a
    mpo = get_diagonal_mpo(L, sites, g)
    return mpo
end


function get_y_op_square_approx(y_mid,L, sites, L_chain, a)
    f(x) =  div(x - 1, L_chain) - y_mid
    g(x) = sin(f(x)/a)*a
    mpo = get_diagonal_mpo(L, sites, g)
    return mpo
end


function get_C_op_MPO_2D(H, L, sites, L_chain; fermi = 0, Nchebychev = 200, maxbonddim = 15)
    Ham = ITensorMPS.truncate!(H; maxdim = maxbonddim, cutoff=1e-8)
    factor = 10
    Tnlist = KPM_Tn(Ham/factor,Nchebychev,sites, maxdim = maxbonddim)
    Id_op = MPO(sites, "Id")

    println("Calculated KPM!")
    x_op = get_x_op_square(L_chain/2,L,sites,L_chain)
    y_op = get_y_op_square(L_chain/2,L,sites,L_chain)
    print("Got positions!")
    P = get_density_from_Tn(Tnlist, Nchebychev, fermi = fermi, maxdim = maxbonddim)
    print("Got density matrix!")
    Q = Id_op - P
    T1 = apply(Q, apply(x_op, apply(P, apply(y_op, Q, maxdim = maxbonddim)), maxdim = maxbonddim), maxdim = maxbonddim)
    T2 = apply(P, apply(x_op, apply(Q, apply(y_op, P, maxdim = maxbonddim)), maxdim = maxbonddim), maxdim = maxbonddim)
    C_op = 2im*pi*-(T1,T2,maxdim = maxbonddim)
    return C_op
end


function get_C_op_MPO_from_P(P, L, sites, L_chain; maxbonddim = 15, type = Float64)
    P = ITensorMPS.truncate!(P; maxdim = maxbonddim, cutoff=1e-8)
    factor = 10
    Id_op = MPO(sites, "Id")
    print("Got density matrix!")
    Q = Id_op - P
    function f(rmid)
        rmid = Int(rmid)
        xmid = mod(rmid-1, L_chain) 
        ymid = div(rmid-1, L_chain)
        x_op = get_x_op_square(xmid,L,sites,L_chain)
        y_op = get_y_op_square(ymid,L,sites,L_chain)
        T1 = apply(Q, apply(x_op, apply(P, apply(y_op, Q, maxdim = maxbonddim, cutoff=1e-5)), maxdim = maxbonddim, cutoff=1e-5), maxdim = maxbonddim, cutoff=1e-5)
        T2 = apply(P, apply(x_op, apply(Q, apply(y_op, P, maxdim = maxbonddim, cutoff=1e-5)), maxdim = maxbonddim, cutoff=1e-5), maxdim = maxbonddim, cutoff=1e-5)
        C_op = 2im*pi*-(T1,T2,maxdim = maxbonddim)
        C_op = ITensorMPS.truncate!(C_op, cutoff = 1e-5)
        rmid_mps = binary_to_MPS(Int(rmid), L, sites)
        res = inner(rmid_mps, apply(C_op,rmid_mps))
        println(real(res))
        return real(res)
    end

    xvals = range(0, (2^L)-1; length=2^(L-3))
    qtt, ranks, errors = quanticscrossinterpolate(Float64, f,  xvals ; tolerance=1e-2)
    return qtt
end

function C2D_Haldane(L,sites, rs, t2, M, i)
    C_op = get_C_op_MPO_Haldane(L,sites, rs, t2, M)
    
    function Cmarker(x)
        psi1 = binary_to_MPS(Int(x), L, sites)
        C = inner(psi1,apply(C_op,psi1))
        return C
    end
    return Cmarker(i)
end


function C_2D_avg_around_center(L, sites, rs, C_op, radius)
    mask = []
    for i in 1:length(rs[:,1])
        if norm(rs[i,:]) < radius
        push!(mask, i)
        end
    end

    Cavg = 0
    for i in mask
        psii = binary_to_MPS(Int(i), L, sites)
        Cavg = Cavg + inner(psii,apply(C_op,psii))/length(mask)
    end
    return Cavg
end


function get_Vortex_op_MPO_Haldane(L, sites, rs, t2, M)
    #truevals = [Haldaneij3(i, j) for i in 1:stepp:N, j in 1:stepp:N]
    function f(i,j)
            return haldane_hoppingf(rs[Int(i),:], rs[Int(j),:], Int(i), Int(j); t2=t2, M=M)
    end

    Ham = hopping2MPO(f, 2^L, sites; tol = 1e-8, type = ComplexF64)
    Ham = ITensorMPS.truncate!(Ham;cutoff=1e-8)
    println("Constructed Hamiltonian!")

    ef = 0.0
    factor = 10
    N1 = 200
    Tnlist = KPM_Tn(Ham/factor,N1,sites)
    Id_op = MPO(sites, "Id")

    println("Calculated KPM!")
    x_op = get_pos_opND_quantics(L,sites,rs,1)
    y_op = get_pos_opND_quantics(L,sites,rs,2)
    z_op = x_op + 1im*y_op
    zb_op = x_op - 1im*y_op
    P = get_density_from_Tn(Tnlist,N1,ef)
    Q = Id_op - P
    V_op = apply(P, apply(zb_op, apply(Q, apply(z_op, P))))
    return V_op
end