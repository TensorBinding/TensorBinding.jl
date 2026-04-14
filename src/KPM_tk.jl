function KPM_Tn(H,N,sites;maxdim=40)
    Id_op = MPO(sites, "Id")
    Ham_n = H
    T_k_minus_2 = Id_op
    T_k_minus_1 = Ham_n   
    Tn_list = [T_k_minus_2,T_k_minus_1]

    for k in 1:N
        if k == 1
            T_k = T_k_minus_2
        elseif k == 2
            T_k = T_k_minus_1
        else
            T_k = +(2 * apply(Ham_n, T_k_minus_1;  cutoff = 1e-8) , -T_k_minus_2;  maxdim = maxdim)
            T_k = ITensorMPS.truncate!(T_k; cutoff = 1e-8)
            T_k_minus_2 = T_k_minus_1 
            T_k_minus_1 =  T_k
            push!(Tn_list,T_k)
            println(ITensorMPS.maxlinkdim(T_k))
        end
    end
    return Tn_list
end

function get_density_from_Tn(Tn_list,N;fermi=0,maxdim=40)  

    jackson_kernel = [(N - n) * cos(π * n / N) + sin(π * n / N) / tan(π / N) for n in 0:N-1]

    function G_n(n)
        if n == 1
            return acos(fermi)
        else
            return sin((n-1) * acos(fermi)) / (n-1)
        end
    end

    # Compute electronic density
    A = Tn_list[1] * G_n(1) * jackson_kernel[1] 
    for n in 2:N
        A = +(A,  2 *  Tn_list[n] * G_n(n) * jackson_kernel[n]; maxdim=maxdim)
        A = ITensorMPS.truncate!(A;cutoff=1e-8)
    end
    A /= (π* N)
    
    return  A
end

function get_Green_retarded_from_Tn(Tn_list, N, ω; η=1e-2, maxdim=40)

    # Jackson kernel for damping
    jackson_kernel = [(N - n) * cos(π * n / N) + sin(π * n / N) / tan(π / N)
                      for n in 0:N-1]

    # Chebyshev propagator term
    function G_n(n, ω, η)
        z = ω + 1im*η         # retarded Green’s function: +iη
        θ = acos(z)
        return -2im/(1+ ==(n-1,0)) * exp(-1im * (n-1) * θ) / (sqrt(1 - z^2))
    end

    # Build Green’s function expansion
    G = Tn_list[1] * G_n(1, ω, η) * jackson_kernel[1]
    for n in 2:N
        G = +(G, Tn_list[n] * G_n(n, ω, η) * jackson_kernel[n]; maxdim=maxdim)
        G = ITensorMPS.truncate!(G; cutoff=1e-8)
    end
    G /= N

    return G
end


function get_ldos_w_from_Tn(Tn_list,N,ω;maxdim=40)  

    jackson_kernel = [(N - n) * cos(π * n / N) + sin(π * n / N) / tan(π / N) for n in 0:N-1]

    function G_n(n)
       return cos((n - 1) * acos(ω))/(π* sqrt(1-ω^2))
    end

    # Compute electronic density
    A = Tn_list[1] * G_n(1) * jackson_kernel[1] 
    for n in 2:N
        A = +(A,  2 *  Tn_list[n] * G_n(n) * jackson_kernel[n]; maxdim=maxdim)
        A = ITensorMPS.truncate!(A;cutoff=1e-8)
    end
    A /= (π* N)
    
    return  A
end


function get_PH_from_Tn(Tn_list,N,ω;maxdim=40)  

    jackson_kernel = [(N - n) * cos(π * n / N) + sin(π * n / N) / tan(π / N) for n in 0:N-1]

    function G_n(n)
       return cos((n - 1) * acos(ω))/(π* sqrt(1-ω^2))
    end

    # Compute electronic density
    A = Tn_list[1] * G_n(1) * jackson_kernel[1] 
    for n in 2:N
        A = +(A,  2 *  Tn_list[n] * G_n(n) * jackson_kernel[n]; maxdim=maxdim)
        A = ITensorMPS.truncate!(A;cutoff=1e-8)
    end
    A /= (π* N)
    
    return  A
end


#for getting electron densities
function get_density_quantics(A,L)
    
    xvals = range(0, (2^L - 1); length=2^L)
    f(x) =  1 -  inner(random_mps(sites,to_binary_vector(Int(x),L))',A, random_mps(sites,to_binary_vector(Int(x),L)))
    qtt, ranks, errors = quanticscrossinterpolate(Float64, f,  xvals ; tolerance=1e-8)

    tt = TCI.tensortrain(qtt.tci)
    density_mps = ITensors.MPS(tt;sites)
  
    density_mpo = outer(density_mps',density_mps)
    for i in 1:L
        density_mpo.data[i] =  Quantics._asdiagonal(density_mps.data[i],sites[i])
    end
    
    return qtt,density_mpo,density_mps
end