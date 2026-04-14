
#################################### KPM


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
            T_k = +(2 * apply(Ham_n, T_k_minus_1;  cutoff = 1e-8) , -T_k_minus_2;  maxdim = maxdim, cutoff = 1e-8)
            T_k_minus_2 = T_k_minus_1 
            T_k_minus_1 =  T_k    
            push!(Tn_list,T_k)
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
        A = +(A,  2 *  Tn_list[n] * G_n(n) * jackson_kernel[n]; maxdim=maxdim, cutoff = 1e-8)
    end
    A /= (π* N)
    
    return  A
end


#################################### SQUARE

function generate_kin_u(sites, num_site)
    L = Int(log2(num_site))
    kinetic_1 = OpSum()
    for i in 1:L
        os = OpSum()
        os += 1,"sigma_plus",L-(i-1)

        for i in 1:L-i 
            os *=  ("Id",i) 
        end


        for i in L+2-i :L 
            os *=  ("sigma_minus",i) 
        end
        
        kinetic_1 += os
    end
    k_mpo_1 = MPO(kinetic_1,sites)
    return k_mpo_1
end

function generate_kin_d(sites, num_site)
    L = Int(log2(num_site))
    kinetic_2 = OpSum()
    for i in 1:L
        os = OpSum()
        os += 1,"sigma_minus",L-(i-1)

        for i in 1:L-i 
            os *=  ("Id",i) 
        end


        for i in L+2-i :L 
            os *=  ("sigma_plus",i) 
        end
        
        kinetic_2 += os
    end
 
    k_mpo_2 = MPO(kinetic_2,sites)
end

#all intra-row hopping
function intrachain_hopping(L_chain, num_site, sites; hopping = MPO(sites, "Id")) 
    L = Int(log2(num_site))
    break_mpo = break_chain(L_chain, L_chain, num_site, sites)
    k_mpo_1 = generate_kin_u(sites, num_site)
    hop_1 = apply(hopping, k_mpo_1)
    true_hop_1 = apply(hop_1, break_mpo)

    k_mpo_2 = generate_kin_d(sites, num_site)
    hop_2 = apply(hopping, k_mpo_2)
    true_hop_2 = apply(hop_2, break_mpo)

    k_mpo =  +(true_hop_1, true_hop_2;  cutoff = 1e-8)
    return k_mpo
end



function interchain_hopping_square(L_chain, num_site, sites; hopping = MPO(sites, "Id"), t=1)
    L = Int(log2(num_site))
    k_mpo_1 = generate_kin_u(sites, num_site)
    hop_1 = apply(hopping,k_mpo_1)
    K_mpo_1_true = apply(hop_1,arbitarty_offline(k_mpo_1,L_chain-1))
        
    
    k_mpo_2 = generate_kin_d(sites, num_site)
    hop_2 = apply(k_mpo_2, hopping)
    K_mpo_2_true = apply(arbitarty_offline(k_mpo_2,L_chain-1),hop_2)
    k_mpo = t*K_mpo_1_true + conj(t)*K_mpo_2_true
    return k_mpo
end



##############################


function interchain_hopping_square_2nd_plus(L_chain, num_site, sites; hopping = MPO(sites, "Id"), t2 = 1)
    L = Int(log2(num_site))
    break_mpo = break_chain(L_chain, L_chain, num_site, sites)
    K_mpo_1 = generate_kin_u(sites, num_site)
    K_mpo_1_broken = apply(break_mpo,K_mpo_1)
    hop_1 = apply(hopping, K_mpo_1_broken)
    K_shift_1 = arbitarty_offline(K_mpo_1,L_chain +1-1)
    K_mpo_1_true = apply(K_shift_1, hop_1)
    
    K_mpo_2 = generate_kin_d(sites, num_site)
    K_mpo_2_broken = apply(K_mpo_2, break_mpo)
    hop_2 = apply(K_mpo_2_broken, hopping)
    K_shift_2 = arbitarty_offline(K_mpo_2,L_chain +1-1)
    K_mpo_2_true = apply(K_shift_2, hop_2)
    k_mpo = t2*K_mpo_1_true + conj(t2)*K_mpo_2_true
    return k_mpo
end


function interchain_hopping_square_2nd_minus(L_chain, num_site, sites; hopping = MPO(sites, "Id"), t2 = 1)
    L = Int(log2(num_site))
    break_mpo = break_chain(1, L_chain, num_site, sites)
    K_mpo_1 = generate_kin_u(sites, num_site)
    K_mpo_1_broken = apply(break_mpo,K_mpo_1)
    hop_1 = apply(hopping,K_mpo_1_broken)
    K_shift_1 = arbitarty_offline(K_mpo_1,L_chain -1-1)
    K_mpo_1_true = apply(hop_1,K_shift_1)
    
    K_mpo_2 = generate_kin_d(sites, num_site)
    K_mpo_2_broken = apply(K_mpo_2,break_mpo)
    hop_2 = apply(K_mpo_2_broken, hopping)
    K_shift_2 = arbitarty_offline(K_mpo_2,L_chain -1-1)
    K_mpo_2_true = apply(K_shift_2, hop_2)
    k_mpo = t2*K_mpo_1_true + conj(t2)*K_mpo_2_true
    return k_mpo
end
