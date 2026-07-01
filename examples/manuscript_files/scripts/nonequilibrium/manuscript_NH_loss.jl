# Shared non-Hermitian loss profile for the APSOS AAH production scripts.

function apsos_commensurate_loss_b()
    return 1.5
end

function apsos_cusped_loss_profile(N_sites::Integer, gamma0::Real, loss_b::Real;
                                   loss_phase::Real = 0.0,
                                   loss_harmonics::Integer = 15)
    N_sites > 0 || error("N_sites must be positive, got $N_sites")
    loss_b > 0 || error("loss_b must be positive, got $loss_b")
    loss_harmonics >= 1 || error("loss_harmonics must be >= 1, got $loss_harmonics")

    period_sites = Float64(loss_b) * N_sites / 6
    odd_harmonics = collect(1:2:loss_harmonics)

    function loss_profile(n)
        theta = 2 * pi * (n - loss_phase) / period_sites
        gamma = 1.0
        for m in odd_harmonics
            gamma -= (8 / pi^2) * cos(m * theta) / m^2
        end
        return Float64(gamma0) * max(gamma, 0.0)
    end

    return loss_profile, period_sites
end

function apsos_tagnum(x::Real; sigdigits::Integer = 5)
    return replace(string(round(Float64(x); sigdigits=sigdigits)), "+" => "")
end
