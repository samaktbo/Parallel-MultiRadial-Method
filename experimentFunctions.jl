include("Problems/qcqpConstructor.jl")

function sampleQCQP(n, m, lambda, sigma, nu)
    best_centers = Array{Vector{Float64}}(undef, m+1)
    Ps = Array{Matrix{Float64}}(undef, m+1)
    qs = Array{Vector{Float64}}(undef, m+1)
    rs = Array{Float64}(undef, m+1)
    for j=1:m+1
        Gj = randn(n, n)
        Ps[j] = Gj' * Gj + lambda*Matrix(I, n, n)
        if j == 1
            qs[j] = sigma*randn(n)
        else
            qs[j] = randn(n)
        end
        rs[j] =  rand(Uniform(0, 1)) + nu
        best_centers[j] = -1 * (Ps[j] \ qs[j])
    end
    
    return Ps, qs, rs, best_centers
end

function makeQCQP(Ps, qs, rs, centers, m, n, tol)
    rad_params = Vector{quadFormConsGaugeParams}(undef, m+1)
    for j=1:m+1
        rad_params[j] = quadFormConsGaugeParamsInitializer(Ps[j], qs[j], rs[j], centers[j])
    end
    qcqp = constructRadialQCQPInstance(Ps[1], qs[1],rs[1], rad_params)
    return qcqp
end

function makeQCQP_sqGauges(Ps, qs, rs, centers, m, n, tol)
    rad_params = Vector{quadFormConsGaugeParams}(undef, m+1)
    for j=1:m+1
        rad_params[j] = quadFormConsGaugeParamsInitializer(Ps[j], qs[j], rs[j], centers[j])
    end
    inst = constructRadialQCQP_sqGaugesInstance(Ps[1], qs[1],rs[1], rad_params)
    return inst
end;

function getSamplingParameters(Ps, qs, rs, best_centers)
    mp1 = size(Ps)[1]
    As = Array{Matrix{Float64}}(undef, mp1)
    rhos = Vector{Float64}(undef, mp1)
    norms_of_Ps = Vector{Float64}(undef, mp1)
    for j = 1:mp1
        Ph = Symmetric(Ps[j])
        As[j] = sqrt(Ph)
        rhos[j] = sqrt(2*rs[j]- best_centers[j]'*qs[j])
        norms_of_Ps[j] = opnorm(Ps[j], 2)
    end
    return As, rhos, norms_of_Ps
end

function sampleFromHilbertBallBoundary(A, center, rho, n)
    # this function uniformly samples a point x satisfying 
    # |A*(x - center)| = rho > 0 and returns e and b, where e = A \ b
    
    dir = rand(Normal(0, 1), n)
    dir = dir / norm(dir)
    x = center + rho * (A \ dir)
    return x
end

function getNewCenter(bdry_point, P, norm_of_P, q, alpha)
    grad = -1 * (P*bdry_point + q)
    
    return bdry_point + (alpha / norm_of_P) * grad
end