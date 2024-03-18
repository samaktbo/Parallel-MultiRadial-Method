struct quadFormConsGaugeParams
    e::Vector{Float64}  # interior point of quadratic constraint, must have r  - q'*e - 0.5*e'*P*e > 0
    M::Matrix{Float64}
    quadGrad::Vector{Float64}  # -P*e - q
    f_of_e::Float64  # r  - q'*e - 0.5*e'*P*e
    n::Int64  # dimension of the domain space
end

function quadFormConsGaugeParamsInitializer(
    P::Matrix{Float64},  # matrix for the quadratic form, must be positive definite
    q::Vector{Float64},  # vector for the linear part of the quadratic form
    r::Float64,  # constant part of the quadratic form
    e::Vector{Float64}  # interior point of quadratic constraint, must have r  - q'*e - 0.5*e'*P*e > 0
    )
    
    n = size(q)[1]
    Pe = P*e
    quadGrad = -1 * (Pe + q)
    f_of_e = r  - q'*e - 0.5*e'*Pe
    M = quadGrad*quadGrad' + 2*f_of_e*P
    return quadFormConsGaugeParams(e, M, quadGrad, f_of_e, n)
end

function quadFormConsGaugeOracle(
        x::Vector{Float64}, 
        params::quadFormConsGaugeParams,
        tol::Float64=10^(-7),
        ret_grad::Bool=False 
    )
    x_minus_e = x - params.e
    Mx_minus_e = params.M * x_minus_e
    quadPart = x_minus_e'*Mx_minus_e
    linearPart = params.quadGrad'*x_minus_e
    divisor = 2*params.f_of_e
    val = (sqrt(quadPart) - linearPart) / divisor
    if ret_grad
        if norm(x-params.e) > tol
            grad = (Mx_minus_e/sqrt(quadPart) - params.quadGrad) / divisor
        else
            val = 0.0
            grad = fill(0.0, params.n)
        end
        ret = (val, grad)
    else
        ret = (val, )
    end
    return ret
end

function quadFormConsGaugeSquareOracle(
        x::Vector{Float64}, 
        params::quadFormConsGaugeParams,
        tol::Float64=10^(-7),
        ret_grad::Bool=False 
    )
    root = quadFormConsGaugeOracle(x, params, tol, ret_grad)
    val = root[1]^2
    if ret_grad
        grad = 2 * root[1] * root[2]
        ret = (val, grad)
    else
        ret = (val, )
    end
    return ret
end

function quadFormRadDualOracle(
        x::Vector{Float64}, 
        params::quadFormConsGaugeParams,
        tau::Float64,
        tol::Float64=10^(-7),
        ret_grad::Bool=true
    )
    # rescale parameters
    M = tau^2 * params.M
    quadGrad = tau * params.quadGrad
    f_of_e = tau * params.f_of_e
    
    # make computations
    x_minus_e = x - params.e
    Mx_minus_e = M * x_minus_e
    quadPart = x_minus_e'*Mx_minus_e
    linearPart = quadGrad'*x_minus_e
    rootPart = sqrt(1 - 2*linearPart + quadPart)
    divisor = 2*f_of_e
    val = (1-linearPart + rootPart) / divisor
    if ret_grad
        grad = ((Mx_minus_e - quadGrad) /rootPart - quadGrad) / divisor
        ret = (val, grad)
    else
        ret = (val, )
    end
    return ret
end

function quadFormPrimalOracle(x, P, q, r, ret_grad)
    Px = P*x
    val = r  - q'*x - 0.5*x'*Px
    ret = (val, )
    if ret_grad
        grad = -1*(Px + q)
        ret = (val, grad)
    end
    return ret
end

struct radialQCQP <: problem_instance
    f::Function
    primal::Function
    objective_dual::Function
    identifiers::Vector{Function}
end

# build radial QCQP
function constructRadialQCQPInstance(
    P0::Matrix{Float64}, 
    q0::Vector{Float64}, 
    r0::Float64, 
    rad_params::Vector{quadFormConsGaugeParams},
    tol::Float64=10^(-7)
)
# create the primal function
primal = (x, ret_grad) -> quadFormPrimalOracle(x, P0, q0, r0, ret_grad)

# dual objects
m = size(rad_params)[1] - 1
dual = Array{Function}(undef, m+1)
identifiers = Array{Function}(undef, m)   
obj_dual = (y, tau, ret_grad) -> quadFormRadDualOracle(y, rad_params[1], tau, tol, ret_grad)
dual[1] = (y, tau, ret_grad) -> quadFormRadDualOracle(y, rad_params[1], tau, tol, ret_grad)
for i=1:m
    dual[i+1] = (y, tau, ret_grad) -> quadFormConsGaugeOracle(y, rad_params[i+1], tol, ret_grad)
    identifiers[i] = (y, ret_grad) -> quadFormConsGaugeOracle(y, rad_params[i+1], tol, ret_grad)
end
f = (y, tau) -> eval_max(dual, y, tau)
return radialQCQP(f, primal, obj_dual, identifiers)
end

### problem instance where the gauges are squared
struct radialQCQP_sqGauges <: problem_instance
    f::Function
    primal::Function
    objective_dual::Function
    identifiers::Vector{Function}
end

function constructRadialQCQP_sqGaugesInstance(
    P0::Matrix{Float64}, 
    q0::Vector{Float64}, 
    r0::Float64, 
    rad_params::Vector{quadFormConsGaugeParams},
    tol::Float64=10^(-7)
)
# create the primal function
primal = (x, ret_grad) -> quadFormPrimalOracle(x, P0, q0, r0, ret_grad)

# dual objects
m = size(rad_params)[1] - 1
dual = Array{Function}(undef, m+1)
identifiers = Array{Function}(undef, m)   
obj_dual = (y, tau, ret_grad) -> quadFormRadDualOracle(y, rad_params[1], tau, tol, ret_grad)
dual[1] = (y, tau, ret_grad) -> quadFormRadDualOracle(y, rad_params[1], tau, tol, ret_grad)
for i=1:m
    dual[i+1] = (y, tau, ret_grad) -> quadFormConsGaugeSquareOracle(y, rad_params[i+1], tol, ret_grad)
    identifiers[i] = (y, ret_grad) -> quadFormConsGaugeSquareOracle(y, rad_params[i+1], tol, ret_grad)
end
f = (y, tau) -> eval_max(dual, y, tau)
return radialQCQP_sqGauges(f, primal, obj_dual, identifiers)
end