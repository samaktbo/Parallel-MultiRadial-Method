using JuMP
using MosekTools
using MathOptInterface


struct fom <: algorithm
    initialize
    step
end;

# the following type must have fields instance for the problem instance
# and iterate for the current iterate produced by the algorithm
abstract type algorithmState end

# helper functions
function firstOrderOracle(
    p::problem_instance, 
    x::Vector{Float64}, 
    tau::Float64,
    ret_grad::Bool, 
    m::Int64, 
    n::Int64)
    h = Vector{Float64}(undef, m+1)
    if ret_grad
        J = Matrix{Float64}(undef, m+1, n) # to store gradients
        (h[1], J[1, :])= p.objective_dual(x, tau, true)
        for j = 1:m
            (h[j+1], J[j+1, :]) = p.identifiers[j](x, true)
        end
        ret = (h, J)
    else
        h[1]= p.objective_dual(x, tau, false)[1]
        for j = 1:m
            h[j+1] = p.identifiers[j](x, false)[1]
        end
        ret = (h, )
    end
    
    return ret
end;

# The universal fast gradient Method
mutable struct fastGradientMethodState <: algorithmState
    instance::problem_instance
    x0::Vector{Float64}
    xk::Vector{Float64}
    iterate::Vector{Float64}
    a::Float64
    L::Float64
    phi::Vector{Float64}
    m::Int64
    n::Int64
    counter::Int64
    burnout::Int64
end

function fastGradientMethodInitialization(
    x0::Vector{Float64},
    eps::Float64,
    inst::problem_instance,
    burnout::Int64
)
    m = size(inst.identifiers)[1]
    n = size(x0)[1]
    phi = fill(0.0, n)
    L = 1 # initial guess of Lipschitz constant
    counter=0
    return fastGradientMethodState(inst, x0, x0, x0, 0.0, L, phi, m, n, 0, burnout)
end

function fastGradientMethodStep(
    st::fastGradientMethodState,
    eps::Float64,
    tau::Float64,
    firstOrderOracle::Function,
    tol::Float64=10^(-9))

    # reset parameters if needed
    if st.counter == st.burnout
        st.x0 = copy(st.iterate)
        st.xk = copy(st.iterate)
        st.phi = fill(0.0, st.n)
        st.a = 0
        st.counter = 0
    end
    xk = st.xk
    yk = st.iterate
    Ak = st.a
    Lk = st.L
    phik = st.phi

    # make a step
    vk = st.x0 - phik
    ik  = 0 
    Lk1 = 2^ik *Lk
    ak1 = 1/(2*Lk1) + sqrt((1/(2*Lk1))^2 + Ak/Lk1)
    Ak1 = Ak + ak1
    tauk= ak1/Ak1
    xk1 = tauk*vk + (1-tauk)*yk

    # compute gradient
    fxk1, nablafxk1 = firstOrderOracle(xk1, true)
    hatxk1 = vk - ak1*nablafxk1
    yk1 = tauk*hatxk1 + (1-tauk)*yk
    fyk1, = firstOrderOracle(yk1, false)
    while fyk1 > fxk1 + dot(nablafxk1, yk1-xk1) + (Lk/2)*norm(yk1-xk1,2)^2 + (eps/2)*tauk && Lk1 < (1/tol)
        ik  = ik+1
        Lk1 = 2^ik*Lk
        ak1 = 1/(2*Lk1) + sqrt((1/(2*Lk1))^2 + Ak/Lk1)
        Ak1 = Ak + ak1
        tauk= ak1/Ak1
        xk1 = tauk*vk + (1-tauk)*yk
        fxk1, nablafxk1 = firstOrderOracle(xk1, true)
        hatxk1 = vk - ak1*nablafxk1
        yk1 = tauk*hatxk1 + (1-tauk)*yk
        fyk1, = firstOrderOracle(yk1, false)
    end
    st.xk=xk1
    st.iterate=yk1
    st.a=Ak1
    st.L=Lk1/2
    st.phi = phik + ak1*nablafxk1
    st.counter = st.counter + 1
end


### The fast smoothing Method
# first order oracle for the smoothing Method
function smoothingMethodOracle(
    p::problem_instance, 
    x::Vector{Float64}, 
    tau::Float64, 
    m::Int64, 
    n::Int64,
    theta::Float64;
    ret_grad::Bool=true)
    
    val = firstOrderOracle(p, x, tau, ret_grad, m, n)
    h = val[1]
    # soft max
    true_max = maximum(h)
    weights = exp.((h .- true_max) / theta)
    normalizer = sum(weights)
    soft_max = theta * log(normalizer) + true_max
    if ret_grad
        # jacobian comps
        J = val[2]
        grad = J' * (weights / normalizer)
        ret = (soft_max, grad)
    else
        ret = (soft_max, )
    end

    return ret
end

function fastSmoothingMethodInitialization(
    x0::Vector{Float64},
    eps::Float64,
    inst::problem_instance)
    burnout = 10^8  # set bournout rate impossibly high to avoid resets
    return fastGradientMethodInitialization(x0, eps, inst, burnout)
end
function fastSmoothingMethodStep(
    st::fastGradientMethodState,
    eps::Float64,
    tau::Float64,
    tol::Float64=10^(-9))
    theta = 0.5*eps / log(st.m + 1)
    oracle = (x, ret_grad) ->  smoothingMethodOracle(st.instance, x, tau, st.m, st.n, theta, ret_grad=ret_grad)
    fastGradientMethodStep(st, 0.5*eps, tau, oracle, tol)
end

fastSmoothingMethod = fom(fastSmoothingMethodInitialization, fastSmoothingMethodStep)

### The fast generalized gradient method
# first order oracle for the generalized gradient Method
function genGradMethodOracle(
    p::problem_instance, 
    x::Vector{Float64}, 
    tau::Float64, 
    lip_cons::Float64, 
    m::Int64, 
    n::Int64;
    tol::Float64=10^(-9),
    ret_grad::Bool=true)
    
    val = firstOrderOracle(p, x, tau, ret_grad, m, n)
    h = val[1]
    max_val = maximum(h)
    if ret_grad
        # jacobian comps
        J = val[2]
        # return zero if gradients are zero
        if sqrt(sum(abs2, J)) < tol/(m+1)
            grad = zeros(m+1)
            println("Gengrad found stationery point")
        else
            GramJ = J * J'
            model = Model(Mosek.Optimizer)
            set_attribute(model, "QUIET", true)
            
            @variable(model, lambdavar[1:m+1])
            @variable(model, t)

            @constraint(model, t*ones(m+1) - h + GramJ*lambdavar in Nonnegatives())

            # make objective with a small quadratic added to avoid mosek numerical issues
            r = min(tol, 10^(-9)) # regularization factor to make sure the objective is strongly convex
            @objective(model, Min, t + 0.5*lip_cons*lambdavar'*GramJ*lambdavar + 0.5*r*(t^2 + lambdavar'*lambdavar))
            optimize!(model)

            opt_lambda = JuMP.value.(lambdavar)

            grad = J'*opt_lambda
        end
        ret = (max_val, grad)
    else
        ret = (max_val, )
    end

    return ret
end

function fastGenGradMethodInitialization(
    x0::Vector{Float64},
    eps::Float64,
    inst::problem_instance)
    burnout = 10^8  # set bournout rate impossibly high to avoid resets
    return fastGradientMethodInitialization(x0, eps, inst, burnout)
end

function fastGenGradMethodStep(
    st::fastGradientMethodState,
    eps::Float64,
    tau::Float64,
    tol::Float64=10^(-9))
    oracle = (x, ret_grad) ->  genGradMethodOracle(st.instance, x, tau, st.L, st.m, st.n, tol=tol, ret_grad=ret_grad)
    fastGradientMethodStep(st, eps, tau, oracle, tol)
end

fastGenGradMethod = fom(fastGenGradMethodInitialization, fastGenGradMethodStep)

## The subgradient Method
mutable struct subgradientState <: algorithmState
    instance::problem_instance
    iterate::Vector{Float64}
end;

function subgradientInitialization( 
    x0::Vector{Float64},
    eps::Float64,
    inst::problem_instance)
    return subgradientState(inst, x0)
end;

function subgradientStep( 
    st::subgradientState,
    eps::Float64,
    tau::Float64,
    tol::Float64=10^(-9)
    )
    # compute the subgradient
    _, subgrad = st.instance.f(st.iterate, tau)
    # make a step if subgradient is not zero
    subgrad_norm_sq = norm(subgrad)^2
    if subgrad_norm_sq > tol^2
        st.iterate = st.iterate - (eps/subgrad_norm_sq) * subgrad
    end
end;

subgradientMethod = fom(subgradientInitialization, subgradientStep)

