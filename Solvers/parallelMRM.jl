using LinearAlgebra
using Printf

include("fomS.jl")

mutable struct parallelMRMState
    instance::problem_instance
    iterate::Vector{Float64}
    tau::Float64
    feas_val::Float64
end;

# make ||-MRM state copible
Base.copy(s::parallelMRMState) = parallelMRMState(s.instance, copy(s.iterate), copy(s.tau), copy(s.feas_val))

function parallelMRMRestart(
    st::algorithmState, 
    fom::algorithm, 
    best_st::parallelMRMState, 
    tau::Float64, 
    delta::Float64;
    rho_tilde::Float64=1.0
    )
    if (1 + rho_tilde * delta) * best_st.tau < tau
        new_state = fom.initialize(copy(best_st.iterate), delta, st.instance)
        # keep old smoothness constant for smooth methods
        if fom == fastSmoothingMethod || fom == fastGenGradMethod
            new_state.L = st.L
        end
        to_ret = (new_state, copy(best_st.tau))
    else
        to_ret = (st, tau)
    end
    return to_ret
end

function getFeasibility(x::Vector{Float64}, inst::problem_instance)
    identifier_vals = [g(x, false)[1] for g in inst.identifiers]
    (feas_val, _) = findmax(identifier_vals)
    return feas_val
end

# parallelMRM scheme
function runParallelMRM(
        fom::algorithm,
        inst::problem_instance,
        x_init::Tuple{Vector{Float64}, Float64}, # a tuple of (x0, 1/f(x0)) where x0 is feasible and f(x0) > 0 
        deltas::Vector{Float64},
        T::Int64,
        tol::Float64=10^(-7),
        rho_tilde::Float64=1.0,
        tot_time::Float64=60.0, # total real time (in seconds) to run
        verbose::Bool=false # a boolean indicating whether to print output
    )
    # function that runs different chanels of the same algorithm
    N = size(deltas)[1]  # number of chanels
    ret = Matrix{Float64}(undef, N, T+1) # matrix of objective values seen
    feas = Matrix{Float64}(undef, N, T+1) # matrix of feasibility values seen
    best_seen = Vector{Float64}(undef, T+1)
    taus = Vector{Float64}(undef, N)
    alg_states = Vector{algorithmState}(undef, N)
    
    # initialization
    x0, tau0 = x_init
    feas_val = getFeasibility(x0, inst)
    
    if feas_val > 1
        throw(error("must start with a feasible point"))
    end

    if tau0 <= tol
        throw(error("must start with positive tau"))
    end
    
    primal_val = 1 / tau0

    for l in 1:N
        ret[l, 1] = primal_val
        feas[l, 1] = feas_val
        taus[l] = copy(tau0)
        alg_states[l] = fom.initialize(x0, deltas[l], inst)
    end

    best_state = parallelMRMState(inst, x0, copy(tau0), copy(feas_val))
    best_seen[1] = primal_val
    times = zeros(N, T+1)

    k = 0
    real_time = time()
    while k < T && time() < real_time + tot_time
        # run all algorithms for T steps
        k += 1
        current_bs = copy(best_state)
        for l in 1:N
            # run the restart function
            start_time = time()
            alg_states[l], taus[l] = parallelMRMRestart(alg_states[l], fom, best_state, taus[l], deltas[l], rho_tilde=rho_tilde)
            
            # make a step
            fom.step(alg_states[l], deltas[l], taus[l], tol)
            feas[l, k+1] = getFeasibility(alg_states[l].iterate, inst)

            # update current_best_state if conditions are met
            if feas[l, k+1] <= 1
                ret[l, k+1] = inst.primal(alg_states[l].iterate, false)[1]
                if 1 / current_bs.tau < ret[l, k+1]
                    current_bs.iterate = copy(alg_states[l].iterate)
                    current_bs.tau = 1 / ret[l, k+1]
                    current_bs.feas_val = copy(feas[l, k+1])
                end
            else
                ret[l, k+1] = ret[l, k]
            end
            end_time = time()
            times[l, k+1] = end_time - start_time
        end
        # update best_state
        best_state = copy(current_bs)
        # store best values seen thus far
        best_seen[k+1] = copy(1 / best_state.tau)
        if verbose
            ratio = tau0 / best_state.tau
            @printf("At iteration number %d, relative improvement is %.6f. \n", k, ratio)
        end
    end
    best_seen, ret, feas, times = best_seen[1:k+1], ret[:, 1:k+1], feas[:, 1:k+1], times[:, 1:k+1]
    return best_state, best_seen, ret, feas, times
end;

# wrapper function
function parallelMRM(
    alg::algorithm, 
    p::problem_instance, 
    x_init::Tuple{Vector{Float64}, Float64}, 
    tot_iter::Int64; # maximum number of iterations
    b::Float64=2.0, # base of that will determine accuracy
    N::Int64=20, # total number of fomS to run
    tol::Float64=10^(-7), # level of precision,
    rho_tilde::Float64=1.0,
    tot_time::Float64=60.0, # total real time (in seconds) to run
    verbose::Bool=false  # whether output should be printed as this solver runs
)

deltas = [b^(-1*l) for l=1:N]
T = tot_iter
return runParallelMRM(alg, p, x_init, deltas, T, tol, rho_tilde, tot_time, verbose)
end;