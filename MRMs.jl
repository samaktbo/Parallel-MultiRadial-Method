module MRM

using LinearAlgebra
using Random, Distributions

# Generic type for describing problems that must have fields f and primal, 
# corresponding to the primal oracle and dual oracles respectively
abstract type problem_instance end;

# generic type that represents an algorithm
abstract type algorithm end;


# include problem instances
include("Problems/qcqpConstructor.jl")

# include solvers
include("Solvers/parallelMRM.jl")

# test and experiments scripts 
include("experimentFunctions.jl")

# first order oracle for max of functions
function eval_max(
    f::Vector{Function},
    y::Vector{Float64},
    tau::Float64
)

# A function that takes a vector of functions f and applies each function at y, tau. 
# This function assumes each fi in f also takes in a third boolean parameter, which 
# which we refer to as ret_grad. This function assumes that fi(y, tau, ret_grad) 
# returns a tuple whose first component is the value of fi at (y, tau). The second component
# is the subgradient of fi at (y, tau) if ret_grad true, otherwise the second component is empty

#
# Returns: (v, g) where v is a float corresponding to max([f1(y, tau), ... , fn(y, tau)])
# and g is the subgradient

# evaluate each function at y
values = map(fi -> fi(y, tau, false)[1], f)

# find the argmax
(value, i) = findmax(values)
ret = f[i](y, tau, true)
return ret
end

end