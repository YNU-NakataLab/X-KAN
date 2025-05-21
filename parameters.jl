"""
X-KAN Hyperparameters
(for further details: julia ./main.jl --help)
"""
mutable struct Parameters
    N::Int64
    beta::Float64
    e0::Float64
    theta_EA::Int64
    chi::Float64
    mu::Float64
    theta_del::Int64
    delta::Float64
    F_I::Float64
    tau::Float64
    m0::Float64
    r0::Float64
    do_subsumption::Bool
    P_hash::Float64
end

function Parameters(args)
    return Parameters(
        args["N"], 
        args["beta"],
        args["e0"],
        args["theta_EA"],
        args["chi"], 
        args["mu"], 
        args["theta_del"], 
        args["delta"], 
        args["F_I"], 
        args["tau"], 
        args["m0"], 
        args["r0"], 
        args["do_subsumption"], 
        args["P_hash"])
end

