using ArgParse
using Random
using CSV
using DataFrames
using Dates
using Pkg

# Python/KAN Integration Setup
# ----------------------------
using PyCall

# Configure Python module import paths
PYTHON_MODULE_PATH = abspath("./pykan/kan")
pushfirst!(PyVector(pyimport("sys")."path"), PYTHON_MODULE_PATH)

# Define Python module dependencies
const PYTHON_MODULES = [
    "LBFGS", "feynman", "spline", "KANLayer", "MultKAN",
    "compiler", "experiment", "utils", "__init__",
    "Symbolic_KANLayer", "hypothesis", "MLP"
]

# Import required Python modules
for mod in PYTHON_MODULES
    @eval $(Symbol(mod)) = pyimport($mod)
end

function parse_commandline()
    s = ArgParseSettings(description="X-KAN classifier system")
    @add_arg_table s begin
        "--num_trials"
            help = "The number of trials"
            arg_type = Int
            default = 30
        "--epoch", "-e"
            help = "The number of epochs"
            arg_type = Int
            default = 10
        "--csv"
            help = "CSV"
            arg_type = String
            default = nothing
        "-a", "--all"
            help = "ALL CSV"
            arg_type = Bool
            default = nothing
        "-N"
            help = "The maximum size of the population"
            arg_type = Int
            default = 50
        "--beta"
            help = "The learning rate for updating fitness and match set size estimate in classifiers"
            arg_type = Float64
            default = 0.2
        "--e0"
            help = "The error threshold under which the accuracy of a classifier is set to one"
            arg_type = Float64
            default = 0.02
        "--theta_EA"
            help = "The threshold for the EA application in a match set"
            arg_type = Int
            default = 100
        "--chi"
            help = "The probability of applying crossover"
            arg_type = Float64
            default = 0.8
        "--mu"
            help = "The probability of mutating one allele and the action"
            arg_type = Float64
            default = 0.04
        "--theta_del"
            help = "The experience threshold over which the fitness of a classifier may be considered in its deletion probability"
            arg_type = Int
            default = 100
        "--delta"
            help = "The fraction of the mean fitness of the population below which the fitness of a classifier may be considered in its vote for deletion"
            arg_type = Float64
            default = 0.1
        "--r0"
            help = "The maximum value of a spread in the covering operator"
            arg_type = Float64
            default = 1.0
        "--m0"
            help = "The maximum change of a spread value or a center value in mutation"
            arg_type = Float64
            default = 0.1
        "--F_I"
            help = "The initial fitness value when generating a new classifier"
            arg_type = Float64
            default = 0.01
        "--tau"
            help = "The tournament size for selection"
            arg_type = Float64
            default = 0.4
        "--do_subsumption"
            help = "Whether offspring are to be tested for possible logical subsumption by parents"
            arg_type = Bool
            default = true
        "--P_hash"
            help = "P_hash"
            arg_type = Float64
            default = 1.0
        "--compare_kan"
            help = "Comparison with a single global KAN model"
            arg_type = Bool
            default = false
    end
    return parse_args(s)
end



function main_csv(args; now_str=Dates.format(Dates.now(), "Y-m-d-H-M-S"))
    env = Environment(args)
    param = Parameters(args)
    helper = Helper(env, param)

    println("[ Settings ]")
    println("    Environment = ", get_environment_name(env))
    println("         #Epoch = ", args["epoch"])
    println("          #Inst = ", size(env.all_data, 1))
    println("           #Fea = ", env.state_length)

    println("[ X-KAN General Parameters ]")
    println("              N = ", param.N)
    println("           beta = ", param.beta)
    println("      epsilon_0 = ", param.e0)
    println("       theta_EA = ", param.theta_EA)
    println("            chi = ", param.chi)
    println("             mu = ", param.mu)
    println("      theta_del = ", param.theta_del)
    println("          delta = ", param.delta)
    println("            m_0 = ", param.m0)
    println("            r_0 = ", param.r0)
    println("            F_I = ", param.F_I)
    println("  doSubsumption = ", Bool(param.do_subsumption))
    println("         P_hash = ", param.P_hash)
    println("            tau = ", param.tau)

    @time for n in 1:args["num_trials"]
        start_time = time()
        xcs::XCS = XCS(env, param)
        env.seed = n
        Random.seed!(env.seed)
        shuffle_train_and_test_data!(env)

        println("\n[ Seed $(env.seed) / $(args["num_trials"]) ]\n")
        time_list = Vector{Float64}(undef, 1)
        env.summary_interval = size(env.train_data, 1)
        summary_list = Array{Any}(undef, args["epoch"], 7)

        for e in 1:args["epoch"]
            # Train
            env.is_exploit = false
            num_iter =  size(env.train_data, 1)
            for i in 1:num_iter
                run_experiment(xcs)
            end

            # Test
            env.is_exploit = true
            train_syserr::Float64, test_syserr::Float64 = get_syserr_per_epoch(xcs, env)
            popsize::Int64 = length(xcs.population)

            # Output log
            summary_list = output_log_for_csv(e, args["epoch"], env, train_syserr, test_syserr, popsize, xcs.covering_occur_num, xcs.subsumption_occur_num, summary_list)
            xcs.covering_occur_num = xcs.subsumption_occur_num = 0.
        end

        elapsed_time = time() - start_time
        time_list[1] = elapsed_time
        if args["all"] == nothing
            dir_path = "./result/" * get_environment_name(env) * "/" * now_str * "/seed" * string(n)
        else
            dir_path = "./all" * now_str * "/" *  basename(get_environment_name(env)) * "/seed" * string(n)
        end

        mkpath(dir_path)
        make_one_column_csv(joinpath(dir_path, "time.csv"), time_list)
        make_matrix_csv(joinpath(dir_path, "classifier.csv"), make_classifier_list(helper, xcs))
        make_matrix_csv(joinpath(dir_path, "summary.csv"), summary_list)
        make_matrix_csv(joinpath(dir_path, "parameter.csv"), make_parameter_list(args))

        # For rule compaction
        rule_compaction!(xcs, env)
        train_syserr, test_syserr = get_syserr_per_epoch(xcs, env)
        popsize = length(xcs.population)
        compacted_summary_list = Vector{Any}(undef, 3)
        compacted_summary_list[1] = train_syserr
        compacted_summary_list[2] = test_syserr
        compacted_summary_list[3] = popsize
        println("\n X-KAN after Rule Compaction:")
        println("Training Mean Absolute Error: $(train_syserr)")
        println(" Testing Mean Absolute Error: $(test_syserr)")
        println("      Compacted Ruleset Size: $(popsize)")
        make_matrix_csv(joinpath(dir_path, "compacted_summary.csv"), compacted_summary_list)
        make_matrix_csv(joinpath(dir_path, "compacted_classifier.csv"), make_classifier_list(helper, xcs))

        # For comparison to a global KAN model
        if args["compare_kan"]
            println("\n A Single Global KAN Model ($(env.epoch)):")
            global_kan_summary_list = Vector{Any}(undef, 2)
            features = torch.tensor(env.train_data[:, 1:end-1], dtype=torch.float32)
            labels = torch.tensor(reshape(env.train_data[:, end], :, 1), dtype=torch.float32)

            # Initialize KAN architecture: [input, 2n+1 hidden, 1 output]
            kan = MultKAN.KAN(
                width=PyVector([env.state_length, 2*env.state_length+1, 1]),
                grid=3,
                k=3,
                seed=42,
                device="cpu"
            )
            kan.speed()

            # Train KAN with L-BFGS optimizer
            py"""
            import torch
            dataset = {
                'train_input': $features.to('cpu'),
                'train_label': $labels.to('cpu')
            }
            pykan = $kan
            pykan.fit(dataset, opt="LBFGS", steps=$env.epoch, lamb=0.001)
            """
            train_syserr = get_error(py"pykan", env.train_data)
            test_syserr = get_error(py"pykan", env.test_data)
            global_kan_summary_list[1] = train_syserr
            global_kan_summary_list[2] = test_syserr
            println("Training Mean Absolute Error: $(train_syserr)")
            println(" Testing Mean Absolute Error: $(test_syserr)")
            make_matrix_csv(joinpath(dir_path, "global_kan_summary.csv"), global_kan_summary_list)
        end

    end


end

function main_all_csv(args)
    now = Dates.now()
    now_str = Dates.format(now, "Y-m-d")
    dir_path::String = "./dataset/"
    csv_list_array::Vector{String} = [
        "f1",
        "f2",
        "f3",
        "f4",
        "airfoil_self_noise",
        "ccpp",
        "Concrete_Data",
        "energy_efficiency_cooling"
    ]

    for csv::String in csv_list_array
        args["csv"] = dir_path * csv * ".csv"
        if csv in ["f1","f2","f3","f4"]
            args["P_hash"] = 0.0
        else
            args["P_hash"] = 0.8
        end
        main_csv(args; now_str)
    end
end


args = parse_commandline()
include("./environment/csv.jl")
include("parameters.jl")

# System Setup
include("./xkan/xcs.jl")
include("helper.jl")

if args["csv"] == nothing && args["all"]
    main_all_csv(args)
elseif args["csv"] != nothing && !args["all"]
    main_csv(args)
else
    error("Set either args[\"csv\"] or args[\"all\"]")
end