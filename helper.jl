using DataFrames, CSV
using Printf

mutable struct Helper
    env::Environment
    parameters::Parameters
end

function Helper(env, parameters)
    return Helper(env, parameters)
end

function make_one_column_csv(filename::String, list)
    dataframe = DataFrame(x=list)
    CSV.write(filename, dataframe, delim=',', writeheader=false)
end

function make_matrix_csv(filename::String, list)
    tbl = Tables.table(list)
    CSV.write(filename, tbl, header=false)
end

function make_classifier_list(self::Helper, xcs::XCS)::Array{Any, 2}
    classifier_list = Array{Any}(undef, length(xcs.population) + 1, 10)

    classifier_list[1, 1] = "ID"
    classifier_list[1, 2] = "Antecedent"
    classifier_list[1, 3] = "Consequent"
    classifier_list[1, 4] = "Fitness"
    classifier_list[1, 5] = "Error"
    classifier_list[1, 6] = "Accuracy"
    classifier_list[1, 7] = "Experience"
    classifier_list[1, 8] = "Time Stamp"
    classifier_list[1, 9] = "Match Set Size"
    classifier_list[1, 10] = "Numerosity"
    

    i::Int64 = 2
    for clf in xcs.population
        condition::String = "["
        for j in 1:length(clf.condition)
            l, u = get_lower_upper_bounds(clf.condition[j])
            condition *= string(round(l, digits=3)) * ":" * string(round(u, digits=3)) * ", "
        end
        condition = chop(condition, tail=2) * "]"
        classifier_list[i, 1] = clf.id
        classifier_list[i, 2] = condition
        classifier_list[i, 3] = "KAN"
        classifier_list[i, 4] = clf.fitness
        classifier_list[i, 5] = clf.error
        classifier_list[i, 6] = clf.accuracy
        classifier_list[i, 7] = clf.experience
        classifier_list[i, 8] = clf.time_stamp
        classifier_list[i, 9] = clf.match_set_size
        classifier_list[i, 10] = clf.numerosity
        i += 1
    end
    return classifier_list
end

function make_parameter_list(args)::Array{Any, 2}
    parameter_list = Array{Any}(undef, 14, 2)

    parameter_list[1,:] = ["N", args["N"]]
    parameter_list[2,:] = ["beta", args["beta"]]
    parameter_list[3,:] = ["e0", args["e0"]]
    parameter_list[4,:] = ["theta_EA", args["theta_EA"]]
    parameter_list[5,:] = ["chi", args["chi"]]
    parameter_list[6,:] = ["mu", args["mu"]]
    parameter_list[7,:] = ["m0", args["m0"]]
    parameter_list[8,:] = ["theta_del", args["theta_del"]]
    parameter_list[9,:] = ["delta", args["delta"]]
    parameter_list[10,:] = ["P_hash", args["P_hash"]]
    parameter_list[11,:] = ["r0", args["r0"]]
    parameter_list[12,:] = ["F_I", args["F_I"]]
    parameter_list[13,:] = ["tau", args["tau"]]
    parameter_list[14,:] = ["doSubsumption", args["do_subsumption"]]

    return parameter_list
end

function val_with_spaces(val::Any)
    s = ""
    str_val = string(val)
    for i in 1:11-length(str_val)
        s *= " "
    end
    return s * str_val * " "
end

function output_log_for_csv(current_epoch::Int64, num_epoch::Int64, env::Environment, train_syserr::Float64, test_syserr::Float64, popsize::Int64, covering_occur_num::Int64, subsumption_occur_num::Int64, summary_list::Array{Any, 2})::Array{Any, 2}
    if current_epoch == 1
        println("      Epoch   Iteration    TrainErr     TestErr     PopSize  CovOccRate   SubOccNum")
        println("=========== =========== =========== =========== =========== =========== ===========")
    end

    train_syserr_str = @sprintf("%.6f", train_syserr)
    test_syserr_str = @sprintf("%.6f", test_syserr)
    popsize_str = @sprintf("%.3f", popsize)
    println(val_with_spaces(current_epoch), val_with_spaces(current_epoch * size(env.train_data, 1)),val_with_spaces(train_syserr_str), val_with_spaces(test_syserr_str), val_with_spaces(popsize_str),val_with_spaces(round(covering_occur_num / env.summary_interval, digits=3)), val_with_spaces(subsumption_occur_num))

    summary_list[round(Int, current_epoch),:] = [current_epoch, current_epoch * size(env.train_data, 1), train_syserr, test_syserr, popsize, covering_occur_num / env.summary_interval, round(Int, subsumption_occur_num)]

    return summary_list
end

function get_syserr_per_epoch(xcs::XCS, env::Environment)::Tuple{Float64, Float64}
    # Initialize error accumulators
    train_error = 0.0
    test_error = 0.0

    # Calculate training MAE
    @inbounds for row in eachrow(env.train_data)
        state = row[1:end-1]  # Convert to Float64 vector
        target = Float64(row[end])
        pred = prediction(xcs, state)
        train_error += abs(pred - target)
    end

    # Calculate testing MAE
    @inbounds for row in eachrow(env.test_data)
        state = row[1:end-1]  # Convert to Float64 vector
        target = Float64(row[end])
        pred = prediction(xcs, state)
        test_error += abs(pred - target)
    end

    # Normalize by dataset size with safety checks
    train_mae = size(env.train_data, 1) > 0 ? train_error / size(env.train_data, 1) : 0.0
    test_mae = size(env.test_data, 1) > 0 ? test_error / size(env.test_data, 1) : 0.0

    return (train_mae, test_mae)
end

function prediction(xcs::XCS, state::Vector{Union{Float64, Int64, String}})::Float64
    match_set::Vector{Classifier} = @views generate_match_set(xcs, state, true)
    _, prediction_array::Vector{Float64} = @views generate_prediction_array(xcs, match_set, state)
    return prediction_array[1]
end

"Single Winner-Based Rule Compaction"
function rule_compaction!(xcs::XCS, env::Environment)
    final_population::Vector{Classifier} = []
    @simd for row_index in 1:size(env.train_data, 1)
        state::Vector{Union{Float64, Int64, String}} = env.train_data[row_index, 1:size(env.train_data,2)-1]
        match_set::Vector{Classifier} = @views generate_match_set(xcs, state, true)
        clf, _ = @views generate_prediction_array(xcs, match_set, state)
        if clf != nothing
            push!(final_population, clf)
        end
    end

    xcs.population = collect(Set(final_population))
end



