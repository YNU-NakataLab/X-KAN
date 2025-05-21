include("classifier.jl")
using Distributions
using LinearAlgebra

"""
    XCS

Main XCS framework container storing environment, parameters, and population state.

# Fields
- `env::Environment`: Problem environment configuration
- `parameters::Parameters`: Hyperparameter collection
- `population::Vector{Classifier}`: Current rule population
- `time_stamp::Int64`: Global iteration counter
- `covering_occur_num::Int64`: Covering operation counter
- `subsumption_occur_num::Int64`: Subsumption operation counter
- `global_id::Int64`: Unique classifier ID generator
"""
mutable struct XCS
    env::Environment
    parameters::Parameters
    population::Vector{Classifier}
    time_stamp::Int64
    covering_occur_num::Int64
    subsumption_occur_num::Int64
    global_id::Int64

    # Inner constructor for initialization validation
    function XCS(env, params)
        new(
            env,
            params,
            Classifier[],
            0,  # time_stamp
            0,  # covering_occur_num
            0,  # subsumption_occur_num
            0   # global_id
        )
    end
end

"""
    run_experiment(xcs::XCS)

Execute one complete XCS learning iteration:
1. Observe current environment state
2. Form/update match set
3. Apply evolutionary operations
4. Increment global timer
"""
function run_experiment(xcs::XCS)
    curr_state = state(xcs.env)
    match_set = generate_match_set(xcs, curr_state)
    update_set!(xcs, match_set)
    run_ea!(xcs, match_set)
    xcs.time_stamp += 1
end

"""
    generate_match_set(xcs::XCS, state, do_exploit=false)

Form match set of classifiers matching current state. Implements covering mechanism
when no matching rules exist (unless in exploit mode).

# Arguments
- `state::Vector`: Current input vector
- `do_exploit::Bool`: Flag for exploitation phase (no covering)

# Returns
- `Vector{Classifier}`: Matching classifiers
"""
function generate_match_set(
    xcs::XCS,
    state::Vector{Union{Float64,Int64,String}},
    do_exploit::Bool=false
)::Vector{Classifier}
    match_set = filter(clf -> does_match(clf.condition, state), xcs.population)
    
    if !do_exploit && isempty(match_set)
        # Covering mechanism: Create new rule for uncovered state
        clf = generate_covering_classifier(xcs, match_set, state)
        push!(xcs.population, clf)
        delete_from_population!(xcs)  # Maintain population size
        push!(match_set, clf)
    end

    return match_set
end

"""
    generate_covering_classifier(xcs::XCS, match_set, state)

Create new classifier through covering mechanism for uncovered state.

# Returns
- `Classifier`: Newly created classifier
"""
function generate_covering_classifier(
    xcs::XCS,
    match_set::Vector{Classifier},
    state::Vector{Union{Float64,Int64,String}}
)::Classifier
    clf = Classifier(xcs.parameters, xcs.env, state)
    clf.time_stamp = xcs.time_stamp
    clf.id = xcs.global_id
    
    # Update system counters
    xcs.covering_occur_num += 1
    xcs.global_id += 1

    return clf
end

"""
    generate_prediction_array(xcs::XCS, match_set, state)

Generate predictions using single-winner inference scheme.

# Returns
- `Tuple`: (winning classifier, prediction array)
"""
function generate_prediction_array(
    xcs::XCS,
    match_set::Vector{Classifier},
    state::Vector{Union{Float64,Int64,String}}
)::Tuple{Union{Classifier,Nothing},Vector{Float64}}
    max_fitness = 0.0
    max_clf = nothing
    prediction = zeros(Float64, 1)

    # Find classifier with maximum fitness
    for clf in match_set
        if clf.fitness >= max_fitness
            max_fitness = clf.fitness
            max_clf = clf
        end
    end

    # Generate prediction if winner found
    if !isnothing(max_clf)
        py"""
        import torch
        state = $state
        kan = $max_clf.kan
        device = 'cpu'
        prediction = kan.forward(torch.tensor([state]).to(device))[0][0].item()
        """
        prediction[1] = py"prediction"
    end

    return (max_clf, prediction)
end

"""
    update_set!(xcs::XCS, match_set)

Update experience and match set size for all classifiers in match set.

# Implementation Details
- Uses Widrow-Hoff update rule with learning rate β
- Maintains moving average of match set size
"""
function update_set!(xcs::XCS, match_set::Vector{Classifier})
    beta = xcs.parameters.beta
    total_numerosity = sum(clf.numerosity for clf in match_set)

    @simd for clf in match_set
        clf.experience += 1
        # Update match set size estimate
        clf.match_set_size += beta * (total_numerosity - clf.match_set_size)
    end

    update_fitness!(xcs, match_set)
end

"""
    update_fitness!(xcs::XCS, match_set)

Update classifier fitness using relative accuracy weighting.

# Mathematical Formulation
Fitness F^k is updated as:
F^k ← F^k + β[(κ^k·num^k)/Σ(κ^q·num^q) - F^k]
where κ^k is accuracy and num^k is generality (numerosity)
"""
function update_fitness!(xcs::XCS, match_set::Vector{Classifier})
    beta = xcs.parameters.beta
    F_I = xcs.parameters.F_I
    accuracy_sum = sum(clf.accuracy * clf.numerosity for clf in match_set)

    @simd for clf in match_set
        # Calculate relative accuracy contribution
        relative_accuracy = (clf.accuracy * clf.numerosity) / accuracy_sum
        clf.fitness += beta * (relative_accuracy - clf.fitness)
        
        # Handle numerical instability
        if isnan(clf.fitness)
            clf.fitness = F_I
        end
    end
end

"""
    run_ea!(xcs::XCS, match_set::Vector{Classifier})

Executes the Evolutionary Algorithm (EA) on the match set. Key operations include:
- Parent selection via tournament selection
- Offspring generation through crossover and mutation
- Subsumption-based population management
- Dynamic population size control

# Arguments
- `xcs::XCS`: Main XCS instance
- `match_set::Vector{Classifier}`: Current set of matching classifiers
"""
function run_ea!(xcs::XCS, match_set::Vector{Classifier})
    if isempty(match_set)
        return
    end

    # Calculate EA trigger condition using moving average of timestamps
    if xcs.time_stamp - mapreduce(clf -> clf.time_stamp * clf.numerosity, +, match_set) / mapreduce(clf -> clf.numerosity, +, match_set) > xcs.parameters.theta_EA
        # Update timestamps for all matching classifiers
        @simd for clf in match_set
            clf.time_stamp = xcs.time_stamp
        end

        # Parent selection and offspring creation
        parent_1::Classifier = select_offspring(xcs, match_set)
        parent_2::Classifier = select_offspring(xcs, match_set)

        # Initialize offspring with proper ID management
        child_1::Classifier = copy_classifier(parent_1)
        child_2::Classifier = copy_classifier(parent_2)
        child_1.id = xcs.global_id
        child_2.id = xcs.global_id + 1
        xcs.global_id += 2

        # Reset offspring parameters
        child_1.fitness /= parent_1.numerosity
        child_2.fitness /= parent_2.numerosity
        child_1.numerosity = 1
        child_2.numerosity = 1
        child_1.experience = 0
        child_2.experience = 0

        # Apply crossover with probability χ
        if rand() < xcs.parameters.chi
            is_changed_x = apply_crossover!(child_1, child_2)
            if is_changed_x
                # Blend parent errors and fitness if crossover occurred
                child_1.fitness = child_2.fitness = (parent_1.fitness + parent_2.fitness) / 2.
            end
        end

        # Fitness reduction for new offspring
        child_1.fitness *= 0.1
        child_2.fitness *= 0.1

        @simd for child in (child_1, child_2)
            # Apply mutation to both offspring
            apply_mutation!(child, xcs.parameters.m0, xcs.parameters.mu)
            # Subsumption check and population insertion
            if xcs.parameters.do_subsumption
                if does_subsume(parent_1, child, xcs.parameters.e0)
                    xcs.subsumption_occur_num += 1
                    parent_1.numerosity = parent_1.numerosity + 1
                elseif does_subsume(parent_2, child, xcs.parameters.e0)
                    xcs.subsumption_occur_num += 1
                    parent_2.numerosity = parent_2.numerosity + 1
                else
                    insert_in_population!(xcs, child)
                end
            else
                insert_in_population!(xcs, child)
            end

            # Maintain population size constraint
            delete_from_population!(xcs)
        end
    end
end

"""
    select_offspring(xcs::XCS, match_set::Vector{Classifier}) -> Classifier

Performs tournament selection to choose parent classifiers.
Uses stochastic universal sampling with tournament size τ.

# Returns
- `Classifier`: Selected parent with highest relative fitness
"""
function select_offspring(xcs::XCS, match_set::Vector{Classifier})::Classifier
    parent::Any = nothing
    while parent == nothing
        for clf in match_set
            if parent == nothing || parent.fitness / parent.numerosity < clf.fitness / clf.numerosity
                for i in 1:clf.numerosity
                    if rand() < xcs.parameters.tau
                        parent = clf
                        break
                    end
                end
            end
        end
    end

    return parent
end

"""
    insert_in_population!(xcs::XCS, clf::Classifier)

Inserts a classifier into the population, either merging with existing rules
or adding as new entry. Initializes KAN consequent for new rules.
"""
function insert_in_population!(xcs::XCS, clf::Classifier)
    # Check for existing matching conditions
    for existing in xcs.population
        if is_equal_condition(existing, clf)
            existing.numerosity += 1
            return
        end
    end

    # Initialize new classifier's KAN model
    train_data = get_train_data_subset(xcs.env.train_data, clf.condition)
    clf.kan = get_consequent(xcs.env.state_length, xcs.env.epoch, train_data)
    clf.error = get_error(clf.kan, train_data)
    clf.accuracy = get_accuracy(clf.error, xcs.parameters.e0)

    push!(xcs.population, clf)
end

"""
    delete_from_population!(xcs::XCS)

Maintains population size using fitness-proportionate deletion.
Implements roulette wheel selection based on deletion votes.
"""
function delete_from_population!(xcs::XCS)
    numerosity_sum::Float64 = mapreduce(clf -> clf.numerosity, +, xcs.population)
    if numerosity_sum <= xcs.parameters.N
        return
    end

    # Calculate deletion votes
    average_fitness::Float64 = mapreduce(clf -> clf.fitness, +, xcs.population) / numerosity_sum
    vote_sum::Float64 = mapreduce(clf -> deletion_vote(clf, average_fitness, xcs.parameters.theta_del, xcs.parameters.delta), +, xcs.population)

    choice_point::Float64 = rand() * vote_sum
    vote_sum = 0.

    # Roulette wheel selection
    for clf in xcs.population
        vote_sum += deletion_vote(clf, average_fitness, xcs.parameters.theta_del, xcs.parameters.delta)
        if vote_sum > choice_point
            clf.numerosity -= 1
            if clf.numerosity == 0
                @views filter!(e -> e != clf, xcs.population)
            end
            return
        end
    end
end