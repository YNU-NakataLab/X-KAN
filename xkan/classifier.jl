include("condition.jl")  # Contains UBR type and related utility functions

# Import Python libraries
const copy_mod = pyimport("copy")
const torch = pyimport("torch")

"""
    Classifier

Represents a single rule in the X-KAN system, combining:
- Hyperrectangular antecedent conditions (UBR)
- KAN consequent model
- Evolutionary parameters

# Fields
- `id::Int64`: Unique identifier for tracking
- `condition::Vector{UBR}`: Antecedent hyperrectangle conditions
- `error::Float64`: Mean absolute error of the KAN model
- `fitness::Float64`: Composite fitness value (accuracy × generality)
- `experience::Int64`: Number of times participated in match sets
- `time_stamp::Int64`: Last iteration of EA application
- `match_set_size::Float64`: Average size of match sets containing this classifier
- `numerosity::Int64`: Number of subsumed rules represented by this classifier
- `accuracy::Float64`: Normalized accuracy measure
- `kan::PyObject`: Reference to Python KAN model object
"""
mutable struct Classifier
    id::Int64
    condition::Vector{UBR}
    error::Float64
    fitness::Float64
    experience::Int64
    time_stamp::Int64
    match_set_size::Float64
    numerosity::Int64
    accuracy::Float64
    kan::PyObject
end


"""
    Classifier(parameters, env, state)

Classifier constructor that initializes:
1. Antecedent conditions through covering mechanism
2. KAN consequent model trained on matching data subset
3. Initial evolutionary parameters

# Arguments
- `parameters::NamedTuple`: X-KAN hyperparameters
- `env::NamedTuple`: Training environment configuration
- `state::Vector`: Input vector triggering covering
"""
function Classifier(parameters, env, state)
    # Initialize antecedent conditions
    condition = Vector{UBR}(undef, env.state_length)
    
    @simd for i in 1:env.state_length
        # Covering mechanism: Create new hyperrectangle
        if rand() < parameters.P_hash || state[i] == "?"
            # Create "Don't Care" condition
            condition[i] = rand() < 0.5 ? UBR(0.0, 1.0) : UBR(1.0, 0.0)
        else
            # Create localized condition around input
            offset = rand() * parameters.r0
            l = clamp(state[i] - offset, 0.0, 1.0)
            u = clamp(state[i] + offset, 0.0, 1.0)
            condition[i] = rand() < 0.5 ? UBR(l, u) : UBR(u, l)
        end
    end

    # Get training data subset matching the antecedent
    train_subset = get_train_data_subset(env.train_data, condition)
    
    # Initialize KAN consequent model
    kan = get_consequent(env.state_length, env.epoch, train_subset)
    
    # Calculate initial model metrics
    error = get_error(kan, train_subset)
    accuracy = get_accuracy(error, parameters.e0)

    return Classifier(0, condition, error, parameters.F_I, 0, 0, 1.0, 1, accuracy, kan)
end

"""
    get_train_data_subset(data, condition)

Filters training data to instances matching the classifier's antecedent conditions.

# Arguments
- `data::Matrix`: Full training dataset
- `condition::Vector{UBR}`: Antecedent hyperrectangles

# Returns
- `Matrix`: Subset of data points matching all conditions
"""
function get_train_data_subset(data::Matrix, condition::Vector{UBR})::Matrix
    match_indices = [i for i in 1:size(data,1) if does_match(condition, data[i, 1:end-1])]
    subset = data[match_indices, :]
    
    # Ensure minimum data for training
    return size(subset, 1) > 1 ? subset : data  # Fallback to full dataset
end

"""
    get_consequent(input_dim, epochs, data)

Initializes and trains a KAN model on the given data subset.

# Arguments
- `input_dim::Int`: Number of input features
- `epochs::Int`: Number of training iterations
- `data::Matrix`: Training data subset

# Returns
- `PyObject`: Trained KAN model
"""
function get_consequent(input_dim::Int, epochs::Int, data::Matrix)::PyObject
    # Convert data to PyTorch tensors
    features = torch.tensor(data[:, 1:end-1], dtype=torch.float32)
    labels = torch.tensor(reshape(data[:, end], :, 1), dtype=torch.float32)

    # Initialize KAN architecture: [input, 2n+1 hidden, 1 output]
    kan = MultKAN.KAN(
        width=PyVector([input_dim, 2*input_dim+1, 1]),
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
    pykan.fit(dataset, opt="LBFGS", steps=$epochs, lamb=0.001)
    """

    return py"pykan"
end

"""
    get_error(kan, data)

Calculates mean absolute error (MAE) of KAN predictions.

# Arguments
- `kan::PyObject`: Trained KAN model
- `data::Matrix`: Evaluation dataset

# Returns
- `Float64`: MAE value
"""
function get_error(kan::PyObject, data::Matrix)::Float64
    size(data, 1) > 1 || error("Insufficient training data for error calculation")
    
    total_error = 0.0
    @simd for row in eachrow(data)
        input = row[1:end-1]
        target = row[end]
        
        # Get KAN prediction
        py"""
        prediction = $kan.forward(torch.tensor([$input]).cpu())[0][0].item()
        """
        total_error += abs(py"prediction" - target)
    end
    
    mae = total_error / size(data, 1)
    return isnan(mae) ? 1.0 : mae  # Handle edge cases
end

"""
    get_accuracy(error, threshold)

Converts absolute error to normalized accuracy measure.

# Arguments
- `error::Float64`: MAE value
- `threshold::Float64`: Error cutoff for perfect accuracy

# Returns
- `Float64`: Accuracy in [0, 1]
"""
function get_accuracy(error::Float64, threshold::Float64)::Float64
    error < threshold ? 1.0 : threshold / error
end


"""
    apply_crossover!(child1, child2)

Performs uniform crossover on antecedent conditions.

# Returns
- `Bool`: True if any modifications occurred
"""
function apply_crossover!(child1::Classifier, child2::Classifier)::Bool
    modified = false
    @inbounds for i in eachindex(child1.condition)
        if rand() < 0.5
            # Swap p bounds
            child1.condition[i].p, child2.condition[i].p = child2.condition[i].p, child1.condition[i].p
            modified = true
        end
        
        if rand() < 0.5
            # Swap q bounds
            child1.condition[i].q, child2.condition[i].q = child2.condition[i].q, child1.condition[i].q
            modified = true
        end
    end
    return modified
end

"""
    apply_mutation!(clf, m0, mu)

Applies mutation to antecedent bounds.

# Arguments
- `clf::Classifier`: Classifier to mutate
- `m0::Float64`: Maximum mutation magnitude
- `mu::Float64`: Mutation probability per dimension
"""
function apply_mutation!(clf::Classifier, m0::Float64, mu::Float64)
    @inbounds for i in eachindex(clf.condition)
        if rand() < mu
            # Mutate p bound
            clf.condition[i].p = clamp(clf.condition[i].p + 2*m0*rand() - m0, 0.0, 1.0)
            
            # Mutate q bound
            clf.condition[i].q = clamp(clf.condition[i].q + 2*m0*rand() - m0, 0.0, 1.0)
        end
    end
end

"""
    copy_classifier(clf::Classifier)

Creates a copy of a Classifier instance, including its condition vector and KAN model reference.
"""
function copy_classifier(clf::Classifier)
    return Classifier(
        clf.id,
        deepcopy(clf.condition),
        clf.error,
        clf.fitness,
        clf.experience,
        clf.time_stamp,
        clf.match_set_size,
        clf.numerosity,
        clf.accuracy,
        clf.kan 
        )
end

"""
    does_match(condition::Vector{UBR}, state::Vector)

Checks if a state vector matches the given antecedent condition (vector of UBRs).
Handles "don't care" values ("?") in the state.
"""
function does_match(condition::Vector{UBR}, state)
    for i in eachindex(state)
        if state[i] == "?"
            continue
        end
        if !(get_lower_bound(condition[i]) <= state[i] <= get_upper_bound(condition[i]))
            return false
        end
    end
    return true
end

"""
    is_equal_condition(self::Classifier, other::Classifier)

Checks if two classifiers have identical antecedent conditions.
"""
function is_equal_condition(clf1::Classifier, clf2::Classifier)::Bool
    for i in eachindex(clf1.condition)
        if !is_equal(clf1.condition[i], clf2.condition[i])
            return false
        end
    end
    return true
end

"""
    is_more_general(general, specific)

Determines if one classifier's antecedent subsumes another.

# Arguments
- `general::Classifier`: Potential general classifier
- `specific::Classifier`: Potential specific classifier

# Returns
- `Bool`: True if general's antecedent fully contains specific's
"""
function is_more_general(general::Classifier, specific::Classifier)::Bool
    for (g_cond, s_cond) in zip(general.condition, specific.condition)
        g_l, g_u = get_lower_upper_bounds(g_cond)
        s_l, s_u = get_lower_upper_bounds(s_cond)
        
        (g_l > s_l || g_u < s_u) && return false
    end
    return true
end

"""
    could_subsume(clf::Classifier, e0::Float64)

Checks if a classifier is eligible to subsume another, based on error threshold.
"""
could_subsume(clf::Classifier, e0::Float64)::Bool = clf.error < e0

"""
    does_subsume(self::Classifier, tos::Classifier, e0::Float64)

Determines if `self` can subsume `tos` (the other classifier), based on error and generality.
"""
does_subsume(self::Classifier, tos::Classifier, e0::Float64)::Bool =
    could_subsume(self, e0) && is_more_general(self, tos)

"""
    deletion_vote(clf::Classifier, average_fitness::Float64, theta_del::Int, delta::Float64)

Calculates the deletion vote for a classifier, used in population control.
"""
function deletion_vote(clf::Classifier, average_fitness::Float64, theta_del::Int, delta::Float64)::Float64
    vote = clf.match_set_size * clf.numerosity
    if clf.experience > theta_del && clf.fitness / clf.numerosity < delta * average_fitness
        vote *= average_fitness / max(clf.fitness / clf.numerosity, 1e-12)
    end
    return vote
end

"""
    is_equal_classifier(clf1::Classifier, clf2::Classifier)

Checks if two classifiers are identical by their unique IDs.
"""
is_equal_classifier(clf1::Classifier, clf2::Classifier)::Bool = clf1.id == clf2.id