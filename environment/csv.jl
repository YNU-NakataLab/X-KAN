using CSV, DataFrames, Random

mutable struct Environment
    epoch::Int64
    summary_interval::Int64
    seed::Int64
    is_exploit::Bool
    state_length::Int64
    row_index::Int64
    train_data::Array{Union{Float64, Int64, String}, 2}
    test_data::Array{Union{Float64, Int64, String}, 2}
    all_data::Array{Union{Float64, Int64, String}, 2}
    file_path::String
    index_array::Vector{Int64}
end

#=
CSV dataset
=#

function Environment(args)::Environment
    train_data, test_data, all_data = get_train_and_test_data_and_all_data(args["csv"], args)
    num_actions, state_length = get_data_information(args["csv"])
    return Environment(args["epoch"], 0, 0, false, state_length, 0, train_data, test_data, all_data, args["csv"], Vector(1:size(train_data, 1)))
end

function get_train_and_test_data_and_all_data(filename::String, args)::Tuple{Array{Union{Float64, Int64, String}, 2}, Array{Union{Float64, Int64, String}, 2}, Array{Union{Float64, Int64, String}, 2}}
    all_data = CSV.File(filename; stringtype=String, header=false) |> DataFrame
    all_data = Matrix(all_data)

    train_ratio = 0.9 # Monte Carlo CV (Training:Testing = 9:1)
    train_data_length::Int64 = Int(floor(size(all_data, 1) * train_ratio))
    test_data_length::Int64 = size(all_data, 1) - train_data_length

    train_data = Array{Any}(undef, train_data_length, size(all_data, 2))
    test_data = Array{Any}(undef, test_data_length, size(all_data, 2))
    
    return train_data, test_data, all_data
end

function get_data_information(filename::String)::Tuple{Int64, Int64}
    all_data = CSV.read(filename, DataFrame, header=false)
    num_actions::Int64 = 1
    state_length::Int64 = size(all_data, 2) - 1
    return num_actions, state_length
end


function normalize_columns(data_train, data_test)::Tuple{Array{Union{Float64, Int64, String}, 2}, Array{Union{Float64, Int64, String}, 2}}
     @simd for j = 1:(size(data_train, 2) - 1)
        col = data_train[:, j]
        # Calculate min and max for each column
        col_without_missing = col[col .!= "?"]
        col_without_missing = map(x -> parse(Float64, string(x)), col_without_missing)
        col_min = minimum(col_without_missing)
        col_max = maximum(col_without_missing)
        for data in [data_train, data_test]
            @simd for i = 1:size(data, 1)
                if data[i, j] != "?"
                    # Normalize data to the range 0 to 1
                    v = parse(Float64, string(data[i, j]))
                    if (col_min == col_max)
                        data[i, j] = 0.5
                    else
                        data[i, j] = max(0, min(1, (v - col_min) / (col_max - col_min)))
                    end
                else
                    data[i, j] = "?"
                end
            end
        end
    end

    j = size(data_train, 2)
    col = data_train[:, j]
    # Calculate min and max for each column
    col_without_missing = col[col .!= "?"]
    col_without_missing = map(x -> parse(Float64, string(x)), col_without_missing)
    col_min = minimum(col_without_missing)
    col_max = maximum(col_without_missing)
    for data in [data_train, data_test]
        @simd for i = 1:size(data, 1)
            # Normalize data to the range -1 to 1
            v = parse(Float64, string(data[i, j]))
            data[i, j] = max(-1, min(1, (2*(v - col_min) / (col_max - col_min)-1)))
        end
    end
    return data_train, data_test
end

# Per epoch
function shuffle_index_array_and_reset_row_index!(self::Environment)
    rng = MersenneTwister(self.seed)
    if self.row_index % size(self.train_data, 1) == 0
        self.index_array = Vector(1:size(self.train_data, 1))
        shuffle!(rng, self.index_array)
        self.row_index = 0
    end
end

# Per seed
function shuffle_train_and_test_data!(self::Environment)
    rng = MersenneTwister(self.seed)
    train_data_length::Int64 = size(self.train_data, 1)
    test_data_length::Int64 = size(self.test_data, 1)

    train_data = Array{Any}(undef, train_data_length, size(self.all_data, 2))
    test_data = Array{Any}(undef, test_data_length, size(self.all_data, 2))

    # Generate a random permutation of row indices using shuffle
    perm::Vector{Int64} = shuffle(rng, 1:train_data_length + test_data_length)

    # Rearrange the rows of the array using the permutation
    self.all_data = self.all_data[perm, :]

    # Generate train data
    for i = 1:train_data_length
        train_data[i, :] = self.all_data[i, :]
    end
    # Generate test data
    for i = 1: test_data_length
        test_data[i, :] = self.all_data[train_data_length + i, :]
    end

    self.train_data, self.test_data = normalize_columns(train_data, test_data)

end

function state(self::Environment)::Vector{Union{Float64,Int64,String}}
    if self.is_exploit == false
        # Shuffle indices and reset pointer when starting new epoch
        shuffle_index_array_and_reset_row_index!(self)
        
        # Safeguard against index overflow
        self.row_index >= length(self.index_array) && error("Data pointer out of bounds")
        
        # Increment pointer and extract features (exclude last column)
        self.row_index += 1
        row_idx = self.index_array[self.row_index]
        return self.train_data[row_idx, 1:end-1]
    else
        throw(ErrorException("State sampling unavailable during exploitation phase. Use test data directly."))
    end
end

function get_environment_name(self::Environment)::String
    return "$(self.file_path)"
end

function make_matrix_csv(filename::String, list)
    tbl = Tables.table(list)
    CSV.write(filename, tbl, header=false)
end
