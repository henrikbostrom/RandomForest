using DataFrames
# dataframes
function fit!(model::PredictionModel, data::DataFrame, labels; features=:)
    y = (typeof(labels) == Symbol || typeof(labels) <: Array{Symbol}) ? data[labels] : labels
    data = data[features]
    if ~(:WEIGHT in names(data))
        data = hcat(data,DataFrame(WEIGHT = ones(size(source,1))))
    end
    if (typeof(labels) == Symbol || typeof(labels) <: Array{Symbol})
        if (typeof(model.method.learningType) == Classifier)
            rename!(data, labels, :CLASS)
        elseif (typeof(model.method.learningType) == Regressor)
            rename!(data, labels, :REGRESSION)
        else
            rename!(data, labels[1], :TIME)
            rename!(data, labels[2], :EVENT)
        end
    else
        df = addTarget(model, df, y)
    end
    fit!(model, data)
end

function fit!(mode::PredictionModel, data::DataFrame)
    global globaldata = data 
    initiate_workers()
    generated_model = generate_model(method=model.method)
    model.method = generated_model.method
    model.classes = generated_model.classes
    model.version = generated_model.version
    model.oobperformance = generated_model.oobperformance
    model.variableimportance = generated_model.variableimportance
    model.trees = generated_model.trees
    model.conformal = generated_model.conformal
end

function fit!(model::PredictionModel{Survival}, X::Matrix, time::Vector, event::Vector)
    data = prepareDF(model, X, hcat(time, event))
    fit!(model, data)
end

function fit!(model::PredictionModel, X::Matrix, y::Vector)
    data = prepareDF(model, X, y)
    fit!(model, data)
end

function predict(model::PredictionModel, data::DataFrame; features=:)
    data = data[features]
    if ~(:WEIGHT in names(data))
        data = hcat(data,DataFrame(WEIGHT = ones(size(source,1))))
    global globaldata = data
    initiate_workers()
    res = apply_model(model)
    return map( i -> i[1], res)
end

function predict(model::PredictionModel, X::Matrix)
    df = DataFrame(X)
    if (typeof(model.method.learningType) == Survival && length(names(df)) == 1)
        rename!(df, names(x)[1], :TIME)
    end
    predict(model, df)
end

function predict(model::PredictionModel{Survival}, X::Matrix, time::Vector)
    df = DataFrame(X)
    df[:TIME] = time
    predict(model, df)
end

function prepareDF(model::PredictionModel, X::Matrix, y)
    df = DataFrame(X)
    if ~(:WEIGHT in names(df))
        df = hcat(df,DataFrame(WEIGHT = ones(size(df,1))))
    end
    df = addTarget(model, df, y)
    return df
end

function addTarget(model::PredictionModel, df::DataFrame, y)
    if (typeof(model.method.learningType) == Classifier)
        df[:CLASS] = y
    elseif (typeof(model.method.learningType) == Regressor)
        df[:REGRESSION] = y
    else
        df[:TIME] = y[:,1]
        df[:EVENT] = y[:,2]
    end
    return df
end

# classification only methods
function predict_proba(model::PredictionModel{Classifier}, data::DataFrame, features)
    if ~(:WEIGHT in names(data))
        global globaldata = hcat(data,DataFrame(WEIGHT = ones(size(source,1))))
    else
        global globaldata = data 
    end
    res = apply_model(model)
    return map( i -> i[3], res)
end

function predict_proba(model::PredictionModel{Classifier}, X::Matrix)
    df = DataFrame(X)
    if ~(:WEIGHT in names(df))
        df = hcat(df,DataFrame(WEIGHT = ones(size(df,1))))
    end
    global globaldata = df
    res = apply_model(model)
    return map( i -> i[3], res)
end
