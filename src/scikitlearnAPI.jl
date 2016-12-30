using DataFrames
# dataframes
function fit!(model::PredictionModel, data::DataFrame, features, labels)
   if ~(:WEIGHT in names(data))
        global globaldata = hcat(data,DataFrame(WEIGHT = ones(size(source,1))))
    else
        global globaldata = data # Made global to allow access from workers
    end
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
    global globaldata = prepareDF(model, X, hcat(time, event))
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

function fit!(model::PredictionModel, X::Matrix, y::Vector)
    global globaldata = prepareDF(model, X, y)
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

# dataframes. Still assumes globaldata
function predict(model::PredictionModel, data::DataFrame, features)
    if ~(:WEIGHT in names(data))
        global globaldata = hcat(data,DataFrame(WEIGHT = ones(size(source,1))))
    else
        global globaldata = data 
    end
    initiate_workers()
    res = apply_model(model)
    return map( i -> i[1], res)
end

function predict(model::PredictionModel, X::Matrix)
    df = DataFrame(X)
    if ~(:WEIGHT in names(df))
        df = hcat(df,DataFrame(WEIGHT = ones(size(df,1))))
    end
    global globaldata = df
    initiate_workers()
    res = apply_model(model)
    return map( i -> i[1], res)
end

function prepareDF(model::PredictionModel, X::Matrix, y)
    df = DataFrame(X)
    if ~(:WEIGHT in names(df))
        df = hcat(df,DataFrame(WEIGHT = ones(size(df,1))))
    end
    #probably not required the class / regression column
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
