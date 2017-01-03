global const rf_ver=v"0.0.10"

"""
To load a dataset from a file:

    julia> load_sparse_data(<filename>, <labels_filename>, predictionType = <predictionType>, separator = <separator>, n = <numberOfFeatures>)

The arguments should be on the following format:

    filename : name of a file containing a sparse dataset (see format requirements above)
    labels_filename :  name of a file containing a vector of labels
    separator : single character (default = ' ')
    predictionType : one of :CLASS, :REGRESSION, or :SURVIVAL
    n : Number of features in the dataset (auto detected if not provided)
"""
function load_sparse_data(source, target; predictionType=:CLASS, separator = ' ', n = -1)
    global useSparseData = true
    df = readdlm(source, separator)
    if n == -1
        n = maximum([parse(split(filter(i->length(i) != 0, df[r,:])[end], ":")[1]) for r in 1:size(df,1)])
    end
    sparseMatrix = spzeros(size(df,1), n)
    for r = 1:size(df,1)
        d = filter(i->length(i) != 0, df[r,:])
        spd = Dict(parse(split(i,":")[1])=>parse(split(i,":")[2]) for i in d)
        sparseMatrix[r, :] = sparsevec(spd, n)
    end
    labels = readdlm(target, separator)[:,1]
    names = Any[1:n...]
    push!(names,predictionType)
    push!(names,:WEIGHT)
    # equivDataFrame = DataFrame(full(sparseMatrix))
    # equivDataFrame[predictionType] = labels
    # equivDataFrame = hcat(equivDataFrame,DataFrame(WEIGHT = ones(size(equivDataFrame,1))))
    # global globaldata = equivDataFrame
    global globaldata = SparseData(names,[sparseMatrix[:,i] for i = 1:n], labels, ones(length(labels)))
    initiate_workers()
    println("Data: $(source)")
end

"""
To load a dataset from a file or dataframe:

    julia> load_data(<filename>, separator = <separator>)
    julia> load_data(<dataframe>)

The arguments should be on the following format:

    filename : name of a file containing a dataset (see format requirements above)
    separator : single character (default = ',')
    dataframe : a dataframe where the column labels should be according to the format requirements above
"""
##
## Functions for working with a single dataset. Amg: loading data should move outside as well
##
function load_data(source; separator = ',', sparse=false)
    global useSparseData = sparse
    if typeof(source) == String
        global globaldata = read_data(source, separator=separator) # Made global to allow access from workers
        initiate_workers()
        println("Data: $(source)")
    elseif typeof(source) == DataFrame
        if ~(:WEIGHT in names(source))
            global globaldata = hcat(source,DataFrame(WEIGHT = ones(size(source,1))))
        else
            global globaldata = source # Made global to allow access from workers
        end
        initiate_workers()
        println("Data loaded")
    else
        println("Data can only be loaded from text files or DataFrames")
    end
end

##
## Function for building a single tree.
##
function build_tree(method,alltrainingrefs,alltrainingweights,allregressionvalues,alltimevalues,alleventvalues,trainingdata,variables,types,varimp)
    leafnodesstats = Int[0, 0] # noleafnodes, noirregularleafnodes
    T1 = typeof(method.learningType) == Classifier ? Array{Int,1} : Int
    T2 = typeof(method.learningType) == Classifier ? Array{Float64,1} : Float64
    PredictType = typeof(method.learningType) == Survival ? Array{Array{Float64,1},1} : Array{Float64,1}
    treeData = TreeData{T1, T2}(0,alltrainingrefs,alltrainingweights,allregressionvalues,alltimevalues,alleventvalues)
    variableimportance = zeros(length(variables))
    tree = get_tree_node(treeData , variableimportance, leafnodesstats, trainingdata,variables,types,method,[],varimp,PredictType)
    if varimp
        s = sum(variableimportance)
        if (s != 0)
            variableimportance = variableimportance/s
        end
    else
        variableimportance = :NONE
    end
    return tree, variableimportance, leafnodesstats[1], leafnodesstats[2]
end

function get_tree_node(treeData, variableimportance, leafnodesstats, trainingdata,variables,types,method,parenttrainingweights,varimp,PredictType)
    if leaf_node(treeData, method)
        leafnodesstats[1] += 1
        return TreeNode{PredictType,Void}(:LEAF,make_leaf(treeData, method, parenttrainingweights))
    else
        bestsplit = find_best_split(treeData,trainingdata,variables,types,method)
        if bestsplit == :NA
            leafnodesstats[1] += 1
            leafnodesstats[2] += 1
            return TreeNode{PredictType,Void}(:LEAF,make_leaf(treeData, method, parenttrainingweights))
        else
            leftrefs,leftweights,leftregressionvalues,lefttimevalues,lefteventvalues,rightrefs,rightweights,rightregressionvalues,righttimevalues,righteventvalues,leftweight = make_split(method,treeData,trainingdata,bestsplit)
            varno, variable, splittype, splitpoint = bestsplit
            if varimp
                if typeof(method.learningType) == Regressor #predictiontask == :REGRESSION
                    variableimp = variance_reduction(treeData.trainingweights,treeData.regressionvalues,leftweights,leftregressionvalues,rightweights,rightregressionvalues)
                elseif typeof(method.learningType) == Classifier #predictiontask == :CLASS
                    variableimp = information_gain(treeData.trainingweights,leftweights,rightweights)
                else #predictiontask == :SURVIVAL
                    variableimp = hazard_score_gain(treeData.trainingweights,treeData.timevalues,treeData.eventvalues,leftweights,lefttimevalues,lefteventvalues,rightweights,righttimevalues,righteventvalues)
                end
                variableimportance[varno] += variableimp
            end
            leftnode=get_tree_node(typeof(treeData)(treeData.depth+1,leftrefs,leftweights,leftregressionvalues,lefttimevalues,lefteventvalues), variableimportance, leafnodesstats, trainingdata,variables,types,method, treeData.trainingweights, varimp, PredictType)
            rightnode=get_tree_node(typeof(treeData)(treeData.depth+1,rightrefs,rightweights,rightregressionvalues,righttimevalues,righteventvalues), variableimportance, leafnodesstats, trainingdata,variables,types,method, treeData.trainingweights, varimp, PredictType)
            return TreeNode{Void,typeof(splitpoint)}(:NODE, varno,splittype,splitpoint,leftweight,
                  leftnode,rightnode)
        end
    end
end

function get_variables_and_types(trainingdata)
    allvariables = names(trainingdata)
    alltypes = eltypes(trainingdata)
    variablechecks = [check_variable(v) for v in allvariables]
    variables = allvariables[variablechecks]
    alltypes = alltypes[variablechecks]
    types::Array{Symbol,1} = [t <: Number ? :NUMERIC : :CATEGORIC for t in alltypes]
    return variables, types
end

function check_variable(v)
    if v in [:CLASS,:REGRESSION,:ID,:FOLD,:TEST,:WEIGHT,:TIME,:EVENT]
        return false
    elseif startswith(string(v),"IGNORE")
        return false
    else
        return true
    end
end

function variance_reduction(trainingweights,regressionvalues,leftweights,leftregressionvalues,rightweights,rightregressionvalues)
    origregressionsum = sum(regressionvalues)
    origweightsum = sum(trainingweights)
    leftregressionsum = sum(leftregressionvalues)
    leftweightsum = sum(leftweights)
    rightregressionsum = origregressionsum-leftregressionsum
    rightweightsum = origweightsum-leftweightsum
    origmean = origregressionsum/origweightsum
    leftmean = leftregressionsum/leftweightsum
    rightmean = rightregressionsum/rightweightsum
    variancereduction = (origmean-leftmean)^2*leftweightsum+(origmean-rightmean)^2*rightweightsum
    return variancereduction
end

function information_gain(trainingweights,leftweights,rightweights)
    origclasscounts = map(sum,trainingweights)
    orignoexamples = sum(origclasscounts)
    leftclasscounts = map(sum,leftweights)
    leftnoexamples = sum(leftclasscounts)
    rightclasscounts = map(sum,rightweights)
    rightnoexamples = sum(rightclasscounts)
    return -orignoexamples*entropy(origclasscounts,orignoexamples)+leftnoexamples*entropy(leftclasscounts,leftnoexamples)+rightnoexamples*entropy(rightclasscounts,rightnoexamples)
end

function hazard_score_gain(trainingweights,timevalues,eventvalues,leftweights,lefttimevalues,lefteventvalues,rightweights,righttimevalues,righteventvalues)
    origcumhazardfunction = generate_cumulative_hazard_function(trainingweights,lefttimevalues,lefteventvalues)
    orighazardscore = hazard_score(trainingweights,timevalues,eventvalues,origcumhazardfunction)
    leftcumhazardfunction = generate_cumulative_hazard_function(leftweights,lefttimevalues,lefteventvalues)
    lefthazardscore = hazard_score(leftweights,lefttimevalues,lefteventvalues,leftcumhazardfunction)
    rightcumhazardfunction = generate_cumulative_hazard_function(rightweights,righttimevalues,righteventvalues)
    righthazardscore = hazard_score(rightweights,righttimevalues,righteventvalues,rightcumhazardfunction)
    return orighazardscore-lefthazardscore-righthazardscore
end

##
## Function for making a prediction with a single tree
##
function make_prediction{T,S}(node::TreeNode{T,S},testdata,exampleno,prediction,weight=1.0)
    if node.nodeType == :LEAF
        prediction += weight*node.prediction
        return prediction
    else
        # varno, splittype, splitpoint, splitweight = node[1]
        examplevalue::Nullable{S} = testdata[node.varno][exampleno]
        if isnull(examplevalue)
            prediction = make_prediction(node.leftnode,testdata,exampleno,prediction,weight*node.leftweight)
            prediction = make_prediction(node.rightnode,testdata,exampleno,prediction,weight*(1-node.leftweight))
            return prediction
        else
            if node.splittype == :NUMERIC
              nextnode=(get(examplevalue) <= node.splitpoint)? node.leftnode: node.rightnode
            else #Catagorical
              nextnode=(get(examplevalue) == node.splitpoint)? node.leftnode: node.rightnode
            end
            return make_prediction(nextnode,testdata,exampleno,prediction,weight)
        end
    end
end

function getDfArrayData(da)
    return typeof(da) <: Array || typeof(da) <: SparseVector ? da : useSparseData ? sparsevec(da.data) : da.data
end
"""
To evaluate a method or several methods for generating a random forest:

    julia> evaluate_method(method = forest(...), protocol = <protocol>)
    julia> evaluate_methods(methods = [forest(...), ...], protocol = <protocol>)

The arguments should be on the following format:

    method : a call to forest(...) as explained above (default = forest())
    methods : a list of calls to forest(...) as explained above (default = [forest()])
    protocol : integer, float, :cv or :test as explained above (default = 10)
"""
# AMG: this is for running a single file. Note: we should allow data to be passed as argument in the next
# three functions !!!
function evaluate_method(;method = forest(),protocol = 10)
    println("Running experiment")
    method = fix_method_type(method)
    totaltime = @elapsed results = [run_single_experiment(protocol,[method])]
    classificationresults = [pt == :CLASS for (pt,f,r) in results]
    regressionresults = [pt == :REGRESSION for (pt,f,r) in results]
    survivalresults = [pt == :SURVIVAL for (pt,f,r) in results]
    present_results(sort(results[classificationresults]),[method],ignoredatasetlabel = true)
    present_results(sort(results[regressionresults]),[method],ignoredatasetlabel = true)
    present_results(sort(results[survivalresults]),[method],ignoredatasetlabel = true)
    println("Total time: $(round(totaltime,2)) s.")
    return results[1][3]["results"][1]
end

function evaluate_methods(;methods = [forest()],protocol = 10)
    println("Running experiment")
    methods = map(m->fix_method_type(m),methods)
    totaltime = @elapsed results = [run_single_experiment(protocol,methods)]
    classificationresults = [pt == :CLASS for (pt,f,r) in results]
    regressionresults = [pt == :REGRESSION for (pt,f,r) in results]
    survivalresults = [pt == :SURVIVAL for (pt,f,r) in results]
    present_results(sort(results[classificationresults]),methods,ignoredatasetlabel = true)
    present_results(sort(results[regressionresults]),methods,ignoredatasetlabel = true)
    present_results(sort(results[survivalresults]),[method],ignoredatasetlabel = true)
    println("Total time: $(round(totaltime,2)) s.")
    return results[1][3]
end

function run_single_experiment(protocol, methods)
    predictiontask = prediction_task(globaldata)
    if predictiontask == :NONE
        println("The loaded dataset is not on the correct format")
        println("This may be due to an incorrectly specified separator, e.g., use: separator = \'\\t\'")
        result = (:NONE,:NONE,:NONE)
    else
        if typeof(protocol) == Float64 || protocol == :test
            results = run_split(protocol,methods)
            result = (predictiontask, "",results)
        elseif typeof(protocol) == Int64 || protocol == :cv
            results = run_cross_validation(protocol,methods)
            result = (predictiontask, "",results)
        else
            throw("Unknown experiment protocol")
        end
        println("Completed experiment")
    end
    return result
end

function fix_method_type(method)
    predictiontask = prediction_task(globaldata)
    if typeof(method.learningType) == Undefined # only redefine method if it does not have proper type
        if predictiontask == :REGRESSION
            method = LearningMethod(Regressor(), (getfield(method,i) for i in fieldnames(method)[2:end])...)
        elseif predictiontask == :CLASS
            method = LearningMethod(Classifier(), (getfield(method,i) for i in fieldnames(method)[2:end])...)
        else # predictiontask == :SURVIVAL
            method = LearningMethod(Survival(), (getfield(method,i) for i in fieldnames(method)[2:end])...)
        end
    end
    return method
end

function getnotrees(method, nocoworkers)
    notrees = [div(method.notrees,nocoworkers) for i=1:nocoworkers]
    for i = 1:mod(method.notrees,nocoworkers)
        notrees[i] += 1
    end
    return notrees
end

function getworkertrees(model, nocoworkers)
    notrees = [div(model.method.notrees,nocoworkers) for i=1:nocoworkers]
    for i = 1:mod(model.method.notrees,nocoworkers)
        notrees[i] += 1
    end
    alltrees = Array(Array,nocoworkers)
    index = 0
    for i = 1:nocoworkers
        alltrees[i] = model.trees[index+1:index+notrees[i]]
        index += notrees[i]
    end
    return alltrees
end

function waitfor(var)
    while !all(i->isdefined(var,i), 1:length(var))
        sleep(0.005)
    end
end

"""
To generate a model from the loaded dataset:

    julia> m = generate_model(method = forest(...))                         

The argument should be on the following format:

    method : a call to forest(...) as explained above (default = forest())
"""
function generate_model(;method = forest())
    # Amg: assumes there is a data preloaded. need to be modified
    predictiontask = prediction_task(globaldata)
    if predictiontask == :NONE # FIXME: MOH We should not be doing this...probably DEAD code
        println("The loaded dataset is not on the correct format: CLASS/REGRESSION column missing")
        println("This may be due to an incorrectly specified separator, e.g., use: separator = \'\\t\'")
        result = :NONE
    else
        method = fix_method_type(method)
        classes = typeof(method.learningType) == Classifier ? getDfArrayData(unique(globaldata[:CLASS])) : Int[]
        nocoworkers = nprocs()-1
        numThreads = Threads.nthreads()
        if nocoworkers > 0
            notrees = getnotrees(method, nocoworkers)
            treesandoobs = pmap(generate_trees, [(method,classes,n,rand(1:1000_000_000)) for n in notrees])
        elseif numThreads > 1
            notrees = getnotrees(method, numThreads)
            treesandoobs = Array{Any,1}(length(notrees))
            Threads.@threads for n in notrees
                treesandoobs[Threads.threadid()] = generate_trees((method,classes,n,rand(1:1000_000_000)))
            end
            waitfor(treesandoobs)
        else
            notrees = [method.notrees]
            treesandoobs = generate_trees.([(method,classes,n,rand(1:1000_000_000)) for n in notrees])
        end
        trees = map(i->i[1], treesandoobs)
        oobs = map(i->i[2], treesandoobs)
        variableimportance = treesandoobs[1][3]
        for i = 2:length(treesandoobs)
            variableimportance += treesandoobs[i][3]
        end
        variableimportance = variableimportance/method.notrees
        variables, types = get_variables_and_types(globaldata)
        variableimportance = hcat(variables,variableimportance)
        oobperformance, conformalfunction = generate_model_internal(method, oobs, classes)
        result = PredictionModel{typeof(method.learningType)}(method,classes,rf_ver,oobperformance,variableimportance,vcat(trees...),conformalfunction)
    end
    return result
end

"""
To store a model in a file:

    julia> store_model(<model>, <file>)                              

The arguments should be on the following format:

    model : a generated or loaded model (see generate_model and load_model)
    file : name of file to store model in
"""
function store_model(model::PredictionModel,file)
    s = open(file,"w")
    serialize(s,model)
    close(s)
    println("Model stored")
end
"""
To load a model from file:

    julia> rf = load_model(<file>)                                  

The argument should be on the following format:

    file : name of file in which a model has been stored
"""
function load_model(file)
    s = open(file,"r")
    model = deserialize(s)
    close(s)
    println("Model loaded")
    return model
end

#=
Infers the prediction task from the data
=#
function prediction_task(method::LearningMethod{Regressor})
    return :REGRESSION
end

function prediction_task(method::LearningMethod{Classifier})
    return :CLASS
end

function prediction_task(method::LearningMethod{Survival})
    return :SURVIVAL
end

function prediction_task(data)
    allnames = names(data)
    if :CLASS in allnames
        return :CLASS
    elseif :REGRESSION in allnames
        return :REGRESSION
    elseif :TIME in allnames && :EVENT in allnames
        return :SURVIVAL
    else
        return :NONE
    end
end

function initiate_workers()
    pr = Array(Future,nprocs())
    for i = 2:nprocs()
        pr[i] = remotecall(load_global_dataset,i)
    end
    for i = 2:nprocs()
        wait(pr[i])
    end
end

function load_global_dataset()
    global globaldata = @fetchfrom(1,globaldata)
end

"""
To apply a model to loaded data:

    julia> apply_model(<model>, confidence = <confidence>)

The argument should be on the following format:

    model : a generated or loaded model (see generate_model and load_model)
    confidence : a float between 0 and 1 or :std (default = :std)
                 - probability of including the correct label in the prediction region
                 - :std means employing the same confidence level as used during training
"""
function apply_model(model::PredictionModel; confidence = :std)
    apply_model_internal(model, confidence=confidence)
end

function update_workers()
    pr = Array(Future,nprocs())
    for i = 2:nprocs()
        pr[i] = remotecall(update_global_dataset,i)
    end
    for i = 2:nprocs()
        wait(pr[i])
    end
end

function update_global_dataset()
    global globaltests = @fetchfrom(1,globaltests)
    global globaldata = hcat(globaltests,globaldata)
end
