global const rf_ver=v"0.0.10"

##
## Function for building a single tree.
##
function build_tree(method,alltrainingrefs,alltrainingweights,allregressionvalues,alltimevalues,alleventvalues,trainingdata,variables,types,varimp)
    leafnodesstats = Int[0, 0] # noleafnodes, noirregularleafnodes
    T1 = typeof(method.learningType) == Classifier ? Array{Int,1} : Int
    T2 = typeof(method.learningType) == Classifier ? Array{Float64,1} : Float64
    T3 = typeof(method.learningType) == Survival ? Array{Float64,1} : Float64
    PredictType = typeof(method.learningType) == Survival ? Array{Array{Float64,1},1} : Array{Float64,1}
    treeData = TreeData{T1, T2, T3}(0,alltrainingrefs,alltrainingweights,allregressionvalues,alltimevalues,alleventvalues,default_prediction(alltrainingweights,allregressionvalues,alltimevalues,alleventvalues,method))
    variableimportance = zeros(length(variables))
    tree = get_tree_node(treeData , variableimportance, leafnodesstats, trainingdata,variables,types,method,varimp,PredictType)
    if varimp
        variableimportance = variableimportance/sum(variableimportance)
    else
        variableimportance = :NONE
    end
    return tree, variableimportance, leafnodesstats[1], leafnodesstats[2]
end

function get_tree_node(treeData, variableimportance, leafnodesstats, trainingdata,variables,types,method,varimp,PredictType)
    if leaf_node(treeData, method)
        leafnodesstats[1] += 1
        return TreeNode{PredictType,Void}(:LEAF,make_leaf(treeData, method))
    else
        bestsplit = find_best_split(treeData,trainingdata,variables,types,method)
        if bestsplit == :NA
            leafnodesstats[1] += 1
            leafnodesstats[2] += 1
            return TreeNode{PredictType,Void}(:LEAF,make_leaf(treeData,method))
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
            defaultprediction = default_prediction(treeData.trainingweights,treeData.regressionvalues,treeData.timevalues,treeData.eventvalues,method)

            leftnode=get_tree_node(typeof(treeData)(treeData.depth+1,leftrefs,leftweights,leftregressionvalues,lefttimevalues,lefteventvalues,defaultprediction), variableimportance, leafnodesstats, trainingdata,variables,types,method,varimp, PredictType)
            rightnode=get_tree_node(typeof(treeData)(treeData.depth+1,rightrefs,rightweights,rightregressionvalues,righttimevalues,righteventvalues,defaultprediction), variableimportance, leafnodesstats, trainingdata,variables,types,method,varimp, PredictType)
            return TreeNode{PredictType,typeof(splitpoint)}(:NODE, varno,splittype,splitpoint,leftweight,
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
function make_prediction{T,S}(node::TreeNode{T,S},testdata,exampleno,prediction::T,weight=1.0)
    local examplevalue::S
    if node.nodeType == :LEAF
        prediction += weight*node.prediction
        return prediction
    else
        # varno, splittype, splitpoint, splitweight = node[1]
        examplevalue = testdata[node.varno][exampleno]
        if isna(examplevalue)
            prediction+=make_prediction(node.leftnode,testdata,exampleno,prediction,weight*node.leftweight)
            prediction+=make_prediction(node.rightnode,testdata,exampleno,prediction,weight*(1-node.leftweight))
            return prediction
        else
            if node.splittype == :NUMERIC
              nextnode=(examplevalue <= node.splitpoint)? node.leftnode: node.rightnode
            else #Catagorical
              nextnode=(examplevalue == node.splitpoint)? node.leftnode: node.rightnode
            end
            return make_prediction(nextnode,testdata,exampleno,prediction,weight)
        end
    end
end

# AMG: this is for running a single file. Note: we should allow data to be passed as argument in the next
# three functions !!!
function evaluate_method(;method = forest(),protocol = 10)
    println("Running experiment")
    method = fix_method_type(method)
    totaltime = @elapsed results = [run_single_experiment(protocol,[method])]
    classificationresults = [pt == :CLASS for (pt,f,r) in results]
    regressionresults = [pt == :REGRESSION for (pt,f,r) in results]
    present_results(sort(results[classificationresults]),[method],ignoredatasetlabel = true)
    present_results(sort(results[regressionresults]),[method],ignoredatasetlabel = true)
    println("Total time: $(round(totaltime,2)) s.")
end

function evaluate_methods(;methods = [forest()],protocol = 10)
    println("Running experiment")
    totaltime = @elapsed results = [run_single_experiment(protocol,methods)]
    classificationresults = [pt == :CLASS for (pt,f,r) in results]
    regressionresults = [pt == :REGRESSION for (pt,f,r) in results]
    present_results(sort(results[classificationresults]),methods,ignoredatasetlabel = true)
    present_results(sort(results[regressionresults]),methods,ignoredatasetlabel = true)
    println("Total time: $(round(totaltime,2)) s.")
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
            result = ("",results)
        elseif typeof(protocol) == Int64 || protocol == :cv
            results = run_cross_validation(protocol,methods)
            result = ("",results)
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
    alltrees = Array(Any,nocoworkers)
    index = 0
    for i = 1:nocoworkers
        alltrees[i] = model.trees[index+1:index+notrees[i]]
        index += notrees[i]
    end
    return alltrees
end

function generate_model(;method = forest())
    # Amg: assumes there is a data preloaded. need to be modified
    predictiontask = prediction_task(globaldata)
    if predictiontask == :NONE # FIXME: MOH We should not be doing this...probably DEAD code
        println("The loaded dataset is not on the correct format: CLASS/REGRESSION column missing")
        println("This may be due to an incorrectly specified separator, e.g., use: separator = \'\\t\'")
        result = :NONE
    else
        method = fix_method_type(method)
        classes = typeof(method.learningType) == Classifier ? unique(globaldata[:CLASS]) : Int[]
        nocoworkers = nprocs()-1
        numThreads = Threads.nthreads()
        treesandoobs = Array{Any,1}()
        if nocoworkers > 0
            notrees = getnotrees(method, nocoworkers)
            treesandoobs = pmap(generate_trees, [(method,classes,n,rand(1:1000_000_000)) for n in notrees])
        elseif numThreads > 1
            notrees = getnotrees(method, numThreads)
            treesandoobs = Array{Any,1}(length(notrees))
            Threads.@threads for n in notrees
                treesandoobs[Threads.threadid()] = generate_trees((method,classes,n,rand(1:1000_000_000)))
            end
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

function store_model(model::PredictionModel,file)
    s = open(file,"w")
    serialize(s,model)
    close(s)
    println("Model stored")
end

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
    pr = Array(Any,nprocs())
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

function update_workers()
    pr = Array(Any,nprocs())
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
