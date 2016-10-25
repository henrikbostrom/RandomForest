# MOH FIXME:should use Julia standardized versioning instead
global majorversion = 0
global minorversion = 0
global patchversion = 10

type Node
    depth::Int
    nodenumber::Int
    trainingrefs::Array{Any,1}
    trainingweights::Array{Any,1}
    regressionvalues::Array{Float64,1} # regression
    timevalues::Array{Float64,1} # survival analysis
    eventvalues::Array{Float64,1} # survivial analysis
    defaultprediction::Array{Any,1}
end
##
## Function for building a single tree.
##

function build_tree(method,alltrainingrefs,alltrainingweights,allregressionvalues,alltimevalues,alleventvalues,trainingdata,variables,types,predictiontask,varimp)
    tree = Any[]
    depth = 0
    nodeno = 1
    noleafnodes = 0
    noirregularleafnodes = 0
    stack = Node[Node(depth,nodeno,alltrainingrefs,alltrainingweights,allregressionvalues,alltimevalues,alleventvalues,default_prediction(alltrainingweights,allregressionvalues,alltimevalues,alleventvalues,predictiontask,method))]
    nextavailablenodeno = 2
    if varimp
        variableimportance = zeros(length(variables))
    end
    while stack != []
        node = pop!(stack)
        if leaf_node(node.trainingweights,node.regressionvalues,node.eventvalues,predictiontask,node.depth,method)
            leaf = (:LEAF,make_leaf(node.trainingweights,node.regressionvalues,node.timevalues,node.eventvalues,predictiontask,node.defaultprediction,method))
            push!(tree,(node.nodenumber,leaf))
            noleafnodes += 1
        else
            bestsplit = find_best_split(node.trainingrefs,node.trainingweights,node.regressionvalues,node.timevalues,node.eventvalues,trainingdata,variables,types,predictiontask,method)
            if bestsplit == :NA
                leaf = (:LEAF,make_leaf(node.trainingweights,node.regressionvalues,node.timevalues,node.eventvalues,predictiontask,node.defaultprediction,method))
                push!(tree,(node.nodenumber,leaf))
                noleafnodes += 1
                noirregularleafnodes += 1
            else
                leftrefs,leftweights,leftregressionvalues,lefttimevalues,lefteventvalues,rightrefs,rightweights,rightregressionvalues,righttimevalues,righteventvalues,leftweight =
                    make_split(method,node.trainingrefs,node.trainingweights,node.regressionvalues,node.timevalues,node.eventvalues,trainingdata,predictiontask,bestsplit)
                varno, variable, splittype, splitpoint = bestsplit
                if varimp
                    if typeof(method.learningType) == Regressor #predictiontask == :REGRESSION
                        variableimp = variance_reduction(node.trainingweights,node.regressionvalues,leftweights,leftregressionvalues,rightweights,rightregressionvalues)
                    elseif typeof(method.learningType) == Classifier #predictiontask == :CLASS
                        variableimp = information_gain(node.trainingweights,leftweights,rightweights)
                    else #predictiontask == :SURVIVAL
                        variableimp = hazard_score_gain(node.trainingweights,node.timevalues,node.eventvalues,leftweights,lefttimevalues,lefteventvalues,rightweights,righttimevalues,righteventvalues)
                    end
                    variableimportance[varno] += variableimp
                end
                push!(tree,(node.nodenumber,((varno,splittype,splitpoint,leftweight),nextavailablenodeno,nextavailablenodeno+1)))
                defaultprediction = default_prediction(node.trainingweights,node.regressionvalues,node.timevalues,node.eventvalues,predictiontask,method)
                push!(stack,Node(node.depth+1,nextavailablenodeno,leftrefs,leftweights,leftregressionvalues,lefttimevalues,lefteventvalues,defaultprediction))
                push!(stack,Node(node.depth+1,nextavailablenodeno+1,rightrefs,rightweights,rightregressionvalues,righttimevalues,righteventvalues,defaultprediction))
                nextavailablenodeno += 2
            end
        end
    end
    if varimp
        variableimportance = variableimportance/sum(variableimportance)
    else
        variableimportance = :NONE
    end
    return restructure_tree(tree), variableimportance, noleafnodes, noirregularleafnodes
end

function get_variables_and_types(trainingdata)
    allvariables = names(trainingdata)
    alltypes = eltypes(trainingdata)
    variablechecks = [check_variable(v) for v in allvariables]
    variables = allvariables[variablechecks]
    alltypes = alltypes[variablechecks]
    types = Array(Any,length(alltypes))
    for t=1:length(alltypes)
        if alltypes[t] in [Float64,Float32,Int,Int32]
            types[t] = :NUMERIC
        else
            types[t] = :CATEGORIC
        end
    end
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
    origclasscounts = [sum(trainingweights[c]) for c=1:size(trainingweights,1)]
    orignoexamples = sum(origclasscounts)
    leftclasscounts = [sum(leftweights[c]) for c=1:size(leftweights,1)]
    leftnoexamples = sum(leftclasscounts)
    rightclasscounts = [sum(rightweights[c]) for c=1:size(rightweights,1)]
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

function restructure_tree(tree)
    nonodes = size(tree,1)
    newtree = Array(Any,nonodes)
    for i = 1:nonodes
        nodeno, node = tree[i]
        newtree[nodeno] = node
    end
    return newtree
end

type StackNode
    nodeno::Int
    weight::Float64
end

##
## Function for making a prediction with a single tree
##

function make_prediction(tree,testdata,exampleno,prediction,nodeno=1,weight=1.0)
    while true
        node = tree[nodeno]
        if node[1] == :LEAF
            prediction += weight*node[2]
            return prediction
        else
            # varno, splittype, splitpoint, splitweight = node[1]
            examplevalue = testdata[node[1][1]][exampleno]
            if isna(examplevalue)
                prediction+=make_prediction(tree,testdata,exampleno,prediction,node[2],weight*node[1][4])
                prediction+=make_prediction(tree,testdata,exampleno,prediction,node[3],weight*(1-node[1][4]))
                return prediction
            else
                if node[1][2] == :NUMERIC
                  nodeno=(examplevalue <= node[1][3])? node[2]: node[3]
                else #Catagorical
                  nodeno=(examplevalue == node[1][3])? node[2]: node[3]
                end
            end
        end
    end
    return prediction
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
            results = run_split(protocol,predictiontask,methods)
            result = (predictiontask,"",results)
        elseif typeof(protocol) == Int64 || protocol == :cv
            results = run_cross_validation(protocol,predictiontask,methods)
            result = (predictiontask,"",results)
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

function generate_model(;method = forest())
    # Amg: assumes there is a data preloaded. need to be modified
    predictiontask = prediction_task(globaldata)
    if predictiontask == :NONE # FIXME: MOH We should not be doing this...probably DEAD code
        println("The loaded dataset is not on the correct format: CLASS/REGRESSION column missing")
        println("This may be due to an incorrectly specified separator, e.g., use: separator = \'\\t\'")
        result = :NONE
    else
        method = fix_method_type(method)
        classes = typeof(method.learningType) == Classifier ? unique(globaldata[:CLASS]) : []
        nocoworkers = nprocs()-1
        numThreads = Threads.nthreads()
        treesandoobs = Array{Any,1}()
        if nocoworkers > 0
            notrees = [div(method.notrees,nocoworkers) for i=1:nocoworkers]
            for i = 1:mod(method.notrees,nocoworkers)
                notrees[i] += 1
            end
            treesandoobs = pmap(generate_trees, [(method,predictiontask,classes,n,rand(1:1000_000_000)) for n in notrees])
        elseif numThreads > 1
            notrees = [div(method.notrees,numThreads) for i=1:numThreads]
            for i = 1:mod(method.notrees,numThreads)
                notrees[i] += 1
            end
            treesandoobs = Array{Any,1}(length(notrees))
            Threads.@threads for n in notrees
                treesandoobs[Threads.threadid()] = generate_trees((method,predictiontask,classes,n,rand(1:1000_000_000)))
            end
        else
            notrees = [method.notrees]
            treesandoobs = generate_trees.([(method,predictiontask,classes,n,rand(1:1000_000_000)) for n in notrees])
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
        result = PredictionModel{typeof(method.learningType)}(method,classes,(majorversion,minorversion,patchversion),oobperformance,variableimportance,vcat(trees...),conformalfunction)
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
