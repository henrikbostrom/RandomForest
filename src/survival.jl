##
## Functions to be executed on each worker
##

function generate_and_test_trees(Arguments::Tuple{LearningMethod{Survival},Symbol,Int64,Int64,Array{Int64,1}})
    method,experimentype,notrees,randseed,randomoobs = Arguments
    s = size(globaldata,1)
    srand(randseed)
    variables, types = get_variables_and_types(globaldata)
    if experimentype == :test
        model,oobpredictions,variableimportance, modelsize, noirregularleafs, randomclassoobs, oob =
            generate_trees((method,Int64[],notrees,randseed);curdata=globaldata[globaldata[:TEST] .== false,:], randomoobs=randomoobs, varimparg = false)
        testdata = globaldata[globaldata[:TEST] .== true,:]
        testmissingvalues, testnonmissingvalues = find_missing_values(method,variables,testdata)
        newtestdata = transform_nonmissing_columns_to_arrays(method,variables,testdata,testmissingvalues)
        replacements_for_missing_values!(method,newtestdata,testdata,variables,types,testmissingvalues,testnonmissingvalues)
        correctvalues = getDfArrayData(testdata[:EVENT])
        timevalues = getDfArrayData(testdata[:TIME])
        nopredictions = size(testdata,1)
        predictions = Array(Array{Float64,1},nopredictions)
        squaredpredictions = Array(Any,nopredictions)
        totalnotrees,squarederror = make_prediction_analysis(method, model, newtestdata, randomclassoobs, oob, predictions, squaredpredictions, correctvalues,timevalues)
        return (modelsize,predictions,[totalnotrees;squarederror],oobpredictions,squaredpredictions,noirregularleafs)
    else # experimentype == :cv
        folds = sort(unique(globaldata[:FOLD]))
        nofolds = length(folds)
        squarederrors = Array(Any,nofolds)
        predictions = Array(Any,size(globaldata,1))
        squaredpredictions = Array(Any,size(globaldata,1))
        oobpredictions = Array(Any,nofolds)
        modelsizes = Array(Int,nofolds)
        noirregularleafs = Array(Int,nofolds)
        testexamplecounter = 0
        foldno = 0
        for fold in folds
            foldno += 1
            trainingdata = globaldata[globaldata[:FOLD] .!= fold,:]
            testdata = globaldata[globaldata[:FOLD] .== fold,:]
            model,oobpredictions[foldno],variableimportance, modelsizes[foldno], noirregularleafs[foldno], randomclassoobs, oob =
                generate_trees((method,Int64[],notrees,randseed);curdata=trainingdata, randomoobs=size(randomoobs,1) > 0 ? randomoobs[foldno] : [], varimparg = false)
            testmissingvalues, testnonmissingvalues = find_missing_values(method,variables,testdata)
            newtestdata = transform_nonmissing_columns_to_arrays(method,variables,testdata,testmissingvalues)
            replacements_for_missing_values!(method,newtestdata,testdata,variables,types,testmissingvalues,testnonmissingvalues)
            correctvalues = getDfArrayData(testdata[:EVENT])
            timevalues = getDfArrayData(testdata[:TIME])
            totalnotrees,squarederror =
                make_prediction_analysis(method, model, newtestdata, randomclassoobs, oob, predictions, squaredpredictions, correctvalues,timevalues; predictionexamplecounter=testexamplecounter)
            testexamplecounter += size(testdata,1)

            squarederrors[foldno] = [totalnotrees;squarederror]
        end
        return (modelsizes,predictions,squarederrors,oobpredictions,squaredpredictions,noirregularleafs)
    end
end

function make_prediction_analysis(method::LearningMethod{Survival}, model, newtestdata, randomoobs, oob, predictions, squaredpredictions,correctvalues,timevalues; predictionexamplecounter = 0)
  squarederror = 0.0
  totalnotrees = 0
  for i = 1:length(correctvalues)
      correctvalue = correctvalues[i]
      prediction = 0.0
      squaredprediction = 0.0
      nosampledtrees = 0
      for t = 1:length(model)
          if method.modpred
              if oob[t][randomoobs[i]]
                  treeprediction = make_survival_prediction(model[t],newtestdata,i,timevalues[i],0)
                  prediction += treeprediction
                  squaredprediction += treeprediction^2
                  squarederror += (treeprediction-correctvalue)^2
                  nosampledtrees += 1
              end
          else
              treeprediction = make_survival_prediction(model[t],newtestdata,i,timevalues[i],0)
              prediction += treeprediction
              squaredprediction += treeprediction^2
              squarederror += (treeprediction-correctvalue)^2
          end
      end
      if ~method.modpred
          nosampledtrees = length(model)
      end
      predictionexamplecounter += 1
      totalnotrees += nosampledtrees
      predictions[predictionexamplecounter] = [nosampledtrees;prediction]
      squaredpredictions[predictionexamplecounter] = [nosampledtrees;squaredprediction]
    end
    return (totalnotrees,squarederror)
end

function generate_trees(Arguments::Tuple{LearningMethod{Survival},Array{Int,1},Int,Int};curdata=globaldata, randomoobs=[], varimparg = true)
    method,classes,notrees,randseed = Arguments
    s = size(curdata,1)
    srand(randseed)
    trainingdata = curdata
    trainingrefs = collect(1:size(trainingdata,1))
    trainingweights = getDfArrayData(trainingdata[:WEIGHT])
    regressionvalues = []
    timevalues = getDfArrayData(trainingdata[:TIME])
    eventvalues = getDfArrayData(trainingdata[:EVENT])
    oobpredictions = Array(Array{Float64,1},s)
    for i = 1:s
        oobpredictions[i] = zeros(3)
    end
    randomclassoobs = Array(Any,size(randomoobs,1))
    for i = 1:size(randomclassoobs,1)
        oobref = randomoobs[i]
        c = 1
        while oobref > size(trainingrefs[c],1)
            oobref -= size(trainingrefs[c],1)
            c += 1
        end
        randomclassoobs[i] = (c,oobref)
    end
    # starting from here till the end of the function is duplicated between here and the Classifier and Regressor dispatchers
    variables, types = get_variables_and_types(curdata)
    modelsize = 0
    noirregularleafs = 0
    missingvalues, nonmissingvalues = find_missing_values(method,variables,trainingdata)
    newtrainingdata = transform_nonmissing_columns_to_arrays(method,variables,trainingdata,missingvalues)
    model = Array(TreeNode,notrees)
    oob = Array(Array,notrees)
    variableimportance = zeros(size(variables,1))
    for treeno = 1:notrees
        sample_replacements_for_missing_values!(method,newtrainingdata,trainingdata,variables,types,missingvalues,nonmissingvalues)
        model[treeno], treevariableimportance, noleafs, treenoirregularleafs, oob[treeno] =
            generate_tree(method,trainingrefs,trainingweights,regressionvalues,timevalues,eventvalues,newtrainingdata,variables,types,oobpredictions,varimp = true)
        modelsize += noleafs
        noirregularleafs += treenoirregularleafs
        if (varimparg)
            variableimportance += treevariableimportance
        end
    end
    return (model,oobpredictions,variableimportance,modelsize,noirregularleafs,randomclassoobs,oob)
end

function find_missing_values(method::LearningMethod{Survival},variables,trainingdata)
    #NOTE: could be further improved with function barrier in val loop. Also identical to the regression version
    missingvalues = Array(Array{Int,1},length(variables))
    nonmissingvalues = Array(Array,length(variables))
    for v = 1:length(variables)
        variable = variables[v]
        missingvalues[v] = Int[]
        nonmissingvalues[v] = typeof(trainingdata[variable]).parameters[1][]
        variable = variables[v]
        if check_variable(variable)
            values = trainingdata[variable]
            for val = 1:length(values)
                value = values[val]
                if isna(value)
                    push!(missingvalues[v],val)
                else
                    push!(nonmissingvalues[v],value)
                end
            end
        end
    end
    return (missingvalues,nonmissingvalues)
end

function transform_nonmissing_columns_to_arrays(method::LearningMethod{Survival},variables,trainingdata,missingvalues)
    #NOTE: Identical to the regression version
    newdata = Array(Array,length(variables))
    for v = 1:length(variables)
        if isempty(missingvalues[v])
            newdata[v] = getDfArrayData(trainingdata[variables[v]])
        end
    end
    return newdata
end

function sample_replacements_for_missing_values!(method::LearningMethod{Survival},newtrainingdata,trainingdata,variables,types,missingvalues,nonmissingvalues)
    #NOTE: Identical to the regression version
    for v = 1:length(variables)
        if !isempty(missingvalues[v])
            values = trainingdata[variables[v]]
            if length(nonmissingvalues[v]) > 0
                for i in missingvalues[v]
                    newvalue = nonmissingvalues[v][rand(1:length(nonmissingvalues[v]))]
                    values[i] = newvalue
                end
            else
                if types[v] == :NUMERIC
                    newvalue = 0
                else
                    newvalue = ""
                end
                for i in missingvalues[v]
                    values[i] =  newvalue # NOTE: The variable (and type) should be removed
                end
            end
            newtrainingdata[v] = getDfArrayData(values)
        end
    end
end

function replacements_for_missing_values!(method::LearningMethod{Survival},newtestdata,testdata,variables,types,missingvalues,nonmissingvalues)
    #NOTE: Identical to the regression version
    for v = 1:length(variables)
        if !isempty(missingvalues[v])
            variableType = typeof(testdata[variables[v]]).parameters[1]
            values = convert(Array{Nullable{variableType},1},testdata[variables[v]],Nullable{variableType}())
            newtestdata[v] = values
        end
    end
end

function generate_tree(method::LearningMethod{Survival},trainingrefs,trainingweights,regressionvalues,timevalues,eventvalues,trainingdata,variables,types,oobpredictions; varimp = false)
    zeroweights = []
    if method.bagging
        newtrainingweights = zeros(length(trainingweights))
        if typeof(method.bagsize) == Int
            samplesize = method.bagsize
        else
            samplesize = round(Int,length(trainingrefs)*method.bagsize)
        end
        selectedsample = rand(1:length(trainingrefs),samplesize)
        newtrainingweights[selectedsample] += 1.0
        nonzeroweights = [newtrainingweights[i] > 0 for i=1:length(trainingweights)]
        newtrainingrefs = trainingrefs[nonzeroweights]
        newtrainingweights = newtrainingweights[nonzeroweights]
        newtimevalues = timevalues[nonzeroweights]
        neweventvalues = eventvalues[nonzeroweights]
        model, variableimportance, noleafs, noirregularleafs = build_tree(method,newtrainingrefs,newtrainingweights,regressionvalues,newtimevalues,neweventvalues,trainingdata,variables,types,varimp)
        zeroweights = ~nonzeroweights
        oobrefs = trainingrefs[zeroweights]
        for oobref in oobrefs
            oobprediction = make_survival_prediction(model,trainingdata,oobref,timevalues[oobref],0)
            oobpredictions[oobref] += [1,oobprediction,oobprediction^2]
        end
    else
        model, variableimportance, noleafs, noirregularleafs = build_tree(method,trainingrefs,trainingweights,regressionvalues,timevalues,eventvalues,trainingdata,variables,types,varimp)
        for i = 1:size(trainingrefs,1)
            trainingref = trainingrefs[i]
            oobprediction = make_survival_prediction(model,trainingdata,trainingref,timevalues[trainingref],0)
            oobpredictions[trainingref] += [1,oobprediction,oobprediction^2]
        end
    end
    # if varimp
    #     return model, variableimportance, noleafs, noirregularleafs, zeroweights
    # else
    #     return model, noleafs, noirregularleafs, zeroweights
    # end
    return model, variableimportance, noleafs, noirregularleafs, zeroweights
end

function default_prediction(trainingweights,regressionvalues,timevalues,eventvalues,method::LearningMethod{Survival})
    return generate_cumulative_hazard_function(trainingweights,timevalues,eventvalues)
end

function generate_cumulative_hazard_function(trainingweights,timevalues,eventvalues) # Assuming all values sorted according to time
    atrisk = sum(trainingweights)
    accweights = 0.0
    accevents = 0.0
#    cumulativehazard = 0.0
    survivalprob = 1.0
    chf = Array{Float64,1}[]
    for t = 1:size(timevalues,1)-1
        if timevalues[t] == timevalues[t+1]
            accweights += trainingweights[t]
            accevents += eventvalues[t]*trainingweights[t]
        elseif eventvalues[t] == 0
            atrisk -= trainingweights[t]
        else
#            cumulativehazard += (accevents+eventvalues[t]*trainingweights[t])/atrisk
            survivalprob *= 1-(accevents+eventvalues[t]*trainingweights[t])/atrisk
            accweights = 0.0
            accevents = 0.0
#            push!(chf,[t,cumulativehazard])
            push!(chf,[t,1-survivalprob])
            atrisk -= accweights+trainingweights[t]
        end
    end
    if accevents+eventvalues[end] > 0
#        cumulativehazard += (accevents+eventvalues[end]*trainingweights[end])/atrisk
        survivalprob *= 1-(accevents+eventvalues[end]*trainingweights[end])/atrisk
#        push!(chf,[timevalues[end],cumulativehazard])
        push!(chf,[timevalues[end],1-survivalprob])
    end
    return chf
end

function leaf_node(node,method::LearningMethod{Survival})
    if method.maxdepth > 0 && method.maxdepth == node.depth
        return true
    else
        noinstances = sum(node.trainingweights)
        if noinstances >= 2*method.minleaf && sum(node.eventvalues) > 1
            return false
        else
            return true
        end
    end
end

function make_leaf(node,method::LearningMethod{Survival}, parenttrainingweights)
    return generate_cumulative_hazard_function(node.trainingweights,node.timevalues,node.eventvalues)
end

function find_best_split(node,trainingdata,variables,types,method::LearningMethod{Survival})
    if (length(variables) == 0)
        method.randsub = :all
    end
    if method.randsub == :all
        sampleselection = collect(1:length(variables))
    elseif method.randsub == :default
        sampleselection = sample(1:length(variables),convert(Int,floor(1/3*length(variables))+1),replace=false)
    elseif method.randsub == :log2
        sampleselection = sample(1:length(variables),convert(Int,floor(log(2,length(variables)))+1),replace=false)
    elseif method.randsub == :sqrt
        sampleselection = sample(1:length(variables),convert(Int,floor(sqrt(length(variables)))),replace=false)
    else
        if typeof(method.randsub) == Int
            if method.randsub > length(variables)
                [1:length(variables)]
            else
                sampleselection = sample(1:length(variables),method.randsub,replace=false)
            end
        else
            sampleselection = sample(1:length(variables),convert(Int,floor(method.randsub*length(variables))+1),replace=false)
        end
    end
    if method.splitsample > 0
        splitsamplesize = method.splitsample
        if sum(node.trainingweights) <= splitsamplesize
            sampletrainingweights = node.trainingweights
            sampletrainingrefs = node.trainingrefs
            sampletimevalues = node.timevalues
            sampleeventvalues = node.eventvalues
        else
            sampletrainingweights = Array(Float64,splitsamplesize)
            sampletrainingrefs = Array(Float64,splitsamplesize)
            sampletimevalues = Array(Float64,splitsamplesize)
            sampleeventvalues = Array(Float64,splitsamplesize)
            for i = 1:splitsamplesize
                sampletrainingweights[i] = 1.0
                randindex = rand(1:length(node.trainingrefs))
                sampletrainingrefs[i] = node.trainingrefs[randindex]
                sampletimevalues[i] = node.timevalues[randindex]
                sampleeventvalues[i] = node.eventvalues[randindex]
            end
        end
    else
        sampletrainingrefs = node.trainingrefs
        sampletrainingweights = node.trainingweights
        sampletimevalues = node.timevalues
        sampleeventvalues = node.eventvalues
    end
    bestsplit = (-Inf,0,:NA,:NA,0.0)
    origweightsum = sum(sampletrainingweights)
    origeventsum = sum(sampleeventvalues .* sampletrainingweights)
    origmean = origeventsum/origweightsum
    for v = 1:length(sampleselection)
        bestsplit = evaluate_variable_survival(bestsplit,sampleselection[v],variables[sampleselection[v]],types[sampleselection[v]],sampletrainingrefs,sampletrainingweights,
                                               origweightsum,origeventsum,origmean,sampletimevalues,sampleeventvalues,trainingdata,method)
    end
    splitvalue, varno, variable, splittype, splitpoint = bestsplit
    if variable == :NA
        return :NA
    else
        return (varno,variable,splittype,splitpoint)
    end
end

function evaluate_variable_survival(bestsplit,varno,variable,splittype,trainingrefs,trainingweights,origweightsum,origeventsum,origmean,timevalues,eventvalues,trainingdata,method)
    allvalues = trainingdata[varno][trainingrefs]
    if splittype == :CATEGORIC
        if method.randval
            bestsplit = evaluate_survival_categoric_variable_randval(bestsplit,varno,variable,splittype,timevalues,eventvalues,allvalues,trainingweights,origweightsum,method)
        else
            bestsplit = evaluate_survival_categoric_variable_allvals(bestsplit,varno,variable,splittype,timevalues,eventvalues,allvalues,trainingweights,origweightsum,origmean,method)
        end
    else # splittype == :NUMERIC
        if method.randval
            bestsplit = evaluate_survival_numeric_variable_randval(bestsplit,varno,variable,splittype,timevalues,eventvalues,allvalues,trainingweights,origweightsum,method)
        else
            bestsplit = evaluate_survival_numeric_variable_allvals(bestsplit,varno,variable,splittype,timevalues,eventvalues,allvalues,trainingweights,origweightsum,origeventsum,origmean,method)
        end

    end
    return bestsplit
end

function evaluate_survival_categoric_variable_randval(bestsplit,varno,variable,splittype,timevalues,eventvalues,allvalues,trainingweights,origweightsum,method)
    key = allvalues[rand(1:end)]
    leftweights = Float64[]
    lefttimevalues = Float64[]
    lefteventvalues = Float64[]
    rightweights = Float64[]
    righttimevalues = Float64[]
    righteventvalues = Float64[]
    for i = 1:length(allvalues)
        if allvalues[i] == key
            push!(leftweights,trainingweights[i])
            push!(lefttimevalues,timevalues[i])
            push!(lefteventvalues,eventvalues[i])
        else
            push!(rightweights,trainingweights[i])
            push!(righttimevalues,timevalues[i])
            push!(righteventvalues,eventvalues[i])
        end
    end
    leftweightsum = sum(leftweights)
    rightweightsum = origweightsum-leftweightsum
    lefteventsum = sum(lefteventvalues)
    righteventsum = sum(righteventvalues)
    if leftweightsum >= method.minleaf && rightweightsum >= method.minleaf && lefteventsum > 0 && righteventsum > 0
        leftcumhazardfunction = generate_cumulative_hazard_function(leftweights,lefttimevalues,lefteventvalues)
        lefthazardscore = hazard_score(leftweights,lefttimevalues,lefteventvalues,leftcumhazardfunction)
        rightcumhazardfunction = generate_cumulative_hazard_function(rightweights,righttimevalues,righteventvalues)
        righthazardscore = hazard_score(rightweights,righttimevalues,righteventvalues,rightcumhazardfunction)
        totalscore = lefthazardscore+righthazardscore
        if -totalscore > bestsplit[1]
            bestsplit = (totalscore,varno,variable,splittype,key)
        end
    end
    return bestsplit
end

function hazard_score(weights,timevalues,eventvalues,cumhazardfunction)
    totalscore = 0.0
    for i = 1:size(weights,1)
        totalscore += weights[i]*abs(eventvalues[i]-get_cumulative_hazard(cumhazardfunction,timevalues[i]))
    end
    return totalscore
end

function get_cumulative_hazard(cumhazardfunction,timevalue)
    cumhazard = 0.0
    i = 1
    while i <= size(cumhazardfunction,1) && timevalue >= cumhazardfunction[i][1]
        cumhazard = cumhazardfunction[i][2]
        i += 1
    end
    return cumhazard
end

function evaluate_survival_categoric_variable_allvals(bestsplit,varno,variable,splittype,timevalues,eventvalues,allvalues,trainingweights,origweightsum,origmean,method) # NOTE: to be fixed
    keys = unique(allvalues)
    for key in keys
        leftregressionsum = 0.0
        leftweightsum = 0.0
        for i = 1:length(allvalues)
            if allvalues[i] == key
                leftweightsum += trainingweights[i]
                leftregressionsum += trainingweights[i]*eventvalues[i]
            end
        end
        rightregressionsum = origeventsum,-leftregressionsum
        rightweightsum = origweightsum-leftweightsum
        if leftweightsum >= method.minleaf && rightweightsum >= method.minleaf
            leftmean = leftregressionsum/leftweightsum
            rightmean = rightregressionsum/rightweightsum
            variancereduction = (origmean-leftmean)^2*leftweightsum+(origmean-rightmean)^2*rightweightsum
            if variancereduction > bestsplit[1]
                bestsplit = (variancereduction,varno,variable,splittype,key)
            end
        end
    end
    return bestsplit
end

function evaluate_survival_numeric_variable_randval(bestsplit,varno,variable,splittype,timevalues,eventvalues,allvalues,trainingweights,origweightsum,method)
    minval = minimum(allvalues)
    maxval = maximum(allvalues)
    splitpoint = minval+rand()*(maxval-minval)
    leftweights = Float64[]
    lefttimevalues = Float64[]
    lefteventvalues = Float64[]
    rightweights = Float64[]
    righttimevalues = Float64[]
    righteventvalues = Float64[]
    for i = 1:length(allvalues)
        if allvalues[i] <= splitpoint
            push!(leftweights,trainingweights[i])
            push!(lefttimevalues,timevalues[i])
            push!(lefteventvalues,eventvalues[i])
        else
            push!(rightweights,trainingweights[i])
            push!(righttimevalues,timevalues[i])
            push!(righteventvalues,eventvalues[i])
        end
    end
    leftweightsum = sum(leftweights)
    rightweightsum = origweightsum-leftweightsum
    lefteventsum = sum(lefteventvalues)
    righteventsum = sum(righteventvalues)
    if leftweightsum >= method.minleaf && rightweightsum >= method.minleaf && lefteventsum > 0 && righteventsum > 0
        leftcumhazardfunction = generate_cumulative_hazard_function(leftweights,lefttimevalues,lefteventvalues)
        lefthazardscore = hazard_score(leftweights,lefttimevalues,lefteventvalues,leftcumhazardfunction)
        rightcumhazardfunction = generate_cumulative_hazard_function(rightweights,righttimevalues,righteventvalues)
        righthazardscore = hazard_score(rightweights,righttimevalues,righteventvalues,rightcumhazardfunction)
        totalscore = lefthazardscore+righthazardscore
        if -totalscore > bestsplit[1]
            bestsplit = (totalscore,varno,variable,splittype,splitpoint)
        end
    end
    return bestsplit
end

function evaluate_survival_numeric_variable_allvals(bestsplit,varno,variable,splittype,timevalues,eventvalues,allvalues,trainingweights,origweightsum,origeventsum,origmean,method) # NOTE: to be fixed!
    #NOTE: Identical to the regression version evaluate_regression_numeric_variable_allvals
    numericvalues = Dict{typeof(allvalues[1]), Array{Float64,1}}()
    for i = 1:length(allvalues)
        numericvalues[allvalues[i]] = get(numericvalues,allvalues[i],[0,0]) .+ [trainingweights[i]*eventvalues[i],trainingweights[i]]
    end
    regressionsum = 0.0
    weightsum = 0.0
    for value in values(numericvalues)
        regressionsum += value[1]
        weightsum += value[2]
    end
    sortedkeys = sort(collect(keys(numericvalues)))
    leftregressionsum = 0.0
    leftweightsum = 0.0
    for s = 1:size(sortedkeys,1)-1
        weightandregressionsum = numericvalues[sortedkeys[s]]
        leftregressionsum += weightandregressionsum[1]
        leftweightsum += weightandregressionsum[2]
        rightregressionsum = origeventsum-leftregressionsum
        rightweightsum = origweightsum-leftweightsum
        if leftweightsum >= method.minleaf && rightweightsum >= method.minleaf
            leftmean = leftregressionsum/leftweightsum
            rightmean = rightregressionsum/rightweightsum
            variancereduction = (origmean-leftmean)^2*leftweightsum+(origmean-rightmean)^2*rightweightsum
        else
            variancereduction = -Inf
        end
        if variancereduction > bestsplit[1]
            bestsplit = (variancereduction,varno,variable,splittype,sortedkeys[s])
        end
    end
    return bestsplit
end

function make_split(method::LearningMethod{Survival},node,trainingdata,bestsplit)
    (varno, variable, splittype, splitpoint) = bestsplit
    leftrefs = Int[]
    leftweights = Float64[]
    lefttimevalues = Float64[]
    lefteventvalues = Float64[]
    rightrefs = Int[]
    rightweights = Float64[]
    righttimevalues = Float64[]
    righteventvalues = Float64[]
    values = trainingdata[varno][node.trainingrefs]
    sumleftweights = 0.0
    sumrightweights = 0.0
    op = splittype == :NUMERIC ? (<=) : (==)
    for r = 1:length(node.trainingrefs)
        ref = node.trainingrefs[r]
        if op(values[r], splitpoint)
            push!(leftrefs,ref)
            push!(leftweights,node.trainingweights[r])
            sumleftweights += node.trainingweights[r]
            push!(lefttimevalues,node.timevalues[r])
            push!(lefteventvalues,node.eventvalues[r])
        else
            push!(rightrefs,ref)
            push!(rightweights,node.trainingweights[r])
            sumrightweights += node.trainingweights[r]
            push!(righttimevalues,node.timevalues[r])
            push!(righteventvalues,node.eventvalues[r])
        end
    end
    leftweight = sumleftweights/(sumleftweights+sumrightweights)
    return leftrefs,leftweights,[],lefttimevalues,lefteventvalues,rightrefs,rightweights,[],righttimevalues,righteventvalues,leftweight
end

function make_survival_prediction{T,S}(node::TreeNode{T,S},testdata,exampleno,time,prediction,weight=1.0)
    if node.nodeType == :LEAF
        prediction += weight* get_cumulative_hazard(node.prediction,time)
        return prediction
    else
        # varno, splittype, splitpoint, splitweight = node[1]
        examplevalue::Nullable{S} = testdata[node.varno][exampleno]
        if isnull(examplevalue)
            prediction =make_survival_prediction(node.leftnode,testdata,exampleno,time,prediction,weight*node.leftweight)
            prediction =make_survival_prediction(node.rightnode,testdata,exampleno,time,prediction,weight*(1-node.leftweight))
            return prediction
        else
            if node.splittype == :NUMERIC
              nextnode=(get(examplevalue) <= node.splitpoint)? node.leftnode: node.rightnode
            else #Catagorical
              nextnode=(get(examplevalue) == node.splitpoint)? node.leftnode: node.rightnode
            end
            return make_survival_prediction(nextnode,testdata,exampleno,time,prediction,weight)
        end
    end
end

function generate_model_internal(method::LearningMethod{Survival},oobs,classes)
    #NOTE: Identical to the regression version
    oobpredictions = oobs[1]
    for r = 2:length(oobs)
        oobpredictions += oobs[r]
    end
    correcttrainingvalues = globaldata[:EVENT]
    oobse = 0.0
    nooob = 0
    ooberrors = Float64[]
    alphas = Float64[]
#            deltas = Float64[]
    for i = 1:length(correcttrainingvalues)
        oobpredcount = oobpredictions[i][1]
        if oobpredcount > 0.0
            ooberror = abs(correcttrainingvalues[i]-(oobpredictions[i][2]/oobpredcount))
            push!(ooberrors,ooberror)
            delta = (oobpredictions[i][3]/oobpredcount-(oobpredictions[i][2]/oobpredcount)^2)
            alpha = 2*ooberror/(delta+0.01)
            push!(alphas,alpha)
#                    push!(deltas,delta)
            oobse += ooberror^2
            nooob += 1
        end
    end
    oobperformance = oobse/nooob
    thresholdindex = Int(floor((nooob+1)*(1-method.confidence)))
    largestrange = maximum(correcttrainingvalues)-minimum(correcttrainingvalues)
    if method.conformal == :default
        conformal = :normalized
    else
        conformal = method.conformal
    end
    if conformal == :std
        if thresholdindex >= 1
            sortedooberrors = sort(ooberrors, rev=true)
            errorrange = minimum([largestrange,2*sortedooberrors[thresholdindex]])
        else
            errorrange = largestrange
        end
        conformalfunction = (:std,errorrange,largestrange,sortedooberrors)
    elseif conformal == :normalized
        sortedalphas = sort(alphas,rev=true)
        if thresholdindex >= 1
            alpha = sortedalphas[thresholdindex]
        else
            alpha = Inf
        end
        conformalfunction = (:normalized,alpha,largestrange,sortedalphas)
    end
    return oobperformance, conformalfunction
end

function apply_model_internal(model::PredictionModel{Survival}; confidence = :std)
    #NOTE: Identical to the regression version
    numThreads = Threads.nthreads()
    nocoworkers = nprocs()-1
    predictions = zeros(size(globaldata,1))
    squaredpredictions = zeros(size(globaldata,1))
    if nocoworkers > 0
        alltrees = getworkertrees(model, nocoworkers)
        results = pmap(apply_trees,[(model.method,[],subtrees) for subtrees in alltrees])
        for r = 1:length(results)
            predictions += results[r][1]
            squaredpredictions += results[r][2]
        end
    elseif numThreads > 1
        alltrees = getworkertrees(model, numThreads)
        predictionResults = Array{Array,1}(length(alltrees))
        squaredpredictionResults = Array{Array,1}(length(alltrees))
        Threads.@threads for subtrees in alltrees
            results = apply_trees((model.method,[],subtrees))
            predictionResults[Threads.threadid()] = results[1]
            squaredpredictionResults[Threads.threadid()] = results[2]
        end
        waitfor(predictionResults)
        waitfor(squaredpredictionResults)
        predictions = sum(predictionResults)
        squaredpredictions = sum(squaredpredictionResults)
    else
        results = apply_trees((model.method,[],model.trees))
        predictions += results[1]
        squaredpredictions += results[2]
    end
    predictions = predictions/model.method.notrees
    squaredpredictions = squaredpredictions/model.method.notrees
    if model.conformal[1] == :std
        if confidence == :std
            errorrange = model.conformal[2]
        else
            nooob = size(model.conformal[4],1)
            thresholdindex = Int(floor((nooob+1)*(1-confidence)))
            if thresholdindex >= 1
                errorrange = minimum([model.conformal[3],2*model.conformal[4][thresholdindex]])
            else
                errorrange = model.conformal[3]
            end
            results = [(p,[p-errorrange/2,p+errorrange/2]) for p in predictions]
        end
    elseif model.conformal[1] == :normalized
        if confidence == :std
            alpha = model.conformal[2]
        else
            nooob = size(model.conformal[4],1)
            thresholdindex = Int(floor((nooob+1)*(1-confidence)))
            if thresholdindex >= 1
                alpha = model.conformal[4][thresholdindex]
            else
                alpha = Inf
            end
        end
        results = Array(Tuple,size(predictions,1))
        for i = 1:size(predictions,1)
            delta = squaredpredictions[i]-predictions[i]^2
            errorrange = alpha*(delta+0.01)
            results[i] = (predictions[i],[predictions[i]-errorrange/2,predictions[i]+errorrange/2])
        end
    end
    return results
end

function apply_trees(Arguments::Tuple{LearningMethod{Survival},Array,Array})
    #NOTE: Identical to the regression version
    method, classes, trees = Arguments
    variables, types = get_variables_and_types(globaldata)
    testmissingvalues, testnonmissingvalues = find_missing_values(method,variables,globaldata)
    newtestdata = transform_nonmissing_columns_to_arrays(method,variables,globaldata,testmissingvalues)
    replacements_for_missing_values!(method,newtestdata,globaldata,variables,types,testmissingvalues,testnonmissingvalues)
    nopredictions = size(globaldata,1)
    timevalues = convert(Array,globaldata[:TIME])
    predictions = Array(Float64,nopredictions)
    squaredpredictions = Array(Float64,nopredictions)
    for i = 1:nopredictions
        predictions[i] = 0.0
        squaredpredictions[i] = 0.0
        for t = 1:length(trees)
            treeprediction = make_survival_prediction(trees[t],newtestdata,i,timevalues[i],0)
            predictions[i] += treeprediction
            squaredpredictions[i] += treeprediction^2
        end
    end
    results = (predictions,squaredpredictions)
    return results
end

function collect_results_split(method::LearningMethod{Survival}, results, time)
    modelsize = sum([result[1] for result in results])
    noirregularleafs = sum([result[6] for result in results])
    predictions = results[1][2]
    for r = 2:length(results)
        predictions += results[r][2]
    end
    nopredictions = size(predictions,1)
    predictions = [predictions[i][2]/predictions[i][1] for i = 1:nopredictions]
    prediction_results = Array(Tuple,size(predictions,1))
    if method.conformal == :default
        conformal = :normalized
    else
        conformal = method.conformal
    end
    if conformal == :normalized 
        squaredpredictions = results[1][5]
        for r = 2:length(results)
            squaredpredictions += results[r][5]
        end
        squaredpredictions = [squaredpredictions[i][2]/squaredpredictions[i][1] for i = 1:nopredictions]
    end
    oobpredictions = results[1][4]
    for r = 2:length(results)
        oobpredictions += results[r][4]
    end
    trainingdata = globaldata[globaldata[:TEST] .== false,:]
    correcttrainingvalues = trainingdata[:EVENT]
    oobse = 0.0
    nooob = 0
    ooberrors = Float64[]
    alphas = Float64[]
    deltas = Float64[]
    largestrange = 2*(maximum(correcttrainingvalues)-minimum(correcttrainingvalues))
    for i = 1:length(correcttrainingvalues)
        oobpredcount = oobpredictions[i][1]
        if oobpredcount > 0.0
            ooberror = abs(correcttrainingvalues[i]-(oobpredictions[i][2]/oobpredcount))
            push!(ooberrors,ooberror)
            if conformal == :normalized ## || conformal == :isotonic
                delta = (oobpredictions[i][3]/oobpredcount-(oobpredictions[i][2]/oobpredcount)^2)
                if 2*ooberror > largestrange
                    alpha = largestrange/(delta+0.01)
                else
                    alpha = 2*ooberror/(delta+0.01)
                end
                push!(alphas,alpha)
                push!(deltas,delta)
            end
            oobse += ooberror^2
            nooob += 1
        end
    end
    oobmse = oobse/nooob
    thresholdindex = Int(floor((nooob+1)*(1-method.confidence)))
    if conformal == :std
        if thresholdindex >= 1
            errorrange = minimum([largestrange,2*sort(ooberrors, rev=true)[thresholdindex]])
        else
            errorrange = largestrange
        end
        prediction_results = [(p,[p-errorrange/2,p+errorrange/2]) for p in predictions]
    elseif conformal == :normalized
        if thresholdindex >= 1
            alpha = sort(alphas, rev=true)[thresholdindex]
        else
            alpha = Inf
        end
    end
    testdata = globaldata[globaldata[:TEST] .== true,:]
    correctvalues = testdata[:EVENT]
    eventprobs = predictions[correctvalues .== 1]
    noeventprobs = predictions[correctvalues .!= 1]
    auc = calculate_auc(eventprobs,noeventprobs)
    mse = 0.0
    validity = 0.0
    rangesum = 0.0
    for i = 1:nopredictions
        error = abs(correctvalues[i]-predictions[i])
        mse += error^2
        if conformal == :normalized && method.modpred
            randomoob = randomoobs[i]
            oobpredcount = oobpredictions[randomoob][1]
            if oobpredcount > 0.0
                thresholdindex = Int(floor(nooob*(1-method.confidence)))
                if thresholdindex >= 1
                    alpha = sort(alphas[[1:randomoob-1;randomoob+1:end]], rev=true)[thresholdindex]
                else
                    alpha = Inf
                end
            else
                println("oobpredcount = $oobpredcount !!! This is almost surely impossible!")
            end
            delta = squaredpredictions[i]-predictions[i]^2
            errorrange = alpha*(delta+0.01)
            if isnan(errorrange) || errorrange > largestrange
                errorrange = largestrange
            end
        elseif conformal == :normalized
            delta = squaredpredictions[i]-predictions[i]^2
            errorrange = alpha*(delta+0.01)
            if isnan(errorrange) || errorrange > largestrange
                errorrange = largestrange
            end
        end
        if conformal != :std
            prediction_results[i] = (predictions[i],[predictions[i]-errorrange/2,predictions[i]+errorrange/2])
        end
        rangesum += errorrange
        if error <= errorrange/2
            validity += 1
        end
    end
    mse = mse/nopredictions
    esterr = oobmse-mse
    absesterr = abs(esterr)
    validity = validity/nopredictions
    region = rangesum/nopredictions
    corrcoeff = cor(correctvalues,predictions)
    totalnotrees = sum([results[r][3][1] for r = 1:length(results)])
    totalsquarederror = sum([results[r][3][2] for r = 1:length(results)])
    avmse = totalsquarederror/totalnotrees
    varmse = avmse-mse
    extratime = toq()
    return SurvivalResult(auc,mse,corrcoeff,avmse,varmse,esterr,absesterr,validity,region,modelsize,noirregularleafs,time+extratime), prediction_results
end

function collect_results_cross_validation(method::LearningMethod{Survival}, results, modelsizes, nofolds, conformal, time)
    folds = collect(1:nofolds)
    allnoirregularleafs = [result[6] for result in results]
    noirregularleafs = allnoirregularleafs[1]
    for r = 2:length(allnoirregularleafs)
        noirregularleafs += allnoirregularleafs[r]
    end
    predictions = results[1][2]
    for r = 2:length(results)
        predictions += results[r][2]
    end
    nopredictions = size(globaldata,1)
    testexamplecounter = 0
    predictions = [predictions[i][2]/predictions[i][1] for i = 1:nopredictions]
    prediction_results = Array(Tuple,size(predictions,1))
    auc = Array(Float64,nofolds)
    mse = Array(Float64,nofolds)
    corrcoeff = Array(Float64,nofolds)
    avmse = Array(Float64,nofolds)
    varmse = Array(Float64,nofolds)
    oobmse = Array(Float64,nofolds)
    esterr = Array(Float64,nofolds)
    absesterr = Array(Float64,nofolds)
    validity = Array(Float64,nofolds)
    region = Array(Float64,nofolds)
    foldno = 0
    if conformal == :normalized ## || conformal == :isotonic
        squaredpredictions = results[1][5]
        for r = 2:length(results)
            squaredpredictions += results[r][5]
        end
        squaredpredictions = [squaredpredictions[i][2]/squaredpredictions[i][1] for i = 1:nopredictions]
    end
    for fold in folds
        foldno += 1
        foldIndeces = globaldata[:FOLD] .== fold
        testdata = globaldata[foldIndeces,:]
        correctvalues = testdata[:EVENT]
        times = testdata[:TIME]
        correcttrainingvalues = globaldata[globaldata[:FOLD] .!= fold,:EVENT]
        oobpredictions = results[1][4][foldno]
        for r = 2:length(results)
            oobpredictions += results[r][4][foldno]
        end
        oobse = 0.0
        nooob = 0
        ooberrors = Float64[]
        alphas = Float64[]
        deltas = Float64[]
        maxcorrect = maximum(correcttrainingvalues)
        mincorrect = minimum(correcttrainingvalues)
        largestrange = 2*(maxcorrect-mincorrect)
        for i = 1:length(correcttrainingvalues)
            oobpredcount = oobpredictions[i][1]
            if oobpredcount > 0
                ooberror = abs(correcttrainingvalues[i]-(oobpredictions[i][2]/oobpredcount))
                push!(ooberrors,ooberror)
                if conformal == :normalized 
                    delta = (oobpredictions[i][3]/oobpredcount)-(oobpredictions[i][2]/oobpredcount)^2
                    if 2*ooberror > largestrange
                        alpha = largestrange/(delta+0.01)
                    else
                        alpha = 2*ooberror/(delta+0.01)
                    end
                    push!(alphas,alpha)
                    push!(deltas,delta)
                end
                oobse += ooberror^2
                nooob += 1
            end
        end
        thresholdindex = Int(floor((nooob+1)*(1-method.confidence)))
        if conformal == :std
            if thresholdindex >= 1
                errorrange = minimum([largestrange,2*sort(ooberrors, rev=true)[thresholdindex]])
            else
                errorrange = largestrange
            end
            prediction_results[foldIndeces] = [(p,[p-errorrange/2,p+errorrange/2]) for p in predictions[testexamplecounter+1:testexamplecounter+length(correctvalues)]]
        elseif conformal == :normalized
            if thresholdindex >= 1
                alpha = sort(alphas,rev=true)[thresholdindex]
            else
                alpha = Inf
            end
        end
        msesum = 0.0
        noinregion = 0.0
        rangesum = 0.0

        foldpredictions = predictions[testexamplecounter+1:testexamplecounter+length(correctvalues)]
        eventprobs = foldpredictions[correctvalues .== 1]
        noeventprobs = foldpredictions[correctvalues .!= 1]
        auc[foldno] = calculate_auc(eventprobs,noeventprobs)
        for i = 1:length(correctvalues)
            error = abs(correctvalues[i]-predictions[testexamplecounter+i])
            msesum += error^2
            if conformal == :normalized && method.modpred
                randomoob = randomoobs[foldno][i]
                oobpredcount = oobpredictions[randomoob][1]
                if oobpredcount > 0.0
                    thresholdindex = Int(floor(nooob*(1-method.confidence)))
                    if thresholdindex >= 1
                        alpha = sort(alphas[[1:randomoob-1;randomoob+1:end]], rev=true)[thresholdindex]
                    else
                        alpha = Inf
                    end
                else
                    println("oobpredcount = $oobpredcount !!! This is almost surely impossible!")
                end
                delta = (squaredpredictions[testexamplecounter+i]-predictions[testexamplecounter+i]^2)
                errorrange = alpha*(delta+0.01)
                if isnan(errorrange) || errorrange > largestrange
                    errorrange = largestrange
                end
            elseif conformal == :normalized
                delta = (squaredpredictions[testexamplecounter+i]-predictions[testexamplecounter+i]^2)
                errorrange = alpha*(delta+0.01)
                if isnan(errorrange) || errorrange > largestrange
                    errorrange = largestrange
                end
            end
            if conformal != :std
                curIndeces = find(foldIndeces)
                curIndex = curIndeces[i]
                prediction_results[curIndex] = (predictions[testexamplecounter+i],[predictions[testexamplecounter+i]-errorrange/2,predictions[testexamplecounter+i]+errorrange/2])
            end
            
            rangesum += errorrange
            if error <= errorrange/2
                noinregion += 1
            end
        end
        mse[foldno] = msesum/length(correctvalues)
        corrcoeff[foldno] = cor(correctvalues,predictions[testexamplecounter+1:testexamplecounter+length(correctvalues)])
        testexamplecounter += length(correctvalues)
        totalnotrees = sum([results[r][3][foldno][1] for r = 1:length(results)])
        totalsquarederror = sum([results[r][3][foldno][2] for r = 1:length(results)])
        avmse[foldno] = totalsquarederror/totalnotrees
        varmse[foldno] = avmse[foldno]-mse[foldno]
        oobmse[foldno] = oobse/nooob
        esterr[foldno] = oobmse[foldno]-mse[foldno]
        absesterr[foldno] = abs(oobmse[foldno]-mse[foldno])
        validity[foldno] = noinregion/length(correctvalues)
        region[foldno] = rangesum/length(correctvalues)
    end
    extratime = toq()
    return SurvivalResult(mean(auc),mean(mse),mean(corrcoeff),mean(avmse),mean(varmse),mean(esterr),mean(absesterr),mean(validity),mean(region),mean(modelsizes),mean(noirregularleafs),
                          time+extratime), prediction_results
end
