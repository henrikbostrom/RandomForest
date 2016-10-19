function generate_trees(Arguments::Tuple{LearningMethod{Survival},Any,Any,Any,Any})
    method,predictiontask,classes,notrees,randseed = Arguments
    s = size(globaldata,1)
    srand(randseed)
    trainingdata = globaldata
    trainingrefs = collect(1:size(trainingdata,1))
    trainingweights = trainingdata[:WEIGHT]
    regressionvalues = []
    timevalues = convert(Array,trainingdata[:TIME])
    eventvalues = convert(Array,trainingdata[:EVENT])
    oobpredictions = Array(Any,size(trainingdata,1))
    for i = 1:size(trainingdata,1)
        oobpredictions[i] = [0,0,0]
    end
    # starting from here till the end of the function is duplicated between here and the Classifier and Regressor dispatchers
    variables, types = get_variables_and_types(globaldata)
    modelsize = 0
    missingvalues, nonmissingvalues = find_missing_values(method,variables,trainingdata)
    newtrainingdata = transform_nonmissing_columns_to_arrays(method,variables,trainingdata,missingvalues)
    model = Array(Any,notrees)
    variableimportance = zeros(size(variables,1))
    for treeno = 1:notrees
        sample_replacements_for_missing_values!(method,newtrainingdata,trainingdata,predictiontask,variables,types,missingvalues,nonmissingvalues)
        model[treeno], treevariableimportance, noleafs, noirregularleafs = generate_tree(method,trainingrefs,trainingweights,regressionvalues,timevalues,eventvalues,newtrainingdata,variables,types,predictiontask,oobpredictions,varimp = true)
        modelsize += noleafs
        variableimportance += treevariableimportance
    end
   return (model,oobpredictions,variableimportance)
end

function find_missing_values(method::LearningMethod{Survival},variables,trainingdata)
    missingvalues = Array(Any,length(variables))
    nonmissingvalues = Array(Any,length(variables))
    for v = 1:length(variables)
        missingvalues[v] = Int[]
        nonmissingvalues[v] = Any[]
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
    newdata = Array(Any,length(variables))
    for v = 1:length(variables)
        if missingvalues[v] == []
            newdata[v] = convert(Array,trainingdata[variables[v]])
        else
            newdata[v] = trainingdata[variables[v]]
        end
    end
    return newdata
end

function sample_replacements_for_missing_values!(method::LearningMethod{Survival},newtrainingdata,trainingdata,predictiontask,variables,types,missingvalues,nonmissingvalues)
    for v = 1:length(variables)
        if missingvalues[v] != []
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
            newtrainingdata[v] = convert(Array,values)
        end
    end
end

function replacements_for_missing_values!(method::LearningMethod{Survival},newtestdata,testdata,variables,types,missingvalues,nonmissingvalues)
    for v = 1:length(variables)
        if missingvalues[v] != []
            values = convert(DataArray,testdata[variables[v]])
            for i in missingvalues[v]
                values[i] =  NA
            end
            newtestdata[v] = values
        end
    end
end

function generate_tree(method::LearningMethod{Survival},trainingrefs,trainingweights,regressionvalues,timevalues,eventvalues,trainingdata,variables,types,predictiontask,oobpredictions; varimp = false)
    if method.bagging
        newtrainingweights = zeros(length(trainingweights))
        if typeof(method.bagsize) == Int
            samplesize = method.bagsize
        else
            samplesize = convert(Int,round(length(trainingrefs)*method.bagsize))
        end
        selectedsample = rand(1:length(trainingrefs),samplesize)
        newtrainingweights[selectedsample] += 1.0
        nonzeroweights = [newtrainingweights[i] > 0 for i=1:length(trainingweights)]
        newtrainingrefs = trainingrefs[nonzeroweights]
        newtrainingweights = newtrainingweights[nonzeroweights]
        newtimevalues = timevalues[nonzeroweights]
        neweventvalues = eventvalues[nonzeroweights]
        model, variableimportance, noleafs, noirregularleafs = build_tree(method,newtrainingrefs,newtrainingweights,regressionvalues,newtimevalues,neweventvalues,trainingdata,variables,types,predictiontask,varimp)
        zeroweights = ~nonzeroweights
        oobrefs = trainingrefs[zeroweights]
        for oobref in oobrefs
            oobprediction = make_survival_prediction(model,trainingdata,oobref,timevalues[oobref],0)
            oobpredictions[oobref] += [1,oobprediction,oobprediction^2]
        end
    else
        model, variableimportance, noleafs, noirregularleafs = build_tree(method,trainingrefs,trainingweights,regressionvalues,timevalues,eventvalues,trainingdata,variables,types,predictiontask,varimp)
        for i = 1:size(trainingrefs,1)
            trainingref = trainingrefs[i]
            emptyleaf, leafstats = make_loo_prediction(model,trainingdata,trainingref,0)
            if emptyleaf
                oobprediction = leafstats[2]/leafstats[1]
            else
                oobprediction = (leafstats[2]-regressionvalues[i])/(leafstats[1]-1)
            end
            oobpredictions[trainingref] += [1,oobprediction,oobprediction^2]
        end
    end
    if varimp
        return model, variableimportance, noleafs, noirregularleafs, zeroweights
    else
        return model, noleafs, noirregularleafs, zeroweights
    end
end

## function make_loo_prediction(tree,testdata,exampleno,prediction)
##     nodeno = 1
##     nonterminal = true
##     emptyleaf = false
##     while nonterminal
##         node = tree[nodeno]
##         if node[1] == :LEAF
##             prediction = emptyleaf,node[2]
##             nonterminal = false
##         else
##             varno, splittype, splitpoint, splitweight = node[1]
##             examplevalue = testdata[varno][exampleno]
##             if splittype == :NUMERIC
##                 if examplevalue <= splitpoint
##                     leftchild = tree[node[2]]
##                     if leftchild[1] == :LEAF && leftchild[2][1] < 2.0
##                         nodeno = node[3]
##                         emptyleaf = true
##                     else
##                         nodeno = node[2]
##                     end
##                 else
##                     rightchild = tree[node[3]]
##                     if rightchild[1] == :LEAF && rightchild[2][1] < 2.0
##                         nodeno = node[2]
##                         emptyleaf = true
##                     else
##                         nodeno = node[3]
##                     end
##                 end
##             else
##                 if examplevalue == splitpoint
##                     leftchild = tree[node[2]]
##                     if leftchild[1] == :LEAF && leftchild[2][1] < 2.0
##                         nodeno = node[3]
##                         emptyleaf = true
##                     else
##                         nodeno = node[2]
##                     end
##                 else
##                     rightchild = tree[node[3]]
##                     if rightchild[1] == :LEAF && rightchild[2][1] < 2.0
##                         nodeno = node[2]
##                         emptyleaf = true
##                     else
##                         nodeno = node[3]
##                     end
##                 end
##             end
##         end
##     end
##     return prediction
## end

function default_prediction(trainingweights,regressionvalues,timevalues,eventvalues,predictiontask,method::LearningMethod{Survival})
    return generate_cumulative_hazard_function(trainingweights,timevalues,eventvalues)
end

function generate_cumulative_hazard_function(trainingweights,timevalues,eventvalues) # Assuming all values sorted according to time
    atrisk = sum(trainingweights)
    accweights = 0.0
    accevents = 0
#    cumulativehazard = 0.0
    survivalprob = 1.0
    chf = Any[]
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
            accevents = 0
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

function leaf_node(trainingweights,regressionvalues,eventvalues,predictiontask,depth,method::LearningMethod{Survival})
    if method.maxdepth > 0 && method.maxdepth == depth
        return true
    else
        noinstances = sum(trainingweights)
        if sum(trainingweights) >= 2*method.minleaf && sum(eventvalues) > 0
            return false
        else
            return true
        end
    end
end

function make_leaf(trainingweights,regressionvalues,timevalues,eventvalues,predictiontask,defaultprediction,method::LearningMethod{Survival})
    return generate_cumulative_hazard_function(trainingweights,timevalues,eventvalues)
end

function find_best_split(trainingrefs,trainingweights,regressionvalues,timevalues,eventvalues,trainingdata,variables,types,predictiontask,method::LearningMethod{Survival})
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
        if sum(trainingweights) <= splitsamplesize
            sampletrainingweights = trainingweights
            sampletrainingrefs = trainingrefs
            sampletimevalues = timevalues
            sampleeventvalues = eventvalues
        else
            sampletrainingweights = Array(Float64,splitsamplesize)
            sampletrainingrefs = Array(Float64,splitsamplesize)
            sampletimevalues = Array(Float64,splitsamplesize)
            sampleeventvalues = Array(Float64,splitsamplesize)
            for i = 1:splitsamplesize
                sampletrainingweights[i] = 1.0
                randindex = rand(1:length(trainingrefs))
                sampletrainingrefs[i] = trainingrefs[randindex]
                sampletimevalues[i] = timevalues[randindex]
                sampleeventvalues[i] = eventvalues[randindex]
            end
        end
    else
        sampletrainingrefs = trainingrefs
        sampletrainingweights = trainingweights
        sampletimevalues = timevalues
        sampleeventvalues = eventvalues
    end
    bestsplit = (-Inf,0,:NA,:NA,0.0)
    origweightsum = sum(sampletrainingweights)
    for v = 1:length(sampleselection)
        bestsplit = evaluate_variable_survival(bestsplit,sampleselection[v],variables[sampleselection[v]],types[sampleselection[v]],sampletrainingrefs,sampletrainingweights,
                                               origweightsum,sampletimevalues,sampleeventvalues,trainingdata,method)
    end
    splitvalue, varno, variable, splittype, splitpoint = bestsplit
    if variable == :NA
        return :NA
    else
        return (varno,variable,splittype,splitpoint)
    end
end

function evaluate_variable_survival(bestsplit,varno,variable,splittype,trainingrefs,trainingweights,origweightsum,timevalues,eventvalues,trainingdata,method)
    allvalues = trainingdata[varno][trainingrefs]
    if splittype == :CATEGORIC
        if method.randval
            bestsplit = evaluate_survival_categoric_variable_randval(bestsplit,varno,variable,splittype,timevalues,eventvalues,allvalues,trainingweights,origweightsum,method)
        else
            bestsplit = evaluate_survival_categoric_variable_allvals(bestsplit,varno,variable,splittype,timevalues,eventvalues,allvalues,trainingweights,origweightsum,method)
        end
    else # splittype == :NUMERIC
        if method.randval
            bestsplit = evaluate_survival_numeric_variable_randval(bestsplit,varno,variable,splittype,timevalues,eventvalues,allvalues,trainingweights,origweightsum,method)
        else
            bestsplit = evaluate_survival_numeric_variable_allvals(bestsplit,varno,variable,splittype,timevalues,eventvalues,allvalues,trainingweights,origweightsum,method)
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
    if leftweightsum >= method.minleaf && rightweightsum >= method.minleaf
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

function evaluate_survival_categoric_variable_allvals(bestsplit,varno,variable,splittype,timevalues,eventvalues,allvalues,trainingweights,origweightsum,method) # NOTE: to be fixed
    keys = unique(allvalues)
    for key in keys
        leftregressionsum = 0.0
        leftweightsum = 0.0
        for i = 1:length(allvalues)
            if allvalues[i] == key
                leftweightsum += trainingweights[i]
                leftregressionsum += trainingweights[i]*regressionvalues[i]
            end
        end
        rightregressionsum = origregressionsum-leftregressionsum
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
    if leftweightsum >= method.minleaf && rightweightsum >= method.minleaf
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

function evaluate_survival_numeric_variable_allvals(bestsplit,varno,variable,splittype,timevalues,eventvalues,allvalues,trainingweights,origweightsum,method) # NOTE: to be fixed!
    numericvalues = Dict{Any, Any}()
    for i = 1:length(allvalues)
        numericvalues[allvalues[i]] = get(numericvalues,allvalues[i],[0,0]) .+ [trainingweights[i]*regressionvalues[i],trainingweights[i]]
    end
    regressionsum = 0.0
    weightsum = 0.0
    for value in values(numericvalues)
        regressionsum += value[1]
        weightsum += value[2]
    end
    splitpoints = Array(Any,length(numericvalues),2)
    splitpoints[:,1] = collect(keys(numericvalues))
    splitpoints[:,2] = collect(values(numericvalues))
    splitpoints = sortrows(splitpoints,by=x->x[1])
    leftregressionsum = 0.0
    leftweightsum = 0.0
    for s = 1:size(splitpoints,1)-1
        weightandregressionsum = splitpoints[s,2]
        leftregressionsum += weightandregressionsum[1]
        leftweightsum += weightandregressionsum[2]
        rightregressionsum = origregressionsum-leftregressionsum
        rightweightsum = origweightsum-leftweightsum
        if leftweightsum >= method.minleaf && rightweightsum >= method.minleaf
            leftmean = leftregressionsum/leftweightsum
            rightmean = rightregressionsum/rightweightsum
            variancereduction = (origmean-leftmean)^2*leftweightsum+(origmean-rightmean)^2*rightweightsum
        else
            variancereduction = -Inf
        end
        if variancereduction > bestsplit[1]
            bestsplit = (variancereduction,varno,variable,splittype,splitpoints[s,1])
        end
    end
    return bestsplit
end

function make_split(method::LearningMethod{Survival},trainingrefs,trainingweights,regressionvalues,timevalues,eventvalues,trainingdata,predictiontask,bestsplit)
    (varno, variable, splittype, splitpoint) = bestsplit
    leftrefs = Int[]
    leftweights = Float64[]
    lefttimevalues = Float64[]
    lefteventvalues = Float64[]
    rightrefs = Int[]
    rightweights = Float64[]
    righttimevalues = Float64[]
    righteventvalues = Float64[]
    values = trainingdata[varno][trainingrefs]
    sumleftweights = 0.0
    sumrightweights = 0.0
    if splittype == :NUMERIC
      for r = 1:length(trainingrefs)
          ref = trainingrefs[r]
          if values[r] <= splitpoint
              push!(leftrefs,ref)
              push!(leftweights,trainingweights[r])
              sumleftweights += trainingweights[r]
              push!(lefttimevalues,timevalues[r])
              push!(lefteventvalues,eventvalues[r])
          else
              push!(rightrefs,ref)
              push!(rightweights,trainingweights[r])
              sumrightweights += trainingweights[r]
              push!(righttimevalues,timevalues[r])
              push!(righteventvalues,eventvalues[r])
          end
      end
  else
      for r = 1:length(trainingrefs)
          ref = trainingrefs[r]
          if values[r] == splitpoint
              push!(leftrefs,ref)
              push!(leftweights,trainingweights[r])
              sumleftweights += trainingweights[r]
              push!(lefttimevalues,timevalues[r])
              push!(lefteventvalues,eventvalues[r])
          else
              push!(rightrefs,ref)
              push!(rightweights,trainingweights[r])
              sumrightweights += trainingweights[r]
              push!(righttimevalues,timevalues[r])
              push!(righteventvalues,eventvalues[r])
          end
      end
  end
  leftweight = sumleftweights/(sumleftweights+sumrightweights)
  return leftrefs,leftweights,[],lefttimevalues,lefteventvalues,rightrefs,rightweights,[],righttimevalues,righteventvalues,leftweight
end

function make_survival_prediction(tree,testdata,exampleno,time,prediction)
    stack = Any[]
    nodeno = 1
    weight = 1.0
    push!(stack,(nodeno,weight))
    while stack != []
        nodeno, weight = pop!(stack)
        node = tree[nodeno]
        if node[1] == :LEAF
            prediction += weight*get_cumulative_hazard(node[2],time)
        else
            varno, splittype, splitpoint, splitweight = node[1]
            examplevalue = testdata[varno][exampleno]
            if isna(examplevalue)
                push!(stack,(node[2],weight*splitweight))
                push!(stack,(node[3],weight*(1-splitweight)))
            else
                if splittype == :NUMERIC
                    if examplevalue <= splitpoint
                        push!(stack,(node[2],weight))
                    else
                        push!(stack,(node[3],weight))
                    end
                else
                    if examplevalue == splitpoint
                        push!(stack,(node[2],weight))
                    else
                        push!(stack,(node[3],weight))
                    end
                end
            end
        end
    end
    return prediction
end

function generate_model_internal(method::LearningMethod{Survival},oobs,classes)
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
    ## elseif conformal == :isotonic
    ##     eds = Array(Float64,nooob,2)
    ##     eds[:,1] = ooberrors
    ##     eds[:,2] = deltas
    ##     eds = sortrows(eds,by=x->x[2])
    ##     isotonicthresholds = Any[]
    ##     mincalibrationsize = maximum([10;Int(floor(1/(round((1-method.confidence)*1000000)/1000000))-1)])
    ##     oobindex = 0
    ##     while size(eds,1)-oobindex >= 2*mincalibrationsize
    ##         isotonicgroup = eds[oobindex+1:oobindex+mincalibrationsize,:]
    ##         threshold = isotonicgroup[1,2]
    ##         isotonicgroup = sortrows(isotonicgroup,by=x->x[1],rev=true)
    ##         push!(isotonicthresholds,(threshold,isotonicgroup[1,1],isotonicgroup))
    ##         oobindex += mincalibrationsize
    ##     end
    ##     isotonicgroup = eds[oobindex+1:end,:]
    ##     threshold = isotonicgroup[1,2]
    ##     isotonicgroup = sortrows(isotonicgroup,by=x->x[1],rev=true)
    ##     thresholdindex = maximum([1,Int(floor(size(isotonicgroup,1)*(1-method.confidence)))])
    ##     push!(isotonicthresholds,(threshold,isotonicgroup[thresholdindex][1],isotonicgroup))
    ##     originalthresholds = copy(isotonicthresholds)
    ##     change = true
    ##     while change
    ##         change = false
    ##         counter = 1
    ##         while counter < size(isotonicthresholds,1) && ~change
    ##             if isotonicthresholds[counter][2] > isotonicthresholds[counter+1][2]
    ##                 newisotonicgroup = [isotonicthresholds[counter][3];isotonicthresholds[counter+1][3]]
    ##                 threshold = minimum(newisotonicgroup[:,2])
    ##                 newisotonicgroup = sortrows(newisotonicgroup,by=x->x[1],rev=true)
    ##                 thresholdindex = maximum([1,Int(floor(size(newisotonicgroup,1)*(1-method.confidence)))])
    ##                 splice!(isotonicthresholds,counter:counter+1,[(threshold,newisotonicgroup[thresholdindex][1],newisotonicgroup)])
    ##                 change = true
    ##             else
    ##                 counter += 1
    ##             end
    ##         end
    ##     end
    ##     conformalfunction = (:isotonic,isotonicthresholds,largestrange)
    end
    return oobperformance, conformalfunction
end

function apply_model(model::PredictionModel{Survival}; confidence = :std)
    nocoworkers = nprocs()-1
    if nocoworkers > 0
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
        results = pmap(apply_trees,[(model.method,model.classes,subtrees) for subtrees in alltrees])
        predictions = results[1][1]
        squaredpredictions = results[1][2]
        for r = 2:length(results)
            predictions += results[r][1]
            squaredpredictions += results[r][2]
        end
    else
        predictions, squaredpredictions = apply_trees((model.method,model.classes,model.trees))
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
        results = Array(Any,size(predictions,1))
        for i = 1:size(predictions,1)
            delta = squaredpredictions[i]-predictions[i]^2
            errorrange = alpha*(delta+0.01)
            results[i] = (predictions[i],[predictions[i]-errorrange/2,predictions[i]+errorrange/2])
        end
    ## elseif model.conformal[1] == :isotonic
    ##     isotonicthresholds = model.conformal[2]
    ##     errorrange = model.conformal[3]
    ##     results = Array(Any,size(predictions,1))
    ##     for i = 1:size(predictions,1)
    ##         delta = squaredpredictions[i]-predictions[i]^2
    ##             foundthreshold = false
    ##             thresholdcounter = size(isotonicthresholds,1)
    ##             while ~foundthreshold && thresholdcounter > 0
    ##                 if delta >= isotonicthresholds[thresholdcounter][1]
    ##                     errorrange = isotonicthresholds[thresholdcounter][2]*2
    ##                     foundthreshold = true
    ##                 else
    ##                     thresholdcounter -= 1
    ##                 end
    ##             end
    ##         results[i] = (predictions[i],[predictions[i]-errorrange/2,predictions[i]+errorrange/2])
    ##     end
    end
    return results
end

function apply_trees(Arguments::Tuple{LearningMethod{Survival},Any,Any})
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

