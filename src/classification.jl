function generate_trees(Arguments::Tuple{LearningMethod{Classifier},Any,Any,Any,Any})
    method,predictiontask,classes,notrees,randseed = Arguments
    s = size(globaldata,1)
    srand(randseed)
    noclasses = length(classes)
    trainingdata = groupby(globaldata, :CLASS)
    trainingrefs = Array(Array{Int,1},noclasses)
    trainingweights = Array(Array{Float64,1},noclasses)
    oobpredictions = Array(Any,noclasses)
    emptyprediction = [0; zeros(noclasses)]
    for c = 1:noclasses
        trainingrefs[c] = collect(1:size(trainingdata[c],1))
        trainingweights[c] = trainingdata[c][:WEIGHT]
        oobpredictions[c] = Array(Any,size(trainingdata[c],1))
        for i = 1:size(trainingdata[c],1)
            oobpredictions[c][i] = emptyprediction
        end
    end
    regressionvalues = []
    timevalues = []
    eventvalues = []
    # starting from here till the end of the function is duplicated between here and the Regressor and Survival dispatchers
    # need to be cleaned
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

function find_missing_values(method::LearningMethod{Classifier},variables,trainingdata)
    noclasses = length(trainingdata)
    missingvalues = Array(Any,noclasses)
    nonmissingvalues = Array(Any,noclasses)
    for c = 1:noclasses
        missingvalues[c] = Array(Any,length(variables))
        nonmissingvalues[c] = Array(Any,length(variables))
        for v = 1:length(variables)
            missingvalues[c][v] = Int[]
            nonmissingvalues[c][v] = Any[]
            variable = variables[v]
            if check_variable(variable)
                values = trainingdata[c][variable]
                for val = 1:length(values)
                    value = values[val]
                    if isna(value)
                        push!(missingvalues[c][v],val)
                    else
                        push!(nonmissingvalues[c][v],value)
                    end
                end
                # missingvalues[c][v] = filter(isna, convert(Array,trainingdata[c][variable]))
                # nonmissingvalues[c][v] = filter(val->!isna(val), convert(Array, trainingdata[c][variable]))
            end
        end
    end
    return (missingvalues,nonmissingvalues)
end

function transform_nonmissing_columns_to_arrays(method::LearningMethod{Classifier},variables,trainingdata,missingvalues)
    noclasses = length(trainingdata)
    newdata = Array(Any,noclasses)
    for c = 1:noclasses
        newdata[c] = Array(Any,length(variables))
        for v = 1:length(variables)
            if missingvalues[c][v] == []
                newdata[c][v] = convert(Array,trainingdata[c][variables[v]])
            else
                newdata[c][v] = trainingdata[c][variables[v]]
            end
        end
    end
    return newdata
end

function sample_replacements_for_missing_values!(method::LearningMethod{Classifier},newtrainingdata,trainingdata,predictiontask,variables,types,missingvalues,nonmissingvalues)
    noclasses = size(newtrainingdata,1)
    for c = 1:noclasses
        for v = 1:length(variables)
            if missingvalues[c][v] != []
                values = trainingdata[c][variables[v]]
                valuefrequencies = map(Float64,[length(nonmissingvalues[cl][v]) for cl = 1:noclasses])
                if sum(valuefrequencies) > 0
                    for i in missingvalues[c][v]
                        sampleclass = wsample(1:noclasses,valuefrequencies)
                        newvalue = nonmissingvalues[sampleclass][v][rand(1:length(nonmissingvalues[sampleclass][v]))]
                        values[i] = newvalue
                    end
                else
                    if types[v] == :NUMERIC
                        newvalue = 0
                    else
                        newvalue = ""
                    end
                    for i in missingvalues[c][v]
                        values[i] =  newvalue # NOTE: The variable (and type) should be removed
                    end
                end
                newtrainingdata[c][v] = convert(Array,values)
            end
        end
    end
end

function replacements_for_missing_values!(method::LearningMethod{Classifier},newtestdata,testdata,variables,types,missingvalues,nonmissingvalues)
    noclasses = size(newtestdata,1)
    for c = 1:noclasses
        for v = 1:length(variables)
            if missingvalues[c][v] != []
                values = convert(DataArray,testdata[c][variables[v]])
                for i in missingvalues[c][v]
                    values[i] =  NA
                end
                newtestdata[c][v] = values
            end
        end
    end
end

function generate_tree(method::LearningMethod{Classifier},trainingrefs,trainingweights,regressionvalues,timevalues,eventvalues,trainingdata,variables,types,predictiontask,oobpredictions; varimp = false)
    noclasses = length(trainingweights)
    zeroweights = Array(Any,noclasses)
    if method.bagging
        classweights = map(Float64,[length(t) for t in trainingrefs])
        if typeof(method.bagsize) == Int
            samplesize = method.bagsize
        else
            samplesize = convert(Int,round(sum(classweights)*method.bagsize))
        end
        newtrainingweights = Array(Array{Float64,1},noclasses)
        newtrainingrefs = Array(Array{Int,1},noclasses)
        for c = 1:noclasses
            newtrainingweights[c] = zeros(length(trainingweights[c]))
        end
        for i = 1:samplesize
            class = wsample(1:noclasses,classweights)
            newtrainingweights[class][rand(1:end)] += 1.0
        end
        for c = 1:noclasses
            nonzeroweights = [newtrainingweights[c][i] > 0 for i=1:length(newtrainingweights[c])]
            zeroweights[c] = ~nonzeroweights
            newtrainingrefs[c] = trainingrefs[c][nonzeroweights]
            newtrainingweights[c] = newtrainingweights[c][nonzeroweights]
        end
        model, variableimportance, noleafs, noirregularleafs = build_tree(method,newtrainingrefs,newtrainingweights,regressionvalues,timevalues,eventvalues,trainingdata,variables,types,predictiontask,varimp)
        for c = 1:noclasses
            oobrefs = trainingrefs[c][zeroweights[c]]
            for oobref in oobrefs
                oobpredictions[c][oobref] += [1;make_prediction(model,trainingdata[c],oobref,zeros(noclasses))]
            end
        end
    else
        model, variableimportance, noleafs, noirregularleafs = build_tree(method,trainingrefs,trainingweights,regressionvalues,timevalues,eventvalues,trainingdata,variables,types,predictiontask,varimp)
    end
    if varimp
        return model, variableimportance, noleafs, noirregularleafs, zeroweights
    else
        return model, noleafs, noirregularleafs, zeroweights
    end
end

function default_prediction(trainingweights,regressionvalues,timevalues,eventvalues,method::LearningMethod{Classifier})
    noclasses = size(trainingweights,1)
    classcounts = map(sum, trainingweights)
    noinstances = sum(classcounts)
    if method.laplace
        return [(classcounts[i]+1)/(noinstances+noclasses) for i=1:noclasses]
    else
        if noinstances > 0
            return [classcounts[i]/noinstances for i=1:noclasses]
        else
            return [1/noclasses for i=1:noclasses]
        end
    end
end

function leaf_node(node,method::LearningMethod{Classifier})
    if method.maxdepth > 0 && method.maxdepth == node.depth
        return true
    else
        noclasses = size(node.trainingweights,1)
        classweights = map(sum, node.trainingweights)
        noinstances = sum(classweights)
        if noinstances >= 2*method.minleaf
            i = 1
            nonzero = 0
            while i <= noclasses &&  nonzero < 2
                if classweights[i] > 0
                    nonzero += 1
                end
                i += 1
            end
            if nonzero >= 2
                return false
            else
                return true
            end
        else
            return true
        end
    end
end

function make_leaf(node,method::LearningMethod{Classifier})
    noclasses = size(node.trainingweights,1)
    classcounts = zeros(noclasses)
    for i=1:noclasses
        classcounts[i] = sum(node.trainingweights[i])
    end
    noinstances = sum(classcounts)
    prediction = node.defaultprediction
    if noinstances > 0
        if method.laplace
            prediction = [(classcounts[i]+1)/(noinstances+noclasses) for i=1:noclasses]
        else
            prediction = [classcounts[i]/noinstances for i=1:noclasses]
        end
    end
    return prediction
end

function find_best_split(node,trainingdata,variables,types,method::LearningMethod{Classifier})
    if method.randsub == :all
        sampleselection = collect(1:length(variables))
    elseif method.randsub == :default
        sampleselection = sample(1:length(variables),convert(Int,floor(log(2,length(variables)))+1),replace=false)
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
        noclasses = size(node.trainingrefs,1)
        sampleregressionvalues = node.regressionvalues
        sampletrainingweights = Array(Any,noclasses)
        sampletrainingrefs = Array(Any,noclasses)
        for c = 1:noclasses
            if sum(node.trainingweights[c]) <= splitsamplesize
                sampletrainingweights[c] = node.trainingweights[c]
                sampletrainingrefs[c] = node.trainingrefs[c]
            else
                sampletrainingweights[c] = Array(Float64,splitsamplesize)
                sampletrainingrefs[c] = Array(Float64,splitsamplesize)
                for i = 1:splitsamplesize
                    sampletrainingweights[c][i] = 1.0
                    sampletrainingrefs[c][i] = node.trainingrefs[c][rand(1:end)]
                end
            end
        end
    else
        sampletrainingrefs = node.trainingrefs
        sampletrainingweights = node.trainingweights
        sampleregressionvalues = node.regressionvalues
    end
    bestsplit = (-Inf,0,:NA,:NA,0.0)
    noclasses = size(node.trainingrefs,1)
    origclasscounts = Array(Float64,noclasses)
    for c = 1:noclasses
        origclasscounts[c] = sum(sampletrainingweights[c])
    end
    for v = 1:length(sampleselection)
        bestsplit = evaluate_variable_classification(bestsplit,sampleselection[v],variables[sampleselection[v]],types[sampleselection[v]],sampletrainingrefs,sampletrainingweights,origclasscounts,noclasses,trainingdata,method)
    end
    splitvalue, varno, variable, splittype, splitpoint = bestsplit
    if variable == :NA
        return :NA
    else
        return (varno,variable,splittype,splitpoint)
    end
end

function evaluate_variable_classification(bestsplit,varno,variable,splittype,trainingrefs,trainingweights,origclasscounts,noclasses,trainingdata,method)
    values = Array(Any,noclasses)
    for c = 1:noclasses
        values[c] = trainingdata[c][varno][trainingrefs[c]]
    end
    if splittype == :CATEGORIC
        if method.randval
            bestsplit = evaluate_classification_categoric_variable_randval(bestsplit,varno,variable,splittype,origclasscounts,noclasses,values,trainingweights,method)
        else
            bestsplit = evaluate_classification_categoric_variable_allvals(bestsplit,varno,variable,splittype,origclasscounts,noclasses,values,trainingweights,method)
        end
    else # splittype == :NUMERIC
        if method.randval
            bestsplit = evaluate_classification_numeric_variable_randval(bestsplit,varno,variable,splittype,origclasscounts,noclasses,values,trainingweights,method)
        else
            bestsplit = evaluate_classification_numeric_variable_allvals(bestsplit,varno,variable,splittype,origclasscounts,noclasses,values,trainingweights,method)
        end
    end
    return bestsplit
end

function evaluate_classification_categoric_variable_randval(bestsplit,varno,variable,splittype,origclasscounts,noclasses,values,trainingweights,method)
    key = values[wsample(1:noclasses,origclasscounts)][rand(1:end)]
    leftclasscounts = zeros(noclasses)
    rightclasscounts = Array(Float64,noclasses)
    for c = 1:noclasses
        for i = 1:length(values[c])
            if values[c][i] == key
                leftclasscounts[c] += trainingweights[c][i]
            end
        end
        rightclasscounts[c] = origclasscounts[c]-leftclasscounts[c]
    end
    if sum(leftclasscounts) >= method.minleaf && sum(rightclasscounts) >= method.minleaf
        splitvalue = -information_content(leftclasscounts,rightclasscounts)
        if splitvalue > bestsplit[1]
            bestsplit = (splitvalue,varno,variable,splittype,key)
        end
    end
    return bestsplit
end

function evaluate_classification_categoric_variable_allvals(bestsplit,varno,variable,splittype,origclasscounts,noclasses,allvalues,trainingweights,method)
    allkeys = allvalues[1]
    for c = 2:noclasses
        allkeys = vcat(allkeys,allvalues[c])
    end
    keys = unique(allkeys)
    rightclasscounts = Array(Float64,noclasses)
    for key in keys
        leftclasscounts = zeros(noclasses)
        for c = 1:noclasses
            for i = 1:length(allvalues[c])
                if allvalues[c][i] == key
                    leftclasscounts[c] += trainingweights[c][i]
                end
            end
            rightclasscounts[c] = origclasscounts[c]-leftclasscounts[c]
        end
        if sum(leftclasscounts) >= method.minleaf && sum(rightclasscounts) >= method.minleaf
            splitvalue = -information_content(leftclasscounts,rightclasscounts)
            if splitvalue > bestsplit[1]
                bestsplit = (splitvalue,varno,variable,splittype,key)
            end
        end
    end
    return bestsplit
end

function evaluate_classification_numeric_variable_randval(bestsplit,varno,variable,splittype,origclasscounts,noclasses,allvalues,trainingweights,method)
    minval = Inf
    maxval = -Inf
    for c = 1:noclasses
        if length(allvalues[c]) > 0
            minvalc = minimum(allvalues[c])
            if minvalc < minval
                minval = minvalc
            end
            maxvalc = maximum(allvalues[c])
            if maxvalc > maxval
                maxval = maxvalc
            end
        end
    end
    if maxval > minval
        splitpoint = minval+rand()*(maxval-minval)
        leftclasscounts = zeros(noclasses)
        rightclasscounts = Array(Float64,noclasses)
        for c = 1:noclasses
            for i = 1:length(allvalues[c])
                if allvalues[c][i] <= splitpoint
                    leftclasscounts[c] += trainingweights[c][i]
                end
            end
            rightclasscounts[c] = origclasscounts[c]-leftclasscounts[c]
        end
        if sum(leftclasscounts) >= method.minleaf && sum(rightclasscounts) >= method.minleaf
            splitvalue = -information_content(leftclasscounts,rightclasscounts)
            if splitvalue > bestsplit[1]
                bestsplit = (splitvalue,varno,variable,splittype,splitpoint)
            end
        end
    end
    return bestsplit
end

function evaluate_classification_numeric_variable_allvals(bestsplit,varno,variable,splittype,origclasscounts,noclasses,allvalues,trainingweights,method)
    numericvalues = Dict{Any, Any}()
    for c = 1:noclasses
        for v = 1:length(allvalues[c])
            value = allvalues[c][v]
            valuecounts = zeros(noclasses)
            valuecounts[c] = trainingweights[c][v]
            numericvalues[value] = get(numericvalues,value,zeros(noclasses)) + valuecounts
        end
    end
    splitpoints = Array(Any,length(numericvalues),2)
    splitpoints[:,1] = collect(keys(numericvalues))
    splitpoints[:,2] = collect(values(numericvalues))
    splitpoints = sortrows(splitpoints,by=x->x[1])
    leftclasscounts = zeros(noclasses)
    weightsum = sum(origclasscounts)
    for s = 1:size(splitpoints,1)-1
        leftclasscounts += splitpoints[s,2]
        rightclasscounts = origclasscounts-leftclasscounts
        splitvalue = -information_content(leftclasscounts,rightclasscounts)
        if sum(leftclasscounts) >= method.minleaf && sum(rightclasscounts) >= method.minleaf
            if splitvalue > bestsplit[1]
                bestsplit = (splitvalue,varno,variable,splittype,splitpoints[s,1])
            end
        end
    end
    return bestsplit
end

function information_content(left,right)
    noleft = sum(left)
    noright = sum(right)
    total = noleft+noright
    if total > 0
        return noleft/total*entropy(left,noleft)+noright/total*entropy(right,noright)
    else
        return Inf
    end
end

function entropy(counts,total)
    if total > 0
        entropyval = 0.0
        for count in counts
            if count > 0
                probability = count/total
                entropyval += -probability*log(probability)
            end
        end
    else
        entropyval = 0.0
    end
    return entropyval
end

function make_split(method::LearningMethod{Classifier},node,trainingdata,bestsplit)
    (varno, variable, splittype, splitpoint) = bestsplit
    noclasses = size(node.trainingrefs,1)
    leftrefs = Array(Any,noclasses)
    leftweights = Array(Any,noclasses)
    rightrefs = Array(Any,noclasses)
    rightweights = Array(Any,noclasses)
    values = Array(Any,noclasses)
    for c = 1:noclasses
        leftrefs[c] = Int[]
        leftweights[c] = Float64[]
        rightrefs[c] = Int[]
        rightweights[c] = Float64[]
        values[c] = trainingdata[c][varno][node.trainingrefs[c]]
        if splittype == :NUMERIC
            for r = 1:length(node.trainingrefs[c])
                ref = node.trainingrefs[c][r]
                if values[c][r] <= splitpoint
                    push!(leftrefs[c],ref)
                    push!(leftweights[c],node.trainingweights[c][r])
                else
                    push!(rightrefs[c],ref)
                    push!(rightweights[c],node.trainingweights[c][r])
                end
            end
        else
            for r = 1:length(node.trainingrefs[c])
                ref = node.trainingrefs[c][r]
                if values[c][r] == splitpoint
                    push!(leftrefs[c],ref)
                    push!(leftweights[c],node.trainingweights[c][r])
                else
                    push!(rightrefs[c],ref)
                    push!(rightweights[c],node.trainingweights[c][r])
                end
            end
        end
    end
    noleftexamples = sum([sum(leftweights[i]) for i=1:noclasses])
    norightexamples = sum([sum(rightweights[i]) for i=1:noclasses])
    leftweight = noleftexamples/(noleftexamples+norightexamples)
    return leftrefs,leftweights,[],[],[],rightrefs,rightweights,[],[],[],leftweight
end

function generate_model_internal(method::LearningMethod{Classifier}, oobs, classes)
    if method.conformal == :default
        conformal = :std
    else
        conformal = method.conformal
    end
    oobpredictions = oobs[1]
    noclasses = length(classes)
    for c = 1:noclasses
        for r = 2:length(oobs)
            oobpredictions[c] += oobs[r][c]
        end
    end
    noobcorrect = 0
    nooob = 0
    if conformal == :std
        alphas = Float64[]
        for c = 1:noclasses
            for i = 1:size(oobpredictions[c],1)
                oobpredcount = oobpredictions[c][i][1]
                if oobpredcount > 0
                    alpha = oobpredictions[c][i][c+1]/oobpredcount-maximum(oobpredictions[c][i][[2:c;c+2:noclasses+1]])/oobpredcount
                    push!(alphas,alpha)
                    noobcorrect += 1-abs(sign(indmax(oobpredictions[c][i][2:end])-c))
                    nooob += 1
                end
            end
        end
        thresholdindex = Int(floor((nooob+1)*(1-method.confidence)))
        sortedalphas = Float64[]
        if thresholdindex >= 1
            sortedalphas = sort(alphas)
            alpha = sortedalphas[thresholdindex]
        else
            alpha = -Inf
        end
        conformalfunction = (:std,alpha,sortedalphas)
    elseif conformal == :classcond
        classalpha = Array(Float64,noclasses)
        classalphas = Array(Any,noclasses)
        for c = 1:noclasses
            classalphas[c] = Float64[]
            noclassoob = 0
            for i = 1:size(oobpredictions[c],1)
                oobpredcount = oobpredictions[c][i][1]
                if oobpredcount > 0
                    alphavalue = oobpredictions[c][i][c+1]/oobpredcount-maximum(oobpredictions[c][i][[2:c;c+2:noclasses+1]])/oobpredcount
                    push!(classalphas[c],alphavalue)
                    noobcorrect += 1-abs(sign(indmax(oobpredictions[c][i][2:end])-c))
                    noclassoob += 1
                end
            end
            thresholdindex = Int(floor((noclassoob+1)*(1-method.confidence)))
            if thresholdindex >= 1
                classalphas[c] = sort(classalphas[c])
                classalpha[c] = classalphas[c][thresholdindex]
            else
                classalpha[c] = -Inf
            end
            nooob += noclassoob
        end
        conformalfunction = (:classcond,classalpha,classalphas)
    end
    oobperformance = noobcorrect/nooob
    return oobperformance, conformalfunction
end

function apply_model(model::PredictionModel{Classifier}; confidence = :std)
    # AMG: still requires global data
    numThreads = Threads.nthreads()
    nocoworkers = nprocs()-1
    predictions = zeros(size(globaldata,1))
    if nocoworkers > 0
        alltrees = getworkertrees(model, nocoworkers)
        results = pmap(apply_trees,[(model.method,model.classes,subtrees) for subtrees in alltrees])
        for r = 1:length(results)
            predictions += results[r]
        end
    elseif numThreads > 1
        alltrees = getworkertrees(model, numThreads)
        Threads.@threads for subtrees in alltrees
            predictions += apply_trees((model.method,model.classes,subtrees))
        end
    else
        predictions += apply_trees((model.method,model.classes,model.trees))
    end
    predictions = predictions/model.method.notrees
    results = Array(Any,size(predictions,1))
    noclasses = length(model.classes)
    if model.conformal[1] == :std
        if confidence == :std
            alpha = model.conformal[2]
        else
            nooob = size(model.conformal[3],1)
            thresholdindex = Int(floor((nooob+1)*(1-confidence)))
            if thresholdindex >= 1
                alpha = model.conformal[3][thresholdindex]
            else
                alpha = -Inf
            end
        end
        for i = 1:size(predictions,1)
            class = model.classes[indmax(predictions[i])]
            plausible = Any[]
            for j=1:noclasses
                if predictions[i][j]-maximum(predictions[i][[1:j-1;j+1:end]]) >= alpha
                    push!(plausible,model.classes[j])
                end
            end
            results[i] = (class,plausible,predictions[i])
        end
    elseif model.conformal[1] == :classcond
        if confidence == :std
            classalpha = model.conformal[2]
        else
            classalpha = Array(Float64,noclasses)
            for c = 1:noclasses
                noclassoob = size(model.conformal[3][c],1)
                thresholdindex = Int(floor((noclassoob+1)*(1-confidence)))
                if thresholdindex >= 1
                    classalpha[c] = model.conformal[3][c][thresholdindex]
                else
                    classalpha[c] = -Inf
                end
            end
        end
        for i = 1:size(predictions,1)
            class = model.classes[indmax(predictions[i])]
            plausible = Any[]
            for j=1:noclasses
                if predictions[i][j]-maximum(predictions[i][[1:j-1;j+1:end]]) >= classalpha[j]
                    push!(plausible,model.classes[j])
                end
            end
            results[i] = (class,plausible,predictions[i])
        end
    end
    return results
end

function apply_trees(Arguments::Tuple{LearningMethod{Classifier},Any,Any})
    method, classes, trees = Arguments
    variables, types = get_variables_and_types(globaldata)
    globalarray = [globaldata]
    testmissingvalues, testnonmissingvalues = find_missing_values(method,variables,globalarray)    
    newtestdata = transform_nonmissing_columns_to_arrays(method,variables,globalarray,testmissingvalues)
    replacements_for_missing_values!(method,newtestdata,globalarray,variables,types,testmissingvalues,testnonmissingvalues)
    nopredictions = size(globaldata,1)
    noclasses = length(classes)
    predictions = Array(Any,nopredictions)
    for i = 1:nopredictions
        predictions[i] = zeros(noclasses)
        for t = 1:length(trees)
            treeprediction = make_prediction(trees[t],newtestdata[1],i,zeros(noclasses))
            predictions[i] += treeprediction
        end
    end
    results = predictions
    return results
end
