
##
## Functions to be executed on each worker
##

function generate_and_test_trees(Arguments::Tuple{LearningMethod{Classifier},Symbol,Int64,Int64,Array{Any,1}})
    method,experimentype,notrees,randseed,randomoobs = Arguments
    classes = getDfArrayData(unique(globaldata[:CLASS]))
    s = size(globaldata,1)
    srand(randseed)
    noclasses = length(classes)
    variables, types = get_variables_and_types(globaldata)
    if experimentype == :test
        model,oobpredictions,variableimportance, modelsize, noirregularleafs, randomclassoobs, oob = generate_trees((method,classes,notrees,randseed);curdata=globaldata[globaldata[:TEST] .== false,:], randomoobs=randomoobs, varimparg = false)
        globaltestdata = globaldata[globaldata[:TEST] .== true,:]
        testdata = Array(DataFrame,noclasses)
        for c = 1:noclasses
            testdata[c] = globaltestdata[globaltestdata[:CLASS] .== classes[c],:]
        end
        testmissingvalues, testnonmissingvalues = find_missing_values(method,variables,testdata)
        newtestdata = transform_nonmissing_columns_to_arrays(method,variables,testdata,testmissingvalues)
        replacements_for_missing_values!(method,newtestdata,testdata,variables,types,testmissingvalues,testnonmissingvalues)
        nopredictions = sum([size(testdata[c],1) for c = 1:noclasses])
        predictions = Array(Array{Float64,1},nopredictions)
        totalnotrees,correctclassificationcounter,squaredproberror = make_prediction_analysis(method, model, newtestdata, randomclassoobs, oob, predictions)
        return (modelsize,predictions,([totalnotrees;correctclassificationcounter],[totalnotrees;squaredproberror]),oobpredictions,noirregularleafs)
    else # experimentype == :cv
        folds = sort(unique(globaldata[:FOLD]))
        nofolds = length(folds)
        nocorrectclassifications = Array(Any,nofolds)
        squaredproberrors = Array(Any,nofolds)
        predictions = Array(Array{Float64,1},size(globaldata,1))
        oobpredictions = Array(Array,nofolds)
        modelsizes = Array(Int,nofolds)
        noirregularleafs = Array(Int,nofolds)
        testexamplecounter = 0
        foldno = 0
        for fold in folds
            foldno += 1
            trainingdata = globaldata[globaldata[:FOLD] .!= fold,:]
            curFoldTestData = globaldata[globaldata[:FOLD] .== fold,:]
            testdata = Array(Any,noclasses)
            for c = 1:noclasses
                testdata[c] = curFoldTestData[curFoldTestData[:CLASS] .== classes[c],:]
            end
            model,oobpredictions[foldno],variableimportance, modelsizes[foldno], noirregularleafs[foldno], randomclassoobs, oob = generate_trees((method,classes,notrees,randseed);curdata=trainingdata, randomoobs=size(randomoobs,1) > 0 ? randomoobs[foldno] : [], varimparg = false)

            testmissingvalues, testnonmissingvalues = find_missing_values(method,variables,testdata)
            newtestdata = transform_nonmissing_columns_to_arrays(method,variables,testdata,testmissingvalues)
            replacements_for_missing_values!(method,newtestdata,testdata,variables,types,testmissingvalues,testnonmissingvalues)

            totalnotrees,correctclassificationcounter,squaredproberror = make_prediction_analysis(method, model, newtestdata, randomclassoobs, oob, predictions; predictionexamplecounter=testexamplecounter)
            testexamplecounter += sum([size(testdata[c],1) for c = 1:noclasses])

            nocorrectclassifications[foldno] = [totalnotrees;correctclassificationcounter]
            squaredproberrors[foldno] = [totalnotrees;squaredproberror]
        end
        return (modelsizes,predictions,(nocorrectclassifications,squaredproberrors),oobpredictions,noirregularleafs)
    end
end

function make_prediction_analysis(method::LearningMethod{Classifier}, model, newtestdata, randomclassoobs, oob, predictions; predictionexamplecounter = 0)
    correctclassificationcounter = 0
    squaredproberror = 0.0
    totalnotrees = 0
    testexamplecounter = 0
    noclasses = length(newtestdata)
    for c = 1:noclasses
        correctclassvector = zeros(noclasses)
        correctclassvector[c] = 1.0
        for i = 1:size(newtestdata[c][1],1)
            testexamplecounter += 1
            classprobabilities = zeros(noclasses)
            nosampledtrees = 0
            if method.modpred
                randomoobclass, randomoobref = randomclassoobs[testexamplecounter]
            end
            for t = 1:length(model)
                if method.modpred
                    if oob[t][randomoobclass][randomoobref]
                        prediction = make_prediction(model[t],newtestdata[c],i,zeros(noclasses))
                        classprobabilities += prediction
                        correctclassificationcounter += 1-abs(sign(indmax(prediction)-c))
                        squaredproberror += sqL2dist(correctclassvector,prediction)
                        nosampledtrees += 1
                    end
                else
                    prediction = make_prediction(model[t],newtestdata[c],i,zeros(noclasses))
                    classprobabilities += prediction
                    correctclassificationcounter += 1-abs(sign(indmax(prediction)-c))
                    squaredproberror += sqL2dist(correctclassvector,prediction)
                end
            end
            if ~method.modpred
                nosampledtrees = length(model)
            end
            predictionexamplecounter += 1
            totalnotrees += nosampledtrees
            predictions[predictionexamplecounter] = [nosampledtrees;classprobabilities]
        end
    end
    return (totalnotrees,correctclassificationcounter,squaredproberror)
end

function generate_trees(Arguments::Tuple{LearningMethod{Classifier},AbstractArray,Int,Int};curdata=globaldata, randomoobs=[], varimparg = true)
    method,classes,notrees,randseed = Arguments
    srand(randseed)
    noclasses = length(classes)
    trainingdata = Array(typeof(curdata), noclasses)
    trainingrefs = Array(Array{Int,1},noclasses)
    trainingweights = Array(Array{Float64,1},noclasses)
    oobpredictions = Array(Array{Array{Float64,1},1},noclasses)
    emptyprediction = [0; zeros(noclasses)]
    for c = 1:noclasses
        trainingdata[c] = curdata[curdata[:CLASS] .== classes[c],:]
        trainingrefs[c] = collect(1:size(trainingdata[c],1))
        trainingweights[c] = getDfArrayData(trainingdata[c][:WEIGHT])
        oobpredictions[c] = Array(Array{Float64,1},size(trainingdata[c],1))
        for i = 1:size(trainingdata[c],1)
            oobpredictions[c][i] = emptyprediction
        end
    end
    regressionvalues = []
    timevalues = []
    eventvalues = []    
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
    variables, types = get_variables_and_types(curdata)
    modelsize = 0
    noirregularleafs = 0
    if useSparseData
        newtrainingdata = trainingdata
    else
        missingvalues, nonmissingvalues = find_missing_values(method,variables,trainingdata)
        newtrainingdata = transform_nonmissing_columns_to_arrays(method,variables,trainingdata,missingvalues)
    end
    model = Array(TreeNode,notrees)
    oob = Array(Array,notrees)
    variableimportance = zeros(size(variables,1))
    for treeno = 1:notrees
        if ~useSparseData
            sample_replacements_for_missing_values!(method, newtrainingdata,trainingdata,variables,types,missingvalues,nonmissingvalues)
        end
        model[treeno], treevariableimportance, noleafs, treenoirregularleafs, oob[treeno] = generate_tree(method,trainingrefs,trainingweights,regressionvalues,timevalues,eventvalues,newtrainingdata,variables,types,oobpredictions;varimp = varimparg)
        modelsize += noleafs
        noirregularleafs += treenoirregularleafs
        if (varimparg)
            variableimportance += treevariableimportance
        end
    end
    return (model,oobpredictions,variableimportance, modelsize, noirregularleafs, randomclassoobs, oob)
end

function find_missing_values(method::LearningMethod{Classifier},variables,trainingdata)
    noclasses = length(trainingdata)
    missingvalues = Array(Array{Array{Int,1},1},noclasses)
    nonmissingvalues = Array(Array{Array,1},noclasses)
    for c = 1:noclasses
        missingvalues[c] = Array(Array{Int,1},length(variables))
        nonmissingvalues[c] = Array(Array,length(variables))
        for v = 1:length(variables)
            variable = variables[v]
            missingvalues[c][v] = Int[]
            nonmissingvalues[c][v] = typeof(trainingdata[c][variable]).parameters[1][]
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
            end
        end
    end
    return (missingvalues,nonmissingvalues)
end

function transform_nonmissing_columns_to_arrays(method::LearningMethod{Classifier},variables,trainingdata,missingvalues)
    noclasses = length(trainingdata)
    newdata = Array(Array{Array,1},noclasses)
    for c = 1:noclasses
        newdata[c] = Array(Array,length(variables))
        for v = 1:length(variables)
            if isempty(missingvalues[c][v])
                newdata[c][v] = getDfArrayData(trainingdata[c][variables[v]])
            end
        end
    end
    return newdata
end

function sample_replacements_for_missing_values!(method::LearningMethod{Classifier},newtrainingdata,trainingdata,variables,types,missingvalues,nonmissingvalues)
    noclasses = length(newtrainingdata)
    for c = 1:noclasses
        for v = 1:length(variables)
            if !isempty(missingvalues[c][v])
                values = trainingdata[c][variables[v]]
                valuefrequencies = [length(nonmissingvalues[cl][v]) for cl = 1:noclasses]
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
                newtrainingdata[c][v] = getDfArrayData(values)
            end
        end
    end
end

function replacements_for_missing_values!(method::LearningMethod{Classifier},newtestdata,testdata,variables,types,missingvalues,nonmissingvalues)
    noclasses = size(newtestdata,1)
    for c = 1:noclasses
        for v = 1:length(variables)
            if !isempty(missingvalues[c][v])
                variableType = typeof(testdata[c][variables[v]]).parameters[1]
                values = convert(Array{Nullable{variableType},1},testdata[c][variables[v]],Nullable{variableType}())
                newtestdata[c][v] = values
            end
        end
    end
end

function generate_tree(method::LearningMethod{Classifier},trainingrefs,trainingweights,regressionvalues,timevalues,eventvalues,trainingdata,variables,types,oobpredictions; varimp = false)
    noclasses = length(trainingweights)
    zeroweights = Array(Array{Bool,1},noclasses)
    if method.bagging
        classweights = [length(t) for t in trainingrefs]
        if typeof(method.bagsize) == Int
            samplesize = method.bagsize
        else
            samplesize = round(Int,sum(classweights)*method.bagsize)
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
        model, variableimportance, noleafs, noirregularleafs = build_tree(method,newtrainingrefs,newtrainingweights,regressionvalues,timevalues,eventvalues,trainingdata,variables,types,varimp)
        for c = 1:noclasses
            oobrefs = trainingrefs[c][zeroweights[c]]
            for oobref in oobrefs
                oobpredictions[c][oobref] += [1;make_prediction(model,trainingdata[c],oobref,zeros(noclasses))]
            end
        end
    else
        model, variableimportance, noleafs, noirregularleafs = build_tree(method,trainingrefs,trainingweights,regressionvalues,timevalues,eventvalues,trainingdata,variables,types,varimp)
    end
    return model, variableimportance, noleafs, noirregularleafs, zeroweights
end

function default_prediction(trainingweights)
    noclasses = size(trainingweights,1)
    classcounts = map(sum,trainingweights)
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
        classweights = map(sum,node.trainingweights)
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

function make_leaf(node,method::LearningMethod{Classifier}, parenttrainingweights)
    noclasses = size(node.trainingweights,1)
    classcounts = [sum(node.trainingweights[i]) for i = 1:noclasses]
    noinstances = sum(classcounts)
    if noinstances > 0
        if method.laplace
            return [(classcounts[i]+1)/(noinstances+noclasses) for i=1:noclasses]
        else
            return [classcounts[i]/noinstances for i=1:noclasses]
        end
    end
    return prediction = default_prediction(parenttrainingweights)
end

function find_best_split(node,trainingdata,variables,types,method::LearningMethod{Classifier})
    if method.randsub == :all
        sampleselection = collect(1:length(variables))
    elseif method.randsub == :default || method.randsub == :log2
        sampleselection = sample(1:length(variables),floor(Int,log(2,length(variables))+1),replace=false)
    elseif method.randsub == :sqrt
        sampleselection = sample(1:length(variables),floor(Int,sqrt(length(variables))),replace=false)
    else
        if typeof(method.randsub) == Int
            if method.randsub > length(variables)
                sampleselection = collect(1:length(variables))
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
        sampletrainingweights = Array(Array{Float64,1},noclasses)
        sampletrainingrefs = Array(Array{Int,1},noclasses)
        for c = 1:noclasses
            if sum(node.trainingweights[c]) <= splitsamplesize
                sampletrainingweights[c] = node.trainingweights[c]
                sampletrainingrefs[c] = node.trainingrefs[c]
            else
                sampletrainingweights[c] = Array(Float64,splitsamplesize)
                sampletrainingrefs[c] = Array(Int,splitsamplesize)
                for i = 1:splitsamplesize
                    sampletrainingweights[c][i] = 1.0
                    sampletrainingrefs[c][i] = node.trainingrefs[c][rand(1:end)]
                end
            end
        end
    else
        sampletrainingrefs = node.trainingrefs
        sampletrainingweights = node.trainingweights
    end
    bestsplit = (-Inf,0,:NA,:NA,0.0)
    noclasses = size(node.trainingrefs,1)
    origclasscounts = map(sum,sampletrainingweights)
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

function non_zero_variables(novariables,trainingdata,node)
    includevariables = falses(novariables)
    for c = 1:size(trainingdata,1)
        for v = 1:novariables
            if countnz(view(trainingdata[c][v],node.trainingrefs[c])) > 0
#            if nnz(trainingdata[c][v][node.trainingrefs[c]]) > 0
                includevariables[v] = true
            end
        end
    end
    return Array(1:novariables)[includevariables]
end

function evaluate_variable_classification(bestsplit,varno,variable,splittype,trainingrefs,trainingweights,origclasscounts,noclasses,trainingdata,method)
    values = Array(SubArray,noclasses)
    for c = 1:noclasses
        values[c] = view(trainingdata[c][varno],trainingrefs[c])
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
    return evaluate_classification_common(key, ==, bestsplit, varno,variable,splittype, values, noclasses, trainingweights, origclasscounts, method)
end

function evaluate_classification_categoric_variable_allvals(bestsplit,varno,variable,splittype,origclasscounts,noclasses,allvalues,trainingweights,method)
    allkeys = allvalues[1]
    for c = 2:noclasses
        allkeys = vcat(allkeys,allvalues[c])
    end
    keys = unique(allkeys)
    for key in keys
        bestsplit = evaluate_classification_common(key, ==, bestsplit, varno,variable,splittype, allvalues, noclasses, trainingweights, origclasscounts, method)
    end
    return bestsplit
end

function evaluate_classification_numeric_variable_randval(bestsplit,varno,variable,splittype,origclasscounts,noclasses,allvalues,trainingweights,method)
    if ~useSparseData
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
            bestsplit = evaluate_classification_common(splitpoint, <=, bestsplit, varno,variable,splittype, allvalues, noclasses, trainingweights, origclasscounts, method)
        end
    else
        splitpoint = 0.5
        bestsplit = evaluate_classification_common(splitpoint, <=, bestsplit, varno,variable,splittype, allvalues, noclasses, trainingweights, origclasscounts, method)
    end
    return bestsplit
end

function evaluate_classification_numeric_variable_allvals(bestsplit,varno,variable,splittype,origclasscounts,noclasses,allvalues,trainingweights,method)
    numericvalues = Dict{typeof(allvalues[1]).parameters[1], Array{Float64,1}}()
    for c = 1:noclasses
        for v = 1:length(allvalues[c])
            value = allvalues[c][v]
            valuecounts = zeros(noclasses)
            valuecounts[c] = trainingweights[c][v]
            numericvalues[value] = get(numericvalues,value,zeros(noclasses)) + valuecounts
        end
    end
    sortedkeys = sort(collect(keys(numericvalues)))
    leftclasscounts = zeros(noclasses)
    weightsum = sum(origclasscounts)
    for s = 1:size(sortedkeys,1)-1
        leftclasscounts += numericvalues[sortedkeys[s]]
        rightclasscounts = origclasscounts-leftclasscounts
        if sum(leftclasscounts) >= method.minleaf && sum(rightclasscounts) >= method.minleaf
            splitvalue = -information_content(leftclasscounts,rightclasscounts)
            if splitvalue > bestsplit[1]
                bestsplit = (splitvalue,varno,variable,splittype,sortedkeys[s])
            end
        end
    end
    return bestsplit
end

function calculateleftclasscount(values, trainingweights, key, op)
    lcc = 0.0
    for i = 1:length(values)
        if op(values[i], key)
            lcc += trainingweights[i]
        end
    end
    return lcc
end

function evaluate_classification_common(key, op, bestsplit, varno,variable,splittype, values, noclasses, trainingweights, origclasscounts, method)
    leftclasscounts = Array(Float64,noclasses)
    rightclasscounts = Array(Float64,noclasses)
    for c = 1:noclasses
        leftclasscounts[c] = calculateleftclasscount(values[c], trainingweights[c], key, op)
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
    
    leftrefs = Array(Array,noclasses)
    leftweights = Array(Array,noclasses)
    rightrefs = Array(Array,noclasses)
    rightweights = Array(Array,noclasses)
    op = splittype == :NUMERIC ? (<=) : (==)
    for c = 1:noclasses
        values = view(trainingdata[c][varno],node.trainingrefs[c])
        leftrefs[c] = Int[]
        leftweights[c] = Float64[]
        rightrefs[c] = Int[]
        rightweights[c] = Float64[]
        for r = 1:length(node.trainingrefs[c])
            ref = node.trainingrefs[c][r]
            if op(values[r], splitpoint)
                push!(leftrefs[c],ref)
                push!(leftweights[c],node.trainingweights[c][r])
            else
                push!(rightrefs[c],ref)
                push!(rightweights[c],node.trainingweights[c][r])
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
                    alpha = 1-(oobpredictions[c][i][c+1]/oobpredcount-maximum(oobpredictions[c][i][[2:c;c+2:noclasses+1]])/oobpredcount)
                    push!(alphas,alpha)
                    noobcorrect += 1-abs(sign(indmax(oobpredictions[c][i][2:end])-c))
                    nooob += 1
                end
            end
        end
        conformalfunction = (:std,sort(alphas,rev=true))
    elseif conformal == :classcond
        classalpha = Array(Float64,noclasses)
        classalphas = Array(Array,noclasses)
        for c = 1:noclasses
            classalphas[c] = Float64[]
            noclassoob = 0
            for i = 1:size(oobpredictions[c],1)
                oobpredcount = oobpredictions[c][i][1]
                if oobpredcount > 0
                    alphavalue = 1-(oobpredictions[c][i][c+1]/oobpredcount-maximum(oobpredictions[c][i][[2:c;c+2:noclasses+1]])/oobpredcount)
                    push!(classalphas[c],alphavalue)
                    noobcorrect += 1-abs(sign(indmax(oobpredictions[c][i][2:end])-c))
                    noclassoob += 1
                end
            end
            classalphas[c] = sort(classalphas[c],rev=true)
            nooob += noclassoob
        end
        conformalfunction = (:classcond,classalphas)
    end
    oobperformance = noobcorrect/nooob
    return oobperformance, conformalfunction
end

function apply_model_internal(model::PredictionModel{Classifier}; confidence = 0.95)
    numThreads = Threads.nthreads()
    nocoworkers = nprocs()-1
    if nocoworkers > 0
        alltrees = getworkertrees(model, nocoworkers)
        results = pmap(apply_trees,[(model.method,model.classes,subtrees) for subtrees in alltrees])
        predictions = sum(results)
    elseif numThreads > 1
        alltrees = getworkertrees(model, numThreads)
        results = Array{Array,1}(length(alltrees))
        Threads.@threads for subtrees in alltrees
            results[Threads.threadid()] = apply_trees((model.method,model.classes,subtrees))
        end
        waitfor(results)
        predictions = sum(results)
    else
        predictions = apply_trees((model.method,model.classes,model.trees))
    end
    predictions = predictions/model.method.notrees
    classes = model.classes
    conformal = model.conformal[1]
    alphas = model.conformal[2]
    results = Array(Any,size(predictions,1))
    for i = 1:size(predictions,1)
        class = classes[indmax(predictions[i])]
        plausible = typeof(class)[]        
        pvalues = Array(Float64,length(classes))
        for j=1:length(classes)
            if conformal == :std
                pvalues[j] = get_p_value(1-(predictions[i][j]-maximum(predictions[i][[1:j-1;j+1:end]])),alphas)
            else # conformal == :classcond
                pvalues[j] = get_p_value(1-(predictions[i][j]-maximum(predictions[i][[1:j-1;j+1:end]])),alphas[j])
            end
            if  pvalues[j] > 1-confidence
                push!(plausible,classes[j])
            end
        end
        results[i] = (class,predictions[i],plausible,pvalues)
    end
    return results
end

function get_p_value(alpha,alphas)
    noalphas = length(alphas)
    i = 1
    foundsmaller = false
    gt = 0
    e = 1
    while i <= noalphas && ~foundsmaller
        if alphas[i] > alpha
            gt += 1
        elseif alphas[i] == alpha
            e += 1
        else
            foundsmaller = true
        end
        i += 1
    end
    p = (gt + rand()*e)/(noalphas+1)
    return p
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
    predictions = Array(Array,nopredictions)
    for i = 1:nopredictions
        predictions[i] = zeros(noclasses)
        for t = 1:length(trees)
            treeprediction = make_prediction(trees[t],newtestdata[1],i,zeros(noclasses))
            predictions[i] += treeprediction
        end
    end
    return predictions
end

function collect_results_split(method::LearningMethod{Classifier}, randomoobs, results, time)
    modelsize = sum([result[1] for result in results])
    noirregularleafs = sum([result[5] for result in results])
    predictions = results[1][2]
    for r = 2:length(results)
        predictions += results[r][2]
    end
    nopredictions = size(predictions,1)
    predictions = [predictions[i][2:end]/predictions[i][1] for i = 1:nopredictions]
    classes = getDfArrayData(unique(globaldata[:CLASS]))
    noclasses = length(classes)
    classdata = Array(Any,noclasses)
    for c = 1:noclasses
        classdata[c] = globaldata[globaldata[:CLASS] .== classes[c],:]
    end
    confidence = method.confidence
    if method.conformal == :default
        conformal = :std
    else
        conformal = method.conformal
    end
    oobpredictions = results[1][4]
    for c = 1:noclasses
        for r = 2:length(results)
            oobpredictions[c] += results[r][4][c]
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
                    alpha = 1-(oobpredictions[c][i][c+1]/oobpredcount-maximum(oobpredictions[c][i][[2:c;c+2:noclasses+1]])/oobpredcount)
                    push!(alphas,alpha)
                    noobcorrect += 1-abs(sign(indmax(oobpredictions[c][i][2:end])-c))
                    nooob += 1
                end
            end
        end
        alphas = sort(alphas,rev=true)
    elseif conformal == :classcond
        randomclassoobs = Array(Any,size(randomoobs,1))
        for i = 1:size(randomclassoobs,1)
            oobref = randomoobs[i]
            c = 1
            while oobref > size(oobpredictions[c],1)
                oobref -= size(oobpredictions[c],1)
                c += 1
            end
            randomclassoobs[i] = (c,oobref)
        end
        alphas = Array(Float64,noclasses)
        for c = 1:noclasses
            alphas = Float64[]
            noclassoob = 0
            for i = 1:size(oobpredictions[c],1)
                oobpredcount = oobpredictions[c][i][1]
                if oobpredcount > 0
                    alpha = 1-(oobpredictions[c][i][c+1]/oobpredcount-maximum(oobpredictions[c][i][[2:c;c+2:noclasses+1]])/oobpredcount)
                    push!(alphas[c],alpha)
                    noobcorrect += 1-abs(sign(indmax(oobpredictions[c][i][2:end])-c))
                    noclassoob += 1
                end
            end
            nooob += noclassoob
            alphas[c] = sort(alphas[c],rev=true)
        end
    end
    oobacc = noobcorrect/nooob
    testdata = Array(Any,noclasses)
    nocorrect = 0
    briersum = 0.0
    marginsum = 0.0
    probsum = 0.0
    noinrangesum = 0
    nolabelssum = 0
    noonec = 0
    testexamplecounter = 0
    for c = 1:noclasses
        correctclassvector = zeros(noclasses)
        correctclassvector[c] = 1.0
        allbutclassvector = [1:c-1;c+1:noclasses]
        testdata[c] = classdata[c][classdata[c][:TEST] .== true,:]
        if size(testdata[c],1) > 0
            for i=testexamplecounter+1:testexamplecounter+size(testdata[c],1)
                mostprobable = indmax(predictions[i])
                correct = 1-abs(sign(mostprobable-c))
                nocorrect += correct
                briersum += sqL2dist(correctclassvector,predictions[i])
                margin = predictions[i][c]-maximum(predictions[i][allbutclassvector])
                marginsum += margin
                probsum += maximum(predictions[i])
                if method.modpred # NOTE: FIXME
                    if conformal == :std
                        randomoob = randomoobs[i]
                        
                        thresholdindex = Int(floor(nooob*(1-confidence)))
                        if thresholdindex >= 1
                            alpha = sort(alphas[[1:randomoob-1;randomoob+1:end]])[thresholdindex] # NOTE: assumes oobpredcount > 0 always is true!
                        else
                            alpha = -Inf
                        end
                    else # conformal == :classcond
                        randomoobclass, randomoobref = randomclassoobs[i]
                        thresholdindex = Int(floor(size(classalphas[randomoobclass],1)*(1-confidence)))
                        origclassalpha = classalpha[randomoobclass]
                        if thresholdindex >= 1
                            classalpha[randomoobclass] = sort(classalphas[randomoobclass][[1:randomoobref-1;randomoobref+1:end]])[thresholdindex] # NOTE: assumes oobpredcount > 0 always is true!
                        else
                            classalpha[randomoobclass] = -Inf
                        end
                        alpha = classalpha[c]
                    end
                else
                    plausible = Int64[]
                    for j = 1:noclasses
                        if conformal == :std
                            p_value = get_p_value(1-(predictions[i][j]-maximum(predictions[i][[1:j-1;j+1:end]])),alphas)
                        elseif conformal == :classcond
                            p_value = get_p_value(1-(predictions[i][j]-maximum(predictions[i][[1:j-1;j+1:end]])),alphas[j])
                        end
                        if p_value > 1-confidence
                            push!(plausible,j)
                        end
                    end
                    nolabels = length(plausible)
                    nolabelssum += nolabels
                    if c in plausible
                        noinrangesum += 1
                        if nolabels == 1
                            noonec += 1
                        end
                    end
                end
            end
        end
        testexamplecounter += size(testdata[c],1)
    end
    notestexamples = sum([size(testdata[c],1) for c = 1:noclasses])
    accuracy = nocorrect/notestexamples
    esterr = oobacc-accuracy
    absesterr = abs(esterr)
    brierscore = briersum/notestexamples
    margin = marginsum/notestexamples
    prob = probsum/notestexamples
    validity = noinrangesum/notestexamples
    avc = nolabelssum/notestexamples
    onec = noonec/notestexamples
    auc = Array(Float64,noclasses)
    testexamplecounter = 0
    for c = 1:noclasses
        if size(testdata[c],1) > 0
            classprobs = [predictions[i][c] for i=1:length(predictions)]
            auc[c] = calculate_auc(sort(classprobs[testexamplecounter+1:testexamplecounter+size(testdata[c],1)],rev = true),
                                   sort([classprobs[1:testexamplecounter];classprobs[testexamplecounter+size(testdata[c],1)+1:end]], rev = true))
        else
            auc[c] = 0.0
        end
        testexamplecounter += size(testdata[c],1)
    end
    classweights = [size(testdata[c],1)/notestexamples for c = 1:noclasses]
    weightedauc = sum(auc .* classweights)
    totalnotrees = sum([results[r][3][1][1] for r = 1:length(results)])
    totalnocorrect = sum([results[r][3][1][2] for r = 1:length(results)])
    avacc = totalnocorrect/totalnotrees
    totalsquarederror = sum([results[r][3][2][2] for r = 1:length(results)])
    avbrier = totalsquarederror/totalnotrees
    varbrier = avbrier-brierscore
    extratime = toq()
    return ClassificationResult(accuracy,weightedauc,brierscore,avacc,esterr,absesterr,avbrier,varbrier,margin,prob,validity,avc,onec,modelsize,noirregularleafs,time+extratime)#, get_predictions_classification(classes, predictions, classalpha), classes

end

function calculate_auc(posclassprobabilities,negclassprobabilities)
    values = Dict{Any, Any}()
    for i = 1:length(posclassprobabilities)
        value = posclassprobabilities[i]
        values[value] = get(values,value,[0,0]) + [1,0]
    end
    for i = 1:length(negclassprobabilities)
        value = negclassprobabilities[i]
        values[value] = get(values,value,[0,0]) + [0,1]
    end
    pairs = sort(collect(values),rev = true,by=x->x[1])
    totnopos = length(posclassprobabilities)
    totnoneg = length(negclassprobabilities)
    poscounter = 0
    negcounter = 0
    auc = 0.0
    for i = 1:length(pairs)
        newpos, newneg = pairs[i][2]
            if newneg == 0
                poscounter += newpos
            elseif newpos == 0
                auc += (newneg/totnoneg)*(poscounter/totnopos)
                negcounter += newneg
            else
                auc += (newneg/totnoneg)*(poscounter/totnopos)+(newpos/totnopos)*(newneg/totnoneg)/2
                poscounter += newpos
                negcounter += newneg
            end
    end
    return auc
end

function collect_results_cross_validation(method::LearningMethod{Classifier}, randomoobs, results, modelsizes, nofolds, time)
    folds = collect(1:nofolds)
    allnoirregularleafs = [result[5] for result in results]
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
    predictions = [predictions[i][2:end]/predictions[i][1] for i = 1:nopredictions]
    confidence = method.confidence
    if method.conformal == :default
        conformal = :std
    else
        conformal = method.conformal
    end
    accuracy = Array(Float64,nofolds)
    auc = Array(Float64,nofolds)
    brierscore = Array(Float64,nofolds)
    avacc = Array(Float64,nofolds)
    avbrier = Array(Float64,nofolds)
    varbrier = Array(Float64,nofolds)
    margin = Array(Float64,nofolds)
    prob = Array(Float64,nofolds)
    oobacc = Array(Float64,nofolds)
    esterr = Array(Float64,nofolds)
    absesterr = Array(Float64,nofolds)
    validity = Array(Float64,nofolds)
    avc = Array(Float64,nofolds)
    onec = Array(Float64,nofolds)
    classes = getDfArrayData(unique(globaldata[:CLASS]))
    noclasses = length(classes)
    foldauc = Array(Float64,noclasses)
    classdata = Array(Any,noclasses)
    returning_prediction = Array(Any,size(predictions,1))
    for c = 1:noclasses
        classdata[c] = globaldata[globaldata[:CLASS] .== classes[c],:]
    end
    testdata = Array(Any,noclasses)
    foldno = 0
    for fold in folds
        foldno += 1
        foldIndeces = globaldata[:FOLD] .== fold
        oobpredictions = results[1][4][foldno]
        for c = 1:noclasses
            for r = 2:length(results)
                oobpredictions[c] += results[r][4][foldno][c]
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
                        alpha = 1-(oobpredictions[c][i][c+1]/oobpredcount-maximum(oobpredictions[c][i][[2:c;c+2:noclasses+1]])/oobpredcount)
                        push!(alphas,alpha)
                        noobcorrect += 1-abs(sign(indmax(oobpredictions[c][i][2:end])-c))
                        nooob += 1
                    end
                end
            end
            alphas = sort(alphas,rev=true)
        elseif conformal == :classcond
            alphas = Array(Float64,noclasses)
            for c = 1:noclasses
                for i = 1:size(oobpredictions[c],1)
                    oobpredcount = oobpredictions[c][i][1]
                    if oobpredcount > 0
                        alpha = 1-(oobpredictions[c][i][c+1]/oobpredcount-maximum(oobpredictions[c][i][[2:c;c+2:noclasses+1]])/oobpredcount)
                        push!(alphas[c],alpha)
                        noobcorrect += 1-abs(sign(indmax(oobpredictions[c][i][2:end])-c))
                        nooob += 1
                    end
                end
                alphas[c] = sort(alphas[c],rev=true)
            end
        end
        oobacc[foldno] = noobcorrect/nooob
        nocorrect = 0
        briersum = 0.0
        marginsum = 0.0
        probsum = 0.0
        noinrangesum = 0
        nolabelssum = 0
        noonec = 0
        origtestexamplecounter = testexamplecounter
        for c = 1:noclasses
            correctclassvector = zeros(noclasses)
            correctclassvector[c] = 1.0
            allbutclassvector = [1:c-1;c+1:noclasses]
            testdata[c] = classdata[c][classdata[c][:FOLD] .== fold,:]
            if size(testdata[c],1) > 0
                for i=testexamplecounter+1:testexamplecounter+size(testdata[c],1)
                    mostprobable = indmax(predictions[i])
                    correct = 1-abs(sign(mostprobable-c))
                    nocorrect += correct
                    briersum += sqL2dist(correctclassvector,predictions[i])
                    examplemargin = predictions[i][c]-maximum(predictions[i][allbutclassvector])
                    marginsum += examplemargin
                    probsum += maximum(predictions[i])
                    plausible = Int64[]
                    for j = 1:noclasses
                        if conformal == :std
                            p_value = get_p_value(1-(predictions[i][j]-maximum(predictions[i][[1:j-1;j+1:end]])),alphas)
                        elseif conformal == :classcond
                            p_value = get_p_value(1-(predictions[i][j]-maximum(predictions[i][[1:j-1;j+1:end]])),alphas[j])
                        end
                        if p_value > 1-confidence
                            push!(plausible,j)
                        end
                    end
                    nolabels = length(plausible)
                    nolabelssum += nolabels
                    if c in plausible
                        noinrangesum += 1
                        if nolabels == 1
                            noonec += 1
                        end
                    end
                end
                testexamplecounter += size(testdata[c],1)
            end
        end
        foldpredictions = predictions[origtestexamplecounter+1:testexamplecounter]
        tempexamplecounter = 0
        for c = 1:noclasses
            if size(testdata[c],1) > 0
                classprobs = [foldpredictions[i][c] for i=1:length(foldpredictions)]
                foldauc[c] = calculate_auc(sort(classprobs[tempexamplecounter+1:tempexamplecounter+size(testdata[c],1)],rev = true),
                                           sort([classprobs[1:tempexamplecounter];classprobs[tempexamplecounter+size(testdata[c],1):end]], rev = true))
                tempexamplecounter += size(testdata[c],1)
            else
                foldauc[c] = 0.0
            end
        end
        notestexamples = sum([size(testdata[c],1) for c = 1:noclasses])
        accuracy[foldno] = nocorrect/notestexamples
        esterr[foldno] = oobacc[foldno]-accuracy[foldno]
        absesterr[foldno] = abs(oobacc[foldno]-accuracy[foldno])
        brierscore[foldno] = briersum/notestexamples
        margin[foldno] = marginsum/notestexamples
        prob[foldno] = probsum/notestexamples
        validity[foldno] = noinrangesum/notestexamples
        avc[foldno] = nolabelssum/notestexamples
        onec[foldno] = noonec/notestexamples
        classweights = [size(testdata[c],1)/notestexamples for c = 1:noclasses]
        auc[foldno] = sum(foldauc .* classweights)
        totalnotrees = sum([results[r][3][1][foldno][1] for r = 1:length(results)])
        totalnocorrect = sum([results[r][3][1][foldno][2] for r = 1:length(results)])
        avacc[foldno] = totalnocorrect/totalnotrees
        totalsquarederror = sum([results[r][3][2][foldno][2] for r = 1:length(results)])
        avbrier[foldno] = totalsquarederror/totalnotrees
        varbrier[foldno] = avbrier[foldno] - brierscore[foldno]
    end
    extratime = toq()
    return ClassificationResult(mean(accuracy),mean(auc),mean(brierscore),mean(avacc),mean(esterr),mean(absesterr),mean(avbrier),mean(varbrier),mean(margin),mean(prob),
                                                mean(validity),mean(avc),mean(onec),mean(modelsizes),mean(noirregularleafs),time+extratime)# , returning_prediction, classes
end
