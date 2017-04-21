
##
## Functions to be executed on each worker
##

function generate_and_test_trees(Arguments::Tuple{LearningMethod{Regressor},Symbol,Int64,Int64,Array{Any,1}})
    method,experimentype,notrees,randseed,randomoobs = Arguments
    s = size(globaldata,1)
    srand(randseed)
    variables, types = get_variables_and_types(globaldata)
    if experimentype == :test
        model,oobpredictions,variableimportance, modelsize, noirregularleafs, oob = generate_trees((method,Int64[],notrees,randseed);curdata=globaldata[globaldata[:TEST] .== false,:], randomoobs=randomoobs, varimparg = false)
        testdata = globaldata[globaldata[:TEST] .== true,:]
        testmissingvalues, testnonmissingvalues = find_missing_values(method,variables,testdata)
        newtestdata = transform_nonmissing_columns_to_arrays(method,variables,testdata,testmissingvalues)
        replacements_for_missing_values!(method,newtestdata,testdata,variables,types,testmissingvalues,testnonmissingvalues)
        correctvalues = getDfArrayData(testdata[:REGRESSION])
        nopredictions = size(testdata,1)
        predictions = Array(Array{Float64,1},nopredictions)
        squaredpredictions = Array(Any,nopredictions)
        totalnotrees,squarederror = make_prediction_analysis(method, model, newtestdata, randomoobs, oob, predictions, squaredpredictions, correctvalues)
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
            model,oobpredictions[foldno],variableimportance, modelsizes[foldno], noirregularleafs[foldno], oob = generate_trees((method,Int64[],notrees,randseed);curdata=trainingdata,randomoobs=randomoobs[foldno], varimparg = false)
            testmissingvalues, testnonmissingvalues = find_missing_values(method,variables,testdata)
            newtestdata = transform_nonmissing_columns_to_arrays(method,variables,testdata,testmissingvalues)
            replacements_for_missing_values!(method,newtestdata,testdata,variables,types,testmissingvalues,testnonmissingvalues)
            correctvalues = getDfArrayData(testdata[:REGRESSION])
            totalnotrees,squarederror = make_prediction_analysis(method, model, newtestdata, randomoobs[foldno], oob, predictions, squaredpredictions, correctvalues; predictionexamplecounter=testexamplecounter)
            testexamplecounter += size(testdata,1)
            squarederrors[foldno] = [totalnotrees;squarederror]
         end
         return (modelsizes,predictions,squarederrors,oobpredictions,squaredpredictions,noirregularleafs)
    end
end

function make_prediction_analysis(method::LearningMethod{Regressor}, model, newtestdata, randomoobs, oob, predictions, squaredpredictions,correctvalues; predictionexamplecounter = 0)
  squarederror = 0.0
  totalnotrees = 0
  for i = 1:size(newtestdata[1],1)
      correctvalue = correctvalues[i]
      prediction = 0.0
      squaredprediction = 0.0
      nosampledtrees = 0
      for t = 1:length(model)
          if method.modpred
              if oob[t][randomoobs[i]]
                  leafstats = make_prediction(model[t],newtestdata,i,0)
                  treeprediction = leafstats[2]/leafstats[1]
                  prediction += treeprediction
                  squaredprediction += treeprediction^2
                  squarederror += (treeprediction-correctvalue)^2
                  nosampledtrees += 1
              end
          else
              leafstats = make_prediction(model[t],newtestdata,i,0)
              treeprediction = leafstats[2]/leafstats[1]
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

function generate_trees(Arguments::Tuple{LearningMethod{Regressor},Array{Int,1},Int,Int};curdata=globaldata, randomoobs=[], varimparg = true)
    method,classes,notrees,randseed = Arguments
    s = size(curdata,1)
    srand(randseed)
    trainingdata = curdata
    trainingrefs = collect(1:s)
    trainingweights = getDfArrayData(trainingdata[:WEIGHT])
    regressionvalues = getDfArrayData(trainingdata[:REGRESSION])
    oobpredictions = Array(Array{Float64,1},s)
    for i = 1:s
        oobpredictions[i] = zeros(3)
    end
    timevalues = []
    eventvalues = []
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
        model[treeno], treevariableimportance, noleafs, treenoirregularleafs, oob[treeno] = generate_tree(method,trainingrefs,trainingweights,regressionvalues,timevalues,eventvalues,newtrainingdata,variables,types,oobpredictions,varimp = true)
        modelsize += noleafs
        noirregularleafs += treenoirregularleafs
        if (varimparg)
            variableimportance += treevariableimportance
        end
    end
   return (model,oobpredictions,variableimportance,modelsize,noirregularleafs,oob)
end

function find_missing_values(method::LearningMethod{Regressor},variables,trainingdata)
    missingvalues = Array(Array{Int,1},length(variables))
    nonmissingvalues = Array(Array,length(variables))
    for v = 1:length(variables)
        variable = variables[v]
        missingvalues[v] = Int[]
        nonmissingvalues[v] = typeof(trainingdata[variable]).parameters[1][]
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

function transform_nonmissing_columns_to_arrays(method::LearningMethod{Regressor},variables,trainingdata,missingvalues)
    newdata = Array(Array,length(variables))
    for v = 1:length(variables)
        if isempty(missingvalues[v])
            newdata[v] = getDfArrayData(trainingdata[variables[v]])
        end
    end
    return newdata
end

function sample_replacements_for_missing_values!(method::LearningMethod{Regressor},newtrainingdata,trainingdata,variables,types,missingvalues,nonmissingvalues)
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

function replacements_for_missing_values!(method::LearningMethod{Regressor},newtestdata,testdata,variables,types,missingvalues,nonmissingvalues)
    for v = 1:length(variables)
        if !isempty(missingvalues[v])
            variableType = typeof(testdata[variables[v]]).parameters[1]
            values = convert(Array{Nullable{variableType},1},testdata[variables[v]],Nullable{variableType}())
            newtestdata[v] = values
        end
    end
end

function generate_tree(method::LearningMethod{Regressor},trainingrefs,trainingweights,regressionvalues,timevalues,eventvalues,trainingdata,variables,types,oobpredictions; varimp = false)
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
        newregressionvalues = regressionvalues[nonzeroweights]
        model, variableimportance, noleafs, noirregularleafs = build_tree(method,newtrainingrefs,newtrainingweights,newregressionvalues,timevalues,eventvalues,trainingdata,variables,types,varimp)
        zeroweights = ~nonzeroweights
        oobrefs = trainingrefs[zeroweights]
        for oobref in oobrefs
            leafstats = make_prediction(model,trainingdata,oobref,0)
            oobprediction = leafstats[2]/leafstats[1]
            oobpredictions[oobref] += [1,oobprediction,oobprediction^2]
        end
    else
        model, variableimportance, noleafs, noirregularleafs = build_tree(method,trainingrefs,trainingweights,regressionvalues,timevalues,eventvalues,trainingdata,variables,types,varimp)
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
    return model, variableimportance, noleafs, noirregularleafs, zeroweights
end

function make_loo_prediction{T,S}(node::TreeNode{T,S},testdata,exampleno,prediction,emptyleaf=false)
    if node.nodeType == :LEAF
        return emptyleaf, node.prediction
    else
        examplevalue = testdata[node.varno][exampleno]
        if node.splittype == :NUMERIC
            nextnode=(examplevalue <= node.splitpoint) ? node.leftnode: node.rightnode
            if (nextnode.nodeType == :LEAF && nextnode.prediction[1] < 2.0)
                nextnode = (examplevalue > node.splitpoint) ? node.leftnode: node.rightnode
                emptyleaf = true
            end
        else # node.splittype == :CATEGORIC
            nextnode=(get(examplevalue) == node.splitpoint) ? node.leftnode: node.rightnode
            if (nextnode.nodeType == :LEAF && nextnode.prediction[1] < 2.0)
                nextnode = (examplevalue != node.splitpoint) ? node.leftnode: node.rightnode
                emptyleaf = true
            end
        end
        return make_loo_prediction(nextnode,testdata,exampleno,prediction,emptyleaf)
    end
end

function default_prediction(trainingweights,regressionvalues,timevalues,eventvalues,method::LearningMethod{Regressor})
    sumweights = sum(trainingweights)
    sumregressionvalues = sum(regressionvalues)
    return [sumweights,sumregressionvalues]
end

function leaf_node(node,method::LearningMethod{Regressor})
    if method.maxdepth > 0 && method.maxdepth == node.depth
        return true
    else
        noinstances = sum(node.trainingweights)
        if noinstances >= 2*method.minleaf
            firstvalue = node.regressionvalues[1]
            i = 2
            multiplevalues = false
            novalues = length(node.regressionvalues)
            while i <= novalues &&  ~multiplevalues
                multiplevalues = firstvalue != node.regressionvalues[i]
                i += 1
            end
            return ~multiplevalues
        else
            return true
        end
    end
end

function make_leaf(node,method::LearningMethod{Regressor}, parenttrainingweights)
    sumweights = sum(node.trainingweights)
    sumregressionvalues = sum(node.regressionvalues)
    return [sumweights,sumregressionvalues]
end

function find_best_split(node,trainingdata,variables,types,method::LearningMethod{Regressor})
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
            sampleregressionvalues = node.regressionvalues
        else
            sampletrainingweights = Array(Float64,splitsamplesize)
            sampletrainingrefs = Array(Float64,splitsamplesize)
            sampleregressionvalues = Array(Float64,splitsamplesize)
            for i = 1:splitsamplesize
                sampletrainingweights[i] = 1.0
                randindex = rand(1:length(node.trainingrefs))
                sampletrainingrefs[i] = node.trainingrefs[randindex]
                sampleregressionvalues[i] = node.regressionvalues[randindex]
            end
        end
    else
        sampletrainingrefs = node.trainingrefs
        sampletrainingweights = node.trainingweights
        sampleregressionvalues = node.regressionvalues
    end
    bestsplit = (-Inf,0,:NA,:NA,0.0)
    origregressionsum = sum(sampleregressionvalues .* sampletrainingweights)
    origweightsum = sum(sampletrainingweights)
    origmean = origregressionsum/origweightsum
    for v = 1:length(sampleselection)
        bestsplit = evaluate_variable_regression(bestsplit,sampleselection[v],variables[sampleselection[v]],types[sampleselection[v]],sampletrainingrefs,sampletrainingweights,
                                                 sampleregressionvalues,origregressionsum,origweightsum,origmean,trainingdata,method)
    end
    splitvalue, varno, variable, splittype, splitpoint = bestsplit
    if variable == :NA
        return :NA
    else
        return (varno,variable,splittype,splitpoint)
    end
end

function evaluate_variable_regression(bestsplit,varno,variable,splittype,trainingrefs,trainingweights,regressionvalues,origregressionsum,origweightsum,origmean,trainingdata,method)
    allvalues = trainingdata[varno][trainingrefs]
    if splittype == :CATEGORIC
        if method.randval
            bestsplit = evaluate_regression_categoric_variable_randval(bestsplit,varno,variable,splittype,regressionvalues,origregressionsum,origweightsum,origmean,allvalues,trainingweights,method)
        else
            bestsplit = evaluate_regression_categoric_variable_allvals(bestsplit,varno,variable,splittype,regressionvalues,origregressionsum,origweightsum,origmean,allvalues,trainingweights,method)
        end
    else # splittype == :NUMERIC
        if method.randval
            bestsplit = evaluate_regression_numeric_variable_randval(bestsplit,varno,variable,splittype,regressionvalues,origregressionsum,origweightsum,origmean,allvalues,trainingweights,method)
        else
            bestsplit = evaluate_regression_numeric_variable_allvals(bestsplit,varno,variable,splittype,regressionvalues,origregressionsum,origweightsum,origmean,allvalues,trainingweights,method)
        end

    end
    return bestsplit
end

function evaluate_regression_categoric_variable_randval(bestsplit,varno,variable,splittype,regressionvalues,origregressionsum,origweightsum,origmean,allvalues,trainingweights,method)
    key = allvalues[rand(1:end)]
    return evaluate_regression_common(key, ==,bestsplit,varno,variable,splittype,regressionvalues,origregressionsum,origweightsum,origmean,allvalues,trainingweights,method)
end

function evaluate_regression_categoric_variable_allvals(bestsplit,varno,variable,splittype,regressionvalues,origregressionsum,origweightsum,origmean,allvalues,trainingweights,method)
    keys = unique(allvalues)
    for key in keys
      bestsplit = evaluate_regression_common(key, ==,bestsplit,varno,variable,splittype,regressionvalues,origregressionsum,origweightsum,origmean,allvalues,trainingweights,method)
    end
    return bestsplit
end

function evaluate_regression_numeric_variable_randval(bestsplit,varno,variable,splittype,regressionvalues,origregressionsum,origweightsum,origmean,allvalues,trainingweights,method)
    minval = minimum(allvalues)
    maxval = maximum(allvalues)
    splitpoint = minval+rand()*(maxval-minval)
    bestsplit = evaluate_regression_common(splitpoint, <=,bestsplit,varno,variable,splittype,regressionvalues,origregressionsum,origweightsum,origmean,allvalues,trainingweights,method)
    return bestsplit
end

function evaluate_regression_numeric_variable_allvals(bestsplit,varno,variable,splittype,regressionvalues,origregressionsum,origweightsum,origmean,allvalues,trainingweights,method)
    numericvalues = Dict{typeof(allvalues[1]), Array{Float64,1}}()
    for i = 1:length(allvalues)
        numericvalues[allvalues[i]] = get(numericvalues,allvalues[i],[0,0]) .+ [trainingweights[i]*regressionvalues[i],trainingweights[i]]
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
            bestsplit = (variancereduction,varno,variable,splittype,sortedkeys[s])
        end
    end
    return bestsplit
end


function evaluate_regression_common(key, op, bestsplit,varno,variable,splittype,regressionvalues,origregressionsum,origweightsum,origmean,allvalues,trainingweights,method)
  leftregressionsum = 0.0
  leftweightsum = 0.0
  for i = 1:length(allvalues)
      if op(allvalues[i], key)
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
  return bestsplit
end

function make_split(method::LearningMethod{Regressor},node,trainingdata,bestsplit)
  (varno, variable, splittype, splitpoint) = bestsplit
  leftrefs = Int[]
  leftweights = Float64[]
  leftregressionvalues = Float64[]
  rightrefs = Int[]
  rightweights = Float64[]
  rightregressionvalues = Float64[]
  allvalues = trainingdata[varno][node.trainingrefs]
  sumleftweights = 0.0
  sumrightweights = 0.0
  op = splittype == :NUMERIC ? (<=) : (==)
  for r = 1:length(node.trainingrefs)
      ref = node.trainingrefs[r]
      if op(allvalues[r], splitpoint)
          push!(leftrefs,ref)
          push!(leftweights,node.trainingweights[r])
          sumleftweights += node.trainingweights[r]
          push!(leftregressionvalues,node.regressionvalues[r])
      else
          push!(rightrefs,ref)
          push!(rightweights,node.trainingweights[r])
          sumrightweights += node.trainingweights[r]
          push!(rightregressionvalues,node.regressionvalues[r])
      end
  end
  leftweight = sumleftweights/(sumleftweights+sumrightweights)
  return leftrefs,leftweights,leftregressionvalues,[],[],rightrefs,rightweights,rightregressionvalues,[],[],leftweight
end

function generate_model_internal(method::LearningMethod{Regressor}, oobs, classes)
    if method.conformal == :default
        conformal = :std
    else
        conformal = method.conformal
    end
    oobpredictions = oobs[1]
    for r = 2:length(oobs)
        oobpredictions += oobs[r]
    end
    correcttrainingvalues = globaldata[:REGRESSION]
    oobse = 0.0
    nooob = 0
    ooberrors = Float64[]
    alphas = Float64[]
    for i = 1:length(correcttrainingvalues)
        oobpredcount = oobpredictions[i][1]
        if oobpredcount > 0.0
            ooberror = abs(correcttrainingvalues[i]-(oobpredictions[i][2]/oobpredcount))
            push!(ooberrors,ooberror)
            delta = (oobpredictions[i][3]/oobpredcount-(oobpredictions[i][2]/oobpredcount)^2)
            alpha = ooberror/(delta+0.01)
            push!(alphas,alpha)
            oobse += ooberror^2
            nooob += 1
        end
    end
    oobperformance = oobse/nooob
    if conformal == :std
        conformalfunction = (:std,minimum(correcttrainingvalues),maximum(correcttrainingvalues),sort(ooberrors, rev=true))
    elseif conformal == :normalized
        conformalfunction = (:normalized,minimum(correcttrainingvalues),maximum(correcttrainingvalues),sort(alphas,rev=true))
    end
    return oobperformance, conformalfunction
end

function apply_model_internal(model::PredictionModel{Regressor}; confidence = 0.95)
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
        nooob = size(model.conformal[4],1)
        minvalue = model.conformal[2]
        maxvalue = model.conformal[3]
        thresholdindex = floor(Int,(nooob+1)*(1-confidence))
        if thresholdindex >= 1
            errorrange = model.conformal[4][thresholdindex]
        else
            errorrange = maxvalue-minvalue
        end
        results = Array(Tuple,size(predictions,1))
        for i = 1:size(predictions,1)
            predictedlower = max(predictions[i]-errorrange,minvalue)
            predictedupper = min(predictions[i]+errorrange,maxvalue)
            results[i] = (predictions[i],[predictedlower,predictedupper])
        end
    elseif model.conformal[1] == :normalized
        nooob = size(model.conformal[4],1)
        minvalue = model.conformal[2]
        maxvalue = model.conformal[3]
        thresholdindex = floor(Int,(nooob+1)*(1-confidence))
        if thresholdindex >= 1
            alpha = model.conformal[4][thresholdindex]
        else
            alpha = Inf
        end
        results = Array(Tuple,size(predictions,1))
        for i = 1:size(predictions,1)
            delta = squaredpredictions[i]-predictions[i]^2
            errorrange = alpha*(delta+0.01)
            predictedlower = max(predictions[i]-errorrange,minvalue)
            predictedupper = min(predictions[i]+errorrange,maxvalue)
            results[i] = (predictions[i],[predictedlower,predictedupper])
        end
    end
    return results
end

function apply_trees(Arguments::Tuple{LearningMethod{Regressor},Array,Array})
    method, classes, trees = Arguments
    variables, types = get_variables_and_types(globaldata)
    testmissingvalues, testnonmissingvalues = find_missing_values(method,variables,globaldata)
    newtestdata = transform_nonmissing_columns_to_arrays(method,variables,globaldata,testmissingvalues)
    replacements_for_missing_values!(method,newtestdata,globaldata,variables,types,testmissingvalues,testnonmissingvalues)
    nopredictions = size(globaldata,1)
    predictions = Array(Float64,nopredictions)
    squaredpredictions = Array(Float64,nopredictions)
    for i = 1:nopredictions
        predictions[i] = 0.0
        squaredpredictions[i] = 0.0
        for t = 1:length(trees)
            leafstats = make_prediction(trees[t],newtestdata,i,0)
            treeprediction = leafstats[2]/leafstats[1]
            predictions[i] += treeprediction
            squaredpredictions[i] += treeprediction^2
        end
    end
    results = (predictions,squaredpredictions)
    return results
end

function collect_results_split(method::LearningMethod{Regressor}, randomoobs, results, time)
    modelsize = sum([result[1] for result in results])
    noirregularleafs = sum([result[6] for result in results])
    predictions = results[1][2]
    for r = 2:length(results)
        predictions += results[r][2]
    end
    nopredictions = size(predictions,1)
    predictions = [predictions[i][2]/predictions[i][1] for i = 1:nopredictions]
    if method.conformal == :default
        conformal = :normalized
    else
        conformal = method.conformal
    end
    if conformal == :normalized || conformal == :binning
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
    correcttrainingvalues = trainingdata[:REGRESSION]
    oobse = 0.0
    nooob = 0
    ooberrors = Float64[]
    alphas = Float64[]
    deltas = Float64[]
    minvalue = minimum(correcttrainingvalues)
    maxvalue = maximum(correcttrainingvalues)
    for i = 1:length(correcttrainingvalues)
        oobpredcount = oobpredictions[i][1]
        if oobpredcount > 0.0
            ooberror = abs(correcttrainingvalues[i]-(oobpredictions[i][2]/oobpredcount))
            push!(ooberrors,ooberror)
            if conformal != :std 
                delta = (oobpredictions[i][3]/oobpredcount-(oobpredictions[i][2]/oobpredcount)^2)
                push!(deltas,delta)
                if conformal == :normalized
                    alpha = ooberror/(delta+0.01)
                    push!(alphas,alpha)
                end
            end            
            oobse += ooberror^2
            nooob += 1
        end
    end
    oobmse = oobse/nooob
    if conformal == :std
        thresholdindex = Int(floor((nooob+1)*(1-method.confidence)))
        if thresholdindex >= 1
            errorrange = sort(ooberrors, rev=true)[thresholdindex]
        else
            errorrange = maxvalue-minvalue
        end
    elseif conformal == :normalized
        thresholdindex = Int(floor((nooob+1)*(1-method.confidence)))
        if thresholdindex >= 1
            alpha = sort(alphas, rev=true)[thresholdindex]
        else
            alpha = Inf
        end
    elseif conformal == :binning
        deltaerrors = sortrows(hcat(deltas,ooberrors),by=x->x[1])        
        mincalibrationsize = max(1,Int(round(1/(1-method.confidence)-1.0)))
        binthresholds = Any[]
        oobindex = 0
        while size(deltaerrors,1)-oobindex >= 2*mincalibrationsize
            bin = deltaerrors[oobindex+1:oobindex+mincalibrationsize,:]
            threshold = bin[end,1]
            push!(binthresholds,(threshold,2*maximum(bin[:,2])))
            oobindex += mincalibrationsize
        end
        bin = deltaerrors[oobindex+1:end,:]
        threshold = bin[end,1]
        push!(binthresholds,(threshold,2*maximum(bin[:,2])))
    end
    testdata = globaldata[globaldata[:TEST] .== true,:]
    correctvalues = testdata[:REGRESSION]
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
            if isnan(errorrange)
                errorrange = maxvalue-minvalue
            end
        elseif conformal == :normalized
            delta = squaredpredictions[i]-predictions[i]^2
            errorrange = alpha*(delta+0.01)
            if isnan(errorrange)
                errorrange = maxvalue-minvalue
            end
        elseif conformal == :binning ## Needs to be modified for modpred
            delta = squaredpredictions[i]-predictions[i]^2
            notfoundthreshold = true
            thresholdcounter = 1
            while notfoundthreshold && thresholdcounter <= size(binthresholds,1)
                if delta > binthresholds[thresholdcounter][1]
                    thresholdcounter += 1
                else
                    errorrange = binthresholds[thresholdcounter][2]
                    notfoundthreshold = false
                end
            end
            if notfoundthreshold
                errorrange = maxvalue-minvalue
            end
        end
        predictedlower = max(predictions[i]-errorrange,minvalue)
        predictedupper = min(predictions[i]+errorrange,maxvalue)
        predictedrange = predictedupper-predictedlower
        rangesum += predictedrange
        if predictedlower <= correctvalues[i] <= predictedupper
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
    return RegressionResult(mse,corrcoeff,avmse,varmse,esterr,absesterr,validity,region,modelsize,noirregularleafs,time+extratime)
end

function collect_results_cross_validation(method::LearningMethod{Regressor}, randomoobs, results, modelsizes, nofolds, time)
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
    if method.conformal == :default
        conformal = :normalized
    else
        conformal = method.conformal
    end
    nopredictions = size(globaldata,1)
    testexamplecounter = 0
    predictions = [predictions[i][2]/predictions[i][1] for i = 1:nopredictions]
    mse = Array(Float64,nofolds)
    corrcoeff = Array(Float64,nofolds)
    avmse = Array(Float64,nofolds)
    varmse = Array(Float64,nofolds)
    oobmse = Array(Float64,nofolds)
    esterr = Array(Float64,nofolds)
    absesterr = Array(Float64,nofolds)
    validity = Array(Float64,nofolds)
    region = Array(Float64,nofolds)
#    errsizecor = Array(Float64,nofolds)
#    valsizecor = Array(Float64,nofolds)
    foldno = 0
    if conformal == :normalized || conformal == :binning
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
        correctvalues = testdata[:REGRESSION]
        correcttrainingvalues = globaldata[globaldata[:FOLD] .!= fold,:REGRESSION]
        oobpredictions = results[1][4][foldno]
        for r = 2:length(results)
            oobpredictions += results[r][4][foldno]
        end
        oobse = 0.0
        nooob = 0
        ooberrors = Float64[]
        alphas = Float64[]
        deltas = Float64[]
        minvalue = minimum(correcttrainingvalues)
        maxvalue = maximum(correcttrainingvalues)
        for i = 1:length(correcttrainingvalues)
            oobpredcount = oobpredictions[i][1]
            if oobpredcount > 0
                ooberror = abs(correcttrainingvalues[i]-(oobpredictions[i][2]/oobpredcount))
                push!(ooberrors,ooberror)
                if conformal != :std 
                    delta = (oobpredictions[i][3]/oobpredcount)-(oobpredictions[i][2]/oobpredcount)^2
                    push!(deltas,delta)
                    if conformal == :normalized
                        alpha = ooberror/(delta+0.01)
                        push!(alphas,alpha)
                    end
                end
                oobse += ooberror^2
                nooob += 1
            end
        end
        if conformal == :std
            thresholdindex = Int(floor((nooob+1)*(1-method.confidence)))
            if thresholdindex >= 1
                errorrange = sort(ooberrors, rev=true)[thresholdindex]
            else
                errorrange = maxvalue-minvalue
            end
        elseif conformal == :normalized
            thresholdindex = Int(floor((nooob+1)*(1-method.confidence)))
            if thresholdindex >= 1
                alpha = sort(alphas,rev=true)[thresholdindex]
            else
                alpha = Inf
            end
        elseif conformal == :binning
            deltaerrors = sortrows(hcat(deltas,ooberrors),by=x->x[1])        
            mincalibrationsize = max(1,Int(round(1/(1-method.confidence)-1.0)))
            binthresholds = Any[]
            oobindex = 0
            while size(deltaerrors,1)-oobindex >= 2*mincalibrationsize
                bin = deltaerrors[oobindex+1:oobindex+mincalibrationsize,:]
                threshold = bin[end,1]
                push!(binthresholds,(threshold,maximum(bin[:,2])))
                oobindex += mincalibrationsize
            end
            bin = deltaerrors[oobindex+1:end,:]
            threshold = bin[end,1]
            push!(binthresholds,(threshold,maximum(bin[:,2])))
        end
        msesum = 0.0
        noinregion = 0.0
        rangesum = 0.0
        testerrors = Array(Float64,length(correctvalues))
#        testsizes = Array(Float64,length(correctvalues))
        testvals = Array(Float64,length(correctvalues))
        for i = 1:length(correctvalues)
            error = abs(correctvalues[i]-predictions[testexamplecounter+i])
            testerrors[i] = error
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
#                        testdeltas[i] = delta
                errorrange = alpha*(delta+0.01)
                if isnan(errorrange)
                    errorrange = maxvalue-minvalue
                end
            elseif conformal == :normalized
                delta = (squaredpredictions[testexamplecounter+i]-predictions[testexamplecounter+i]^2)
                #                        testdeltas[i] = delta
                errorrange = alpha*(delta+0.01)
                if isnan(errorrange)
                    errorrange = maxvalue-minvalue
                end
            elseif conformal == :binning ## Needs to be modified for modpred
                delta = squaredpredictions[testexamplecounter+i]-predictions[testexamplecounter+i]^2
                notfoundthreshold = true
                thresholdcounter = 1
                while notfoundthreshold && thresholdcounter <= size(binthresholds,1)
                    if delta > binthresholds[thresholdcounter][1]
                        thresholdcounter += 1
                    else
                        errorrange = binthresholds[thresholdcounter][2]
                        notfoundthreshold = false
                    end
                end
                if notfoundthreshold
                    errorrange = maxvalue-minvalue
                end
            end
            curIndeces = find(foldIndeces)
            curIndex = curIndeces[i]
            predictedlower = max(predictions[testexamplecounter+i]-errorrange,minvalue)
            predictedupper = min(predictions[testexamplecounter+i]+errorrange,maxvalue)
            predictedrange = predictedupper-predictedlower
#            testsizes[i] = predictedrange
#            prediction_results[curIndex] = (predictions[testexamplecounter+i],[predictedlower,predictedupper])
            rangesum += predictedrange
            if predictedlower <= correctvalues[i] <= predictedupper
                noinregion += 1
                testvals[i] = 1
            else
                testvals[i] = 0
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
        ## errsizecor[foldno] = cor(testerrors,testsizes)
        ## if isnan(errsizecor[foldno])
        ##     errsizecor[foldno] = 0
        ## end
        ## valsizecor[foldno] = cor(testvals,testsizes)
        ## if isnan(valsizecor[foldno])
        ##     valsizecor[foldno] = 0
        ## end        
    end
    extratime = toq()
    return RegressionResult(mean(mse),mean(corrcoeff),mean(avmse),mean(varmse),mean(esterr),mean(absesterr),mean(validity),mean(region),
                            mean(modelsizes),mean(noirregularleafs),time+extratime)
end
