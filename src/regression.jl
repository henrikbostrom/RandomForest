function generate_trees(Arguments::Tuple{LearningMethod{Regressor},Array{Int,1},Int,Int})
    method,classes,notrees,randseed = Arguments
    s = size(globaldata,1)
    srand(randseed)
    trainingdata = globaldata
    trainingrefs = collect(1:size(trainingdata,1))
    trainingweights = trainingdata[:WEIGHT]
    regressionvalues = trainingdata[:REGRESSION]
    oobpredictions = Array(Any,size(trainingdata,1))
    for i = 1:size(trainingdata,1)
        oobpredictions[i] = [0,0,0]
    end
    timevalues = []
    eventvalues = []
    # starting from here till the end of the function is duplicated between here and the Classifier and Survival dispatchers
    variables, types = get_variables_and_types(globaldata)
    modelsize = 0
    missingvalues, nonmissingvalues = find_missing_values(method,variables,trainingdata)
    newtrainingdata = transform_nonmissing_columns_to_arrays(method,variables,trainingdata,missingvalues)
    model = Array(Any,notrees)
    variableimportance = zeros(size(variables,1))
    for treeno = 1:notrees
        sample_replacements_for_missing_values!(method,newtrainingdata,trainingdata,variables,types,missingvalues,nonmissingvalues)
        model[treeno], treevariableimportance, noleafs, noirregularleafs = generate_tree(method,trainingrefs,trainingweights,regressionvalues,timevalues,eventvalues,newtrainingdata,variables,types,oobpredictions,varimp = true)
        modelsize += noleafs
        variableimportance += treevariableimportance
    end
   return (model,oobpredictions,variableimportance)
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
            newdata[v] = convert(Array,trainingdata[variables[v]])
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
            newtrainingdata[v] = convert(Array,values)
        end
    end
end

function replacements_for_missing_values!(method::LearningMethod{Regressor},newtestdata,testdata,variables,types,missingvalues,nonmissingvalues)
    for v = 1:length(variables)
        if !isempty(missingvalues[v])
            variableType = typeof(testdata[variables[v]]).parameters[1]
            values = convert(Array{Nullable{variableType},1},testdata[variables[v]],Nullable{variableType}())
            for i in missingvalues[v]
                values[i] =  Nullable{variableType}()
            end
            newtestdata[v] = values
        end
    end
end

function generate_tree(method::LearningMethod{Regressor},trainingrefs,trainingweights,regressionvalues,timevalues,eventvalues,trainingdata,variables,types,oobpredictions; varimp = false)
    zeroweights = Array{Bool,1}()
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
    if varimp
        return model, variableimportance, noleafs, noirregularleafs, zeroweights
    else
        return model, noleafs, noirregularleafs, zeroweights
    end
end

function make_loo_prediction(tree,testdata,exampleno,prediction)
    nodeno = 1
    nonterminal = true
    emptyleaf = false
    while nonterminal
        node = tree[nodeno]
        if node[1] == :LEAF
            prediction = emptyleaf,node[2]
            nonterminal = false
        else
            varno, splittype, splitpoint, splitweight = node[1]
            examplevalue = testdata[varno][exampleno]
            if splittype == :NUMERIC
                if examplevalue <= splitpoint
                    leftchild = tree[node[2]]
                    if leftchild[1] == :LEAF && leftchild[2][1] < 2.0
                        nodeno = node[3]
                        emptyleaf = true
                    else
                        nodeno = node[2]
                    end
                else
                    rightchild = tree[node[3]]
                    if rightchild[1] == :LEAF && rightchild[2][1] < 2.0
                        nodeno = node[2]
                        emptyleaf = true
                    else
                        nodeno = node[3]
                    end
                end
            else
                if examplevalue == splitpoint
                    leftchild = tree[node[2]]
                    if leftchild[1] == :LEAF && leftchild[2][1] < 2.0
                        nodeno = node[3]
                        emptyleaf = true
                    else
                        nodeno = node[2]
                    end
                else
                    rightchild = tree[node[3]]
                    if rightchild[1] == :LEAF && rightchild[2][1] < 2.0
                        nodeno = node[2]
                        emptyleaf = true
                    else
                        nodeno = node[3]
                    end
                end
            end
        end
    end
    return prediction
end

function default_prediction(trainingweights,regressionvalues,timevalues,eventvalues,method::LearningMethod{Regressor})
    sumweights = sum(trainingweights)
    sumregressionvalues = sum(regressionvalues)
    return [sumweights,sumregressionvalues]
    ## if sumweights > 0
    ##     return sum(trainingweights .* regressionvalues)/sumweights
    ## else
    ##     return :NA
    ## end
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

function make_leaf(node,method::LearningMethod{Regressor})
    sumweights = sum(node.trainingweights)
    sumregressionvalues = sum(node.regressionvalues)
    return [sumweights,sumregressionvalues]
    ## if sumweights > 0
    ##     prediction = sum(trainingweights .* regressionvalues)/sumweights
    ## else
    ##     prediction = defaultprediction
    ## end
    return prediction
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
    return bestsplit
end

function evaluate_regression_categoric_variable_allvals(bestsplit,varno,variable,splittype,regressionvalues,origregressionsum,origweightsum,origmean,allvalues,trainingweights,method)
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

function evaluate_regression_numeric_variable_randval(bestsplit,varno,variable,splittype,regressionvalues,origregressionsum,origweightsum,origmean,allvalues,trainingweights,method)
    minval = minimum(allvalues)
    maxval = maximum(allvalues)
    splitpoint = minval+rand()*(maxval-minval)
    leftregressionsum = 0.0
    leftweightsum = 0.0
    for i = 1:length(allvalues)
        if allvalues[i] <= splitpoint
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
            bestsplit = (variancereduction,varno,variable,splittype,splitpoint)
        end
    end
    return bestsplit
end

function evaluate_regression_numeric_variable_allvals(bestsplit,varno,variable,splittype,regressionvalues,origregressionsum,origweightsum,origmean,allvalues,trainingweights,method)
    numericvalues = Dict{typeof(allvalues[1]).parameters[1], Array{Float64,1}}()
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
    # splitpoints = sortrows(splitpoints,by=x->x[1])
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

function make_split(method::LearningMethod{Regressor},node,trainingdata,bestsplit)
  (varno, variable, splittype, splitpoint) = bestsplit
  leftrefs = Int[]
  leftweights = Float64[]
  leftregressionvalues = Float64[]
  rightrefs = Int[]
  rightweights = Float64[]
  rightregressionvalues = Float64[]
  values = trainingdata[varno][node.trainingrefs]
  sumleftweights = 0.0
  sumrightweights = 0.0
  if splittype == :NUMERIC
      for r = 1:length(node.trainingrefs)
          ref = node.trainingrefs[r]
          if values[r] <= splitpoint
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
  else
      for r = 1:length(node.trainingrefs)
          ref = node.trainingrefs[r]
          if values[r] == splitpoint
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
  end
  leftweight = sumleftweights/(sumleftweights+sumrightweights)
  return leftrefs,leftweights,leftregressionvalues,[],[],rightrefs,rightweights,rightregressionvalues,[],[],leftweight
end

function generate_model_internal(method::LearningMethod{Regressor}, oobs, classes)
    oobpredictions = oobs[1]
    for r = 2:length(oobs)
        oobpredictions += oobs[r]
    end
    correcttrainingvalues = globaldata[:REGRESSION]
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
    thresholdindex = floor(Int,((nooob+1)*(1-method.confidence)))
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

function apply_model(model::PredictionModel{Regressor}; confidence = :std)
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
        Threads.@threads for subtrees in notrees
            results = apply_trees((model.method,[],subtrees))
            predictions += results[1]
            squaredpredictions += results[2]
        end
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
            thresholdindex = floor(Int,(nooob+1)*(1-confidence))
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
            thresholdindex = floor(Int((nooob+1)*(1-confidence)))
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
