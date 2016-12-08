function run_split_internal(method::LearningMethod{Classifier}, results, time)
    modelsize = sum([result[1] for result in results])
    noirregularleafs = sum([result[5] for result in results])
    predictions = results[1][2]
    for r = 2:length(results)
        predictions += results[r][2]
    end
    nopredictions = size(predictions,1)

    predictions = [predictions[i][2:end]/predictions[i][1] for i = 1:nopredictions]
    classes = unique(globaldata[:CLASS])
    noclasses = length(classes)
    classdata = Array(Any,noclasses)
    for c = 1:noclasses
        classdata[c] = globaldata[globaldata[:CLASS] .== classes[c],:]
    end
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
                    alpha = oobpredictions[c][i][c+1]/oobpredcount-maximum(oobpredictions[c][i][[2:c;c+2:noclasses+1]])/oobpredcount
                    push!(alphas,alpha)
                    noobcorrect += 1-abs(sign(indmax(oobpredictions[c][i][2:end])-c))
                    nooob += 1
                end
            end
        end
        thresholdindex = Int(floor((nooob+1)*(1-method.confidence)))
        if thresholdindex >= 1
            alpha = sort(alphas)[thresholdindex]
        else
            alpha = -Inf
        end
        classalpha = fill(alpha, noclasses)
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
        classalpha = Array(Float64,noclasses)
        classalphas = Array(Any,noclasses)
        for c = 1:noclasses
            alphas = Float64[]
            noclassoob = 0
            for i = 1:size(oobpredictions[c],1)
                oobpredcount = oobpredictions[c][i][1]
                if oobpredcount > 0
                    alphavalue = oobpredictions[c][i][c+1]/oobpredcount-maximum(oobpredictions[c][i][[2:c;c+2:noclasses+1]])/oobpredcount
                    push!(alphas,alphavalue)
                    noobcorrect += 1-abs(sign(indmax(oobpredictions[c][i][2:end])-c))
                    noclassoob += 1
                end
            end
            classalphas[c] = alphas
            thresholdindex = Int(floor((noclassoob+1)*(1-method.confidence)))
            if thresholdindex >= 1
                classalpha[c] = sort(alphas)[thresholdindex]
            else
                classalpha[c] = -Inf
            end
            nooob += noclassoob
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
                if method.modpred
                    if conformal == :std
                        randomoob = randomoobs[i]
                        thresholdindex = Int(floor(nooob*(1-method.confidence)))
                        if thresholdindex >= 1
                            alpha = sort(alphas[[1:randomoob-1;randomoob+1:end]])[thresholdindex] # NOTE: assumes oobpredcount > 0 always is true!
                        else
                            alpha = -Inf
                        end
                    else # conformal == :classcond
                        randomoobclass, randomoobref = randomclassoobs[i]
                        thresholdindex = Int(floor(size(classalphas[randomoobclass],1)*(1-method.confidence)))
                        origclassalpha = classalpha[randomoobclass]
                        if thresholdindex >= 1
                            classalpha[randomoobclass] = sort(classalphas[randomoobclass][[1:randomoobref-1;randomoobref+1:end]])[thresholdindex] # NOTE: assumes oobpredcount > 0 always is true!
                        else
                            classalpha[randomoobclass] = -Inf
                        end
                        alpha = classalpha[c]
                    end
                else
                    if conformal == :classcond
                        alpha = classalpha[c]
                    end
                end
                if margin >= alpha
                    noinrangesum += 1
                end
                nolabels = 0
                if conformal == :std
                    for j = 1:noclasses
                        if j == mostprobable
                            if predictions[i][j]-maximum(predictions[i][[1:j-1;j+1:end]]) >= alpha
                                nolabels += 1
                            end
                        else
                            if predictions[i][j]-predictions[i][mostprobable] >= alpha # classalphas[j]
                                nolabels += 1
                            end
                        end
                    end
                else # conformal == :classcond
                    for j = 1:noclasses
                        if j == mostprobable
                            if predictions[i][j]-maximum(predictions[i][[1:j-1;j+1:end]]) >= classalpha[j]
                                nolabels += 1
                            end
                        else
                            if predictions[i][j]-predictions[i][mostprobable] >= classalpha[j]
                                nolabels += 1
                            end
                        end
                    end
                    if method.modpred
                        classalpha[randomoobclass] = origclassalpha
                    end
                end
                nolabelssum += nolabels
                if nolabels == 1
                    noonec += correct
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
    return ClassificationResult(accuracy,weightedauc,brierscore,avacc,esterr,absesterr,avbrier,varbrier,margin,prob,validity,avc,onec,modelsize,noirregularleafs,time+extratime), get_predictions_classification(classes, predictions, classalpha)

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


function run_cross_validation_internal(method::LearningMethod{Classifier}, results, modelsizes, nofolds, conformal, time)
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
    classes = unique(globaldata[:CLASS])
    noclasses = length(classes)
    foldauc = Array(Float64,noclasses)
    classdata = Array(Any,noclasses)
    returning_prediction = []
    for c = 1:noclasses
        classdata[c] = globaldata[globaldata[:CLASS] .== classes[c],:]
    end
    testdata = Array(Any,noclasses)
    foldno = 0
    for fold in folds
        foldno += 1
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
                        alpha = oobpredictions[c][i][c+1]/oobpredcount-maximum(oobpredictions[c][i][[2:c;c+2:noclasses+1]])/oobpredcount
                        push!(alphas,alpha)
                        noobcorrect += 1-abs(sign(indmax(oobpredictions[c][i][2:end])-c))
                        nooob += 1
                    end
            end
            end
            thresholdindex = Int(floor((nooob+1)*(1-method.confidence)))
            if thresholdindex >= 1
                alpha = sort(alphas)[thresholdindex]
            else
                alpha = -Inf
            end
            classalphas = [alpha for j=1:noclasses]
        elseif conformal == :classcond
            classalphas = Array(Float64,noclasses)
            for c = 1:noclasses
                alphas = Float64[]
                noclassoob = 0
                for i = 1:size(oobpredictions[c],1)
                    oobpredcount = oobpredictions[c][i][1]
                    if oobpredcount > 0
                        alphavalue = oobpredictions[c][i][c+1]/oobpredcount-maximum(oobpredictions[c][i][[2:c;c+2:noclasses+1]])/oobpredcount
                        push!(alphas,alphavalue)
                        noobcorrect += 1-abs(sign(indmax(oobpredictions[c][i][2:end])-c))
                        noclassoob += 1
                    end
                end
                thresholdindex = Int(floor((noclassoob+1)*(1-method.confidence)))
                if thresholdindex >= 1
                    classalphas[c] = sort(alphas)[thresholdindex]
                else
                classalphas[c] = -Inf
                end
                nooob += noclassoob
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
                    if examplemargin >= classalphas[c]
                        noinrangesum += 1
                    end
                    nolabels = 0
                    for j = 1:noclasses
                        if j == mostprobable
                            if predictions[i][j]-maximum(predictions[i][[1:j-1;j+1:end]]) >= classalphas[j]
                                nolabels += 1
                            end
                        else
                            if predictions[i][j]-predictions[i][mostprobable] >= classalphas[j]
                                nolabels += 1
                            end
                        end
                    end
                    nolabelssum += nolabels
                    if nolabels == 1
                        noonec += correct
                    end
                end
                testexamplecounter += size(testdata[c],1)
            end
        end
        foldpredictions = predictions[origtestexamplecounter+1:testexamplecounter]
        append!(returning_prediction, get_predictions_classification(classes, foldpredictions, classalphas))
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
                                                mean(validity),mean(avc),mean(onec),mean(modelsizes),mean(noirregularleafs),time+extratime), returning_prediction
end

##
## Functions to be executed on each worker
##
function generate_and_test_trees(Arguments::Tuple{LearningMethod{Classifier},Symbol,Int64,Int64,Array{Int64,1}})
    method,experimentype,notrees,randseed,randomoobs = Arguments
    classes = unique(globaldata[:CLASS])
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
