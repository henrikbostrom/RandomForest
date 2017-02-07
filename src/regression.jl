
function collect_results_split(method::LearningMethod{Regressor}, results, time)
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
    if conformal == :normalized ## || conformal == :isotonic
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
    ## if conformal == :rfok
    ##     labels = names(trainingdata)[[!(l in [:REGRESSION,:WEIGHT,:TEST,:FOLD,:ID]) for l in names(trainingdata)]]
    ##     rows = Array{Float64}[]
    ##     knnalphas = Float64[]
    ## end
    largestrange = maximum(correcttrainingvalues)-minimum(correcttrainingvalues)
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
            ## if conformal == :rfok
            ##     push!(rows,hcat(ooberror,convert(Array{Float64},trainingdata[i,labels])))
            ## end
            oobse += ooberror^2
            nooob += 1
        end
    end
    ## if conformal == :rfok
    ##     maxk = minimum([45,nooob-1])
    ##     deltasalphas = Array(Float64,nooob,maxk,2)
    ##     distancematrix = Array(Float64,nooob,nooob)
    ##     for r = 1:nooob-1
    ##         distancematrix[r,r] = 0.001
    ##         for k = r+1:nooob
    ##             distance = sqL2dist(rows[r][2:end],rows[k][2:end])+0.001 # Small term added to prevent infinite weights
    ##             distancematrix[r,k] = distance
    ##             distancematrix[k,r] = distance
    ##         end
    ##     end
    ##     distancematrix[nooob,nooob] = 0.001
    ##     for r = 1:nooob
    ##         distanceerrors = Array(Float64,nooob,2)
    ##         for n = 1:nooob
    ##             distanceerrors[n,1] = distancematrix[r,n]
    ##             distanceerrors[n,2] = rows[n][1]
    ##         end
    ##         distanceerrors = sortrows(distanceerrors,by=x->x[1])
    ##         knndelta = 0.0
    ##         distancesum = 0.0
    ##         for k = 2:maxk+1
    ##             knndelta += distanceerrors[k,2]/distanceerrors[k,1]
    ##             distancesum += 1/distanceerrors[k,1]
    ##             deltasalphas[r,k-1,1] = knndelta/distancesum
    ##             if 2*rows[r][1] > largestrange
    ##                 deltasalphas[r,k-1,2] = largestrange/(deltasalphas[r,k-1,1]+0.01)
    ##             else
    ##                 deltasalphas[r,k-1,2] = 2*rows[r][1]/(deltasalphas[r,k-1,1]+0.01)
    ##             end
    ##         end
    ##     end
    ##     thresholdindex = Int(floor((nooob+1)*(1-method.confidence)))
    ##     if thresholdindex >= 1
    ##         lowestrangesum = Inf
    ##         bestk = Inf
    ##         bestalpha = Inf
    ##         for k = 1:maxk
    ##             knnalpha = sort(deltasalphas[:,k,2],rev=true)[thresholdindex]
    ##             knnrangesum = 0.0
    ##             for j = 1:nooob
    ##                 knnrangesum += knnalpha*(deltasalphas[j,k,1]+0.01)
    ##             end
    ##             if knnrangesum < lowestrangesum
    ##                 lowestrangesum = knnrangesum
    ##                 bestk = k
    ##                 bestalpha = knnalpha
    ##             end
    ##         end
    ##     else
    ##         bestalpha = Inf
    ##         bestk = 1
    ##     end
    ## end
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
    ## elseif conformal == :rfok
    ##     alpha = bestalpha
    ## elseif conformal == :isotonic
    ##     eds = Array(Any,nooob,2)
    ##     eds[:,1] = ooberrors
    ##     eds[:,2] = deltas
    ##     eds = sortrows(eds,by=x->x[2])
    ##     mincalibrationsize = maximum([10,Int(floor(1/(round((1-method.confidence)*1000000)/1000000))-1)])
    ##     isotonicthresholds = Any[]
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
            if isnan(errorrange) || errorrange > largestrange
                errorrange = largestrange
            end
        elseif conformal == :normalized
            delta = squaredpredictions[i]-predictions[i]^2
            errorrange = alpha*(delta+0.01)
            if isnan(errorrange) || errorrange > largestrange
                errorrange = largestrange
            end
        ## elseif conformal == :rfok
        ##     testrow = convert(Array{Float64},testdata[i,labels])
        ##     distanceerrors = Array(Float64,nooob,2)
        ##     for n = 1:nooob
        ##         distanceerrors[n,1] = sqL2dist(testrow,rows[n][2:end])+0.001 # Small term added to prevent infinite weights
        ##         distanceerrors[n,2] = rows[n][1]
        ##     end
        ##     distanceerrors = sortrows(distanceerrors,by=x->x[1])
        ##     knndelta = 0.0
        ##     distancesum = 0.0
        ##     for j = 1:bestk
        ##         knndelta += distanceerrors[j,2]/distanceerrors[j,1]
        ##         distancesum += 1/distanceerrors[j,1]
        ##     end
        ##     knndelta /= distancesum
        ##     errorrange = alpha*(knndelta+0.01)
        ##     if isnan(errorrange) || errorrange > largestrange
        ##         errorrange = largestrange
        ##     end
        ## elseif conformal == :isotonic
        ##     delta = squaredpredictions[i]-predictions[i]^2
        ##     foundthreshold = false
        ##     thresholdcounter = size(isotonicthresholds,1)
        ##     while ~foundthreshold && thresholdcounter > 0
        ##         if delta >= isotonicthresholds[thresholdcounter][1]
        ##             errorrange = isotonicthresholds[thresholdcounter][2]*2
        ##             foundthreshold = true
        ##         else
        ##             thresholdcounter -= 1
        ##         end
        ## end
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
    return RegressionResult(mse,corrcoeff,avmse,varmse,esterr,absesterr,validity,region,modelsize,noirregularleafs,time+extratime), prediction_results
end

function collect_results_cross_validation(method::LearningMethod{Regressor}, results, modelsizes, nofolds, conformal, time)
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
    mse = Array(Float64,nofolds)
    corrcoeff = Array(Float64,nofolds)
    avmse = Array(Float64,nofolds)
    varmse = Array(Float64,nofolds)
    oobmse = Array(Float64,nofolds)
    esterr = Array(Float64,nofolds)
    absesterr = Array(Float64,nofolds)
    validity = Array(Float64,nofolds)
    region = Array(Float64,nofolds)
#            errcor = Array(Float64,nofolds)
#            errcor = Float64[]
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
        ## if conformal == :rfok
        ##     trainingdata = globaldata[globaldata[:FOLD] .!= fold,:]
        ##     labels = names(trainingdata)[[!(l in [:REGRESSION,:WEIGHT,:TEST,:FOLD,:ID]) for l in names(trainingdata)]]
        ##     rows = Array{Float64}[]
        ##     knnalphas = Float64[]
        ## end
        largestrange = maximum(correcttrainingvalues)-minimum(correcttrainingvalues)
        for i = 1:length(correcttrainingvalues)
            oobpredcount = oobpredictions[i][1]
            if oobpredcount > 0
                ooberror = abs(correcttrainingvalues[i]-(oobpredictions[i][2]/oobpredcount))
                push!(ooberrors,ooberror)
                if conformal == :normalized ## || conformal == :isotonic
                    delta = (oobpredictions[i][3]/oobpredcount)-(oobpredictions[i][2]/oobpredcount)^2
                    if 2*ooberror > largestrange
                        alpha = largestrange/(delta+0.01)
                    else
                        alpha = 2*ooberror/(delta+0.01)
                    end
                    push!(alphas,alpha)
                    push!(deltas,delta)
                end
                ## if conformal == :rfok
                ##     push!(rows,hcat(ooberror,convert(Array{Float64},trainingdata[i,labels])))
                ## end
                oobse += ooberror^2
                nooob += 1
            end
        end
        ## if conformal == :rfok
        ##     maxk = minimum([45,nooob-1])
        ##     deltasalphas = Array(Float64,nooob,maxk,2)
        ##     distancematrix = Array(Float64,nooob,nooob)
        ##     for r = 1:nooob-1
        ##         distancematrix[r,r] = 0.001
        ##         for k = r+1:nooob
        ##             distance = sqL2dist(rows[r][2:end],rows[k][2:end])+0.001 # Small term added to prevent infinite weights
        ##             distancematrix[r,k] = distance
        ##             distancematrix[k,r] = distance
        ##         end
        ##     end
        ##     distancematrix[nooob,nooob] = 0.001
        ##     for r = 1:nooob
        ##         distanceerrors = Array(Float64,nooob,2)
        ##         for n = 1:nooob
        ##             distanceerrors[n,1] = distancematrix[r,n]
        ##             distanceerrors[n,2] = rows[n][1]
        ##         end
        ##         distanceerrors = sortrows(distanceerrors,by=x->x[1])
        ##         knndelta = 0.0
        ##         distancesum = 0.0
        ##         for k = 2:maxk+1
        ##             knndelta += distanceerrors[k,2]/distanceerrors[k,1]
        ##             distancesum += 1/distanceerrors[k,1]
        ##             deltasalphas[r,k-1,1] = knndelta/distancesum
        ##             if 2*rows[r][1] > largestrange
        ##                 deltasalphas[r,k-1,2] = largestrange/(deltasalphas[r,k-1,1]+0.01)
        ##             else
        ##                 deltasalphas[r,k-1,2] = 2*rows[r][1]/(deltasalphas[r,k-1,1]+0.01)
        ##             end
        ##         end
        ##     end
        ##     thresholdindex = Int(floor((nooob+1)*(1-method.confidence)))
        ##     if thresholdindex >= 1
        ##         lowestrangesum = Inf
        ##         bestk = Inf
        ##         bestalpha = Inf
        ##         for k = 1:maxk
        ##             knnalpha = sort(deltasalphas[:,k,2],rev=true)[thresholdindex]
        ##             knnrangesum = 0.0
        ##             for j = 1:nooob
        ##                 knnrangesum += knnalpha*(deltasalphas[j,k,1]+0.01)
        ##             end
        ##             if knnrangesum < lowestrangesum
        ##                 lowestrangesum = knnrangesum
        ##                 bestk = k
        ##                 bestalpha = knnalpha
        ##             end
        ##         end
        ##     else
        ##         bestalpha = Inf
        ##         bestk = 1
        ##     end
        ## end
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
        ## elseif conformal == :rfok
        ##     alpha = bestalpha
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
        end
        msesum = 0.0
        noinregion = 0.0
        rangesum = 0.0
#                testdeltas = Array(Float64,length(correctvalues))
#                testerrors = Array(Float64,length(correctvalues))
        for i = 1:length(correctvalues)
            error = abs(correctvalues[i]-predictions[testexamplecounter+i])
#                    testerrors[i] = error
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
                if isnan(errorrange) || errorrange > largestrange
                    errorrange = largestrange
                end
            elseif conformal == :normalized
                delta = (squaredpredictions[testexamplecounter+i]-predictions[testexamplecounter+i]^2)
#                        testdeltas[i] = delta
                errorrange = alpha*(delta+0.01)
                if isnan(errorrange) || errorrange > largestrange
                    errorrange = largestrange
                end
            ## elseif conformal == :rfok
            ##     testrow = convert(Array{Float64},testdata[i,labels])
            ##     distanceerrors = Array(Float64,nooob,2)
            ##     for n = 1:size(rows,1)
            ##         distanceerrors[n,1] = sqL2dist(testrow,rows[n][2:end])+0.001 # Small term added to prevent infinite weights
            ##         distanceerrors[n,2] = rows[n][1]
            ##     end
            ##     distanceerrors = sortrows(distanceerrors,by=x->x[1])
            ##     knndelta = 0.0
            ##     distancesum = 0.0
            ##     for j = 1:bestk
            ##         knndelta += distanceerrors[j,2]/distanceerrors[j,1]
            ##         distancesum += 1/distanceerrors[j,1]
            ##     end
            ##     knndelta /= distancesum
            ##     testdeltas[i] = knndelta
            ##     errorrange = alpha*(knndelta+0.01)
            ##     if isnan(errorrange) || errorrange > largestrange
            ##         errorrange = largestrange
            ##     end
            ## elseif conformal == :isotonic
            ##     delta = (squaredpredictions[testexamplecounter+i]-predictions[testexamplecounter+i]^2)
            ##     foundthreshold = false
            ##     thresholdcounter = size(isotonicthresholds,1)
            ##     while ~foundthreshold && thresholdcounter > 0
            ##         if delta >= isotonicthresholds[thresholdcounter][1]
            ##             errorrange = isotonicthresholds[thresholdcounter][2]*2
            ##             foundthreshold = true
            ##         else
            ##             thresholdcounter -= 1
            ##         end
            ##     end
            end
            
            if conformal != :std
                curIndeces = find(foldIndeces)
                curIndex = curIndeces[i]
                prediction_results[curIndex] = (predictions[testexamplecounter+i],[predictions[testexamplecounter+i]-errorrange/2,predictions[testexamplecounter+i]+errorrange/2])
            end
            
            rangesum += errorrange
            if error <= errorrange/2
                noinregion += 1
#                        testerrors[i] = 0
#                    else
#                        testerrors[i] = 1
            end
        end
#                if sum(testerrors) > 0 && sum(testerrors) < length(testerrors)
#                    println("****************************************")
#                    println("ranges: $testdeltas")
#                    println("errors: $testerrors")
#                    push!(errcor,cor(testdeltas,testerrors))
#                    println("corr: $(cor(testdeltas,testerrors))")
#                end
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
    return RegressionResult(mean(mse),mean(corrcoeff),mean(avmse),mean(varmse),mean(esterr),mean(absesterr),mean(validity),mean(region),mean(modelsizes),mean(noirregularleafs),
                                            time+extratime), prediction_results
end


##
## Functions to be executed on each worker
##
function generate_and_test_trees(Arguments::Tuple{LearningMethod{Regressor},Symbol,Int64,Int64,Array{Int64,1}})
    method,experimentype,notrees,randseed,randomoobs = Arguments
    s = size(globaldata,1)
    srand(randseed)
    variables, types = get_variables_and_types(globaldata)
    if experimentype == :test
        model,oobpredictions,variableimportance, modelsize, noirregularleafs, randomclassoobs, oob = generate_trees((method,Int64[],notrees,randseed);curdata=globaldata[globaldata[:TEST] .== false,:], randomoobs=randomoobs, varimparg = false)
        testdata = globaldata[globaldata[:TEST] .== true,:]
        testmissingvalues, testnonmissingvalues = find_missing_values(method,variables,testdata)
        newtestdata = transform_nonmissing_columns_to_arrays(method,variables,testdata,testmissingvalues)
        replacements_for_missing_values!(method,newtestdata,testdata,variables,types,testmissingvalues,testnonmissingvalues)
        correctvalues = getDfArrayData(testdata[:REGRESSION])
        nopredictions = size(testdata,1)
        predictions = Array(Array{Float64,1},nopredictions)
        squaredpredictions = Array(Any,nopredictions)
        totalnotrees,squarederror = make_prediction_analysis(method, model, newtestdata, randomclassoobs, oob, predictions, squaredpredictions, correctvalues)
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
            model,oobpredictions[foldno],variableimportance, modelsizes[foldno], noirregularleafs[foldno], randomclassoobs, oob = generate_trees((method,Int64[],notrees,randseed);curdata=trainingdata, randomoobs=size(randomoobs,1) > 0 ? randomoobs[foldno] : [], varimparg = false)
            testmissingvalues, testnonmissingvalues = find_missing_values(method,variables,testdata)
            newtestdata = transform_nonmissing_columns_to_arrays(method,variables,testdata,testmissingvalues)
            replacements_for_missing_values!(method,newtestdata,testdata,variables,types,testmissingvalues,testnonmissingvalues)
            correctvalues = getDfArrayData(testdata[:REGRESSION])
            totalnotrees,squarederror = make_prediction_analysis(method, model, newtestdata, randomclassoobs, oob, predictions, squaredpredictions, correctvalues; predictionexamplecounter=testexamplecounter)
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

    # starting from here till the end of the function is duplicated between here and the Classifier and Survival dispatchers
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
        # variableimportance += treevariableimportance
        noirregularleafs += treenoirregularleafs
        if (varimparg)
            variableimportance += treevariableimportance
        end
    end
   return (model,oobpredictions,variableimportance,modelsize,noirregularleafs,randomclassoobs,oob)
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
    # if varimp
    #     return model, variableimportance, noleafs, noirregularleafs, zeroweights
    # else
    #     return model, noleafs, noirregularleafs, zeroweights
    # end
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
        else #Catagorical
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

function make_leaf(node,method::LearningMethod{Regressor}, parenttrainingweights)
    sumweights = sum(node.trainingweights)
    sumregressionvalues = sum(node.regressionvalues)
    return [sumweights,sumregressionvalues]
    ## if sumweights > 0
    ##     prediction = sum(trainingweights .* regressionvalues)/sumweights
    ## else
    ##     prediction = defaultprediction
    ## end
    # return prediction
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

function apply_model_internal(model::PredictionModel{Regressor}; confidence = :std)
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
        results = Array(Tuple,size(predictions,1))
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
