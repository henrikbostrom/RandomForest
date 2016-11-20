#= AMG
build model and test and get results
=#
function run_split_internal(method::LearningMethod{Regressor}, results, time)
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
    return RegressionResult(mse,corrcoeff,avmse,varmse,esterr,absesterr,validity,region,modelsize,noirregularleafs,time+extratime)
end


function run_cross_validation_internal(method::LearningMethod{Regressor}, results, modelsizes, nofolds, conformal, time)
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
        testdata = globaldata[globaldata[:FOLD] .== fold,:]
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
                                            time+extratime)
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
        correctvalues = testdata[:REGRESSION]
        nopredictions = size(testdata,1)
        predictions = Array(Array{Float64,1},nopredictions)
        squaredpredictions = Array(Any,nopredictions)
        totalnotrees,squarederror = make_prediction_analysis(method, model, newtestdata, randomclassoobs, oob, predictions, squaredpredictions, correctvalues)
        return (modelsize,predictions,squarederrors,oobpredictions,squaredpredictions,noirregularleafs)
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
            correctvalues = testdata[:REGRESSION]
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
