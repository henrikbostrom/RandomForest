## RandomForest.jl
## v. 0.0.9
##
## Random forests for classification and regression with conformal prediction
##
## Developed for Julia 0.4 (http://julialang.org/)
##
## Copyright Henrik Boström 2016
## Email: henrik.bostrom@dsv.su.se
##
## TODO for version 1.0:
##
## *** MUST ***
##
## - check definition of OneC
## - output should either be presented as text or as a dataframe (or possibly more than one)
## - basic information about each dataset should be displayed in result table, e.g. no classes
##
## *** SHOULD ***
##
## - allow stored models to be used with different confidence levels (requires storing all alphas)
## - allow for using modpred in stored models (requires storing info. from all oob predictions)
## - leave-one-out cross validation
## - handling of sparse data
## - make everything work for single trees, including nonconformity measures, etc.
## - warnings/errors should be reported for incorrect format of datasets
## - warnings/errors should be reported for incorrect parameter settings
##
## *** COULD ***
##
## - variable importance alternatively calculated using permutations
## - consider alternative ways of distributing tasks, e.g., w/o copying dataset
## - employ weight vectors (from StatsBase)
## - handle uncertainty
## - functions to "foldify", i.e., add "FOLD" or "TEST" columns to exported files
## - statistical tests (Friedman)
##
## *** WONT ***
##
## - allow for original weights that have to be taken into account when performing bagging, etc.
## - visualize single tree

__precompile__()

module RandomForest

using DataFrames

export
    experiment,
    tree,
    forest,
    doc,
    read_data,
    load_data,
    describe_data,
    evaluate_method,
    evaluate_methods,
    generate_model,
    store_model,
    load_model,
    describe_model,
    apply_model,
    runexp

global majorversion = 0
global minorversion = 0
global patchversion = 9

"""`runexp` is used to test the performance of the library on a number of test sets"""
function runexp()
    experiment(files = ["uci/glass.txt"]) # Warmup
    experiment(files="uci",methods=[forest(),forest(notrees=500)],resultfile="uci-results.txt")
    experiment(files = ["regression/cooling.txt"]) # Warmup
    experiment(files="regression",methods=[forest(),forest(notrees=500)],resultfile="regression-results.txt")
end

include("types.jl")

##
## Functions for running experiments
##

function experiment(;files = ".", separator = ',', protocol = 10, normalizetarget = false, normalizeinput = false, methods = [forest()], resultfile = :none)
    println("RandomForest v. $(majorversion).$(minorversion).$(patchversion)")
    if typeof(files) == ASCIIString
        if isdir(files)
            dirfiles = readdir(files)
            datafiles = dirfiles[[splitext(file)[2] in [".txt",".csv"] for file in dirfiles]]
            filenames = [string(files,"/",filename) for filename in datafiles]
        else
            throw("Not a directory: $files")
        end
    else
        filenames = files
    end
    totaltime = @elapsed results = [run_experiment(file,separator,protocol,normalizetarget,normalizeinput,methods) for file in filenames]
    classificationresults = [pt == :CLASS for (pt,f,r) in results]
    regressionresults = [pt == :REGRESSION for (pt,f,r) in results]
    present_results(sort(results[classificationresults]),methods)
    present_results(sort(results[regressionresults]),methods)
    println("Total time: $(round(totaltime,2)) s.")
    if resultfile != :none
        origstdout = STDOUT
        resultfilestream = open(resultfile,"w+")
        redirect_stdout(resultfilestream)
        present_results(sort(results[classificationresults]),methods)
        present_results(sort(results[regressionresults]),methods)
        println("Total time: $(round(totaltime,2)) s.")
        redirect_stdout(origstdout)
        close(resultfilestream)
    end
end

function run_experiment(file, separator, protocol, normalizetarget, normalizeinput, methods)
    global globaldata = read_data(file, separator=separator) # Made global to allow access from workers
    predictiontask = prediction_task(globaldata)
    if predictiontask == :NONE
        warn("File excluded: $file - no column is labeled CLASS or REGRESSION\n\tThis may be due to incorrectly specified separator, e.g., use: separator = \'\\t\'")
        result = (:NONE,:NONE,:NONE)
    else
        if predictiontask == :REGRESSION && normalizetarget
            regressionvalues = globaldata[:REGRESSION]
            minval = minimum(regressionvalues)
            maxval = maximum(regressionvalues)
            normalizedregressionvalues = [(v-minval)/(maxval-minval) for v in regressionvalues]
            globaldata[:REGRESSION] = convert(Array{Float64},normalizedregressionvalues)
        end
        if normalizeinput # NOTE: currently assumes that all input is numeric and that there are no missing values
            for label in names(globaldata)
                if ~(label in [:REGRESSION,:CLASS,:ID,:WEIGHT,:TEST,:FOLD])
                    min = minimum(globaldata[label])
                    max = maximum(globaldata[label])
                    if min < max
                        globaldata[label] = convert(Array{Float64},[(x-min)/(max-min) for x in globaldata[label]])
                    else
                        globaldata[label] = convert(Array{Float64},[0.5 for x in globaldata[label]])
                    end
                end
            end
        end
        initiate_workers()
        if typeof(protocol) == Float64 || protocol == :test
            results = run_split(protocol,predictiontask,methods)
            result = (predictiontask,file,results)
        elseif typeof(protocol) == Int64 || protocol == :cv
            results = run_cross_validation(protocol,predictiontask,methods)
            result = (predictiontask,file,results)
        else
            throw("Unknown experiment protocol")
        end
        println("Completed experiment with: $file")
    end
    return result
end

function read_data(file; separator = ',')
    df = readtable(file,separator = separator)
    if ~(:WEIGHT in names(df))
        df = hcat(df,DataFrame(WEIGHT = ones(size(df,1))))
    end
    return df
end

function run_split(testoption,predictiontask,methods)
    if typeof(testoption) == Float64
        noexamples = size(globaldata,1)
        notestexamples = convert(Int,floor(testoption*noexamples))
        notrainingexamples = noexamples-notestexamples
        tests = shuffle([trues(notestexamples);falses(notrainingexamples)])
        if ~(:TEST in names(globaldata))
            global globaltests = DataFrame(TEST = tests)
            global globaldata = hcat(globaltests,globaldata)
        else
            globaldata[:TEST] = tests
            global globaldata = globaldata
        end
    elseif testoption == :test
        if ~(:TEST in names(globaldata))
            throw("Missing TEST column in dataset")
        elseif typeof(globaldata[:TEST]) != DataArrays.DataArray{Bool,1}
            throw("TEST column contains non-Boolean values")
        end
    end
    update_workers()
    nocoworkers = nprocs()-1
    methodresults = Array(Any,length(methods))
    origseed = rand(1:1000_000_000)
    for m = 1:length(methods)
        srand(origseed)
        if nocoworkers > 0
            notrees = [div(methods[m].notrees,nocoworkers) for i=1:nocoworkers]
            for i = 1:mod(methods[m].notrees,nocoworkers)
                notrees[i] += 1
            end
        else
            notrees = [methods[m].notrees]
        end
        if methods[m].modpred
            randomoobs = Array(Int64,notestexamples)
            for i = 1:notestexamples
                randomoobs[i] = rand(1:notrainingexamples)
            end
        else
            randomoobs = Array(Int64,0)
        end
        time = @elapsed results = pmap(generate_and_test_trees,[(methods[m],predictiontask,:test,n,rand(1:1000_000_000),randomoobs) for n in notrees])
        tic()
        methodresults[m] = run_split_internal(methods[m], results)
    end
    return methodresults
end

function run_split_internal(method::LearningMethod{Regressor}, results)
    modelsize = sum([result[1] for result in results])
    noirregularleafs = sum([result[6] for result in results])
    predictions = results[1][2]
    for r = 2:length(results)
        predictions += results[r][2]
    end
    nopredictions = size(predictions,1)

    predictions = [predictions[i][2]/predictions[i][1] for i = 1:nopredictions]
    if methods[m].conformal == :default
        conformal = :normalized
    else
        conformal = methods[m].conformal
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
    ##     thresholdindex = Int(floor((nooob+1)*(1-methods[m].confidence)))
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
    thresholdindex = Int(floor((nooob+1)*(1-methods[m].confidence)))
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
    ##     mincalibrationsize = maximum([10,Int(floor(1/(round((1-methods[m].confidence)*1000000)/1000000))-1)])
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
    ##     thresholdindex = maximum([1,Int(floor(size(isotonicgroup,1)*(1-methods[m].confidence)))])
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
    ##                 thresholdindex = maximum([1,Int(floor(size(newisotonicgroup,1)*(1-methods[m].confidence)))])
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
        if conformal == :normalized && methods[m].modpred
            randomoob = randomoobs[i]
            oobpredcount = oobpredictions[randomoob][1]
            if oobpredcount > 0.0
                thresholdindex = Int(floor(nooob*(1-methods[m].confidence)))
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

function run_split_internal(method::LearningMethod{Classifier}, results)
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
    if methods[m].conformal == :default
        conformal = :std
    else
        conformal = methods[m].conformal
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
        thresholdindex = Int(floor((nooob+1)*(1-methods[m].confidence)))
        if thresholdindex >= 1
            alpha = sort(alphas)[thresholdindex]
        else
            alpha = -Inf
        end
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
            thresholdindex = Int(floor((noclassoob+1)*(1-methods[m].confidence)))
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
                if methods[m].modpred
                    if conformal == :std
                        randomoob = randomoobs[i]
                        thresholdindex = Int(floor(nooob*(1-methods[m].confidence)))
                        if thresholdindex >= 1
                            alpha = sort(alphas[[1:randomoob-1;randomoob+1:end]])[thresholdindex] # NOTE: assumes oobpredcount > 0 always is true!
                        else
                            alpha = -Inf
                        end
                    else # conformal == :classcond
                        randomoobclass, randomoobref = randomclassoobs[i]
                        thresholdindex = Int(floor(size(classalphas[randomoobclass],1)*(1-methods[m].confidence)))
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
                    if methods[m].modpred
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
    return ClassificationResult(accuracy,weightedauc,brierscore,avacc,esterr,absesterr,avbrier,varbrier,margin,prob,validity,avc,onec,modelsize,noirregularleafs,time+extratime)

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

function initiate_workers()
    pr = Array(Any,nprocs())
    for i = 2:nprocs()
        pr[i] = remotecall(i,RandomForest.load_global_dataset)
    end
    for i = 2:nprocs()
        wait(pr[i])
    end
end

function load_global_dataset()
    global globaldata = @fetchfrom(1,RandomForest.globaldata)
end

function update_workers()
    pr = Array(Any,nprocs())
    for i = 2:nprocs()
        pr[i] = remotecall(i,RandomForest.update_global_dataset)
    end
    for i = 2:nprocs()
        wait(pr[i])
    end
end

function update_global_dataset()
    global globaltests = @fetchfrom(1,RandomForest.globaltests)
    global globaldata = hcat(globaltests,globaldata)
end

function run_cross_validation(protocol,predictiontask,methods)
    if typeof(protocol) == Int64
        nofolds = protocol
        folds = collect(1:nofolds)
        foldsizes = Array(Int64,nofolds)
        noexamples = size(globaldata,1)
        if nofolds > noexamples
            nofolds = noexamples
        end
        basesize = div(noexamples,nofolds)
        remainder = mod(noexamples,nofolds)
        foldnos = Array(Int,noexamples)
        counter = 0
        for foldno = 1:nofolds
            foldsize = basesize
            if remainder > 0
                foldsize += 1
                remainder -= 1
            end
            foldsizes[foldno] = foldsize
            foldnos[counter+1:counter+foldsize] = foldno
            counter += foldsize
        end
        shuffle!(foldnos)
        if ~(:FOLD in names(globaldata))
            global globaltests = DataFrame(FOLD = foldnos)
            global globaldata = hcat(globaltests,globaldata)
        else
            globaldata[:FOLD] = foldnos
            global globaldata = globaldata
        end
    else
        if ~(:FOLD in names(globaldata))
            throw("Missing FOLD column in dataset")
        else
            folds = sort(unique(globaldata[:FOLD]))
            nofolds = length(folds)
        end
    end
    update_workers()
    nocoworkers = nprocs()-1
    methodresults = Array(Any,length(methods))
    origseed = rand(1:1000_000_000)
    for m = 1:length(methods)
        srand(origseed)
        if methods[m].conformal == :default
            conformal = :normalized
        else
            conformal = methods[m].conformal
        end
        if nocoworkers > 0
            notrees = [div(methods[m].notrees,nocoworkers) for i=1:nocoworkers]
            for i = 1:mod(methods[m].notrees,nocoworkers)
                notrees[i] += 1
            end
        else
            notrees = [methods[m].notrees]
        end
        if methods[m].modpred
            randomoobs = Array(Any,nofolds)
            for i = 1:nofolds
                randomoobs[i] = Array(Int64,foldsizes[i])
                for j = 1:foldsizes[i]
                    randomoobs[i][j] = rand(1:(noexamples-foldsizes[i]))
                end
            end
        else
            randomoobs = Array(Int64,0)
        end
        time = @elapsed results = pmap(generate_and_test_trees,[(methods[m],predictiontask,:cv,n,rand(1:1000_000_000),randomoobs) for n in notrees])
        tic()
        allmodelsizes = try
            [result[1] for result in results]
        catch
            origstdout = STDOUT
            dumpfilestream = open("dump.txt","w+")
            redirect_stdout(dumpfilestream)
            println("***** ERROR *****")
            println("results:\n $results")
            redirect_stdout(origstdout)
            close(dumpfilestream)
            error("Something went wrong - output written to dump.txt")
        end
        modelsizes = allmodelsizes[1]
        for r = 2:length(allmodelsizes)
            modelsizes += allmodelsizes[r]
        end
        methodresults[m] = run_cross_validation_internal(methods[m], results, modelsizes)
    end
    return methodresults
end

function run_cross_validation_internal(method::LearningMethod{Regressor}, results, modelsizes)
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
        ##     thresholdindex = Int(floor((nooob+1)*(1-methods[m].confidence)))
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
        thresholdindex = Int(floor((nooob+1)*(1-methods[m].confidence)))
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
        ##     mincalibrationsize = maximum([10;Int(floor(1/(round((1-methods[m].confidence)*1000000)/1000000))-1)])
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
        ##     thresholdindex = maximum([1,Int(floor(size(isotonicgroup,1)*(1-methods[m].confidence)))])
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
        ##                 thresholdindex = maximum([1,Int(floor(size(newisotonicgroup,1)*(1-methods[m].confidence)))])
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
            if conformal == :normalized && methods[m].modpred
                randomoob = randomoobs[foldno][i]
                oobpredcount = oobpredictions[randomoob][1]
                if oobpredcount > 0.0
                    thresholdindex = Int(floor(nooob*(1-methods[m].confidence)))
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

function run_cross_validation_internal(method::LearningMethod{Classifier}, results, modelsizes)
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
    if methods[m].conformal == :default
        conformal = :std
    else
        conformal = methods[m].conformal
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
            thresholdindex = Int(floor((nooob+1)*(1-methods[m].confidence)))
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
                thresholdindex = Int(floor((noclassoob+1)*(1-methods[m].confidence)))
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
                                                mean(validity),mean(avc),mean(onec),mean(modelsizes),mean(noirregularleafs),time+extratime)
end

"""
Infers the prediction task from the data
"""
function prediction_task(method::LearningMethod{Regressor})
    return :REGRESSION
end

function prediction_task(method::LearningMethod{Classifier})
    return :CLASS
end

function prediction_task(method) # AMGAD: why there is none prediction task ?
    return :NONE
end

##
## Functions to be executed on each worker
##

function generate_and_test_trees(method::LearningMethod{Regressor},predictiontask,experimentype,notrees,randseed,randomoobs)
    s = size(globaldata,1)
    srand(randseed)
    if experimentype == :test
        variables, types = get_variables_and_types(globaldata)
        modelsize = 0
        noirregularleafs = 0
        testdata = globaldata[globaldata[:TEST] .== true,:]
        trainingdata = globaldata[globaldata[:TEST] .== false,:]
        trainingrefs = collect(1:size(trainingdata,1))
        trainingweights = trainingdata[:WEIGHT]
        regressionvalues = trainingdata[:REGRESSION]
        oobpredictions = Array(Any,size(trainingdata,1))
        for i = 1:size(trainingdata,1)
            oobpredictions[i] = [0,0,0]
        end
        missingvalues, nonmissingvalues = find_missing_values(predictiontask,variables,trainingdata)
        newtrainingdata = transform_nonmissing_columns_to_arrays(predictiontask,variables,trainingdata,missingvalues)
        testmissingvalues, testnonmissingvalues = find_missing_values(predictiontask,variables,testdata)
        newtestdata = transform_nonmissing_columns_to_arrays(predictiontask,variables,testdata,testmissingvalues)
        model = Array(Any,notrees)
        oob = Array(Any,notrees)
        for treeno = 1:notrees
            sample_replacements_for_missing_values!(newtrainingdata,trainingdata,predictiontask,variables,types,missingvalues,nonmissingvalues)
            model[treeno], noleafs, treenoirregularleafs, oob[treeno] = generate_tree(method,trainingrefs,trainingweights,regressionvalues,newtrainingdata,variables,types,predictiontask,oobpredictions)
            modelsize += noleafs
            noirregularleafs += treenoirregularleafs
        end
        nopredictions = size(testdata,1)
        predictions = Array(Any,nopredictions)
        squaredpredictions = Array(Any,nopredictions)
        squarederror = 0.0
        totalnotrees = 0
        for i = 1:nopredictions
            correctvalue = testdata[i,:REGRESSION]
            prediction = 0.0
            squaredprediction = 0.0
            nosampledtrees = 0
            randomoob = randomoobs[i]
            for t = 1:length(model)
                if method.modpred
                    if oob[t][randomoob]
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
            totalnotrees += nosampledtrees
            predictions[i] = [nosampledtrees;prediction]
            squaredpredictions[i] = [nosampledtrees;squaredprediction]
        end
        squarederrors = [totalnotrees;squarederror]
        return (modelsize,predictions,squarederrors,oobpredictions,squaredpredictions,noirregularleafs)
    else # experimentype == :cv
        folds = sort(unique(globaldata[:FOLD]))
        nofolds = length(folds)
        squarederrors = Array(Any,nofolds)
        predictions = Array(Any,size(globaldata,1))
        squaredpredictions = Array(Any,size(globaldata,1))
        variables, types = get_variables_and_types(globaldata)
        oobpredictions = Array(Any,nofolds)
        modelsizes = Array(Int64,nofolds)
        noirregularleafs = Array(Int64,nofolds)
        testexamplecounter = 0
        foldno = 0
        for fold in folds
            foldno += 1
            trainingdata = globaldata[globaldata[:FOLD] .!= fold,:]
            testdata = globaldata[globaldata[:FOLD] .== fold,:]
            trainingrefs = collect(1:size(trainingdata,1))
            trainingweights = trainingdata[:WEIGHT]
            regressionvalues = trainingdata[:REGRESSION]
            oobpredictions[foldno] = Array(Any,size(trainingdata,1))
            for i = 1:size(trainingdata,1)
                oobpredictions[foldno][i] = [0,0,0]
            end
            missingvalues, nonmissingvalues = find_missing_values(predictiontask,variables,trainingdata)
            newtrainingdata = transform_nonmissing_columns_to_arrays(predictiontask,variables,trainingdata,missingvalues)
            testmissingvalues, testnonmissingvalues = find_missing_values(predictiontask,variables,testdata)
            newtestdata = transform_nonmissing_columns_to_arrays(predictiontask,variables,testdata,testmissingvalues)
            model = Array(Any,notrees)
            modelsize = 0
            totalnoirregularleafs = 0
            oob = Array(Any,notrees)
            for treeno = 1:notrees
                sample_replacements_for_missing_values!(newtrainingdata,trainingdata,predictiontask,variables,types,missingvalues,nonmissingvalues)
                model[treeno], noleafs, treenoirregularleafs, oob[treeno] = generate_tree(method,trainingrefs,trainingweights,regressionvalues,newtrainingdata,variables,types,predictiontask,oobpredictions[foldno])
                modelsize += noleafs
                totalnoirregularleafs += treenoirregularleafs
            end
            modelsizes[foldno] = modelsize
            noirregularleafs[foldno] = totalnoirregularleafs
                squarederror = 0.0
            totalnotrees = 0
            for i = 1:size(testdata,1)
                correctvalue = testdata[i,:REGRESSION]
                prediction = 0.0
                nosampledtrees = 0
                squaredprediction = 0.0
                if method.modpred
                    randomoob = randomoobs[foldno][i]
                end
                for t = 1:length(model)
                    if method.modpred
                        if oob[t][randomoob]
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
                totalnotrees += nosampledtrees
                testexamplecounter += 1
                predictions[testexamplecounter] = [nosampledtrees;prediction]
                squaredpredictions[testexamplecounter] = [nosampledtrees;squaredprediction]
            end
            squarederrors[foldno] = [totalnotrees;squarederror]
        end
        return (modelsizes,predictions,squarederrors,oobpredictions,squaredpredictions,noirregularleafs)
    end
end

function generate_and_test_trees(method::LearningMethod{Classifier},predictiontask,experimentype,notrees,randseed,randomoobs)
    s = size(globaldata,1)
    srand(randseed)
    if experimentype == :test
        classes = unique(globaldata[:CLASS])
        noclasses = length(classes)
        classdata = Array(Any,noclasses)
        for c = 1:noclasses
            classdata[c] = globaldata[globaldata[:CLASS] .== classes[c],:]
        end
        variables, types = get_variables_and_types(globaldata)
        modelsize = 0
        noirregularleafs = 0
        trainingdata = Array(Any,noclasses)
        trainingrefs = Array(Any,noclasses)
        trainingweights = Array(Any,noclasses)
        testdata = Array(Any,noclasses)
        oobpredictions = Array(Any,noclasses)
        emptyprediction = zeros(noclasses)
        for c = 1:noclasses
            testdata[c] = classdata[c][classdata[c][:TEST] .== true,:]
            trainingdata[c] = classdata[c][classdata[c][:TEST] .== false,:]
            trainingrefs[c] = collect(1:size(trainingdata[c],1))
            trainingweights[c] = trainingdata[c][:WEIGHT]
            oobpredictions[c] = Array(Any,size(trainingdata[c],1))
            for i = 1:size(trainingdata[c],1)
                oobpredictions[c][i] = [0;emptyprediction]
            end
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
        regressionvalues = []
        missingvalues, nonmissingvalues = find_missing_values(predictiontask,variables,trainingdata)
        newtrainingdata = transform_nonmissing_columns_to_arrays(predictiontask,variables,trainingdata,missingvalues)
        testmissingvalues, testnonmissingvalues = find_missing_values(predictiontask,variables,testdata)
        newtestdata = transform_nonmissing_columns_to_arrays(predictiontask,variables,testdata,testmissingvalues)
        model = Array(Any,notrees)
        oob = Array(Any,notrees)
        for treeno = 1:notrees
            sample_replacements_for_missing_values!(newtrainingdata,trainingdata,predictiontask,variables,types,missingvalues,nonmissingvalues)
            model[treeno], noleafs, treenoirregularleafs, oob[treeno] = generate_tree(method,trainingrefs,trainingweights,regressionvalues,newtrainingdata,variables,types,predictiontask,oobpredictions)
            modelsize += noleafs
            noirregularleafs += treenoirregularleafs
        end
        nopredictions = sum([size(testdata[c],1) for c = 1:noclasses])
        testexamplecounter = 0
        predictions = Array(Any,nopredictions)
        correctclassificationcounter = 0.0
        totalnotrees = 0
        squaredproberror = 0.0
        for c = 1:noclasses
            correctclassvector = zeros(noclasses)
            correctclassvector[c] = 1.0
            for i = 1:size(testdata[c],1)
                classprobabilities = zeros(noclasses)
                nosampledtrees = 0
                testexamplecounter += 1
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
                totalnotrees += nosampledtrees
                predictions[testexamplecounter] = [nosampledtrees;classprobabilities]
            end
        end
        return (modelsize,predictions,([totalnotrees;correctclassificationcounter],[totalnotrees;squaredproberror]),oobpredictions,noirregularleafs)
    else # experimentype == :cv
        folds = sort(unique(globaldata[:FOLD]))
        nofolds = length(folds)
        classes = unique(globaldata[:CLASS])
        noclasses = length(classes)
        classdata = Array(Any,noclasses)
        for c = 1:noclasses
            classdata[c] = globaldata[globaldata[:CLASS] .== classes[c],:]
        end
        nocorrectclassifications = Array(Any,nofolds)
        squaredproberrors = Array(Any,nofolds)
        predictions = Array(Any,size(globaldata,1))
        variables, types = get_variables_and_types(globaldata)
        oobpredictions = Array(Any,nofolds)
        modelsizes = Array(Int64,nofolds)
        noirregularleafs = Array(Int64,nofolds)
        testexamplecounter = 0
        foldno = 0
        for fold in folds
            foldno += 1
            trainingdata = Array(Any,noclasses)
            trainingrefs = Array(Any,noclasses)
            trainingweights = Array(Any,noclasses)
            testdata = Array(Any,noclasses)
            oobpredictions[foldno] = Array(Any,noclasses)
            emptyprediction = zeros(noclasses)
            for c = 1:noclasses
                trainingdata[c] = classdata[c][classdata[c][:FOLD] .!= fold,:]
                testdata[c] = classdata[c][classdata[c][:FOLD] .== fold,:]
                trainingrefs[c] = collect(1:size(trainingdata[c],1))
                trainingweights[c] = trainingdata[c][:WEIGHT]
                oobpredictions[foldno][c] = Array(Any,size(trainingdata[c],1))
                for i = 1:size(trainingdata[c],1)
                    oobpredictions[foldno][c][i] = [0;emptyprediction]
                end
            end
            if size(randomoobs,1) > 0
                randomclassoobs = Array(Any,size(randomoobs[foldno],1))
                for i = 1:size(randomclassoobs,1)
                    oobref = randomoobs[foldno][i]
                    c = 1
                    while oobref > size(trainingrefs[c],1)
                        oobref -= size(trainingrefs[c],1)
                        c += 1
                    end
                    randomclassoobs[i] = (c,oobref)
                end
            end
            regressionvalues = []
            missingvalues, nonmissingvalues = find_missing_values(predictiontask,variables,trainingdata)
            newtrainingdata = transform_nonmissing_columns_to_arrays(predictiontask,variables,trainingdata,missingvalues)
            testmissingvalues, testnonmissingvalues = find_missing_values(predictiontask,variables,testdata)
            newtestdata = transform_nonmissing_columns_to_arrays(predictiontask,variables,testdata,testmissingvalues)
            model = Array(Any,notrees)
            modelsize = 0
            totalnoirregularleafs = 0
            oob = Array(Any,notrees)
            for treeno = 1:notrees
                sample_replacements_for_missing_values!(newtrainingdata,trainingdata,predictiontask,variables,types,missingvalues,nonmissingvalues)
                model[treeno], noleafs, treenoirregularleafs, oob[treeno] = generate_tree(method,trainingrefs,trainingweights,regressionvalues,newtrainingdata,variables,types,predictiontask,oobpredictions[foldno])
                modelsize += noleafs
                totalnoirregularleafs += treenoirregularleafs
            end
            modelsizes[foldno] = modelsize
            noirregularleafs[foldno] = totalnoirregularleafs
            correctclassificationcounter = 0
            squaredproberror = 0.0
            totalnotrees = 0
            foldtestexamplecounter = 0
            for c = 1:noclasses
                correctclassvector = zeros(noclasses)
                correctclassvector[c] = 1.0
                for i = 1:size(testdata[c],1)
                    foldtestexamplecounter += 1
                    classprobabilities = zeros(noclasses)
                    nosampledtrees = 0
                    if method.modpred
                        randomoobclass, randomoobref = randomclassoobs[foldtestexamplecounter]
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
                    testexamplecounter += 1
                    totalnotrees += nosampledtrees
                    predictions[testexamplecounter] = [nosampledtrees;classprobabilities]
                end
            end
            nocorrectclassifications[foldno] = [totalnotrees;correctclassificationcounter]
            squaredproberrors[foldno] = [totalnotrees;squaredproberror]
        end
        return (modelsizes,predictions,(nocorrectclassifications,squaredproberrors),oobpredictions,noirregularleafs)
    end
end

function generate_trees(method::LearningMethod{Regressor},predictiontask,classes,notrees,randseed)
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

    # AMGAD: starting from here till the end of the function is duplicated between here and the classifier dispatcher
    variables, types = get_variables_and_types(globaldata)
    modelsize = 0
    missingvalues, nonmissingvalues = find_missing_values(predictiontask,variables,trainingdata)
    newtrainingdata = transform_nonmissing_columns_to_arrays(predictiontask,variables,trainingdata,missingvalues)
    model = Array(Any,notrees)
    variableimportance = zeros(size(variables,1))
    for treeno = 1:notrees
        sample_replacements_for_missing_values!(newtrainingdata,trainingdata,predictiontask,variables,types,missingvalues,nonmissingvalues)
        model[treeno], treevariableimportance, noleafs, noirregularleafs = generate_tree(method,trainingrefs,trainingweights,regressionvalues,newtrainingdata,variables,types,predictiontask,oobpredictions,varimp = true)
        modelsize += noleafs
        variableimportance += treevariableimportance
    end
   return (model,oobpredictions,variableimportance)
end

function generate_trees(method::LearningMethod{Classifier},predictiontask,classes,notrees,randseed)
    s = size(globaldata,1)
    srand(randseed)
    noclasses = length(classes)
    trainingdata = Array(Any,noclasses)
    trainingrefs = Array(Any,noclasses)
    trainingweights = Array(Any,noclasses)
    oobpredictions = Array(Any,noclasses)
    emptyprediction = zeros(noclasses)
    for c = 1:noclasses
        trainingdata[c] = globaldata[globaldata[:CLASS] .== classes[c],:]
        trainingrefs[c] = collect(1:size(trainingdata[c],1))
        trainingweights[c] = trainingdata[c][:WEIGHT]
        oobpredictions[c] = Array(Any,size(trainingdata[c],1))
        for i = 1:size(trainingdata[c],1)
            oobpredictions[c][i] = [0;emptyprediction]
        end
    end
    regressionvalues = []

    # AMGAD: starting from here till the end of the function is duplicated between here and the regressor dispatcher
    variables, types = get_variables_and_types(globaldata)
    modelsize = 0
    missingvalues, nonmissingvalues = find_missing_values(predictiontask,variables,trainingdata)
    newtrainingdata = transform_nonmissing_columns_to_arrays(predictiontask,variables,trainingdata,missingvalues)
    model = Array(Any,notrees)
    variableimportance = zeros(size(variables,1))
    for treeno = 1:notrees
        sample_replacements_for_missing_values!(newtrainingdata,trainingdata,predictiontask,variables,types,missingvalues,nonmissingvalues)
        model[treeno], treevariableimportance, noleafs, noirregularleafs = generate_tree(method,trainingrefs,trainingweights,regressionvalues,newtrainingdata,variables,types,predictiontask,oobpredictions,varimp = true)
        modelsize += noleafs
        variableimportance += treevariableimportance
    end
    return (model,oobpredictions,variableimportance)
end

function find_missing_values(method::LearningMethod{Regressor},variables,trainingdata)
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

function find_missing_values(method::LearningMethod{Classifier},variables,trainingdata)
    noclasses = size(trainingdata,1)
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
            end
        end
    end
    return (missingvalues,nonmissingvalues)
end

function transform_nonmissing_columns_to_arrays(mathod::LearningMethod{Regressor},variables,trainingdata,missingvalues)
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

function transform_nonmissing_columns_to_arrays(mathod::LearningMethod{Classifier},variables,trainingdata,missingvalues)
    noclasses = size(trainingdata,1)
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

function sample_replacements_for_missing_values!(method::LearningMethod{Regressor},newtrainingdata,trainingdata,predictiontask,variables,types,missingvalues,nonmissingvalues)
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

function replacements_for_missing_values!(method::LearningMethod{Classifier},newtestdata,testdata,predictiontask,variables,types,missingvalues,nonmissingvalues)
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

function replacements_for_missing_values!(method::LearningMethod{Regressor},newtestdata,testdata,predictiontask,variables,types,missingvalues,nonmissingvalues)
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

function generate_tree(method::LearningMethod{Classifier},trainingrefs,trainingweights,regressionvalues,trainingdata,variables,types,predictiontask,oobpredictions; varimp = false)
    if method.bagging
        noclasses = length(trainingweights)
        classweights = map(Float64,[length(t) for t in trainingrefs])
        if typeof(method.bagsize) == Int
            samplesize = method.bagsize
        else
            samplesize = convert(Int,round(sum(classweights)*method.bagsize))
        end
        newtrainingweights = Array(Any,noclasses)
        newtrainingrefs = Array(Any,noclasses)
        for c = 1:noclasses
            newtrainingweights[c] = zeros(length(trainingweights[c]))
        end
        for i = 1:samplesize
            class = wsample(1:noclasses,classweights)
            newtrainingweights[class][rand(1:end)] += 1.0
        end
        zeroweights = Array(Any,noclasses)
        for c = 1:noclasses
            nonzeroweights = [newtrainingweights[c][i] > 0 for i=1:length(newtrainingweights[c])]
            zeroweights[c] = ~nonzeroweights
            newtrainingrefs[c] = trainingrefs[c][nonzeroweights]
            newtrainingweights[c] = newtrainingweights[c][nonzeroweights]
        end
        model, variableimportance, noleafs, noirregularleafs = build_tree(method,newtrainingrefs,newtrainingweights,regressionvalues,trainingdata,variables,types,predictiontask,varimp)
        for c = 1:noclasses
            oobrefs = trainingrefs[c][zeroweights[c]]
            for oobref in oobrefs
                oobpredictions[c][oobref] += [1;make_prediction(model,trainingdata[c],oobref,zeros(noclasses))]
            end
        end
    else
        model, variableimportance, noleafs, noirregularleafs = build_tree(method,trainingrefs,trainingweights,regressionvalues,trainingdata,variables,types,predictiontask,varimp)
    end
    if varimp
        return model, variableimportance, noleafs, noirregularleafs, zeroweights
    else
        return model, noleafs, noirregularleafs, zeroweights
    end
end

function generate_tree(method::LearningMethod{Regressor},trainingrefs,trainingweights,regressionvalues,trainingdata,variables,types,predictiontask,oobpredictions; varimp = false)
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
        model, variableimportance, noleafs, noirregularleafs = build_tree(method,newtrainingrefs,newtrainingweights,newregressionvalues,trainingdata,variables,types,predictiontask,varimp)
        zeroweights = ~nonzeroweights
        oobrefs = trainingrefs[zeroweights]
        for oobref in oobrefs
            leafstats = make_prediction(model,trainingdata,oobref,0)
            oobprediction = leafstats[2]/leafstats[1]
            oobpredictions[oobref] += [1,oobprediction,oobprediction^2]
        end
    else
        model, variableimportance, noleafs, noirregularleafs = build_tree(method,trainingrefs,trainingweights,regressionvalues,trainingdata,variables,types,predictiontask,varimp)
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

##
## Function for building a single tree
##

function build_tree(method,alltrainingrefs,alltrainingweights,allregressionvalues,trainingdata,variables,types,predictiontask,varimp)
    tree = Any[]
    depth = 0
    nodeno = 1
    noleafnodes = 0
    noirregularleafnodes = 0
    stack = [(depth,nodeno,alltrainingrefs,alltrainingweights,allregressionvalues,default_prediction(alltrainingweights,allregressionvalues,predictiontask,method))]
    nextavailablenodeno = 2
    if varimp
        variableimportance = zeros(length(variables))
    end
    while stack != []
        depth, nodenumber, trainingrefs, trainingweights, regressionvalues, defaultprediction = pop!(stack)
        if leaf_node(trainingweights,regressionvalues,predictiontask,depth,method)
            leaf = (:LEAF,make_leaf(trainingweights,regressionvalues,predictiontask,defaultprediction,method))
            push!(tree,(nodenumber,leaf))
            noleafnodes += 1
        else
            bestsplit = find_best_split(trainingrefs,trainingweights,regressionvalues,trainingdata,variables,types,predictiontask,method)
            if bestsplit == :NA
                leaf = (:LEAF,make_leaf(trainingweights,regressionvalues,predictiontask,defaultprediction,method))
                push!(tree,(nodenumber,leaf))
                noleafnodes += 1
                noirregularleafnodes += 1
            else
                leftrefs,leftweights,leftregressionvalues,rightrefs,rightweights,rightregressionvalues,leftweight =
                    make_split(trainingrefs,trainingweights,regressionvalues,trainingdata,predictiontask,bestsplit)
                varno, variable, splittype, splitpoint = bestsplit
                if varimp
                    if typeof(method.learningType) == Regressor #predictiontask == :REGRESSION
                        variableimp = variance_reduction(trainingweights,regressionvalues,leftweights,leftregressionvalues,rightweights,rightregressionvalues)
                    else
                        variableimp = information_gain(trainingweights,leftweights,rightweights)
                    end
                    variableimportance[varno] += variableimp
                end
                push!(tree,(nodenumber,((varno,splittype,splitpoint,leftweight),nextavailablenodeno,nextavailablenodeno+1)))
                defaultprediction = default_prediction(trainingweights,regressionvalues,predictiontask,method)
                push!(stack,(depth+1,nextavailablenodeno,leftrefs,leftweights,leftregressionvalues,defaultprediction))
                push!(stack,(depth+1,nextavailablenodeno+1,rightrefs,rightweights,rightregressionvalues,defaultprediction))
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
    if v in [:CLASS,:REGRESSION,:ID,:FOLD,:TEST,:WEIGHT]
        return false
    elseif startswith(string(v),"IGNORE")
        return false
    else
        return true
    end
end

function default_prediction(trainingweights,regressionvalues,predictiontask,method::LearningMethod{Classifier})
    noclasses = size(trainingweights,1)
    classcounts = [sum(trainingweights[i]) for i=1:noclasses]
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

function default_prediction(trainingweights,regressionvalues,predictiontask,method::LearningMethod{Regressor})
    sumweights = sum(trainingweights)
    sumregressionvalues = sum(regressionvalues)
    return [sumweights,sumregressionvalues]
    ## if sumweights > 0
    ##     return sum(trainingweights .* regressionvalues)/sumweights
    ## else
    ##     return :NA
    ## end
end

function leaf_node(trainingweights,regressionvalues,predictiontask,depth,method::LearningMethod{Classifier})
    if method.maxdepth > 0 && method.maxdepth == depth
        return true
    else
        noclasses = size(trainingweights,1)
        classweights = [sum(trainingweights[c]) for c = 1:noclasses]
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

function leaf_node(trainingweights,regressionvalues,predictiontask,depth,method::LearningMethod{Regressor})
    if method.maxdepth > 0 && method.maxdepth == depth
        return true
    else
        noinstances = sum(trainingweights)
        if noinstances >= 2*method.minleaf
            firstvalue = regressionvalues[1]
            i = 2
            multiplevalues = false
            novalues = length(regressionvalues)
            while i <= novalues &&  ~multiplevalues
                multiplevalues = firstvalue != regressionvalues[i]
                i += 1
            end
            return ~multiplevalues
        else
            return true
        end
    end
end

function make_leaf(trainingweights,regressionvalues,predictiontask,defaultprediction,method::LearningMethod{Classifier})
    noclasses = size(trainingweights,1)
    classcounts = zeros(noclasses)
    for i=1:noclasses
        classcounts[i] = sum(trainingweights[i])
    end
    noinstances = sum(classcounts)
    if noinstances > 0
        if method.laplace
            prediction = [(classcounts[i]+1)/(noinstances+noclasses) for i=1:noclasses]
        else
            prediction = [classcounts[i]/noinstances for i=1:noclasses]
        end
    else
        prediction = defaultprediction
    end
    return prediction
end

function make_leaf(trainingweights,regressionvalues,predictiontask,defaultprediction,method::LearningMethod{Classifier})
    sumweights = sum(trainingweights)
    sumregressionvalues = sum(regressionvalues)
    return [sumweights,sumregressionvalues]
    ## if sumweights > 0
    ##     prediction = sum(trainingweights .* regressionvalues)/sumweights
    ## else
    ##     prediction = defaultprediction
    ## end
    return prediction
end

# AMG: this is not split yet
function find_best_split(trainingrefs,trainingweights,regressionvalues,trainingdata,variables,types,predictiontask,method)
    if method.randsub == :all
        sampleselection = collect(1:length(variables))
    elseif method.randsub == :default
        if predictiontask == :CLASS
            sampleselection = sample(1:length(variables),convert(Int,floor(log(2,length(variables)))+1),replace=false)
        else
            sampleselection = sample(1:length(variables),convert(Int,floor(1/3*length(variables))+1),replace=false)
        end
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
        if predictiontask == :CLASS
            noclasses = size(trainingrefs,1)
            sampleregressionvalues = regressionvalues
            sampletrainingweights = Array(Any,noclasses)
            sampletrainingrefs = Array(Any,noclasses)
            for c = 1:noclasses
                if sum(trainingweights[c]) <= splitsamplesize
                    sampletrainingweights[c] = trainingweights[c]
                    sampletrainingrefs[c] = trainingrefs[c]
                else
                    sampletrainingweights[c] = Array(Float64,splitsamplesize)
                    sampletrainingrefs[c] = Array(Float64,splitsamplesize)
                    for i = 1:splitsamplesize
                        sampletrainingweights[c][i] = 1.0
                        sampletrainingrefs[c][i] = trainingrefs[c][rand(1:end)]
                    end
                end
            end
        else
            if sum(trainingweights) <= splitsamplesize
                sampletrainingweights = trainingweights
                sampletrainingrefs = trainingrefs
                sampleregressionvalues = regressionvalues
            else
                sampletrainingweights = Array(Float64,splitsamplesize)
                sampletrainingrefs = Array(Float64,splitsamplesize)
                sampleregressionvalues = Array(Float64,splitsamplesize)
                for i = 1:splitsamplesize
                    sampletrainingweights[i] = 1.0
                    randindex = rand(1:length(trainingrefs))
                    sampletrainingrefs[i] = trainingrefs[randindex]
                    sampleregressionvalues[i] = regressionvalues[randindex]
                end
            end
        end
    else
        sampletrainingrefs = trainingrefs
        sampletrainingweights = trainingweights
        sampleregressionvalues = regressionvalues
    end
    bestsplit = (-Inf,0,:NA,:NA,0.0)
    if predictiontask == :CLASS
        noclasses = size(trainingrefs,1)
        origclasscounts = Array(Float64,noclasses)
        for c = 1:noclasses
            origclasscounts[c] = sum(sampletrainingweights[c])
        end
        for v = 1:length(sampleselection)
            bestsplit = evaluate_variable_classification(bestsplit,sampleselection[v],variables[sampleselection[v]],types[sampleselection[v]],sampletrainingrefs,sampletrainingweights,origclasscounts,noclasses,trainingdata,method)
        end
        splitvalue, varno, variable, splittype, splitpoint = bestsplit
    else
        origregressionsum = sum(sampleregressionvalues .* sampletrainingweights)
        origweightsum = sum(sampletrainingweights)
        origmean = origregressionsum/origweightsum
        for v = 1:length(sampleselection)
            bestsplit = evaluate_variable_regression(bestsplit,sampleselection[v],variables[sampleselection[v]],types[sampleselection[v]],sampletrainingrefs,sampletrainingweights,
                                                     sampleregressionvalues,origregressionsum,origweightsum,origmean,trainingdata,method)
        end
        splitvalue, varno, variable, splittype, splitpoint = bestsplit
    end
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

function make_split(method::LearningMethod{Classiffier},trainingrefs,trainingweights,regressionvalues,trainingdata,predictiontask,bestsplit)
    (varno, variable, splittype, splitpoint) = bestsplit
    noclasses = size(trainingrefs,1)
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
        values[c] = trainingdata[c][varno][trainingrefs[c]]
        if splittype == :NUMERIC
            for r = 1:length(trainingrefs[c])
                ref = trainingrefs[c][r]
                if values[c][r] <= splitpoint
                    push!(leftrefs[c],ref)
                    push!(leftweights[c],trainingweights[c][r])
                else
                    push!(rightrefs[c],ref)
                    push!(rightweights[c],trainingweights[c][r])
                end
            end
        else
            for r = 1:length(trainingrefs[c])
                ref = trainingrefs[c][r]
                if values[c][r] == splitpoint
                    push!(leftrefs[c],ref)
                    push!(leftweights[c],trainingweights[c][r])
                else
                    push!(rightrefs[c],ref)
                    push!(rightweights[c],trainingweights[c][r])
                end
            end
        end
    end
    noleftexamples = sum([sum(leftweights[i]) for i=1:noclasses])
    norightexamples = sum([sum(rightweights[i]) for i=1:noclasses])
    leftweight = noleftexamples/(noleftexamples+norightexamples)
    return leftrefs,leftweights,[],rightrefs,rightweights,[],leftweight
end

function make_split(method::LearningMethod{Regressor},trainingrefs,trainingweights,regressionvalues,trainingdata,predictiontask,bestsplit)
    (varno, variable, splittype, splitpoint) = bestsplit
    leftrefs = Int[]
    leftweights = Float64[]
    leftregressionvalues = Float64[]
    rightrefs = Int[]
    rightweights = Float64[]
    rightregressionvalues = Float64[]
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
                push!(leftregressionvalues,regressionvalues[r])
            else
                push!(rightrefs,ref)
                push!(rightweights,trainingweights[r])
                sumrightweights += trainingweights[r]
                push!(rightregressionvalues,regressionvalues[r])
            end
        end
    else
        for r = 1:length(trainingrefs)
            ref = trainingrefs[r]
            if values[r] == splitpoint
                push!(leftrefs,ref)
                push!(leftweights,trainingweights[r])
                sumleftweights += trainingweights[r]
                push!(leftregressionvalues,regressionvalues[r])
            else
                push!(rightrefs,ref)
                push!(rightweights,trainingweights[r])
                sumrightweights += trainingweights[r]
                push!(rightregressionvalues,regressionvalues[r])
            end
        end
    end
  end
  leftweight = sumleftweights/(sumleftweights+sumrightweights)
  return leftrefs,leftweights,leftregressionvalues,rightrefs,rightweights,rightregressionvalues,leftweight
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

function restructure_tree(tree)
    nonodes = size(tree,1)
    newtree = Array(Any,nonodes)
    for i = 1:nonodes
        nodeno, node = tree[i]
        newtree[nodeno] = node
    end
    return newtree
end

##
## Function for making a prediction with a single tree
##

function make_prediction(tree,testdata,exampleno,prediction)
    stack = Any[]
    nodeno = 1
    weight = 1.0
    push!(stack,(nodeno,weight))
    while stack != []
        nodeno, weight = pop!(stack)
        node = tree[nodeno]
        if node[1] == :LEAF
            prediction += weight*node[2]
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

##
## Functions for outputting result summaries
##

function present_results(results,methods;ignoredatasetlabel = false)
    if results != []
        if results[1][1] == :CLASS # NOTE: Assuming all tasks of the same type
            resultlabels = fieldnames(ClassificationResult) #FIXME: MOH Can be extracted from the model information
        else
            resultlabels = fieldnames(RegressionResult)
        end
        methodresults = Array(Float64,length(results),length(methods),length(resultlabels))
        for datasetno = 1:length(results)
            for methodno = 1:length(methods)
                for resultno = 1:length(resultlabels)
                    methodresults[datasetno,methodno,resultno] = results[datasetno][3][methodno].(resultlabels[resultno])
                end
            end
        end
        rankingresults = Array(Float64,length(results),length(methods),length(resultlabels))
        for datasetno = 1:length(results)
            for resultno = 1:length(resultlabels)
                tempres = vec(methodresults[datasetno,:,resultno])
                resultlabel = resultlabels[resultno]
                for methodno = 1:length(methods)
                    rankingresults[datasetno,methodno,resultno] = get_rank(tempres[methodno],tempres,resultlabel)
                end
            end
        end
        maxsizes = Array(Int,length(methods),length(resultlabels))
        for methodno = 1:length(methods)
            for resultno = 1:length(resultlabels)
                tempresults = methodresults[:,methodno,resultno]
                maxsize = maximum([length(string(round(v,4))) for v in [mean(tempresults);tempresults]])
                if maxsize < 6
                    maxsize = 6
                end
                maxsizes[methodno,resultno] = maxsize
            end
        end
        methodlabels = [string("M",i) for i=1:length(methods)]
        maxdatasetnamesize = maximum([length(dataset) for (task,dataset,results) in results])
        if results[1][1] == :CLASS
            println("\nClassification results")
        else
            println("\nRegression results")
        end
        print_aligned_l("",maxdatasetnamesize)
        for resultno = 1:length(resultlabels)
            print("\t")
            print_aligned_r("$(resultlabels[resultno])",maxsizes[1,resultno])
            for methodno = 2:length(methodlabels)
                print("\t")
                print(" "^maxsizes[methodno,resultno])
            end
        end
        println("")
        if ignoredatasetlabel
            print_aligned_l("",maxdatasetnamesize)
        else
            print_aligned_l("Dataset",maxdatasetnamesize)
        end
        if ~(ignoredatasetlabel && length(methods) == 1)
            for resultno = 1:length(resultlabels)
                print("\t")
                print_aligned_r("$(methodlabels[1])",maxsizes[1,resultno])
                for methodno = 2:length(methodlabels)
                    print("\t")
                    print_aligned_r("$(methodlabels[methodno])",maxsizes[methodno,resultno])
                end
            end
            println("")
        end
        println("")
        for datasetno = 1:length(results)
            print_aligned_l(results[datasetno][2],maxdatasetnamesize)
            for resultno = 1:length(resultlabels)
                for methodno = 1:length(methodlabels)
                    print("\t")
                    print_aligned_r("$(round(methodresults[datasetno,methodno,resultno],4))",maxsizes[methodno,resultno])
                end
            end
            println("")
        end
        if length(results) > 1
            println("")
            print_aligned_l("Mean",maxdatasetnamesize)
            for resultno = 1:length(resultlabels)
                for methodno = 1:length(methodlabels)
                    print("\t")
                    print_aligned_r("$(round(mean(methodresults[:,methodno,resultno]),4))",maxsizes[methodno,resultno])
                end
            end
            println("")
            println("")
        end
        if length(methods) > 1
            println("")
            println("Ranks")
            print_aligned_l("",maxdatasetnamesize)
            for resultno = 1:length(resultlabels)
                print("\t")
                print_aligned_r("$(resultlabels[resultno])",maxsizes[1,resultno])
                for methodno = 2:length(methodlabels)
                    print("\t")
                    print(" "^maxsizes[methodno,resultno])
                end
            end
            println("")
            if ignoredatasetlabel
                print_aligned_l("",maxdatasetnamesize)
            else
                print_aligned_l("Dataset",maxdatasetnamesize)
            end
            for resultno = 1:length(resultlabels)
                print("\t")
                print_aligned_r("$(methodlabels[1])",maxsizes[1,resultno])
                for methodno = 2:length(methodlabels)
                    print("\t")
                    print_aligned_r("$(methodlabels[methodno])",maxsizes[methodno,resultno])
                end
            end
            println("")
            println("")
            for datasetno = 1:length(results)
                print_aligned_l(results[datasetno][2],maxdatasetnamesize)
                for resultno = 1:length(resultlabels)
                    for methodno = 1:length(methodlabels)
                        print("\t")
                        print_aligned_r("$(round(rankingresults[datasetno,methodno,resultno],4))",maxsizes[methodno,resultno])
                    end
                end
                println("")
            end
            if length(results) > 1
                println("")
                print_aligned_l("Mean",maxdatasetnamesize)
                for resultno = 1:length(resultlabels)
                    for methodno = 1:length(methodlabels)
                        print("\t")
                        print_aligned_r("$(round(mean(rankingresults[:,methodno,resultno]),4))",maxsizes[methodno,resultno])
                    end
                end
                println("")
                println("")
            end
            println("Methods")
        else
            println("")
            println("Method")
        end
        println("")
        for m = 1:length(methods)
            present_method(m,methods[m],showmethodlabel = ~(ignoredatasetlabel && length(methods) == 1))
            println("")
        end
    end
end

function print_aligned_r(Str,Size)
    noblanks = Size-length(Str)
    if noblanks > 0
        print(" "^noblanks,Str)
    else
        print(Str)
    end
end

function print_aligned_l(Str,Size)
    noblanks = Size-length(Str)
    if noblanks > 0
        print(Str," "^noblanks)
    else
        print(Str)
    end
end

function get_rank(value,values,metric) # NOTE: value must be present in values
    if metric in [:MSE,:AvMSE,:OOBErr,:DEOAcc,:AEEMSE,:AEEAcc,:Brier,:AvBrier,:Region,:Size,:NoIrr,:Time]
        sortedvalues = sort(values)
        range = searchsorted(sortedvalues,value)
    else
        sortedvalues = sort(values,rev = true)
        range = searchsorted(sortedvalues,value, rev = true)
    end
    resultrank = sum(range)/length(range)
    return resultrank
end

function present_method(methodno,method;showmethodlabel = true)
    if showmethodlabel
        print("M$(methodno): ")
    end
    for n in fieldnames(method)
        println("\t$(n) = $(method.(n))")
    end
end

##
## Functions for working with a single dataset
##

function load_data(source; separator = ',')
    if typeof(source) == ASCIIString
        global globaldata = read_data(source, separator=separator) # Made global to allow access from workers
        initiate_workers()
        println("Data loaded")
    elseif typeof(source) == DataFrame
        if ~(:WEIGHT in names(source))
            global globaldata = hcat(source,DataFrame(WEIGHT = ones(size(source,1))))
        else
            global globaldata = source # Made global to allow access from workers
        end
        initiate_workers()
        println("Data loaded")
    else
        println("Data can only be loaded from text files or DataFrames")
    end
end

function describe_data()
    println("No. of examples: $(size(globaldata,1))")
    println("No. of columns: $(size(globaldata,2)-1)")
    println("")
    println("Columns:")
    describe(globaldata)
end

function evaluate_method(;method = forest(),protocol = 10)
    println("Running experiment")
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

function generate_model(method)
    predictiontask = prediction_task(globaldata)
    if predictiontask == :NONE # FIXME: MOH We should not be doing this...probably DEAD code
        println("The loaded dataset is not on the correct format: CLASS/REGRESSION column missing")
        println("This may be due to an incorrectly specified separator, e.g., use: separator = \'\\t\'")
        result = :NONE
    else
        if typeof(method.learningType) == Classifier
            classes = unique(globaldata[:CLASS])
        else
            classes = []
        end
        nocoworkers = nprocs()-1
        if nocoworkers > 0
            notrees = [div(method.notrees,nocoworkers) for i=1:nocoworkers]
            for i = 1:mod(method.notrees,nocoworkers)
                notrees[i] += 1
            end
        else
            notrees = [method.notrees]
        end
        treesandoobs = pmap(generate_trees,[(method,predictiontask,classes,n,rand(1:1000_000_000)) for n in notrees])
        trees = [treesandoobs[i][1] for i=1:length(treesandoobs)]
        oobs = [treesandoobs[i][2] for i=1:length(treesandoobs)]
        variableimportance = treesandoobs[1][3]
        for i = 2:length(treesandoobs)
            variableimportance += treesandoobs[i][3]
        end
        variableimportance = variableimportance/method.notrees
        variables, types = get_variables_and_types(globaldata)
        variableimportance = hcat(variables,variableimportance)
        oobperformance, conformalfunction = generate_model_internal(method, oobs)
        result = PredictionModel(predictiontask,classes,(majorversion,minorversion,patchversion),method,oobperformance,variableimportance,vcat(trees...),conformalfunction)
        println("Model generated")
    end
    return result
end

function generate_model_internal(method::LearningMethod{Classifier})
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

function generate_model_internal(method::LearningMethod{Regressor})
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

function describe_model(model::PredictionModel)
    println("Generated by: RandomForest v. $(model.version[1]).$(model.version[2]).$(model.version[3])")
    if model.predictiontask == :CLASS
        println("Prediction task: classification")
        println("Class labels: $(model.classes)")
    else
        println("Prediction task: regression")
    end
    println("Learning method:")
    present_method(0,model.method,showmethodlabel = false)
    if model.predictiontask == :CLASS
        println("OOB accuracy: $(model.oobperformance)")
    else
        println("OOB MSE: $(model.oobperformance)")
    end
    varimp = sortrows(model.variableimportance,by=x->x[2],rev=true)
    println("Variable importance:")
    for i = 1:size(varimp,1)
        println("$(varimp[i,1])\t$(varimp[i,2])")
    end
end

# AMG: not dispatched yet
function apply_model(model; confidence = :std)
    predictiontask = prediction_task(globaldata)
    if predictiontask != model.predictiontask
        println("The model is not consistent with the loaded dataset")
        predictions = :NONE
    else
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
            results = pmap(apply_trees,[(predictiontask,model.classes,subtrees) for subtrees in alltrees])
            if predictiontask == :REGRESSION
                predictions = results[1][1]
                squaredpredictions = results[1][2]
                for r = 2:length(results)
                    predictions += results[r][1]
                    squaredpredictions += results[r][2]
                end
            else
                predictions = results[1]
                for r = 2:length(results)
                    predictions += results[r]
                end
            end
        else
            if predictiontask == :REGRESSION
                predictions, squaredpredictions = apply_trees((predictiontask,model.classes,model.trees))
            else
                predictions = apply_trees((predictiontask,model.classes,model.trees))
            end
        end
        predictions = predictions/model.method.notrees
        if predictiontask == :REGRESSION
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
        else # predictiontask == :CLASS
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
        end
    end
    return results
end

# AMG: not dispatched yet
function apply_trees(predictiontask, classes, trees)
    variables, types = get_variables_and_types(globaldata)
    testmissingvalues, testnonmissingvalues = find_missing_values(:UNKNOWN,variables,globaldata)
    newtestdata = transform_nonmissing_columns_to_arrays(:UNKNOWN,variables,globaldata,testmissingvalues)
    replacements_for_missing_values!(newtestdata,globaldata,:UNKNOWN,variables,types,testmissingvalues,testnonmissingvalues)
    nopredictions = size(globaldata,1)
    if predictiontask == :CLASS
        noclasses = length(classes)
        predictions = Array(Any,nopredictions)
        for i = 1:nopredictions
            predictions[i] = zeros(noclasses)
            for t = 1:length(trees)
                treeprediction = make_prediction(trees[t],newtestdata,i,zeros(noclasses))
                predictions[i] += treeprediction
            end
        end
        results = predictions
    else
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
    end
    return results
end

end
