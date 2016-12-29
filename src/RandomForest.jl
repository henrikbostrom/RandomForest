## jl
## v. 0.10.0
##
## Random forests for classification, regression and survival analysis with conformal prediction
## NOTE: survival analysis under development!
##
## Developed for Julia 0.5 (http://julialang.org/)
##
## Copyright Henrik Boström 2016
## Email: henrik.bostrom@dsv.su.se
##
## TODO for version 1.0:
##
## *** MUST ***
##
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
    load_sparse_data,
    describe_data,
    evaluate_method,
    evaluate_methods,
    generate_model,
    store_model,
    load_model,
    describe_model,
    apply_model,
    runexp,
    fit!,
    predict,
    predict_proba,
    # LearningMethod,
    # Classifier,
    # Regressor,
    forestClassifier,
    treeClassifier,
    forestRegressor,
    treeRegressor,
    forestSurvival,
    treeSurvival

include("types.jl")
include("common.jl")
include("print.jl")
include("classification.jl")
include("regression.jl")
include("survival.jl")
include("classificationWithTest.jl")
include("regressionWithTest.jl")
include("scikitlearnAPI.jl")
include("survivalWithTest.jl")
include("sparseData.jl")
global useSparseData = false

"""`runexp` is used to test the performance of the library on a number of test sets"""
function runexp()
    experiment(files = ["uci/glass.txt"]) # Warmup
    experiment(files="uci",methods=[forest(),forest(notrees=500)],resultfile="uci-results.txt")
    experiment(files = ["regression/cooling.txt"]) # Warmup
    experiment(files="regression",methods=[forest(),forest(notrees=500)],resultfile="regression-results.txt")
    experiment(files = ["survival/pharynx.csv"]) # Warmup
    experiment(files="survival",methods=[forest(notrees=500,minleaf=10),forest(notrees=1000,minleaf=10)],resultfile="survival-results.txt")
end


##
## Functions for running experiments
##

function experiment(;files = ".", separator = ',', protocol = 10, normalizetarget = false, normalizeinput = false, methods = [forest()], resultfile = :none, printable = true, sparse = false)
    global useSparseData = sparse
    println("RandomForest v. $rf_ver")
    if typeof(files) == String
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
    survivalresults = [pt == :SURVIVAL for (pt,f,r) in results]
    if printable
      present_results(sort(results[classificationresults]),methods)
      present_results(sort(results[regressionresults]),methods)
      present_results(sort(results[survivalresults]),methods)
      println("Total time: $(round(totaltime,2)) s.")
    end
    if resultfile != :none
        origstdout = STDOUT
        resultfilestream = open(resultfile,"w+")
        redirect_stdout(resultfilestream)
        present_results(sort(results[classificationresults]),methods)
        present_results(sort(results[regressionresults]),methods)
        present_results(sort(results[survivalresults]),methods)
        println("Total time: $(round(totaltime,2)) s.")
        redirect_stdout(origstdout)
        close(resultfilestream)
    end
    return results
end

function run_experiment(file, separator, protocol, normalizetarget, normalizeinput, methods)
    global globaldata = read_data(file, separator=separator) # Made global to allow access from workers
    predictiontask = prediction_task(globaldata)
    if predictiontask == :NONE
        warn("File excluded: $file - no column is labeled CLASS or REGRESSION\n\tThis may be due to incorrectly specified separator, e.g., use: separator = \'\\t\'")
        result = (:NONE,:NONE,:NONE)
    else
        if predictiontask == :REGRESSION
            methods = map(x->LearningMethod(Regressor(), (getfield(x,i) for i in fieldnames(x)[2:end])...), methods)
        elseif predictiontask == :CLASS
            methods = map(x->LearningMethod(Classifier(), (getfield(x,i) for i in fieldnames(x)[2:end])...), methods)
        else # predictiontask == :SURVIVAL
            methods = map(x->LearningMethod(Survival(), (getfield(x,i) for i in fieldnames(x)[2:end])...), methods)
        end
        if predictiontask == :REGRESSION && normalizetarget
            regressionvalues = globaldata[:REGRESSION]
            minval = minimum(regressionvalues)
            maxval = maximum(regressionvalues)
            normalizedregressionvalues = [(v-minval)/(maxval-minval) for v in regressionvalues]
            globaldata[:REGRESSION] = convert(Array{Float64},normalizedregressionvalues)
        end
        if normalizeinput # NOTE: currently assumes that all input is numeric and that there are no missing values
            for label in names(globaldata)
                if ~(label in [:REGRESSION,:CLASS,:ID,:WEIGHT,:TEST,:FOLD,:TIME,:EVENT])
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
            results = run_split(protocol,methods)
            result = (predictiontask,file,results)
        elseif typeof(protocol) == Int64 || protocol == :cv
            results = run_cross_validation(protocol,methods)
            result = (predictiontask,file,results)
        else
            throw("Unknown experiment protocol")
        end
        typeof(file)==String && println("Completed experiment with: $file")
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

function run_split(testoption,methods)
    if typeof(testoption) == Float64
        noexamples = size(globaldata,1)
        notestexamples = floor(Int,testoption*noexamples)
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
    numThreads = Threads.nthreads()
    time = 0
    methodresults = Array(Any,length(methods))
    origseed = rand(1:1000_000_000) #FIXME: randomizing the random seed doesn't help
    for m = 1:length(methods)
        results = Array{Any,1}()
        srand(origseed) #NOTE: To remove configuration order dependance
        if methods[m].modpred
            randomoobs = Array(Int64,notestexamples)
            for i = 1:notestexamples
                randomoobs[i] = rand(1:notrainingexamples)
            end
        else
            randomoobs = Array(Int,0)
        end
        if nocoworkers > 0
            notrees = getnotrees(methods[m], nocoworkers)
            time = @elapsed results = pmap(generate_and_test_trees,[(methods[m],:test,n,rand(1:1000_000_000),randomoobs) for n in notrees])
        elseif numThreads > 1
            notrees = getnotrees(methods[m], numThreads)
            results = Array{Any,1}(length(notrees))
            time = @elapsed Threads.@threads for n in notrees
                results[Threads.threadid()] = generate_and_test_trees((methods[m],:test,n,rand(1:1000_000_000),randomoobs))
            end
            waitfor(results)
        else
            notrees = [methods[m].notrees]
            time = @elapsed results = generate_and_test_trees.([(methods[m],:test,n,rand(1:1000_000_000),randomoobs) for n in notrees])
        end

        tic()
        methodresult = run_split_internal(methods[m], results, time)
        methodresults[m] = Dict("performance"=>methodresult[1], "predictions"=>methodresult[2])
        if length(methodresult) > 2
            methodresults[m]["classLabels"] = methodresult[3]
        end
    end
    return Dict("results"=>methodresults, "testSplit"=>map(i->i ? 1 : 0, globaldata[:TEST]), "type"=>prediction_task(globaldata))
end

function run_cross_validation(protocol,methods)
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
    numThreads = Threads.nthreads()
    time = 0
    methodresults = Array(Any,length(methods))
    origseed = rand(1:1000_000_000)
    for m = 1:length(methods)
        srand(origseed)
        if methods[m].conformal == :default
            conformal = :normalized
        else
            conformal = methods[m].conformal
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
        if nocoworkers > 0
            notrees = getnotrees(methods[m], nocoworkers)
            time = @elapsed results = pmap(generate_and_test_trees,[(methods[m],:cv,n,rand(1:1000_000_000),randomoobs) for n in notrees])
        elseif numThreads > 1
            notrees = getnotrees(methods[m], numThreads)
            results = Array{Any,1}(length(notrees))
            time = @elapsed Threads.@threads for n in notrees
                results[Threads.threadid()] = generate_and_test_trees((methods[m],:cv,n,rand(1:1000_000_000),randomoobs))
            end
            waitfor(results)
        else
            notrees = [methods[m].notrees]
            time = @elapsed results = generate_and_test_trees.([(methods[m],:cv,n,rand(1:1000_000_000),randomoobs) for n in notrees])
        end
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
        methodresult = run_cross_validation_internal(methods[m], results, modelsizes, nofolds, conformal, time)
        methodresults[m] = Dict("performance"=>methodresult[1], "predictions"=>methodresult[2])
        if length(methodresult) > 2
            methodresults[m]["classLabels"] = methodresult[3]
        end
    end
    return Dict("results"=>methodresults, "type"=>prediction_task(globaldata))
end

## 
function load_sparse_data(source, target; predictionType=:CLASS, separator = ' ', n = -1)
    global useSparseData = true
    df = readdlm(source, separator)
    if n == -1
        n = maximum([parse(split(filter(i->length(i) != 0, df[r,:])[end], ":")[1]) for r in 1:size(df,1)])
    end
    sparseMatrix = spzeros(size(df,1), n)
    for r = 1:size(df,1)
        d = filter(i->length(i) != 0, df[r,:])
        spd = Dict(parse(split(i,":")[1])=>parse(split(i,":")[2]) for i in d)
        sparseMatrix[r, :] = sparsevec(spd, n)
    end
    labels = readdlm(target, separator)[:,1]
    names = Any[1:n...]
    push!(names,predictionType)
    push!(names,:WEIGHT)
    # equivDataFrame = DataFrame(full(sparseMatrix))
    # equivDataFrame[predictionType] = labels
    # equivDataFrame = hcat(equivDataFrame,DataFrame(WEIGHT = ones(size(equivDataFrame,1))))
    # global globaldata = equivDataFrame
    global globaldata = SparseData(names,[sparseMatrix[:,i] for i = 1:n], labels, ones(length(labels)))
    initiate_workers()
    println("Data: $(source)")
end

##
## Functions for working with a single dataset. Amg: loading data should move outside as well
##
function load_data(source; separator = ',', sparse=false)
    global useSparseData = sparse
    if typeof(source) == String
        global globaldata = read_data(source, separator=separator) # Made global to allow access from workers
        initiate_workers()
        println("Data: $(source)")
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

end
