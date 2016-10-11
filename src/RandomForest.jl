## jl
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
    runexp,
    fit!,
    predict,
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
include("classificationWithTest.jl")
include("regressionWithTest.jl")
include("scikitlearnAPI.jl")

# MOH FIXME:should use Julia standardized versioning instead
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


##
## Functions for running experiments
##

function experiment(;files = ".", separator = ',', protocol = 10, normalizetarget = false, normalizeinput = false, methods = [forest()], resultfile = :none)
    println("RandomForest v. $(majorversion).$(minorversion).$(patchversion)")
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
        if predictiontask == :REGRESSION
            methods = map(x->LearningMethod(Regressor(), (getfield(x,i) for i in fieldnames(x)[2:end])...), methods)
        else
            methods = map(x->LearningMethod(Classifier(), (getfield(x,i) for i in fieldnames(x)[2:end])...), methods)
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
    # MOH FIXME: At this point you should know that method result to expect
    methodresults = Array(Any,length(methods))
    origseed = rand(1:1000_000_000)
    for m = 1:length(methods)
        srand(origseed) #NOTE: To remove configuration order dependance
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

function initiate_workers()
    pr = Array(Any,nprocs())
    for i = 2:nprocs()
        pr[i] = remotecall(i,load_global_dataset)
    end
    for i = 2:nprocs()
        wait(pr[i])
    end
end

function load_global_dataset()
    global globaldata = @fetchfrom(1,globaldata)
end

function update_workers()
    pr = Array(Any,nprocs())
    for i = 2:nprocs()
        pr[i] = remotecall(i,update_global_dataset)
    end
    for i = 2:nprocs()
        wait(pr[i])
    end
end

function update_global_dataset()
    global globaltests = @fetchfrom(1,globaltests)
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
        methodresults[m] = run_cross_validation_internal(methods[m], results, modelsizes, nofolds, conformal, time)
    end
    return methodresults
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

function prediction_task(data)
    allnames = names(data)
    if :CLASS in allnames
        return :CLASS
    elseif :REGRESSION in allnames
        return :REGRESSION
    else
        return :NONE
    end
end

##
## Functions for working with a single dataset. Amg: loading data should move outside as well
##
function load_data(source; separator = ',')
    if typeof(source) == String
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

end
