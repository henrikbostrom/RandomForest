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

"""`runexp` is used to test the performance of the library on a number of test sets"""
function runexp()
    experiment(files = ["uci/glass.txt"]) # Warmup
    experiment(files="uci",methods=[forest(),forest(notrees=500)],resultfile="uci-results.txt")
    experiment(files = ["regression/cooling.txt"]) # Warmup
    experiment(files="regression",methods=[forest(),forest(notrees=500)],resultfile="regression-results.txt")
    experiment(files = ["survival/pharynx.csv"]) # Warmup
    experiment(files="survival",methods=[forest(notrees=500,minleaf=10),forest(notrees=1000,minleaf=10)],resultfile="survival-results.txt")
end

"""
An experiment is run by calling experiment(...) in the following way:

    julia> experiment(files = <files>, separator = <separator>, protocol = <protocol>,
                      normalizetarget = <normalizetarget>, normalizeinput = <normalizeinput>,
                      methods = [<method>, ...])

The arguments should be on the following format:

    files : list of file names or path to directory (default = ".")
        - in a specified directory, files with extensions other than .txt and .csv are ignored
        - each file should contain a dataset (see format requirements below)
        - example: files = [\"uci/house-votes.txt\", \"uci/glass.txt\"],
        - example: files = "uci"

    separator : single character (default = ',')
        - the character to use as field separator
        - example: separator = '\t' (the tab character)

    protocol : integer, float, :cv, :test (default = 10)
        - the experiment protocol:
            an integer means using cross-validation with this no. of folds.
            a float between 0 and 1 means using this fraction of the dataset for testing
            :cv means using cross-validation with folds specified by a column labeled FOLD
            :test means dividing the data into training and test according to boolean values
             in a column labeled TEST (true means that the example is used for testing)
        - example: protocol = 0.25 (25% of the data is for testing)

    normalizetarget : boolean (default = false)
        - true means that each regression value v will be replaced by
          (v-v_min)/(v_max-v_min), where v_min and V-max are the minimum and maximum values
        - false means that the original regression values are kept

    normalizeinput : boolean (default = false)
        - true means that each numeric input value v will be replaced by
          (v-v_min)/(v_max-v_min), where v_min and V-max are the minimum and maximum values
        - false means that the original values are kept

    method : a call on the form forest(...) (default = forest())

        [FIXME: tree() method is missing here]: #
        - The call may have the following (optional) arguments:

            notrees : integer (default = 100)
                - no. of trees to generate in the forest

            minleaf : integer (default = 1)
                - minimum no. of required examples to form a leaf

            maxdepth : integer (default = 0)
                - maximum depth of generated trees (0 means that there is no depth limit)

            randsub : integer, float, :default, :all, :log2, or :sqrt (default = :default)
                - no. of randomly selected features to evaluate at each split:
                   :default means :log2 for classification and 1/3 for regression
                   :all means that all features are used (no feature subsampling takes place)
                   :log2 means that log2 of the no. of features are sampled
                   :sqrt means that sqrt of the no. of features are sampled
                   an integer (larger than 0) means that this number of features are sampled
                   a float (between 0.0 and 1.0) means that this fraction of features are sampled

            randval : boolean (default = true)
                - true means that a single randomly selected value is used to form conditions for each
                  feature in each split
                - false mean that all values are used to form conditions when evaluating features for
                  each split

            splitsample : integer (default = 0)
                - no. of randomly selected examples to use for evaluating each split
                - 0 means that no subsampling of the examples will take place

            bagging : boolean (default = true)
                - true means that a bootstrap replicate of the training examples is used for each tree
                - false means that the original training examples are used when building each tree

            bagsize : float or integer (default = 1.0)
                - no. of randomly selected examples to include in the bootstrap replicate
                - an integer means that this number of examples are sampled with replacement
                - a float means that the corresponding fraction of examples are sampled with replacement

            modpred : boolean (default = false)
                - true means that for each test instance, the trees for which a randomly selected training
                  instance is out-of-bag is used for prediction and the training instance is not used for
                  calculating a calibration score
                - false means that all trees in the forest are used for prediction and all out-of-bag scores
                  are used for calibration

            laplace : boolean (default = false)
                - true means that class probabilities at each leaf node is Laplace corrected
                - false means that class probabilities at each leaf node equal the relative class
                  frequencies

            confidence : a float between 0 and 1 (default = 0.95)
                - probability of including the correct label in the prediction region

            conformal : :default, :std, :normalized or :classcond (default = :default)
                - method used to calculate prediction regions
                - For classification, the following options are allowed:
                   :default is the same as :std
                   :std means that validity is guaranteed in general, but not for each class
                   :classcond means that validity is guaranteed for each class
                - For regression, the following options are allowed:
                   :default is the same as :normalized
                   :std results in the same region size for all predictions
                   :normalized means that each region size is dependent on the spread
                    of predictions among the individual trees

---------

    Examples:

    The call experiment(files = "uci") is hence the same as

    experiment(files = "uci", separator = ´,´, protocol = 10, methods = [forest()])

    The following compares the default random forest to one with 1000 trees and a maxdepth of 10:

    julia> experiment(files = "uci", methods = [forest(), forest(notrees = 1000, maxdepth = 10)])

---------

A dataset should have the following format:

    <names-row>
    <data-row>
    ...
    <data-row>

where

    <names-row> = <name><separator><name><separator>...<name>
and

    <data-row>  = <value><separator><value><separator>...<value>

\<name\> can be any of the following:

        CLASS            - declares that the column contains class labels
        REGRESSION       - declares that the column contains regression values
        ID               - declares that the column contains identifier labels
        IGNORE           - declares that the column should be ignored
        FOLD             - declares that the column contains labels for cross-validation folds
        WEIGHT           - declares that the column contains instance weights
        any other value  - is used to create a variable name

\<separator\> is a single character (as specified above)

\<value\> can be any of the following:

        integer          - is handled as a number if all values in the same column are of type integer,
                           float or NA, and as a string otherwise
        float            - is handled as a number if all values in the same column are of type integer,
                           float or NA, and as a string otherwise
        NA               - is handled as a missing value
        any other value  - is handled as a string

Example:

    ID,RI,Na,Mg,Al,Si,K,Ca,Ba,Fe,CLASS
    1,1.52101,NA,4.49,1.10,71.78,0.06,8.75,0.00,0.00,1
    2,1.51761,13.89,NA,1.36,72.73,0.48,7.83,0.00,0.00,1
    3,1.51618,13.53,3.55,1.54,72.99,0.39,7.78,0.00,0.00,1
    ...

---------

For classification tasks the following measures are reported:

        Acc        - accuracy, i.e., fraction of examples correctly predicted
        AUC        - area under ROC curve
        Brier      - Brier score
        AvAcc      - average accuracy for single trees in the forest
        DEOAcc     - difference of the estimated and observed accuracy
        AEEAcc     - absolute error of the estimated accuracy
        AvBrier    - average Brier score for single trees in the forest
        VBrier     - average squared deviation of single tree predictions from forest predictions
        Margin     - diff. between prob. for correct class and prob. for most prob. other class
        Prob       - probability for predicted class
        Valid      - fraction of true labels included in prediction region
        Region     - size, i.e., number of labels, in prediction region
        OneC       - fraction of prediction regions containing exactly one true label
        Size       - the number of nodes in the forest
        Time       - the total time taken for both training and testing

For regression tasks the following measures are reported:

        MSE        - mean squared error
        Corr       - the Pearson correlation between predicted and actual values
        AvMSE      - average mean squared error for single trees in the forest
        VarMSE     - average squared deviation of single tree predictions from forest predictions
        DEOMSE     - difference of the estimated and observed MSE
        AEEMSE     - absolute error of the estimated MSE
        Valid      - fraction of true labels included in prediction region
        Region     - average size of prediction region
        Size       - the number of nodes in the forest
        Time       - the total time taken for both training and testing
"""
##
## Functions for running experiments
##

function experiment(;files = ".", separator = ',', protocol = 10, normalizetarget = false, normalizeinput = false, methods = [forest()], resultfile = :none, printable = true)
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

end
