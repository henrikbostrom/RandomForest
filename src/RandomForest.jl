## jl
## v. 0.0.11
##
## Random forests for classification, regression and survival analysis with conformal prediction
## NOTE: survival analysis under development!
##
## Developed for Julia 0.5 (http://julialang.org/)
##
## Copyright Henrik Boström 2017
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
    forestClassifier,
    treeClassifier,
    forestRegressor,
    treeRegressor,
    forestSurvival,
    treeSurvival

include("types.jl")
include("print.jl")
include("classification.jl")
include("regression.jl")
include("survival.jl")
include("scikitlearnAPI.jl")
include("sparseData.jl")

global useSparseData = false
global const rf_ver=v"0.0.11"

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


    Examples:

    The call experiment(files = "uci") is hence the same as

    experiment(files = "uci", separator = ´,´, protocol = 10, methods = [forest()])

    The following compares the default random forest to one with 1000 trees and a maxdepth of 10:

    julia> experiment(files = "uci", methods = [forest(), forest(notrees = 1000, maxdepth = 10)])


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


A sparse dataset should have the following format:

    <data-row>
    ...
    <data-row>

where

    <data-row> = <column number>:<value><separator><column number>:<value><separator> ... <column number>:<value><separator>

An example for a sparse dataset: https://archive.ics.uci.edu/ml/machine-learning-databases/dexter/DEXTER/dexter_test.data

\<column number\> an integer number representing column index

\<value\> can be integer, or float

\<separator\> is a single character (as specified above)


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
        methodresult = collect_results_split(methods[m], results, time)
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
        methodresult = collect_results_cross_validation(methods[m], results, modelsizes, nofolds, conformal, time)
        methodresults[m] = Dict("performance"=>methodresult[1], "predictions"=>methodresult[2])
        if length(methodresult) > 2
            methodresults[m]["classLabels"] = methodresult[3]
        end
    end
    return Dict("results"=>methodresults, "type"=>prediction_task(globaldata))
end

##
## Functions for working with a single file
##

"""
To load a dataset from a file or dataframe:

    julia> load_data(<filename>, separator = <separator>)
    julia> load_data(<dataframe>)

The arguments should be on the following format:

    filename : name of a file containing a dataset (see format requirements above)
    separator : single character (default = ',')
    dataframe : a dataframe where the column labels should be according to the format requirements above
"""
            
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

"""
To load a dataset from a file:

    julia> load_sparse_data(<filename>, <labels_filename>, predictionType = <predictionType>, separator = <separator>, n = <numberOfFeatures>)

The arguments should be on the following format:

    filename : name of a file containing a sparse dataset (see format requirements above)
    labels_filename :  name of a file containing a vector of labels
    separator : single character (default = ' ')
    predictionType : one of :CLASS, :REGRESSION, or :SURVIVAL
    n : Number of features in the dataset (auto detected if not provided)
"""
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
## Function for building a single tree.
##
function build_tree(method,alltrainingrefs,alltrainingweights,allregressionvalues,alltimevalues,alleventvalues,trainingdata,variables,types,varimp)
    leafnodesstats = Int[0, 0] # noleafnodes, noirregularleafnodes
    T1 = typeof(method.learningType) == Classifier ? Array{Int,1} : Int
    T2 = typeof(method.learningType) == Classifier ? Array{Float64,1} : Float64
    PredictType = typeof(method.learningType) == Survival ? Array{Array{Float64,1},1} : Array{Float64,1}
    treeData = TreeData{T1, T2}(0,alltrainingrefs,alltrainingweights,allregressionvalues,alltimevalues,alleventvalues)
    variableimportance = zeros(length(variables))
    tree = get_tree_node(treeData , variableimportance, leafnodesstats, trainingdata,variables,types,method,[],varimp,PredictType)
    if varimp
        s = sum(variableimportance)
        if (s != 0)
            variableimportance = variableimportance/s
        end
    else
        variableimportance = :NONE
    end
    return tree, variableimportance, leafnodesstats[1], leafnodesstats[2]
end

function get_tree_node(treeData, variableimportance, leafnodesstats, trainingdata,variables,types,method,parenttrainingweights,varimp,PredictType)
    if leaf_node(treeData, method)
        leafnodesstats[1] += 1
        return TreeNode{PredictType,Void}(:LEAF,make_leaf(treeData, method, parenttrainingweights))
    else
        bestsplit = find_best_split(treeData,trainingdata,variables,types,method)
        if bestsplit == :NA
            leafnodesstats[1] += 1
            leafnodesstats[2] += 1
            return TreeNode{PredictType,Void}(:LEAF,make_leaf(treeData, method, parenttrainingweights))
        else
            leftrefs,leftweights,leftregressionvalues,lefttimevalues,lefteventvalues,rightrefs,rightweights,rightregressionvalues,righttimevalues,righteventvalues,leftweight = make_split(method,treeData,trainingdata,bestsplit)
            varno, variable, splittype, splitpoint = bestsplit
            if varimp
                if typeof(method.learningType) == Regressor #predictiontask == :REGRESSION
                    variableimp = variance_reduction(treeData.trainingweights,treeData.regressionvalues,leftweights,leftregressionvalues,rightweights,rightregressionvalues)
                elseif typeof(method.learningType) == Classifier #predictiontask == :CLASS
                    variableimp = information_gain(treeData.trainingweights,leftweights,rightweights)
                else #predictiontask == :SURVIVAL
                    variableimp = hazard_score_gain(treeData.trainingweights,treeData.timevalues,treeData.eventvalues,leftweights,lefttimevalues,lefteventvalues,rightweights,righttimevalues,righteventvalues)
                end
                variableimportance[varno] += variableimp
            end
            leftnode=get_tree_node(typeof(treeData)(treeData.depth+1,leftrefs,leftweights,leftregressionvalues,lefttimevalues,lefteventvalues), variableimportance, leafnodesstats, trainingdata,variables,types,method, treeData.trainingweights, varimp, PredictType)
            rightnode=get_tree_node(typeof(treeData)(treeData.depth+1,rightrefs,rightweights,rightregressionvalues,righttimevalues,righteventvalues), variableimportance, leafnodesstats, trainingdata,variables,types,method, treeData.trainingweights, varimp, PredictType)
            return TreeNode{Void,typeof(splitpoint)}(:NODE, varno,splittype,splitpoint,leftweight,
                  leftnode,rightnode)
        end
    end
end

function get_variables_and_types(trainingdata)
    allvariables = names(trainingdata)
    alltypes = eltypes(trainingdata)
    variablechecks = [check_variable(v) for v in allvariables]
    variables = allvariables[variablechecks]
    alltypes = alltypes[variablechecks]
    types::Array{Symbol,1} = [t <: Number ? :NUMERIC : :CATEGORIC for t in alltypes]
    return variables, types
end

function check_variable(v)
    if v in [:CLASS,:REGRESSION,:ID,:FOLD,:TEST,:WEIGHT,:TIME,:EVENT]
        return false
    elseif startswith(string(v),"IGNORE")
        return false
    else
        return true
    end
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
    origclasscounts = map(sum,trainingweights)
    orignoexamples = sum(origclasscounts)
    leftclasscounts = map(sum,leftweights)
    leftnoexamples = sum(leftclasscounts)
    rightclasscounts = map(sum,rightweights)
    rightnoexamples = sum(rightclasscounts)
    return -orignoexamples*entropy(origclasscounts,orignoexamples)+leftnoexamples*entropy(leftclasscounts,leftnoexamples)+rightnoexamples*entropy(rightclasscounts,rightnoexamples)
end

function hazard_score_gain(trainingweights,timevalues,eventvalues,leftweights,lefttimevalues,lefteventvalues,rightweights,righttimevalues,righteventvalues)
    origcumhazardfunction = generate_cumulative_hazard_function(trainingweights,lefttimevalues,lefteventvalues)
    orighazardscore = hazard_score(trainingweights,timevalues,eventvalues,origcumhazardfunction)
    leftcumhazardfunction = generate_cumulative_hazard_function(leftweights,lefttimevalues,lefteventvalues)
    lefthazardscore = hazard_score(leftweights,lefttimevalues,lefteventvalues,leftcumhazardfunction)
    rightcumhazardfunction = generate_cumulative_hazard_function(rightweights,righttimevalues,righteventvalues)
    righthazardscore = hazard_score(rightweights,righttimevalues,righteventvalues,rightcumhazardfunction)
    return orighazardscore-lefthazardscore-righthazardscore
end

##
## Function for making a prediction with a single tree
##
function make_prediction{T,S}(node::TreeNode{T,S},testdata,exampleno,prediction,weight=1.0)
    if node.nodeType == :LEAF
        prediction += weight*node.prediction
        return prediction
    else
        # varno, splittype, splitpoint, splitweight = node[1]
        examplevalue::Nullable{S} = testdata[node.varno][exampleno]
        if isnull(examplevalue)
            prediction = make_prediction(node.leftnode,testdata,exampleno,prediction,weight*node.leftweight)
            prediction = make_prediction(node.rightnode,testdata,exampleno,prediction,weight*(1-node.leftweight))
            return prediction
        else
            if node.splittype == :NUMERIC
              nextnode=(get(examplevalue) <= node.splitpoint)? node.leftnode: node.rightnode
            else #Catagorical
              nextnode=(get(examplevalue) == node.splitpoint)? node.leftnode: node.rightnode
            end
            return make_prediction(nextnode,testdata,exampleno,prediction,weight)
        end
    end
end

function getDfArrayData(da)
    return typeof(da) <: Array || typeof(da) <: SparseVector ? da : useSparseData ? sparsevec(da.data) : da.data
end
"""
To evaluate a method or several methods for generating a random forest:

    julia> evaluate_method(method = forest(...), protocol = <protocol>)
    julia> evaluate_methods(methods = [forest(...), ...], protocol = <protocol>)

The arguments should be on the following format:

    method : a call to forest(...) as explained above (default = forest())
    methods : a list of calls to forest(...) as explained above (default = [forest()])
    protocol : integer, float, :cv or :test as explained above (default = 10)
"""
# AMG: this is for running a single file. Note: we should allow data to be passed as argument in the next
# three functions !!!
function evaluate_method(;method = forest(),protocol = 10)
    println("Running experiment")
    method = fix_method_type(method)
    totaltime = @elapsed results = [run_single_experiment(protocol,[method])]
    classificationresults = [pt == :CLASS for (pt,f,r) in results]
    regressionresults = [pt == :REGRESSION for (pt,f,r) in results]
    survivalresults = [pt == :SURVIVAL for (pt,f,r) in results]
    present_results(sort(results[classificationresults]),[method],ignoredatasetlabel = true)
    present_results(sort(results[regressionresults]),[method],ignoredatasetlabel = true)
    present_results(sort(results[survivalresults]),[method],ignoredatasetlabel = true)
    println("Total time: $(round(totaltime,2)) s.")
    return results[1][3]["results"][1]
end

function evaluate_methods(;methods = [forest()],protocol = 10)
    println("Running experiment")
    methods = map(m->fix_method_type(m),methods)
    totaltime = @elapsed results = [run_single_experiment(protocol,methods)]
    classificationresults = [pt == :CLASS for (pt,f,r) in results]
    regressionresults = [pt == :REGRESSION for (pt,f,r) in results]
    survivalresults = [pt == :SURVIVAL for (pt,f,r) in results]
    present_results(sort(results[classificationresults]),methods,ignoredatasetlabel = true)
    present_results(sort(results[regressionresults]),methods,ignoredatasetlabel = true)
    present_results(sort(results[survivalresults]),methods,ignoredatasetlabel = true)
    println("Total time: $(round(totaltime,2)) s.")
    return results[1][3]
end

function run_single_experiment(protocol, methods)
    predictiontask = prediction_task(globaldata)
    if predictiontask == :NONE
        println("The loaded dataset is not on the correct format")
        println("This may be due to an incorrectly specified separator, e.g., use: separator = \'\\t\'")
        result = (:NONE,:NONE,:NONE)
    else
        if typeof(protocol) == Float64 || protocol == :test
            results = run_split(protocol,methods)
            result = (predictiontask, "",results)
        elseif typeof(protocol) == Int64 || protocol == :cv
            results = run_cross_validation(protocol,methods)
            result = (predictiontask, "",results)
        else
            throw("Unknown experiment protocol")
        end
        println("Completed experiment")
    end
    return result
end

function fix_method_type(method)
    predictiontask = prediction_task(globaldata)
    if typeof(method.learningType) == Undefined # only redefine method if it does not have proper type
        if predictiontask == :REGRESSION
            method = LearningMethod(Regressor(), (getfield(method,i) for i in fieldnames(method)[2:end])...)
        elseif predictiontask == :CLASS
            method = LearningMethod(Classifier(), (getfield(method,i) for i in fieldnames(method)[2:end])...)
        else # predictiontask == :SURVIVAL
            method = LearningMethod(Survival(), (getfield(method,i) for i in fieldnames(method)[2:end])...)
        end
    end
    return method
end

function getnotrees(method, nocoworkers)
    notrees = [div(method.notrees,nocoworkers) for i=1:nocoworkers]
    for i = 1:mod(method.notrees,nocoworkers)
        notrees[i] += 1
    end
    return notrees
end

function getworkertrees(model, nocoworkers)
    notrees = [div(model.method.notrees,nocoworkers) for i=1:nocoworkers]
    for i = 1:mod(model.method.notrees,nocoworkers)
        notrees[i] += 1
    end
    alltrees = Array(Array,nocoworkers)
    index = 0
    for i = 1:nocoworkers
        alltrees[i] = model.trees[index+1:index+notrees[i]]
        index += notrees[i]
    end
    return alltrees
end

function waitfor(var)
    while !all(i->isdefined(var,i), 1:length(var))
        sleep(0.005)
    end
end

"""
To generate a model from the loaded dataset:

    julia> m = generate_model(method = forest(...))                         

The argument should be on the following format:

    method : a call to forest(...) as explained above (default = forest())
"""
function generate_model(;method = forest())
    predictiontask = prediction_task(globaldata)
    if predictiontask == :NONE
        println("The loaded dataset is not on the correct format: CLASS/REGRESSION column missing")
        println("This may be due to an incorrectly specified separator, e.g., use: separator = \'\\t\'")
        result = :NONE
    else
        method = fix_method_type(method)
        classes = typeof(method.learningType) == Classifier ? getDfArrayData(unique(globaldata[:CLASS])) : Int[]
        nocoworkers = nprocs()-1
        numThreads = Threads.nthreads()
        if nocoworkers > 0
            notrees = getnotrees(method, nocoworkers)
            treesandoobs = pmap(generate_trees, [(method,classes,n,rand(1:1000_000_000)) for n in notrees])
        elseif numThreads > 1
            notrees = getnotrees(method, numThreads)
            treesandoobs = Array{Any,1}(length(notrees))
            Threads.@threads for n in notrees
                treesandoobs[Threads.threadid()] = generate_trees((method,classes,n,rand(1:1000_000_000)))
            end
            waitfor(treesandoobs)
        else
            notrees = [method.notrees]
            treesandoobs = generate_trees.([(method,classes,n,rand(1:1000_000_000)) for n in notrees])
        end
        trees = map(i->i[1], treesandoobs)
        oobs = map(i->i[2], treesandoobs)
        variableimportance = treesandoobs[1][3]
        for i = 2:length(treesandoobs)
            variableimportance += treesandoobs[i][3]
        end
        variableimportance = variableimportance/method.notrees
        variables, types = get_variables_and_types(globaldata)
        variableimportance = hcat(variables,variableimportance)
        oobperformance, conformalfunction = generate_model_internal(method, oobs, classes)
        result = PredictionModel{typeof(method.learningType)}(method,classes,rf_ver,oobperformance,variableimportance,vcat(trees...),conformalfunction)
    end
    return result
end

"""
To store a model in a file:

    julia> store_model(<model>, <file>)                              

The arguments should be on the following format:

    model : a generated or loaded model (see generate_model and load_model)
    file : name of file to store model in
"""
function store_model(model::PredictionModel,file)
    s = open(file,"w")
    serialize(s,model)
    close(s)
    println("Model stored")
end
"""
To load a model from file:

    julia> rf = load_model(<file>)                                  

The argument should be on the following format:

    file : name of file in which a model has been stored
"""
function load_model(file)
    s = open(file,"r")
    model = deserialize(s)
    close(s)
    println("Model loaded")
    return model
end

#=
Infers the prediction task from the data
=#
function prediction_task(method::LearningMethod{Regressor})
    return :REGRESSION
end

function prediction_task(method::LearningMethod{Classifier})
    return :CLASS
end

function prediction_task(method::LearningMethod{Survival})
    return :SURVIVAL
end

function prediction_task(data)
    allnames = names(data)
    if :CLASS in allnames
        return :CLASS
    elseif :REGRESSION in allnames
        return :REGRESSION
    elseif :TIME in allnames && :EVENT in allnames
        return :SURVIVAL
    else
        return :NONE
    end
end

function initiate_workers()
    pr = Array(Future,nprocs())
    for i = 2:nprocs()
        pr[i] = remotecall(load_global_dataset,i)
    end
    for i = 2:nprocs()
        wait(pr[i])
    end
end

function load_global_dataset()
    global globaldata = @fetchfrom(1,globaldata)
end

"""
To apply a model to loaded data:

    julia> apply_model(<model>, confidence = <confidence>)

The argument should be on the following format:

    model : a generated or loaded model (see generate_model and load_model)
    confidence : a float between 0 and 1 or :std (default = :std)
                 - probability of including the correct label in the prediction region
                 - :std means employing the same confidence level as used during training
"""
function apply_model(model::PredictionModel; confidence = :std)
    apply_model_internal(model, confidence=confidence)
end

function update_workers()
    pr = Array(Future,nprocs())
    for i = 2:nprocs()
        pr[i] = remotecall(update_global_dataset,i)
    end
    for i = 2:nprocs()
        wait(pr[i])
    end
end

function update_global_dataset()
    global globaltests = @fetchfrom(1,globaltests)
    global globaldata = hcat(globaltests,globaldata)
end

            
end
