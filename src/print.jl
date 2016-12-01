@doc """
       Functions for outputting result summaries
       """ ->

function present_results(results,methods;ignoredatasetlabel = false)
    if results != []
        if results[1][1] == :CLASS # NOTE: Assuming all tasks of the same type
            resultlabels = fieldnames(ClassificationResult) #FIXME: MOH Can be extracted from the model information
        elseif results[1][1] == :REGRESSION
            resultlabels = fieldnames(RegressionResult)
        else # results[1][1] == :SURVIVAL
            resultlabels = fieldnames(SurvivalResult)
        end
        methodresults = Array(Float64,length(results),length(methods),length(resultlabels))
        for datasetno = 1:length(results)
            for methodno = 1:length(methods)
                for resultno = 1:length(resultlabels)
                    methodresults[datasetno,methodno,resultno] = getfield(results[datasetno][3][methodno], resultlabels[resultno])
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
        elseif results[1][1] == :REGRESSION
            println("\nRegression results")
        else # results[1][1] == :SURVIVAL
            println("\nSurvival results")
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
        println("\t$(n) = $(getfield(method,n))")
    end
end

function describe_data()
    println("No. of examples: $(size(globaldata,1))")
    println("No. of columns: $(size(globaldata,2)-1)")
    println("")
    println("Columns:")
    describe(globaldata)
end

function describe_model(model::PredictionModel)
    method = model.method
    println("Generated by: RandomForest v. $rf_ver")
    if typeof(method.learningType) == Classifier # model.predictiontask == :CLASS
        println("Prediction task: classification")
        println("Class labels: $(model.classes)")
    elseif typeof(method.learningType) == Regressor # model.predictiontask == :REGRESSION
        println("Prediction task: regression")
    else # typeof(method.learningType) == Survival # model.predictiontask == :SURVIVAL
        println("Prediction task: survival")
    end
    println("Learning method:")
    present_method(0,model.method,showmethodlabel = false)
    if typeof(method.learningType) == Classifier # model.predictiontask == :CLASS
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
