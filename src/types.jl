## Type declarations

abstract LearningType

type Undefined <: LearningType
end

type Classifier <: LearningType
end

type Regressor <: LearningType
end

type LearningMethod{T<:LearningType}
    learningType::T
    modeltype::Any
    notrees::Int
    minleaf::Int
    maxdepth::Int
    randsub::Any
    randval::Bool
    splitsample::Int
    bagging::Bool
    bagsize::Number
    modpred::Bool
    laplace::Bool
    confidence::Float64
    conformal::Any
end



function tree(;minleaf = 5, maxdepth = 0, randsub = :all, randval = false,
              splitsample = 0, bagging = false, bagsize = 1.0, modpred = false, laplace = true, confidence = 0.95, conformal = :default)
    return LearningMethod(Undefined(),:tree,1,minleaf,maxdepth,randsub,randval,splitsample,bagging,bagsize,modpred,laplace,confidence,conformal)
end

function forest(;minleaf = 1, maxdepth = 0, randsub = :default, randval = true,
                splitsample = 0, bagging = true, bagsize = 1.0, modpred = false, laplace = false, confidence = 0.95, conformal = :default, notrees = 100)
    return LearningMethod(Undefined(),:forest,notrees,minleaf,maxdepth,randsub,randval,splitsample,bagging,bagsize,modpred,laplace,confidence,conformal)
end

function treeClassifier(;minleaf = 5, maxdepth = 0, randsub = :all, randval = false,
              splitsample = 0, bagging = false, bagsize = 1.0, modpred = false, laplace = true, confidence = 0.95, conformal = :default)
    return LearningMethod(Classifier(), :tree,1,minleaf,maxdepth,randsub,randval,splitsample,bagging,bagsize,modpred,laplace,confidence,conformal)
end

function treeRegressor(;minleaf = 5, maxdepth = 0, randsub = :all, randval = false,
              splitsample = 0, bagging = false, bagsize = 1.0, modpred = false, laplace = true, confidence = 0.95, conformal = :default)
    return LearningMethod(Regressor(), :tree,1,minleaf,maxdepth,randsub,randval,splitsample,bagging,bagsize,modpred,laplace,confidence,conformal)
end

function forestClassifier(;minleaf = 1, maxdepth = 0, randsub = :default, randval = true,
                splitsample = 0, bagging = true, bagsize = 1.0, modpred = false, laplace = false, confidence = 0.95, conformal = :default, notrees = 100)
    return LearningMethod(Classifier(), :forest,notrees,minleaf,maxdepth,randsub,randval,splitsample,bagging,bagsize,modpred,laplace,confidence,conformal)
end

function forestRegressor(;minleaf = 1, maxdepth = 0, randsub = :default, randval = true,
                splitsample = 0, bagging = true, bagsize = 1.0, modpred = false, laplace = false, confidence = 0.95, conformal = :default, notrees = 100)
    return LearningMethod(Regressor(), :forest,notrees,minleaf,maxdepth,randsub,randval,splitsample,bagging,bagsize,modpred,laplace,confidence,conformal)
end

type RegressionResult
    MSE::Float64
    Corr::Float64
    AvMSE::Float64
    VarMSE::Float64
    DEOMSE::Float64
    AEEMSE::Float64
    Valid::Float64
    Region::Float64
    Size::Float64
    NoIrr::Float64
    Time::Float64
end

type ClassificationResult
    Acc::Float64
    AUC::Float64
    Brier::Float64
    AvAcc::Float64
    DEOAcc::Float64
    AEEAcc::Float64
    AvBrier::Float64
    VBrier::Float64
    Margin::Float64
    Prob::Float64
    Valid::Float64
    Region::Float64
    OneC::Float64
    Size::Float64
    NoIrr::Float64
    Time::Float64
end

type PredictionModel
    predictiontask::Any
    classes::Any
    version::Any
    method::LearningMethod
    oobperformance::Any
    variableimportance::Any
    trees::Any
    conformal::Any
end
