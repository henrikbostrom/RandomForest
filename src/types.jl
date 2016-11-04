## Type declarations

immutable TreeData{T1, T2}
    depth::Int
    trainingrefs::Array{T1,1}
    trainingweights::Array{T2,1}
end

immutable TreeNode{T,ST}
   nodeType::Symbol
   prediction::T
   variable::Symbol
   splittype::Symbol
   splitpoint::ST # Int for CATEGORIC and Float64 NUMERIC
   leftweight::Float64
   leftnode::TreeNode
   rightnode::TreeNode
   TreeNode(n,p)=new(n,p)
   TreeNode(n,v,splt,spltp,lw,left,right)=new(n,T(),v,splt,spltp,lw,left,right)
end

type NodeSplit
    splitvalue::Float64
    varno::Int
    splittype::Symbol
    splitpoint::Float64
end

abstract LearningType

type Undefined <: LearningType
end

type Classifier <: LearningType
end

type Regressor <: LearningType
end

type Survival <: LearningType
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

function tree(;learningType = Undefined(), minleaf = 5, maxdepth = 0, randsub = :all, randval = false,
              splitsample = 0, bagging = false, bagsize = 1.0, modpred = false, laplace = true, confidence = 0.95, conformal = :default)
    return LearningMethod(learningType,:tree,1,minleaf,maxdepth,randsub,randval,splitsample,bagging,bagsize,modpred,laplace,confidence,conformal)
end

function forest(;learningType = Undefined(), minleaf = 1, maxdepth = 0, randsub = :default, randval = true,
                splitsample = 0, bagging = true, bagsize = 1.0, modpred = false, laplace = false, confidence = 0.95, conformal = :default, notrees = 100)
    return LearningMethod(learningType,:forest,notrees,minleaf,maxdepth,randsub,randval,splitsample,bagging,bagsize,modpred,laplace,confidence,conformal)
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
    OneAcc::Float64
    Size::Float64
    NoIrr::Float64
    Time::Float64
end

type SurvivalResult
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

type PredictionModel{T}
    method::LearningMethod{T}
    classes::Any
    version::VersionNumber
    oobperformance::Any
    variableimportance::Any
    trees::Array{TreeNode,1}
    conformal::Any
    PredictionModel(m) = new(m)
    PredictionModel(method,class,ver,oob,varImp,trees,conformal) = new(method,class,ver,oob,varImp,trees,conformal)
end

function treeClassifier(;minleaf = 5, maxdepth = 0, randsub = :all, randval = false,
              splitsample = 0, bagging = false, bagsize = 1.0, modpred = false, laplace = true, confidence = 0.95, conformal = :default)
    method = LearningMethod(Classifier(), :tree,1,minleaf,maxdepth,randsub,randval,splitsample,bagging,bagsize,modpred,laplace,confidence,conformal)
    return PredictionModel{Classifier}(method)
end

function treeRegressor(;minleaf = 5, maxdepth = 0, randsub = :all, randval = false,
              splitsample = 0, bagging = false, bagsize = 1.0, modpred = false, laplace = true, confidence = 0.95, conformal = :default)
    method = LearningMethod(Regressor(), :tree,1,minleaf,maxdepth,randsub,randval,splitsample,bagging,bagsize,modpred,laplace,confidence,conformal)
    return PredictionModel{Regressor}(method)
end

function treeSurvival(;minleaf = 5, maxdepth = 0, randsub = :all, randval = false,
              splitsample = 0, bagging = false, bagsize = 1.0, modpred = false, laplace = true, confidence = 0.95, conformal = :default)
    method = LearningMethod(Survival(), :tree,1,minleaf,maxdepth,randsub,randval,splitsample,bagging,bagsize,modpred,laplace,confidence,conformal)
    return PredictionModel{Survival}(method)
end

function forestClassifier(;minleaf = 1, maxdepth = 0, randsub = :default, randval = true,
                splitsample = 0, bagging = true, bagsize = 1.0, modpred = false, laplace = false, confidence = 0.95, conformal = :default, notrees = 100)
    method = LearningMethod(Classifier(), :forest,notrees,minleaf,maxdepth,randsub,randval,splitsample,bagging,bagsize,modpred,laplace,confidence,conformal)
    return PredictionModel{Classifier}(method)
end

function forestRegressor(;minleaf = 1, maxdepth = 0, randsub = :default, randval = true,
                splitsample = 0, bagging = true, bagsize = 1.0, modpred = false, laplace = false, confidence = 0.95, conformal = :default, notrees = 100)
    method = LearningMethod(Regressor(), :forest,notrees,minleaf,maxdepth,randsub,randval,splitsample,bagging,bagsize,modpred,laplace,confidence,conformal)
    return PredictionModel{Regressor}(method)
end

function forestSurvival(;minleaf = 1, maxdepth = 0, randsub = :default, randval = true,
                splitsample = 0, bagging = true, bagsize = 1.0, modpred = false, laplace = false, confidence = 0.95, conformal = :default, notrees = 100)
    return LearningMethod(Survival(), :forest,notrees,minleaf,maxdepth,randsub,randval,splitsample,bagging,bagsize,modpred,laplace,confidence,conformal)
    return PredictionModel{Survival}(method)
end
