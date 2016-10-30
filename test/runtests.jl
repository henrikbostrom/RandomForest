using RandomForest
using Base.Test
using FactCheck

tests = [
    "classification.jl",
    "regression.jl",
    "survival.jl"
]

println("Running tests:")

for test in tests
    include(test)
end
