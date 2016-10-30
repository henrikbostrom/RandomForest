using RandomForest
using FactCheck
using Requests
using DataFrames

tests = [
    "classification.jl",
    "regression.jl",
    "survival.jl"
]

println("Running tests:")

for test in tests
    include(test)
end
