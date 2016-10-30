using RandomForest, FactCheck, Requests, DataFrames
include("testhelpers.jl")

eta = 0.05;

facts("*** Run Classifcation Test ***") do


  context("Function: experiment") do
    file="glass.txt"
    @fact test_exp(file)[1][3][1].Acc --> roughly(0.79; atol = eta)
    file="autos.txt"
    @fact test_exp(file)[1][3][1].Acc --> roughly(0.83; atol = eta)
  end

  context("Function: generate_model") do
    file="glass.txt"
    @fact test_gen(file).oobperformance --> roughly(0.79; atol = eta)
    file="autos.txt"
    @fact test_gen(file).oobperformance --> roughly(0.83; atol = eta)
  end

end
