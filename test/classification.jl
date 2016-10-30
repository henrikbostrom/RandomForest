isdefined(:helperloaded) || include("testhelpers.jl")

eta = 0.1;
facts("*** Run Classifcation Test ***") do

  context("Function: experiment") do
    @fact test_exp("glass.txt")[1][3][1].Acc --> greater_than(0.79 - eta)
    @fact test_exp("autos.txt")[1][3][1].Acc --> greater_than(0.83 - eta)
  end

  context("Function: generate_model") do
    @fact test_gen("glass.txt").oobperformance --> greater_than(0.79 - eta)
    @fact test_gen("autos.txt").oobperformance --> greater_than(0.83 - eta)
  end

end
