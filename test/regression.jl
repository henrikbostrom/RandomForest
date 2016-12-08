isdefined(:helperloaded) || include("testhelpers.jl")

percent = 0.1
facts("*** Run Regression Test ***") do

  context("Function: experiment") do
    @fact test_exp("laser.txt")[1][3][1].MSE --> less_than(72.5 + 72.5 * percent)
    @fact test_exp("plastic.txt")[1][3][1].MSE --> less_than(3.0 + 3.0 * percent)
  end

  context("Function: generate_model") do
    @fact test_gen("laser.txt").oobperformance --> less_than(72.5 + 72.5 * percent)
    @fact test_gen("plastic.txt").oobperformance --> less_than(3.0 + 3.0 * percent)
  end

end
