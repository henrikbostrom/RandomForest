isdefined(:helperloaded) || include("testhelpers.jl")

percent = 0.1
facts("*** Run Survival Test ***") do

  context("Function: experiment") do
    @fact test_exp("pbc.csv")[1][3][1].MSE --> less_than(0.25 + 0.25 * percent)
    @fact test_exp("pharynx.csv")[1][3][1].MSE --> less_than(0.44 + 0.44 * percent)
  end

  context("Function: generate_model") do
    @fact test_gen("pbc.csv").oobperformance --> less_than(0.25 + 0.25 * percent)
    @fact test_gen("pharynx.csv").oobperformance --> less_than(0.44 + 0.44 * percent)
  end

end
