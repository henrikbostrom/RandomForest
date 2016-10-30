using RandomForest, Base.Test, FactCheck

facts("*** Run Regression Test ***") do

  context("Function: experiment") do
    @fact experiment(files=["../regression/laser.txt"], printable = false)[1][3][1].MSE --> roughly(72.5; atol = 10.0)
    @fact experiment(files=["../regression/plastic.txt"], printable = false)[1][3][1].MSE --> roughly(3.0; atol = 0.5)
  end

  context("Function: generate_model") do
    load_data("../regression/laser.txt");
    @fact generate_model().oobperformance --> roughly(72.5; atol = 10.0)
    load_data("../regression/plastic.txt");
    @fact generate_model().oobperformance --> roughly(3.0; atol = 0.5)
  end

end
