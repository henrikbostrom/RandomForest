using RandomForest, Base.Test, FactCheck

percent = 0.1
facts("*** Run Regression Test ***") do

  context("Function: experiment") do
    @fact experiment(files=["./regression/laser.txt"], printable = false)[1][3][1].MSE --> less_than(72.5 + 72.5 * percent)
    @fact experiment(files=["./regression/plastic.txt"], printable = false)[1][3][1].MSE --> less_than(3.0 + 3.0 * percent)
  end

  context("Function: generate_model") do
    load_data("./regression/laser.txt");
    @fact generate_model().oobperformance --> less_than(72.5 + 72.5 * percent)
    load_data("./regression/plastic.txt");
    @fact generate_model().oobperformance --> less_than(3.0 + 3.0 * percent)
  end

end
