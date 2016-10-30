using RandomForest, Base.Test, FactCheck

eta = 0.05;
facts("*** Run Classifcation Test ***") do

  context("Function: experiment") do
    @fact experiment(files=["../uci/glass.txt"], printable = false)[1][3][1].Acc --> roughly(0.79; atol = eta)
    @fact experiment(files=["../uci/autos.txt"], printable = false)[1][3][1].Acc --> roughly(0.83; atol = eta)
  end

  context("Function: generate_model") do
    load_data("../uci/glass.txt");
    @fact generate_model().oobperformance --> roughly(0.79; atol = eta)
    load_data("../uci/autos.txt");
    @fact generate_model().oobperformance --> roughly(0.83; atol = eta)
  end

end
