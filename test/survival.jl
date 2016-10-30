using RandomForest, Base.Test, FactCheck

facts("*** Run Survival Test ***") do

  context("Function: experiment") do
    @fact experiment(files=["./survival/pbc.csv"], printable = false)[1][3][1].MSE --> roughly(0.25; atol = 0.05)
    @fact experiment(files=["./survival/pharynx.csv"], printable = false)[1][3][1].MSE --> roughly(0.44; atol = 0.05)
  end

  context("Function: generate_model") do
    load_data("./survival/pbc.csv");
    @fact generate_model().oobperformance --> roughly(0.25; atol = 0.05)
    load_data("./survival/pharynx.csv");
    @fact generate_model().oobperformance --> roughly(0.44; atol = 0.05)
  end

end
