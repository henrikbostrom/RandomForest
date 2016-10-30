using RandomForest, FactCheck, Requests, DataFrames

eta = 0.05;
facts("*** Run Classifcation Test ***") do

  context("Function: experiment") do
    @fact experiment(files=[Requests.get_streaming("https://raw.githubusercontent.com/henrikbostrom/RandomForest/testing/testData/glass.txt")], printable = false)[1][3][1].Acc --> roughly(0.79; atol = eta)
    @fact experiment(files=[Requests.get_streaming("https://raw.githubusercontent.com/henrikbostrom/RandomForest/testing/testData/autos.txt")], printable = false)[1][3][1].Acc --> roughly(0.83; atol = eta)
  end

  context("Function: generate_model") do
    load_data(readtable(Requests.get_streaming("https://raw.githubusercontent.com/henrikbostrom/RandomForest/testing/testData/glass.txt")));
    @fact generate_model().oobperformance --> roughly(0.79; atol = eta)

    load_data(readtable(Requests.get_streaming("https://raw.githubusercontent.com/henrikbostrom/RandomForest/testing/testData/autos.txt")));
    @fact generate_model().oobperformance --> roughly(0.83; atol = eta)
  end

end
