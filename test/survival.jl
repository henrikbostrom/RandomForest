using RandomForest, FactCheck, Requests, DataFrames

percent = 0.1
facts("*** Run Survival Test ***") do

  context("Function: experiment") do
    @fact experiment(files=[Requests.get_streaming("https://raw.githubusercontent.com/henrikbostrom/RandomForest/testing/testData/pbc.csv")], printable = false)[1][3][1].MSE --> less_than(0.25 + 0.25 * percent)
    @fact experiment(files=[Requests.get_streaming("https://raw.githubusercontent.com/henrikbostrom/RandomForest/testing/testData/pharynx.csv")], printable = false)[1][3][1].MSE --> less_than(0.44 + 0.44 * percent)
  end

  context("Function: generate_model") do

        d=readtable(Requests.get_streaming("https://raw.githubusercontent.com/henrikbostrom/RandomForest/testing/testData/pbc.csv"));
        load_data(d);
        @fact generate_model().oobperformance --> less_than(0.25 + 0.25 * percent)

        d=readtable(Requests.get_streaming("https://raw.githubusercontent.com/henrikbostrom/RandomForest/testing/testData/pharynx.csv"));
        load_data(d);
        @fact generate_model().oobperformance --> less_than(0.44 + 0.44 * percent)

  end

end
