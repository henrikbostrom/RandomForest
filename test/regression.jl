using RandomForest, FactCheck, Requests, DataFrames

percent = 0.1
facts("*** Run Regression Test ***") do

  context("Function: experiment") do
    @fact experiment(files=[Requests.get_streaming("https://raw.githubusercontent.com/henrikbostrom/RandomForest/testing/testData/laser.txt")], printable = false)[1][3][1].MSE --> less_than(72.5 + 72.5 * percent)
    @fact experiment(files=[Requests.get_streaming("https://raw.githubusercontent.com/henrikbostrom/RandomForest/testing/testData/plastic.txt")], printable = false)[1][3][1].MSE --> less_than(3.0 + 3.0 * percent)
  end

    context("Function: generate_model") do
        d=readtable(Requests.get_streaming("https://raw.githubusercontent.com/henrikbostrom/RandomForest/testing/testData/laser.txt"));
        load_data(d);
        @fact generate_model().oobperformance --> less_than(72.5 + 72.5 * percent)

        d=readtable(Requests.get_streaming("https://raw.githubusercontent.com/henrikbostrom/RandomForest/testing/testData/plastic.txt"));
        load_data(d);
        @fact generate_model().oobperformance --> less_than(3.0 + 3.0 * percent)
    end

end
