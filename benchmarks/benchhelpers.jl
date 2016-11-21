using RandomForest, FactCheck, Requests, DataFrames

baseurl="https://raw.githubusercontent.com/henrikbostrom/RandomForest/testing/testData/"

function test_exp(file)
  println("Testing_exp: $file")
  experiment(files=[Requests.get_streaming(baseurl*file)], printable = false)[1][1]
end

function test_gen(file)
  println("Testing_gen: $file")
  load_data(readtable(Requests.get_streaming(baseurl*file)))
  generate_model()
  return "Prdection Model"
end
