baseurl="https://raw.githubusercontent.com/henrikbostrom/RandomForest/testing/testData/"
function test_exp(file)
  println("Testing: $file")
  experiment(files=[Requests.get_streaming(baseurl*file)], printable = false)
end

function test_gen(file)
  println("Testing: $file")
    d=readtable(Requests.get_streaming(baseurl*file))
    load_data(d)
    generate_model()
end
