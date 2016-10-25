using RandomForest, Base.Test, FactCheck

facts("*** Run Classifcation Test ***") do
  @fact_throws experiment(files=["../uci/glass.txt"]) "every thing is alright"
end
