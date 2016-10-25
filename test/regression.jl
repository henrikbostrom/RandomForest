using RandomForest, Base.Test, FactCheck

facts("*** Run Regression Test ***") do
  @fact_throws experiment(files=["../regression/stock.txt"]) "every thing is alright"
end
