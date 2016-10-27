using RandomForest, Base.Test, FactCheck

facts("*** Run Regression Test ***") do
  @fact_throws experiment(files=["../regression/stock.txt"]) "everything is alright"
end
