using RandomForest, Base.Test, FactCheck

facts("*** Run Survival Test ***") do
  @fact_throws experiment(files=["../survival/pbc.txt"]) "everything is alright"
end
