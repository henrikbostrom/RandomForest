include("benchhelpers.jl")

funcs = [:test_exp]#, :test_gen];

files = ["glass", "laser", "plastic", "pbc", "pharynx"];

f = open("timing.txt", "w");
for func in funcs
  write(f, "*** $(func) *** \n")
  write(f, join(["fileName", "type", "elapsedTime"], "\t \t") * "\n")
  for file in files
    els = @elapsed fileType = eval(funcs[1])("$(file).txt")
    write(f, join([file, fileType, els], "\t \t") * "\n")
  end
end
close(f)
