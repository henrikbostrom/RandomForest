include("benchhelpers.jl")

funcs = [:test_gen]; #[:test_exp, 

files = ["glass.txt", "laser.txt", "plastic.txt", "pbc.csv", "pharynx.csv"];

f = open("timing.txt", "w");
for func in funcs
  write(f, "*** $(func) *** \n")
  write(f, join(["fileName", "type", "elapsedTime"], "\t \t") * "\n")
  for file in files
    els = @elapsed fileType = eval(func)("$(file)")
    write(f, join([file, fileType, els], "\t \t") * "\n")
  end
end
close(f)
