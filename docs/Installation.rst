To install the package
========

Clone or download the package to some suitable directory.

From this directory, start Julia (which can be downloaded from http://julialang.org/) at a command prompt:

    julia

Install a requested package:

    julia> Pkg.add("DataFrames")

Try to load the RandomForest package (assuming that the current directory is in your load path,
e.g., add "push!(LOAD_PATH, pwd())" to the file ".juliarc.jl" in your home directory):

    julia> using RandomForest

Then exit by:

    julia> exit()