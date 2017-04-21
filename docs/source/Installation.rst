To install the package
========

Start Julia (which can be downloaded from http://julialang.org/) at a command prompt:

    julia

at the Julia REPL, give the following command to clone the package from GitHub:

    julia> Pkg.clone("https://github.com/henrikbostrom/RandomForest.git")

Unless you do not already have it installed, install also the DataFrames package:

    julia> Pkg.add("DataFrames")
    
Load the RandomForest package:

    julia> using RandomForest

