using RandomForest

# functions are sorted in alphabetical order
myFunctions = names(RandomForest)

# not ordered. but still not in the exported order
# myFunctions = ccall(:jl_module_names, Array{Symbol,1}, (Any,Cint,Cint), RandomForest, false, false)
dict_headings =  Dict()
dict_headings["RandomForest.rst"] = "Functions for running experiments"
dict_headings["common.rst"] = "Functions for working with a single dataset"
dict_headings["print.rst"] = "Functions for printing"

dict =  Dict()


for myFunc in myFunctions
    try
        # println(myFunc)
        p = methods(eval(myFunc))
        if (length(p) > 0) 
            # function may be in more than one file 
            file_symbol = p.ms[1].file
            file_str = string(file_symbol)
            fileName = split(file_str,"/")
            fileName = fileName[length(fileName)]
            md = Base.doc(eval(myFunc))
            if (haskey(dict,fileName)) 
                push!(dict[fileName],(md ,myFunc))  
            else
                dict[fileName] = [(md ,myFunc)]
            end 
        end 
    catch
        warn("Can't find method: '$myFunc'")
    end
end 



cd(joinpath(dirname(@__FILE__),"source")) do

    for fileName in  keys(dict)
        fname = replace(fileName, ".jl", ".rst")
        f = open(fname,"w")
        list = dict[fileName]

        if (haskey(dict_headings,fname)) 
            fname = dict_headings[fname]
        else 
            fname = replace(fileName, ".rst", "")
        end

        println(f,".. _$fname:")
        println(f)
        println(f,"$fname")
        println(f,"==============================================================")
        println(f)
        println(f, ".. DO NOT EDIT: this file is generated from Julia source.")
        println(f)
            
        for x in list
            md = x[1]
            myFunc = x[2]
            if (isa(md,Markdown.MD))

                if (!isa(md.content[1],Markdown.Paragraph))
                    # println(f,".. function:: $myFunc \n")
                    println(f,"$myFunc \n^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
                    println(f,Markdown.rst(md.content[1]))
                    println(f,"\n---------\n")
                end
            end    

        end
        close(f)
    end 

end
