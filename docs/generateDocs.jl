using RandomForest

myFunctions = names(RandomForest)
dict =  Dict()

for myFunc in myFunctions
    try
        p = methods(eval(myFunc))
        if (length(p) > 0) 
            # function may be in more than one file 
            file_symbol = p.ms[1].file
            file_str = string(file_symbol)
            fileName = split(file_str,"/")
            fileName = fileName[length(fileName)]
            md = Base.doc(eval(myFunc))
            if (haskey(dict,fileName)) 
                push!(dict[fileName],md)  
            else
                dict[fileName] = [md]
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
        md_list = dict[fileName]

        println(f,".. _$fname:")
        println(f)
        println(f,"$fname")
        println(f,"----------------------------------------------------")
        println(f)
        println(f, ".. DO NOT EDIT: this file is generated from Julia source.")
        println(f)
            
        for md in md_list
            if (isa(md,Markdown.MD))

                if (!isa(md.content[1],Markdown.Paragraph))
                    println(md.content[1], " " , fname)
                    println(f,Markdown.rst(md.content[1]))
                end
            end    

        end
        close(f)
    end 

end