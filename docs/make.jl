using Documenter, RandomForest

makedocs(doctest=false)

deploydocs(deps=Deps.pip("pygments", "mkdocs", "mkdocs-material"),
           repo="github.com/henrikbostrom/RandomForest.git", #TODO repo name could change
           julia="0.4",
           osname="linux")
