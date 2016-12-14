RandomForest v. 0.0.10
======================

Copyright 2016 Henrik Boström

A `Julia <http://julialang.org>`_ package that implements random forests for classification, regression and survival analysis with conformal prediction.
[NOTE: survival analysis under development]

There are two basic ways of working with the package:

- running an experiment with multiple datasets, possibly comparing multiple methods,
  i.e., random forests with different parameter settings, or

- working with a single dataset, to evaluate, generate, store, load or
  apply a random forest

All named arguments below are optional, while the others are mandatory.

The classification datasets included in uci.zip have been downloaded and adapted from:

Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

The regression datasets included in regression.zip have been downloaded and adapted from the above source and from:

Rasmussen,  C.E.,  Neal,  R.M.,  Hinton,  G.,  van  Camp,  D.,  Revow,  M.,  Ghahramani, Z., Kustra, R., and Tibshirani, R. (1996) Delve data for evaluating learning in valid experiments [http://www.cs.toronto.edu/~delve/data/datasets.html]

The survival datasets included in survival.zip have been downloaded and adapted from:

Statistical Software Information, University of Massachusetts Amherst, Index of Survival Analysis Datasets,
[https://www.umass.edu/statdata/statdata/stat-survival.html]