.. _To run an experiment:

To run an experiment
==============================================================

.. DO NOT EDIT: this file is generated from Julia source.

experiment 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
An experiment is run by calling experiment(...) in the following way:

.. code-block:: julia

    julia> experiment(files = <files>, separator = <separator>, protocol = <protocol>,
                      normalizetarget = <normalizetarget>, normalizeinput = <normalizeinput>,
                      methods = [<method>, ...])

The arguments should be on the following format:

.. code-block:: julia

    files : list of file names or path to directory (default = ".")
        - in a specified directory, files with extensions other than .txt and .csv are ignored
        - each file should contain a dataset (see format requirements below)
        - example: files = ["uci/house-votes.txt", "uci/glass.txt"],
        - example: files = "uci"
    
    separator : single character (default = ',')
        - the character to use as field separator
        - example: separator = '	' (the tab character)
    
    protocol : integer, float, :cv, :test (default = 10)
        - the experiment protocol:
            an integer means using cross-validation with this no. of folds.
            a float between 0 and 1 means using this fraction of the dataset for testing
            :cv means using cross-validation with folds specified by a column labeled FOLD
            :test means dividing the data into training and test according to boolean values
             in a column labeled TEST (true means that the example is used for testing)
        - example: protocol = 0.25 (25% of the data is for testing)
    
    normalizetarget : boolean (default = false)
        - true means that each regression value v will be replaced by
          (v-v_min)/(v_max-v_min), where v_min and V-max are the minimum and maximum values
        - false means that the original regression values are kept
    
    normalizeinput : boolean (default = false)
        - true means that each numeric input value v will be replaced by
          (v-v_min)/(v_max-v_min), where v_min and V-max are the minimum and maximum values
        - false means that the original values are kept
    
    method : a call on the form forest(...) (default = forest())
    
        [FIXME: tree() method is missing here]: #
        - The call may have the following (optional) arguments:
    
            notrees : integer (default = 100)
                - no. of trees to generate in the forest
    
            minleaf : integer (default = 1)
                - minimum no. of required examples to form a leaf
    
            maxdepth : integer (default = 0)
                - maximum depth of generated trees (0 means that there is no depth limit)
    
            randsub : integer, float, :default, :all, :log2, or :sqrt (default = :default)
                - no. of randomly selected features to evaluate at each split:
                   :default means :log2 for classification and 1/3 for regression
                   :all means that all features are used (no feature subsampling takes place)
                   :log2 means that log2 of the no. of features are sampled
                   :sqrt means that sqrt of the no. of features are sampled
                   an integer (larger than 0) means that this number of features are sampled
                   a float (between 0.0 and 1.0) means that this fraction of features are sampled
    
            randval : boolean (default = true)
                - true means that a single randomly selected value is used to form conditions for each
                  feature in each split
                - false mean that all values are used to form conditions when evaluating features for
                  each split
    
            splitsample : integer (default = 0)
                - no. of randomly selected examples to use for evaluating each split
                - 0 means that no subsampling of the examples will take place
    
            bagging : boolean (default = true)
                - true means that a bootstrap replicate of the training examples is used for each tree
                - false means that the original training examples are used when building each tree
    
            bagsize : float or integer (default = 1.0)
                - no. of randomly selected examples to include in the bootstrap replicate
                - an integer means that this number of examples are sampled with replacement
                - a float means that the corresponding fraction of examples are sampled with replacement
    
            modpred : boolean (default = false)
                - true means that for each test instance, the trees for which a randomly selected training
                  instance is out-of-bag is used for prediction and the training instance is not used for
                  calculating a calibration score
                - false means that all trees in the forest are used for prediction and all out-of-bag scores
                  are used for calibration
    
            laplace : boolean (default = false)
                - true means that class probabilities at each leaf node is Laplace corrected
                - false means that class probabilities at each leaf node equal the relative class
                  frequencies
    
            confidence : a float between 0 and 1 (default = 0.95)
                - probability of including the correct label in the prediction region
    
            conformal : :default, :std, :normalized or :classcond (default = :default)
                - method used to calculate prediction regions
                - For classification, the following options are allowed:
                   :default is the same as :std
                   :std means that validity is guaranteed in general, but not for each class
                   :classcond means that validity is guaranteed for each class
                - For regression, the following options are allowed:
                   :default is the same as :normalized
                   :std results in the same region size for all predictions
                   :normalized means that each region size is dependent on the spread
                    of predictions among the individual trees

–––––

.. code-block:: julia

    Examples:
    
    The call experiment(files = "uci") is hence the same as
    
    experiment(files = "uci", separator = ´,´, protocol = 10, methods = [forest()])
    
    The following compares the default random forest to one with 1000 trees and a maxdepth of 10:
    
    julia> experiment(files = "uci", methods = [forest(), forest(notrees = 1000, maxdepth = 10)])

–––––

A dataset should have the following format:

.. code-block:: julia

    <names-row>
    <data-row>
    ...
    <data-row>

where

.. code-block:: julia

    <names-row> = <name><separator><name><separator>...<name>

and

.. code-block:: julia

    <data-row>  = <value><separator><value><separator>...<value>

<name> can be any of the following:

.. code-block:: julia

        CLASS            - declares that the column contains class labels
        REGRESSION       - declares that the column contains regression values
        ID               - declares that the column contains identifier labels
        IGNORE           - declares that the column should be ignored
        FOLD             - declares that the column contains labels for cross-validation folds
        WEIGHT           - declares that the column contains instance weights
        any other value  - is used to create a variable name

<separator> is a single character (as specified above)

<value> can be any of the following:

.. code-block:: julia

        integer          - is handled as a number if all values in the same column are of type integer,
                           float or NA, and as a string otherwise
        float            - is handled as a number if all values in the same column are of type integer,
                           float or NA, and as a string otherwise
        NA               - is handled as a missing value
        any other value  - is handled as a string

Example:

.. code-block:: julia

    ID,RI,Na,Mg,Al,Si,K,Ca,Ba,Fe,CLASS
    1,1.52101,NA,4.49,1.10,71.78,0.06,8.75,0.00,0.00,1
    2,1.51761,13.89,NA,1.36,72.73,0.48,7.83,0.00,0.00,1
    3,1.51618,13.53,3.55,1.54,72.99,0.39,7.78,0.00,0.00,1
    ...

–––––

For classification tasks the following measures are reported:

.. code-block:: julia

        Acc        - accuracy, i.e., fraction of examples correctly predicted
        AUC        - area under ROC curve
        Brier      - Brier score
        AvAcc      - average accuracy for single trees in the forest
        DEOAcc     - difference of the estimated and observed accuracy
        AEEAcc     - absolute error of the estimated accuracy
        AvBrier    - average Brier score for single trees in the forest
        VBrier     - average squared deviation of single tree predictions from forest predictions
        Margin     - diff. between prob. for correct class and prob. for most prob. other class
        Prob       - probability for predicted class
        Valid      - fraction of true labels included in prediction region
        Region     - size, i.e., number of labels, in prediction region
        OneC       - fraction of prediction regions containing exactly one true label
        Size       - the number of nodes in the forest
        Time       - the total time taken for both training and testing

For regression tasks the following measures are reported:

.. code-block:: julia

        MSE        - mean squared error
        Corr       - the Pearson correlation between predicted and actual values
        AvMSE      - average mean squared error for single trees in the forest
        VarMSE     - average squared deviation of single tree predictions from forest predictions
        DEOMSE     - difference of the estimated and observed MSE
        AEEMSE     - absolute error of the estimated MSE
        Valid      - fraction of true labels included in prediction region
        Region     - average size of prediction region
        Size       - the number of nodes in the forest
        Time       - the total time taken for both training and testing


---------

runexp 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
``runexp`` is used to test the performance of the library on a number of test sets


---------

