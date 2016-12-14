To work with a single dataset
=============================

To load a dataset from a file or dataframe:

    julia> load_data(<filename>, separator = <separator>)
    julia> load_data(<dataframe>)

The arguments should be on the following format:

    filename : name of a file containing a dataset (see format requirements above)
    separator : single character (default = ',')
    dataframe : a dataframe where the column labels should be according to the format requirements above

-----

To get a description of a loaded dataset:

    julia> describe_data(data)

-----

To evaluate a method or several methods for generating a random forest:

    julia> evaluate_method(method = forest(...), protocol = <protocol>)
    julia> evaluate_methods(methods = [forest(...), ...], protocol = <protocol>)

The arguments should be on the following format:

    method : a call to forest(...) as explained above (default = forest())
    methods : a list of calls to forest(...) as explained above (default = [forest()])
    protocol : integer, float, :cv or :test as explained above (default = 10)

-----

To generate a model from the loaded dataset:

    julia> m = generate_model(method = forest(...))                         

The argument should be on the following format:

    method : a call to forest(...) as explained above (default = forest())

-----

To get a description of a model:

    julia> describe_model(<model>)                                   

The argument should be on the following format:

    model : a generated or loaded model (see generate_model and load_model)

-----

To store a model in a file:

    julia> store_model(<model>, <file>)                              

The arguments should be on the following format:

    model : a generated or loaded model (see generate_model and load_model)
    file : name of file to store model in

-----

To load a model from file::

    julia> rf = load_model(<file>)                                  

The argument should be on the following format:

    file : name of file in which a model has been stored

-----

To apply a model to loaded data:

    julia> apply_model(<model>, confidence = <confidence>)

The argument should be on the following format:

    model : a generated or loaded model (see generate_model and load_model)
    confidence : a float between 0 and 1 or :std (default = :std)
                 - probability of including the correct label in the prediction region
                 - :std means employing the same confidence level as used during training
