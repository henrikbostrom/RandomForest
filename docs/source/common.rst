.. _common.rst:

common.rst
=========================================

.. DO NOT EDIT: this file is generated from Julia source.

evaluate_method 
^^^^^^^^^^^^
To evaluate a method or several methods for generating a random forest:

.. code-block:: julia

    julia> evaluate_method(method = forest(...), protocol = <protocol>)
    julia> evaluate_methods(methods = [forest(...), ...], protocol = <protocol>)

The arguments should be on the following format:

.. code-block:: julia

    method : a call to forest(...) as explained above (default = forest())
    methods : a list of calls to forest(...) as explained above (default = [forest()])
    protocol : integer, float, :cv or :test as explained above (default = 10)


---------

generate_model 
^^^^^^^^^^^^
To generate a model from the loaded dataset:

.. code-block:: julia

    julia> m = generate_model(method = forest(...))

The argument should be on the following format:

.. code-block:: julia

    method : a call to forest(...) as explained above (default = forest())


---------

load_data 
^^^^^^^^^^^^
To load a dataset from a file or dataframe:

.. code-block:: julia

    julia> load_data(<filename>, separator = <separator>)
    julia> load_data(<dataframe>)

The arguments should be on the following format:

.. code-block:: julia

    filename : name of a file containing a dataset (see format requirements above)
    separator : single character (default = ',')
    dataframe : a dataframe where the column labels should be according to the format requirements above


---------

load_model 
^^^^^^^^^^^^
To load a model from file:

.. code-block:: julia

    julia> rf = load_model(<file>)

The argument should be on the following format:

.. code-block:: julia

    file : name of file in which a model has been stored


---------

store_model 
^^^^^^^^^^^^
To store a model in a file:

.. code-block:: julia

    julia> store_model(<model>, <file>)

The arguments should be on the following format:

.. code-block:: julia

    model : a generated or loaded model (see generate_model and load_model)
    file : name of file to store model in


---------

