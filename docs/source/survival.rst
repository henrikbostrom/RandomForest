.. _survival.rst:

survival.rst
=========================================

.. DO NOT EDIT: this file is generated from Julia source.

apply_model 
^^^^^^^^^^^^
To apply a model to loaded data:

.. code-block:: julia

    julia> apply_model(<model>, confidence = <confidence>)

The argument should be on the following format:

.. code-block:: julia

    model : a generated or loaded model (see generate_model and load_model)
    confidence : a float between 0 and 1 or :std (default = :std)
                 - probability of including the correct label in the prediction region
                 - :std means employing the same confidence level as used during training


---------

