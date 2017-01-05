Summary of all functions
=======================

All named arguments are optional, while the others are mandatory.

To run an experiment:

        experiment(files = <files>, separator = <separator>, protocol = <protocol>, 
                   methods = [<method>, ...])

To work with a single dataset:

        load_data(<filename>, separator = <separator>)

        load_data(<dataframe>)

        describe_data(<dataframe>)

        evaluate_method(method = forest(...), protocol = <protocol>)

        evaluate_methods(methods = [forest(...), ...], protocol = <protocol>)

        m = generate_model(method = forest(...))                

        describe_model(<model>)                                   

        store_model(<model>, <file>)                              

        m = load_model(<file>)                                  

        apply_model(<model>, confidence = <confidence>)
