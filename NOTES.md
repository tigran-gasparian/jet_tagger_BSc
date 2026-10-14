The model tries to assign a particle to one out of 8 labels
    

20 input variables  - found in input_vars.json variables.json metadata.json 
                    - were plot in shap_summary_class as y-axis
8 output variables  - correspond to a particle type estimated by the model
                    - found in class_label.json variables.json metadata.json
                    - were plot everywhere except for the loss history plot
16 most important particles - events or jets with <16 particles are padded with zeros
                            - events or jets with >16 particles are truncated to 16

    training_data_CMS/X_train.npy
    Shape: (14679587, 16, 20)  <--14679587 samples
    Data Type: float64,

    training_data_CMS/X_test.npy
    Shape: (1631066, 16, 20)
    Data Type: float64

    training_data_CMS/y_train.npy
    Shape: (14679587, 8)
    Data Type: float64

    training_data_CMS/y_test.npy
    Shape: (1631066, 8)
    Data Type: float64

    data_chunk_457.root
    Shape: (55287, 16, 20)
        