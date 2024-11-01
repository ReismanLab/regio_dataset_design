## utils

Contains the python scripts used for data (descriptors), model_validation (modeling and figure plots), active_learning (aquisition functions).

 - acquisition.py:       details the acquisition function used in the manuscript or reported in SI.
 - descriptors.py:       details the descriptor generation for each class of descriptors (xTB-Morfeus, Gasteiger, DBSTEP, local environnements, AIMNET, BDE, Rdkit-Vbur).
 - feature_selection.py: details on how the feature selection is performed for the Machine-Selection descriptors.
 - metrics.py:           helper for metric calculation.
 - preprocessing.py:     details on the preprocessing and the reactive site identification.
 - visualization.py:     helper for vizualizing the performances or plotting the molecules with predictions.