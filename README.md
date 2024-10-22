# regio_dataset_design

This code supports the results reported in the Chemrxiv working paper: [Tailoring Datasets for Regioselectivity Predictions on Complex Substrates](https://chemrxiv.org/engage/chemrxiv/article-details/670af7ffcec5d6c142f850d3)

The repository is divided in 5 folders:
 - data:             experimental data and descriptors
 - model_validation: scripts to run the modeling of the C(sp<sup>3</sup>)–H oxidations and the C(sp<sup>2</sup>)–H borylations
 - active_learning:  scripts to run the acquisition functions and evaluate the learning curves
 - results:          contains the raw files obtained when running the active_learning or model_validation scripts
 - utils:            main functions used for preprocessing/descriptor_generation/modeling/active_learning...

Following is a folder by folder description.

## installation

For best usage please install a conda environement using the ```regio_data.yml``` file.


## data

Contains the raw reaction data and scripts to featurize the reactants for the modellind and active learning sections

   **reaction_data:** contains the raw reaction dataset for C(sp<sup>3</sup>)–H dioxirane oxidations [dataset_crude.xlsx](https://github.com/ReismanLab/regiochem/blob/main/data/reaction_data/dataset_crude.xlsx), a list of canonical_smiles to featurize and a python script to preprocess a crude dataset that can be run using the following commands:
     
 
       python protonate_amines_discard_unspec_diastereocenters.py
       python preprocess_reactions.py --out 'folder_out' --in_folder 'reaction_data' --in_file 'dataset_crude_cleaned_doi_amine.xlsx'


   **borylation_reaction_data:** contains the raw reaction dataset for C(sp<sup>3</sup>)–H dioxirane oxidations [borylation_regio.csv](https://github.com/ReismanLab/regiochem/blob/main/data/borylation_reaction_data/borylation_regio.csv), and notebooks used for the preprocessing.


   **descriptors:**
           aimnet2 and bdes are folders containing helper tools to featurize the reactants using the [BDE prediction tool](https://github.com/patonlab/BDE-db2) and [AIMNET2](https://github.com/isayevlab/AIMNet2)
           the "preprocessed_..." folders contains csv file of already preprocessed and featurized reactions for the oxidation and borylation datasets, enabling to run the validations faster.
           smiles_descriptors contains json files with the SMILES strings of the canonicalized SMILES that were featurized and their bond and atomic descriptors. These files are updated each time compute_desc.py are called. They can be run using the following command line with any csv file containing a "SMILES" column.
           
       python compute_desc.py --desc 'ALL' --csv '../reaction_data/can_smiles.csv' --njobs 1


   
## model_validation

   **regression:**  contains two folders large_mol and loo to run the validation for the large_mol validation task and leave-one-out task. These validations and the plot of the figures can obtained using the following command lines for the large molecule in *large_mol* and *loo* respectively. 
     
       python perf_bm.py
       python figures_bm.py --run average --desc Custom --model RF2 --rxn dioxirane

Generates this type of figures for TOP-1, TOP-2, TOP-3, TOP-5 and TOP-AVG:
       
![TOP-1 for the predictions on large molecules](model_validation/regression/large_mol/dioxirane/average/heatmap_TOP1.png)

       python perf_loo.py
       python figure_loo.py --run run_01 --desc Selected --model RF2

Generates this type of figures for TOP-1, TOP-2, TOP-3, TOP-5 and TOP-AVG:
       
![TOP-1 for the predictions on loo](model_validation/regression/loo/dioxirane/average/heatmap_TOP-1.png)
       
## active_learning

**regression:** contains scripts to generate results of acquisition functions on targets as well as to generate visualizations using a random forest model

- To generate results of one acquisition function on one target using the same model to select molecules and to evaluate performance, use ```main.py```. SMILES, acquistion function, batch size, warm/cold start, feature type, among other parameters can be specified as command line arguments.
         
- To generate results for multiple acquisition functions and multiple targets, the ```run_acqf.sh``` script can be used.
         
- To change the model used for performance evaluation of the acquisition functions, use ```eval_perf_new_model.py```. The path to a folder containing the initial acquisition function results must be specified, and a new folder containing the results with performance recomputed with a new model will be created. Performance can be recomputed using a ranking or random forest model. The input and output folders, the descriptor type, and the model choice can be passed in as command line arguments.
         
- To generate plots comparing each acquisition function to the random baseline, use script ```baseline_comparisons.py```. There is a choice of metric type (AUC or training set size at stable performance) and a choice of figure type (bar plot, box plot, or violin plot) which can be passed in as command line arguments. For the metrics, performance threshold, duration and dataset (to remove) can also be specified as command line arguments.


       python baseline_comparisons.py --path "regression/clean_run" --out "clean_run"


![Random Baseline Comparison](active_learning/regression/clean_run/standard_dist_clust-all_metric-SSP_disttype-box.png)
         
- To compare AFs on a same plot for a specific SMILES:

       python learning_curve_comp.py --acqf_list "["acqf_1", "acqf_2", "acqf_6"]" --smi "CC(=O)O[C@H]1CC[C@@]2(C)[C@@H](CC(=O)[C@@H]3[C@@H]2CC[C@]2(C)[C@@H]([C@H](C)CCCC(C)C)CC[C@@H]32)C1"
 
![Active Learning Target Comparison](active_learning/regression/clean_run/learning_curves_comp/lc_comp_tmp_1.png)    

Note that the results detailed for each target can be found in [active_learning/regression/clean_run/learning_curves_comp](active_learning/regression/clean_run/learning_curves_comp) and can be computed using ```print_all_lc_comp.py```.

- To visualize the learnig curves for each molecules by acquisition functions featuring TOP-1, TOP-2, TOP-3, TOP-5, and TOP-10, can be obtained using ```learning_curve.py```. Precomputed learning curves can be found in [active_learning/regression/clean_run/learning_curves_10](active_learning/regression/clean_run/learning_curves_10).

       python learning_curve.py --overwrite True

 ![Active Learning Summary](active_learning/regression/clean_run/learning_curves_10/summary_5_CC(=O)O[C@H]1CC[C@@]2(C)[C@@H](CC(=O)[C@@H]3[C@@H]2CC[C@]2(C)[C@@H]([C@H](C)CCCC(C)C)CC[C@@H]32)C1.png) 


**experimental:** Same workflow for additional targets.


## results

Contains the results of the model_validation and active_learning sections, the results are stored for ease of figure reproduction. This folder is organized with the same tree as model_validation.

   
## utils

Contains the python scripts used for data (descriptors), model_validation (modeling and figure plots), active_learning (aquisition functions).

 - acquisition.py:       details the acquisition function used in the manuscript or reported in SI.
 - descriptors.py:       details the descriptor generation for each class of descriptors (xTB-Morfeus, Gasteiger, DBSTEP, local environnements, AIMNET, BDE, Rdkit-Vbur).
 - feature_selection.py: details on how the feature selection is performed for the Machine-Selection descriptors.
 - metrics.py:           helper for metric calculation.
 - preprocessing.py:     details on the preprocessing and the reactive site identification.
 - visualization.py:     helper for vizualizing the performances or plotting the molecules with predictions.

