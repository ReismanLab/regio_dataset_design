# data

Contains the raw reaction data and scripts to featurize the reactants for the modelling and active learning sections

## dioxirane reaction data:

Contains the raw reaction dataset for C(sp<sup>3</sup>)–H dioxirane oxidations [dataset_crude.xlsx](https://github.com/ReismanLab/regiochem/blob/main/data/reaction_data/dataset_crude.xlsx), a list of canonical_smiles to featurize and a python script to preprocess a crude dataset that can be run using the following commands:
   
```
python preprocess_reactions.py --out 'preprocessed_dioxirane_reactions' --in_folder 'dioxirane_reaction_data' --in_file 'dataset_crude_filtered.xlsx'
```

## borylation reaction data:

Contains the raw reaction dataset for C(sp<sup>3</sup>)–H dioxirane oxidations [borylation_regio.csv](https://github.com/ReismanLab/regiochem/blob/main/data/borylation_reaction_data/borylation_regio.csv), and notebooks used for the preprocessing.


## descriptors:

Aimnet2 and bdes are folders containing helper tools to featurize the reactants using the [BDE prediction tool](https://github.com/patonlab/BDE-db2) and [AIMNET2](https://github.com/isayevlab/AIMNet2)
the "preprocessed_..." folders contains csv file of already preprocessed and featurized reactions for the oxidation and borylation datasets, enabling to run the validations faster.
smiles_descriptors contains json files with the SMILES strings of the canonicalized SMILES that were featurized and their bond and atomic descriptors. These files are updated each time compute_desc.py are called. They can be run using the following command line with any csv file containing a "SMILES" column.

```           
python compute_desc.py --desc 'ALL' --csv test_smiles.csv
```
