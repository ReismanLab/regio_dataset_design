
import sys
import warnings
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
import os

root = os.getcwd()
try:
    base_cwd = os.getcwd().split('regiochem')[0]
    base_cwd = f"{base_cwd}/regiochem"
except:
    raise ValueError("You are not in the right directory, need to be in the 'notebooks' directory or subdirectory of it.")

sys.path.append(f"{base_cwd}/utils/")
pd.set_option('mode.chained_assignment', None)
warnings.simplefilter(action='ignore', category=FutureWarning) 

import modelling as md
import preprocessing as pp


import argparse
parser = argparse.ArgumentParser(description='Preprocess reactions')
parser.add_argument('--in_folder',
                    help='Folder containing input SMILES')
parser.add_argument('--in_file',
                    help='data file with input SMILES')
parser.add_argument('--out',
                    help='Output folder')
parser.add_argument('--atom',
                    help='O or B',
                    default='O')
parser.add_argument('--thresh_corr',
                    help='Threshold to remove highly correlated features',
                    default=0.9)
parser.add_argument('--normalize',
                    help='Flag to normalize features',
                    default="False")
parser.add_argument('--descs', type=str, nargs='+',
                    help='Names of descriptors to generate')



args = parser.parse_args()
rxn_folder = args.in_folder
assert rxn_folder is not None, "No input folder specified!"
assert args.in_file is not None, "No data file specified!"
assert args.out is not None, "No output folder specified!"

data_file            = f'{base_cwd}/data/{rxn_folder}/{args.in_file}' # this is the file from which all the model validations and active learning will be run
out                  = args.out
out_folder           = f"{base_cwd}/data/descriptors/{out}/"
atom                 = args.atom
threshold_correlated = float(args.thresh_corr)
if args.normalize == "False":
    normalize = False
else:
    normalize = True

if args.descs is None:
    descs = ["XTB", "Gasteiger", "DBSTEP", "ENV-1", "ENV-2", "Rdkit-Vbur", "BDE", "Custom", "Selected"]
else:
    descs = args.descs

if not os.path.isdir(out_folder):
    os.mkdir(out_folder)

print("Preparing phys-chem descriptors")
# phys-chem descriptors
if "XTB" in descs:
    print("Preparing XTB descriptors")

    df_xtb    = md.prepare_reactivity_mapping('XTB', file=data_file, 
                                            preprocess=True,
                                            normalize=normalize, threshold_correlated=threshold_correlated,
                                            rxn_folder=rxn_folder, atom=atom)
    df_xtb = pp.add_dois_to_df(df_xtb, rxn_folder=rxn_folder)
    df_xtb.to_csv(f"{out_folder}df_xtb.csv")

if "Gasteiger" in descs:
    print("Preparing Gasteiger descriptors")
    df_gas    = md.prepare_reactivity_mapping('Gasteiger', file=data_file, 
                                            preprocess=True,
                                            normalize=normalize, threshold_correlated=threshold_correlated,
                                            rxn_folder=rxn_folder, atom=atom)
    df_gas = pp.add_dois_to_df(df_gas, rxn_folder=rxn_folder)
    df_gas.to_csv(f"{out_folder}df_gas.csv")

if "DBSTEP" in descs:
    print("Preparing DBSTEP descriptors")
    df_dbs    = md.prepare_reactivity_mapping('DBSTEP', file=data_file, 
                                            preprocess=True,
                                            normalize=normalize, threshold_correlated=threshold_correlated,
                                            rxn_folder=rxn_folder, atom=atom)
    df_dbs = pp.add_dois_to_df(df_dbs, rxn_folder=rxn_folder)
    df_dbs.to_csv(f"{out_folder}df_dbstep.csv")

if "ENV-1" in descs:
    print("Preparing ENV1 descriptors")
    df_en1    = md.prepare_reactivity_mapping('ENV-1', file=data_file, 
                                            preprocess=True,
                                            normalize=normalize, threshold_correlated=threshold_correlated,
                                            rxn_folder=rxn_folder, atom=atom)
    df_en1 = pp.add_dois_to_df(df_en1, rxn_folder=rxn_folder)
    df_en1.to_csv(f"{out_folder}df_en1.csv")

    # OHE descriptors
    from sklearn.preprocessing import OneHotEncoder
    enc        = OneHotEncoder()
    df_en1_ohe = enc.fit_transform(df_en1.drop(columns=['Reactive Atom', 'Selectivity', 'Atom_nº', 'Reactant_SMILES']))
    df_en1_ohe = pd.DataFrame(df_en1_ohe.toarray(), columns=enc.get_feature_names_out())
    df_en1_ohe = pd.concat([df_en1_ohe, df_en1[['Reactive Atom', 'Selectivity', 'Atom_nº', 'Reactant_SMILES']]], axis=1)
    df_en1_ohe = pp.add_dois_to_df(df_en1_ohe, rxn_folder=rxn_folder)
    df_en1_ohe.to_csv(f"{out_folder}df_en1_ohe.csv")

if "ENV-2" in descs:
    print("Preparing ENV2 descriptors")
    df_en2    = md.prepare_reactivity_mapping('ENV-2', file=data_file, 
                                            preprocess=True,
                                            normalize=normalize, threshold_correlated=threshold_correlated,
                                            rxn_folder=rxn_folder, atom=atom)
    df_en2 = pp.add_dois_to_df(df_en2, rxn_folder=rxn_folder)
    df_en2.to_csv(f"{out_folder}df_en2.csv")

if "Rdkit-Vbur" in descs:
    print("Preparing rdkit-Vbur descriptors")
    df_rdkVbur = md.prepare_reactivity_mapping('Rdkit-Vbur', file=data_file, 
                                            preprocess=True,
                                            normalize=normalize, threshold_correlated=threshold_correlated,
                                            rxn_folder=rxn_folder, atom=atom)
    df_rdkVbur = pp.add_dois_to_df(df_rdkVbur, rxn_folder=rxn_folder)
    df_rdkVbur.to_csv(f"{out_folder}df_rdkVbur.csv")

if "BDE" in descs:
    print("Preparing BDE descriptors")
    df_bde    = md.prepare_reactivity_mapping('BDE', file=data_file, 
                                            preprocess=True,
                                            normalize=normalize, threshold_correlated=threshold_correlated,
                                            rxn_folder=rxn_folder, atom=atom)
    df_bde = pp.add_dois_to_df(df_bde, rxn_folder=rxn_folder)
    df_bde.to_csv(f"{out_folder}df_bde.csv")

if "Custom" in descs:
    # Chemist features selection
    print(df_bde.columns)
    print(df_gas.columns)
    df_custom = df_bde[['Reactant_SMILES', 'Atom_nº', 'Selectivity', 'Reactive Atom', 'bde_avg', 'DOI']].drop(columns=["DOI"], axis=1).merge(df_gas[["Reactant_SMILES", "Atom_nº", "Selectivity", "Reactive Atom", "gas_charge_C", "gas_charge_H_max"]], on = ["Reactant_SMILES", "Atom_nº", "Selectivity", "Reactive Atom"])
    df_custom = df_custom.merge(df_xtb[['Buried_Volume_C', 'Buried_Volume_H_max', "Reactant_SMILES", "Atom_nº"]], on=["Reactant_SMILES", "Atom_nº"])
    df_custom = df_custom.loc[:,~df_custom.columns.duplicated()].copy()
    df_custom = pp.add_dois_to_df(df_custom, rxn_folder=rxn_folder)
    df_custom.to_csv(f"{out_folder}df_custom.csv")

if "Selected" in descs:
    # ML feature selection descriptors
    import feature_selection as fs
    from sklearn.ensemble import RandomForestRegressor
    rf2 = RandomForestRegressor(n_estimators=250,
                                max_features=0.5,
                                max_depth=10,
                                min_samples_leaf=3)
    df_sel, param = fs.main(rf2, out=out, file=data_file, normalize=normalize, threshold_cor=threshold_correlated)
    df_sel = pp.add_dois_to_df(df_sel, rxn_folder=rxn_folder)
    df_sel.to_csv(f"{out_folder}df_selected.csv")

