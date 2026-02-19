### IMPORTS ###

# add path to utils
import sys
import os

try:
    base_cwd = os.getcwd().split('regio_dataset_design')[0]
    base_cwd = f"{base_cwd}regio_dataset_design"
except:
    raise ValueError("You are not in the right directory, need to be in the 'notebooks' directory or subdirectory of it.")
sys.path.append(f"{base_cwd}/utils/")

# remaining_imports
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
import modelling as md
import acquisition as aq
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from tqdm import tqdm
from rdkit.Chem import MCS
from datetime import date
import pickle

pd.set_option('mode.chained_assignment', None)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('mode.chained_assignment', None)

### PARAMETERS ###
n_runs = 1

# remove large molecules from training set and remaining set:
def remove_large_molecules(df, target_SMILES, max_num_C=15):
    num_C = []
    for smi in df.Reactant_SMILES:
        if smi != target_SMILES:
            mol = Chem.MolFromSmiles(smi)
            if not mol: # catch edge case when mol is None
                print("Mol is None", smi)
                num_C.append(15)
            else:
                num_C.append([at.GetSymbol() for at in mol.GetAtoms()].count('C'))
        else:
            num_C.append(0)
    df['num_C'] = num_C
    df = df[df.num_C <= max_num_C]
    df.drop(columns=['num_C'], inplace=True)
    return df

def find_reactive_at(f, obs):
    top_at = []
    for s in f.Reactant_SMILES.unique():
        f_sub = f.loc[f.Reactant_SMILES == s]
        f_sub = f_sub.sort_values(obs, ascending=False)
        reactive_at = f_sub.loc[:, 'Atom_nº'].values[0]
        top_at.extend([reactive_at] * len(f_sub))
    return top_at

def make_descriptors_basic(option = 'custom', df_folder = 'preprocessed_dioxirane_reactions', feat = "Selectivity"):
    """
    Input:
    option      : str, the type of descriptors to use
    df_folder   : str, the folder with the datasets
    feat        : str, the name of the parameter to use as a prediction target 
    Output:
    features[option] : pd.DataFrame, formatted data frame with 
    """
    # read_data
    dict_descs = {  'bde'       : 'df_bde.csv',
                    'xtb'       : 'df_xtb.csv',
                    'gas'       : 'df_gas.csv',
                    'env1'      : 'df_en1.csv',
                    'env2'      : 'df_en2.csv',
                    'dbstep'    : 'df_dbstep.csv',
                    'selected'  : 'df_selected.csv',
                    'custom'    : 'df_custom.csv'}
     
    assert option in dict_descs.keys(), f"option should be in {dict_descs.keys()}"
    
    df = pd.read_csv(f"{base_cwd}/data/descriptors/{df_folder}/{dict_descs[option]}", index_col=0)

    if "Reactive Atom" not in df.columns: # handling for new observables
        df["Reactive Atom"] = find_reactive_at(df, feat)
        df.to_csv(f"{base_cwd}/data/descriptors/{df_folder}/{dict_descs[option]}")
    return df

def benchmark_aqcf_on_smiles(aqcf_type,      # the type of acquisition function
             target_SMILES,  # the target smiles
             tset_type, # warm or cold
             reg_,
             batch_size,
             distance_balance,
             n_repeat,
             feature_choice,
             selection_strategy,
             n_runs,
             alpha,
             df_folder = 'preprocessed_dioxirane_reactions',
             feat = "Selectivity",
             max_tset=np.inf
             ):
    # get the descriptors and data
    df = make_descriptors_basic(option=feature_choice, df_folder=df_folder, feat=feat)
    df_small    = remove_large_molecules(df, target_SMILES)

    if tset_type == "cold":
        target_mol      =  Chem.MolFromSmiles(target_SMILES)
        training_SMILES = df_small[df_small.Reactant_SMILES != target_SMILES].Reactant_SMILES.unique()
        training_mols   = [Chem.MolFromSmiles(smi) for smi in training_SMILES]
        
        idx=None
        largest_atom_ct = 0
        for i in tqdm(range(len(training_mols))):
            mol = training_mols[i]
            at_ct = MCS.FindMCS([target_mol, mol], ringMatchesRingOnly=True, matchValences=True).numAtoms
            if  at_ct > largest_atom_ct:
                largest_atom_ct = at_ct
                idx = i

        initial = training_SMILES[idx]
        training_SMILES = [training_SMILES[idx]]

        df_training = df.loc[df['Reactant_SMILES'] == training_SMILES[0]]
        df_remaining  = df.loc[df['Reactant_SMILES'] != training_SMILES[0]]

    if tset_type == "warm":
        dois_to_start_with = ['10.1021/ja00199a039', '10.1021/jo9604189']
        df_training   = df.loc[df['DOI'].isin(dois_to_start_with)]
        df_remaining  = df.loc[df['DOI'].isin(dois_to_start_with) == False]
        training_SMILES = df_training.Reactant_SMILES.unique()

    df_remaining_small = remove_large_molecules(df_remaining, target_SMILES)
    df_training_small  = remove_large_molecules(df_training, target_SMILES)
    training_SMILES_small = df_training_small.Reactant_SMILES.unique()
    reg = clone(reg_)

    if selection_strategy == "simple":
        top_5_scores_, smiles_, carbon_preds_, max_aqcf_score_ = evaluate_acqf(target_SMILES, training_SMILES_small, reg, df_small,
                aqcf_type, batch_size=batch_size, distance_balance=distance_balance, n_runs=n_runs, n_repeat=n_repeat, alpha=alpha,
                feat=feat, df_folder=df_folder, max_tset=max_tset)
        
    return top_5_scores_, smiles_, initial, carbon_preds_, max_aqcf_score_, df_remaining_small.columns

def evaluate_acqf(target_smi, train_SMILES,
                        reg, df, acqf, n_runs=1,
                        distance_balance=0.01, batch_size=1, n_repeat=10, alpha=1, feat="Selectivity",
                        df_folder="preprocessed_dioxirane_reactions",
                        max_tset=np.inf):
    # Perform acqf_1 evaluation:
    top_5_scores = []
    smiles = []
    carbon_preds = []
    max_aqcf_sco = []
    for i in tqdm(range(int(n_runs))):
        top_5_score_aqcf_, list_smiles_, C_pred_, max_aqcf_score_ = aq.eval_perf(target_smi, train_SMILES, df,
                                                     reg, acqf, acqf_args_dict = {'n_repet':n_repeat}, batch_size=batch_size,
                                                     distance_balance=distance_balance, alpha=alpha, feat=feat,
                                                     df_folder=df_folder,
                                                     max_tset=max_tset)
        top_5_scores.append(top_5_score_aqcf_)
        smiles.append(list_smiles_)
        carbon_preds.append(C_pred_)
        max_aqcf_sco.append(max_aqcf_score_)

    return top_5_scores, smiles, carbon_preds, max_aqcf_sco

