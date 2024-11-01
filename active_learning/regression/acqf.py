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

def make_descriptors_basic(option = 'custom', df_folder = 'preprocessed_reactions'):
    options = ['custom', 'bde', 'xtb', 'gas', 'env1', 'env2', 'dbstep', 'selected']

    assert option in options, f"option should be in {options}"
    # read_data

    print(base_cwd)
    df_bde    = pd.read_csv(f'{base_cwd}/data/descriptors/{df_folder}/df_bde.csv', index_col=0)
    df_xtb    = pd.read_csv(f'{base_cwd}/data/descriptors/{df_folder}/df_xtb.csv', index_col=0)
    df_gas    = pd.read_csv(f'{base_cwd}/data/descriptors/{df_folder}/df_gas.csv', index_col=0)
    df_env1   = pd.read_csv(f'{base_cwd}/data/descriptors/{df_folder}/df_en1.csv', index_col=0)
    df_env2   = pd.read_csv(f'{base_cwd}/data/descriptors/{df_folder}/df_en2.csv', index_col=0)
    df_dbstep = pd.read_csv(f'{base_cwd}/data/descriptors/{df_folder}/df_dbstep.csv', index_col=0)
    df_select = pd.read_csv(f'{base_cwd}/data/descriptors/{df_folder}/df_selected.csv', index_col=0)
    df_custom = pd.read_csv(f'{base_cwd}/data/descriptors/{df_folder}/df_custom.csv', index_col=0)

    if option == 'custom':
        df = df_custom
    elif option == 'bde':
        df = df_bde
    elif option == 'xtb':
        df = df_xtb
    elif option == 'gas':
        df = df_gas
    elif option == 'env1':
        df = df_env1
    elif option == 'env2':
        df = df_env2
    elif option == 'dbstep':
        df = df_dbstep
    elif option == 'selected':
        df = df_select
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
             df_folder = 'preprocessed_reactions',
             ):
    # get the descriptors and data
    df = make_descriptors_basic(option=feature_choice, df_folder=df_folder)
    df_small    = remove_large_molecules(df, target_SMILES)

    if tset_type == "cold":
        target_mol      =  Chem.MolFromSmiles(target_SMILES)
        training_SMILES = df_small[df_small.Reactant_SMILES != target_SMILES].Reactant_SMILES.unique()
        training_mols   = [Chem.MolFromSmiles(smi) for smi in training_SMILES]
        mcs_res         = [MCS.FindMCS([target_mol, mol], ringMatchesRingOnly=True, matchValences=True).numAtoms for mol in training_mols]
        idx             = np.argmax(mcs_res)
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
                aqcf_type, batch_size=batch_size, distance_balance=distance_balance, n_runs=n_runs, n_repeat=n_repeat, alpha=alpha)
    return top_5_scores_, smiles_, initial, carbon_preds_, max_aqcf_score_, df_remaining_small.columns

def evaluate_acqf(target_smi, train_SMILES,
                        reg, df, acqf = 'random', n_runs=1,
                        distance_balance=0.01, batch_size=1, n_repeat=10, alpha=1):
    # Perform acqf_1 evaluation:
    top_5_scores = []
    smiles = []
    carbon_preds = []
    max_aqcf_sco = []
    for i in tqdm(range(int(n_runs))):
        top_5_score_aqcf_, list_smiles_, C_pred_, max_aqcf_score_ = aq.eval_perf(target_smi, train_SMILES, df,
                                                     reg, acqf, acqf_args_dict = {'n_repet':n_repeat}, batch_size=batch_size,
                                                     distance_balance=distance_balance, alpha=alpha)
        top_5_scores.append(top_5_score_aqcf_)
        smiles.append(list_smiles_)
        carbon_preds.append(C_pred_)
        max_aqcf_sco.append(max_aqcf_score_)

    return top_5_scores, smiles, carbon_preds, max_aqcf_sco

