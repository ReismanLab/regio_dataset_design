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

def make_descriptors_basic(option = 'custom', df_folder = 'preprocessed_dioxirane_reactions', feat = "Selectivity", maximize = True):
    """
    Input:
    option      : str, the type of descriptors to use
    df_folder   : str, the folder with the datasets
    feat        : str, the name of the parameter to use as a prediction target 
    pred_max    : bool, if True then the model ranks the "most reactive atoms" as the atom that has the largest target values, otherwise it takes the minimum. 
    Output:
    features[option] : pd.DataFrame, formatted data frame with 
    """
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

    features = {
            'bde'       : df_bde,
            'xtb'       : df_xtb,
            'dbstep'    : df_dbstep, 
            'gs'        : df_gas,
            'env1'      : df_env1,
            'env2'      : df_env2,
            'selected'  : df_select,
            'custom'    : df_custom}

    if feat != "Selectivity":
        # search for obs in all descriptor dataframes
        obs_col = None 
        for f in features: # look for feat in any of the proposed dataframes
            if feat in features[f].columns:
                obs_col = features[f][feat]
                if not maximize:
                    obs_col = -1 * obs_col
        if obs_col is None:
            assert False, "Observable not found in any descriptor dataframe, exiting."

        # add obs to all descriptor dataframes
        for f in features: 
            features[f][feat] = obs_col # does that work if some dataframe have a different shape, for example if one compound could not be featurized with one method?

        # find reactive atom, can do it on the first dataframe as they now all have that same column
        f = features[list(features.keys())[0]]
        top_at = []
        for s in f.Reactant_SMILES.unique():
            f_sub = f.loc[f.Reactant_SMILES == s]
            if pred_max:
                f_sub = f_sub.sort_values(feat, ascending=False) # looks for the maximum value
            else:
                f_sub = f_sub.sort_values(feat, ascending=True) # looks for the minimum value
            reactive_at = f_sub.loc[:, 'Atom_nº'].values[0]
            top_at.extend([reactive_at] * len(f_sub))
        
        # add reactive atom to all descriptor dataframes
        for f in features: 
            features[f]["Reactive Atom"] = top_at
            features[f] = features[f].drop("Selectivity", axis=1)
    
    # option to scale the target value between 0 and 100 with 100 being the best possible value -- idea = match selecitivity training and make sure high value correspond to desired property for the AF-Al to make sense
    if feat != "Selectivity":
         print(f"Renormalizing target values for {feat} such that values go from 0 to 100 with 100 being the most desirable value")
         normalized_df = features[option]
         max_ft = max(normalized_df[feat].values)
         min_ft = min(normalized_df[feat].values)
         assert max_ft != min_ft, "Observable is monotonous -- exiting."
         normalized_df[feat] = normalized_df[feat].map(lambda x : 100 * (x - min_ft) / (max_ft - min_ft))
         if pred_max != True:
             print(f"Changing observable values such that original min values are large")
             normalized_df[feat] = normalized_df[feat].map(lambda x : 100 - x)
         features[option] = normalized_df
 
    # sanity check:
    features[option].to_csv("features_prepared.csv")
    
    return features[option]

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
             feat = "Selectivity"
             ):
    # get the descriptors and data
    df = make_descriptors_basic(option=feature_choice, df_folder=df_folder, feat=feat, pred_max = False)
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
                aqcf_type, batch_size=batch_size, distance_balance=distance_balance, n_runs=n_runs, n_repeat=n_repeat, alpha=alpha,
                feat=feat, df_folder=df_folder)
    return top_5_scores_, smiles_, initial, carbon_preds_, max_aqcf_score_, df_remaining_small.columns

def evaluate_acqf(target_smi, train_SMILES,
                        reg, df, acqf = 'random', n_runs=1,
                        distance_balance=0.01, batch_size=1, n_repeat=10, alpha=1, feat="Selectivity",
                        df_folder="preprocessed_dioxirane_reactions"):
    # Perform acqf_1 evaluation:
    top_5_scores = []
    smiles = []
    carbon_preds = []
    max_aqcf_sco = []
    for i in tqdm(range(int(n_runs))):
        top_5_score_aqcf_, list_smiles_, C_pred_, max_aqcf_score_ = aq.eval_perf(target_smi, train_SMILES, df,
                                                     reg, acqf, acqf_args_dict = {'n_repet':n_repeat}, batch_size=batch_size,
                                                     distance_balance=distance_balance, alpha=alpha, feat=feat,
                                                     df_folder=df_folder)
        top_5_scores.append(top_5_score_aqcf_)
        smiles.append(list_smiles_)
        carbon_preds.append(C_pred_)
        max_aqcf_sco.append(max_aqcf_score_)

    return top_5_scores, smiles, carbon_preds, max_aqcf_sco

