import sys
sys.path.append('..')
import preprocessing as pp
import modelling as md
import descriptors as ds
import pandas as pd
import numpy as np
pd.set_option('mode.chained_assignment', None)
import ast
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from datetime import date
import matplotlib.pyplot as plt

import os

try:
    base_cwd = os.getcwd().split('regiochem')[0]
    base_cwd = f"{base_cwd}/regiochem"
except:
    raise ValueError("You are not in the right directory, need to be in the 'notebooks' directory or subdirectory of it.")

def norm(df, cols_to_exclude=["Reactant_SMILES", 'Atom_nº', 'Selectivity', "Reactive Atom", 'DOI']):
    out = pd.DataFrame()
    for col in df.columns:
        if col in cols_to_exclude or sum(df[col]) == 0:
            out[col] = df[col]
        else:
            out[col] = (df[col] - df[col].mean())/(df[col].max() - df[col].min())
    return out

def corr(df, thresh_corr, cols_to_exclude=["Reactant_SMILES", 'Atom_nº', 'Selectivity', "Reactive Atom", 'DOI']):
    out = pd.DataFrame()
    for col in df.columns:
        if col not in cols_to_exclude:
            out[col] = df[col]
    out = md.remove_correlated_features(out, False, thresh_corr)
    for col in cols_to_exclude:
        out[col] = df[col]
    return out



def load_data(file, out, normalize=True, threshold_corr=None):
    data_file = f"{base_cwd}/data/{file}"

    try:
        df_bde = pd.read_csv(f"{base_cwd}/data/descriptors/{out}/df_bde.csv", index_col=0)
        if normalize:
            df_bde = norm(df_bde)
        if threshold_corr and threshold_corr != 1:
            df_bde = corr(df_bde, threshold_corr)            
    except:
        print("Re-loading BDE dataframe")
        df_bde = md.prepare_reactivity_mapping(descriptor='BDE', file=data_file, 
                                            normalize=normalize, threshold_correlated=0.9)
        #df_bde.to_csv(f"{base_cwd}/data/descriptors/{out}/df_bde.csv")

    try:
        df_en2 = pd.read_csv(f"{base_cwd}/data/descriptors/{out}/df_en2.csv", index_col=0)
        if normalize:
            df_en2 = norm(df_en2)
        if threshold_corr and threshold_corr != 1:
            df_en2 = corr(df_en2, threshold_corr)
    except:
        print("Re-loading ENV-2 dataframe")
        df_en2 = md.prepare_reactivity_mapping(descriptor='ENV-2', file=data_file, 
                                            normalize=normalize, threshold_correlated=0.9)
        #df_en2.to_csv(f"{base_cwd}/data/descriptors/{out}/df_en2.csv")

    try:
        df_xtb = pd.read_csv(f"{base_cwd}/data/descriptors/{out}/df_xtb.csv", index_col=0)
        if normalize:
            df_xtb = norm(df_xtb)
        if threshold_corr and threshold_corr != 1:
            df_xtb = corr(df_xtb, threshold_corr)
    except:
        print("Re-loading XTB dataframe")
        df_xtb = md.prepare_reactivity_mapping(descriptor='XTB', file=data_file,
                                            normalize=normalize, threshold_correlated=0.9)
        #df_xtb.to_csv(f"{base_cwd}/data/descriptors/{out}/df_xtb.csv")

    try:
        df_gas = pd.read_csv(f"{base_cwd}/data/descriptors/{out}/df_gas.csv", index_col=0)
        if normalize:
            df_gas = norm(df_gas)
        if threshold_corr and threshold_corr != 1:
            df_gas = corr(df_gas, threshold_corr)
    except:
        print("Re-loading Gasteiger dataframe")
        df_gas = md.prepare_reactivity_mapping(descriptor='Gasteiger', file=data_file,
                                            normalize=normalize, threshold_correlated=0.9)
        #df_gas.to_csv(f"{base_cwd}/data/descriptors/{out}/df_gas.csv")

    try:
        df_dbs = pd.read_csv(f"{base_cwd}/data/descriptors/{out}/df_dbstep.csv", index_col=0)
        if normalize:
            df_dbs = norm(df_dbs)
        if threshold_corr and threshold_corr != 1:
            df_dbs = corr(df_dbs, threshold_corr)
    except:
        print("Re-loading DBSTEP dataframe")
        df_dbs = md.prepare_reactivity_mapping(descriptor='DBSTEP', file=data_file,
                                            normalize=normalize, threshold_correlated=0.9)
        #df_dbs.to_csv(f"{base_cwd}/data/descriptors/{out}/df_dbstep.csv")

    try:
        df_en1 = pd.read_csv(f"{base_cwd}/data/descriptors/{out}/df_en1.csv", index_col=0)
        if normalize:
            df_en1 = norm(df_en1)
        if threshold_corr and threshold_corr != 1:
            df_en1 = corr(df_en1, threshold_corr)
    except:
        print("Re-loading ENV-1 dataframe")
        df_en1 = md.prepare_reactivity_mapping(descriptor='ENV-1', file=data_file,
                                            normalize=normalize, threshold_correlated=0.9)
        #df_en1.to_csv(f"{base_cwd}/data/descriptors/{out}/df_en1.csv")

    try:
        df_rdkVbur = pd.read_csv(f"{base_cwd}/data/descriptors/{out}/df_rdkVbur.csv", index_col=0)
        if normalize:
            df_rdkVbur = norm(df_rdkVbur)
        if threshold_corr and threshold_corr != 1:
            df_rdkVbur = corr(df_rdkVbur, threshold_corr)
    except:
        print("Re-loading Rdkit-Vbur dataframe")
        df_rdkVbur = md.prepare_reactivity_mapping(descriptor='Rdkit-Vbur', file=data_file,
                                            normalize=normalize, threshold_correlated=0.9)
        #df_rdkVbur.to_csv(f"{base_cwd}/data/descriptors/{out}/df_rdkVbur.csv")

    return df_bde, df_en2, df_xtb, df_gas, df_dbs, df_en1, df_rdkVbur


def main(reg, 
         encodings_list=['XTB', 'Gasteiger', 'DBSTEP', 'ENV-1', 'ENV-2', 'BDE', 'Rdkit-Vbur'], #'DFT'
         threshold_imp=0.1,
         threshold_cor=0.9,
         out="preprocessed_reactions",
         file="reaction_data/dataset_crude.xlsx",
         normalize=True):
    """
    Main function to run the feature selection.
    Add description here.
    """
    
    df_bde, df_en2, df_xtb, df_gas, df_dbs, df_en1, df_rdkVbur = load_data(file, out, normalize=True, threshold_corr=0.9)

    # retrieve best parameters for all descriptors:
    list_param = []
    encoded_dfs = {'XTB': df_xtb, 'Gasteiger': df_gas, 'DBSTEP': df_dbs, 'ENV-1': df_en1, 'ENV-2': df_en2, 'BDE': df_bde, 'Rdkit-Vbur': df_rdkVbur} # 'DFT': df_dft, 

    for encoding in encodings_list:
        df          = encoded_dfs[encoding]
        print(f"Encoding: {encoding}, Shape: {df.shape}")
        res         = train_model_permutation_importance(df, reg)
        list_param += get_list_parameters(df, res, threshold_imp=threshold_imp)

    if not normalize:
        df_bde, df_en2, df_xtb, df_gas, df_dbs, df_en1, df_rdkVbur = load_data(file, out, normalize=False)

    df_all = pd.concat([df_xtb,
                        df_gas.drop(columns=['Reactant_SMILES', 'Atom_nº', 'Selectivity', 'Reactive Atom', 'DOI']),
                        df_dbs.drop(columns=['Reactant_SMILES', 'Atom_nº', 'Selectivity', 'Reactive Atom', 'DOI']),
                        df_en1.drop(columns=['Reactant_SMILES', 'Atom_nº', 'Selectivity', 'Reactive Atom', 'DOI']),
                        df_en2.drop(columns=['Reactant_SMILES', 'Atom_nº', 'Selectivity', 'Reactive Atom', 'DOI']),
                        df_bde.drop(columns=['Reactant_SMILES', 'Atom_nº', 'Selectivity', 'Reactive Atom', 'DOI']),
                        df_rdkVbur.drop(columns=['Reactant_SMILES', 'Atom_nº', 'Selectivity', 'Reactive Atom', 'DOI'])
                        ], axis=1)
    
    list_params = list(set(list_param) & set(df_all.columns.to_list()))
    df_selected = df_all[list_params + ['Reactant_SMILES', 'Atom_nº', 'Selectivity', 'Reactive Atom']]
    corr_matrix = df_selected.corr().abs()
    upper       = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop     = [column for column in upper.columns if any(upper[column] > threshold_cor)]

    print(f"DROPPING FEATURES: {to_drop}")

    df_selected.drop(to_drop, axis=1, inplace=True)
    df_selected.dropna(axis=1, inplace=True) 

    return df_selected, list_params   

def get_only_dft_ft_smiles(df, df_dft):
    """ 
    Get only the SMILES that are in the DFT descriptor.
    """
    df_ = df.copy()
    df_ = df_[df_.loc[:, 'Reactant_SMILES'].isin(df_dft.Reactant_SMILES.unique())]
    df_.reset_index(inplace=True, drop=True)
    return df_

def train_model_permutation_importance(df, reg):
    """ 
    Train a model and get the permutation importance.
    """
    X    = df.drop(columns=['Reactant_SMILES', 'Atom_nº', 'Selectivity', 'Reactive Atom', 'DOI'])
    X    = X.values
    y    = df.loc[:, "Selectivity"].values
    reg.fit(X, y)
    result = permutation_importance(reg, X, y, n_repeats=20)
    return result

def get_list_parameters(df, res, threshold_imp=0.1):
    """
    Get the list of parameters that are important.
    """
    sorted_idx = res.importances_mean.argsort()
    list_parameters = []
    for idx in sorted_idx:
        if res.importances_mean[idx] + res.importances_std[idx] > threshold_imp and res.importances_mean[idx] - 2 * res.importances_std[idx] > 0:
            list_parameters.append(df.columns[idx])
    return list_parameters