# add path to utils
import sys
import os
root = os.getcwd()

try:
    base_cwd = root.split('regiochem')[0]
    base_cwd = f"{base_cwd}regiochem"
except:
    raise ValueError("You are not in the right directory, need to be in the 'notebooks' directory or subdirectory of it.")

sys.path.append(f"{base_cwd}/utils/")

### Parse arguments
import argparse
from rdkit import Chem
parser = argparse.ArgumentParser(description='Acquisition function result computation')
parser.add_argument('--reactant',
                    help='Target smiles for acquisition function',
                    default="C[C@@H]1CC[C@H]2C(C)([C@H]3C[C@@]12CC[C@]3(O)C)C")
parser.add_argument('--products', type=str, nargs='+',
                    help='List of products',
                    default=["C[C@]1(O)CC[C@H]2C([C@H]3C[C@@]12CC[C@@]3(C)O)(C)C"])
parser.add_argument('--sels', type=int, nargs='+',
                    help='List of experimental selectivities',
                    default=[100])
parser.add_argument('--acqf',
                    help='Acquisition function to compute (labels defined in utils/acquisition)',
                    default='acqf_1')
parser.add_argument('--batch',
                    help='Batch size',
                    default=1)
parser.add_argument('--start',
                    help='Warm- or cold-start',
                    default='cold')
parser.add_argument('--n_repet',
                    help='Number of models to use for each ACQF-1 evaluation',
                    default=10)
parser.add_argument('--db',
                    help='Distance balance for ACQF-1 evaluation',
                    default=1)
parser.add_argument('--feat',
                    help='Features to use for RF computation',
                    default='custom')
parser.add_argument('--n_est',
                    help='Argument to RandomForestRegressor',
                    default=250)
parser.add_argument('--max_feats',
                    help='Argument to RandomForestRegressor',
                    default=0.5)
parser.add_argument('--max_depth',
                    help='Argument to RandomForestRegressor',
                    default=10)
parser.add_argument('--min_samples_leaf',
                    help='Argument to RandomForestRegressor',
                    default=3)
parser.add_argument('--model',
                    help='String description of model being used',
                    default='regression_rf')
parser.add_argument('--selection_strat',
                    help='Additional selection strategies for acqf evaluation',
                    default="simple")
parser.add_argument('--res', 
                    help='Name for results folder',
                    default='experimental')
parser.add_argument('--run', 
                    help='Name for the run ',
                    default='0')
parser.add_argument('--df_folder',
                    help='Name for the folder where the precomputed descriptors are',
                    default='preprocessed_reactions_no_unspec_no_intra_unnorm')
parser.add_argument('--alpha',
                    help='balance between uncertainty and reactivity weighting for the target site to orient selection alpha must be between 0 and 2',
                    default=1)
parser.add_argument('--thresh_corr',
                    help='threshold correlated',
                    default=0.9)
parser.add_argument('--large',
                    help='True if large molecules are included',
                    default="False")
parser.add_argument('--atom',
                    help='O or B',
                    default="O")

args             = parser.parse_args()
target_SMILES    = Chem.CanonSmiles(args.reactant)
products         = args.products
sels             = [int(sel) for sel in args.sels]
acqf             = args.acqf
batch            = int(args.batch)
start            = args.start
n_repet          = int(args.n_repet)
distance_balance = float(args.db)
feature_choice   = args.feat
n_estimators     = int(args.n_est)
max_features     = float(args.max_feats)
max_depth        = int(args.max_depth)
min_samples_leaf = float(args.min_samples_leaf)
min_samples_leaf = min_samples_leaf if min_samples_leaf <1 else int(min_samples_leaf)
model            = args.model
selection_strategy = args.selection_strat
res              = args.res
run              = args.run
df_folder        = args.df_folder
alpha            = float(args.alpha)
thresh_corr      = float(args.thresh_corr)
include_large    = args.large
if include_large == "False":
    include_large = False
else:
    include_large = True
atom = args.atom

import os
path = f"{base_cwd}/results/active_learning/regression/{res}"
if not os.path.exists(path):
    os.mkdir(path)

if "/" in target_SMILES: # handling for molecules with double bond stereochemistry
    s = target_SMILES.replace("/", "-")
else:
    s = target_SMILES

if os.path.exists(f"{path}/res_rf_{s}_{acqf}_{run}_{batch}_{start}start_{feature_choice}.pkl"):
    print(f"\n\nSkipping {target_SMILES} with {acqf} {run} {batch} {start} {feature_choice}\n\n because already computed!", flush=True)
    exit()

## remaining imports
sys.path.append(f"../regression")
import acqf as a
import acquisition as aq
import modelling as md
import pickle
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from rdkit.Chem import rdFMCS
import pandas as pd
import numpy as np
from sklearn.base import clone

reg_ = RandomForestRegressor(n_estimators=n_estimators,
                                max_features=max_features,
                                max_depth=max_depth,
                                min_samples_leaf=min_samples_leaf)
params = {"target_SMILES": target_SMILES,
          "ACQF": acqf,
          "Batch_Size": batch,
          "Start": start,
          "n_repet": n_repet,
          "distance_balance": distance_balance,
          "feature_choice": feature_choice,
          "n_estimators": n_estimators,
          "max_features": max_features,
          "max_depth": max_depth,
          "min_samples_leaf": min_samples_leaf,
          "model": model,
          "selection_strategy": selection_strategy,
          "folder_for_descriptors": df_folder,
          "alpha": alpha}

print(params, flush=True)

target_df = pd.DataFrame({"Reactant_SMILES": [target_SMILES]*len(products), "Product_SMILES": products, "rxn_ID":[0], "Selectivity (%)": sels})
target_df.to_csv("target.csv")
dfs = []
if feature_choice in ["xtb", "custom", "selected"]:
    target_df = md.prepare_reactivity_mapping('XTB', file="target.csv", 
                                            preprocess=True,
                                            normalize=False, threshold_correlated=1,
                                            rxn_folder="target_data", atom=atom)
    dfs.append(target_df)
if feature_choice in ["bde", "custom", "selected"]:
    target_df = md.prepare_reactivity_mapping('BDE', file="target.csv", 
                                            preprocess=True,
                                            normalize=False, threshold_correlated=1,
                                            rxn_folder="target_data", atom=atom)
    dfs.append(target_df)
if feature_choice in ["gas", "custom", "selected"]:
    target_df = md.prepare_reactivity_mapping('Gasteiger', file="target.csv", 
                                            preprocess=True,
                                            normalize=False, threshold_correlated=1,
                                            rxn_folder="target_data", atom=atom)
    dfs.append(target_df)
if feature_choice in ["env1", "selected"]:
    target_df = md.prepare_reactivity_mapping('ENV-1', file="target.csv", 
                                            preprocess=True,
                                            normalize=False, threshold_correlated=1,
                                            rxn_folder="target_data", atom=atom)
    dfs.append(target_df)
if feature_choice in ["env2", "selected"]:
    target_df = md.prepare_reactivity_mapping('ENV-2', file="target.csv", 
                                            preprocess=True,
                                            normalize=False, threshold_correlated=1,
                                            rxn_folder="target_data", atom=atom)
    dfs.append(target_df)
if feature_choice in ["dbstep", "selected"]:
    target_df = md.prepare_reactivity_mapping('DBSTEP', file="target.csv", 
                                            preprocess=True,
                                            normalize=False, threshold_correlated=1,
                                            rxn_folder="target_data", atom=atom)
    dfs.append(target_df)


df = a.make_descriptors_basic(option=feature_choice, df_folder=df_folder)

import json
with open('../../data/reaction_data/clusters.json') as json_file:
    clusters = json.load(json_file)

# make sure all smiles are in a cluster
def smi_to_cluster(s, clusters):
    for k in clusters:
        if s in clusters[k]:
            return k
cluster = smi_to_cluster(target_SMILES, clusters)
if cluster is not None and cluster != "misc":
    cluster_smis = clusters[cluster]
    df = df.loc[~np.isin(df.Reactant_SMILES, cluster_smis)]

if len(dfs) == 1:
    targ = dfs[0]
else:
    targ = pd.DataFrame()
    for col in df.columns:
        for df_ in dfs:
            if col in df_.columns:
                targ[col] = df_[col]

df_c_ox = pd.DataFrame()
print("Normalized dataframe means:")
norms = {}
for col in df.columns:
    if col in ["Reactant_SMILES", "Atom_nº", "Selectivity", "Reactive Atom", "DOI"]:
        continue
    df_c_ox[col] = (df[col] - np.mean(df[col]))/(max(df[col]) - min(df[col]))
    norms[col] = [np.mean(df[col]), max(df[col]), min(df[col])]
    print(col, np.mean(df_c_ox[col]))

if thresh_corr != None: # remove correlated features
    df_c_ox = md.remove_correlated_features(df_c_ox, False, thresh_corr)

df_c_ox["Reactant_SMILES"] = df["Reactant_SMILES"]
df_c_ox["Atom_nº"] = df["Atom_nº"]
df_c_ox["Selectivity"] = df["Selectivity"]
df_c_ox["Reactive Atom"] = df["Reactive Atom"]
df_c_ox["DOI"] = df["DOI"]

df = df_c_ox

target_df = pd.DataFrame()
print("Normalized target dataframe means:")
for col in df.columns:
    if col == "DOI":
        continue
    if col in ["Reactant_SMILES", "Atom_nº", "Selectivity", "Reactive Atom"]:
        target_df[col] = targ[col]
    else:
        target_df[col] = (targ[col] - norms[col][0])/(norms[col][1] - norms[col][2])
        print(col, np.mean(target_df[col]))
        
df = pd.concat([df, target_df])
df = df.dropna(subset=['Reactant_SMILES'], axis=0)

if not include_large:
    df    = a.remove_large_molecules(df, target_SMILES)

if start == "cold":
    target_mol      =  Chem.MolFromSmiles(target_SMILES)
    training_SMILES = df[df.Reactant_SMILES != target_SMILES].Reactant_SMILES.unique()
    training_mols   = [Chem.MolFromSmiles(smi) for smi in training_SMILES]
    mcs_res         = [rdFMCS.FindMCS([mol, target_mol], ringMatchesRingOnly=True, completeRingsOnly=True, bondCompare=rdFMCS.BondCompare.CompareOrderExact, timeout=60).numAtoms for mol in training_mols]
    idx             = np.argmax(mcs_res)
    initial = training_SMILES[idx]
    training_SMILES = [training_SMILES[idx]]

    df_training = df.loc[df['Reactant_SMILES'] == training_SMILES[0]]
    df_remaining  = df.loc[df['Reactant_SMILES'] != training_SMILES[0]]

if start == "warm":
    dois_to_start_with = ['10.1021/ja00199a039', '10.1021/jo9604189']
    df_training   = df.loc[df['DOI'].isin(dois_to_start_with)]
    df_remaining  = df.loc[df['DOI'].isin(dois_to_start_with) == False]
    training_SMILES = df_training.Reactant_SMILES.unique()
    initial = None

training_SMILES = df_training.Reactant_SMILES.unique()

reg = clone(reg_)
if selection_strategy == "simple":
    t5, smis, y, max_aqcf_score = a.evaluate_acqf(target_SMILES, training_SMILES, reg, df,
            acqf, batch_size=batch, distance_balance=distance_balance, n_runs=1, n_repeat=n_repet, alpha=alpha)
params["cols"] = df_remaining.columns

with open(f"{path}/res_rf_{s}_{acqf}_{run}_{batch}_{start}start_{feature_choice}.pkl", "wb") as f:
    pickle.dump([t5, [initial] + smis, y, max_aqcf_score, params], f)


