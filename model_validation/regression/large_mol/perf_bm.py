import pandas as pd
import ast
import sys
import os
from rdkit import Chem
from tqdm import tqdm
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
from datetime import date
import argparse

root = os.getcwd()
print(os.getcwd())
try:
    base_cwd = os.getcwd().split('regiochem')[0]
    base_cwd = f"{base_cwd}/regiochem"
except:
    raise ValueError("You are not in the right directory, need to be in the 'notebooks' directory or subdirectory of it.")

sys.path.append(f"{base_cwd}/utils/")
import modelling as md

pd.set_option('mode.chained_assignment', None)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) 

## params
parser = argparse.ArgumentParser(description='Performance of models on big molecules')
parser.add_argument('--run',
                    help='name of the folder to put the results in, if already taken will abort')
parser.add_argument('--df_folder',
                    help='name of the folder to read the csvs from: \n --preprocessed_borylation_reactions \n --preprocessed_reactions',
                    default='preprocessed_reactions_no_unspec_no_intra')

args = parser.parse_args()
if args.run == None:
    print("Please provide a name for the run")
    sys.exit()
else:
    run = args.run

if args.df_folder == None:
    print("Please provide a folder to read the data from: \n --preprocessed_borylation_reactions \n --preprocessed_reactions \n --preprocessed_reactions_no_unspec_center")
    sys.exit()
else:
    rxn_folder = args.df_folder

if args.df_folder in ['preprocessed_borylation_reactions',  'preprocessed_borylation_reactions_unnorm']:
    out_folder = f"borylation/{run}"
elif args.df_folder in ['preprocessed_reactions', 'preprocessed_reactions_no_unspec_center', 'preprocessed_reactions_no_unspec_no_intra']:
    out_folder = f"dioxirane/{run}"
else:
    print("Unexpected folder name.")
    sys.exit()

if os.path.exists(f"{base_cwd}/results/model_validation/regression/large_mol/{out_folder}"):
    print(f"Folder {out_folder} already exists, please provide a different name")
    sys.exit()
else:
    os.makedirs(f"{base_cwd}/results/model_validation/regression/large_mol/{out_folder}")


## import models
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

rf2 = RandomForestRegressor(n_estimators=250,
                            max_features=0.5,
                            max_depth=10,
                            min_samples_leaf=3)
rfoptxtb = RandomForestRegressor(**{'bootstrap': True, 
                                    'max_depth': 13, 
                                    'max_features': 'log2', 
                                    'min_samples_leaf': 1, 
                                    'min_samples_split': 2, 
                                    'n_estimators': 500})
mlp = MLPRegressor(hidden_layer_sizes=(20,), 
                   activation='logistic', 
                   solver='lbfgs',
                   max_iter=100000, 
                   learning_rate='adaptive', 
                   learning_rate_init=0.01,)
mlp2 = MLPRegressor(hidden_layer_sizes=(200,20), 
                   activation='logistic', 
                   solver='lbfgs',
                   max_iter=100000, 
                   learning_rate='adaptive', 
                   learning_rate_init=0.001,)
svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
knn = KNeighborsRegressor(n_neighbors=50, weights='distance')
gpr = GaussianProcessRegressor(kernel=Matern(length_scale=2.0, nu=1.5),
                                n_restarts_optimizer=3, 
                                normalize_y=True)
lr  = LinearRegression()

models =  {'RF-OPT-XTB' : rfoptxtb,
           'RF2'        : rf2,
           'MLP'        : mlp,
           'MLP2'       : mlp2,
           'SVR'        : svr,
           'KNN'        : knn,
           'GPR'        : gpr,
           'LR'         : lr}

## import data - featurization
df_xtb     = pd.read_csv(f"{base_cwd}/data/descriptors/{rxn_folder}/df_xtb.csv", index_col=0)
df_gas     = pd.read_csv(f"{base_cwd}/data/descriptors/{rxn_folder}/df_gas.csv", index_col=0)
df_dbs     = pd.read_csv(f"{base_cwd}/data/descriptors/{rxn_folder}/df_dbstep.csv", index_col=0)
df_en1     = pd.read_csv(f"{base_cwd}/data/descriptors/{rxn_folder}/df_en1.csv", index_col=0)
df_en2     = pd.read_csv(f"{base_cwd}/data/descriptors/{rxn_folder}/df_en2.csv", index_col=0)
df_bde     = pd.read_csv(f"{base_cwd}/data/descriptors/{rxn_folder}/df_bde.csv", index_col=0)
df_rdkVbur = pd.read_csv(f"{base_cwd}/data/descriptors/{rxn_folder}/df_rdkVbur.csv", index_col=0)
df_sel     = pd.read_csv(f"{base_cwd}/data/descriptors/{rxn_folder}/df_selected.csv", index_col=0)
df_custom  = pd.read_csv(f"{base_cwd}/data/descriptors/{rxn_folder}/df_custom.csv", index_col=0)
df_en1_ohe = pd.read_csv(f"{base_cwd}/data/descriptors/{rxn_folder}/df_en1_ohe.csv", index_col=0)

features = {#'AIMNET-A'  : df_a,
            #'AIMNET-AIM': df_aim,
            'BDE'       : df_bde.drop(["DOI"], axis=1),
            'XTB'       : df_xtb.drop(["DOI"], axis=1),
            'DBSTEP'    : df_dbs.drop(["DOI"], axis=1), 
            'Gasteiger' : df_gas.drop(["DOI"], axis=1),
            'ENV-1'     : df_en1.drop(["DOI"], axis=1),
            'ENV-1-OHE' : df_en1_ohe.drop(["DOI"], axis=1),
            'ENV-2'     : df_en2.drop(["DOI"], axis=1),
            'Rdkit-Vbur': df_rdkVbur.drop(["DOI"], axis=1),
            'Selected'  : df_sel.drop(["DOI"], axis=1),
            'Custom'    : df_custom.drop(["DOI"], axis=1)
            }

## get big smiles
smiles = df_custom.Reactant_SMILES.unique()
smiles = set([Chem.CanonSmiles(s) for s in smiles])

big_smiles   = []
small_smiles = []
for smiles in smiles:
    mol = Chem.MolFromSmiles(smiles)
    num_C = [atom.GetAtomicNum() for atom in mol.GetAtoms()].count(6)
    if num_C > 15:
        big_smiles.append(smiles)
    else:
        small_smiles.append(smiles)
#big_smiles = [Chem.CanonSmiles('[H][C@]1(C2)[C@@]([C@@H](OC(C)=O)[C@H](OC(C)=O)C3=C(C)[C@@H](OC(C)=O)C[C@]2([H])C3(C)C)(C)[C@@H](OC(C)=O)C[C@H](O[Si](CC)(CC)CC)[C@@]14OC4')]

## produce results    

# first print baseline:
df_test = df_bde[df_bde.Reactant_SMILES.isin(big_smiles)]
print(f"Baseline: {md.baseline(df_test)}, tested on {len(big_smiles)}/{len(df_test)} big molecules")

for m, model in models.items():
    print(f"Model: {m}")
    for feature, df_f in features.items():
        print(f"Feature: {feature}")
        reg_     = md.train_model(small_smiles, df_f, model)
        TOP5     = []
        SMILES   = []
        Rel_reac = []
        Y_pred   = []
        for smiles in big_smiles:
            _, y_pred   = md.predict_site(reg_, smiles, df_f, classif=False)
            df_f_       = df_f[df_f.Reactant_SMILES == smiles]
            df_f_['Predicted_Selectivity'] = [y_pred[x] for x in df_f_['Atom_nº']]
            top5 = []
            
            df_sub    =  df_f_.sort_values(by='Selectivity', ascending=False)
            top1      = df_sub['Atom_nº'].iloc[0]
            for i in [1, 2, 3, 4, 5]:
                df_sub = df_f_.sort_values(by='Predicted_Selectivity', ascending=False)
                pred_atoms = df_sub['Atom_nº'].values
                if top1 in pred_atoms[:i]:
                    top5.append(1)
                else:
                    top5.append(0)

            TOP5.append(top5)
            SMILES.append(smiles)
            Y_pred.append(y_pred)
        df_top5 = pd.DataFrame(index=SMILES, data=TOP5)
        df_top5["Y_pred"] = Y_pred
        df_top5.to_csv(f"{base_cwd}/results/model_validation/regression/large_mol/{out_folder}/eval_bm_{feature}_{m}.csv")
