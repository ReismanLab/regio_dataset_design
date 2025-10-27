import pandas as pd
import sys
import os
from rdkit import Chem
from sklearn.ensemble import RandomForestRegressor
import argparse

root = os.getcwd()

try:
    base_cwd = os.getcwd().split('regio_dataset_design')[0]
    base_cwd = f"{base_cwd}/regio_dataset_design"
except:
    raise ValueError("You are not in the right directory, need to be in the 'notebooks' directory or subdirectory of it.")

sys.path.append(f"{base_cwd}/utils/")
import modelling as md

pd.set_option('mode.chained_assignment', None)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) 

## parameterss
parser = argparse.ArgumentParser(description='Performance of models on big molecules')
parser.add_argument('--run',
                    help='name of the folder to put the results in, if already taken will abort')
parser.add_argument('--df_folder',
                    help='name of the folder to read the csvs from: \n --preprocessed_borylation_reactions \n --preprocessed_dioxirane_reactions',
                    default='preprocessed_dioxirane_reactions')
parser.add_argument('--model',
                    help="list of the models to use, can be all or a subset of 'RF-OPT-XTB', 'RF2', 'MLP', 'MLP2', 'SVR', 'KNN', 'GPR', 'LR' or 'all'",
                    default='RF2 KNN SVR LR')
parser.add_argument('--y',
                    help="observable to predict. Any descriptor, or Selectivity by default",
                    default='Selectivity')

args = parser.parse_args()
if args.run == None:
    print("Please provide a name for the run")
    sys.exit()
else:
    run = args.run

rxn_folder = args.df_folder

if rxn_folder in ['preprocessed_borylation_reactions',  'preprocessed_borylation_reactions_unnorm']:
    out_folder = f"borylation/{run}"
elif rxn_folder in ['preprocessed_dioxirane_reactions', 'preprocessed_reactions_no_unspec_no_intra_unnorm']:
    out_folder = f"dioxirane/{run}"
else:
    print("Unexpected folder name.")
    sys.exit()

if os.path.exists(f"{base_cwd}/results/model_validation/regression/large_mol/{out_folder}"):
    print(f"Folder {out_folder} already exists, please provide a different name")
    sys.exit()
else:
    os.makedirs(f"{base_cwd}/results/model_validation/regression/large_mol/{out_folder}")

models = args.model.split()
if models == ['all']:
    models = ['RF-OPT-XTB', 'RF2', 'MLP', 'MLP2', 'SVR', 'KNN', 'GPR', 'LR']
print(f"Will be using the following models: {models}")

obs = args.y
print(f"Will be predicting {obs}")

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

dict_models =  {'RF-OPT-XTB' : rfoptxtb,
           'RF2'        : rf2,
           'MLP'        : mlp,
           'MLP2'       : mlp2,
           'SVR'        : svr,
           'KNN'        : knn,
           'GPR'        : gpr,
           'LR'         : lr}

models = {m: dict_models[m] for m in models}

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

feats_to_drop = ["DOI"]
if obs != "Selectivity": 
    print("Dropping Selectivity...")
    feats_to_drop.append("Selectivity")

features = {'BDE'       : df_bde.drop(feats_to_drop, axis=1),
            'XTB'       : df_xtb.drop(feats_to_drop, axis=1),
            'DBSTEP'    : df_dbs.drop(feats_to_drop, axis=1), 
            'Gasteiger' : df_gas.drop(feats_to_drop, axis=1),
            'ENV-1'     : df_en1.drop(feats_to_drop, axis=1),
            'ENV-1-OHE' : df_en1_ohe.drop(feats_to_drop, axis=1),
            'ENV-2'     : df_en2.drop(feats_to_drop, axis=1),
            'Rdkit-Vbur': df_rdkVbur.drop(feats_to_drop, axis=1),
            'Selected'  : df_sel.drop(feats_to_drop, axis=1),
            'Custom'    : df_custom.drop(feats_to_drop, axis=1)
            }

# find obs if it's not selectivity
if obs != "Selectivity": 
    obs_col = None # search for obs in all descriptor dataframes
    for f in features:
        if obs in features[f].columns:
            obs_col = features[f][obs]
    if obs_col is None:
        assert False, "Observable not found in any descriptor dataframe, exiting."

    for f in features: # add obs to all descriptor dataframes
        features[f][obs] = obs_col

# update Reactive Atom col
f = features[list(features.keys())[0]]
top_at = []
for s in f.Reactant_SMILES.unique():
    f_sub = f.loc[f.Reactant_SMILES == s]
    f_sub = f_sub.sort_values(obs, ascending=False)
    reactive_at = f_sub.loc[:, 'Atom_nº'].values[0]
    top_at.extend([reactive_at] * len(f_sub))

for feat in features: # add obs to all descriptor dataframes
    features[feat]["Reactive Atom"] = top_at

# drop features without any descriptors
bad_descs = []
for feat in features:
    if len(features[feat].columns) == len([obs, 'Reactant_SMILES', 'Atom_nº', 'Reactive Atom']):
        bad_descs.append(feat)

for feat in bad_descs:
    del features[feat]

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

## produce results    

# first print baseline:
df_test = f[f.Reactant_SMILES.isin(big_smiles)]
print(f"Baseline: {md.baseline(df_test, feat=obs)}, tested on {len(big_smiles)}/{len(df_test.Reactant_SMILES.unique())} big molecules")

for m, model in models.items():
    print(f"Model: {m}")
    for feature, df_f in features.items():
        print(f"Feature: {feature}")
        reg_     = md.train_model(small_smiles, df_f, model, feat=obs)
        TOP5     = []
        SMILES   = []
        Rel_reac = []
        Y_pred   = []
        for smiles in big_smiles:
            _, y_pred   = md.predict_site(reg_, smiles, df_f, classif=False, feat=obs)
            df_f_       = df_f[df_f.Reactant_SMILES == smiles]
            df_f_[f'Predicted_{obs}'] = [y_pred[x] for x in df_f_['Atom_nº']]
            top5 = []
            
            df_sub    =  df_f_.sort_values(by=obs, ascending=False)
            top1      = df_sub['Atom_nº'].iloc[0]
            for i in [1, 2, 3, 4, 5]:
                df_sub = df_f_.sort_values(by=f'Predicted_{obs}', ascending=False)
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