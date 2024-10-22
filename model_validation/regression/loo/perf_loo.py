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

root = os.getcwd()

try:
    base_cwd = os.getcwd().split('regio_dataset_design')[0]
    base_cwd = f"{base_cwd}/regio_dataset_design"
except:
    raise ValueError("You are not in the right directory, need to be in the 'notebooks' directory or subdirectory of it.")

import argparse
### Parse arguments
parser = argparse.ArgumentParser(description='Descriptor computation')
parser.add_argument('--run',
                    help='name of the folder to put the results in, if already taken will abort')
parser.add_argument('--df_folder',
                    help='name of the folder to read the csvs from',
                    default='preprocessed_reactions_no_unspec_no_intra')
parser.add_argument('--rxn',
                    help='name of the rxn folder to read the csvs from',
                    default='dioxirane')


args = parser.parse_args()
if args.df_folder is None:
    print(f"Taking default folder preprocessed_reactions_no_unspec_no_intra")
    df_folder = str(args.df_folder)
else:
    df_folder = str(args.df_folder)
    if df_folder not in ['preprocessed_borylation_reactions', 'preprocessed_reactions', 'preprocessed_reactions_no_unspec_center', 'preprocessed_reactions_no_unspec_no_intra']:
        print(f"Please specify a folder to take csv from (C_H ox or Borylation) with --df_folder folder_name, folder_name should be:\n - preprocessed_borylation_reactions\n - preprocessed_reactions\n - preprocessed_reactions_no_unspec_center")
        exit()
    elif df_folder == 'preprocessed_borylation_reactions':
        rxn_folder = 'borylation_filt'
    elif df_folder == 'preprocessed_reactions_no_unspec_center':
        rxn_folder = 'dioxirane'
    else:
        rxn_folder = 'dioxirane'

print(f"Reading csvs from {df_folder} folder")
print(f"Writing csvs in {rxn_folder} folder")

if args.run is None:
    print(f"Please specify a folder with --run folder_name")
    exit()
else:
   folder = str(args.run)
   os.chdir(f"{base_cwd}/results/model_validation/regression/loo/{rxn_folder}")
   folders = os.listdir('.')
   print(folders)
   if folder in folders:
       print(f"{folder} is already in {folders}, please choose another name")
       #exit()
   else:
       os.mkdir(folder)
   os.chdir(root)


sys.path.append(f"{base_cwd}/utils/")
import modelling as md
import metrics as mt

pd.set_option('mode.chained_assignment', None)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) 

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
rfoptxtb = RandomForestRegressor(**{'bootstrap': True, 'max_depth': 13, 'max_features': 'log2', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 500})
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

models =  {'LR'         : lr,
           'RF-OPT-XTB' : rfoptxtb,
           'RF2'        : rf2,
           'SVR'        : svr,
           'KNN'        : knn,
#           'MLP'        : mlp,
#           'MLP2'       : mlp2,
#           'GPR'        : gpr,
           }

## import data - featurization
df_xtb     = pd.read_csv(f"{base_cwd}/data/descriptors/{args.df_folder}/df_xtb.csv", index_col=0)
df_gas     = pd.read_csv(f"{base_cwd}/data/descriptors/{args.df_folder}/df_gas.csv", index_col=0)
df_dbs     = pd.read_csv(f"{base_cwd}/data/descriptors/{args.df_folder}/df_dbstep.csv", index_col=0)
df_en1     = pd.read_csv(f"{base_cwd}/data/descriptors/{args.df_folder}/df_en1.csv", index_col=0)
df_en2     = pd.read_csv(f"{base_cwd}/data/descriptors/{args.df_folder}/df_en2.csv", index_col=0)
df_bde     = pd.read_csv(f"{base_cwd}/data/descriptors/{args.df_folder}/df_bde.csv", index_col=0)
df_rdkVbur = pd.read_csv(f"{base_cwd}/data/descriptors/{args.df_folder}/df_rdkVbur.csv", index_col=0)
df_sel     = pd.read_csv(f"{base_cwd}/data/descriptors/{args.df_folder}/df_selected.csv", index_col=0)
df_custom  = pd.read_csv(f"{base_cwd}/data/descriptors/{args.df_folder}/df_custom.csv", index_col=0)
df_en1_ohe = pd.read_csv(f"{base_cwd}/data/descriptors/{args.df_folder}/df_en1_ohe.csv", index_col=0)

features = {#'AIMNET-A'  : df_a,
            #'AIMNET-AIM': df_aim,
            'BDE'       : df_bde.drop(columns=["DOI"]),
            'XTB'       : df_xtb.drop(columns=["DOI"]),
            'DBSTEP'    : df_dbs.drop(columns=["DOI"]), 
            'Gasteiger' : df_gas.drop(columns=["DOI"]),
            'ENV-1'     : df_en1.drop(columns=["DOI"]),
            'ENV-1-OHE' : df_en1_ohe.drop(columns=["DOI"]),
            'ENV-2'     : df_en2.drop(columns=["DOI"]),
            'Rdkit-Vbur': df_rdkVbur.drop(columns=["DOI"]),
            'Selected'  : df_sel.drop(columns=["DOI"]),
            'Custom'    : df_custom.drop(columns=["DOI"])
            }


## performance evaluation
# first print baseline:
df_test = df_bde.copy()
print(f"Baseline: {md.baseline(df_test)}, tested on {len(df_test.Reactant_SMILES.unique())} reactions")

results = pd.DataFrame(columns=['Model', 'Feature', 'TOP-1', 'TOP-2', 'TOP-3', 'TOP-5', 'TOP-AVG'])

for m, model in models.items():
    for f, feature in features.items():
        p, df_p = md.leave_one_out(feature, model)
        print(f"{m} - {f}: {p}")
        df_p.to_csv(f"{base_cwd}/results/model_validation/regression/loo/{rxn_folder}/{folder}/pred_loo_{m}_{f}.csv")
        top_n = []
        for i in [1,2,3,5]:
            top_n.append(md.get_top_n_accuracy(df_p, i))

        print(f"TOP-1: {top_n[0]}, TOP-2: {top_n[1]}, TOP-3: {top_n[2]}, TOP-5: {top_n[3]}, TOP-AVG: {mt.top_avg(df_p)}")
        results = results.append({'Model'   : m,
                                  'Feature' : f,
                                  'TOP-1'   : top_n[0],
                                  'TOP-2'   : top_n[1],
                                  'TOP-3'   : top_n[2],
                                  'TOP-5'   : top_n[3],
                                  'TOP-AVG' : mt.top_avg(df_p)
                                  }, ignore_index=True)
        
results.to_csv(f"{base_cwd}/results/model_validation/regression/loo/{rxn_folder}/{folder}/df_results.csv")
