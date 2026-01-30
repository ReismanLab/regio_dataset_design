import pandas as pd
import sys
import os
from rdkit import Chem

root = os.getcwd()
try:
    base_cwd = os.getcwd().split('regio_dataset_design')[0]
    base_cwd = f"{base_cwd}/regio_dataset_design"
except:
    raise ValueError("You are not in the right directory, need to be in the 'notebooks' directory or subdirectory of it.")

sys.path.append(f"{base_cwd}/utils/")
import modelling as md
import metrics as mt

pd.set_option('mode.chained_assignment', None)
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) 

import argparse
### Parse arguments
parser = argparse.ArgumentParser(description='Descriptor computation')
parser.add_argument('--run',
                    help='name of the folder to put the results in, if already taken will abort')
parser.add_argument('--df_folder',
                    help='name of the folder to read the csvs from',
                    default='preprocessed_dioxirane_reactions')
parser.add_argument('--rxn',
                    help='name of the rxn folder to read the csvs from',
                    default='dioxirane')
parser.add_argument('--model',
                    help="list of the models to use, can be \"all\" or a subset of 'RF-OPT-XTB', 'RF2', 'MLP', 'MLP2', 'SVR', 'KNN', 'GPR', 'LR' or 'all'",
                    default='RF2 KNN SVR LR')
parser.add_argument('--desc',
                    help="list of the models to use, can be \"all\" or a subset of 'BDE', 'Gasteiger', 'DBSTEP', 'XTB', 'Selected', 'Custom', 'ENV-1', 'ENV-2', 'ENV-1-OHE', 'Rdkit-Vbur' or 'all'",
                    default='XTB Selected Custom')
parser.add_argument('--y',
                    help="The variable to predict, can be a descriptor or Selectivity (default)",
                    default='Selectivity')

args = parser.parse_args()
if args.df_folder is None:
    print(f"Taking default folder preprocessed_dioxirane_reactions")

df_folder = str(args.df_folder)
if df_folder not in os.listdir(f"{base_cwd}/data/descriptors/"):
    print(f"df_folder {df_folder} not found in data/descriptors/ Exiting...")
    exit()

rxn_folder = str(args.rxn)    
print(f"Reading csvs from {df_folder} folder")
print(f"Writing csvs in {rxn_folder} folder")

# make folder for results
if args.run is None:
    print(f"Please specify a folder with --run folder_name")
    exit()

folder = str(args.run)
os.chdir(f"{base_cwd}/results/model_validation/regression/loo/{rxn_folder}")
if folder in os.listdir('.'):
    print(f"{folder} is already present, please choose another name. Exiting...")
    exit()
os.mkdir(folder)
os.chdir(root)

models = args.model.split()
if models == ['all']:
    models = ['RF-OPT-XTB', 'RF2', 'MLP', 'MLP2', 'SVR', 'KNN', 'GPR', 'LR']
print(f"\n        Will be using the following models: {models}\n")

desc = args.desc.split()
if desc == ['all']:
    desc = ['BDE', 'Gasteiger', 'DBSTEP', 'XTB', 'Selected', 'Custom', 'ENV-1', 'ENV-2', 'ENV-1-OHE', 'Rdkit-Vbur']
print(f"        Will be using the following descriptors: {desc}\n")

obs = args.y
print(f"        Will be predicting {obs}")

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

dict_models =  {'LR'         : lr,
           'RF-OPT-XTB' : rfoptxtb,
           'RF2'        : rf2,
           'SVR'        : svr,
           'KNN'        : knn,
           'MLP'        : mlp,
           'MLP2'       : mlp2,
           'GPR'        : gpr,
           }

models = {m: dict_models[m] for m in models}

## import data - featurization
dict_descs = {"BDE"        : "df_bde.csv",
              "Gasteiger"  : "df_gas.csv",
              "DBSTEP"     : "df_dbstep.csv",
              "XTB"        : "df_xtb.csv",
              "Selected"   : "df_selected.csv",
              "Custom"     : "df_custom.csv",
              "ENV-1"      : "df_en1.csv",
              "ENV-1-OHE"  : "df_en1_ohe.csv",
              "ENV-2"      : "df_en2.csv",
              "Rdkit-Vbur" : "df_rdkVbur.csv"
              }

def find_reactive_at(f, obs):
    top_at = []
    for s in f.Reactant_SMILES.unique():
        f_sub = f.loc[f.Reactant_SMILES == s]
        f_sub = f_sub.sort_values(obs, ascending=False)
        reactive_at = f_sub.loc[:, 'Atom_nº'].values[0]
        top_at.extend([reactive_at] * len(f_sub))
    return top_at

features = {}
for d in desc:
    try:
        df_temp = pd.read_csv(f"{base_cwd}/data/descriptors/{args.df_folder}/{dict_descs[d]}", index_col=0)
    except:
        print(f"Descriptor {d} not found, skipping...")
        continue

    if "Reactive Atom" not in df_temp.columns: # handling for new observables
        df_temp["Reactive Atom"] = find_reactive_at(df_temp, obs)
        df_temp.to_csv(f"{base_cwd}/data/descriptors/{args.df_folder}/{dict_descs[d]}")

    df_temp.drop(columns=["DOI"], inplace=True, errors='ignore')
    features[d] = df_temp


## performance evaluation
# first print baseline:
f = features[list(features.keys())[0]] # select any feature set
df_test = f.copy()
print(f"Baseline: {md.baseline(df_test, feat=obs)}, tested on {len(df_test.Reactant_SMILES.unique())} reactions")

results = pd.DataFrame(columns=['Model', 'Feature', 'TOP-1', 'TOP-2', 'TOP-3', 'TOP-5', 'TOP-AVG'])

for m, model in models.items():
    for f, feature in features.items():
        p, df_p = md.leave_one_out(feature, model, feat=obs)
        print(f"{m} - {f}: {p}")
        df_p.to_csv(f"{base_cwd}/results/model_validation/regression/loo/{rxn_folder}/{folder}/pred_loo_{m}_{f}.csv")
        top_n = []
        for i in [1,2,3,5]:
            top_n.append(md.get_top_n_accuracy(df_p, i, feat=obs))

        print(f"TOP-1: {top_n[0]}, TOP-2: {top_n[1]}, TOP-3: {top_n[2]}, TOP-5: {top_n[3]}, TOP-AVG: {mt.top_avg(df_p, feat=obs)}")
        results = results.append({'Model'   : m,
                                  'Feature' : f,
                                  'TOP-1'   : top_n[0],
                                  'TOP-2'   : top_n[1],
                                  'TOP-3'   : top_n[2],
                                  'TOP-5'   : top_n[3],
                                  'TOP-AVG' : mt.top_avg(df_p, feat=obs)
                                  }, ignore_index=True)
        
results.to_csv(f"{base_cwd}/results/model_validation/regression/loo/{rxn_folder}/{folder}/df_results.csv")