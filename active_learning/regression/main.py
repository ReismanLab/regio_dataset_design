# add path to utils
import sys
import os
root = os.getcwd()

try:
    base_cwd = root.split('regio_dataset_design')[0]
    base_cwd = f"{base_cwd}regio_dataset_design"
except:
    raise ValueError("You are not in the right directory, need to be in the 'notebooks' directory or subdirectory of it.")

sys.path.append(f"{base_cwd}/utils/")

### Parse arguments
import argparse
parser = argparse.ArgumentParser(description='Acquisition function result computation')
parser.add_argument('--smi',
                    help='Target smiles for acquisition function',
                    default="COC(=O)CC[C@@H](C)[C@H]1CC[C@H]2[C@@H]3CC[C@@H]4C[C@H](OC(C)=O)CC[C@]4(C)[C@H]3CC[C@]12C")
parser.add_argument('--acqf',
                    help='Acquisition function to compute (labels defined in utils/acquisition)',
                    default='acqf_2-1')
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
                    default='test')
parser.add_argument('--run', 
                    help='Name for the run ',
                    default='test')
parser.add_argument('--df_folder',
                    help='Name for the folder where the precomputed descriptors are',
                    default='preprocessed_reactions_no_unspec_no_intra')
parser.add_argument('--alpha',
                    help='balance between uncertainty and reactivity weighting for the target site to orient selection alpha must be between 0 and 2',
                    default=1)

args             = parser.parse_args()
smi              = args.smi
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
alpha            = args.alpha

import os
path = f"{base_cwd}/results/active_learning/regression/{res}"
if not os.path.exists(path):
    os.mkdir(path)

if "/" in smi: # handling for molecules with double bond stereochemistry
    s = smi.replace("/", "-")
else:
    s = smi

if os.path.exists(f"{path}/res_rf_{s}_{acqf}_{run}_{batch}_{start}start_{feature_choice}.pkl"):
    print(f"\n\nSkipping {smi} with {acqf} {run} {batch} {start} {feature_choice}\n\n because already computed!", flush=True)
    exit()


## remaining imports
import acqf as a
import pickle
from sklearn.ensemble import RandomForestRegressor

reg = RandomForestRegressor(n_estimators=n_estimators,
                                max_features=max_features,
                                max_depth=max_depth,
                                min_samples_leaf=min_samples_leaf)
params = {"target_SMILES": smi,
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

def final_eval(smi, aqcf_type, run):
    if "/" in smi: # handling for molecules with double bond stereochemistry 
        s = smi.replace("/", "-")
    else:
        s = smi

    if not os.path.exists(f"{path}/res_rf_{s}_{aqcf_type}_{run}_{batch}_{start}start_{feature_choice}.pkl"):
        print(f"\n\nComputing {smi} with {aqcf_type} {run} {batch} {start} {feature_choice}\n\n", flush=True)
        t5, smis, initial, y, max_aqcf_score, cols = a.benchmark_aqcf_on_smiles(aqcf_type,      # the type of acquisition function
                                                    smi,  # the target smiles
                                                    start, # warm or cold
                                                    reg,
                                                    batch,
                                                    distance_balance,
                                                    n_repet,
                                                    feature_choice=feature_choice,
                                                    selection_strategy=selection_strategy,
                                                    n_runs=1,
                                                    alpha=alpha,
                                                    df_folder=df_folder)
        params["cols"] = cols

        with open(f"{path}/res_rf_{s}_{aqcf_type}_{run}_{batch}_{start}start_{feature_choice}.pkl", "wb") as f:
            pickle.dump([t5, [initial] + smis, y, max_aqcf_score, params], f)
    else:
        print(f"\n\nSkipping {smi} with {aqcf_type} {run} {batch} {start} {feature_choice}\n\n because already computed!", flush=True)

final_eval(smi, acqf, run)
