## imports
import sys
import os
root = os.getcwd()

try:
    base_cwd = root.split('regiochem')[0]
    base_cwd = f"{base_cwd}regiochem"
except:
    raise ValueError("You are not in the right directory, need to be in the 'notebooks' directory or subdirectory of it.")

sys.path.append(f"{base_cwd}/utils/")

from sklearn.base import clone
import pandas as pd
import modelling as md
import visualization as viz
import pandas as pd
from rdkit import Chem
import numpy as np
import argparse
import ast

## parse parameters

parser = argparse.ArgumentParser(description='Make_image_from_prediction')
parser.add_argument('--smi',
                    help='the SMILES of the molecule you want to make prediction on')
parser.add_argument('--training',
                    help='the type of model training you want to use, e.g. loo or large_mol')
parser.add_argument('--model',
                    help='the type of model you want to use, e.g. rf, mlp, svr, knn, gpr')
parser.add_argument('--desc',
                    help='the descriptor you want to use, e.g. Custom, Selected, BDEs, etc.')
parser.add_argument('--reaction',
                    help='The reaction type you want to make prediction on: e.g. borylation or dioxirane')
parser.add_argument('--run',
                    help='The subfolder you want the predictions from, can be set to average if you want to average the predictions from all runs')
parser.add_argument('--figtype',
                    help='The type of figure you want to generate, can be set to numbers or colors')


args = parser.parse_args()
if args.smi == None:
    raise ValueError("SMILES not provided, please provide a SMILES string")
else:
    smiles = args.smi
    try:
        smiles = Chem.CanonSmiles(smiles)
        print(f"\nSMILES used: {smiles}\n")
    except:
        raise ValueError("SMILES string provided is not valid")
    
if args.training == None:
    training = 'loo'
    prefix   = 'pred_loo'
    print(f"\nTraining type not provided, using {training}\n")
elif args.training not in ['loo', 'large_mol']:
    raise ValueError("Training type not found in the list of available training types: 'loo', 'bm'")
else:
    training = args.training
    if training == 'loo':
        prefix = 'pred_loo'
    else:
        prefix = 'eval_bm'
    print(f"\nTraining type used: {training}\n")

if args.model == None:
    model = 'RF2'
    print(f"\nModel not provided, using {model}\n")
else:
    model = args.model
    if model not in ['GPR', 'KNN', 'LR', 'MLP', 'MLP2', 'RF-OPT-XTB', 'RF2', 'SVR']:
        raise ValueError(f"Model {model} not found in the list of available models: 'GPR', 'KNN', 'LR', 'MLP', 'MLP2', 'RF-OPT-XTB', 'RF2', 'SVR'")
    else:
        print(f"\nModel used: {model}\n")

if args.desc == None:
    desc = 'Custom'
    print(f"No descriptors provided, will use Custom\n")
elif args.desc not in ['Custom', 'Selected', 'BDE', 'Rdkit-Vbur', 'XTB', 'ENV-1', 'ENV-1-OHE', 'ENV-2', 'DBSTEP', 'Gasteiger']:
    raise ValueError("Descriptor not found in the list of available descriptors: 'Custom', 'Selected', 'BDE', 'Rdkit-Vbur', 'XTB', 'ENV-1', 'ENV-1-OHE', 'ENV-2', 'DBSTEP', 'Gasteiger'")
else:
    desc = args.desc
    print(f"Descriptos used: {desc}\n")

if args.reaction == None:
    reaction = 'dioxirane'
    print(f"No reaction type provided, will use dioxirane\n")
elif args.reaction not in ['borylation', 'borylation_filt', 'dioxirane']:
    raise ValueError("Reaction type not found in the list of available reactions: 'borylation', 'dioxirane'")
else:
    reaction = args.reaction
    print(f"\nReaction type used: {reaction}\n")

if args.run == None:
    run = 'run_01'
    print(f"No run provided, will use {run}\n")
else:
    run = args.run
    print(f"\nRun used: {run}\n")

if args.figtype == None:
    figtype = 'colors'
    print(f"No figure type provided, will use {figtype}\n")
elif args.figtype not in ['numbers', 'colors']:
    raise ValueError("Figure type not found in the list of available figure types: 'numbers', 'colors'")
else:
    figtype = args.figtype
    print(f"Figure type used: {figtype}\n")

## read results

if run != 'average':
    if training == 'loo':
        df_res = pd.read_csv(f"{base_cwd}/results/model_validation/regression/{training}/{reaction}/{run}/{prefix}_{model}_{desc}.csv", index_col=0)  
        df_smi = df_res[df_res.Reactant_SMILES == smiles]
        if df_smi.empty:
            raise ValueError(f"SMILES {smiles} not found in the dataset")
        y_pred = {at_idx: pred for at_idx, pred in zip(df_smi["Atom_nº"].values, df_smi.Predicted_Selectivity.values)}
    elif training == 'large_mol':
        df_res = pd.read_csv(f"{base_cwd}/results/model_validation/regression/{training}/{reaction}/{run}/{prefix}_{desc}_{model}.csv", index_col=0)
        df_smi = df_res[df_res.index == smiles]
        if df_smi.empty:
            raise ValueError(f"SMILES {smiles} not found in the dataset")
        y_pred = ast.literal_eval(df_smi["Y_pred"].values[0])

    else:
        raise ValueError("Training type not found in the list of available training types: 'loo', 'bm'")
    

else:
    if  training == 'loo':
        runs = os.listdir(f"{base_cwd}/results/model_validation/regression/{training}/{reaction}/")
        runs = [run for run in runs if 'run' in run]
        dfs = []
        for run in runs:
            try:
                df_res = pd.read_csv(f"{base_cwd}/results/model_validation/regression/{training}/{reaction}/{run}/{prefix}_{model}_{desc}.csv", index_col=0)
                df_smi = df_res[df_res.Reactant_SMILES == smiles]
                if df_smi.empty:
                    raise ValueError(f"SMILES {smiles} not found in the dataset")
                y_pred = {at_idx: pred for at_idx, pred in zip(df_smi["Atom_nº"].values, df_smi.Predicted_Selectivity.values)}
                dfs.append(y_pred)
            except:
                print(f"Could not read {reaction}/{run}/{prefix}_{model}_{desc}.csv")

        y_pred = {at_idx: np.mean([d[at_idx] for d in dfs]) for at_idx in dfs[0].keys()}
    
    elif training == 'large_mol':
        runs = os.listdir(f"{base_cwd}/results/model_validation/regression/{training}/{reaction}/")
        runs = [run for run in runs if 'run' in run]
        dfs = []
        for run in runs:
            try:
                df_res = pd.read_csv(f"{base_cwd}/results/model_validation/regression/{training}/{reaction}/{run}/{prefix}_{desc}_{model}.csv", index_col=0)
                df_smi = df_res[df_res.index == smiles]
                if df_smi.empty:
                    raise ValueError(f"SMILES {smiles} not found in the dataset")
                y_pred = ast.literal_eval(df_smi["Y_pred"].values[0])
                print(y_pred)
                dfs.append(y_pred)
            except:
                print(f"Could not read {reaction}/{run}/{prefix}_{desc}_{model}.csv")

        y_pred = {at_idx: np.mean([d[at_idx] for d in dfs]) for at_idx in dfs[0].keys()}
    
    else:
        raise ValueError("Training type not found in the list of available training types: 'loo', 'bm'")

if figtype == "numbers":
    img_pred_rank = viz.visualize_regio_pred(smiles, y_pred, draw="numbers")
    img_pred_rank.save(f"temp.png")
if figtype == "colors":
    img_pred_rank = viz.visualize_regio_pred(smiles, y_pred, draw="colors")
    img_pred_rank.savefig(f"temp.png", bbox_inches='tight', dpi=300)
