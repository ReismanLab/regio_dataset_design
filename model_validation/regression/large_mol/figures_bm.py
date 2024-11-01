import seaborn as sns
import matplotlib.pyplot as plt
import os
import ast
import sys
import pandas as pd
import numpy as np
import warnings

pd.set_option('mode.chained_assignment', None)
warnings.simplefilter(action='ignore', category=FutureWarning)

### Parse arguments
recompute = True

import argparse
parser = argparse.ArgumentParser(description='Descriptor computation')
parser.add_argument('--run',
                    help='Folder with results files',
                    default = 'average')
parser.add_argument('--model',
                    help='Model to evaluate the regioselectivity plot',
                    default = 'RF2')
parser.add_argument('--desc',
                    help='Descriptor to use for the regioselectivity plot',
                    default = 'Custom')
parser.add_argument('--vmin',
                    help='vmin')
parser.add_argument('--vmax',
                    help='vmax')
parser.add_argument('--rxn',
                    help='Folder with preprocessed reactions', 
                    default = 'preprocessed_dioxirane_reactions')
parser.add_argument('--draw',
                    help='Type of images: numbers or heatmap with colors: "colors" or "numbers"', 
                    default = "colors")
parser.add_argument('--dataset',
                    help='The dataset the runs were performed on, could be all the data (folders starting by "run_"), only the ones with specified stereo ("run_no_unspec_dia_") or the totally cleaned dataset (no unspecified data, no intra molecular reactions, protonated amines: "clean_run_)"', 
                    default = "clean_run")

args = parser.parse_args()

folder = args.run
rxn    = args.rxn

if rxn == None:
    rxn_folder = "preprocessed_reactions_no_unspec_no_intra"
    rxn        = "dioxirane"
    dataset    = "clean_run"
    print(f"Reaction default: {rxn}")
elif rxn == 'borylation':
    rxn_folder = "preprocessed_borylation_reactions"
    dataset    = "run_"
    print(f"Reaction: {rxn}")
elif rxn == 'dioxirane':
    rxn_folder = "preprocessed_dioxirane_reactions"
    dataset    = "clean_run"  
    print(f"Reaction: {rxn}")
else:
    print(f"Unexpected reaction: {rxn}.\n please give one of these: \n'borylation', 'dioxirane'")
    sys.exit()

if rxn_folder in ['preprocessed_borylation_reactions',  'preprocessed_borylation_reactions_unnorm']:
    rxn          = 'borylation'
    norm_big_mol = 22
    dataset      = "run_"
    print(f"Reaction used: {rxn}")

elif rxn_folder in ['preprocessed_dioxirane_reactions', 'preprocessed_reactions', 'preprocessed_reactions_no_unspec_center', 'preprocessed_reactions_no_unspec_no_intra']:
    rxn          = "dioxirane"
    norm_big_mol = 50
    dataset      = "clean_run"
    print(f"Reaction used: {rxn}")

else:
    print(f"Unexpected folder name: {rxn_folder}.\n please give one of these: \n'preprocessed_reactions', 'preprocessed_reactions_no_unspec_center', 'preprocessed_reactions_no_unspec_no_intra'\n'preprocessed_borylation_reactions',  'preprocessed_borylation_reactions_unnorm'")
    sys.exit()
    
draw = args.draw  

root = os.getcwd()
try:
    base_cwd = os.getcwd().split('regio_dataset_design')[0]
    base_cwd = f"{base_cwd}/regio_dataset_design"
except:
    raise ValueError("You are not in the right directory, need to be in the 'notebooks' directory or subdirectory of it.")

sys.path.append(f"{base_cwd}/utils/")

import metrics as mt    

def get_top_avg(df, df_bde):
    big_smiles = df.index.to_list()
    df_exp_ = df_bde[df_bde.Reactant_SMILES.isin(big_smiles)]
    df_exp  = pd.DataFrame()  
    for smiles in big_smiles:
        df_smi = df_exp_[df_exp_.Reactant_SMILES == smiles]
        y_pred = ast.literal_eval(df.loc[smiles, 'Y_pred'])
        df_smi['Predicted_Selectivity'] = df_smi['Atom_nº'].apply(lambda x: y_pred[x])
        df_exp = pd.concat([df_exp, df_smi])
    try:
        return mt.top_avg(df_exp.drop(columns=['Unnamed: 0']))
    except:
        return mt.top_avg(df_exp)
    
def get_top_n(df, df_bde, n):
    big_smiles = df.index.to_list()
    df_exp_ = df_bde[df_bde.Reactant_SMILES.isin(big_smiles)]
    df_exp  = pd.DataFrame()  
    for smiles in big_smiles:
        df_smi = df_exp_[df_exp_.Reactant_SMILES == smiles]
        y_pred = ast.literal_eval(df.loc[smiles, 'Y_pred'])
        df_smi['Predicted_Selectivity'] = df_smi['Atom_nº'].apply(lambda x: y_pred[x])
        df_exp = pd.concat([df_exp, df_smi])
    out = mt.top_n(df_exp, n)
    return float(np.sum(out))
    
## read results files
def get_df_res(rxn, folder, rxn_folder, base_cwd, recompute=False): # if recompute is True, will recompute the results
    if recompute == False:
        try:
            print(f"{base_cwd}/results/model_validation/regression/large_mol/{rxn}/{folder}/df_results.csv")
            df_results = pd.read_csv(f"{base_cwd}/results/model_validation/regression/large_mol/{rxn}/{folder}/df_results.csv", index_col=0)
        except:
            print(f"Could not read {rxn}/{folder}/df_results.csv")
            recompute = True

    if recompute:
        print(f"Recomputing {rxn}/{folder}/df_results.csv")
        files = os.listdir(f"{base_cwd}/results/model_validation/regression/large_mol/{rxn}/{folder}")
        files = [f"{base_cwd}/results/model_validation/regression/large_mol/{rxn}/{folder}/" + f for f in files if 'eval_bm' in f]
        files = [f for f in files if '.csv' in f]

        df_bde     = pd.read_csv(f"{base_cwd}/data/descriptors/{rxn_folder}/df_bde.csv", index_col=0)
        df_results = pd.DataFrame(columns=['Descriptor', 'Model', 'TOP1', 'TOP2', 'TOP3', 'TOPAVG'])

        for f in files:
            df = pd.read_csv(f, index_col=0)
            feature = f.split('_')[-2]
            model   = f.split('_')[-1].split('.')[0]
            df_results = df_results.append({'Descriptor': feature, 'Model': model,
                                            'TOP1'      : float(get_top_n(df, df_bde, 1)), 
                                            'TOP2'      : float(get_top_n(df, df_bde, 2)),
                                            'TOP3'      : float(get_top_n(df, df_bde, 3)),
                                            'TOPAVG'    : get_top_avg(df, df_bde)}, ignore_index=True)

        df_results.sort_values(by=['Descriptor', 'Model'], ascending=False, inplace=True)
        df_results.to_csv(f"{base_cwd}/results/model_validation/regression/large_mol/{rxn}/{folder}/df_results.csv")
    return df_results

if folder != 'average':
    df_results = get_df_res(rxn, folder, rxn_folder, base_cwd, recompute=recompute)
    df_res_std = None

else:
    folders = os.listdir(f"{base_cwd}/results/model_validation/regression/large_mol/{rxn}")
    folders = [f for f in folders if 'run' in f]
    folders = [f for f in folders if dataset in f]
    if dataset == "run":
       folders = [f for f in folders if f[0] == "r"]
    #folders = [f for f in folders if 'run_03' not in f] # nan in run_03
    df_res_all = []
    for f in folders:
        df_results = get_df_res(rxn, f, rxn_folder, base_cwd, recompute=False)
        df_results.reset_index(inplace=True, drop=True)
        df_results.set_index(['Model', 'Descriptor'],  inplace=True, drop=True)
        df_res_all.append(df_results)
    
    df_res = df_res_all[0].copy()
    # average over all runs
    for i in range(1, len(df_res_all)):
        print(f"There are {df_res_all[i].isna().sum().sum()} nan in dataframe {i}, folder {folders[i]}")
        df_res_all[i].fillna(0, inplace=True) 
        df_res += df_res_all[i]

    df_res = df_res / len(df_res_all)
    df_results = df_res.copy()
    
    def multiply(df1, df2):
        df = pd.DataFrame(df1.values*df2.values, columns=df1.columns, index=df1.index)
        return df
    # compute standard deviation
    df_res_sub = df_res_all[0] - df_results
    df_res_std = multiply(df_res_sub, df_res_sub) 
    for i in range(1, len(df_res_all)):
        df_res_sub = df_res_all[i] - df_results
        df_square = multiply(df_res_sub, df_res_sub) 
        df_res_std += df_square 
        
    df_res_std = df_res_std / len(df_res_all)
    df_res_std = np.sqrt(df_res_std)

    # reset index
    df_res_std.reset_index(inplace=True) 
    df_results.reset_index(inplace=True) 

    # save results
    df_results.to_csv(f"df_mean_runs.csv")
    df_res_std.to_csv(f"df_std_runs.csv")

## plot heatmaps
    
df_bde     = pd.read_csv(f"{base_cwd}/data/descriptors/{rxn_folder}/df_bde.csv", index_col=0)

try:
    os.mkdir(f"{rxn}/{folder}")
except:
    pass

if args.vmin == None:
    vmin = 0
else:
    vmin = int(args.vmin)

if args.vmax == None:
    vmax = len(df_bde.index)
else:
    vmax = int(args.vmax)

def plot_heat_map(df_res_mean, df_res_std, target, rxn, folder, vmin, vmax, reverse = False, round_ = 1): 
     
    table = df_res_mean.pivot(index="Descriptor", columns="Model", values=target)
    table = table.reindex(['BDE', 'Rdkit-Vbur', 'DBSTEP', 'Gasteiger', 'ENV-1', 'ENV-2', 'XTB', 'Custom', 'Selected'])
    table = table.reindex(['RF2', 'RF-OPT-XTB', 'KNN', 'LR', 'MLP2', 'MLP', 'SVR', 'GPR'], axis =1)
    for i in table.columns:
        for j in table.index:
            if target not in ['TOPAVG', 'TOP-AVG']:
                table.loc[j,i] = 100*table.loc[j,i]/norm_big_mol

    if folder == 'average': 
        table_std = df_res_std.pivot(index="Descriptor", columns="Model", values=target)
        table_std = table_std.reindex(['BDE', 'Rdkit-Vbur', 'DBSTEP', 'Gasteiger', 'ENV-1', 'ENV-2', 'XTB', 'Custom', 'Selected'])
        table_std = table_std.reindex(['RF2', 'RF-OPT-XTB', 'KNN', 'LR', 'MLP2', 'MLP', 'SVR', 'GPR'], axis =1)
        for i in table_std.columns:
            for j in table_std.index:
                if target not in ['TOPAVG', 'TOP-AVG']:
                    print(i, j)
                    table_std.loc[j,i] = 100*table_std.loc[j,i]/norm_big_mol

    
    annots = pd.DataFrame(columns=table.columns, index=table.index)
    for i in range(len(annots.index)):
        for j in range(len(annots.columns)):
            if folder == 'average':
                annots.iloc[i,j] = f"{round(table.iloc[i,j], round_)} ± {round(table_std.iloc[i,j], round_)}" 
            else:
                annots.iloc[i,j] = f"{round(table.iloc[i,j], round_)}"

    cmap = "viridis"
    if reverse:
        cmap = "viridis_r"
        
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.heatmap(data = table, 
                annot=annots, 
                cmap=cmap,
                fmt='',
                vmin=vmin, vmax=vmax,
                ax = ax)
    ax.set_title(f"{target} per model and descriptor on large molecules dataset")
    fig.savefig(f"{rxn}/{folder}/heatmap_{target}.png", dpi=300)
    print(f"Saved {rxn}/{folder}/heatmap_{target}.png")

vmin = 0
vmax = 100
plot_heat_map(df_results, df_res_std, 'TOP1', rxn, folder, vmin, vmax)
plot_heat_map(df_results, df_res_std, 'TOP2', rxn, folder, vmin, vmax)
plot_heat_map(df_results, df_res_std, 'TOP3', rxn, folder, vmin, vmax)
plot_heat_map(df_results, df_res_std, 'TOPAVG', rxn, folder, 1, 5, reverse=True, round_=2)


## plot regioselectivity prediction
import visualization as viz
import numpy as np
import matplotlib.image as img
from rdkit import Chem

if args.model == None:
    model = 'RF2'
else:
    print(f"Model: {args.model} for the regioselectivity plot   ")
    model = args.model

if args.desc == None:
    desc = 'Selected'
else:
    print(f"Descriptor: {args.desc} for the regioselectivity plot")
    desc = args.desc

if folder != 'average':
    try:
        file    = f"{base_cwd}/results/model_validation/regression/large_mol/{rxn}/{folder}/eval_bm_{desc}_{model}.csv"
        figname = file.replace('csv', 'png')
        figname = figname.replace('results/', '')
        figname = figname.replace('.png', f"_{draw}.png")
        print(f"\n\nfigname: {figname}\n\n")
        df = pd.read_csv(file, index_col=0)

    except:
        try:
            file    = f"{base_cwd}/results/model_validation/regression/large_mol/{rxn}/{folder}/eval_sm_{desc}_{model}.csv"
            figname = file.replace('csv', 'png')
            figname = figname.replace('results/', '')
            figname = figname.replace('.png', f"_{draw}.png")
            print(f"\n\nfigname: {figname}\n\n")
            df = pd.read_csv(file, index_col=0)
        except:
            print(f"File {file} does not exist")
            exit()

else:
    folders = os.listdir(f"{base_cwd}/results/model_validation/regression/large_mol/{rxn}")
    folders = [f for f in folders if 'run' in f]
    folders = [f for f in folders if 'unspec' not in f]
    figname = f"{rxn}/{folder}/regioselectivity_{desc}_{model}_{draw}.png"
    print(f"\n\nfigname: {figname}\n\n")
    dfs = []
    for f in folders:
        try:
            file    = f"{base_cwd}/results/model_validation/regression/large_mol/{rxn}/{f}/eval_bm_{desc}_{model}.csv"
            df = pd.read_csv(file, index_col=0)
            dfs.append(df)

        except:
            try:
                file    = f"{base_cwd}/results/model_validation/regression/large_mol/{rxn}/{f}/eval_sm_{desc}_{model}.csv"
                df = pd.read_csv(file, index_col=0)
                dfs.append(df)
            except:
                print(f"File {file} does not exist")
    
    if len(dfs) == 0:
        print("No files found")
        exit()

mol_per_line = 3 
max_mols  = len(df.index)
num_lines = int(np.floor(max_mols/mol_per_line)+1)
fig, ax = plt.subplots(num_lines, mol_per_line, figsize=(mol_per_line*mol_per_line, mol_per_line*num_lines))

for i, smiles in enumerate(sorted(list(df.index))):
    smiles = Chem.CanonSmiles(smiles)
    x_i = i // mol_per_line
    y_i = i % mol_per_line
    if folder != 'average':
        y_pred = df.loc[smiles]['Y_pred']
        y_pred = ast.literal_eval(y_pred)
    else: # average over all runs
        y_preds = []
        for df in dfs:
            y_preds.append(ast.literal_eval(df.loc[smiles]['Y_pred']))
        y_pred = {k: 0 for k in y_preds[0].keys()}   
        for y in y_preds:
            for k, v in y.items():
                y_pred[k] += v
        for k, v in y_pred.items():
            y_pred[k] = v/len(y_preds)
        # could add a standard deviation here...

    # normalize ypred:
    y_pred_max = np.max(list(y_pred.values()))
    y_pred_min = min(list(y_pred.values()))
    y_pred = {k: (v - y_pred_min) / (y_pred_max - y_pred_min) for k, v in y_pred.items()}
    y_pred_sum = sum(list(y_pred.values()))
    y_pred = {k: 100*v / y_pred_sum for k, v in y_pred.items()}
    # img_pred_rank = viz.visualize_regio_exp(smiles, df_bde)
    img_pred_rank = viz.visualize_regio_pred(smiles, y_pred, draw = draw)
    img_pred_rank.savefig(f"smiles_{i}.png", bbox_inches='tight', dpi=300)
    del img_pred_rank
    ax[x_i,y_i].imshow(img.imread(f"smiles_{i}.png"))
    ax[x_i,y_i].set_xticks([])
    ax[x_i,y_i].set_yticks([])
    ax[x_i,y_i].set_title(f"Pred. with {model}-{desc}")
    if df.loc[smiles, '0'] == 1:
        ax[x_i,y_i].patch.set_edgecolor('darkcyan')  
        ax[x_i,y_i].patch.set_linewidth(10)
    elif df.loc[smiles, '1'] == 1:
        ax[x_i,y_i].patch.set_edgecolor('mediumaquamarine')  
        ax[x_i,y_i].patch.set_linewidth(10)
    elif df.loc[smiles, '2'] == 1:
        ax[x_i,y_i].patch.set_edgecolor('goldenrod')  
        ax[x_i,y_i].patch.set_linewidth(10)
    else:
        ax[x_i,y_i].patch.set_edgecolor('orangered')  
        ax[x_i,y_i].patch.set_linewidth(10)
try:
    os.remove('tmp.png')
except:
    pass
for j in range(i, len(list(df.index))):
    x_i = j // mol_per_line
    y_i = j % mol_per_line
    ax[x_i,y_i].axis('off')

print(f"Saving {figname}")
fig.savefig(figname, bbox_inches='tight', dpi=600)
os.system(f"rm smiles_*.png")
print(f"Saved {figname}")


