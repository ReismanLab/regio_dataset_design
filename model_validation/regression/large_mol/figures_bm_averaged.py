import seaborn as sns
import matplotlib.pyplot as plt
import os
import ast
import sys
import pandas as pd
import warnings
pd.set_option('mode.chained_assignment', None)
warnings.simplefilter(action='ignore', category=FutureWarning)

### Parse arguments
import argparse
parser = argparse.ArgumentParser(description='Descriptor computation')
parser.add_argument('--folders',
                    help='Folder with results files')
parser.add_argument('--rxn',
                    help='Folder with preprocessed reactions', default = "preprocessed_reactions")


args = parser.parse_args()
if args.folders == None:
    folders  = [f"run_0{i}" for i in range(1, 9)]
    folders += [f"run_1{i}" for i in range(0, 1)]
else:
    folders = args.folders

if args.rxn == None:
    rxn_folder = "preprocessed_reactions"
    rxn = "dioxirane"
    print(f"Reaction default: {rxn}")
elif args.rxn == 'borylation':
    rxn = 'borylation'
    rxn_folder = "preprocessed_borylation_reactions"
    print(f"Reaction used: {rxn}")
elif args.rxn == 'borylation_filt':
    rxn = 'borylation_filt'
    rxn_folder = "preprocessed_borylation_reactions"
    print(f"Reaction used: {rxn}")
else:
    rxn_folder = "preprocessed_reactions"
    rxn = "dioxirane"
    print(f"Reaction default: {rxn} and not the one provided: {args.rxn}")

root = os.getcwd()
try:
    base_cwd = os.getcwd().split('regiochem')[0]
    base_cwd = f"{base_cwd}/regiochem"
except:
    raise ValueError("You are not in the right directory, need to be in the 'notebooks' directory or subdirectory of it.")


# by default:
epochs = [2, 10, 50, 100]


## read results files
files = [f"{base_cwd}/results/model_validation/regression/large_mol/{rxn}/{folder}/df_results.csv" for folder in folders]
files = [f for f in files if '.csv' in f]

dfs = []
for i, f in enumerate(files):
    df = pd.read_csv(f, index_col=0)
    df.sort_values(by=['Descriptor', 'Model'], inplace=True)  
    df.reset_index(drop=True, inplace=True)
    dfs.append(df.copy())
    

## average results
#df_mean = p
panel = pd.concat(dfs)
print(f"\n\n{panel.columns}\n\n")

#df_mean = 
df_mean = panel.groupby(['Descriptor', 'Model']).mean()
df_std  = panel.groupby(['Descriptor', 'Model']).std()
df_mean.reset_index(inplace=True)
df_std.reset_index(inplace=True)
df_mean.to_csv("df_mean_runs.csv")
df_std.to_csv("df_std_runs.csv")

## plot heatmaps

try:
    os.mkdir(f"{rxn}/run_avg")
except:
    pass

table                = df_mean.pivot('Descriptor', 'Model' ,'TOP1') 
table_std            = df_std.pivot('Descriptor', 'Model' ,'TOP1')
table             = table.reindex(['Rdkit-Vbur', 'DBSTEP', 'Gasteiger', 'ENV-1', 'ENV-2', 'XTB', 'Custom', 'Selected'])
table_std         = table_std.reindex(['Rdkit-Vbur', 'DBSTEP', 'Gasteiger', 'ENV-1', 'ENV-2', 'XTB', 'Custom', 'Selected'])
table = table.reindex(['RF2', 'RF-OPT-XTB', 'KNN', 'LR', 'MLP2', 'MLP', 'SVR', 'GPR'], axis =1)
table_std = table_std.reindex(['RF2', 'RF-OPT-XTB', 'KNN', 'LR', 'MLP2', 'MLP', 'SVR', 'GPR'], axis =1)
fig, ax = plt.subplots(2, 1, figsize=(10, 10))
sns.heatmap(data = table, 
            annot=True, 
            cmap="viridis_r",
            vmin=0, vmax=55,
            ax = ax[0])
ax[0].set_title(f"TOP1 accuracy per models and descriptor on large molecules dataset")
sns.heatmap(data = table_std, 
            annot=True, 
            cmap="viridis",
            vmin=0, vmax=2,
            ax = ax[1])
ax[1].set_title(f"TOP1 standard deviation per models and descriptor on large molecules dataset")
fig.savefig(f"{rxn}/run_avg/heatmap_top1.png", dpi=300)

table             = df_mean.pivot('Descriptor', 'Model' ,'TOP2') 
table_std         = df_std.pivot('Descriptor', 'Model' ,'TOP2')
table             = table.reindex(['Rdkit-Vbur', 'DBSTEP', 'Gasteiger', 'ENV-1', 'ENV-2', 'XTB', 'Custom', 'Selected'])
table_std         = table_std.reindex(['Rdkit-Vbur', 'DBSTEP', 'Gasteiger', 'ENV-1', 'ENV-2', 'XTB', 'Custom', 'Selected'])
table = table.reindex(['RF2', 'RF-OPT-XTB', 'KNN', 'LR', 'MLP2', 'MLP', 'SVR', 'GPR'], axis =1)
table_std = table_std.reindex(['RF2', 'RF-OPT-XTB', 'KNN', 'LR', 'MLP2', 'MLP', 'SVR', 'GPR'], axis =1)
fig, ax = plt.subplots(2, 1, figsize=(10, 10))
sns.heatmap(data = table, 
            annot=True, 
            cmap="viridis_r",
            vmin=0, vmax=55,
            ax = ax[0])
ax[0].set_title(f"TOP2 accuracy per models and descriptor on large molecules dataseth")
sns.heatmap(data = table_std,
            annot=True, 
            cmap="viridis",
            vmin=0, vmax=2,
            ax = ax[1])
ax[1].set_title(f"TOP2 standard deviation per models and descriptor on large molecules dataset")
fig.savefig(f"{rxn}/run_avg/heatmap_top2.png", dpi=300)

table             = df_mean.pivot('Descriptor', 'Model' ,'TOP3')
table_std         = df_std.pivot('Descriptor', 'Model' ,'TOP3')
table             = table.reindex(['Rdkit-Vbur', 'DBSTEP', 'Gasteiger', 'ENV-1', 'ENV-2', 'XTB', 'Custom', 'Selected'])
table_std         = table_std.reindex(['Rdkit-Vbur', 'DBSTEP', 'Gasteiger', 'ENV-1', 'ENV-2', 'XTB', 'Custom', 'Selected'])
table = table.reindex(['RF2', 'RF-OPT-XTB', 'KNN', 'LR', 'MLP2', 'MLP', 'SVR', 'GPR'], axis =1)
table_std = table_std.reindex(['RF2', 'RF-OPT-XTB', 'KNN', 'LR', 'MLP2', 'MLP', 'SVR', 'GPR'], axis =1)
fig, ax = plt.subplots(2, 1, figsize=(10, 10))
sns.heatmap(data = table, 
            annot=True, 
            cmap="viridis_r",
            vmin=0, vmax=55,
            ax = ax[0])
ax[0].set_title(f"TOP3 accuracy per models and descriptor on large molecules dataset")
sns.heatmap(data = table_std,
            annot=True, 
            cmap="viridis",
            vmin=0, vmax=2,
            ax = ax[1])
ax[1].set_title(f"TOP3 standard deviation per models and descriptor on large molecules dataset")
fig.savefig(f"{rxn}/run_avg/heatmap_top3.png", dpi=300)

table       = df_mean.pivot('Descriptor', 'Model' ,'TOPAVG')
table_std   = df_std.pivot('Descriptor', 'Model' ,'TOPAVG')
table             = table.reindex(['Rdkit-Vbur', 'DBSTEP', 'Gasteiger', 'ENV-1', 'ENV-2', 'XTB', 'Custom', 'Selected'])
table_std         = table_std.reindex(['Rdkit-Vbur', 'DBSTEP', 'Gasteiger', 'ENV-1', 'ENV-2', 'XTB', 'Custom', 'Selected'])
table = table.reindex(['RF2', 'RF-OPT-XTB', 'KNN', 'LR', 'MLP2', 'MLP', 'SVR', 'GPR'], axis =1)
table_std = table_std.reindex(['RF2', 'RF-OPT-XTB', 'KNN', 'LR', 'MLP2', 'MLP', 'SVR', 'GPR'], axis =1)
fig, ax = plt.subplots(2, 1, figsize=(10, 10))
sns.heatmap(data = table, 
            annot=True, 
            cmap="viridis",
            vmin=1, vmax=5,
            ax = ax[0])
ax[0].set_title(f"TOPAVG accuracy per models and descriptor on large molecules dataset")
sns.heatmap(data = table_std,
            annot=True, 
            cmap="viridis",
            vmin=0, vmax=2,
            ax = ax[1])
ax[1].set_title(f"TOPAVG standard deviation per models and descriptor on large molecules dataset")

fig.savefig(f"{rxn}/run_avg/heatmap_topavg.png", dpi=300)
