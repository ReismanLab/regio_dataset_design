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

args = parser.parse_args()
if args.folders == None:
    folders  = [f"run_0{i}" for i in range(1, 5)]
else:
    folders = args.folders
    
root = os.getcwd()
try:
    base_cwd = os.getcwd().split('regiochem')[0]
    base_cwd = f"{base_cwd}/regiochem"
except:
    raise ValueError("You are not in the right directory, need to be in the 'notebooks' directory or subdirectory of it.")



## read results files
files = [f"{base_cwd}/results/model_validation/regression/loo/{folder}/results_loo.csv" for folder in folders]
files = [f for f in files if '.csv' in f]

dfs = []
for i, f in enumerate(files):
    df = pd.read_csv(f)
    df.sort_values(by=['Feature', 'Model'], inplace=True)  
    df.reset_index(drop=True, inplace=True)
    dfs.append(df.copy())
    

## average results
#df_mean = p
panel = pd.concat(dfs)
#print(f"\n\n{panel.columns}\n\n")

#df_mean = 
df_mean = panel.groupby(['Feature', 'Model']).mean()
df_std  = panel.groupby(['Feature', 'Model']).std()
df_mean.reset_index(inplace=True)
df_std.reset_index(inplace=True)

df_mean.to_csv("df_mean_runs.csv")
df_std.to_csv("df_std_runs.csv")

## plot heatmaps

try:
    os.mkdir(f"run_avg")
except:
    pass

table                = df_mean.pivot('Feature', 'Model' ,'TOP-1') 
table_std            = df_std.pivot('Feature', 'Model' ,'TOP-1')
table             = table.reindex(['Rdkit-Vbur', 'DBSTEP', 'Gasteiger', 'ENV-1', 'ENV-2', 'XTB', 'Custom', 'Selected'])
table_std         = table_std.reindex(['Rdkit-Vbur', 'DBSTEP', 'Gasteiger', 'ENV-1', 'ENV-2', 'XTB', 'Custom', 'Selected'])
table = table.reindex(['RF2', 'RF-OPT-XTB', 'KNN', 'LR', 'MLP2', 'MLP', 'SVR', 'GPR'], axis =1)
table_std = table_std.reindex(['RF2', 'RF-OPT-XTB', 'KNN', 'LR', 'MLP2', 'MLP', 'SVR', 'GPR'], axis =1)
fig, ax = plt.subplots(2, 1, figsize=(10, 10))
sns.heatmap(data = table, 
            annot=True, 
            cmap="viridis_r",
            vmin=0, vmax=1,
            ax = ax[0])
ax[0].set_title(f"TOP-1 accuracy per models and descriptor on large molecules dataset")
sns.heatmap(data = table_std, 
            annot=True, 
            cmap="viridis",
            vmin=0, vmax=2,
            ax = ax[1])
ax[1].set_title(f"TOP-1 standard deviation per models and descriptor on large molecules dataset")
fig.savefig(f"run_avg/heatmap_top1.png", dpi=300)

table             = df_mean.pivot('Feature', 'Model' ,'TOP-2') 
table_std         = df_std.pivot('Feature', 'Model' ,'TOP-2')
table             = table.reindex(['Rdkit-Vbur', 'DBSTEP', 'Gasteiger', 'ENV-1', 'ENV-2', 'XTB', 'Custom', 'Selected'])
table_std         = table_std.reindex(['Rdkit-Vbur', 'DBSTEP', 'Gasteiger', 'ENV-1', 'ENV-2', 'XTB', 'Custom', 'Selected'])
table = table.reindex(['RF2', 'RF-OPT-XTB', 'KNN', 'LR', 'MLP2', 'MLP', 'SVR', 'GPR'], axis =1)
table_std = table_std.reindex(['RF2', 'RF-OPT-XTB', 'KNN', 'LR', 'MLP2', 'MLP', 'SVR', 'GPR'], axis =1)
fig, ax = plt.subplots(2, 1, figsize=(10, 10))
sns.heatmap(data = table, 
            annot=True, 
            cmap="viridis_r",
            vmin=0, vmax=1,
            ax = ax[0])
ax[0].set_title(f"TOP-2 accuracy per models and descriptor on large molecules dataseth")
sns.heatmap(data = table_std,
            annot=True, 
            cmap="viridis",
            vmin=0, vmax=2,
            ax = ax[1])
ax[1].set_title(f"TOP-2 standard deviation per models and descriptor on large molecules dataset")
fig.savefig(f"run_avg/heatmap_top2.png", dpi=300)

table             = df_mean.pivot('Feature', 'Model' ,'TOP-3')
table_std         = df_std.pivot('Feature', 'Model' ,'TOP-3')
table             = table.reindex(['Rdkit-Vbur', 'DBSTEP', 'Gasteiger', 'ENV-1', 'ENV-2', 'XTB', 'Custom', 'Selected'])
table_std         = table_std.reindex(['Rdkit-Vbur', 'DBSTEP', 'Gasteiger', 'ENV-1', 'ENV-2', 'XTB', 'Custom', 'Selected'])
table = table.reindex(['RF2', 'RF-OPT-XTB', 'KNN', 'LR', 'MLP2', 'MLP', 'SVR', 'GPR'], axis =1)
table_std = table_std.reindex(['RF2', 'RF-OPT-XTB', 'KNN', 'LR', 'MLP2', 'MLP', 'SVR', 'GPR'], axis =1)
fig, ax = plt.subplots(2, 1, figsize=(10, 10))
sns.heatmap(data = table, 
            annot=True, 
            cmap="viridis_r",
            vmin=0, vmax=1,
            ax = ax[0])
ax[0].set_title(f"TOP-3 accuracy per models and descriptor on large molecules dataset")
sns.heatmap(data = table_std,
            annot=True, 
            cmap="viridis",
            vmin=0, vmax=2,
            ax = ax[1])
ax[1].set_title(f"TOP-3 standard deviation per models and descriptor on large molecules dataset")
fig.savefig(f"run_avg/heatmap_top3.png", dpi=300)

table       = df_mean.pivot('Feature', 'Model' ,'TOP-AVG')
table_std   = df_std.pivot('Feature', 'Model' ,'TOP-AVG')
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
ax[0].set_title(f"TOP-AVG accuracy per models and descriptor on large molecules dataset")
sns.heatmap(data = table_std,
            annot=True, 
            cmap="viridis",
            vmin=0, vmax=2,
            ax = ax[1])
ax[1].set_title(f"TOP-AVG standard deviation per models and descriptor on large molecules dataset")

fig.savefig(f"run_avg/heatmap_topavg.png", dpi=300)
