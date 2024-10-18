## plot regioselectivity prediction
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as img
import os
import ast
import sys
import pandas as pd
import numpy as np
from rdkit import Chem
import warnings
pd.set_option('mode.chained_assignment', None)
warnings.simplefilter(action='ignore', category=FutureWarning)


### Parse arguments
import argparse
parser = argparse.ArgumentParser(description='Descriptor computation')
parser.add_argument('--run',
                    help='Folder with results files')
parser.add_argument('--model',
                    help='Model to evaluate the regioselectivity plot')
parser.add_argument('--desc',
                    help='Descriptor to use for the regioselectivity plot')
parser.add_argument('--smiles',
                    help='SMILES to evaluate the regioselectivity plot')
parser.add_argument('--rxn',
                    help='Folder with preprocessed reactions', default = "preprocessed_reactions")

args = parser.parse_args()
if args.smiles == None:
    print("Need to provide a SMILES (--smiles your_smiles) to evaluate the regioselectivity plot")
    exit()
else:
    smiles = Chem.CanonSmiles(args.smiles)

if args.run == None:
    folder = 'run_01'
    print(f"Default folder: {folder} for the regioselectivity plot")
else:
    folder = args.run

if args.model == None:
    model = 'RF2'
    print(f"Default model: {model} for the regioselectivity plot")
else:
    model = args.model

if args.desc == None:
    desc = 'Selected'
    print(f"Default descriptor: {desc} for the regioselectivity plot")
else:
    desc = args.desc

# for data loading:
rxn_folder = args.rxn
if rxn_folder == "preprocessed_reactions":
    sub_folder = "dioxirane"
    print(f"Default sub folder: {sub_folder} for the regioselectivity plot")
elif rxn_folder == "preprocessed_reactions_borylation":
    sub_folder = "borylation_filt"
    print(f"Default sub folder: {sub_folder} for the regioselectivity plot")


root = os.getcwd()
try:
    base_cwd = os.getcwd().split('regiochem')[0]
    base_cwd = f"{base_cwd}/regiochem"
except:
    raise ValueError("You are not in the right directory, need to be in the 'notebooks' directory or subdirectory of it.")

sys.path.append(f"{base_cwd}/utils/")
import metrics as mt    
import visualization as viz

### Load data
try:
    file    = f"{base_cwd}/results/model_validation/regression/large_mol/{sub_folder}/{folder}/eval_bm_{desc}_{model}.csv"
    figname = file.replace('csv', 'png')
    figname = figname.replace('results/', '')
    figname = figname.replace(f"{sub_folder}/", '')
    figname = figname.replace('eval_bm', 'fig_smi')
    print(f"\n\nfigname: {figname}\n\n")
    df = pd.read_csv(file, index_col=0)

except:
    try:
        file    = f"{base_cwd}/results/model_validation/regression/large_mol/{sub_folder}/{folder}/eval_sm_{desc}_{model}.csv"
        figname = file.replace('csv', 'png')
        figname = figname.replace('results/', '')
        figname = figname.replace(f"{sub_folder}/", '')
        print(f"\n\nfigname: {figname}\n\n")
        df = pd.read_csv(file, index_col=0)
    except:
        print(f"File {file} does not exist")
        exit()

### Make plot!
fig, ax = plt.subplots(2, 1, figsize=(8, 4))

smiles = Chem.CanonSmiles(smiles)

if smiles not in df.index:
    print(f"SMILES {smiles} not in the dataframe, are you sure it is in the dataset?")
    exit()

y_pred = df.loc[smiles]['Y_pred']
y_pred = ast.literal_eval(y_pred)
# normalize ypred:
y_pred_max = np.max(list(y_pred.values()))
y_pred_min = min(list(y_pred.values()))
y_pred     = {k: (v - y_pred_min) / (y_pred_max - y_pred_min) for k, v in y_pred.items()}
y_pred_sum = sum(list(y_pred.values()))
y_pred     = {k: 100*v / y_pred_sum for k, v in y_pred.items()}
df_bde = pd.read_csv(f"{base_cwd}/data/descriptors/{rxn_folder}/df_bde.csv", index_col=0)
img_exp_rank  = viz.visualize_regio_exp(smiles, df_bde)
img_pred_rank = viz.visualize_regio_pred(smiles, y_pred)
img_pred_rank.savefig('tmp_pred.png', bbox_inches='tight', dpi=300)
img_exp_rank.savefig('tmp_exp.png', bbox_inches='tight', dpi=300)
del img_pred_rank, img_exp_rank
ax[1].imshow(img.imread('tmp_pred.png'))
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_title(f"{model}-{desc}")

ax[0].imshow(img.imread('tmp_exp.png'))
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_title(f"Experimental")

if df.loc[smiles, '0'] == 1:
    ax[1].patch.set_edgecolor('darkcyan')  
    ax[1].patch.set_linewidth(10)
elif df.loc[smiles, '1'] == 1:
    ax[1].patch.set_edgecolor('mediumaquamarine')  
    ax[1].patch.set_linewidth(10)
elif df.loc[smiles, '2'] == 1:
    ax[1].patch.set_edgecolor('goldenrod')  
    ax[1].patch.set_linewidth(10)
else:
    ax[1].patch.set_edgecolor('orangered')  
    ax[1].patch.set_linewidth(10)

os.remove('tmp_pred.png')
os.remove('tmp_exp.png')

ax[0].axis('off')
ax[1].axis('off')

fig.savefig(figname, bbox_inches='tight', dpi=300)
