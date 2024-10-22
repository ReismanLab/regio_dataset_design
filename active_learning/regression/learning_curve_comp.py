## imports 
import os
import sys
import pickle
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import image as img
from rdkit import Chem
from rdkit.Chem import Draw
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
try:
    base_cwd = os.getcwd().split('regio_dataset_design')[0]
    base_cwd = f"{base_cwd}/regio_dataset_design"
except:
    raise ValueError("You are not in the right directory, need to be in the 'notebooks' directory or subdirectory of it.")
sys.path.append(f"{base_cwd}/utils/")
import metrics as mt
import visualization as viz
import modelling as md
import preprocessing as pp
import ast


## parser
import argparse
parser = argparse.ArgumentParser(description='Descriptor computation')
parser.add_argument('--folder',
                    help='Folder with results files',
                    default = 'clean_run')    
parser.add_argument('--acqf_list', type=str, nargs='+',
                    help='List of acquisition functions to compare',
                    default=["acqf_1-db0-01-a1", "acqf_2-1", "acqf_6"])
parser.add_argument('--smi',
                    help='Smiles to plot for the learning curves',
                    default='CC(=O)O[C@H]1CC[C@@]2(C)[C@@H](CC(=O)[C@@H]3[C@@H]2CC[C@]2(C)[C@@H]([C@H](C)CCCC(C)C)CC[C@@H]32)C1')
parser.add_argument('--suffix',
                    help='Name of the figure to save',
                    default='0')
parser.add_argument('--stability_length',
                    help='Length of the stability window',
                    default=10)

# large aqcuisition function list
l_af_list = ['random', 
                             'acqf_1-db0-01-a1', 'acqf_10',
                             'acqf_2-1', 'acqf_3', 'acqf_4-1', 
                             'acqf_5', 'acqf_6', 'acqf_7', 'acqf_9']
dict_idx_to_acqf = dict(zip(l_af_list, range(len(l_af_list))))
print(f"Dictionary: {dict_idx_to_acqf}")
print(f"\n\n ****WARNING****\nmake sure the l_af_list: {l_af_list}\nis in the one that has been used in the learning curve computation with overwrite == True before!!.")

args             = parser.parse_args()
folder           = args.folder
acqf_list        = args.acqf_list
try:
    idx_acqf         = [dict_idx_to_acqf[x] for x in acqf_list]
except:
    print(f"acqf_list: {ast.literal_eval(acqf_list[0])}")
    acqf_list = ast.literal_eval(acqf_list[0])
    idx_acqf         = [dict_idx_to_acqf[x] for x in acqf_list]
smi              = args.smi
stability_length = int(args.stability_length)
try:
    smi = Chem.CanonSmiles(smi)
except:
    raise ValueError("Smiles is not valid.")
# check if smiles exists...


### plot summary
file_name = f"{base_cwd}/results/active_learning/regression/{folder}/results_{smi}_1_coldstart_custom.pkl"

with open(file_name, 'rb') as f:
    results_1 = pickle.load(f)
t5, smis, carbon_preds_ = results_1

rand = t5[0]
af_1 = t5[idx_acqf[0]]
af_2 = t5[idx_acqf[1]]
af_3 = t5[idx_acqf[2]]

labels = ["Random", "Active Learning", "Mol. Similarity", "Site Similarity"]
colors = ['gray', 'skyblue', 'green', 'plum']  
for i, af in enumerate([rand, af_1, af_2, af_3]):
    data = [af[i][0] for i in range(len(af))]
    print(f"Data shape: {np.array(data).shape}")
    num_acqf    = len(data[0])
    print(f"Number of aquisition functions: {num_acqf}")
    mean_scores = np.mean(data, axis=0)
    # find stability of the model
    count     = 0
    stability = []

    for j, x in enumerate(mean_scores[:, 0]):
        if x > 0.9: 
            count += 1   
        else:
            count = 0
        if count == stability_length:
            stability = [j - stability_length + 1, j + 1]
            print(f"Stability: {stability}")
            break

    var_scores  = np.var(data, axis=0)
    print(f"Mean scores shape: {mean_scores.shape}")
    print(f"Var scores shape: {var_scores.shape}")
    plt.plot([0, num_acqf], [0.9, 0.9], color="black", linestyle='--', alpha=0.8)
    try:
        plt.axvspan(stability[0], stability[1], color=colors[i], alpha=0.4)
    except:
        pass
    plt.plot(mean_scores[:, 0], color=colors[i], linewidth=2)
    plt.fill_between(range(num_acqf), 
                     mean_scores[:, 0] - var_scores[:, 0], 
                     mean_scores[:, 0] + var_scores[:, 0], 
                     alpha=0.5, color=colors[i]) 
    try:
        plt.plot(mean_scores[stability[0]:stability[1], 0], [x for x in range(stability[0], stability[1])], color=colors[i], linewidth=10)
    except:
        pass
    plt.legend(labels)

handles = [plt.Rectangle((0,0),1,1, color='gray'), 
           plt.Rectangle((0,0),1,1, color='skyblue'), 
           plt.Rectangle((0,0),1,1, color='green'), 
           plt.Rectangle((0,0),1,1, color='plum')]

plt.legend(handles, labels)
plt.ylim([-0.05, 1.05])
plt.xlabel('# Experiments Realized')
plt.ylabel('Average Top-1 Accuracy')

plt.savefig(f"{folder}/lc_comp_tmp_{args.suffix}.png", dpi=300)

smi_af_1 = smis[idx_acqf[0]]
smi_af_2 = smis[idx_acqf[1]]    
smi_af_6 = smis[idx_acqf[2]]

def find_smiles(smi_af, name):
    smi_af_avg = []
    for x in range(10):
        smi_af_avg += smi_af[x][1][:10]
    smi_af_avg = np.unique(smi_af_avg)
    print(f"{name}: {smi_af_avg}")

find_smiles(smi_af_1, "SMI AF 1")
find_smiles(smi_af_2, "SMI AF 2")
find_smiles(smi_af_6, "SMI AF 6")

print(f"SMI AF 1 0 : {smi_af_1[0][1][:10]}")
