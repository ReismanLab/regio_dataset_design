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

### Parse arguments
import argparse
parser = argparse.ArgumentParser(description='Descriptor computation')
parser.add_argument('--folder',
                    help='Folder with results files',
                    default='clean_run')
parser.add_argument('--overwrite',
                    help='If you need to overwrite the results files and recompute them')
parser.add_argument('--rxn',
                    help='Reaction to be used, default is dioxirane can be borylation.\nAttributes the folders with descriptors and results to plot the figures',
                    default='dioxirane')

args = parser.parse_args()

### argument parsing ###
# folder data extraction
folders = np.nan
if args.folder == None:
    folder = 'clean_run'
    print(f"\n\nFolder not provided, using {folder}\n\n")  
else:
    folder = args.folder
    if folder not in os.listdir(f"{base_cwd}/results/active_learning/regression/") and folder != 'custom_db':
        raise ValueError(f"Folder {folder} not found in {base_cwd}/results/active_learning/regression/")
    else:
        print(f"\n\nFolder used: {folder}\n\n")

# overwrite ?
if args.overwrite == None:
    overwrite = False
    print(f"\n\nOverwrite not provided, will not overwrite {overwrite}\nMake sure you have a results pkl file already created!!! If not, retry with --overwrite True\n\n")
elif args.overwrite not in ['True', 'False']:
    raise ValueError("Overwrite should be True or False")
else:
    if args.overwrite == 'True':
        overwrite = True
    else: 
        overwrite = False

# rxn type and corresponing AFs
rxn = args.rxn
if rxn == 'dioxirane':
    aqcf_list = ['random', 
                 'acqf_1-db0-01-a1', 'acqf_10',
                 'acqf_2-1', 'acqf_3', 'acqf_4-1', 
                 'acqf_5', 'acqf_6', 'acqf_7', 'acqf_9']
    folder_desc = 'preprocessed_dioxirane_reactions'
elif rxn == 'borylation':
    aqcf_list = ['random', 
                 'acqf_1', 'acqf_10',
                 'acqf_2-1', 'acqf_3', 'acqf_4-1', 
                 'acqf_5', 'acqf_6', 'acqf_7', 'acqf_9']
    folder_desc = 'preprocessed_borylation_reactions'
### end parsing ###

#### UTILS FUNCTIONS
## metrics
def get_metrics(df, smiles, carbon_preds, aqcf_list=aqcf_list):
    reac_smiles = []
    for s in df.Reactant_SMILES:
        s = s.replace('/', '-')
        s = s.replace('\\', '-')
        reac_smiles.append(s)
    df['Reactant_SMILES'] = reac_smiles
    df_smi = df[df.Reactant_SMILES == smiles]
    if len(df_smi) == 0:
        print(f"SMILES {smiles} not found in the dataframe")
        return None
    metric_all = []
    for ai, aqcf in enumerate(aqcf_list):
        metric_aqcfs = []
        for repet in range(len(carbon_preds[ai])): # iterate on the number of repetitions of the aqcf aquisitions
            metric_repet = []
            for n_molecules in range(len(carbon_preds[ai][0])-1): # iterate on the number of molecules added 
                sub_dict    = carbon_preds[ai][repet][n_molecules]
                selectivity                     = [sub_dict[x] for x in df_smi['Atom_nº'].values]
                df_smi['Predicted_Selectivity'] = selectivity
                metric_                         = [mt.top_n(df_smi, 1)[0], mt.m2(df_smi)[0], mt.m3(df_smi)[0], mt.m4(df_smi)[0], mt.m5(df_smi)[0], mt.m6(df_smi)[0], mt.m7(df_smi)[0]]   
                metric_repet.append(metric_)
            metric_aqcfs.append(metric_repet)
        metric_all.append(metric_aqcfs)              
    return metric_all

## plot functions
def plot_evolution(ax, data, title=False, labels=['Top1', 'Top2', 'Top3', 'Top5', 'Top10']):
    """
    data: list of list of scores
    """
    print(f"Data shape: {np.array(data).shape}")
    colors      = ['black', 'blue', 'purple', 'orange', 'red', 'pink', 'brown']
    linewidths  = [5, 3.5, 2, 1, 1, 1, 1]
    num_acqf    = len(data[0])
    #print(f"Number of aquisition functions: {num_acqf}")
    data = [data[i][0] for i in range(len(data))]
    #print(f"New data shape: {np.array(data).shape}")
    num_acqf    = len(data[0])
    #print(f"New number of aquisition functions: {num_acqf}")
    mean_scores = np.mean(data, axis=0)
    var_scores  = np.var(data, axis=0)
    for i, j in enumerate(labels):
        ax.plot(mean_scores[:, i], label=j, color=colors[i], linewidth=linewidths[i])
        ax.fill_between(range(num_acqf), 
                        mean_scores[:, i] - var_scores[:, i], 
                        mean_scores[:, i] + var_scores[:, i], 
                        alpha=0.2, color=colors[i])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('Training Size')
    ax.set_ylabel('Top-n Acc.')
    ax.legend()
    
    if title != False:
        ax.set_title(title)
    else:
        ax.set_title('Top-n Accuracies Training')

def draw_n_first_smiles(smiles_list_list, n=10):
    unique_smiles = []
    for smiles_list in smiles_list_list:
        unique_smiles = unique_smiles + smiles_list[:n]
    unique_smiles = list(set(unique_smiles))
    img = Draw.MolsToGridImage([Chem.MolFromSmiles(smi) for smi in unique_smiles])
    return unique_smiles

### util data:
df = pd.read_csv(f"{base_cwd}/data/descriptors/{folder_desc}/df_bde.csv", index_col=0) # chande

## load data for random training:
df_bde = pd.read_csv(f"{base_cwd}/data/descriptors/{folder_desc}/df_bde.csv", index_col=0) # chande
df_xtb = pd.read_csv(f"{base_cwd}/data/descriptors/{folder_desc}/df_xtb.csv", index_col=0) # chande
df_gas = pd.read_csv(f"{base_cwd}/data/descriptors/{folder_desc}/df_gas.csv", index_col=0) # chande

## get a list of SMILES for training:
df_custom = pd.concat([df_bde, df_gas, df_xtb[['Buried_Volume_C', 'Buried_Volume_H_max']]], axis=1)
df_custom = df_custom.loc[:,~df_custom.columns.duplicated()].copy()

def remove_large_molecules(df, max_num_C=15):
    num_C = []
    for smi in df.Reactant_SMILES:
        mol = Chem.MolFromSmiles(smi)
        num_C.append([at.GetSymbol() for at in mol.GetAtoms()].count('C'))
    df['num_C'] = num_C
    df = df[df.num_C <= max_num_C]
    df.drop(columns=['num_C'], inplace=True)
    return df

df_custom_small = remove_large_molecules(df_custom, max_num_C=15)
train_smiles    = df_custom_small.Reactant_SMILES.unique()

## define model:    
reg = RandomForestRegressor(n_estimators=250,
                            max_features=0.5,
                            max_depth=10,
                            min_samples_leaf=3)

def plot_img(target, y_pred):
    img = viz.visualize_regio_pred(target, y_pred)
    return img, None

### plot summary function:
def plot_summary(smiles, t5, df, _, carbo_preds, save=True, labels=['Top1', 'Top2', 'Top3', 'Top5', 'Top10'], aqcf_list=aqcf_list):
    check_ = {}
    
    dict_name = dict(zip(range(len(aqcf_list)), aqcf_list))
    print('Plot Summary Function')
    print(f'dict_name: {dict_name}')

    print(f't5: {t5}') 
    fig, ax = plt.subplots(4, 3, figsize=(12, 16))
    for i, ax_ in enumerate(ax.flatten()):
        # plot aquisition function evolution
        if i < 10:
            plot_evolution(ax_, t5[i], dict_name[i], labels=labels)
        if i % 3 != 0:
            ax_.set_ylabel('')
            ax_.set_yticklabels([])
        if i != 9:
            ax_.set_xlabel('')
            ax_.set_xticklabels([])
        if i in range(1, 9):
            ax_.get_legend().remove()

        # plot molecule and predictions
        if i == 10:
            try:
                img_ = viz.visualize_regio_exp(smiles, df, scale=1000)
                img_.savefig('tmp.png', bbox_inches='tight')
                ax_.imshow(img.imread('tmp.png'))
            except:
                print(f"Error rdkit with SMILES - not plotting structure")
                pass
            ax_.axis("off")
            ax_.set_title('Reported Ox.')

        if i == 11:
            y     = carbo_preds[0][0][0][-1]
            y_    = list(y.values())
            min_y = np.min(y_)
            max_y = np.max(y_)
            for i, y_i in enumerate(y_):
                y_[i] = 100*(y_i - min_y) / (max_y - min_y) # normalize y
            for i, key in enumerate(y.keys()):
                y.update({key: y_[i]})
            img_, y = plot_img(smiles, y)
            try:
                img_.savefig('tmp.png', bbox_inches='tight')
                ax_.imshow(img.imread('tmp.png'))
            except:
                print(f"Error rdkit with SMILES - not plotting structure")
                pass
            ax_.axis("off")
            ax_.set_title('Predicted Ox.')

            check_.update({smiles: y})
    
    fig.suptitle(f"SMILES {_}", fontsize=16)
    plt.suptitle(smiles)
    plt.tight_layout()
    #plt.show()
    
    if save:
        if folder in os.listdir("."):
            pass
        else:
            os.mkdir(folder)
        fig.savefig(f"{folder}/summary_{_}_{smiles}.png", bbox_inches='tight')
        plt.close()

#### MAIN
### make concatenated file for easier plotting
if overwrite:
    files  = os.listdir(f"{base_cwd}/results/active_learning/regression/{folder}")
    files  = [f for f in files if f.endswith('.pkl')]
    smiles = [f.split('_')[2] for f in files if f.split('_')[0] != "results"]
    smiles = list(set(smiles))
    valid_smiles = []
    for smi in smiles:
        try:
            mol = Chem.MolFromSmiles(smi)
            valid_smiles.append(smi)
        except:
            print(f"SMILES {smi} not valid")
            pass

    print(f"TARGET SMILES: {smiles}")
    smiles = sorted(smiles)

    aqcfs  = aqcf_list
    
    batch  = [f.split('_')[-3] for f in files]
    batch  = list(set(batch))
    if len(batch) > 1:
        raise ValueError(f"More than one batch found: {batch}")
    else:
        batch = batch[0]
    print(f"Batch: {batch}")

    start_type = [f.split('_')[-2] for f in files]   
    start_type = list(set(start_type))
    if len(start_type) > 1:
        raise ValueError(f"More than one start type found: {start_type}")
    else:
        start_type = start_type[0]
    print(f"Start type: {start_type}")

    desc = [f.split('_')[-1].split('.')[0] for f in files]
    desc = list(set(desc))
    if len(desc) > 1:
        raise ValueError(f"More than one descriptor found: {desc}")
    else:
        desc = desc[0]
    print(f"Descriptor: {desc}")

    select_files = {}
    for smi in smiles:
        if "/" in smi:
            smi = smi.replace("/", "-")
        new_file = f"{base_cwd}/results/active_learning/regression/{folder}/results_{smi}_{batch}_{start_type}_{desc}.pkl"
        TOP5     = []
        SMILES   = []
        Y_PRED   = []
        for aqcf in aqcf_list:
            top5   = []
            smiles = [] 
            y_pred = []
            for i in range(1, 30):
                file = f"{base_cwd}/results/active_learning/regression/{folder}/res_rf_{smi}_{aqcf}_{i}_{batch}_{start_type}_{desc}.pkl"
                try:    
                    with open(file, 'rb') as f:
                        t5, smis, y, _, parameter = pickle.load(f)
                    top5.append(t5)
                    smiles.append(smis)
                    y_pred.append(y)
                except:
                    print(f"res_rf_{smi}_{aqcf}_{i}_{batch}_{start_type}_{desc}.pkl not found")
                    pass 


            TOP5.append(top5)
            SMILES.append(smiles)
            Y_PRED.append(y_pred)
        
        with open(new_file, 'wb') as f:
            pickle.dump([TOP5, SMILES, Y_PRED], f)

        select_files.update({smi: new_file.split('/')[-1]})

else:
    files = os.listdir(f"{base_cwd}/results/active_learning/regression/{folder}")
    files = [f for f in files if f.endswith('.pkl') and f.startswith('results')]
    print(f"Number of files found: {len(files)}")
    try:
        print(f"Files 0: {files[0]}")
    except:
        raise ValueError("No files foun, you might wat to check if you have a results pkl file already created, if not restart with --overwrite True")
    smiles = [f.split('_')[1] for f in files]
    #smiles = [s for s in smiles if s != '1']
    batch = list(set([f.split('_')[-3] for f in files]))[0]
    start_type = list(set([f.split('_')[-2] for f in files]))[0]
    desc = list(set([f.split('_')[-1].split('.')[0] for f in files]))[0]
    print(f"Batch: {batch}")
    print(f"Start type: {start_type}")
    print(f"Descriptor: {desc}")
    select_files = {}
    for i, smi in enumerate(smiles):
        if smi != '1':
            select_files.update({smi: files[i]})
    select_files = dict(sorted(select_files.items()))
    

### plot summary
_ = 0
reac_smiles = []

for s in df.Reactant_SMILES:
    s = s.replace('/', '-')
    s = s.replace('\\', '-')
    reac_smiles.append(s)

df['Reactant_SMILES'] = reac_smiles

### need to correct here df when the values of the target are not Selectivity ###

### need to correct here df when the values of the target have been renormalized ###

for smiles_1 in list(select_files.keys()):
    print(smiles_1)
    with open(f"{base_cwd}/results/active_learning/regression/{folder}/{select_files[smiles_1]}", 'rb') as f:
        results_1 = pickle.load(f)
    t5, smis, carbon_preds_ = results_1
    plot_summary(smiles_1, t5, df, _, carbon_preds_)
    _ += 1
