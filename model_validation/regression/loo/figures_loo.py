import seaborn as sns
import matplotlib.pyplot as plt
import os
import ast
import sys
import pandas as pd
import warnings
import numpy as np
pd.set_option('mode.chained_assignment', None)
warnings.simplefilter(action='ignore', category=FutureWarning)


## Parse arguments
import argparse
parser = argparse.ArgumentParser(description='Descriptor computation')
parser.add_argument('--run',
                    help='Folder with results files')
parser.add_argument('--rxn',
                    help='Folder with preprocessed reactions', 
                    default = "dioxirane")

args = parser.parse_args()

if args.run == None:
    folder = 'clean_run_01'
else:
    folder = args.run

if args.rxn == None:
    rxn_folder = "preprocessed_reactions_no_unspec_no_intra"
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
    rxn_folder = "preprocessed_reactions_no_unspec_no_intra"
    rxn = "dioxirane"
    print(f"Reaction default: {rxn} and not the one provided: {args.rxn}")

root = os.getcwd()
try:
    base_cwd = os.getcwd().split('regio_dataset_design')[0]
    base_cwd = f"{base_cwd}/regio_dataset_design"
except:
    raise ValueError("You are not in the right directory, need to be in the 'notebooks' directory or subdirectory of it.")

sys.path.append(f"{base_cwd}/utils/")
import modelling as md
import metrics as mt

## read results files

features = {'BDE'       : 0,
            'XTB'       : 0,
            'DBSTEP'    : 0, 
            'Gasteiger' : 0,
            'ENV-1'     : 0,
            'ENV-1-OHE' : 0,
            'ENV-2'     : 0,
            'Rdkit-Vbur': 0,
            'Selected'  : 0,
            'Custom'    : 0}

models =  {'LR'         : 0,
           'RF-OPT-XTB' : 0,
           'RF2'        : 0,
           #'MLP'        : 0,
           #'MLP2'       : 0,
           'SVR'        : 0,
           'KNN'        : 0,
           'GPR'        : 0,}

results = pd.DataFrame(columns=['Model', 'Feature', 'TOP-1', 'TOP-2', 'TOP-3', 'TOP-5', 'TOP-AVG'])

if folder != 'average':
    for m, model in models.items():
        for f, feature in features.items():
            try:
                df_p = pd.read_csv(f"{base_cwd}/results/model_validation/regression/loo/{rxn}/{folder}/pred_loo_{m}_{f}.csv", index_col=0)
                print(f"{m} - {f} have been run in {folder}")
                top_n = []
                for i in [1,2,3,5]:
                    top_n.append(md.get_top_n_accuracy(df_p, i))
                results = results.append({'Model'   : m,
                                        'Feature' : f,
                                        'TOP-1'   : 100*top_n[0],
                                        'TOP-2'   : 100*top_n[1],
                                        'TOP-3'   : 100*top_n[2],
                                        'TOP-5'   : 100*top_n[3],
                                        'TOP-AVG' : mt.top_avg(df_p)
                                        }, ignore_index=True)
            except:
                print(f"{m} - {f} have not been run in model_validation/regression/loo/{rxn}/{folder}/pred_loo_{m}_{f}.csv -- skipping")

    results.to_csv(f"{base_cwd}/results/model_validation/regression/loo/{rxn}/{folder}/results_loo.csv", index=False)

else:
    folders = os.listdir(f"{base_cwd}/results/model_validation/regression/loo/{rxn}/")
    folders = [f for f in folders if 'run' in f]
    results = []
    for fold in folders:
        res_ = pd.DataFrame(columns=['Model', 'Feature', 'TOP-1', 'TOP-2', 'TOP-3', 'TOP-5', 'TOP-AVG'])
        for m, model in models.items():
            for f, feature in features.items():
                try:
                    df_p = pd.read_csv(f"{base_cwd}/results/model_validation/regression/loo/{rxn}/{fold}/pred_loo_{m}_{f}.csv", index_col=0)
                    top_n = []
                    for i in [1,2,3,5]:
                        top_n.append(md.get_top_n_accuracy(df_p, i))
                    res_ = res_.append({'Model'   : m,
                                            'Feature' : f,
                                            'TOP-1'   : 100*top_n[0],
                                            'TOP-2'   : 100*top_n[1],
                                            'TOP-3'   : 100*top_n[2],
                                            'TOP-5'   : 100*top_n[3],
                                            'TOP-AVG' : mt.top_avg(df_p)
                                            }, ignore_index=True)
                except:
                    print(f"{m} - {f} have not been run in {fold} -- skipping")

        res_.to_csv(f"{base_cwd}/results/model_validation/regression/loo/{rxn}/{fold}/results_loo.csv", index=False)
        results.append(res_)

    res = results[0].set_index(['Model', 'Feature'])
    for r in results[1:]:
        res += r.set_index(['Model', 'Feature'])

    res = res/len(results)

    df_results = res.copy()
    
    def multiply(df1, df2):
        df = pd.DataFrame(df1.values*df2.values, columns=df1.columns, index=df1.index)
        return df
    
    # compute standard deviation
    df_count_all  = results[0].set_index(['Model', 'Feature'])
    for idx in df_count_all.index:
        for col in df_count_all.columns:
            if df_count_all.loc[idx, col] == df_count_all.loc[idx, col]:
                df_count_all.loc[idx, col] = 1
            else:
                df_count_all.loc[idx, col] = 0

    df_res_sub = results[0].set_index(['Model', 'Feature']) - df_results
    df_res_std = multiply(df_res_sub, df_res_sub) 
    for i in range(1, len(results)):
        df_res_sub = results[i].set_index(['Model', 'Feature']) - df_results
        df_square  = multiply(df_res_sub, df_res_sub) 
        df_res_std += df_square 
        df_count   = results[i].set_index(['Model', 'Feature'])
        for idx in df_count.index:
            for col in df_count.columns:
                if df_count.loc[idx, col] == df_count.loc[idx, col]:
                    df_count.loc[idx, col] = 1
                else:
                    df_count.loc[idx, col] = 0
        df_count_all += df_count

    df_res_std = df_res_std / df_count_all
    df_res_std = np.sqrt(df_res_std)

    # reset index
    df_res_std.reset_index(inplace=True) 
    df_results.reset_index(inplace=True) 

    # save results
    df_res_std.to_csv(f"df_std_runs.csv")
    df_results.to_csv(f"df_mean_runs.csv")
    results = df_results.copy()

    try:
        os.mkdir(f"{base_cwd}/results/model_validation/regression/loo/{rxn}/average/")
    except:
        pass
    results.to_csv(f"{base_cwd}/results/model_validation/regression/loo/{rxn}/average/results_loo.csv", index=False)
    

## plot results
import seaborn as sns
import matplotlib.pyplot as plt

try:
    os.chdir(f"{rxn}")  
    os.mkdir(f"{folder}")
    os.chdir(root)
except:
    print(f"{rxn}/{folder}  could not be created, maybe already there: double check!")
    pass

round_ = 2

print("Plotting results")
#print(f"\n\n{results}\n\n") 
#print(f"\n\n{df_res_std}\n\n")

for col in ['TOP-1', 'TOP-2', 'TOP-3', 'TOP-5', 'TOP-AVG']:
    table = results.pivot("Feature", "Model", col) 
    table = table.reindex(['BDE', 'Rdkit-Vbur', 'DBSTEP', 'Gasteiger', 'ENV-1', 'ENV-2', 'XTB', 'Custom', 'Selected'])
    table = table.reindex(['RF2', 'RF-OPT-XTB', 'KNN', 'LR',  'SVR', 'GPR'], axis =1) # 'MLP2', 'MLP',
    if folder == 'average':
        table_std = df_res_std.pivot("Feature", "Model" , col)
        table_std = table_std.reindex(['BDE', 'Rdkit-Vbur', 'DBSTEP', 'Gasteiger', 'ENV-1', 'ENV-2', 'XTB', 'Custom', 'Selected'])
        table_std = table_std.reindex(['RF2', 'RF-OPT-XTB', 'KNN', 'LR',  'SVR', 'GPR'], axis =1) # 'MLP2', 'MLP',

    annots = pd.DataFrame(columns=table.columns, index=table.index)
    for i in range(len(annots.index)):
        for j in range(len(annots.columns)):
            if folder == 'average':
                annots.iloc[i,j] = f"{round(table.iloc[i,j], round_)} ± {round(table_std.iloc[i,j], round_)}" 
            else:
                annots.iloc[i,j] = f"{round(table.iloc[i,j], round_)}" 

    fig, ax = plt.subplots(figsize=(20, 10))
    if col != 'TOP-AVG':
       vmin = 0
       vmax = 100
       cmap = "viridis_r"
    else:
       vmin = 5
       vmax = 1
       cmap = "viridis"
    sns.heatmap(data = table, 
                annot=annots, 
                fmt='',
                cmap=cmap, vmin=vmin, vmax=vmax, ax=ax)
    ax.set_title(col)   

    if rxn in os.getcwd():
        os.chdir("..")

    fig.savefig(f"{rxn}/{folder}/heatmap_{col}.png", dpi=300)

