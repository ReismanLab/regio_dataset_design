# add path to utils
import sys
import os
import subprocess
root = os.getcwd()

try:
    base_cwd = root.split('regio_dataset_design')[0]
    base_cwd = f"{base_cwd}regio_dataset_design"
except:
    raise ValueError("You are not in the right directory, need to be in the 'notebooks' directory or subdirectory of it.")

sys.path.append(f"{base_cwd}/utils/")

import pandas as pd
from rdkit import Chem
import json
import numpy as np
import pickle
import os
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

acqf_lst = ["acqf_1-db100-a1", "acqf_1-db10-a1", "acqf_1-db1-a1", "acqf_1-db0-1-a1", "acqf_1-db0-01-a1", "acqf_1-db0-001-a1",
            "acqf_1-db1-a2", "acqf_1-db1-a1-5", "acqf_1-db1-a0-5", "acqf_1-db1-a0", "acqf_10",
            "acqf_2", "acqf_2-1", "acqf_3", "acqf_4", "acqf_4-1",
            "acqf_5", "acqf_6", "acqf_7", "acqf_8", "acqf_9", 
            "random" ]
acqf_lst_bor = ["acqf_1", "acqf_10",
            "acqf_2", "acqf_2-1", "acqf_3", "acqf_4", "acqf_4-1",
            "acqf_5", "acqf_6", "acqf_7", "acqf_8", "acqf_9", 
            "random" ]
acqf2col = {"acqf_1-db1-a1": "lightskyblue", 
            "acqf_1-db100-a1": "lightskyblue",
            "acqf_1-db10-a1": "lightskyblue",
            "acqf_1-db0-1-a1": "lightskyblue",
            "acqf_1-db0-01-a1": "lightskyblue",
            "acqf_1-db0-001-a1": "lightskyblue",
            "acqf_1-db1-a2": "lightskyblue",
            "acqf_1-db1-a0": "lightskyblue",
            "acqf_1-db1-a0-5": "lightskyblue",
            "acqf_1-db1-a1-5": "lightskyblue",
            "acqf_1": "lightskyblue",
            "acqf_10": "lightskyblue",
            "acqf_2": "palegreen", 
            "acqf_2-1": "palegreen", 
            "acqf_3": "palegreen",
            "acqf_4": "palegreen",
            "acqf_4-1": "palegreen",
            "acqf_5": "plum", 
            "acqf_6": "plum",
            "acqf_7": "plum", 
            "acqf_8": "plum", 
            "acqf_9": "plum",
            "random": "gray"}

### Parse arguments
import argparse
parser = argparse.ArgumentParser(description='Acquisition function result computation')
parser.add_argument('--path',
                    help='Input directory of pkl files in ../../results/active_learning, if "custom_db" will look for custom descriptors and plot all db values for AF-1',
                    default="regression/clean_run")
parser.add_argument('--cluster',
                    help='Results filtered for a single cluster "macrocycle", "steroids", "amino-acid", "taxol", "misc" or "all" molecules',
                    default='all')
parser.add_argument('--figtype',
                    help='One of "bar", "box", "violin',
                    default="box")
parser.add_argument('--metric',
                    help='One of AUC and SSP',
                    default='SSP')
parser.add_argument('--thresh',
                    help='Performance-based thresholding for both AUC and SSP',
                    default=0.9)
parser.add_argument('--duration',
                    help='Duration for SSP calculation',
                    default=10)
parser.add_argument('--prefix',
                    help='Output png file prefix',
                    default='standard')
parser.add_argument('--out',
                    help='Results folder',
                    default="test")
parser.add_argument('--acqf_lst',
                    help='List of acquisition functions to plot',
                    default="all")
parser.add_argument('--paper_style',
                    help='List of acquisition functions to plot',
                    default=False)

args           = parser.parse_args()
path_          = "../../results/active_learning/" + args.path
cluster        = args.cluster
figtype        = args.figtype
metric         = args.metric
perf_threshold = float(args.thresh)
duration       = int(args.duration)
prefix         = args.prefix
out            = args.out

if args.acqf_lst == "all":
    acqf_lst       = acqf_lst
    new_acqfs      = []
    for af in acqf_lst:
        num_files = subprocess.check_output(f"ls -la {path_}/*{af}* | wc -l", shell=True)
        num_files = num_files.decode('utf-8')
        num_files = num_files.split(' ')[-1].replace('\n','')
        num_files = int(num_files)
        if num_files > 0:
            new_acqfs.append(af)
        else:
            print(f"No files found for AF: {af} -- skipping")
    acqf_lst = new_acqfs
    aqcfs_included = "all"
else:
    aqcfs_included = list(args.acqf_lst.split(' '))
    aqcfs_included = [x.replace('[', '').replace(']', '').replace(',','') for x in aqcfs_included]
paper_style    = args.paper_style
print(f"AFs included: {aqcfs_included}")

if not os.path.exists(out):
    print(f"Creating {out}")
    os.mkdir(out)
else:
    print(f"Output folder {out} already exists")

# relate the dataset to the path
if args.path.split('/')[1] in ["custom_db", "custom_db=1", "custom_db=0-1", "custom_db=0-01", "custom_db=0-01_b=5", "custom_db=0-01_b=10"]:
    dataset = "preprocessed_reactions"
elif args.path.split('/')[1] in ["no_unspec_dia_db_1"]:
    dataset = "preprocessed_reactions_no_unspec_center"
elif args.path.split('/')[1] in ["clean_run", "clean_run_db_0-1", "clean_run_db_0-01", "clean_run_xtb_db0-01_alpha1", "clean_run_selected_db0-01_alpha1"]:
    dataset = "preprocessed_reactions_no_unspec_no_intra"
elif "borylation" in args.path.split('/')[1]:
    dataset = "preprocessed_borylation_reactions"
    acqf_lst       = acqf_lst_bor
    #aqcfs_included = list(args.acqf_lst.split(' '))
    #aqcfs_included = [x.replace('[', '').replace(']', '').replace(',','') for x in aqcfs_included]
else:
    print(f"Cannot determine dataset for {args.path}")
    exit()

# special case for custom descriptors with old dataset - enables to have all db values for acqf_1
if args.path == "custom_db": 
    acqf_lst = ["acqf_1_db-1", "acqf_1_db-01", "acqf_1_db-001", 
                "acqf_2", "acqf_3", "acqf_4", "acqf_5", "acqf_6",
                "acqf_7", "acqf_8", "acqf_9", "random"]
    acqf2col = {"acqf_1_db-1"  : "lightskyblue", 
                "acqf_1_db-01" : "lightskyblue",
                "acqf_1_db-001": "lightskyblue",
                "acqf_2": "palegreen", 
                "acqf_3": "palegreen",
                "acqf_4": "palegreen",
                "acqf_5": "plum", 
                "acqf_6": "plum",
                "acqf_7": "plum", 
                "acqf_8": "plum", 
                "acqf_9": "plum",
                "random": "gray"}

# case where not all acqf are included
if aqcfs_included != "all":
    new_acqf_lst = []
    for x in aqcfs_included:
        if x in acqf_lst:
            new_acqf_lst.append(x)
        else:
            print(f"Acquisition function {x} not in list")
    
    acqf_lst = new_acqf_lst
    print(f"Acquisition functions included: {acqf_lst}")
    if "random" not in acqf_lst:
        acqf_lst.append("random")
    acqf2col = {k: v for k, v in acqf2col.items() if k in acqf_lst}
    acqf2col["random"] = "gray"
    print(f"Acquisition functions included: {acqf_lst}")
    print(f"Colors: {acqf2col}")

assert metric == "AUC" or metric == "SSP", "Invalid metric type"
assert figtype == "box" or figtype == "bar" or figtype == "violin", "Invalid figure type"

if not os.path.exists(out):
    os.mkdir(out)

# Evaluation 1: AUC
# assemble AUC results
def make_auc_results(acqf_lst, avg_run_data, rem_smiles, good_performing_smiles):
    auc_results = pd.DataFrame(columns=["SMILES"] + acqf_lst)
    for smi in avg_run_data:
        res = [smi]
        for acqf in acqf_lst:
            y = avg_run_data[smi][acqf]
            x = np.array(range(0, rem_smiles))
            res.append(auc(x, y)/rem_smiles)
        new_df = pd.DataFrame([res], columns = auc_results.columns)
        auc_results = pd.concat([auc_results, new_df])
    auc_results.reset_index(drop=True, inplace=True)

    trunc_auc_results = auc_results.loc[np.isin(auc_results.SMILES, good_performing_smiles)]
    return trunc_auc_results

def prep_auc_results(results, cluster):
    cols_to_drop = ["SMILES", "random"]
    if cluster != "all":
        cols_to_drop.append("cluster")
        results["cluster"] = [smi_to_cluster(s, clusters) for s in results["SMILES"]]
        results = results.loc[results["cluster"] == cluster]
        results.reset_index(drop=True, inplace=True)
        print(f"Cluster size: {len(results)}")

    baseline = results["random"]
    acqfs = results.drop(cols_to_drop, axis=1)
    for col in acqfs.columns:
        acqfs[col] = acqfs[col] - baseline
    return acqfs

# Evaluation 2: tset size at stabilized performance (SSP)
# assemble SSP results

# assemble performances about the number of reactions needed to achieve top-1 consistently.
def get_stable_index(run, max_, duration, threshold):
    count = 0
    for i, p in enumerate(run):
        if p >= threshold:
            count += 1
            if count == duration: # duration is the stability period demanded
                return i
        else:
            count = 0
    return max_

# assemble results
def make_ssp_results(acqf_lst, avg_run_data, threshold, duration):
    tset_results = pd.DataFrame(index=avg_run_data.keys(), columns=acqf_lst)
    for s in avg_run_data:
        for acqf in acqf_lst:
            tset_results.loc[s, acqf] = get_stable_index(avg_run_data[s][acqf], max_=len(avg_run_data[s][acqf]), duration=duration, threshold=perf_threshold)
    smis = tset_results.index
    tset_results["SMILES"] = smis
    tset_results.reset_index(drop=True, inplace=True)
    return tset_results

def prep_ssp_results(results, cluster, acqf_lst, rem_smiles):
    if cluster != "all":
        results["cluster"] = [smi_to_cluster(s, clusters) for s in results["SMILES"]]
        results = results.loc[results["cluster"] == cluster]
        results.reset_index(drop=True, inplace=True)
        print(f"Cluster size: {len(results)}")

    results = results.loc[:,~results.columns.duplicated()].copy()

    acqfs = {}
    max_samples = 0
    print(f"AFs: {acqf_lst}")
    for col in acqf_lst:
        acqf = []
        for i in range(len(results)):
            if results[col][i] != rem_smiles or results["random"][i] !=rem_smiles:

                acqf.append(results["random"][i] - results[col][i])
        print(f"{col} samples after filtering = {len(acqf)}")
        if len(acqf) > max_samples:
            max_samples = len(acqf)
        acqfs[col] = acqf
    
    del acqfs["random"]
    for col in acqfs:
        acqf = acqfs[col]
        diff = max_samples - len(acqf)
        if diff != 0:
            acqf.extend([np.nan] * diff)
        
    acqfs = pd.DataFrame(acqfs)
    return acqfs

def make_bar(acqf_lst, cluster, metric, prefix, out, avg_run_data,
              rem_smiles, good_performing_smiles, duration, threshold):
    if metric == "AUC":
        label = metric
        res = make_auc_results(acqf_lst, avg_run_data, rem_smiles, good_performing_smiles)
        acqfs = prep_auc_results(res, cluster)
    if metric == "SSP":
        label = "Size at Stabilized Performance"
        res = make_ssp_results(acqf_lst, avg_run_data, threshold, duration)
        acqfs = prep_ssp_results(res, cluster, acqf_lst, rem_smiles)
    means = acqfs.mean(axis=0)
    sds   = acqfs.std(axis=0)
    means.plot.bar(color=[acqf2col[a] for a in acqf_lst])

    plt.errorbar(x=means.index, y=means, yerr=sds, fmt="o", color="black")
    plt.ylabel(f"Mean improvement in {label} over baseline")
    handles = [plt.Rectangle((0,0),1,1, color='lightskyblue'), plt.Rectangle((0,0),1,1, color='palegreen'), plt.Rectangle((0,0),1,1, color='plum')]
    plt.legend(handles=handles, labels=['ACQF-1', 'Substructure-based methods', 'Carbon-based methods'], loc="lower right")
    plt.xticks(rotation=90)
    plt.savefig(f"{out}/{prefix}_bar_clust-{cluster}_metric-{metric}.png")
    plt.show()

def make_dist(acqf_lst, cluster, metric, dist_type, prefix, out, avg_run_data,
              rem_smiles, good_performing_smiles, duration, threshold):
    if metric == "AUC":
        label = metric
        res = make_auc_results(acqf_lst, avg_run_data, rem_smiles, good_performing_smiles)
        acqfs = prep_auc_results(res, cluster)
    
    if metric == "SSP":
        label = "Size at Stabilized Performance"
        res = make_ssp_results(acqf_lst, avg_run_data, threshold, duration)
        acqfs = prep_ssp_results(res, cluster, acqf_lst, rem_smiles)
    
    acqfs["SMILES"] = res["SMILES"]
    melted  = acqfs.melt(id_vars=["SMILES"])
    if paper_style:
        fig, ax = plt.subplots(figsize=(len(acqf_lst)*1, 5)) # 6,3
    else:
        fig, ax = plt.subplots(figsize=(len(acqf_lst)*1.5, 5)) #12,6
    
    if dist_type == "box":
        sns.boxplot(data=melted, x="variable", y="value", hue="variable",
                    dodge=False,
                    palette=acqf2col, ax=ax)
        sns.stripplot(data=melted, x="variable", y="value", color="black",
                      dodge=False, ax=ax)
        
    elif dist_type == "violin":
        sns.violinplot(data=melted,
                       cut=0, 
                       density_norm="area",
                       native_scale=True,
                       dodge=False,
                       x="variable", y="value", 
                       hue="variable", palette=acqf2col, 
                       ax=ax)
        sns.stripplot(data=melted, x="variable", y="value", color="black",
                      dodge=False, ax=ax)


    handles = [plt.Rectangle((0,0),1,1, color='lightskyblue'), 
               plt.Rectangle((0,0),1,1, color='palegreen'), plt.Rectangle((0,0),1,1, color='palegreen'), 
               plt.Rectangle((0,0),1,1, color='plum'), plt.Rectangle((0,0),1,1, color='plum')]
    #, loc="lower right")
    if paper_style:
        ax.set_ylabel(f"# Exp. spared vs. Baseline")
        try:
            ax.set_xticklabels(["AF-1", "AF-10", "AF-2", "AF-6"])
        except:
            pass
        ax.set_xlabel("Acquisition Function")
        ax.get_legend().remove()
        #ax.legend(handles=handles, labels=['Active Learning', 'Substructure-based methods', 'Carbon-based methods'], 
        #          bbox_to_anchor=(1, 1.0))
    else:
        ax.set_ylabel(f"Improvement in {label} over baseline")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='right')
        ax.set_xlabel("Acquisition function")
        ax.legend(handles=handles, labels=['Uncertainty Exploration', 'Mol. Sim', 'Mol. Sim', 'Site Sim.', 'Site Sim.'])
    fig.tight_layout()
    fig.savefig(f"{out}/{prefix}_dist_clust-{cluster}_metric-{metric}_disttype-{dist_type}.png", dpi=300)
    plt.show()
    return res

if args.path == "custom_db": # special case for custom descriptors with old dataset - enables to have all db values for acqf_1
    fnames = []
    paths = os.listdir("../../results/active_learning/regression")
    paths = [p for p in paths if "custom_db" in p] # keep only custom descriptors
    paths = [p for p in paths if "_b=" not in p]   # remove the batch experiments
    print(f"Looking for files in {paths} directories")
    for p in paths:
        print(p)
        path_       = f"../../results/active_learning/regression/{p}"
        path_fnames = os.listdir(path_) 
        path_fnames = [f for f in path_fnames if 'res_' in f]
        fnames = [f for f in fnames if 'acqf_10' not in f] # remove acqf_10 for now
        path_fnames = [f"../../results/active_learning/regression/{p}" + '/' + f for f in path_fnames if f[-3:] == "pkl"]
        fnames += path_fnames
    print(f"Found {len(fnames)} for {args.path}")
    print(f"{fnames[0]}")

else:        
    fnames = os.listdir(path_)
    fnames = [f for f in fnames if 'res_' in f]
    fnames = [f"../../results/active_learning/{args.path}" + '/' + f for f in fnames if f[-3:] == "pkl"]
    print(f"Found {len(fnames)} files in {path_}")
    print(f"{fnames[0]}")

all_smiles = {}
for fname in sorted(fnames):
    if fname[-3:] != "pkl":
        continue
    fname_split = fname.split("active_learning")[1]
    fname_split = fname_split.split("/")[-1]
    smi = [x for x in fname_split[:-4].split("_") if len(x) > 10][0]
    if smi not in all_smiles:
        all_smiles[smi] = [fname]
    else:
        all_smiles[smi].append(fname)

print(f"Acquisition function results for {len(all_smiles)} SMILES.")

df     = pd.read_csv(f"../../data/descriptors/{dataset}/df_bde.csv", index_col=0)
smiles = df.Reactant_SMILES.unique()
smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(smi)) for smi in smiles]
smiles = list(set(smiles))

def is_num_C_more_than_n(smiles, n):
    mol = Chem.MolFromSmiles(smiles)
    num_C = len([atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6])
    if num_C > n:
        return True
    else:
        return False
    
smiles_to_test = [smi for smi in smiles if is_num_C_more_than_n(smi, 15)]
print(f"{len(smiles_to_test)} target molecules in dataset")
good_smiles = [s for s in all_smiles if s in smiles_to_test]

for s in smiles_to_test:
    if s.replace("/", "-") in all_smiles: # fix smiles where "/" was removed
        s_vals = all_smiles[s.replace("/", "-")]
        del all_smiles[s.replace("/", "-")]
        all_smiles[s] = s_vals
    if s not in all_smiles:
        print(f"SMILES not calculated: {s}")

rem_smiles = len([smi for smi in smiles if not is_num_C_more_than_n(smi, 15)])
print(f"{rem_smiles} remaining molecules in dataset")

if dataset != "preprocessed_borylation_reactions": 
    with open('../../data/reaction_data/clusters.json') as json_file:
        clusters = json.load(json_file)
else:
    clusters = {}
    clusters["all"] = [s for s in all_smiles]

# make sure all smiles are in a cluster
def smi_to_cluster(s, clusters):
    for k in clusters:
        if s in clusters[k]:
            return k
    print(f"SMILES: {s}")
       
for s in all_smiles:
    assert smi_to_cluster(s, clusters) is not None

# assemble files + select smiles where good performance is achieved
# for this: just need average performance
maxes = []
avg_run_data = {}

for s in all_smiles:
    sub_max = 0
    fnames = all_smiles[s]
    
    # group by acqf
    acqfs = {a:[] for a in acqf_lst} 
    for fname in fnames:
        for acqf in acqfs:
            if acqf in fname:
                acqfs[acqf].append(fname)

    run_df = pd.DataFrame()

    for acqf, acqf_fnames in acqfs.items():
        runs = []
        for fname in acqf_fnames:
            with open(f"{fname}", 'rb') as f:
                 data = pickle.load(f)
            top_1 = list(np.array(data[0][0])[:, 0])
            if len(top_1) > rem_smiles + 2:
                print(f"check if this is OK maybbe you are using the large molecule?")
                print(len(top_1), fname)
            else:
                runs.append(top_1)
        
        avg_run = np.mean(np.array(runs), axis=0)
        
        try: 
            __, __, __ = acqf, s, max(avg_run)
        except:
            print(acqf, s, "No RUNs, please get the data!")
            exit()
        if sub_max < max(avg_run):
            sub_max = max(avg_run)
        run_df[acqf] = avg_run[:rem_smiles]
    maxes.append(sub_max)
    avg_run_data[s] = run_df

good_performing_smiles = [list(all_smiles.keys())[i] for i in range(len(all_smiles)) if maxes[i] > perf_threshold]
print(f"{len(good_performing_smiles)} achieve performance > {perf_threshold}")

if figtype == "bar":
    performance = make_bar(acqf_lst, cluster, metric, prefix, out, avg_run_data,
              rem_smiles, good_performing_smiles, duration, perf_threshold)
if figtype == "violin" or figtype == "box":
    performance = make_dist(acqf_lst, cluster, metric, figtype, prefix, out, avg_run_data,
              rem_smiles, good_performing_smiles, duration, perf_threshold)
              
performance.to_csv(f"perf_{out}.csv")


