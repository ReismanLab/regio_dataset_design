import sys
sys.path.append('../utils/')

import os
root = os.getcwd()

try:
    base_cwd = os.getcwd().split('regio_dataset_design')[0]
    base_cwd = f"{base_cwd}/regio_dataset_design"
except:
    raise ValueError("You are not in the right directory.")

import numpy as np
rng = np.random.default_rng()
import pandas as pd
import modelling as md
import copy
import math
import random
from tqdm import tqdm
from random import randrange
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from rdkit.Chem import rdFMCS
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs

### PERFORMANCE EVALUATION ALONG DATASET GATHERING ###
def eval_perf(target_SMILES, training_SMILES, df, reg, acqf, 
              acqf_args_dict = {}, batch_size=1, 
              distance_balance=0.01, n_repet=20, alpha=1, feat="Selectivity",
              df_folder="preprocessed_dioxirane_reactions"):
    """
    Inputs:
        target_SMILES  : str, SMILES of the target molecule
        training_SMILES: list, of SMILES of the training molecules
        df             : dataframe, with the training data for all molecules 
        acqf           : str, acquisition function to be used
    Output:
        score_by_smiles_weighted: Series, with the score of each remaining molecule, sorted from the best to the worst
    """

    acqf_dict = {'random' : add_random_molecule,
                 'acqf_1' : acqf_1,
                 'acqf_2' : acqf_2,
                 'acqf_2-1': acqf_2_1,
                 'acqf_3' : acqf_out,
                 'acqf_4' : acqf_out,
                 'acqf_4-1': acqf_out,
                 'acqf_5' : acqf_out,
                 'acqf_6' : acqf_out,
                 'acqf_7' : acqf_out,
                 'acqf_8' : acqf_out,
                 'acqf_9' : acqf_out,
                 'acqf_10': acqf_10}
    
    #print(f"\n\nacqf_args_dict keys:  {acqf_args_dict.keys()}\n\n", flush=True)

    try:
        distance_balance = acqf_args_dict['distance_balance']
        n_repet          = acqf_args_dict['n_repet']
    except:
        pass
    
    acqf_args_dict          = {}  # reinit the dictionnary for aqcf > 3 

    top_5_scores             = []
    smiles_added             = []
    carbo_reac_predictions   = []
    df_remaining_            = df[df['Reactant_SMILES'].isin(training_SMILES) == False]
    df_remaining_            = df_remaining_[df_remaining_['Reactant_SMILES'] != target_SMILES]
    df_training              = df[df['Reactant_SMILES'].isin(training_SMILES)]
    
    # training the model round0 
    start_model              = md.train_model(df_training['Reactant_SMILES'].unique(), df_training.drop(columns=['DOI']), reg, feat=feat)

    # use the model to predict the site reactivities of the target molecule
    start_valid_, start_pred = md.predict_site(start_model, target_SMILES, df.drop(columns=['DOI']), classif=False, feat=feat)
    
    carbo_reac_predictions.append(start_pred)
    start_pred               = list(start_pred.items())
    top_5_scores.append(get_top_5(df, start_pred, target_SMILES))
    
    if acqf == 'acqf_1' or acqf == 'acqf_10':
        acqf_args_dict = {'n_repet': n_repet, 'distance_balance': distance_balance, 'alpha': alpha}
    elif acqf == 'acqf_3':
        out, max_scores              = make_acqf_3_order(target_SMILES, training_SMILES, df, df_folder)
        acqf_args_dict["out"]        = out
        acqf_args_dict["max_scores"] = max_scores
    elif acqf == 'acqf_4':
        out, max_scores              = make_acqf_4_order(target_SMILES, training_SMILES, df, df_folder)
        acqf_args_dict["out"]        = out
        acqf_args_dict["max_scores"] = max_scores
    elif acqf == 'acqf_4-1':
        out, max_scores              = make_acqf_4_1_order(target_SMILES, training_SMILES, df, df_folder)
        acqf_args_dict["out"]        = out
        acqf_args_dict["max_scores"] = max_scores
    elif acqf == 'acqf_5':
        out, max_scores              = make_acqf_5_order(target_SMILES, training_SMILES, df, feat=feat)
        acqf_args_dict["out"]        = out
        acqf_args_dict["max_scores"] = max_scores
    elif acqf == 'acqf_6':
        out, max_scores              = make_acqf_6_order(target_SMILES, training_SMILES, df, feat=feat)
        acqf_args_dict["out"]        = out
        acqf_args_dict["max_scores"] = max_scores
    elif acqf == 'acqf_7':
        out, max_scores              = make_acqf_7_order(target_SMILES, training_SMILES, df, feat=feat)
        acqf_args_dict["out"]        = out
        acqf_args_dict["max_scores"] = max_scores
    elif acqf == 'acqf_8':
        out, max_scores              = make_acqf_8_order(target_SMILES, training_SMILES, df, feat=feat)
        acqf_args_dict["out"]        = out
        acqf_args_dict["max_scores"] = max_scores
    elif acqf == 'acqf_9':
        out, max_scores              = make_acqf_9_order(target_SMILES, training_SMILES, df, feat=feat)
        acqf_args_dict["out"]        = out
        acqf_args_dict["max_scores"] = max_scores
    
    acqf_args_dict['batch_size'] = batch_size
    acqf_args_dict['feat'] = feat

    n_remaining = np.inf # some non-zero initialization
    
    max_scores_ = []
    while n_remaining >= batch_size: # remove the 40 to get the original behavior
        reg__                                                = clone(reg)
        acqf_args_dict["reg"]                                = reg__
        training_SMILES, n_remaining, new_SMILES, max_scores = acqf_dict[acqf](target_SMILES, training_SMILES, df, **acqf_args_dict)
        if acqf in ['acqf_1', 'acqf_10', 'acqf_2', 'acqf_2-1']:
            max_scores_.append(max_scores[0])
        smiles_added.extend(new_SMILES)

        reg__          = clone(reg)
        new_model      = md.train_model(training_SMILES, df.drop(columns=['DOI']), reg__, feat=feat)
        valid_, y_pred = md.predict_site(new_model, target_SMILES, df.drop(columns=['DOI']), classif=False, feat=feat)
        y_pred_list    = list(y_pred[x] for x in y_pred.keys())
        top5           = get_top_5(df, y_pred_list, target_SMILES)
        top_5_scores.append(top5)
        carbo_reac_predictions.append(y_pred)      
    
    if acqf in ['acqf_1', 'acqf_10', 'acqf_2', 'acqf_2-1']:
        max_scores = max_scores_
    
    return top_5_scores, smiles_added, carbo_reac_predictions, max_scores

def get_top_5(df, pred, target_SMILES, feat="Selectivity"):
    """
    Inputs:
        df           : dataframe, with the training data for all molecules 
        pred         : list, of predicted values
        target_SMILES: str, SMILES of the target molecule
    Output:
        top_5        : list, of the top 5 predicted values
    """
    df_pred = df[df['Reactant_SMILES'] == target_SMILES]
    df_pred[f'Predicted_{feat}']   = pred
    df_pred.sort_values(by=f'Predicted_{feat}', ascending=False, inplace=True)
    reactive_atom = df_pred['Reactive Atom'].unique()[0]
    #print(f"reactive_atom: {reactive_atom}", flush=True)

    top_5 = []
    for n in [1,2,3,5,10]:
        sub_df = df_pred.head(n)
        n_most_reactive_atoms = list(sub_df['Atom_nº'].values)[:n]
        #print(f"{n}_most_reactive_atoms: {n_most_reactive_atoms}", flush=True)
        if reactive_atom in n_most_reactive_atoms:
            top_5.append(1)
        else:
            top_5.append(0)
    return top_5


### RANDOM BASELINE ###
def add_random_molecule(target_SMILES, training_SMILES, df, 
                        reg=None, n_repet=None, distance_balance=None, batch_size=1, feat="Selectivity"):
    """"
    Inputs:
        training_SMILES: list, of SMILES of the training molecules
        df                : dataframe, with the training data for all molecules 
        target_SMILES     : str, SMILES of the target molecule
    Output:
        training_SMILES   : list, of SMILES of the training molecules updated with a random molecule
        n_remaining       : int, number of remaining molecules
        new_SMILES_lst    : list, of SMILES that were added in this batch
    """
    new_SMILES_lst = []
    for _ in range(batch_size):
        smiles_remaining   = df.loc[df['Reactant_SMILES'].isin(training_SMILES) == False, 'Reactant_SMILES'].unique()
        smiles_remaining_  = []
        for smi in smiles_remaining:
            if smi != target_SMILES:
                smiles_remaining_.append(smi)
        training_SMILES = list(training_SMILES)
        new_SMILES      = smiles_remaining_[randrange(len(smiles_remaining_))]
        training_SMILES.append(new_SMILES)
        new_SMILES_lst.append(new_SMILES)
    return training_SMILES, len(smiles_remaining_)-len(new_SMILES_lst), new_SMILES_lst, None


### FUNCTION 1 ###
def acqf_1(target_SMILES, training_SMILES, df, reg, n_repet=20, distance_balance=0.01, batch_size=1, alpha=1,
           feat="Selectivity"):
    """
    1. get the product of uncertainty avg reactivity per carbon of the target molecule based on the training of n_repet models
    2. get the matrix distance between the target C and the remaining molecules C
    3. compute the score of each remaining molecule based on the distance, the uncertainty and the reactivity
    Inputs:
        target_SMILES   : str, SMILES of the target molecule
        training_SMILES : list, of SMILES of the training molecules
        df              : dataframe, with the training data for all molecules    
        reg             : sklearn regressor, regression model
        n_repet         : int, number of repetitions of the training ad evaluation with different random seeds
        distance_balance: float, parameter to balance the distance with the score
        alpha           : float, parameter to balance the uncertainty with the reactivity, 
                            if alpha = 1, the balance is equal
                            if alpha < 1, the reactivity is more important
                            if alpha > 1, the uncertainty is more important
    Output:
        str, SMILES of the molecule with the highest score
    """
    df_remaining_            = df[df['Reactant_SMILES'].isin(training_SMILES) == False]
    df_remaining_            = df_remaining_[df_remaining_['Reactant_SMILES'] != target_SMILES]
    score                    = rank_carbon_uncertainty(target_SMILES, training_SMILES, df, reg, n_repet=n_repet, feat=feat, alpha=alpha)
    df_dists                 = target_C_distance_to_remaining(target_SMILES, df, df_remaining_, feat=feat)
    score_by_smiles_weighted = select_smiles_1(df_dists, score, distance_balance = distance_balance)
    training_SMILES  = list(training_SMILES) 
    new_SMILES_lst   = []
    for i in range(batch_size):
        new_SMILES      = score_by_smiles_weighted.index[i]
        training_SMILES.append(new_SMILES)
        new_SMILES_lst.append(new_SMILES)
    
    return training_SMILES, len(score_by_smiles_weighted.index) - batch_size, new_SMILES_lst, list(score_by_smiles_weighted.values)  

def rank_carbon_uncertainty(target_SMILES, training_SMILES, df, reg, feat, 
                            n_repet=20,
                            alpha=1):
    """
    Returns score of each carbon of the target molecule based on the uncertainty and the reactivity
    score = uncertainty * reactivity (std-deviation * mean reactivity)
    Inputs:
    target_SMILES  : str, SMILES of the target molecule
    training_SMILES: list, of SMILES of the training molecules
    df             : dataframe, with the training data for all molecules 
    reg            : sklearn regressor, regression model
    n_repet        : int, number of repetitions of the training ad evaluation with different random seeds
    alpha          : float, parameter to balance the uncertainty with the reactivity
    """
    df_training = df[df['Reactant_SMILES'].isin(training_SMILES)]
    # the ranking is based on the uncertainty and the reactivity:
    preds = []
    for i in range(n_repet):
        reg_ = clone(reg)
        reg_.random_state = i
        reg_.neighbors    = i + 1
        rf_trained  = md.train_model(df_training['Reactant_SMILES'].unique(),
                                      df_training.drop(columns=['DOI']), reg_, feat=feat)
        bool_, pred = md.predict_site(rf_trained, target_SMILES, df.drop(columns=['DOI']), feat=feat)
        preds.append(list(pred.values()))

    preds_mean = list(np.mean(preds, axis=0))
    preds_std  = list(np.std(preds, axis=0))
    # the best atom is the one with the maximum product of uncertainty and average reactivity
    score = [x**(2-alpha)*(y**(alpha)) for x,y in zip(preds_mean, preds_std)]
    return score

def target_C_distance_to_remaining(target_SMILES, df_custom, df_remaining, feat="Selectivity"):
    """
    target_SMILES: SMILES of the target molecule
    df_custom    : dataframe with the training data including the target SMILES descriptors
    df_remaining : dataframe with the remaining molecules descriptors
    """
    # get target SMILES C descriptors:
    target_C = df_custom[df_custom['Reactant_SMILES'] == target_SMILES]
    if len(target_C.DOI.unique()) != 1:
        print("WARNING: target molecule is not in only one reaction.... need to change target_C_distance_to_remaining function!")
    target_C.drop(columns=['DOI', 'Reactant_SMILES', feat, 'Reactive Atom'], inplace=True)
    target_C.set_index('Atom_nº', inplace=True)
    X_t     = target_C.values

    # get remaining molecules descriptors:
    remaining_molecules = df_remaining.drop(columns=['DOI', feat, 'Reactive Atom'])
    remaining_molecules.set_index(['Reactant_SMILES', 'Atom_nº'] , inplace=True)
    X_r     = remaining_molecules.values

    # get distance between remaining molecules C and target C carbons:
    dists   = euclidean_distances(X_r, X_t)
    columns = [f"dist_to_{x}" for x in target_C.index]
    dists   = pd.DataFrame(dists, index=remaining_molecules.index, columns=columns)
    return dists


def select_smiles_1(df_dists, score, distance_balance = 0.01):
    """
    Inputs:
        df_dists        : dataframe, with the distances between the target C and the remaining molecules C
        score           : list, score of the target C
        distance_balance: float, parameter to balance the distance with the score
            Note that the distace distribution is not normal, it ranges from 0 to 1.4 with a main peak around 0.2 for one target SMILES tested, 
            It might be advantageous to tune this parameter for each target SMILES 
    Output:
        score_by_smiles_weighted: Series, with the score of each remaining molecule, sorted from the best to the worst
    """
    df_selection = df_dists.copy()
    # 1.2.1. balance nearest neighbors distance with score
    for i, col in enumerate(df_selection.columns):
        # the affinity between target C and potential tested molecules is:
        # the product of the score of the target C with the inverse of it's distance to the candidate carbon
        df_selection[col] = df_selection[col].map(lambda x: score[i]/(x+distance_balance)) # +0.05 to avoid division by 0 THIS CONTROLS THE BALANCE BETWEEN DISTANCE AND SCORE

    # 1.2.2. compute a score per molecule in the remaining set: 
    # sum over target C columns:
    df_score        = df_selection.sum(axis=1)    
    score_by_smiles = df_score.sum(axis=0, level='Reactant_SMILES')


    # weight by the number of carbons in the molecule:
    df_selection.drop(columns=df_selection.columns, inplace=True)
    df_selection.reset_index(inplace=True)
    df_selection.set_index('Reactant_SMILES', inplace=True)
    df_selection['Atom_nº'] = df_selection.index.map(lambda x: 1)
    df_num_atoms            = df_selection.sum(axis=0, level='Reactant_SMILES')

    score_by_smiles_weighted = score_by_smiles/df_num_atoms['Atom_nº']
    score_by_smiles_weighted.sort_values(ascending=False, inplace=True)

    return score_by_smiles_weighted


### FUNCTION 2 ###
def acqf_2(target_SMILES, training_SMILES, df, 
           reg=None, n_repet=None, distance_balance=None, batch_size=1,
           feat="Selectivity"):
    """
    1. rank molecules in training_SMILES based on MCS similarity to target_SMILES
    2. return most similar molecule

    Inputs:
        target_SMILES   : str, SMILES of the target molecule
        training_SMILES : list, of SMILES of the training molecules
        df              : dataframe, with the training data for all molecules    
    Output:
        training_SMILES with additional SMILES added
        number of remaining molecules to be added to training set (before adding additional SMILES)
        str, SMILES of the molecule that was added to the training set
    """
    df_remaining_            = df[df['Reactant_SMILES'].isin(training_SMILES) == False]
    df_remaining_            = df_remaining_[df_remaining_['Reactant_SMILES'] != target_SMILES]
    
    target = Chem.MolFromSmiles(target_SMILES)
    remaining_molecules = df_remaining_.Reactant_SMILES.unique()

    # compute MCS similarity
    sim = []
    for smi in remaining_molecules:
        mol = Chem.MolFromSmiles(smi)
        res = rdFMCS.FindMCS([mol, target], ringMatchesRingOnly=True, timeout=60)
        num_atoms = res.numAtoms
        score = num_atoms/max(target.GetNumAtoms(), mol.GetNumAtoms())
        sim.append(score)
    
    out = pd.DataFrame({"Reactant_SMILES": remaining_molecules, "Score": sim})
    out.sort_values("Score", ascending=False, inplace=True)

    max_score_SMILES = list(out["Score"].values)  
    training_SMILES = list(training_SMILES)
    new_SMILES_lst  = out.Reactant_SMILES.to_list()[:batch_size]
    training_SMILES.extend(new_SMILES_lst)

    return training_SMILES, len(out.index) - batch_size, new_SMILES_lst, max_score_SMILES

def acqf_2_1(target_SMILES, training_SMILES, df, 
           reg=None, n_repet=None, distance_balance=None, batch_size=1, feat="Selectivity"):
    """
    1. rank molecules in training_SMILES based on MCS similarity to target_SMILES
    2. return most similar molecule
    
    this time...using MCS similarity that is stricter about ring matches

    Inputs:
        target_SMILES   : str, SMILES of the target molecule
        training_SMILES : list, of SMILES of the training molecules
        df              : dataframe, with the training data for all molecules    
    Output:
        training_SMILES with additional SMILES added
        number of remaining molecules to be added to training set (before adding additional SMILES)
        str, SMILES of the molecule that was added to the training set
    """
    df_remaining_            = df[df['Reactant_SMILES'].isin(training_SMILES) == False]
    df_remaining_            = df_remaining_[df_remaining_['Reactant_SMILES'] != target_SMILES]
    
    target = Chem.MolFromSmiles(target_SMILES)
    remaining_molecules = df_remaining_.Reactant_SMILES.unique()

    # compute MCS similarity
    sim = []
    for smi in remaining_molecules:
        mol = Chem.MolFromSmiles(smi)
        res = rdFMCS.FindMCS([mol, target], ringMatchesRingOnly=True, completeRingsOnly=True, bondCompare=rdFMCS.BondCompare.CompareOrderExact, timeout=60)
        num_atoms = res.numAtoms
        score = num_atoms/max(target.GetNumAtoms(), mol.GetNumAtoms())
        sim.append(score)
    
    out = pd.DataFrame({"Reactant_SMILES": remaining_molecules, "Score": sim})
    out.sort_values("Score", ascending=False, inplace=True)

    max_score_SMILES = list(out["Score"].values)  
    training_SMILES = list(training_SMILES)
    new_SMILES_lst  = out.Reactant_SMILES.to_list()[:batch_size]
    training_SMILES.extend(new_SMILES_lst)

    return training_SMILES, len(out.index) - batch_size, new_SMILES_lst, max_score_SMILES


### FUNCTION 3 ###
def compute_fingerprint_matrix(df, df_folder):
    smis = df.Reactant_SMILES.unique()
    fpgen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048)
    fps = []

    print("Computing fingerprints for all molecules...")
    for smi in tqdm(smis):
        m = Chem.MolFromSmiles(smi)
        fps.append(fpgen.GetFingerprint(m))

    print("Computing Tanimoto similarity for all molecule pairs:")
    matrix = np.empty((len(fps), len(fps)))
    for i in tqdm(range(len(fps))):
        for j in range(i, len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i],fps[j])
            matrix[i, j] = sim
            matrix[j, i] = sim
    out = pd.DataFrame(matrix)
    out["SMILES"] = smis
    out.to_csv(f"{base_cwd}/data/descriptors/{df_folder}/fp_sim.csv")


def make_fp_clusters(target_SMILES, training_SMILES, df, df_folder):
    """
    1. Cluster all training molecules based on fingerprint similarity (using affinity propagation)

    Inputs: 
        target_SMILES   : str, SMILES of the target molecule
        training_SMILES : list, of SMILES of the training molecules
        df              : dataframe, with the training data for all molecules (but not the large ones!!)

    Outputs:
        dictionary, with key as cluster index and value as the list of SMILES belonging to the cluster
    """
    try:
        matrix = pd.read_csv(f"{base_cwd}/data/descriptors/{df_folder}/fp_sim.csv")
        assert len(matrix) == len(df.Reactant_SMILES.unique())
    except:
        compute_fingerprint_matrix(df, df_folder)
        matrix = pd.read_csv(f"{base_cwd}/data/descriptors/{df_folder}/fp_sim.csv")

    df_remaining_            = df[df['Reactant_SMILES'].isin(training_SMILES) == False]
    df_remaining_            = df_remaining_[df_remaining_['Reactant_SMILES'] != target_SMILES]

    # remove rows of matrix
    matrix = matrix.loc[matrix["SMILES"].isin(df_remaining_['Reactant_SMILES']) == True]
    truncated_smi_lst = list(matrix["SMILES"])

    # remove columns of matrix 
    completed_mols = list(matrix.index)
    completed_mols = [str(x) for x in completed_mols]
    matrix         = matrix[completed_mols]

    rni = rng.integers(0, 1000, 1)[0]
    clustering = SpectralClustering(n_clusters=10, 
                                    random_state=rni,
                                    affinity="precomputed").fit(matrix)

    clusters = {}
    for i in range(len(clustering.labels_)):
        l   = clustering.labels_[i]
        smi = truncated_smi_lst[i]
        if l not in clusters:
            clusters[l] = []
        clusters[l].append(smi)
    return clusters

def make_acqf_3_order(target_SMILES, training_SMILES, df, df_folder):
    """
    1. Cluster all molecules based on MCS similarity (using affinity propagation)
    2. Alternate returning a random molecule from each cluster

    Inputs: 
        target_SMILES   : str, SMILES of the target molecule
        training_SMILES : list, of SMILES of the training molecules
        df              : dataframe, with the training data for all molecules (but not the large ones!!)

    Outputs:
        list of SMILES in order to be output
    """
    df_remaining_            = df[df['Reactant_SMILES'].isin(training_SMILES) == False]
    df_remaining_            = df_remaining_[df_remaining_['Reactant_SMILES'] != target_SMILES]

    clusters    = make_fp_clusters(target_SMILES, training_SMILES, df, df_folder)
    out         = []
    
    i           = 0
    size        = len(df_remaining_.Reactant_SMILES.unique())
    while size > 0:
        c_idx = i % len(clusters)
        c     = clusters[c_idx]
        i    += 1
        if len(c) == 0:
            continue
        smi   = random.choice(c)
        out.append(smi)
        c.remove(smi)
        size -= 1

    return out, None


### FUNCTION 4 ###
def make_acqf_4_order(target_SMILES, training_SMILES, df, df_folder):
    """
    1. Cluster all molecules based on MCS similarity (using affinity propagation)
    2. Sort clusters by MCS similarity to target_SMILES
    3. Alternate returning the most similar molecule from each cluster

    Inputs: 
        target_SMILES   : str, SMILES of the target molecule
        training_SMILES : list, of SMILES of the training molecules
        df              : dataframe, with the training data for all molecules (but not the large ones!!)

    Outputs:
        list of SMILES in order to be output
    """
    clusters = make_fp_clusters(target_SMILES, training_SMILES, df, df_folder)

    df_remaining_            = df[df['Reactant_SMILES'].isin(training_SMILES) == False]
    df_remaining_            = df_remaining_[df_remaining_['Reactant_SMILES'] != target_SMILES]

    # sort clusters by similarity to target smiles
    target              = Chem.MolFromSmiles(target_SMILES)
    remaining_molecules = df_remaining_.Reactant_SMILES.unique()

    # compute MCS similarity
    sim = []
    for smi in remaining_molecules:
        mol       = Chem.MolFromSmiles(smi)
        res       = rdFMCS.FindMCS([mol, target], ringMatchesRingOnly=True, timeout=60)
        num_atoms = res.numAtoms
        score     = num_atoms/max(target.GetNumAtoms(), mol.GetNumAtoms())
        sim.append(score)
    
    mcs_target = pd.DataFrame({"Reactant_SMILES": remaining_molecules, "Score": sim})
    mcs_target.sort_values("Score", ascending=False)

    for k in clusters:
        c           = clusters[k]
        subset      = mcs_target.loc[mcs_target['Reactant_SMILES'].isin(c) == True]
        subset      = subset.sort_values(by="Score", ascending=False)
        clusters[k] = list(subset["Reactant_SMILES"])

    out  = []
    i    = 0
    size = len(df_remaining_.Reactant_SMILES.unique())

    while size > 0:
        c_idx = i % len(clusters)
        c     = clusters[c_idx]
        i    += 1
        if len(c) == 0:
            continue
        smi   = c[0]
        out.append(smi)
        c.remove(smi)
        size -= 1

    return out, None

### FUNCTION 4.1 ###
def make_acqf_4_1_order(target_SMILES, training_SMILES, df, df_folder):
    """
    1. Cluster all molecules based on MCS similarity (using affinity propagation)
    2. Sort clusters by MCS similarity to target_SMILES
    3. Alternate returning the most similar molecule from each cluster

    Inputs: 
        target_SMILES   : str, SMILES of the target molecule
        training_SMILES : list, of SMILES of the training molecules
        df              : dataframe, with the training data for all molecules (but not the large ones!!)

    Outputs:
        list of SMILES in order to be output
    """
    clusters = make_fp_clusters(target_SMILES, training_SMILES, df, df_folder)

    df_remaining_            = df[df['Reactant_SMILES'].isin(training_SMILES) == False]
    df_remaining_            = df_remaining_[df_remaining_['Reactant_SMILES'] != target_SMILES]

    # sort clusters by similarity to target smiles
    target              = Chem.MolFromSmiles(target_SMILES)
    remaining_molecules = df_remaining_.Reactant_SMILES.unique()

    # compute MCS similarity
    sim = []
    for smi in remaining_molecules:
        mol       = Chem.MolFromSmiles(smi)
        res       = rdFMCS.FindMCS([mol, target], ringMatchesRingOnly=True, completeRingsOnly=True, bondCompare=rdFMCS.BondCompare.CompareOrderExact, timeout=60)
        num_atoms = res.numAtoms
        score     = num_atoms/max(target.GetNumAtoms(), mol.GetNumAtoms())
        sim.append(score)
    
    mcs_target = pd.DataFrame({"Reactant_SMILES": remaining_molecules, "Score": sim})
    mcs_target.sort_values("Score", ascending=False)

    for k in clusters:
        c           = clusters[k]
        subset      = mcs_target.loc[mcs_target['Reactant_SMILES'].isin(c) == True]
        subset      = subset.sort_values(by="Score", ascending=False)
        clusters[k] = list(subset["Reactant_SMILES"])

    out  = []
    i    = 0
    size = len(df_remaining_.Reactant_SMILES.unique())

    while size > 0:
        c_idx = i % len(clusters)
        c     = clusters[c_idx]
        i    += 1
        if len(c) == 0:
            continue
        smi   = c[0]
        out.append(smi)
        c.remove(smi)
        size -= 1

    return out, None

### FUNCTION 5 ###
def get_carbon_scores(target_SMILES, df_custom, df_remaining, feat="Selectivity"):
    """
    1. Compute carbon similarity matrix (using negative square of euclidean distance).
    2. Determine carbon score. Carbon score is the maximum similarity score with any target carbon.

    Inputs: 
        target_SMILES: SMILES of the target molecule
        df_custom    : dataframe with the training data including the target SMILES descriptors
        df_remaining : dataframe with the remaining molecules descriptors

    Outputs:
        Return list of carbon scores
    """
    target_C = df_custom[df_custom['Reactant_SMILES'] == target_SMILES]
    if len(target_C.DOI.unique()) != 1:
        print("WARNING: target molecule is not in only one reaction.... need to change target_C_distance_to_remaining function!")
    target_C.drop(columns=['DOI', 'Reactant_SMILES', feat, 'Reactive Atom', 'Atom_nº'], inplace=True)
    X_t     = target_C.values

    # get remaining molecules descriptors:
    remaining_molecules = df_remaining.drop(columns=['DOI', feat, 'Reactive Atom', 'Reactant_SMILES', 'Atom_nº'])
    X_r     = remaining_molecules.values

    # get distance between remaining molecules C and target C carbons:
    dists   = np.array(euclidean_distances(X_r, X_t))
    sims = - dists * dists
    columns = [f"sim_to_{x}" for x in target_C.index]
    sims   = pd.DataFrame(sims, index=remaining_molecules.index, columns=columns)

    target_dist_maxes = sims.apply(max, axis=1) # each carbon is represented by the max similarity to a carbon in the target
    return target_dist_maxes


def make_acqf_5_order(target_SMILES, training_SMILES, df, feat):
    """
    1. Compute carbon similarity matrix (using negative square of euclidean distance).
    2. Determine carbon score. Carbon score is the maximum similarity score with any target carbon.
    3. Determine molecule score. Molecule score is the average of all carbon scores of the molecule.
    4. Return list of SMILES, sorted by molecule score.

    Inputs: 
        target_SMILES   : str, SMILES of the target molecule
        training_SMILES : list, of SMILES of the training molecules
        df              : dataframe, with the training data for all molecules (but not the large ones!!)

    Outputs:
        list of SMILES in order to be output
    """
    df_remaining_            = df[df['Reactant_SMILES'].isin(training_SMILES) == False]
    df_remaining_            = df_remaining_[df_remaining_['Reactant_SMILES'] != target_SMILES]

    target_dist_maxes = get_carbon_scores(target_SMILES, df, df_remaining_, feat=feat)

    avg_carbon = []
    for smi in df_remaining_['Reactant_SMILES'].unique():
        smi_carbons = df.loc[df['Reactant_SMILES'] == smi].index
        avg_carbon.append(np.mean(target_dist_maxes[smi_carbons])) # each smiles is represented by the avg of carbon scores

    csim = pd.DataFrame({"smi": df_remaining_['Reactant_SMILES'].unique(), "avg": avg_carbon})
    csim = csim.sort_values("avg", ascending=False)

    return list(csim.smi), list(csim.avg)

### FUNCTION 6 ###
def make_acqf_6_order(target_SMILES, training_SMILES, df, feat):
    """
    1. Compute carbon similarity matrix (using negative square of euclidean distance).
    2. Determine carbon score. Carbon score is the maximum similarity score with any target carbon.
    3. Determine molecule score. Molecule score is the maximum of all carbon scores of the molecule.
    4. Return list of smiles, sorted by molecule score.

    Inputs: 
        target_SMILES   : str, SMILES of the target molecule
        training_SMILES : list, of SMILES of the training molecules
        df              : dataframe, with the training data for all molecules (but not the large ones!!)

    Outputs:
        list of SMILES in order to be output
    """
    df_remaining_            = df[df['Reactant_SMILES'].isin(training_SMILES) == False]
    df_remaining_            = df_remaining_[df_remaining_['Reactant_SMILES'] != target_SMILES]

    target_dist_maxes = get_carbon_scores(target_SMILES, df, df_remaining_, feat=feat)

    max_carbon = []
    for smi in df_remaining_['Reactant_SMILES'].unique():
        smi_carbons = df.loc[df['Reactant_SMILES'] == smi].index
        max_carbon.append(np.max(target_dist_maxes[smi_carbons])) # each smiles is represented by the max of carbon scores

    csim = pd.DataFrame({"smi": df_remaining_['Reactant_SMILES'].unique(), "max": max_carbon})
    csim = csim.sort_values("max", ascending=False)
    return list(csim.smi), list(csim["max"])

### FUNCTION 7 ###
def make_acqf_7_order(target_SMILES, training_SMILES, df, feat):
    """
    1. Combine acqf-5 and acqf-6, alternating adding from each list.

    Inputs: 
        target_SMILES   : str, SMILES of the target molecule
        training_SMILES : list, of SMILES of the training molecules
        df              : dataframe, with the training data for all molecules (but not the large ones!!)

    Outputs:
        list of SMILES in order to be output
    """
    csim_avg, val = make_acqf_5_order(target_SMILES, training_SMILES, df, feat=feat)
    csim_max, val = make_acqf_6_order(target_SMILES, training_SMILES, df, feat=feat)
    
    avg_idx = 0
    max_idx = 0
    tset = []
    size = len(df.Reactant_SMILES.unique())
    while len(tset) <= size:
        lst_size = len(tset)
        if lst_size % 2 == 0: # add average
            if avg_idx == len(csim_avg):
                tset += csim_max
                return tset, None

            smi_to_add = csim_avg[avg_idx]
            avg_idx += 1
        else: # add max
            if max_idx == len(csim_max):
                tset += csim_avg
                return tset, None
            smi_to_add = csim_max[max_idx]
            max_idx += 1
        
        if smi_to_add not in tset:
            tset.append(smi_to_add)
    return tset, None

### FUNCTION 8 ###
def get_carbon_clusters(target_SMILES, training_SMILES, df, feat="Selectivity"):
    """
    1. Cluster carbons using affinity propagation.
    2. For each SMILES in a cluster, get the number of carbons that belongs to that cluster.
    3. Rank SMILES in each cluster by number of carbons (descending order)

    Inputs: 
        target_SMILES   : str, SMILES of the target molecule
        training_SMILES : list, of SMILES of the training molecules
        df              : dataframe, with the training data for all molecules (but not the large ones!!)

    Outputs:
        dict, with key as cluster index and value as list of SMILES 
              sorted by which SMILES is most representative of a cluster
    """
    df_remaining_            = df[df['Reactant_SMILES'].isin(training_SMILES) == False]
    df_remaining_            = df_remaining_[df_remaining_['Reactant_SMILES'] != target_SMILES]
    df_remaining_            = df_remaining_.reset_index(drop=True)
    rem_C = df_remaining_.drop(columns=['DOI', 'Reactant_SMILES', feat, 'Reactive Atom', 'Atom_nº'], inplace=False)

    rni = rng.integers(0, 1000, 1)[0]
    clustering = KMeans(n_clusters=10, random_state=rni).fit(rem_C)
    clusters = {}

    # for each SMILES in a cluster, get the number of carbons that belongs to that cluster
    for i in range(len(df_remaining_.index)): 
        l = clustering.labels_[i]

        if l not in clusters:
            clusters[l] = {}
        
        smi = df_remaining_.Reactant_SMILES[i]

        if smi not in clusters[l]:
            clusters[l][smi] = 0
        clusters[l][smi] += 1
    
    # rank clusters by number of carbons
    for k in clusters: 
        smi = []
        count = []
        c = clusters[k]
        for j in c:
            smi.append(j)
            count.append(c[j])
        df = pd.DataFrame({"SMILES": smi, "count": count})
        df = df.sort_values("count", ascending=False)
        clusters[k] = list(df.SMILES)
    return clusters

def make_acqf_8_order(target_SMILES, training_SMILES, df, feat):
    """
    1. Compute carbon similarity clusters.
    2. Alternate between clusters, selecting the most representative molecule each time.

    Inputs: 
        target_SMILES   : str, SMILES of the target molecule
        training_SMILES : list, of SMILES of the training molecules
        df              : dataframe, with the training data for all molecules (but not the large ones!!)

    Outputs:
        list of SMILES in order to be output
    """
    clusters = copy.deepcopy(get_carbon_clusters(target_SMILES, training_SMILES, df, feat=feat))
    out = []
    i = 0

    df_remaining_            = df[df['Reactant_SMILES'].isin(training_SMILES) == False]
    df_remaining_            = df_remaining_[df_remaining_['Reactant_SMILES'] != target_SMILES]
    size = len(df_remaining_.Reactant_SMILES.unique())

    while size > 0:
        c_idx = i % len(clusters)
        c = clusters[c_idx]
        i += 1
        if len(c) == 0:
            continue
        smi = c[0]
        c.remove(smi)
        if smi in out:
            i -= 1
            continue
        out.append(smi)
        size -= 1
    return out, None

### FUNCTION 9 ###
def make_target_carbon_clusters(target_SMILES, training_SMILES, df, feat="Selectivity"):
    """
    1. Compute carbon similarity matrix.
    2. For each target carbon, rank all other carbons by similarity.

    Inputs: 
        target_SMILES   : str, SMILES of the target molecule
        training_SMILES : list, of SMILES of the training molecules
        df              : dataframe, with the training data for all molecules (but not the large ones!!)

    Outputs:
        dict, with key as target carbon index and value as list of SMILES 
              sorted by which SMILES contains the carbon most similar to the target carbon
    """
    target_carbons = df.loc[df['Reactant_SMILES'] == target_SMILES].reset_index(drop=True)
    target_carbons.drop(columns=['DOI', feat, 'Reactive Atom', 'Reactant_SMILES', 'Atom_nº'], inplace=True)

    df_remaining_            = df[df['Reactant_SMILES'].isin(training_SMILES) == False]
    df_remaining_            = df_remaining_[df_remaining_['Reactant_SMILES'] != target_SMILES]
    
    # get remaining molecules descriptors:
    remaining_molecules = df_remaining_.drop(columns=['DOI', feat, 'Reactive Atom', 'Reactant_SMILES', 'Atom_nº'])
    X_r     = remaining_molecules.values

    carbon_dict = {}
    for i in range(len(target_carbons)):
        target_C = target_carbons.iloc[i].copy()
        X_t     = target_C.values
        X_t = X_t.reshape(1, -1)
        
        # get similarity between remaining molecules C and target carbon:
        dists   = np.array(euclidean_distances(X_r, X_t))
        sims = - dists * dists
        sims   = pd.DataFrame(sims, index=remaining_molecules.index)
        c_dist = sims.apply(max, axis=1) # each carbon is represented by the max similarity to a carbon in the target
        c_dist = pd.DataFrame({"max": c_dist, "Reactant_SMILES": df_remaining_.Reactant_SMILES})

        sorted_c_dist = c_dist.sort_values(by = "max", ascending=False) 
        carbon_dict[i] = list(sorted_c_dist.Reactant_SMILES) # sorted SMILES for each target carbon
    
    return carbon_dict

def make_acqf_9_order(target_SMILES, training_SMILES, df, feat):
    """
    1. Compute carbon similarity clusters.
    2. Alternate between clusters, selecting the most representative molecule each time.

    Inputs: 
        target_SMILES   : str, SMILES of the target molecule
        training_SMILES : list, of SMILES of the training molecules
        df              : dataframe, with the training data for all molecules (but not the large ones!!)

    Outputs:
        list of SMILES in order to be output
    """
    carbon_dict = copy.deepcopy(make_target_carbon_clusters(target_SMILES, training_SMILES, df, feat=feat))

    out = []

    i = 0
    df_remaining_            = df[df['Reactant_SMILES'].isin(training_SMILES) == False]
    df_remaining_            = df_remaining_[df_remaining_['Reactant_SMILES'] != target_SMILES]
    size = len(df_remaining_.Reactant_SMILES.unique())
    while size > 0:
        c_idx = i % len(carbon_dict)
        c = carbon_dict[c_idx]
        i += 1
        if len(c) == 0:
            continue
        smi = c[0]
        c.remove(smi)
        if smi in out:
            i -= 1
            continue
        out.append(smi)
        size -= 1
    return out, None

def acqf_out(target_SMILES, training_SMILES, df, out, max_scores, batch_size=1, reg=None, feat="Selectivity"):
    """
    1. Output top molecule in out list that is not in training_SMILES

    Inputs:
        target_SMILES   : str, SMILES of the target molecule
        training_SMILES : list, of SMILES of the training molecules
        df              : dataframe, with the training data for all molecules    
        out             : list of SMILES in the order to be output
    Output:
        training_SMILES with additional SMILES added
        number of remaining molecules to be added to training set (before adding additional SMILES)
        str, SMILES of the molecule that was added to the training set
    """
    out        = [x for x in out if x not in training_SMILES]
    new_SMILES = out[:batch_size]
    
    training_SMILES = list(training_SMILES)
    training_SMILES.extend(new_SMILES)

    return training_SMILES, len(out) - batch_size, new_SMILES, max_scores

### FUNCTION 10  ###
def acqf_10(target_SMILES, training_SMILES, df, reg, n_repet=20, distance_balance=0.01, batch_size=1, alpha=1,
            feat="Selectivity"):
    """
    1. get the product of uncertainty avg reactivity per carbon of the target molecule based on the training of n_repet models
    2. get the matrix distance between the target C and the remaining molecules C
    3. compute the score of each remaining molecule based on the distance, the uncertainty and the reactivity

    ACQF 1, but biased towards molecules with fewer reactive sites
    Inputs:
        target_SMILES   : str, SMILES of the target molecule
        training_SMILES : list, of SMILES of the training molecules
        df              : dataframe, with the training data for all molecules    
        reg             : sklearn regressor, regression model
        n_repet         : int, number of repetitions of the training ad evaluation with different random seeds
        distance_balance: float, parameter to balance the distance with the score
    Output:
        str, SMILES of the molecule with the highest score
    """
    df_remaining_            = df[df['Reactant_SMILES'].isin(training_SMILES) == False]
    df_remaining_            = df_remaining_[df_remaining_['Reactant_SMILES'] != target_SMILES]
    score                    = rank_carbon_uncertainty(target_SMILES, training_SMILES, df, reg, n_repet=n_repet, feat=feat, alpha=alpha)
    df_dists                 = target_C_distance_to_remaining(target_SMILES, df, df_remaining_, feat=feat)
    score_by_smiles_weighted = select_smiles_2(df_dists, score, distance_balance = distance_balance)
    training_SMILES  = list(training_SMILES) 
    new_SMILES_lst   = []
    for i in range(batch_size):
        new_SMILES      = score_by_smiles_weighted.index[i]
        training_SMILES.append(new_SMILES)
        new_SMILES_lst.append(new_SMILES)
    
    return training_SMILES, len(score_by_smiles_weighted.index) - batch_size, new_SMILES_lst, list(score_by_smiles_weighted.values)  


def select_smiles_2(df_dists, score, distance_balance = 0.01):
    """
    select_smiles_1, now with carbon count bias!
    Inputs:
        df_dists        : dataframe, with the distances between the target C and the remaining molecules C
        score           : list, score of the target C
        distance_balance: float, parameter to balance the distance with the score
            Note that the distace distribution is not normal, it ranges from 0 to 1.4 with a main peak around 0.2 for one target SMILES tested, 
            It might be advantageous to tune this parameter for each target SMILES 
    Output:
        score_by_smiles_weighted: Series, with the score of each remaining molecule, sorted from the best to the worst
    """
    df_selection = df_dists.copy()
    # 1.2.1. balance nearest neighbors distance with score
    for i, col in enumerate(df_selection.columns):
        # the affinity between target C and potential tested molecules is:
        # the product of the score of the target C with the inverse of it's distance to the candidate carbon
        df_selection[col] = df_selection[col].map(lambda x: score[i]/(x+distance_balance)) # +0.05 to avoid division by 0 THIS CONTROLS THE BALANCE BETWEEN DISTANCE AND SCORE

    # 1.2.2. compute a score per molecule in the remaining set: 
    # sum over target C columns:
    df_score        = df_selection.sum(axis=1)    
    score_by_smiles = df_score.sum(axis=0, level='Reactant_SMILES')

    # weight by the number of carbons in the molecule:
    df_selection.drop(columns=df_selection.columns, inplace=True)
    df_selection.reset_index(inplace=True)
    df_selection.set_index('Reactant_SMILES', inplace=True)
    df_selection['Atom_nº'] = df_selection.index.map(lambda x: 1)
    df_num_atoms            = df_selection.sum(axis=0, level='Reactant_SMILES')

    score_by_smiles_weighted = score_by_smiles/(df_num_atoms['Atom_nº'] ** 2)   
    score_by_smiles_weighted.sort_values(ascending=False, inplace=True)

    return score_by_smiles_weighted

### FUNCTION 11 ###
def acqf_11(target_SMILES, training_SMILES, df, reg, lit_dict, n_repet=20, distance_balance=0.01, batch_size=1, feat="Selectivity"):
    """
    1. get the product of uncertainty avg reactivity per carbon of the target molecule based on the training of n_repet models
    2. get the matrix distance between the target C and the remaining molecules C
    3. compute the score of each remaining molecule based on the distance, the uncertainty and the reactivity

    ACQF 1, but biased towards molecules with fewer reactive sites
    Inputs:
        target_SMILES   : str, SMILES of the target molecule
        training_SMILES : list, of SMILES of the training molecules
        df              : dataframe, with the training data for all molecules    
        reg             : sklearn regressor, regression model
        lit_dict      : dict labelling literature molecules as 1 and non-literature molecules as 0
        n_repet         : int, number of repetitions of the training ad evaluation with different random seeds
        distance_balance: float, parameter to balance the distance with the score
    Output:
        str, SMILES of the molecule with the highest score
    """
    df_remaining_            = df[df['Reactant_SMILES'].isin(training_SMILES) == False]
    df_remaining_            = df_remaining_[df_remaining_['Reactant_SMILES'] != target_SMILES]
    score                    = rank_carbon_uncertainty(target_SMILES, training_SMILES, df, reg, n_repet=n_repet, feat=feat)
    df_dists                 = target_C_distance_to_remaining(target_SMILES, df, df_remaining_, feat=feat)
    score_by_smiles_weighted = select_smiles_2(df_dists, score, lit_dict, distance_balance = distance_balance)
    training_SMILES  = list(training_SMILES) 
    new_SMILES_lst   = []
    for i in range(batch_size):
        new_SMILES      = score_by_smiles_weighted.index[i]
        training_SMILES.append(new_SMILES)
        new_SMILES_lst.append(new_SMILES)
    
    return training_SMILES, len(score_by_smiles_weighted.index) - batch_size, new_SMILES_lst, list(score_by_smiles_weighted.values)  
