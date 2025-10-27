# this python file computes metyrics on a dataframe that has the following columns:
# 'Atom_nº', 'Predicted_Selectivity', 'Selectivity', 'Reactant_SMILES'
# returns a list of scores for all molecules.

import pandas as pd
import numpy as np

def top_n(df, n, loose = False, feat="Selectivity"):
    """
    df contains the following columns:
    - Reactant_SMILES
    - Selectivity
    - Predicted_Selectivity
    - Atom_º
    loose: if True, if two atoms are ranked similarly the more experimentaly reactive one is considered as the most reactive (cheating)
           if False, if two atoms are ranked similarly the more experimentaly reactive one is considered as the less reactive (strict)
    Returns: number of reactive sites in the good ranking (good_count)
             number of reactive sites (n_reactive_sites)
    """

    topn = []
    for smiles in df.Reactant_SMILES.unique():
        df_smi           = df[df.Reactant_SMILES == smiles]
        df_smi_          = df_smi.sort_values(feat, ascending = False)
        atoms_order_true = df_smi_['Atom_nº'].values
        if loose == False:
            df_smi       = df_smi.sort_values(feat, ascending = True)
        df_smi           = df_smi.sort_values(f'Predicted_{feat}', ascending = False)
        atoms_order_pred = df_smi['Atom_nº'].values
        good_count       = 0
        if atoms_order_true[0] in atoms_order_pred[:n]:
            good_count = 1
        topn.append(good_count)
    return topn

def top_avg(df, loose = False, feat="Selectivity"):
    """
    df contains the following columns:
    - Reactant_SMILES
    - Selectivity
    - Predicted_Selectivity
    - Atom_º
    loose: of True, if two atoms are ranked similarly the more experimentaly reactive one is considered as the most reactive (cheating)
           if False, if two atoms are ranked similarly the more experimentaly reactive one is considered as the less reactive (strict)
    Returns: number of reactive sites in the good ranking (good_count)
             number of reactive sites (n_reactive_sites)
    """

    topn = []
    for smiles in df.Reactant_SMILES.unique():
        df_smi           = df[df.Reactant_SMILES == smiles]
        df_smi_          = df_smi.sort_values(feat, ascending = False)
        atoms_order_true = df_smi_['Atom_nº'].values
        if loose == False:
            df_smi       = df_smi.sort_values(feat, ascending = True)
        df_smi           = df_smi.sort_values(f'Predicted_{feat}', ascending = False)
        atoms_order_pred = df_smi['Atom_nº'].values
        topn_            = 1
        while atoms_order_true[0] not in atoms_order_pred[:topn_]:
            topn_ += 1
        topn.append(topn_)
    return np.sum(topn)/len(topn)  

def m2(df):
    """
    df contains the following columns:
    - Reactant_SMILES
    - Selectivity
    - Predicted_Selectivity
    - Atom_º
    Returns: number of reactive sites in the good ranking (good_count)/number of reactive sites
    """
    gc  = []
    for smiles in df.Reactant_SMILES.unique():
        df_smi           = df[df.Reactant_SMILES == smiles]
        df_smi_          = df_smi.sort_values('Selectivity', ascending = False)
        n_reactive_sites = df_smi_[df_smi_.Selectivity > 0].shape[0]
        atoms_order_true = df_smi_['Atom_nº'].values
        df_smi           = df_smi.sort_values('Predicted_Selectivity', ascending = False)
        atoms_order_pred = df_smi['Atom_nº'].values
        good_count = 0
        for i in range(n_reactive_sites):
            if atoms_order_true[i] == atoms_order_pred[i]:
                good_count += 1 
        gc.append(good_count/n_reactive_sites)
    return gc

def m3(df):
    # note has to be balanced y the size of the molecule, 
    # for a molecule with 2 reactive sites and 2 selectivity >0 the metric will always be 2  
        # add an if n reactive sites = n sites then check ordering 
    # done by returning gr - gr/possibilities
    """
    df contains the following columns:
    - Reactant_SMILES
    - Selectivity
    - Predicted_Selectivity
    - Atom_º
    Returns: number of reactive sites that are in the top n_reactive sites, does not care about their ordering
    """  
    go = []
    for smiles in df.Reactant_SMILES.unique():
        df_smi           = df[df.Reactant_SMILES == smiles]
        df_smi_          = df_smi.sort_values('Selectivity', ascending = False)
        n_reactive_sites = df_smi_[df_smi_.Selectivity > 0].shape[0]
        atoms_order_true = df_smi_['Atom_nº'].values
        df_smi           = df_smi.sort_values('Predicted_Selectivity', ascending = False)
        atoms_order_pred = df_smi['Atom_nº'].values
        good_reactive = 0
        for i in atoms_order_pred[:n_reactive_sites]:
            if i in atoms_order_true[:n_reactive_sites]:
                good_reactive += 1        
        go.append(round(good_reactive/n_reactive_sites, 3))
    return go

def m4(df):
    """
    df contains the following columns:
    - Reactant_SMILES
    - Selectivity
    - Predicted_Selectivity
    - Atom_º
    Returns: number of permutations needed to get the good ordering of reactive sites divided by the number of reactive sites and the number of posibilities 
    1 - np/(nrs*nsites) 
    """
    n_perm = []
    for smiles in df.Reactant_SMILES.unique():
        df_smi           = df[df.Reactant_SMILES == smiles]
        df_smi_          = df_smi.sort_values('Selectivity', ascending = False)
        atoms_order_true = df_smi_['Atom_nº'].values
        df_smi           = df_smi.sort_values('Predicted_Selectivity', ascending = False)
        atoms_order_pred = df_smi['Atom_nº'].values
        n_reactive_sites = df_smi[df_smi.Selectivity > 0].shape[0]
        n_permutations = 0
        for i in range(n_reactive_sites):
            n_permutations += abs(i - list(atoms_order_pred).index(atoms_order_true[i]))
        n_perm.append(round(1 - n_permutations/(len(atoms_order_true)*n_reactive_sites - 2*(n_reactive_sites - 1)),3))
        #print(atoms_order_true, atoms_order_pred, n_permutations, n_reactive_sites)
    return n_perm


def m5(df):
    """
    df contains the following columns:
    - Reactant_SMILES
    - Selectivity
    - Predicted_Selectivity
    - Atom_º
    Returns: MAE
    """
    m6 = []
    for smiles in df.Reactant_SMILES.unique():
        df_smi   = df[df.Reactant_SMILES == smiles]
        preds    = df_smi['Predicted_Selectivity'].values
        pred_sel = [100*(x - min(preds))/(max(preds)- min(preds)) for x in preds]   # Normalize data to between 0 and 100 witrh the sum being a 100% 
        mae      = 1 - sum([abs(x-y) for x,y in zip(pred_sel, df_smi['Selectivity'].values)])/(100*len(pred_sel))
        m6.append(round(mae, 3))
    return m6

def m6(df):
    """
    df contains the following columns:
    - Reactant_SMILES
    - Selectivity
    - Predicted_Selectivity
    - Atom_º
    Returns: MAE on exp of outputs
    """
    m5 = []
    for smiles in df.Reactant_SMILES.unique():
        df_smi = df[df.Reactant_SMILES == smiles]
      #  df_smi = df_smi.sort_values('Selectivity', ascending = False)
      #  df_smi = df_smi.sort_values('Predicted_Selectivity', ascending = False)
        preds  = df_smi['Predicted_Selectivity'].values
        pred_sel = [100*(np.exp(x) - np.exp(min(preds)))/(np.exp(max(preds)) - np.exp(min(preds))) for x in preds]
        mae = 1 - sum([abs(x-y) for x,y in zip(pred_sel, df_smi['Selectivity'].values)])/(100*len(pred_sel))
        m5.append(round(mae, 3))
    return m5

def m7(df):
    """
    df contains the following columns:
    - Reactant_SMILES
    - Selectivity
    - Predicted_Selectivity
    - Atom_º
    Returns: RMSE on exp of outputs
    """
    m5 = []
    for smiles in df.Reactant_SMILES.unique():
        df_smi = df[df.Reactant_SMILES == smiles]
     #   df_smi = df_smi.sort_values('Selectivity', ascending = False)
     #   df_smi = df_smi.sort_values('Predicted_Selectivity', ascending = False)
        preds = df_smi['Predicted_Selectivity'].values
        pred_sel = [100*(np.exp(x) - np.exp(min(preds)))/(np.exp(max(preds)) - np.exp(min(preds))) for x in preds]
        mae = 1 - sum([abs(x-y)**2 for x,y in zip(pred_sel, df_smi['Selectivity'].values)])/(10000*len(pred_sel))
        m5.append(round(mae, 3))
    return m5

def ns_pm(df):
    """
    Number of sites per molecules
    """
    n_sites = []
    for smiles in df.Reactant_SMILES.unique():
        df_smi = df[df.Reactant_SMILES == smiles]
        n_sites.append(df_smi.shape[0])
    return n_sites





