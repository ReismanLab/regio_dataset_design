import sys 
sys.path.append('..')
import preprocessing as pp
import descriptors as ds
import metrics as mt
import pandas as pd
import numpy as np
import ast
from tqdm import tqdm
from sklearn.base import clone
from rdkit import Chem

import os

try:
    base_cwd = os.getcwd().split('regio_dataset_design')[0]
    base_cwd = f"{base_cwd}/regio_dataset_design"
except:
    raise ValueError("You are not in the right directory, need to be in the 'notebooks' directory or subdirectory of it.")

# assembling pre-processing and featurization
def extract_features(descriptor, 
                     preprocess         = False,
                     file               = f"{base_cwd}/data/reaction_data/new_data_template.xlsx", 
                     print_             = False,
                     num_reactions_file = 'numbered_reaction_1.csv',
                     smi_col            = 'Reactant_SMILES',
                     rxn_folder="reaction_data", atom="O"):
    """
    1. Preprocess data using pp.preprocess (if preprocess is True)
    2. Extract descriptors and save to dataframe 
    descriptor: str(), 
        'Gasteiger', 'XTB', 'DBSTEP', 'DFT', 'ENV-1', 'ENV-2', 'BDE', 'Rdkit-Vbur', 'AIMNET-EMB'
    return value: dataframe with a column for SMILES and a column containing a dictionary of descriptors for all atoms in that molecule
    """

    if preprocess:
        num_reactions_file = pp.preprocess(file, smi_col=smi_col, rxn_folder=rxn_folder, atom=atom)
        df                 = pd.read_csv(num_reactions_file, index_col=0)
    else:
        df                 = pd.read_csv(f"{base_cwd}/data/{rxn_folder}/{num_reactions_file}", index_col=0)
    df.reset_index(drop=True, inplace=True)
    
    if print_:
        print(f"PREPROCESSING FINISHED: STARTING FEATURIZATION WITH {descriptor}")
    
    description = []

    # check if the SMILES used are canonical:
    for reactant in df.loc[:, 'Reactant_SMILES']:
        if reactant != Chem.CanonSmiles(reactant):
            print(f"{reactant} is not canonical")
            print("GRRRRRRRRRR")

    if descriptor == 'Gasteiger':
        for reactant in tqdm(df.loc[:, 'Reactant_SMILES']):
            desc = ds.Gasteiger(reactant, print_=print_, write=True, df_json=None) 
            description.append(desc)
            
    elif descriptor == 'XTB':
        for reactant in tqdm(df.loc[:, 'Reactant_SMILES']):
            desc = ds.xtb_CH(reactant, print_=print_, write=True, df_json=None) 
            description.append(desc)
            
    elif descriptor == 'DFT':
        for reactant in tqdm(df.loc[:, 'Reactant_SMILES']):
            desc = ds.dft_CH(reactant, print_=print_, write=True, df_json=None) 
            description.append(desc)

    elif descriptor == 'DBSTEP':
        for reactant in tqdm(df.loc[:, 'Reactant_SMILES']):
            desc = ds.dbstep_CH(reactant, print_=print_, write=True, df_json=None)  
            description.append(desc)

    elif descriptor == 'ENV-1':
        for reactant in tqdm(df.loc[:, 'Reactant_SMILES']):
            desc = ds.env1(reactant, print_=print_, write=True, df_json=None) 
            description.append(desc)
        if print_:
            print("ENV-1 DESCRIPTORS")
            print(f"len description: {description}")

    elif descriptor == 'ENV-2':
        for reactant in tqdm(df.loc[:, 'Reactant_SMILES']):
            desc = ds.env2(reactant, print_=print_) 
            description.append(desc)

    elif descriptor == 'BDE':
        for reactant in tqdm(df.loc[:, 'Reactant_SMILES']):
            desc = ds.bde(reactant, print_=print_) 
            description.append(desc)

    elif descriptor == 'Rdkit-Vbur':
        for reactant in tqdm(df.loc[:, 'Reactant_SMILES']):
            desc = ds.rdkit_conf_Vbur(reactant, print_=print_) 
            description.append(desc)

    elif descriptor == 'AIMNET-EMB':
        for reactant in tqdm(df.loc[:, 'Reactant_SMILES']):
            desc = ds.aimnet_embeddings(reactant, print_=print_) 
            description.append(desc)

    else:
        print(f"Descriptor: {descriptor} not found")
        return None
    
    df.loc[:, 'Descriptor'] = description
    return df

def remove_correlated_features(df_c_ox, print_, threshold_correlated):
    if print_:
        print(f"DISCARDING ZERO VARIANCE COLUMNS")
        print(f"columns before: {list(df_c_ox.columns)}")
    df_c_ox = df_c_ox.loc[:, df_c_ox.std() != 0]

    if print_:
        print(f"columns after: {list(df_c_ox.columns)}")
        print(f"DISCARDING CORRELATED DESCRIPTORS, threshold = {threshold_correlated}")
    corr_matrix = df_c_ox.corr().abs()
    upper       = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop     = [column for column in upper.columns if any(upper[column] > threshold_correlated)]
    if print_:
        print(f"DROPPING FEATURES: {to_drop}")
    df_c_ox = df_c_ox.drop(to_drop, axis=1)
    return df_c_ox

def drop_molecules_with_nas(df_c_ox, print_):
    if print_:
        print("REMOVING MOLECULES WITH NULLS.")
    
    null_reactants = []
    for reactant in df_c_ox['Reactant_SMILES'].unique():
        null_count = df_c_ox.loc[df_c_ox["Reactant_SMILES"] == reactant].isnull().sum().sum()
        if null_count > 0:
            null_reactants.append(reactant)

    df_c_ox = df_c_ox.loc[df_c_ox['Reactant_SMILES'].isin(null_reactants) == False]
    return df_c_ox   

def prepare_reactivity_mapping(descriptor, 
                               preprocess           = True,
                               file                 = f"{base_cwd}/data/reaction_data/new_data_template.xlsx", 
                               normalize            = False,
                               threshold_correlated = None,
                               print_               = False,
                               num_reactions_file   = 'numbered_reaction_1.csv',
                               smi_col              = 'Reactant_SMILES',
                               rxn_folder="reaction_data", atom="O"):
                    
    """"
    descriptor: str(),
        'Gasteiger', 'XTB', 'DBSTEP', 'DFT', 'ENV-1', 'ENV-2', 'BDE', 'Rdkit-Vbur'
        # the type of descriptor to use for the featurization see extract_features
    preprocess: bool(),
        # if True preprocesses the data again directly from the file
        # if False uses the preprocessed data (num_reactions_file)
        # default: True
    file: str(),
        # the file to preprocess
    normalize: bool(),
        # if True normalizes the descriptors
    threshold_correlated: float(),
    print_: bool(),
    num_reactions_file: str()
    """
    df = extract_features(descriptor, 
                          preprocess         = preprocess,
                          file               = file,     
                          print_             = print_,
                          num_reactions_file = num_reactions_file,
                          smi_col            = smi_col,
                          rxn_folder=rxn_folder, atom=atom)

    if print_:
        print("FINISHED FEATURIZATION: STARTING MAPPING DESCRIPTORS AND REACTIVITY")
        print("DF length: ", len(df))

    indexes       = []
    reactants     = []
    atom_number   = []
    selectivities = []
    descriptors   = {}
    columns_desc  = None
    index = 0

    for idx, sel in enumerate(df.loc[:, 'Selectivity_Reduced']):
        descriptor = df.loc[idx, 'Descriptor']
        try:
            _ = ast.literal_eval(sel)

        except:
            print(f"'Reactant_SMILES': {df.loc[idx, 'Reactant_SMILES']}")
            print(f"'Selectivity': {df.loc[idx, 'Selectivity_Reduced']}")
            raise ValueError("Issue with the selectivity")
        for key, value in ast.literal_eval(sel).items():
            try:
                descriptors.update({index: descriptor[key].values()})
                reactants.append(df.loc[idx, 'Reactant_SMILES'])
                atom_number.append(key)
                selectivities.append(value)
                indexes.append(index)
                index +=1
                if columns_desc is None:
                    columns_desc = list(descriptor[key].keys())
            except:
                if print_:
                    print(f"{df.loc[idx, 'Reactant_SMILES']} atom number {key} was not featurized ") #{descriptor}   
                pass
            
    if print_:
        print("FINISHED MAPPING DESCRIPTORS AND REACTIVITY, STARTING CONCATENATING MULTIPLE PRODUCT REACTIONS")

    df_c_ox = pd.DataFrame(descriptors)
    df_c_ox = df_c_ox.transpose()
    df_c_ox.columns = columns_desc

    if print_:
        print(df_c_ox.head(10))
    if normalize: # normalize the descriptors
        if print_:
            print("NORMALIZING DESCRIPTORS")
            print(f"len df_c_ox: {len(df_c_ox)}")
        for col in df_c_ox.columns:
            if sum(df_c_ox[col]) != 0:
                df_c_ox[col] = (df_c_ox[col] - df_c_ox[col].mean())/(df_c_ox[col].max() - df_c_ox[col].min())
        if print_:
            print(df_c_ox.head(10))
            print(f"len df_c_ox: {len(df_c_ox)}")
    
    if threshold_correlated != None: # remove correlated features
        df_c_ox = remove_correlated_features(df_c_ox, print_, threshold_correlated)
        if print_:
            print(f"len df_c_ox: {len(df_c_ox)} after removing correlated features")   

    df_c_ox.loc[:, 'Reactant_SMILES'] = reactants
    df_c_ox.loc[:, 'Atom_nº']         = atom_number
    df_c_ox.loc[:, 'Selectivity']     = selectivities
    
    # add the number of the most reactive when there are two products similarly reactive the first is chosen
    most_reactive = []
    for reactant in df.loc[:, 'Reactant_SMILES']:
        if reactant in reactants: # added for dft with reactant not featurized
            df_red = df_c_ox[df_c_ox.loc[:, 'Reactant_SMILES'] == reactant]
            idx_   = df_red[df_red.loc[:, 'Selectivity'] == max(df_red.loc[:, 'Selectivity'])].index[0]
            most_r = df_red.loc[idx_, 'Atom_nº']
            for i in range(len(df_red)):
                most_reactive.append(most_r)
        else:
            print(f"{reactant} has not been featurized")

    df_c_ox.loc[:, 'Reactive Atom'] = most_reactive
    if print_:
        print(f"len df_c_ox: {len(df_c_ox)} before dropping Nans")  
        print(df_c_ox.head(10))
    
    # remove molecules where the descriptors contain NAs:
    df_c_ox = df_c_ox.dropna(axis=0)
    df_c_ox = drop_molecules_with_nas(df_c_ox, print_)

    if print_:
        print(f"len df_c_ox before return: {len(df_c_ox)}")
    return df_c_ox

def prepare_featurization(descriptor, 
                               preprocess=False,
                               file=f"{base_cwd}/data/reaction_data/new_data_template.xlsx", 
                               normalize=False,
                               threshold_correlated=None,
                               print_=False,
                               num_reactions_file='numbered_reaction_1.csv',
                               smi_col='smiles', rxn_folder="reaction_data", atom="O"):
    
    df = extract_features(descriptor, 
                         preprocess=preprocess,
                         file=file, 
                         print_=True,
                         num_reactions_file=num_reactions_file,
                         smi_col=smi_col, rxn_folder=rxn_folder, atom=atom)

    description = df["Descriptor"]
    if print_:
        print("FINISHED FEATURIZATION: REMOVING SYMMETRIC ATOMS...")
    
    can_r    = []
    gr_idxs  = []
    for idx in tqdm(df.index):
        r_canon        = Chem.CanonSmiles(df.loc[idx, 'Reactant_SMILES'])
        can_r.append(r_canon)
        m = Chem.MolFromSmiles(r_canon)
        if pp.is_mol_symmetric(r_canon):
            sym_group = pp.group_symmetric_atoms(r_canon)[1]
            idx_in_group = pp.get_idx_per_groups(sym_group)
            final_groups = {}
            for group in idx_in_group:
                at_idx = idx_in_group[group][0]
                at = m.GetAtoms()[at_idx]
                if at.GetSymbol() == 'C' and at.GetTotalNumHs() > 0:
                    final_groups[group] = idx_in_group[group]
            gr_idxs.append(final_groups)
        else:
            keys = []
            list_keys = []
            for at in m.GetAtoms():
                if at.GetSymbol() == 'C' and at.GetTotalNumHs() > 0:
                    keys.append(at.GetIdx())
                    list_keys.append([at.GetIdx()])
            g_ = dict(zip(keys, list_keys))
            gr_idxs.append(g_)
        
    # account for symmetry
    atom_nos = []
    atom_desc = []
    reactants = []
    columns_desc  = None

    for i in tqdm(df.index):
        d = description[i]
        gr_idx = gr_idxs[i]
        
        for k in gr_idx:
            atom_idx_to_keep = gr_idx[k][0]
            atom_nos.append(atom_idx_to_keep)
            reactants.append(can_r[i])
            try:
                atom_desc.append(d[atom_idx_to_keep])
                if columns_desc is None:
                    columns_desc = list(d[atom_idx_to_keep].keys())
            except:
                atom_desc.append(np.nan)

    # make into dataframe:
    desc_dict = {col: [] for col in columns_desc}
    for atom in atom_desc:
        for k in desc_dict:
            try:
                desc_dict[k].append(atom[k])
            except:
                desc_dict[k].append(np.nan)

    df_c_ox = pd.DataFrame(desc_dict)
    df_c_ox.columns = columns_desc
    
    if normalize == True: # normalize the descriptors
        if print_:
            print("NORMALIZING DESCRIPTORS")
        df_c_ox = (df_c_ox - df_c_ox.mean())/(df_c_ox.max() - df_c_ox.min())

    if threshold_correlated != None: # remove correlated features
        df_c_ox = remove_correlated_features(df_c_ox, print_, threshold_correlated)
             

    df_c_ox.loc[:, 'Reactant_SMILES'] = reactants
    df_c_ox.loc[:, 'Atom_nº']         = atom_nos

    
    # remove molecules where the descriptors contain NAs:
    df_c_ox = drop_molecules_with_nas(df_c_ox, print_)

    return df_c_ox   

def leave_one_out(df_, reg, classif=False, feat="Selectivity"):
    """
    df_:
    reg:
    Output:
    """
    df_out = df_.copy()
    df     = df_.copy()
    count_true = 0
    l = list(df.loc[:, 'Reactant_SMILES'].unique())
    predictions = []
    
    for reactant in tqdm(df.loc[:, "Reactant_SMILES"].unique()):
        l_ = l.copy()
        l_.remove(reactant)
        reg_ = train_model(l_, df, reg, feat=feat)
        if reg_ is not None:
            bool_, y_pred = predict_site(reg_, reactant, df, classif=classif, feat=feat)
            #print(f"reactant: {reactant}, bool: {bool_}, y_pred: {y_pred.values()}")
            if bool_:
                count_true += 1

            idxs = df[df.loc[:, "Reactant_SMILES"] == reactant].index
            df_out.loc[idxs, f"Predicted_{feat}"] = list(y_pred.values()) # y_pred
        else:
            print(f"Issue training regressor with {reactant}")
    
    return count_true/len(l), df_out

# modelling tools
def train_model(reactant_list, df, reg, feat="Selectivity"):
    """
    Inputs:
        reactant_list: list of reactants to train the model
        df:            dataframe with the features of the reactant at least, can contain more, but not necessary. 
                       Has to contain the columns: 'Reactant_SMILES', 'Atom_nº', 'Selectivity', 'Reactive Atom'
        reg:           sklearn regressor
    Output:
        reg_:          trained model
    """
    df_c = df[df.loc[:, "Reactant_SMILES"].isin(reactant_list)]

    X = df_c.drop(columns=["Reactant_SMILES", "Atom_nº", feat, "Reactive Atom"]).values
    y = df_c.loc[:, feat].values

    reg_ = clone(reg)
    #try:
    reg_.fit(X, y)
    return reg_
    #except:
    #    print("Can't train model here... ?")
    #    print(X, y)
    #    return None

def predict_site(reg, reactant, df, classif=False, feat="Selectivity"):
    """
    Inputs:
        reg:       trained model
        reactant:  SMILES of the reactant to predict
        df:        dataframe with the features of the reactant at least, can contain more, but not necessary. Has to contain the columns: 'Reactant_SMILES', 'Atom_nº', 'Selectivity', 'Reactive Atom'
    Outputs:
        bool:      True if the prediction is correct, False otherwise
        y_pred:    list of the predicted selectivities for each site
    """
    df_r     = df[df.loc[:, "Reactant_SMILES"] == reactant]
    X        = df_r.drop(columns=["Reactant_SMILES", "Atom_nº", feat, "Reactive Atom"]).values    
    if classif:
        y_pred   = reg.predict_proba(X)
        y_pred   = list(y_pred)
        y_pred   = [y[list(reg.classes_).index(1)] for y in y_pred]
        if np.sum(y_pred) > 0:
            y_pred   = 100*np.array(y_pred)/(np.sum(y_pred))

    else:
        y_pred   = reg.predict(X)
        if np.sum(y_pred) > 0:
            y_pred   = 100*np.array(y_pred)/(np.sum(y_pred))
        y_pred   = list(y_pred)

    df_r.loc[:, 'Prediction'] = y_pred

    predictions = dict(zip(df_r.loc[:, 'Atom_nº'], y_pred))

    try:    
        idx_max  = df_r[df_r.loc[:, 'Prediction'] == max(df_r.loc[:, 'Prediction'])].index[0]
    except:
        print(f"Can't predict {reactant}")
        print(f"max = {max(df_r.loc[:, 'Prediction'])}, predictions  = {df_r.loc[:, 'Prediction'].to_list()}")
        return False, predictions # y_pred
    at_pred  = df_r.loc[idx_max, 'Atom_nº']
    at_true  = df_r.loc[idx_max, 'Reactive Atom']

    if at_true == at_pred:
        #print(reactant, at_true, at_pred)
        return True, predictions # y_pred
    else:
        #print(reactant, at_true, at_pred)
        return False, predictions # y_pred


def get_top_n_accuracy(df, n, feat="Selectivity"):
    accuracy = 0
    for reactant in df.Reactant_SMILES.unique():
        df_sub = df[df.Reactant_SMILES == reactant]
        df_sub = df_sub.sort_values(by=feat, ascending=False)
        top1 =  df_sub['Atom_nº'].iloc[0] # get top1 atom no

        sub_df = df[df.Reactant_SMILES == reactant]
        sub_df = sub_df.sort_values(by=f'Predicted_{feat}', ascending=False)
        n_most_reactive_atoms = list(sub_df['Atom_nº'].values)[:n]
        if top1 in n_most_reactive_atoms:
            accuracy += 1

    return accuracy/len(df.Reactant_SMILES.unique())

def baseline_alt(df, feat="Selectivity"):
    """
    This baseline takes as an input a DataFrame that has the colums: 'Reactant_SMILES', 'Atom_nº', 'Selectivity', 'Reactive Atom'
    
    This baseline is pretty simple that selects:

    1. Benzylic carbons when available
    2. The carbon with the minimum number of hydrogens bonded to it

    returns a top-1 accuracy
    This baseline performs slightly worse than the one reported below.
    """
    df = df[['Reactant_SMILES', 'Atom_nº', 'Reactive Atom', 'Selectivity']]

    H_neigbors  = []
    C_neigbors  = []
    Is_Benzylic = []

    for r in df.Reactant_SMILES.unique():
        for at_num in df[df.loc[:, 'Reactant_SMILES'] == r].loc[:, 'Atom_nº'].values:
            H_neigbors.append(num_h_neigbors(r, at_num))
            C_neigbors.append(num_c_neigbors(r, at_num))
            Is_Benzylic.append(is_benzylic(r, at_num))
            
    df.loc[:, 'H_neighbors'] = H_neigbors
    df.loc[:, 'C_neighbors'] = C_neigbors
    df.loc[:, 'Is_benzylic'] = Is_Benzylic

    predicted_atom   = []
    pred_selectivity = []  

    for r in tqdm(df.Reactant_SMILES.unique()):

        df_r  = df[df.Reactant_SMILES == r] # subset to specific reactant
        df_r_ = df_r.copy()
        num_C = len(df_r)
        df_benz = df_r[df_r.loc[:, 'Is_benzylic']]
        if len(df_benz) > 0:
            print("benzylic protons!")
            df_r = df_benz
        df_r = df_r.sort_values("H_neighbors", ascending=True)
        reactive_at = df_r.loc[:, 'Atom_nº'].values[0]

        for _ in range(num_C):
            predicted_atom.append(reactive_at)

        selectivity = [1 if x == reactive_at else 0 for x in df_r_.loc[:, 'Atom_nº'].values]
        pred_selectivity += selectivity

    df.loc[:, "Predicted_Atom"] = predicted_atom
    df.loc[:, f"Predicted_{feat}"] = pred_selectivity
    # analysis of the performance:

    for k in [1,2,3,5]:
        print(f"TOP-{k}: {get_top_n_accuracy(df, k, feat=feat)}")
    print(f"TOP-AVG: {mt.top_avg(df, feat=feat)}")

    count_true = 0 
    for r in df.Reactant_SMILES.unique():
        df_r  = df[df.Reactant_SMILES == r]
        r_atom = df_r.loc[:, 'Reactive Atom'].unique()[0]

        p_atom = df_r.loc[:, 'Predicted_Atom'].unique()[0]
        if r_atom == p_atom:
            count_true += 1

    return count_true/len(df.Reactant_SMILES.unique())
        

def baseline(df, feat="Selectivity"):
    """
    This baseline takes as an input a DataFrame that has the colums: 'Reactant_SMILES', 'Atom_nº', 'Selectivity', 'Reactive Atom'
    
    This baseline is pretty simple that selects:

    1. The carbon with the smallest positive number of hydrogen bonded
    2. In case of ambiguity: the atom with the most Csp2 neighbors
    3. In case of ambiguity: chooses randomly betwee best candidates

    returns a top-1 accuracy
    """
    df = df[['Reactant_SMILES', 'Atom_nº', 'Reactive Atom', feat]]

    H_neigbors  = []
    C_neigbors  = []
    Is_Benzylic = []

    for r in df.Reactant_SMILES.unique():
        for at_num in df[df.loc[:, 'Reactant_SMILES'] == r].loc[:, 'Atom_nº'].values:
            H_neigbors.append(num_h_neigbors(r, at_num))
            C_neigbors.append(num_c_neigbors(r, at_num))
            Is_Benzylic.append(is_benzylic(r, at_num))
            
    df.loc[:, 'H_neighbors'] = H_neigbors
    df.loc[:, 'C_neighbors'] = C_neigbors
    df.loc[:, 'Is_benzylic'] = Is_Benzylic

    predicted_atom   = []
    pred_selectivity = []  

    for r in tqdm(df.Reactant_SMILES.unique()):

        df_r  = df[df.Reactant_SMILES == r] # subset to specific reactant
        df_r_ = df_r.copy()
        num_C = len(df_r)
        min_H = min(df_r.H_neighbors.values) 
        df_r  = df_r[df_r.H_neighbors == min_H] # subset to just those with the minimal number of Hs

        if len(df_r) == 1: # easy case
            reactive_at    = df_r.loc[:, 'Atom_nº'].values[0]
        else:
            # check if all atoms are benzylic/non-benzylic
            if len(df_r.Is_benzylic.unique()) == 1: 
                reactive_at = df_r.loc[:, 'Atom_nº'].values[0] # random attribution between remaining atoms
            # take only the benzylic ones
            else:
                df_r = df_r[df_r.loc[:, 'Is_benzylic']]
                if len(df_r) == 1:
                    reactive_at    = df_r.loc[:, 'Atom_nº'].values[0]
                else:
                    reactive_at = df_r.loc[:, 'Atom_nº'].values[0] # random attribution between remaining atoms
                    
        for _ in range(num_C):
            predicted_atom.append(reactive_at)
        
        #print(f"reactant: {r}, reactive_at: {reactive_at}")
        #print(f"reactive atoms: {df_r.loc[:, 'Atom_nº'].values}")
        #print(f"{[1 if x == reactive_at else 0 for x in df_r.loc[:, 'Atom_nº'].values]}")
        #print(f"{len(df_r.loc[:, 'Atom_nº'].values)}, {num_C}")
        selectivity = [1 if x == reactive_at else 0 for x in df_r_.loc[:, 'Atom_nº'].values]
        pred_selectivity += selectivity

    df.loc[:, "Predicted_Atom"] = predicted_atom
    df.loc[:, f"Predicted_{feat}"] = pred_selectivity
    # analysis of the performance:

    for k in [1,2,3,5]:
        print(f"TOP-{k}: {get_top_n_accuracy(df, k, feat=feat)}")
    print(f"TOP-AVG: {mt.top_avg(df, feat=feat)}")

    count_true = 0 
    for r in df.Reactant_SMILES.unique():
        df_r  = df[df.Reactant_SMILES == r]
        r_atom = df_r.loc[:, 'Reactive Atom'].unique()[0]

        p_atom = df_r.loc[:, 'Predicted_Atom'].unique()[0]
        if r_atom == p_atom:
            count_true += 1

    return count_true/len(df.Reactant_SMILES.unique())

# baseline utils functions:

def num_h_neigbors(smiles, idx):
    m   = Chem.MolFromSmiles(smiles)
    m   = Chem.AddHs(m)
    n_H = 0
    for at in m.GetAtoms():
        if at.GetIdx() == idx:
            for n in at.GetNeighbors():
                if n.GetSymbol() == 'H':
                    n_H += 1
    return n_H

def num_c_neigbors(smiles, idx):
    m   = Chem.MolFromSmiles(smiles)
    m   = Chem.AddHs(m)
    n_H = 0
    for at in m.GetAtoms():
        if at.GetIdx() == idx:
            for n in at.GetNeighbors():
                if n.GetSymbol() == 'C':
                    n_H += 1
    return n_H

def is_benzylic(smiles, idx):
    m   = Chem.MolFromSmiles(smiles)
    m   = Chem.AddHs(m)
    benzylic = False
    for at in m.GetAtoms():
        if at.GetIdx() == idx:
            for n in at.GetNeighbors():
                if n.GetSymbol() == 'C' and n.GetHybridization() == Chem.HybridizationType.SP2:
                        benzylic = True
    return benzylic

