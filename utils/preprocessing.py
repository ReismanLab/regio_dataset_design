# this file is used to generate a reaction dataset cleaned from the raw excel sheet. It can either preprocess the data by attributing binary classification for the reactive and non reactive sites or the relative selectivity observed.

import ast
import numpy as np
import pandas as pd
from   tqdm import tqdm
from   rdkit import Chem
from   rdkit.Chem import AllChem, Draw
import os 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from rxnmapper import RXNMapper
rxn_mapper = RXNMapper()

root = os.getcwd()

try:
    base_cwd = os.getcwd().split('regio_dataset_design')[0]
    base_cwd = f"{base_cwd}/regio_dataset_design"
except:
    raise ValueError("You are not in the right directory, need to be in the 'notebooks' directory or subdirectory of it.")

def preprocess(file, rxn_folder="reaction_data", smi_col='Reactant_SMILES', atom="O"):
    try:
        df = pd.read_csv(file)
    except:
        df = pd.read_excel(file)
    if "Selectivity (%)" or "Yield (%)" in df.columns:
        return preprocess_reactions(df, rxn_folder=rxn_folder, smi_col=smi_col, atom=atom)
    else:
        return preprocess_molecules(df, file, rxn_folder=rxn_folder, smi_col=smi_col)

def preprocess_molecules(df, file, smi_col='Reactant_SMILES', rxn_folder="reaction_data"):
    # drop discard reactions
    try:
        df = df[df.loc[:, 'Discard'] != 'Yes']
    except:
        pass
    
    # drop reactions with no reactant
    df['Reactant_SMILES'] = df[smi_col]
    df = df[df.loc[:, 'Reactant_SMILES'] == df.loc[:, 'Reactant_SMILES']] 
    df.reset_index(drop=True, inplace=True)
    
    # make sure smiles is canonical
    df.Reactant_SMILES = df.Reactant_SMILES.map(lambda x: Chem.CanonSmiles(x))

    # write smiles to file
    df_ = df[['Reactant_SMILES']]
    df_.reset_index(inplace=True, drop=True)
    df_.drop_duplicates(subset='Reactant_SMILES', keep='first', inplace=True)

    print("First part of the cleaning completed")

    outpath = f"{base_cwd}/data/{rxn_folder}/numbered_molecules_" + file.split("/")[-1].split(".")[0] + '.csv'
    df_.to_csv(outpath)
    return outpath

def preprocess_reactions(df, smi_col="Reactant_SMILES", rxn_folder="reaction_data", atom="O") :
    '''
    This function reads the raw data, and takes in just the reactants and products, combining them into a reaction. 
    All of the reactions in the raw data are compiled into an array.

    Input:
    - df: 
    '''
    # basic pre cleaning:
    try:
        df = df[df.loc[:, 'Discard'] != 'Yes'] # drop discard reactions
    except:
        pass

    df['Reactant_SMILES'] = df[smi_col]
    df = df[df.loc[:, 'Reactant_SMILES'] == df.loc[:, 'Reactant_SMILES']] # drop reactions with no reactant
    df.reset_index(drop=True, inplace=True)

    # make sure smiles reactant and product are canonical
    df.Reactant_SMILES = df.Reactant_SMILES.map(lambda x: Chem.CanonSmiles(x))

    # drop rows with no yield or selectivity data
    for idx in df.index:
        if df.loc[idx, 'Selectivity (%)'] == df.loc[idx, 'Selectivity (%)']:
            df.loc[idx, 'Selectivity (%)'] = float(df.loc[idx, 'Selectivity (%)']) 
        elif df.loc[idx, 'Yield (%)'] == df.loc[idx, 'Yield (%)']:
            df.loc[idx, 'Selectivity (%)'] = float(df.loc[idx, 'Yield (%)'])
        else:
            df.drop(index = [idx], inplace=True) 
    df.reset_index(drop=True, inplace=True)
    
    can_r    = []
    map_r    = []
    can_p    = []
    map_p    = []
    symmetry = []
    u_sym    = []
    groups   = []
    gr_idxs  = [] 
    map_rxn  = []
    map_reac = []
    idx_rxn_ = []
    idx_sel  = []
    

    for idx in tqdm(df.index):
        try:
            r_canon        = Chem.CanonSmiles(df.loc[idx, 'Reactant_SMILES'])
            p_canon        = Chem.CanonSmiles(df.loc[idx, 'Product_SMILES'])
        except:
            print(f"Error in the canonical smiles for reaction {idx}")
            print(f"Reactant: {df.loc[idx, 'Reactant_SMILES']}")
            print(f"Product: {df.loc[idx, 'Product_SMILES']}")
            print(f"Reaction: {df.loc[idx, 'Reaction']}, selectivity: {df.loc[idx, 'Selectivity (%)']}")
            raise ValueError("Error in the canonical smiles")
        rxn_smiles     = f"{r_canon}>>{p_canon}"
        mapped_rxn     = rxn_mapper.get_attention_guided_atom_maps([rxn_smiles])
        mapped_rxn     = mapped_rxn[0]['mapped_rxn']
        r_m, p_m       = mapped_rxn.split('>>')
        try:
            map_reactivity = reactive_carbon(r_m, p_m, product_atom=atom)
        except:
            print (f"Error in the reactive_carbon function for reaction {idx}")
            print (f"Reactant: {df.loc[idx, 'Reactant_SMILES']}")
            print (f"Product: {df.loc[idx, 'Product_SMILES']}")
            print (f"Reaction: {rxn_smiles}, selectivity: {df.loc[idx, 'Selectivity (%)']}")
            raise ValueError("Error in the reactive_carbon function")
        idx_to_rxn     = atom_idx_to_rxn_map(r_canon, r_m, draw=False)
        can_r.append(r_canon)
        can_p.append(p_canon)
        map_rxn.append(mapped_rxn)
        map_r.append(r_m)
        map_p.append(p_m)
        map_reac.append(map_reactivity)
        idx_rxn_.append(idx_to_rxn)
        idx_sel.append(get_selectivity(map_reactivity, idx_to_rxn, df.loc[idx, 'Selectivity (%)']))
        
        if is_mol_symmetric(r_canon):
            symmetry.append(True)
            sym_group = group_symmetric_atoms(r_canon)[1]
            groups.append(sym_group)
            gr_idxs.append(get_idx_per_groups(sym_group))
            u_sym.append(drop_symmetric_carbons(sym_group))
            
        else:
            symmetry.append(False)
            m = Chem.MolFromSmiles(r_canon)
            keys = []
            list_keys = []
            for at in m.GetAtoms():
                if at.GetSymbol() == 'C':
                    keys.append(at.GetIdx())
                    list_keys.append([at.GetIdx()])
            g  = dict(zip(keys, keys))
            g_ = dict(zip(keys, list_keys))
            groups.append(g)
            gr_idxs.append(g_)
            u_sym.append(g)
            
    df.loc[:, 'Reactant_SMILES']   = can_r
    df.loc[:, 'Reactant_mapped']   = map_r
    df.loc[:, 'Product_SMILES']    = can_p
    df.loc[:, 'Product_mapped']    = map_p
    df.loc[:, 'Symmetry']          = symmetry
    df.loc[:, 'Symmetry_groups']   = groups
    df.loc[:, 'Group_idxes']       = gr_idxs
    df.loc[:, 'Unique symmetry']   = u_sym
    df.loc[:, 'Mapped_rxn']        = map_rxn
    df.loc[:, 'Mapped_reactivity'] = map_reac
    df.loc[:, 'idx_to_rxn_map']    = idx_rxn_
    df.loc[:, 'idx_to_selectivity']= idx_sel

    df.to_csv(f"{base_cwd}/data/{rxn_folder}/numbered_reaction.csv") # used for adding dois
    print("First part of the cleaning completed")
    
    # make a single dictionnary per reaction:
    selectivities_f = {}
    count = 0
    for idx in tqdm(df.rxn_ID.unique()):
        df_red  = df[df.loc[:, 'rxn_ID'] == idx]
        idx_rxn = df_red.index.to_list()
        df_red.reset_index(inplace=True)
        # print(df_red.Reactant_SMILES.unique()[0])
        # Case where there is only one reaction (just symmetry reduction)
        if len(df_red) == 1:
            sel       = df_red.loc[0, 'idx_to_selectivity']
            group_idx = df_red.loc[0, 'Group_idxes']
            red_sel = {}
            for key, value in group_idx.items():
                red_sel[min(value)] = np.mean([sel[v] for v in value])
        
        # case in which there are at least two products
        else:
            red_sels = []
            for idx_ in df_red.index:
                sel       = df_red.loc[idx_, 'idx_to_selectivity']
                group_idx = df_red.loc[idx_, 'Group_idxes']
                red_sel_  = {}
                for key, value in group_idx.items():
                    red_sel_[min(value)] = np.mean([sel[v] for v in value])
                red_sels.append(red_sel_)
            red_sel = {}
            for key in red_sels[0].keys():
                try:
                    red_sel[key] = np.sum([red[key] for red in red_sels])
                except:
                    raise ValueError(f"Error in the sum of the selectivities for reaction {idx}")
        
        # normalize red_sel so that the sum equals 100
        
        sum_sel = np.sum(list(red_sel.values())) 
        if sum_sel == 0:
            count += 1

        for key, value in red_sel.items():
            if sum_sel == 0:
                red_sel[key] = 0
            else:
                red_sel[key] = 100*value/sum_sel
            
        for idx_ in idx_rxn:
            selectivities_f[idx_] = red_sel

    selectivities_f_lst = []
    for i in range(len(df)):
        selectivities_f_lst.append(selectivities_f[i])

    print(str(count) + " molecules with no selectivity.")  
    df.loc[:, 'Selectivity_Reduced'] = selectivities_f_lst
    
    df_ = df[['Reactant_SMILES', 'Selectivity_Reduced']]
    df_.reset_index(inplace=True, drop=True)
    df_.drop_duplicates(subset='Reactant_SMILES', keep='first', inplace=True)
    outpath = f"{base_cwd}/data/{rxn_folder}/numbered_reaction_1.csv"
    df_.to_csv(outpath)
    return outpath


def group_symmetric_atoms(smiles):
    """
    input : reactant Canonical SMILES
    output:
        mol, Chem.Mol() object annotated with the symmetry group atoms are belonging to
        idx_to_group, dict() with keys being atom idx in the CanonicalSMILES and values beig the label of the group they belong to.
    """
    
    smiles = Chem.CanonSmiles(smiles)
    mol    = Chem.MolFromSmiles(smiles)
    Chem.RemoveStereochemistry(mol)
    groups = Chem.CanonicalRankAtoms(mol, breakTies=False)
    
    idx_to_group = {}
    
    for at in mol.GetAtoms():
        at.SetProp('atomNote', f"{groups[at.GetIdx()]}")  
        if at.GetSymbol() == 'C':
            idx_to_group.update({at.GetIdx(): groups[at.GetIdx()]})

    return mol, idx_to_group


def is_mol_symmetric(smiles):
    """
    input : reactant Canonical SMILES
    output:
        boolean, True if the carbon squelettom has equivalent carbons, False if not
    """
    smiles = Chem.CanonSmiles(smiles)
    mol = Chem.MolFromSmiles(smiles)
    # remove stereochemistry: helps find symmetries...
    Chem.RemoveStereochemistry(mol)
    
    groups = list(Chem.CanonicalRankAtoms(mol, breakTies=False))

    if len(groups) - len(set(groups)) > 0:
        return True
    else:
        return False

def reactive_carbon(reactant, product, product_atom='O'):
    """
    Input (type: product - str, reactant - str): Takes in the SMILES of product and reactant
    !!! WARNING --> Only takes in SMILES that has been mapped <-- !!!
    Output: returns an array where the most reactive carbon is 1, the rest are 0 
    """

    reactant = Chem.MolFromSmiles(reactant)
    reactant = Chem.AddHs(reactant)
    product  = Chem.MolFromSmiles(product)
    product  = Chem.AddHs(product)
    o_in_react = []
    h_in_react = []

    # iterate over reactant atoms
    for atom in reactant.GetAtoms():
        if atom.GetAtomicNum() == 6:
            properties     = atom.GetPropsAsDict()
            list_neighbors = atom.GetNeighbors()
            list_s = [at.GetSymbol() for at in list_neighbors]
            if product_atom in list_s:
                o_in_react.append(properties["molAtomMapNumber"])  
                h_in_react.append(list_s.count('H'))
    C_mapping = {}
    
    for atom in product.GetAtoms():
        if atom.GetAtomicNum() == 6:
            properties = atom.GetPropsAsDict()
            list_neighbors = atom.GetNeighbors()
            list_s = [at.GetSymbol() for at in list_neighbors]
            # code to store atom index with O in product
            atom_idx = properties["molAtomMapNumber"]
            # case of the CH_n oxidation:
            if product_atom in list_s and atom_idx not in o_in_react:
                C_mapping.update({atom_idx: True})
            # case of the C(OH)H_-1 oxidations:
            elif product_atom in list_s and list_s.count('H') < h_in_react[o_in_react.index(atom_idx)]:
                C_mapping.update({atom_idx: True})
            # else
            else:
                C_mapping.update({atom_idx: False})
    
    return C_mapping


def atom_idx_to_rxn_map(r, r_m, draw=False):
    """
    r   : str() reactant Canonical SMILES
    r_m : str() reactant Mapped SMILES
    draw: bool True if you want to check the mapping
    """

    m_m = Chem.MolFromSmiles(r_m)
    m_r = Chem.MolFromSmiles(r)
    for at in m_m.GetAtoms():
        if at.GetSymbol() == 'C':
            at.SetProp('atomNote', str(at.GetAtomMapNum()))
            at.SetAtomMapNum(0)
        else:
            at.SetAtomMapNum(0)
            
    at_idx    = []
    at_mapNum = []

    r_m_      = Chem.MolToSmiles(m_m)
    order     = m_m.GetPropsAsDict(True,True)["_smilesAtomOutputOrder"]
    m_canonic = Chem.RenumberAtoms(m_m, order)
    
    for at in m_canonic.GetAtoms():
        if at.GetSymbol() == 'C':
            at_idx.append(at.GetIdx())
            at_mapNum.append(int(at.GetProp('atomNote')))
    
    mapping = dict(zip(at_idx, at_mapNum))

    if draw:
        for at in m_r.GetAtoms():
            at.SetAtomMapNum(at.GetIdx())
        img = Draw.MolsToGridImage([m_r, m_m], subImgSize=(500, 500))
        display(img)

    return mapping

def drop_symmetric_carbons(sym_group):
    """
    Input:    sym_group, dict() 
        keys: atom idx 
        values: idx of the group they belong to
    Output:  unique_dict, dict() 
        keys: atom idx of reduced number of atoms
        values: number of atoms per group 
    """
    unique_dict = {}
    seen_values = set()

    for key, value in sym_group.items():
        if value not in seen_values:
            unique_dict[key] = list(sym_group.values()).count(value)
            seen_values.add(value)
     
    return unique_dict

def get_idx_per_groups(sym_group):
    """
    Input:    sym_group, dict() 
        keys: atom idx 
        values: idx of the group they belong to
    Output:  group_idx, dict() 
        keys: group idx
        values: list() of the atoms in this group 
    """

    group_idx   = {}
    seen_values = set()

    for key, value in sym_group.items():
        if value not in seen_values:
            group_idx[value] = [key]
            seen_values.add(value)
        else:
            l = list(group_idx[value])
            l.append(key)
            group_idx[value] = l
    
    return group_idx

def get_selectivity(map_reac, idx_to_rxn, sel):
    """
    """
    # map at_idx to the binary reactivity
    idx_to_reactivity = {}
    for key, value in idx_to_rxn.items():
        idx_to_reactivity.update({key: map_reac[value]})
    
    # map at_idx to the selectivity
    idx_to_selectivity = {}
    for key, value in idx_to_reactivity.items():
        if value:
            idx_to_selectivity.update({key: sel})
        else:
            idx_to_selectivity.update({key: 0.0})
    
    return idx_to_selectivity
    

def add_dois_to_df(df, rxn_folder="reaction_data",dois_to_start_with=['10.1021/ja00199a039', '10.1021/jo9604189']):
    """
    Retunrs a dataframe with the DOI of the reactant. Some selection had to be made in case of multiple DOI for the same reactant.
    Might want to improve that by considering reactions rather tha reactants...
    input:
        df: dataframe with the reaction data
        dois_to_start_with: list of dois to start with as an initial training set
    output: 
        df: dataframe with the reaction data and the DOI of the reaction
    """
    print(f"Root: {root}")
    new_root = root.split('regio_dataset_design')[0]
    print(f"New Root: {new_root}")
    path = f"{new_root}regio_dataset_design/data/{rxn_folder}/numbered_reaction.csv"
    print(f"Path: {path}")
    dois = []
    
    df_doi = pd.read_csv(path)
    for r in df['Reactant_SMILES']:
        doi  = df_doi.loc[df_doi['Reactant_SMILES'] == r, 'DOI'].unique()
        if len(doi) == 1:
            dois.append(doi[0])
        elif dois_to_start_with[0] in doi:
            dois.append(dois_to_start_with[0])
            #print(r, doi)
        elif dois_to_start_with[1] in doi:
            dois.append(dois_to_start_with[1])
            #print(r, doi)
        elif '10.1002/047084289X.rm267.pub3' in doi:
            doi = list(doi)
            doi.pop(doi.index('10.1002/047084289X.rm267.pub3'))
            if len(doi) == 1:
                dois.append(doi[0])
            #else:
            #    print(r,doi)
        else:
            dois.append(doi[0])

    df['DOI'] = dois
    return df
