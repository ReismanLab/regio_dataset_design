import json
import os
import subprocess
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolfiles
from rdkit.Chem.rdmolops import GetFormalCharge
from morfeus import Sterimol, BiteAngle, BuriedVolume, ConeAngle, read_xyz, xtb, Pyramidalization, XTB
import dbstep.Dbstep as db
import ast
import torch
import random

try:
    base_cwd = os.getcwd().split('regio_dataset_design')[0]
    base_cwd = f"{base_cwd}/regio_dataset_design"
except:
    raise ValueError("You are not in the right directory, need to be in the notebooks directory or subdirectory of it.")

model = torch.jit.load(f"{base_cwd}/data/descriptors/aimnet2/models/aimnet2_wb97m-d3_ens.jpt", map_location="cpu")

# this descriptors give descriptors for all unique C-H bonds in a reactant

### the function below take a Canonical SMILES as an input and returns a dictionnary of descriptors for each Carbon center bond. Carbons are identified by their idx in the Canonical SMILES of the reactant.

def Gasteiger(smiles, print_=False, write=True, df_json=None):
    """
    input : Canonical SMILES
    output: dictionnary of descriptors for each C-H bond
    
    The code first tries to retrieve the descriptors from a file where they have already been computed, if the descriptors have not been computed it will compute them from scratch and update the file with the descriptors.
    """
    
    if df_json is not None:
        df = df_json
    
    if write:
        f = open(f"{base_cwd}/data/descriptors/smiles_descriptors/reactions.json")
        df = json.load(f)
        f.close()
    
    # TO DO: add something to remove AtomMapNum if there is one
    
    smiles = Chem.CanonSmiles(smiles)
    
    try:
        ds = df[smiles]['gasteiger']
        ds_corrected = {}
        for key, values in ds.items():
            ds_corrected.update({int(key): values})
        return ds_corrected # something to change here because ast do not work...
        #print("data retrieved from computed file")
        
    except:
        if print_:
            print(f"Can't load the descriptors: Computing Gasteiger descriptors for {smiles}")
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.ComputeGasteigerCharges(mol)
        desc = {}
        
        for C in mol.GetAtoms():
            if C.GetAtomicNum() == 6:
                charge_C = round(C.GetPropsAsDict()["_GasteigerCharge"], 5) 
                charge_H = []
                for H in C.GetNeighbors():
                    if H.GetAtomicNum() == 1:
                        charge_H.append(round(H.GetPropsAsDict()["_GasteigerCharge"], 5))
                try:        
                    desc.update({C.GetIdx() : {'gas_charge_C'      : charge_C,
                                               'gas_charge_H_mean' : np.mean(charge_H),
                                               'gas_charge_H_max'  : max(charge_H),
                                               'gas_charge_H_min'  : min(charge_H)}})
                except:
                    #print("No hydrogen on this carbon: no descriptors attributed")
                    pass
        
        if write:
            if smiles in df.keys():                
                df[smiles].update({'gasteiger': desc})
            else:
                df.update({smiles: {}})
                df[smiles].update({'gasteiger': desc})
        
            with open(f"{base_cwd}/data/descriptors/smiles_descriptors/reactions.json", "w") as f:
                json.dump(df, f, sort_keys=True, indent=1)
            
        return desc
    
def xtb_CH(smiles, print_=False, write=True, df_json=None):
    """
    input :   smiles: Canonical SMILES
    output:   desc: dictionnary of descriptors for each C-H bond
    write:
    recompute: 
    The code first tries to retrieve the descriptors from a file where they have already been computed, if the descriptors have not been computed it will compute them from scratch and update the file with the descriptors.
    """
    
    smiles = Chem.CanonSmiles(smiles)
    
    if df_json is not None:
        df = df_json

    if write:
        f  = open(f"{base_cwd}/data/descriptors/smiles_descriptors/xtb.json")
        df = json.load(f)
        f.close()
    
    xtb_f_name = str(random.random()).split('.')[1] 
    
    try:
        ds = df[smiles]['xtb_CH']
        ds_corrected = {}
        for key, values in ds.items():
            ds_corrected.update({int(key): values})
        if print_:
            print(f"{smiles} already computed")
        return ds_corrected 
        
    except:
        if print_:   
            print(f"Can't load the descriptors: Computing XTB-C descriptors for {smiles}")
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        if AllChem.EmbedMolecule(mol) == -1:
            print(f"Embedding failed - XTB not running - {smiles}")
            return None
        else:
            AllChem.UFFOptimizeMolecule(mol)
        cwd = os.getcwd()
        os.chdir(f"{base_cwd}/utils/xtb_utils")
        os.mkdir(f"{xtb_f_name}")
        os.chdir(f"{xtb_f_name}")
        rdmolfiles.MolToXYZFile(mol, 'temp.xyz')

        subprocess.run('xtb temp.xyz --opt extreme --gnf=2 --json', shell=True, check=True, capture_output=True)
        print(f"XTB: {smiles} done")
        subprocess.run('mv xtbopt.xyz temp.xyz', shell=True, check=True, capture_output=True)
        elements, coordinates = read_xyz("temp.xyz")
        os.chdir(cwd)
        xtb = XTB(elements, coordinates)

        desc = {}
        
        for C in mol.GetAtoms():
            if C.GetAtomicNum() == 6:
                C_idx = C.GetIdx()
                C_idx_xtb = C_idx + 1
                charge_C                 = round(xtb.get_charges()[C_idx_xtb],5)
                electrophilicity_C       = round(xtb.get_fukui("electrophilicity")[C_idx_xtb],5)
                Nucleophilicity_C        = round(xtb.get_fukui("nucleophilicity")[C_idx_xtb],5)
                Radical_C                = round(xtb.get_fukui("radical")[C_idx_xtb],5)
                dual_C                   = round(xtb.get_fukui("dual")[C_idx_xtb],5) 
                Local_Nucleophilicity_C  = round(xtb.get_fukui("local_nucleophilicity")[C_idx_xtb],5) 
                Local_Electrophilicity_C = round(xtb.get_fukui("local_electrophilicity")[C_idx_xtb],5)
                Buried_Volume_C          = round(BuriedVolume(elements, coordinates, C_idx_xtb).fraction_buried_volume,5)
                Pyramidalization_C       = round(Pyramidalization(coordinates, C_idx_xtb).P,5)
                
                charge_H           = []
                electrophilicity_H = []
                Nucleophilicity_H  = []
                Radical_H          = []
                dual_H             = []
                Local_Nucleophilicity_H  = []
                Local_Electrophilicity_H = []
                Buried_Volume_H          = []
                Pyramidalization_H       = []
                for H in C.GetNeighbors():
                    if H.GetAtomicNum() == 1:
                        H_idx = H.GetIdx()
                        H_idx_xtb = H_idx + 1
                        charge_H.append(xtb.get_charges()[H_idx_xtb])
                        electrophilicity_H.append(xtb.get_fukui("electrophilicity")[H_idx_xtb])
                        Nucleophilicity_H.append(xtb.get_fukui("nucleophilicity")[H_idx_xtb])
                        Radical_H.append(xtb.get_fukui("radical")[H_idx_xtb])
                        dual_H.append(xtb.get_fukui("dual")[H_idx_xtb])
                        Local_Nucleophilicity_H.append(xtb.get_fukui("local_nucleophilicity")[H_idx_xtb])
                        Local_Electrophilicity_H.append(xtb.get_fukui("local_electrophilicity")[H_idx_xtb])
                        Buried_Volume_H.append(BuriedVolume(elements, coordinates, H_idx_xtb).fraction_buried_volume)
                        Pyramidalization_H.append(Pyramidalization(coordinates, H_idx_xtb).P)
                        
                        
                try:
                    desc.update({ C_idx : {
                                'charge_C'                     : round(charge_C,6),
                                'electrophilicity_C'           : round(electrophilicity_C,6),
                                'Nucleophilicity_C'            : round(Nucleophilicity_C,6),
                                'Radical_C'                    : round(Radical_C,6),
                                'dual_C'                       : round(dual_C,6),
                                'Local_Nucleophilicity_C'      : round(Local_Nucleophilicity_C,6),
                                'Local_Electrophilicity_C'     : round(Local_Electrophilicity_C,6),
                                'Buried_Volume_C'              : round(Buried_Volume_C,6),
                                'Pyramidalization_C'           : round(Pyramidalization_C,6),
                                'charge_H_mean'                : round(np.mean(charge_H),6),
                                'charge_H_max'                 : round(max(charge_H),6),
                                'charge_H_min'                 : round(min(charge_H),6),
                                'electrophilicity_H_mean'      : round(np.mean(electrophilicity_H),6),
                                'electrophilicity_H_max'       : round(max(electrophilicity_H),6),
                                'electrophilicity_H_min'       : round(min(electrophilicity_H),6),
                                'Nucleophilicity_H_mean'       : round(np.mean(Nucleophilicity_H),6),
                                'Nucleophilicity_H_max'        : round(max(Nucleophilicity_H),6),
                                'Nucleophilicity_H_min'        : round(min(Nucleophilicity_H),6),
                                'Radical_H_mean'               : round(np.mean(Radical_H),6),
                                'Radical_H_max'                : round(max(Radical_H),6),
                                'Radical_H_min'                : round(min(Radical_H),6),   
                                'dual_H_mean'                  : round(np.mean(dual_H),6),
                                'dual_H_max'                   : round(max(dual_H),6),
                                'dual_H_min'                   : round(min(dual_H),6),
                                'Local_Nucleophilicity_H_mean' : round(np.mean(Local_Nucleophilicity_H),6),
                                'Local_Nucleophilicity_H_max'  : round(max(Local_Nucleophilicity_H),6),
                                'Local_Nucleophilicity_H_min'  : round(min(Local_Nucleophilicity_H),6), 
                                'Local_Electrophilicity_H_mean': round(np.mean(Local_Electrophilicity_H),6),
                                'Local_Electrophilicity_H_max' : round(max(Local_Electrophilicity_H),6),
                                'Local_Electrophilicity_H_min' : round(min(Local_Electrophilicity_H),6), 
                                'Buried_Volume_H_mean'         : round(np.mean(Buried_Volume_H),6),
                                'Buried_Volume_H_max'          : round(max(Buried_Volume_H),6),
                                'Buried_Volume_H_min'          : round(min(Buried_Volume_H),6),
                                'Pyramidalization_H_mean'      : round(np.mean(Pyramidalization_H),6),
                                'Pyramidalization_H_max'       : round(max(Pyramidalization_H),6),
                                'Pyramidalization_H_min'       : round(min(Pyramidalization_H),6)
                                             }
                                    })
                except:
                    pass
        if write: 
            if smiles in df.keys():                
                df[smiles].update({'xtb_CH': desc})
            else:
                df.update({smiles: {}})
                df[smiles].update({'xtb_CH': desc})
        
            with open(f"{base_cwd}/data/descriptors/smiles_descriptors/xtb.json", "w") as f:
                json.dump(df, f, sort_keys=True, indent=1)
              
        return desc
    
def dbstep_CH(smiles, print_=False, write=True, df_json=None):
    """
    input : Canonical SMILES
    output: dictionary of descriptors for each C-H bond
    The code first tries to retrieve the descriptors from a file where they have already been computed, if the descriptors have not been computed it will compute them from scratch and update the file with the descriptors.
    """
    
    smiles = Chem.CanonSmiles(smiles)
    
    db_step_desc_names = ['L_max', 'V_occ_max', 
                          'Bmin_2.0_max', 'Bmin_2.5_max', 'Bmin_3.0_max', 'Bmin_3.5_max', 'Bmin_4.0_max', 'Bmin_4.5_max',
                          'Bmax_2.0_max', 'Bmax_2.5_max', 'Bmax_3.0_max', 'Bmax_3.5_max', 'Bmax_4.0_max', 'Bmax_4.5_max',
                          'V_bur_2.0_max', 'V_bur_2.5_max', 'V_bur_3.0_max', 'V_bur_3.5_max', 'V_bur_4.0_max', 'V_bur_4.5_max',
                          'S_bur_2.0_max', 'S_bur_2.5_max', 'S_bur_3.0_max', 'S_bur_3.5_max', 'S_bur_4.0_max', 'S_bur_4.5_max',
                          'L_min', 'V_occ_min', 
                          'Bmin_2.0_min', 'Bmin_2.5_min', 'Bmin_3.0_min', 'Bmin_3.5_min', 'Bmin_4.0_min', 'Bmin_4.5_min',
                          'Bmax_2.0_min', 'Bmax_2.5_min', 'Bmax_3.0_min', 'Bmax_3.5_min', 'Bmax_4.0_min', 'Bmax_4.5_min',
                          'V_bur_2.0_min', 'V_bur_2.5_min', 'V_bur_3.0_min', 'V_bur_3.5_min', 'V_bur_4.0_min', 'V_bur_4.5_min',
                          'S_bur_2.0_min', 'S_bur_2.5_min', 'S_bur_3.0_min', 'S_bur_3.5_min', 'S_bur_4.0_min', 'S_bur_4.5_min',
                          'L_avg', 'V_occ_avg', 
                          'Bmin_2.0_avg', 'Bmin_2.5_avg', 'Bmin_3.0_avg', 'Bmin_3.5_avg', 'Bmin_4.0_avg', 'Bmin_4.5_avg',
                          'Bmax_2.0_avg', 'Bmax_2.5_avg', 'Bmax_3.0_avg', 'Bmax_3.5_avg', 'Bmax_4.0_avg', 'Bmax_4.5_avg',
                          'V_bur_2.0_avg', 'V_bur_2.5_avg', 'V_bur_3.0_avg', 'V_bur_3.5_avg', 'V_bur_4.0_avg', 'V_bur_4.5_avg',
                          'S_bur_2.0_avg', 'S_bur_2.5_avg', 'S_bur_3.0_avg', 'S_bur_3.5_avg', 'S_bur_4.0_avg', 'S_bur_4.5_avg',]

    if df_json is not None:
        df = df_json

    if write:
        f = open(f"{base_cwd}/data/descriptors/smiles_descriptors/dbstep.json")
        df = json.load(f)
        f.close()

    xtb_f_name = str(random.random()).split('.')[1]

    try:
        ds = df[smiles]['dbstep_CH']
        if ds == {}:
            print("Descriptor Empty...")
            raise ValueError
        ds_corrected = {}
        for key, values in ds.items():
            ds_corrected.update({int(key): values})
        return ds_corrected 
        

    except:
        if print_:
            print(f"Can't load the descriptors: Computing DBSTEP descriptors for {smiles}")
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        cwd = os.getcwd()
        os.chdir(f"{base_cwd}/utils/xtb_utils")
        os.mkdir(f"{xtb_f_name}")
        os.chdir(f"{xtb_f_name}")
        AllChem.rdmolfiles.MolToXYZFile(mol, "temp.xyz")
        if print_:
            print("XTB optimization", flush= True)
        subprocess.run('xtb temp.xyz --opt extreme --gnf=2 --json', shell=True, check=True, capture_output=True)
        subprocess.run('mv xtbopt.xyz temp.xyz', shell=True, check=True, capture_output=True)
        os.chdir(cwd)

        desc = {}

        for C in mol.GetAtoms():
            if C.GetAtomicNum() == 6 and 'H' in [atom.GetSymbol() for atom in C.GetNeighbors()]:
                dbstep_desc = []
                C_idx = C.GetIdx()
                for H in C.GetNeighbors():
                    if H.GetAtomicNum() == 1:
                        H_idx        = H.GetIdx()
                        H_idx_dbstep = H_idx + 1

                        mol = db.dbstep(f"{base_cwd}/utils/xtb_utils/{xtb_f_name}/temp.xyz", 
                                        atom1 = C_idx+1, 
                                        atom2 = H_idx_dbstep,
                                        commandline = True, 
                                        verbose     = False, 
                                        sterimol    = True, 
                                        volume      = True, 
                                        scan        = '2.0:4.5:0.5', 
                                        measure     = 'classic')
                        
                        dbstep_desc.append([mol.L] + [mol.occ_vol] + mol.Bmin + mol.Bmax + mol.bur_vol + mol.bur_shell)

                dbstep_desc = np.array(dbstep_desc)        
                        
                max_dbstep_desc = [max(dbstep_desc[:, i]) for i in range(len(dbstep_desc[0]))]
                min_dbstep_desc = [min(dbstep_desc[:, i]) for i in range(len(dbstep_desc[0]))]
                avg_dbstep_desc = [np.mean(dbstep_desc[:, i]) for i in range(len(dbstep_desc[0]))]

                dbstep_desc = max_dbstep_desc + min_dbstep_desc + avg_dbstep_desc
                dbstep_desc = [round(d,6) for d in dbstep_desc]

                try:
                    desc.update({C_idx : dict(zip(db_step_desc_names, dbstep_desc))})
                except:
                    print(f"could not update the descriptors for {C_idx}")
                    pass
        
        if write:
            if smiles in df.keys():                
                df[smiles].update({'dbstep_CH': desc})
            else:
                df.update({smiles: {}})
                df[smiles].update({'dbstep_CH': desc})
        
            with open(f"{base_cwd}/data/descriptors/smiles_descriptors/dbstep.json", "w") as f:
                json.dump(df, f, sort_keys=True, indent=1)
              
        return desc
    
def dft_CH(smiles, print_=False):
    """
    input : Canonical SMILES
    output: dictionnary of descriptors for each C-H bond
    
    The code first tries to retrieve the descriptors from a file where they have already been computed, if the descriptors have not been computed it will compute them from scratch and update the file with the descriptors.
    """

    smiles = Chem.CanonSmiles(smiles)
    
    f  = open(f"{base_cwd}/data/descriptors/smiles_descriptors/reactions.json")
    df = json.load(f)
    f.close()
    
    try:
        ds = df[smiles]['dft_CH']

        if ds == {}:
            print('Desc empty')
            raise ValueError
        
        ds_corrected = {}

        for key, values in ds.items():
            ds_corrected.update({int(key): values})

        return ds_corrected 
    
    except:
        if print_:
            print(f"Can't load the descriptors: retrieving DFT-C descriptors for {smiles}")
        mol = Chem.MolFromSmiles(smiles)    
        mol = Chem.AddHs(mol)   
        desc = {}   
        
        faut  = open(f"{base_cwd}/data/descriptors/smiles_descriptors/autoqchem-descriptors.json")
        dfaut = json.load(faut)
        faut.close()
        
        for at in mol.GetAtoms():
            if at.GetSymbol() == 'C':
                if 'H' in [n.GetSymbol() for n in at.GetNeighbors()]:
                    C_idx = at.GetIdx()    
                    try:
                        df_smi = dfaut[smiles]
                    except:
                        #print(f"{smiles} is not in the autoqchem json file")
                        return None
                    
                    # C descriptors
                    desc_C = df_smi[str(C_idx)]

                    # Adding H descriptors
                    desc_H = []
                    for H in at.GetNeighbors():
                        if H.GetAtomicNum() == 1:
                            H_idx = H.GetIdx()
                            desc_H.append(list(df_smi[str(H_idx)].values()))
                    
                    desc_H     = np.array(desc_H)
                    max_H_desc = [max(desc_H[:, i]) for i in range(len(desc_H[0]))]
                    min_H_desc = [min(desc_H[:, i]) for i in range(len(desc_H[0]))]
                    avg_H_desc = [np.mean(desc_H[:, i]) for i in range(len(desc_H[0]))]

                    desc_H = max_H_desc + min_H_desc + avg_H_desc
                    keys_H = [ k + "_H_max" for k in df_smi[str(H_idx)].keys()] + [ k + "_H_min" for k in df_smi[str(H_idx)].keys()] + [ k + "_H_avg" for k in df_smi[str(H_idx)].keys()]

                    desc_H = dict(zip(keys_H, desc_H))  

                    # Merging H and C descriptors

                    desc_C.update(desc_H)

                    for key in ['X', 'Y', 'Z', 'X_H_max', 'Y_H_max', 'Z_H_max', 'X_H_min', 'Y_H_min', 'Z_H_min', 'X_H_avg', 'Y_H_avg', 'Z_H_avg']:
                        desc_C.pop(key)
                    
                    desc.update({C_idx : desc_C})

        if smiles in df.keys():
            df[smiles].update({'dft_CH': desc})
        else:
            df.update({smiles: {'dft_CH': desc}})

        with open(f"{base_cwd}/data/descriptors/smiles_descriptors/reactions.json", "w") as f:
            json.dump(df, f, sort_keys=True, indent=1)

        return desc

def env1(smiles, print_=False, write=True, df_json=None):
    """
    input : Canonical SMILES
    output: dictionnary of descriptors for each C-H bond
    
    The code first tries to retrieve the descriptors from a file where they have already been computed, if the descriptors have not been computed it will compute them from scratch and update the file with the descriptors.
    """
    smiles = Chem.CanonSmiles(smiles)
    
    if df_json is not None:
        df = df_json

    if write:
        f  = open(f"{base_cwd}/data/descriptors/smiles_descriptors/C_env.json")
        df = json.load(f)
        f.close()
    
    try:
        ds = df[smiles]['env1']  
        ds_corrected = {}

        for key, values in ds.items():
            ds_corrected.update({int(key): values})

        return ds_corrected 
    
    except:
        if print_:
            print(f"Can't load the descriptors: computing OHE environment descriptors {smiles}")
        mol = Chem.MolFromSmiles(smiles)    
        mol = Chem.AddHs(mol)   
        desc = {}   
        
        for at in mol.GetAtoms():
            if at.GetSymbol() == 'C':
                neighbors        = at.GetNeighbors()
                neighbors_symbol = [n.GetSymbol() for n in neighbors]
                if 'H' in neighbors_symbol:
                    C_idx  = at.GetIdx()    

                    # get number of neighbors of elements H, C, N, O
                    num_H  = neighbors_symbol.count('H')
                    num_C  = neighbors_symbol.count('C')
                    num_N  = neighbors_symbol.count('N')
                    num_O  = neighbors_symbol.count('O')

                    # get carbon hybridization
                    if str(at.GetHybridization()) == 'SP3':
                        C_spX = 0
                    else:
                        C_spX = 1  

                    # get number of neighbors of carbon with hybridization SP2, SP3
                    n_Csp2 = 0
                    n_Csp3 = 0 
                    n_CAro = 0
                    for n in neighbors:
                        if n.GetSymbol() == 'C':
                            if str(n.GetHybridization()) == 'SP2':
                                n_Csp2 += 1
                                if n.GetIsAromatic():
                                    n_CAro += 1
                            elif str(n.GetHybridization()) == 'SP3':  
                                n_Csp3 += 1

                    # list of descriptors

                    list_desc = ['num_H', 'num_C', 'num_N', 'num_O', 'C_spX', 'n_Csp2', 'n_Csp3', 'n_CAro']

                    desc_C = dict(zip(list_desc, [num_H, num_C, num_N, num_O, C_spX, n_Csp2, n_Csp3, n_CAro]))

                    desc.update({C_idx : desc_C})
        
        if write:
            if smiles in df.keys():
                df[smiles].update({'env1': desc})
            else:
                df.update({smiles: {'env1': desc}})

            with open(f"{base_cwd}/data/descriptors/smiles_descriptors/C_env.json", "w") as f:
                json.dump(df, f, sort_keys=True, indent=1)

        return desc

def env2(smiles, print_=False, write=True, df_json=None):
    """
    input : Canonical SMILES
    output: dictionnary of descriptors for each C-H bond
    
    The code first tries to retrieve the descriptors from a file where they have already been computed, if the descriptors have not been computed it will compute them from scratch and update the file with the descriptors.
   
    This descriptors are:
    - L, Bmin, Bmax, Vbur, Sbur for the C-H bonds, both directions
    - L, Bmin, Bmax, Vbur, Sbur for the C-C bonds, both directions
    - charges for C and H
    - charges for neighbors of C if they are carbons

    Theses descriptors are intendended to be used in comnbination with the descriptors from env1.
    """

    smiles = Chem.CanonSmiles(smiles)
    
    if df_json is not None:
        df = df_json

    if write:
        f  = open(f"{base_cwd}/data/descriptors/smiles_descriptors/C_env.json")
        df = json.load(f)
        f.close()

    xtb_f_name = str(random.random()).split('.')[1]

    try:
        ds = df[smiles]['env2']
        if ds == {}:
            print('Desc empty')
            raise ValueError
        
        ds_corrected = {}

        for key, values in ds.items():
            ds_corrected.update({int(key): values})

        return ds_corrected 
    
    except:
        if print_:
            print(f"Can't load the descriptors: Computing DBSTEP + charges descriptors for {smiles}")
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        cwd = os.getcwd()
        os.chdir(f"{base_cwd}/utils/xtb_utils")
        os.mkdir(f"{xtb_f_name}")
        os.chdir(f"{xtb_f_name}")
        AllChem.rdmolfiles.MolToXYZFile(mol, "temp.xyz")
        subprocess.run('xtb temp.xyz --opt extreme --gnf=2 --json', shell=True, check=True, capture_output=True)
        subprocess.run('mv xtbopt.xyz temp.xyz', shell=True, check=True, capture_output=True)
        os.chdir(cwd)

        coord   = torch.as_tensor([mol.GetConformer().GetPositions()])               # for mol in mols])
        charge  = torch.as_tensor([GetFormalCharge(mol)])                            # for mol in mols])
        numbers = torch.as_tensor([[atom.GetAtomicNum() for atom in mol.GetAtoms()]])# for mol in mols])

        # predict charges
        _in  = dict(coord=coord, numbers=numbers, charge=charge)
        _out = model(_in)

        desc = {}

        for C in mol.GetAtoms():
            if C.GetAtomicNum() == 6 and 'H' in [atom.GetSymbol() for atom in C.GetNeighbors()]:
                dbstep_desc_CH = []
                dbstep_desc_CC = []
                C_idx = C.GetIdx()
                C_charge  = float(_out['charges'][0][C_idx])
                H_charges = []
                C_charges = [] # charges for C neighbors
                for H in C.GetNeighbors():
                    if H.GetAtomicNum() == 1:
                        H_idx        = H.GetIdx()
                        H_idx_dbstep = H_idx + 1

                        mol1 = db.dbstep(f"{base_cwd}/utils/xtb_utils/{xtb_f_name}/temp.xyz", 
                                        atom1 = C_idx+1, 
                                        atom2 = H_idx_dbstep,
                                        commandline = True, 
                                        verbose     = False, 
                                        sterimol    = True, 
                                        volume      = True,
                                        measure     = 'classic')
                        
                        mol2 = db.dbstep(f"{base_cwd}/utils/xtb_utils/{xtb_f_name}/temp.xyz", 
                                        atom1 = H_idx_dbstep, 
                                        atom2 = C_idx+1,
                                        commandline = True, 
                                        verbose     = False, 
                                        sterimol    = True, 
                                        volume      = True,
                                        measure     = 'classic')
                        
                        dbstep_desc_CH.append([mol1.L, mol1.occ_vol, mol1.Bmin, mol1.Bmax, mol1.bur_vol, mol1.bur_shell, 
                                               mol2.L, mol2.occ_vol, mol2.Bmin, mol2.Bmax, mol2.bur_vol, mol2.bur_shell])

                        # adding charges
                        H_charges.append(float(_out['charges'][0][H_idx]))
                    
                    else: #for all other bonded atoms
                        H_idx        = H.GetIdx()
                        H_idx_dbstep  = H_idx + 1

                        mol1 = db.dbstep(f"{base_cwd}/utils/xtb_utils/{xtb_f_name}/temp.xyz", 
                                        atom1 = C_idx+1, 
                                        atom2 = H_idx_dbstep,
                                        commandline = True, 
                                        verbose     = False, 
                                        sterimol    = True, 
                                        volume      = True,
                                        measure     = 'classic')
                        
                        mol2 = db.dbstep(f"{base_cwd}/utils/xtb_utils/{xtb_f_name}/temp.xyz", 
                                        atom1 = H_idx_dbstep, 
                                        atom2 = C_idx+1,
                                        commandline = True, 
                                        verbose     = False, 
                                        sterimol    = True, 
                                        volume      = True,
                                        measure     = 'classic')
                        
                        C_charges.append(float(_out['charges'][0][H_idx]))

                        dbstep_desc_CC.append([mol1.L, mol1.occ_vol, mol1.Bmin, mol1.Bmax, mol1.bur_vol, mol1.bur_shell, 
                                               mol2.L, mol2.occ_vol, mol2.Bmin, mol2.Bmax, mol2.bur_vol, mol2.bur_shell])     

                # C-H descriptors
                dbstep_desc = np.array(dbstep_desc_CH)        
                        
                max_dbstep_desc = [max(dbstep_desc[:, i]) for i in range(len(dbstep_desc[0]))]
                min_dbstep_desc = [min(dbstep_desc[:, i]) for i in range(len(dbstep_desc[0]))]
                avg_dbstep_desc = [np.mean(dbstep_desc[:, i]) for i in range(len(dbstep_desc[0]))]

                dbstep_desc_all = max_dbstep_desc + min_dbstep_desc + avg_dbstep_desc

                # C-C descriptors
                dbstep_desc = np.array(dbstep_desc_CC)        
                        
                max_dbstep_desc = [max(dbstep_desc[:, i]) for i in range(len(dbstep_desc[0]))]
                min_dbstep_desc = [min(dbstep_desc[:, i]) for i in range(len(dbstep_desc[0]))]
                avg_dbstep_desc = [np.mean(dbstep_desc[:, i]) for i in range(len(dbstep_desc[0]))]

                dbstep_desc_all =  dbstep_desc_all + max_dbstep_desc + min_dbstep_desc + avg_dbstep_desc + [max(C_charges), min(C_charges), np.mean(C_charges)] + [max(H_charges), min(H_charges), np.mean(H_charges)] + [C_charge]
                dbstep_desc_all =  [round(d,6) for d in dbstep_desc_all]

                db_step_desc_names = ['L_ch1_max', 'OccV_ch1_max', 'Bmin_ch1_max', 'Bmax_ch1_max', 'Vbur_ch1_max', 'Sbur_ch1_max',
                                      'L_ch2_max', 'OccV_ch2_max', 'Bmin_ch2_max', 'Bmax_ch2_max', 'Vbur_ch2_max', 'Sbur_ch2_max',
                                      'L_ch1_min', 'OccV_ch1_min', 'Bmin_ch1_min', 'Bmax_ch1_min', 'Vbur_ch1_min', 'Sbur_ch1_min',
                                      'L_ch2_min', 'OccV_ch2_min', 'Bmin_ch2_min', 'Bmax_ch2_min', 'Vbur_ch2_min', 'Sbur_ch2_min',
                                      'L_ch1_avg', 'OccV_ch1_avg', 'Bmin_ch1_avg', 'Bmax_ch1_avg', 'Vbur_ch1_avg', 'Sbur_ch1_avg',
                                      'L_ch2_avg', 'OccV_ch2_avg', 'Bmin_ch2_avg', 'Bmax_ch2_avg', 'Vbur_ch2_avg', 'Sbur_ch2_avg',
                                      'L_cc1_max', 'OccV_cc1_max', 'Bmin_cc1_max', 'Bmax_cc1_max', 'Vbur_cc1_max', 'Sbur_cc1_max',
                                      'L_cc2_max', 'OccV_cc2_max', 'Bmin_cc2_max', 'Bmax_cc2_max', 'Vbur_cc2_max', 'Sbur_cc2_max',
                                      'L_cc1_min', 'OccV_cc1_min', 'Bmin_cc1_min', 'Bmax_cc1_min', 'Vbur_cc1_min', 'Sbur_cc1_min',
                                      'L_cc2_min', 'OccV_cc2_min', 'Bmin_cc2_min', 'Bmax_cc2_min', 'Vbur_cc2_min', 'Sbur_cc2_min',
                                      'L_cc1_avg', 'OccV_cc1_avg', 'Bmin_cc1_avg', 'Bmax_cc1_avg', 'Vbur_cc1_avg', 'Sbur_cc1_avg',
                                      'L_cc2_avg', 'OccV_cc2_avg', 'Bmin_cc2_avg', 'Bmax_cc2_avg', 'Vbur_cc2_avg', 'Sbur_cc2_avg',
                                      'C_n_charge_max', 'C_n_charge_min', 'C_n_charge_avg', 'H_charge_max', 'H_charge_min', 'H_charge_avg', 'C_charge']

                try:
                    desc.update({str(C_idx) : dict(zip(db_step_desc_names, dbstep_desc_all))})
                except:
                    print(f"could not update the descriptors for {C_idx}")
                    pass
        if write:
            if smiles in df.keys():
                print(f"{smiles} already there")
                df[smiles].update({'env2': desc})
            else:
                df.update({smiles : {'env2': desc}})
        
        if write:
            with open(f"{base_cwd}/data/descriptors/smiles_descriptors/C_env.json", "w") as f:
                json.dump(df, f, sort_keys=True, indent=1)

        return desc    

def bde(smiles, print_=False):
    """
    input : Canonical SMILES
    output: dictionnary of descriptors for each C atom
    
    The code first tries to retrieve the descriptors from a file where they have already been computed, if the descriptors have not been computed it will compute them from scratch and update the file with the descriptors.
    """

    smiles = Chem.CanonSmiles(smiles)
    
    f  = open(f"{base_cwd}/data/descriptors/smiles_descriptors/bdes.json")
    df = json.load(f)
    f.close()
    
    try:
        ds = df[smiles]['pred_bdes']
        if ds == {}:
            print(f"Desc empty for smiles {smiles}: please update the bdes.json file")
            return None
        
        else:
            ds_corrected = {}
            for key, values in ds.items():
                ds_corrected.update({int(key): values})
            return ds_corrected 
    
    except:
        print(f"SMILES: {smiles} not featurized so please update the bdes.json file using the bde_predictions tools.")
        return None

def aimnet_embeddings(smiles, print_=False):
    """
    input : Canonical SMILES
    output: dictionnary of descriptors for each C atom
    
    The code first tries to retrieve the descriptors from a file where they have already been computed, if the descriptors have not been computed it will compute them from scratch and update the file with the descriptors.
    """

    smiles = Chem.CanonSmiles(smiles)
    
    f  = open(f"{base_cwd}/data/descriptors/smiles_descriptors/aimnet_embeddings.json")
    df = json.load(f)
    f.close()
    
    try:
        ds = df[smiles]#['a']
        if ds == {}:
            print(f"Desc empty for smiles {smiles}: please update the aimnet_embeddings.json file")
            return None
        
        else:
            ds_corrected = {}
            for key, values in ds.items():
                ds_corrected.update({int(key): values})
            return ds_corrected 
    
    except:
        print(f"SMILES: {smiles} not featurized so please update the aimnet_embeddings.json file using the bde_predictions tools.")
        return None

def rdkit_conf_Vbur(smiles, print_=False, write=True, df_json=None):
    """
    input : Canonical SMILES
    output: dictionnary of descriptors for each C-H bond
    
    The code first tries to retrieve the descriptors from a file where they have already been computed, if the descriptors have not been computed it will compute them from scratch and update the file with the descriptors.
    """
    
    smiles = Chem.CanonSmiles(smiles)
    
    if df_json is not None:
        df = df_json

    if write:
        f = open(f"{base_cwd}/data/descriptors/smiles_descriptors/vbur_rdkit.json")
        df = json.load(f)
        f.close()
    
    try:
        ds = df[smiles]['vbur_rdkit']
        ds_corrected = {}
        for key, values in ds.items():
            ds_corrected.update({int(key): values})
        if print_:
            print(f"{smiles} already computed")
        return ds_corrected 
        
    except:
        if print_:   
            print(f"Can't load the descriptors: Computing RDKIT-Vbur descriptors for {smiles}")
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        try:
            AllChem.EmbedMolecule(mol)
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            print(f"Could not embed or optimize {smiles}")
            return None
        
        cwd = os.getcwd()
        os.chdir(f"{base_cwd}/utils/xtb_utils")
        if print_:
            print(f"Vbur Morfeus - MFFOptSutructure: {smiles}")

        rdmolfiles.MolToXYZFile(mol, 'mff_temp.xyz')

        elements, coordinates = read_xyz("mff_temp.xyz")
        os.chdir(cwd)
        xtb = XTB(elements, coordinates)

        desc = {}
        
        for C in mol.GetAtoms():
            if C.GetAtomicNum() == 6:
                C_idx = C.GetIdx()
                C_idx_xtb = C_idx + 1
                Buried_Volume_C          = BuriedVolume(elements, coordinates, C_idx_xtb).fraction_buried_volume
                Pyramidalization_C       = Pyramidalization(coordinates, C_idx_xtb).P
                
                Buried_Volume_H          = []
                Pyramidalization_H       = []
                for H in C.GetNeighbors():
                    if H.GetAtomicNum() == 1:
                        H_idx = H.GetIdx()
                        H_idx_xtb = H_idx + 1
                        Buried_Volume_H.append(BuriedVolume(elements, coordinates, H_idx_xtb).fraction_buried_volume)
                        Pyramidalization_H.append(Pyramidalization(coordinates, H_idx_xtb).P)
                        
                        
                try:
                    desc.update({ C_idx : {
                                'Buried_Volume_C_MMFF'             : round(Buried_Volume_C,6),
                                'Pyramidalization_C_MFF'           : round(Pyramidalization_C,6),
                                'Buried_Volume_H_mean_MFF'         : round(np.mean(Buried_Volume_H),6),
                                'Buried_Volume_H_max_MFF'          : round(max(Buried_Volume_H),6),
                                'Buried_Volume_H_min_MFF'          : round(min(Buried_Volume_H),6),
                                'Pyramidalization_H_mean_MFF'      : round(np.mean(Pyramidalization_H),6),
                                'Pyramidalization_H_max_MFF'       : round(max(Pyramidalization_H),6),
                                'Pyramidalization_H_min_MFF'       : round(min(Pyramidalization_H),6)
                                             }
                                    })
                except:
                    pass

        if write:
            if smiles in df.keys():                
                df[smiles].update({'vbur_rdkit': desc})
            else:
                df.update({smiles: {}})
                df[smiles].update({'vbur_rdkit': desc})
        
            if print_:
                print(f"Saving {smiles} to vbur_rdkit.json")
            
            with open(f"{base_cwd}/data/descriptors/smiles_descriptors/vbur_rdkit.json", "w") as f:
                json.dump(df, f, sort_keys=True, indent=1)
              
        return desc          
        
