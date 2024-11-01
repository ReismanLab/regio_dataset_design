# this script protonates amines in the dataset_crude.xlsx file
# it also removes some entries that are not found in the data referenced like the intra molecular reactions

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Fragments  import fr_amide, fr_nitro, fr_alkyl_carbamate, fr_azide
from tqdm import tqdm
import os
import sys
sys.path.append('../utils')
from preprocessing import is_mol_symmetric, group_symmetric_atoms

df = pd.read_excel('dioxirane_reaction_data/dataset_crude.xlsx')

smiles_to_remove = ['O=C(CCC[C@]12C)[C@]2([H])CC[C@@H]1[C@@H](C)CCCC(C)C', # 11 ot found in the data referenced
                    '[H][C@@]1(CCCC[C@@]12C)CC[C@]3([H])[C@]2([H])CC[C@@]4([H])[C@@]3([H])CC[C@@H]4[C@H](C)CCCC(C)C', # 41 not found in the data referenced
                    ]
smiles_to_remove = [Chem.CanonSmiles(s) for s in smiles_to_remove]

smiles_to_swap   = {'[H][C@@]1(CCCC[C@@]1([C@@]2([H])CC[C@@]34C)C)CC[C@@]2([H])[C@]3([H])CC[C@H]4[C@@H](C)CCCC(C)C': '[H][C@@]1(CCCC[C@@]1([C@@]2([H])CC[C@@]34C)C)CC[C@@]2([H])[C@]3([H])CC[C@@H]4[C@H](C)CCCC(C)C'} # 51 sterochemistry was wrong
smiles_to_swap   = {Chem.CanonSmiles(k): Chem.CanonSmiles(v) for k, v in smiles_to_swap.items()}

doi_to_remove    = ['10.1021/ja980916u', '10.1021/jo0347011'] # intra molecular reaction with in situ dioxiran formation
smiles_intra     = ['O=C(CCl)CCC1CCCCC1', 'CCC(C1CCCCC1)CCC(C(OC)=O)=O', 'O=C(C(F)(F)F)CCO[C@]1(C)CCC[C@@H](C)C1']
smiles_intra     = [Chem.CanonSmiles(s) for s in smiles_intra]

# swapping smiles and removing wrong entries
smiles_list = df['Reactant_SMILES'].tolist()
smiles_can  = []
for s in smiles_list:
    try:
        s_can = Chem.CanonSmiles(s)
        if s_can in smiles_to_swap.keys():
            print(f"Swapping {s} for {smiles_to_swap[s_can]}")
            s_can = smiles_to_swap[s_can]
        smiles_can.append(s_can)
    except:
        if s == s:
            print(f"{s}  cannot be canonized")
        smiles_can.append(s)
        
df['Reactant_SMILES'] = smiles_can

df = df[~df['Reactant_SMILES'].isin(smiles_to_remove)]
df = df[~df['Reactant_SMILES'].isin(smiles_intra)]

# removing doi with intra molecular reactions
df = df[~df['DOI'].isin(doi_to_remove)]

df.to_excel('dioxirane_reaction_data/dataset_crude_filtered.xlsx', index=False)

# protonating amines
df.reset_index(drop=True, inplace=True)
reac_can = []
prod_can  = []
for i, s in enumerate(df['Reactant_SMILES'].tolist()):
    mod_reac = False
    mod_prod = False
    try:
        m = Chem.MolFromSmiles(s)
        count_N =  len([at for at in m.GetAtoms() if at.GetSymbol() == 'N'])
        count_no2   = fr_nitro(m)
        count_amide = fr_amide(m)
        count_azide = fr_azide(m)
        
        if count_N > count_no2 + count_amide + 3*count_azide: # only select molecules with nitrogen
            smiles_changed = False
            print(f"Reactant: {s}")
            for at in m.GetAtoms():
                if at.GetSymbol() == 'N' and at.GetFormalCharge() == 0: # select nitrogen atoms that are not charged
                    if str(at.GetHybridization()) != 'SP' and at.GetIsAromatic() == False:     # remove CN groups and aromatic nitrogens
                        # need to remove amides:
                        is_amide = False
                        for at2 in at.GetNeighbors():
                            if at2.GetSymbol() == 'C' and str(at.GetHybridization()) == 'SP2':
                                if 'O' in [at3.GetSymbol() for at3 in at2.GetAtoms()]:
                                    is_amide = True
                        
                        if not is_amide:
                            at.SetFormalCharge(1)
                        print(f"Reactant: {s} -> {Chem.MolToSmiles(m)}")
                        smiles_changed = True
                        reac_can.append(Chem.MolToSmiles(m))
                        mod_reac = True

            if smiles_changed:
                # looking at the product
                s_p = df.loc[i, 'Product_SMILES']
                m = Chem.MolFromSmiles(s_p)
                for at in m.GetAtoms():
                    if at.GetSymbol() == 'N' and at.GetFormalCharge() == 0: # select nitrogen atoms that are not charged
                        if str(at.GetHybridization()) != 'SP' and at.GetIsAromatic() == False:     # remove CN groups and aromatic nitrogens
                            # need to remove amides:
                            is_amide = False
                            for at2 in at.GetNeighbors():
                                if at2.GetSymbol() == 'C' and str(at.GetHybridization()) == 'SP2':
                                    if 'O' in [at3.GetSymbol() for at3 in at2.GetAtoms()]:
                                        is_amide = True
                            if not is_amide:
                                at.SetFormalCharge(1)
                            print(f"Product : {s} -> {Chem.MolToSmiles(m)}")
                            prod_can.append(Chem.MolToSmiles(m))
                            mod_prod = True
    except:
        pass
    if not mod_reac:
        reac_can.append(s)
    if not mod_prod:
        prod_can.append(df.loc[i, 'Product_SMILES'])

df['Reactant_SMILES'] = reac_can
df['Product_SMILES'] = prod_can

# removing non stereo specified molecules
# discard the smiles with no specified stereocenters.
# racemic mixtures are ok, but we discard non specific diastereoisomers
def discard(smi):
    """
    input: smiles
    output: True if the molecule should be discarded according to the fact that diastereoisomers are not correctly assigned, False if not
    """
    m = Chem.MolFromSmiles(smi)
    chiral_tags1      = Chem.FindMolChiralCenters(m, includeUnassigned=True)
    mol_sym           = is_mol_symmetric(smi)
    if mol_sym:
        m, sym_at = group_symmetric_atoms(smi)
    num_stereocenters = len(chiral_tags1)

    # if there are no stereocenters, we can't discard it
    if num_stereocenters == 0:
        #print("No stereocenters: keeper")
        return False
    
    # if there is only one stereocenter, we can't discard it
    elif num_stereocenters == 1:
        #print("1 stereocenters: keeper")
        return False
    
    # if there are more than one stereocenters:
    else:
        # if all stereocenters are assigned, we can't discard it
        if all([x[1] != "?" for x in chiral_tags1]):
            #print("All stereocenters are assigned: keeper")
            #print(chiral_tags1)
            return False
        
        else:
            mol_sym  = is_mol_symmetric(smi)
            if mol_sym:
                m, sym_at = group_symmetric_atoms(smi)
                #print("Molecule is symmetric - should we discard?: discard for now")
                return True
            else:
                #print("Molecule is not symmetric - and not all stereocenters are assigned: discard")
                return True

smiles = df['Reactant_SMILES'].tolist() 
keepers   = []
for s in smiles:
    try:
        discard_ = discard(s)
    except:
        discard_ = True
    
    if not discard_:
        keepers.append(s)
    else:
        if s == s:
            print(f"Discarding {s}")

discarded = [x for x in smiles if x not in keepers]

print("Keepers: ", len(keepers))
print("Discarded: ", len(discarded))

df = df[df['Reactant_SMILES'].isin(keepers)]

df.to_excel('dioxirane_reaction_data/dataset_crude_filtered.xlsx', index=False)
