import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

# Read the data
df = pd.read_csv('../../data/descriptors/preprocessed_reactions_no_unspec_no_intra/df_bde.csv')

# Get the SMILES
smiles = df['Reactant_SMILES'].unique()

# Get smiles with more than 15 C atoms
def get_C_count(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return [atom.GetSymbol() for atom in mol.GetAtoms()].count('C') 

smiles = [s for s in smiles if get_C_count(s) > 15]

print(len(smiles))

for i, s in enumerate(smiles):
    #print(i, s)
    #if i == 1:
    os.system(f"python learning_curve_comp.py --smi \"{s}\" --acqf_list \"['acqf_10', 'acqf_2-1', 'acqf_6']\" --suffix {i}_af10")
