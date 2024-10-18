import pandas as pd
from rdkit import Chem
import argparse 
import numpy as np

parser = argparse.ArgumentParser(description='Descriptor computation')
parser.add_argument('--smiles',
                    help='Smiles of the molecule')
args = parser.parse_args()

smiles = args.smiles
try:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
except:
    raise ValueError("Invalid SMILES")

df = pd.read_excel('reaction_data/dataset_crude.xlsx')

smiles = Chem.CanonSmiles(smiles)

can_smiles = []
for s in df['Reactant_SMILES']:
    try:
        can_smiles.append(Chem.CanonSmiles(s))
    except:
        can_smiles.append(None)
        if s == s:
            print(f"Error with {s} in canonicalization")
df['Reactant_SMILES'] = can_smiles

for x in np.unique(df[df['Reactant_SMILES'] == smiles]['DOI'].values):
    print(f"DOI: {x}")