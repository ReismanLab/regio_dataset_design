import pandas as pd
from rdkit import Chem

df = pd.read_excel('dioxirane_reaction_data/dataset_crude_filtered.xlsx')
smiles = df.Reactant_SMILES.unique()

smi_can = []
for s in smiles:
   try:
       smi_can.append(Chem.CanonSmiles(s))
   except:
       if s == s:
           print(f"Ccould not canonicalize {s}")

df_ = pd.DataFrame(columns=["SMILES"], data= smi_can)
df_.to_csv('descriptors/can_smiles.csv')
