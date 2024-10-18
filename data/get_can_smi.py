import pandas as pd
from rdkit import Chem

df = pd.read_excel('reaction_data/dataset_crude_cleaned_doi_amine.xlsx')
smiles = df.Reactant_SMILES.unique()

smi_can = []
for s in smiles:
   try:
       smi_can.append(Chem.CanonSmiles(s))
   except:
       if s == s:
           print(f"Ccould not canonicalize {s}")

df_ = pd.DataFrame(columns=["SMILES"], data= smi_can)
df_.to_csv('can_smiles_N_proton.csv')
