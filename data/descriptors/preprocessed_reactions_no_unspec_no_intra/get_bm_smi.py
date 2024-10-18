import pandas as pd
from rdkit import Chem

df = pd.read_csv('df_bde.csv')
smi = df.Reactant_SMILES.unique()
smi = sorted(smi)
for s in smi:
   m = Chem.MolFromSmiles(s)
   c_count = len([at for at in m.GetAtoms() if at.GetSymbol() == 'C'])
   if c_count > 15:
       print(f"    '{s}'")
