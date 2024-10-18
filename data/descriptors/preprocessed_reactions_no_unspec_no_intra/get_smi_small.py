import pandas as pd
from rdkit import Chem

df = pd.read_csv('df_en1.csv')
smi = df.Reactant_SMILES.unique()
smi = sorted(smi)
all_smi = ''
for s in smi:
    m = Chem.MolFromSmiles(s)
    if m is None:
        print(f"Invalid SMILES: {s}")
        continue
    else:
        count_c = [a.GetSymbol() for a in m.GetAtoms()].count('C')
        if count_c < 15:
            all_smi +=  s + '.'

print(all_smi)
