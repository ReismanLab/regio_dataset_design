import pandas as pd

df = pd.read_csv('df_en1.csv')
smi = df.Reactant_SMILES.unique()
smi = sorted(smi)
all_smi = ''
for s in smi:
    all_smi +=  s + '.'
print(all_smi)
