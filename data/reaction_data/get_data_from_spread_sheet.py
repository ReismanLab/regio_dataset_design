import pandas as pd
from datetime import date
sheet_id  = "1OijQ0fiJTJn8OOOU9pJ9qm5JxmQWKpmb"
url       = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?gid=751558775&format=csv"
df_xlsx   = pd.read_csv(url, index_col=0)
data_file = f"data-{date.today()}.xlsx"
df_xlsx.to_excel(data_file)

# get canonical smiles in reaction dataset
from rdkit import Chem
smiles  = df_xlsx['Reactant_SMILES'].unique()
can_smi = [] 
for smi in smiles:
    try:
        can_smi.append(Chem.CanonSmiles(smi))
    except:
        print(f"{smi} is not a valid SMILES")

df_can = pd.DataFrame(can_smi, columns=['SMILES'])
df_can.to_csv(f"can_smiles.csv")