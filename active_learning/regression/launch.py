import subprocess
import os
import json
import sys
import pandas as pd
from rdkit import Chem

root = os.getcwd()

try:
    base_cwd = os.getcwd().split('regio_dataset_design')[0]
    base_cwd = f"{base_cwd}/regio_dataset_design"
except:
    raise ValueError("You are not in the right directory, need to be in the 'notebooks' directory or subdirectory of it.")

sys.path.append(f"{base_cwd}/utils/")

files_pkl = os.listdir()

df_bde     = pd.read_csv(f"{base_cwd}/data/descriptors/preprocessed_dioxirane_reactions/df_bde.csv", index_col=0)

## get big smiles
smiles = df_bde.Reactant_SMILES.unique()
smiles = set([Chem.CanonSmiles(s) for s in smiles])

big_smiles   = []
small_smiles = []
for smiles in smiles:
    mol = Chem.MolFromSmiles(smiles)
    num_C = [atom.GetAtomicNum() for atom in mol.GetAtoms()].count(6)
    if num_C > 15:
        big_smiles.append(smiles)
    else:
        small_smiles.append(smiles)


idx_to_smi={}
for i, smi in enumerate(big_smiles):
   print(i, smi)
   idx_to_smi[i] = smi
   sh_name  = f"smi_{i}_selected.sh"
   sub_name = f"smi_{i}_submit_xtb.sh"
   os.system(f"cp run_selected.sh {sh_name}")
   os.system(f"cp submit.sh {sub_name}")
   os.system(f"sed -i 's,smi_to_calc,{smi},g' {sh_name}") 
   os.system(f"sed -i 's,submit_file,{sh_name},g' {sub_name}")
   os.system(f"sbatch {sub_name}")

with open("idx_to_smiles_selected.json", "w") as f:
   json.dump(idx_to_smi, f, indent=1)

