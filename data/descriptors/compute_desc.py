import sys
import os
import argparse
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from rdkit import Chem
from joblib import Parallel, delayed

root = os.getcwd()
print(os.getcwd())

os.environ['KMP_DUPLICATE_LIB_OK']='True'

base_cwd = os.getcwd().split('regiochem')[0]
base_cwd = f"{base_cwd}/regiochem"
sys.path.append(f"{base_cwd}/utils/")

import descriptors as ds

### Parse arguments
parser = argparse.ArgumentParser(description='Descriptor computation')
parser.add_argument('--desc',
                    help='Descriptor to compute. Options are: XTB, DBSTEP, ENV1, ENV2, VBUR-RDKIT, GASTEIGER, ALL')

parser.add_argument('--csv',
                    help='CSV file with SMILES to compute descriptors, one columns has to be smiles, SMILES or Smiles')

parser.add_argument('--njobs',
                    help='Number of jobs to run in parallel, default is -1 (all cores)')

args = parser.parse_args()

if args.desc not in ['XTB', 'DBSTEP', 'ENV1', 'ENV2', 'VBUR-RDKIT', 'GASTEIGER', 'ALL']:
    exit(f"desc not in XTB, DBSTEP, ENV1, ENV2, VBUR-RDKIT, GASTERIGER, ALL, please choose one of these")

try:
    df     = pd.read_csv(args.csv)
    if 'SMILES' in df.columns:
        smiles = df['SMILES']
    elif 'Smiles' in df.columns:
        smiles = df['Smiles']
    elif 'smiles' in df.columns:
        smiles = df['smiles']
    else:
        exit(f"CSV file: {args.csv} does not have a column with SMILES, Smiles or smiles")
    can_smiles = [] 
    for smi in smiles:
        try:
            can_smiles.append(Chem.CanonSmiles(smi))
        except:
            print(f"{smi} is not a valid SMILES")

except:
    exit(f"CSV file: {args.csv} incorrect")

njobs = int(args.njobs) 
if njobs == None:
    njobs = -1

### Compute descriptors

def get_xtb(smiles, df):
    return smiles, ds.xtb_CH(smiles, print_=False, write=False, df_json=df)

def get_dbstep(smiles, df):
    return smiles, ds.dbstep_CH(smiles, print_=False, write=False, df_json=df)

def get_env1(smiles, df):
    return smiles, ds.env1(smiles, print_=False, write=False, df_json=df)

def get_env2(smiles, df):
    return smiles, ds.env2(smiles, print_=False, write=False, df_json=df)

def get_vbur_rdkit(smiles, df):
    return smiles, ds.rdkit_conf_Vbur(smiles, print_=False, write=False, df_json=df)

def get_gas(smiles, df):
    return smiles, ds.Gasteiger(smiles, print_=False, write=False, df_json=df)


if args.desc == 'XTB' or args.desc == 'ALL' :
    print(f"Computing XTB")
    json_file = f"{base_cwd}/data/descriptors/smiles_descriptors/xtb.json"
    f  = open(json_file)
    df = json.load(f)
    f.close()
    results = Parallel(n_jobs=njobs)(delayed(get_xtb)(smi, df) for smi in tqdm(can_smiles))
    for smi, desc in results:
        if smi in df.keys():
            df[smi].update({'xtb_CH': desc})
        else:
            df.update({smi : {'xtb_CH': desc}})

    with open(json_file, "w") as f:
        json.dump(df, f, sort_keys=True, indent=1)

if args.desc == 'DBSTEP' or args.desc == 'ALL' :
    print(f"Computing DBSTEP")
    json_file = f"{base_cwd}/data/descriptors/smiles_descriptors/dbstep.json"
    f  = open(json_file)
    df = json.load(f)
    f.close()
    results = Parallel(n_jobs=njobs)(delayed(get_dbstep)(smi, df) for smi in tqdm(can_smiles))
    for smi, desc in results:
        if smi in df.keys():
            df[smi].update({'dbstep_CH': desc})
        else:
            df.update({smi : {'dbstep_CH': desc}})

    with open(json_file, "w") as f:
        json.dump(df, f, sort_keys=True, indent=1)

if args.desc == 'ENV1' or args.desc == 'ALL' :
    print(f"Computing ENV1")
    json_file = f"{base_cwd}/data/descriptors/smiles_descriptors/C_env.json"
    f  = open(json_file)
    df = json.load(f)
    f.close()
    results = Parallel(n_jobs=njobs)(delayed(get_env1)(smi, df) for smi in tqdm(can_smiles))
    for smi, desc in results:
        if smi in df.keys():
            df[smi].update({'env1': desc})
        else:
            df.update({smi : {'env1': desc}})

    with open(json_file, "w") as f:
        json.dump(df, f, sort_keys=True, indent=1)

if args.desc == 'ENV2' or args.desc == 'ALL' :
    print(f"Computing ENV2")
    json_file = f"{base_cwd}/data/descriptors/smiles_descriptors/C_env.json"
    f  = open(json_file)
    df = json.load(f)
    f.close()
    results = Parallel(n_jobs=njobs)(delayed(get_env2)(smi, df) for smi in tqdm(can_smiles))
    for smi, desc in results:
        if smi in df.keys():
            df[smi].update({'env2': desc})
        else:
            df.update({smi : {'env2': desc}})

    with open(json_file, "w") as f:
        json.dump(df, f, sort_keys=True, indent=1)

if args.desc == 'VBUR-RDKIT' or args.desc == 'ALL' :
    print(f"Computing VBUR-RDKIT")
    json_file = f"{base_cwd}/data/descriptors/smiles_descriptors/vbur_rdkit.json"
    f  = open(json_file)
    df = json.load(f)
    f.close()
    results = Parallel(n_jobs=njobs)(delayed(get_vbur_rdkit)(smi, df) for smi in tqdm(can_smiles))
    for smi, desc in results:
        if smi in df.keys():
            df[smi].update({'vbur_rdkit': desc})
        else:
            df.update({smi : {'vbur_rdkit': desc}})

    with open(json_file, "w") as f:
        json.dump(df, f, sort_keys=True, indent=1)

if args.desc == 'GASTEIGER' or args.desc == 'ALL' :
    print(f"Computing GASTEIGER")
    json_file = f"{base_cwd}/data/descriptors/smiles_descriptors/reactions.json"
    f  = open(json_file)
    df = json.load(f)
    f.close()
    results = Parallel(n_jobs=njobs)(delayed(get_gas)(smi, df) for smi in tqdm(can_smiles))
    for smi, desc in results:
        if smi in df.keys():
            df[smi].update({'gasteiger': desc})
        else:
            df.update({smi : {'gasteiger': desc}})

    with open(json_file, "w") as f:
        json.dump(df, f, sort_keys=True, indent=1)




