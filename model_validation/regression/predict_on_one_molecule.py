# add path to utils
import sys
import os
root = os.getcwd()

try:
    base_cwd = root.split('regio_dataset_design')[0]
    base_cwd = base_cwd + "regio_dataset_design"
except:
    raise ValueError("You are not in the right directory, need to be in the 'notebooks' directory or subdirectory of it.")

sys.path.append(base_cwd+"/utils/")

from sklearn.base import clone
import pandas as pd
import modelling as md
import visualization as viz
import pandas as pd
from rdkit import Chem
import numpy as np

# params
model = "RF2"
smiles = 'C[C@]12CC[C@H]3[C@H]([C@@H]1CC[C@@H]2O)CCC4=C3C=CC(=C4)O'
feature_choice = "Selected"
include_large = True
atom = "O"
reg_args = {"n_estimators":250, "max_features":0.5,
            "max_depth":10, "min_samples_leaf":3}
rxn_folder = "preprocessed_reactions_no_unspec_no_intra_unnorm"
smiles = Chem.CanonSmiles(smiles)
outfile = "estradiol"
figtype = "numbers"
threshold_correlated = 0.9

options = ['Custom', 'BDE', 'XTB', 'Gasteiger', 'ENV-1', 'ENV-2', 'DBSTEP', 'Selected']

assert feature_choice in options, f"option should be in {options}"

# smiles equality check accounting for unassigned chiral tags
def smiles_are_same(s1, s2):
    # ensure they are canonical so chiral tags are the same index
    s1, s2 = Chem.CanonSmiles(s1), Chem.CanonSmiles(s2)
    if s1 == s2: # if canonical smiles are same, we don't need to check all the chiral centers
        return True
    # if the non-isomeric smiles are different, we don't need to check all the chiral centers
    if Chem.CanonSmiles(s1, useChiral=False) != Chem.CanonSmiles(s2, useChiral=False):
        return False
    # smiles are isomeric, but not identical -- need to check if this is due to not all centers being assigned
    mol1 = Chem.MolFromSmiles(s1)
    Chem.AssignAtomChiralTagsFromStructure(mol1)
    chiral_tags1 = Chem.FindMolChiralCenters(mol1, includeUnassigned=True)
    mol2 = Chem.MolFromSmiles(s2)
    Chem.AssignAtomChiralTagsFromStructure(mol2)
    chiral_tags2 = Chem.FindMolChiralCenters(mol2, includeUnassigned=True)
    assert len(chiral_tags1) == len(chiral_tags2), f"number of chiral tags is not the same for: {s1} {s2}"
    mismatch_count = 0
    for i in range(len(chiral_tags1)):
        idx1, assign1 = chiral_tags1[i]
        idx2, assign2 = chiral_tags2[i]
        if idx1 != idx2:
            print(f"chiral tag indices not the same, assigning as diastereomers: {s1} {s2}")
            return False

        if assign1 != "?" and assign2 != "?" and assign1 != assign2:
            mismatch_count += 1

    if mismatch_count == 0 or mismatch_count == len(chiral_tags1):
        return True
    else:
        return False

# read_data
df_xtb     = pd.read_csv(f"{base_cwd}/data/descriptors/{rxn_folder}/df_xtb.csv", index_col=0)
df_gas     = pd.read_csv(f"{base_cwd}/data/descriptors/{rxn_folder}/df_gas.csv", index_col=0)
df_dbs     = pd.read_csv(f"{base_cwd}/data/descriptors/{rxn_folder}/df_dbstep.csv", index_col=0)
df_en1     = pd.read_csv(f"{base_cwd}/data/descriptors/{rxn_folder}/df_en1.csv", index_col=0)
df_en2     = pd.read_csv(f"{base_cwd}/data/descriptors/{rxn_folder}/df_en2.csv", index_col=0)
df_bde     = pd.read_csv(f"{base_cwd}/data/descriptors/{rxn_folder}/df_bde.csv", index_col=0)
df_rdkVbur = pd.read_csv(f"{base_cwd}/data/descriptors/{rxn_folder}/df_rdkVbur.csv", index_col=0)
df_sel     = pd.read_csv(f"{base_cwd}/data/descriptors/{rxn_folder}/df_selected.csv", index_col=0)
df_custom  = pd.read_csv(f"{base_cwd}/data/descriptors/{rxn_folder}/df_custom.csv", index_col=0)
df_en1_ohe = pd.read_csv(f"{base_cwd}/data/descriptors/{rxn_folder}/df_en1_ohe.csv", index_col=0)

features = {#'AIMNET-A'  : df_a,
            #'AIMNET-AIM': df_aim,
            'BDE'       : df_bde.drop(["DOI"], axis=1),
            'XTB'       : df_xtb.drop(["DOI"], axis=1),
            'DBSTEP'    : df_dbs.drop(["DOI"], axis=1), 
            'Gasteiger' : df_gas.drop(["DOI"], axis=1),
            'ENV-1'     : df_en1.drop(["DOI"], axis=1),
            'ENV-1-OHE' : df_en1_ohe.drop(["DOI"], axis=1),
            'ENV-2'     : df_en2.drop(["DOI"], axis=1),
            'Rdkit-Vbur': df_rdkVbur.drop(["DOI"], axis=1),
            'Selected'  : df_sel.drop(["DOI"], axis=1),
            'Custom'    : df_custom.drop(["DOI"], axis=1)
            }
df = features[feature_choice]

## get big smiles
smis = df.Reactant_SMILES.unique()
smis = set([Chem.CanonSmiles(s) for s in smis])

big_smiles   = []
small_smiles = []
for s in smis:
    mol = Chem.MolFromSmiles(s)
    num_C = [atom.GetAtomicNum() for atom in mol.GetAtoms()].count(6)
    if num_C > 15:
        big_smiles.append(s)
    else:
        small_smiles.append(s)
print("Big SMILES", len(big_smiles))

# ensure target is not included in training
df = df.loc[[not smiles_are_same(s, smiles) for s in df["Reactant_SMILES"]]]

if not include_large:
    print("Removing large molecules!")
    df = df.loc[~np.isin(df.Reactant_SMILES, big_smiles)]
       
df_c_ox = pd.DataFrame()
norms = {}
for col in df.columns:
    if col in ["Reactant_SMILES", "Atom_nº", "Selectivity", "Reactive Atom"]:
        continue
    df_c_ox[col] = (df[col] - np.mean(df[col]))/(max(df[col]) - min(df[col]))
    norms[col] = [np.mean(df[col]), max(df[col]), min(df[col])]

if threshold_correlated != None: # remove correlated features
    df_c_ox = md.remove_correlated_features(df_c_ox, False, threshold_correlated)

df_c_ox["Reactant_SMILES"] = df["Reactant_SMILES"]
df_c_ox["Atom_nº"] = df["Atom_nº"]
df_c_ox["Selectivity"] = df["Selectivity"]
df_c_ox["Reactive Atom"] = df["Reactive Atom"]

df = df_c_ox

target_df = pd.DataFrame({"Reactant_SMILES": [smiles], "Product_SMILES": [smiles], "rxn_ID":[0], "Selectivity (%)": [0]})
target_df.to_csv("target.csv")

dfs = []
if feature_choice in ["XTB", "Custom", "Selected"]:
    print("Loading XTB...")
    target_df = md.prepare_reactivity_mapping('XTB', file="target.csv", 
                                            preprocess=True,
                                            normalize=False, threshold_correlated=None,
                                            rxn_folder="target_data", atom=atom)
    dfs.append(target_df)
if feature_choice in ["BDE", "Custom", "Selected"]:
    print("Loading BDE...")
    target_df = md.prepare_reactivity_mapping('BDE', file="target.csv", 
                                            preprocess=True,
                                            normalize=False, threshold_correlated=None,
                                            rxn_folder="target_data", atom=atom)
    dfs.append(target_df)
if feature_choice in ["Gasteiger", "Custom", "Selected"]:
    print("Loading Gasteiger...")
    target_df = md.prepare_reactivity_mapping('Gasteiger', file="target.csv", 
                                            preprocess=True,
                                            normalize=False, threshold_correlated=None,
                                            rxn_folder="target_data", atom=atom)
    dfs.append(target_df)
if feature_choice in ["ENV-1", "Custom", "Selected", "ENV-1-OHE"]:
    print("Loading ENV-1...")
    target_df = md.prepare_reactivity_mapping('ENV-1', file="target.csv", 
                                            preprocess=True,
                                            normalize=False, threshold_correlated=None,
                                            rxn_folder="target_data", atom=atom)
    if feature_choice in ["ENV-1", "Custom", "Selected"]:
        dfs.append(target_df)
    if feature_choice in ["ENV-1-OHE", "Custom", "Selected"]:
        # OHE descriptors
        from sklearn.preprocessing import OneHotEncoder
        enc        = OneHotEncoder()
        df_en1_ohe = enc.fit_transform(target_df.drop(columns=['Reactive Atom', 'Selectivity', 'Atom_nº', 'Reactant_SMILES']))
        df_en1_ohe = pd.DataFrame(df_en1_ohe.toarray(), columns=enc.get_feature_names_out())
        df_en1_ohe = pd.concat([df_en1_ohe, target_df[['Reactive Atom', 'Selectivity', 'Atom_nº', 'Reactant_SMILES']]], axis=1)
        dfs.append(df_en1_ohe)

if feature_choice in ["ENV-2", "Custom", "Selected"]:
    print("Loading ENV-2...")
    target_df = md.prepare_reactivity_mapping('ENV-2', file="target.csv", 
                                            preprocess=True,
                                            normalize=False, threshold_correlated=None,
                                            rxn_folder="target_data", atom=atom)
    dfs.append(target_df)
if feature_choice in ["DBSTEP", "Custom", "Selected"]:
    print("Loading DBSTEP...")
    target_df = md.prepare_reactivity_mapping('DBSTEP', file="target.csv", 
                                            preprocess=True,
                                            normalize=False, threshold_correlated=None,
                                            rxn_folder="target_data", atom=atom)
    dfs.append(target_df)
if feature_choice in ['Rdkit-Vbur', "Custom", "Selected"]:
    print("Loading Rdkit-Vbur...")
    target_df = md.prepare_reactivity_mapping('Rdkit-Vbur', file="target.csv", 
                                            preprocess=True,
                                            normalize=False, threshold_correlated=None,
                                            rxn_folder="target_data", atom=atom)
    dfs.append(target_df)

if len(dfs) == 1:
    target_df = dfs[0]
else:
    target_df = pd.DataFrame()
    for col in df.columns:
        for df_ in dfs:
            if col in df_.columns:
                if col in ["Reactant_SMILES", "Atom_nº", "Selectivity", "Reactive Atom"]:
                    target_df[col] = df_[col]
                else:
                    target_df[col] = (df_[col] - norms[col][0])/(norms[col][1] - norms[col][2])
print(len(target_df))

df = pd.concat([df, target_df])

if model == "RF2" or model == 'RF-OPT-XTB':
    from sklearn.ensemble import RandomForestRegressor
    reg = RandomForestRegressor(**reg_args)
if model == "MLP" or model == "MLP2":
    from sklearn.neural_network import MLPRegressor
    reg = MLPRegressor(**reg_args)
if model == "SVR":
    from sklearn.svm import SVR
    reg = SVR(**reg_args)
if model == "KNN":
    from sklearn.neighbors import KNeighborsRegressor
    reg = KNeighborsRegressor(**reg_args)
if model == "GPR":
    from sklearn.gaussian_process import GaussianProcessRegressor
    reg = GaussianProcessRegressor(**reg_args)
if model == "LR":
    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()

def get_reg_outs(reg, df, target):
    training_smiles = [s for s in df.Reactant_SMILES.unique() if s != target]
    reg_      = md.train_model(training_smiles, df, reg)
    valid_, y_pred = md.predict_site(reg_, target, df, classif=False)
    return y_pred


y_pred = get_reg_outs(reg, df, smiles)
for i in range(9):
    tmp = get_reg_outs(reg, df, smiles)
    for at in y_pred:
        y_pred[at] += tmp[at]
for at in y_pred:
    y_pred[at] = y_pred[at]/10

if figtype == "numbers":
    img_pred_rank = viz.visualize_regio_pred(smiles, y_pred, draw="numbers")
    img_pred_rank.save(f"{outfile}.png", dpi=(1200,1200))
if figtype == "colors":
    img_pred_rank = viz.visualize_regio_pred(smiles, y_pred, draw="colors")
    img_pred_rank.savefig(f"{outfile}.png", bbox_inches='tight', dpi=300)
