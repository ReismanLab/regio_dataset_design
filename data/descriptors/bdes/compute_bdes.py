import argparse
import numpy as np
import pandas as pd
import json
from rdkit import Chem
from tqdm import tqdm
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers

### Parse arguments
parser = argparse.ArgumentParser(description='BDE computation')

parser.add_argument('--csv',
                    help='CSV file with SMILES to compute descriptors, one columns has to be smiles, SMILES or Smiles')

args = parser.parse_args()

try:
    df     = pd.read_csv(args.csv)
    print(df)
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

### bde utils functions
#load preprocess - used to convert to graph structue
import nfp
from preprocess_inputs_cfc import preprocessor
preprocessor.from_json('model_3_tfrecords_multi_halo_cfc/preprocessor.json')

class Slice(layers.Layer):
    def call(self, inputs):
        input_shape = tf.shape(inputs)
        num_bonds = input_shape[1] / 2
        output = tf.slice(inputs, [0, 0, 0], [-1, num_bonds, -1])
        output.set_shape(self.compute_output_shape(inputs.shape))
        return output

    def compute_output_shape(self, input_shape):
        return [input_shape[0], None, input_shape[2]]
    
custom_objects = {**nfp.custom_objects,'Slice':Slice}

model = tf.keras.models.load_model('model_3_multi_halo_cfc/best_model.hdf5', custom_objects=custom_objects)

def get_bdes(smiles_):
    #make the test data graphs
    smiles = Chem.CanonSmiles(smiles_)  
      
    pred_bdes = predict_bdes(smiles)

    dict_CH_bdes = {}

    dict_atom_to_bond = get_bond_idx_for_C(smiles)

    for atom, bond in dict_atom_to_bond.items():
        # need to complete the bde min max mean for each site and then dump it into a json file...
        bdes = pred_bdes['pred_bde'].iloc[bond]
        bdfes = pred_bdes['pred_bdfe'].iloc[bond]
        param_atoms =  {'bde_min':min(bdes), 
                        'bde_max':max(bdes), 
                        'bde_avg':float(np.mean(bdes)),
                        'bdfe_min':min(bdfes), 
                        'bdfe_max':max(bdfes), 
                        'bfde_avg':float(np.mean(bdfes))}
        dict_CH_bdes.update({atom:param_atoms})

    return dict_CH_bdes

def get_bdes_graph(smiles_):
    smiles = Chem.CanonSmiles(smiles_)  
      
    pred_bdes = predict_bdes(smiles)

    dict_CH_bdes = {}

    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    
    for bond in mol.GetBonds():
        bdes  = pred_bdes['pred_bde'].iloc[bond.GetIdx()]
        bdfes = pred_bdes['pred_bdfe'].iloc[bond.GetIdx()]

        bond_desc =  {'bde': float(bdes), 'bdfe': float(bdfes)}

        dict_CH_bdes.update({str(bond.GetBeginAtomIdx()) + '-' + str(bond.GetEndAtomIdx()):bond_desc})

    return dict_CH_bdes

def predict_bdes(smiles_):
    smiles = Chem.CanonSmiles(smiles_)
    test_smiles = (
        tf.data.Dataset.from_generator(
            lambda:  ([get_data(smiles)]), 
            output_signature= { **preprocessor.output_signature,'n_atom': tf.TensorSpec(shape=(), dtype=tf.int32, name=None),\
            'n_bond': tf.TensorSpec(shape=(), dtype=tf.int32, name=None) })
        .padded_batch(batch_size=1000, padding_values={**preprocessor.padding_values,'n_atom': tf.constant(0, dtype="int32"),\
            'n_bond': tf.constant(0, dtype="int32")})
        )
    
    predicted_bdes = model.predict(test_smiles, verbose=True)

    df = pd.DataFrame(predicted_bdes.reshape(-1, 2), columns=['pred_bde','pred_bdfe'])
    
    return df

def get_data(smiles):
    input_dict = preprocessor(smiles)
    input_dict['n_atom'] = len(input_dict['atom'] )
    input_dict['n_bond'] = len(input_dict['bond'] )
    return input_dict

def func(x):
    x['bond_index'] = range(0, predicted_bdes.shape[1])
    return x

def get_bond_idx_for_C(smiles_):
    """
    input : reactant Canonical SMILES
    output:
        dict_atom_to_bond, dict() with keys being atom idx in the CanonicalSMILES 
                                       values being the bond idx of the corresponding C-H bonds.
    """
    smiles = Chem.CanonSmiles(smiles_)
    m = Chem.MolFromSmiles(smiles)
    m = Chem.AddHs(m)

    m_, g = group_symmetric_atoms(smiles)
    groups_done = []
    idx_to_keep = [] 

    for at in m.GetAtoms():
        if at.GetSymbol() == 'C' and 'H' in [a.GetSymbol() for a in at.GetNeighbors()]:
            if g[at.GetIdx()] not in groups_done:
                groups_done.append(g[at.GetIdx()])
                idx_to_keep.append(at.GetIdx())

    dict_atom_to_bond = {}    
    for at in m.GetAtoms():
        if at.GetIdx() in idx_to_keep:
            bonds_to_keep = []
            bonds = at.GetBonds()
            for b in bonds:
                b_atoms = [b.GetBeginAtom().GetSymbol(), b.GetEndAtom().GetSymbol()]
                if 'H' in b_atoms and 'C' in b_atoms:
                    bonds_to_keep.append(b.GetIdx())
            dict_atom_to_bond.update({at.GetIdx():bonds_to_keep})
            
    return dict_atom_to_bond

def group_symmetric_atoms(smiles_):
    """
    input : reactant Canonical SMILES
    output:
        mol, Chem.Mol() object annotated with the symmetry group atoms are belonging to
        idx_to_group, dict() with keys being atom idx in the CanonicalSMILES and values beig the label of the group they belong to.
    """
    
    smiles = Chem.CanonSmiles(smiles_)
    mol    = Chem.MolFromSmiles(smiles)
    Chem.RemoveStereochemistry(mol)
    groups = Chem.CanonicalRankAtoms(mol, breakTies=False)
    
    idx_to_group = {}
    

    for at in mol.GetAtoms():
        at.SetProp('atomNote', f"{groups[at.GetIdx()]}")  
        if at.GetSymbol() == 'C':
            idx_to_group.update({at.GetIdx(): groups[at.GetIdx()]})

    return mol, idx_to_group

def is_mol_symmetric(smiles_):
    """
    input : reactant Canonical SMILES
    output:
        boolean, True if the carbon squelettom has equivalent carbons, False if not
    """
    smiles = Chem.CanonSmiles(smiles_)
    mol = Chem.MolFromSmiles(smiles)
    
    # remove stereochemistry: helps find symmetries...
    Chem.RemoveStereochemistry(mol)
    
    groups = list(Chem.CanonicalRankAtoms(mol, breakTies=False))

    if len(groups) - len(set(groups)) > 0:
        return True
    else:
        return False
    
def get_bdes_and_diplay_m(smiles_):
    smiles = Chem.CanonSmiles(smiles_)   
    
    pred_bdes = predict_bdes(smiles)

    m = Chem.MolFromSmiles(smiles)
    m = Chem.AddHs(m)
    for b in m.GetBonds():
        b.SetProp('bondNote', f"{round(pred_bdes['pred_bde'].iloc[b.GetIdx()],0)}")
    
    return m

### main
desc_file = open('../smiles_descriptors/bdes.json')
df_desc = json.load(desc_file)
desc_file.close()

for k, smiles in tqdm(enumerate(can_smiles)):

    smiles = Chem.CanonSmiles(smiles)

    if smiles in df_desc.keys():                
        df_desc[smiles].update({'pred_bdes': get_bdes(smiles)})
    
    else:
        df_desc.update({smiles: {}})
        df_desc[smiles].update({'pred_bdes': get_bdes(smiles)})

    if k % 100 == 0:
        with open('../smiles_descriptors/bdes.json', "w") as desc_file:
            json.dump(df_desc, desc_file, sort_keys=True, indent=1)
        print("\n\nBDE updated\n\n")

with open('../smiles_descriptors/bdes.json', "w") as desc_file:
    json.dump(df_desc, desc_file, sort_keys=True, indent=1)
print("\n\nBDE updated\n\n")


desc_file = open('../smiles_descriptors/bdes_graphs.json')
df_desc = json.load(desc_file)
desc_file.close()

for k, smiles in tqdm(enumerate(can_smiles)):

    smiles = Chem.CanonSmiles(smiles)

    if smiles in df_desc.keys():                
        df_desc[smiles].update({'pred_bdes': get_bdes_graph(smiles)})
    
    else:
        df_desc.update({smiles: {}})
        df_desc[smiles].update({'pred_bdes': get_bdes_graph(smiles)})

    if k % 100 == 0:
        with open('../smiles_descriptors/bdes_graphs.json', "w") as desc_file:
            json.dump(df_desc, desc_file, sort_keys=True, indent=1)
        print("\n\nBDE GRAPHS updated\n\n")

with open('../smiles_descriptors/bdes_graphs.json', "w") as desc_file:
    json.dump(df_desc, desc_file, sort_keys=True, indent=1)
print("\n\nBDE GRAPHS updated\n\n")