# add path to utils
import sys
import os
root = os.getcwd()

try:
    base_cwd = root.split('regio_dataset_design')[0]
    base_cwd = f"{base_cwd}regio_dataset_design"
except:
    raise ValueError("You are not in the right directory, need to be in the 'notebooks' directory or subdirectory of it.")

sys.path.append(f"{base_cwd}/utils/")

import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
import pandas as pd
import numpy as np
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser(description='Change model for performance evaluation of previously computed ACQF evaluations.')
parser.add_argument('--model',
                    help='One of "rank" or "reg"',
                    default="reg")
parser.add_argument('--feat',
                    help='Descriptor type',
                    default="selected")
parser.add_argument('--out',
                    help='Folder to store outputs',
                    default='test')
parser.add_argument('--input',
                    help='Folder containing previously computed ACQF evals',
                    default='regression/custom_db=1')
parser.add_argument('--n_est',
                    help='Argument to RandomForestRegressor',
                    default=250)
parser.add_argument('--max_feats',
                    help='Argument to RandomForestRegressor',
                    default=0.5)
parser.add_argument('--max_depth',
                    help='Argument to RandomForestRegressor',
                    default=10)
parser.add_argument('--min_samples_leaf',
                    help='Argument to RandomForestRegressor',
                    default=3)
parser.add_argument('--epochs',
                    help='Argument to RankNet',
                    default=10)
args = parser.parse_args()

model_type = args.model
reg = RandomForestRegressor(n_estimators=args.n_est,
                                max_features=args.max_feats,
                                max_depth=args.max_depth,
                                min_samples_leaf=args.min_samples_leaf)
feature_choice = args.feat
out = f"../../results/active_learning/{args.out}"
perf_desc = f"{model_type}_{feature_choice}"
epochs = args.epochs

if not os.path.exists(out):
    os.mkdir(out)

if model_type == "reg":
    import modelling as md
    import acquisition as aq
    import acqf as a
elif model_type == "rank":
    import rank as rk
    import tensorflow as tf
    import acquisition_ranking as ar

def get_rk_outs(feature_choice, target, epochs):
    # get the descriptors and data
    xi, xj, pij, pair_id, pair_query_id, smi_to_query, df = rk.make_descriptors_basic(option=feature_choice)
    df = df.drop(["DOI"], axis=1)
    def pred_func(training):
        # get the test and train data
        xitr, xjtr, pijtr, xitst, xjtst, pijtst = ar.make_test_and_train(
                                                                    xi, xj, pij, 
                                                                    pair_query_id, 
                                                                    smi_to_query,
                                                                    target,
                                                                    training)
        # make the first model
        m, history     = rk.train_and_validate(xitr, xitst, 
                                            xjtr, xjtst, 
                                            pijtr, pijtst, 
                                            epoch=epochs)
        
        del history
        
        df__, rel_reac = rk.call_model_on_validation(m, 
                                                    df, 
                                                    target)
        
        y_pred         = rk.predict_reactivities(m, df, target)
        top5           = rk.get_top_5_modified(df, rel_reac, target)
        tf.keras.backend.clear_session()
        return top5, y_pred
    return pred_func

def get_rf_outs(reg, feature_choice, target):
    df = a.make_descriptors_basic(option=feature_choice)
    df    = a.remove_large_molecules(df, target)
    df.drop(["DOI"], axis=1, inplace=True)
    def pred_func(training):
        reg__ = clone(reg)
        new_model      = md.train_model(training, df, reg__)
        valid_, y_pred = md.predict_site(new_model, target, df, classif=False)
        y_pred_list    = list(y_pred[x] for x in y_pred.keys())
        top5           = aq.get_top_5(df, y_pred_list, target)
        return top5, y_pred
    return pred_func

folder = f"../../results/active_learning/{args.input}"
for fname in tqdm(os.listdir(folder)):
    if fname[-4:] != ".pkl":
        continue
    with open(f"{folder}/{fname}", 'rb') as f:
        data = pickle.load(f)
    order_of_addition = [data[1][0]] + data[1][1]

    top_5_scores             = []
    carbo_reac_predictions   = []

    target_SMILES = [x for x in fname[:-4].split("_") if len(x) > 10][0]

    df   = pd.read_csv("../../data/descriptors/preprocessed_reactions/df_bde.csv", index_col=0)
    smiles = df.Reactant_SMILES.unique()
    if target_SMILES not in smiles:
        target_SMILES = target_SMILES.replace("-", "/")

    training_SMILES = []
    if model_type == "reg":
        pred_func = get_rf_outs(reg, feature_choice, target_SMILES)
    elif model_type == "rank":
        pred_func = get_rk_outs(feature_choice, target_SMILES, epochs)

    for smi in order_of_addition:
        training_SMILES.append(smi)
        top5, y_pred = pred_func(training_SMILES)
        top_5_scores.append(top5)
        carbo_reac_predictions.append(y_pred) 
    
    if "/" in target_SMILES: # handling for molecules with double bond stereochemistry 
        s = target_SMILES.replace("/", "-")
    else:
        s = target_SMILES
    with open(f"{out}/{fname[:-4]}_perf={perf_desc}.pkl", "wb") as f:
        pickle.dump([[top_5_scores], data[1], [carbo_reac_predictions], data[3], data[4]], f)
    