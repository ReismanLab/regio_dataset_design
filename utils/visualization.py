from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps

import numpy as np
import pylab

pylab.ioff()

def visualize_regio_pred(smiles, y_pred, 
                         draw         = 'colors', 
                         scale        = -1, 
                         coordScale   = 1.0, 
                         colors       = 'r', 
                         contourLines = 5,
                         step         = 0.001,
                         alpha        = 0.3,
                         colorMap     = 'Spectral_r'
                         ): 
    """
    smiles: SMILES of the reactant
    y_pred a dict with indexes being atom num and keys being reactivity
    draw: 'colors' or 'numbers', colors gives a intensity map, numbers gives the values
    """
    smiles = Chem.CanonSmiles(smiles)
    # print(smiles)
    mol    = Chem.MolFromSmiles(smiles)
    mol    = Chem.AddHs(mol)
    atom_nums = []
    for at in mol.GetAtoms():
        if at.GetSymbol() == 'C':
            if 'H' in [n.GetSymbol() for n in at.GetNeighbors()]:
                atom_nums.append(at.GetIdx())

    sel = y_pred #dict(zip(atom_nums, y_pred))
    contribs = []
    for at in mol.GetAtoms():
        if at.GetIdx() in atom_nums:
            try:
                at.SetProp('atomNote', str(round(sel[at.GetIdx()], 2)))
                contribs.append(sel[at.GetIdx()])
            except:
                #print(at.GetIdx(), " is probably discarded because of symmetry")
                contribs.append(0)
                pass
        else:
            contribs.append(0)
    mol = Chem.RemoveHs(mol)

    contribs = (np.array(contribs) - np.min(contribs))/(np.max(contribs) - np.min(contribs))
    contribs = contribs.tolist()    
    
    if draw == 'colors':    
        d2d = Draw.MolDraw2DCairo(350,300)
        dopts = d2d.drawOptions()
        dopts.setBackgroundColour((0,.9,.9,.3))
        fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, contribs, 
                                                         colorMap=colorMap, 
                                                         contourLines=contourLines,
                                                         scale=scale, 
                                                         coordScale=coordScale,
                                                         step=step,
                                                         colors=colors,
                                                         alpha=alpha,
                                                         )
        d2d.FinishDrawing()
        return fig
    
    elif draw == 'numbers':
        img = Draw.MolToImage(mol, size=(1200, 1200))
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.axis('off')
        return fig
    


def visualize_regio_exp(smiles, df, 
                         draw         = 'colors',  
                         scale        = -1, 
                         coordScale   = 1.0, 
                         colors       = 'r', 
                         contourLines = 5,
                         step         = 0.001,
                         alpha        = 0.3,
                         colorMap     = 'Spectral_r',
                         obs          = "Selectivity"):
    """"
    plots the experimental reactivity of the reactant if the reactat has been investigated
    df: dataframe containing the experimental reactivity of the reactant
        requires columns : 'Atom_nº' and 'Selectivity'
    """

    if obs not in df.columns or 'Atom_nº' not in df.columns:
        print(f"The dataframe should contain the column '{obs}' and 'Atom_nº'")
        raise ValueError

    smiles = Chem.CanonSmiles(smiles)
    mol    = Chem.MolFromSmiles(smiles)
    mol    = Chem.AddHs(mol)
    sub_df = df[df.Reactant_SMILES == smiles]
    
    if len(sub_df) == 0:
        print(f"{smiles} is not in the dataset")
        return None

    contribs = []

    for at in mol.GetAtoms():
        if at.GetSymbol() == 'C' and 'H' in [n.GetSymbol() for n in at.GetNeighbors()]:
            try:
                at.SetProp('atomNote', str(round(sub_df[sub_df['Atom_nº'] == at.GetIdx()][obs].values[0], 2)))
                contribs.append(sub_df[sub_df['Atom_nº'] == at.GetIdx()][obs].values[0])
            except:
                contribs.append(0)
                pass
        else:
            contribs.append(0)

    contribs = np.array(contribs) 
    mol = Chem.RemoveHs(mol)

    if draw == 'colors':   
        d2d = Draw.MolDraw2DCairo(350,300)
        dopts = d2d.drawOptions()
        dopts.setBackgroundColour((0,.9,.9,.3))

        min_y = np.min(contribs)
        max_y = np.max(contribs)
        contribs = (contribs - min_y)/(max_y - min_y)
        contribs = contribs.tolist() 

        fig = SimilarityMaps.GetSimilarityMapFromWeights(mol, contribs, 
                                                         colorMap=colorMap, 
                                                         contourLines=contourLines,
                                                         scale=scale, 
                                                         coordScale=coordScale,
                                                         step=step,
                                                         colors=colors,
                                                         alpha=alpha,
                                                         )
        return fig
    
    elif draw == 'numbers':
        img = Draw.MolToImage(mol)
        return img
    

# making videos of predictions along the training of a model on a target molecule
# note should be able to adapt this function to any acquisition function
import os
try:
    import cv2
except:
    print("Could not import cv2, you might have trouble trying to make video animations... but the rest should be fine.")

import modelling as md
from random import randrange
from sklearn.ensemble import RandomForestRegressor

def make_video_from_data(y_preds, video_name, target_SMILES):

    for v, y in enumerate(y_preds):
        img = visualize_regio_pred(target_SMILES, y)
        v   = ("%03d" % (v,))
        img.savefig(f"tmp/pred_{v}.png", bbox_inches='tight')

    path           = 'tmp/'
    out_path       = 'tmp/'
    out_video_name = f"evol_pred_{video_name}.mp4"
    out_video_path = os.path.join(out_path, out_video_name)

    pre_imgs       = os.listdir(path)
    pre_imgs       = list(pre_imgs)
    pre_imgs       = sorted(pre_imgs)
    
    pre_imgs_      = []
    for img_ in pre_imgs:
        if 'png' in img_:
            pre_imgs_.append(img_)
    pre_imgs       = pre_imgs_

    img = []
    for i in pre_imgs:
        i =  path + i 
        img.append(i)

    cv2_fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    frame = cv2.imread(img[0])
    size  = list(frame.shape)
    
    del size[2]
    size.reverse()

    video = cv2.VideoWriter(out_video_path,  cv2_fourcc, 48, size)

    for i in range(len(img)):
        for _ in range(3):  # image duration
            video.write(cv2.imread(img[i])) 

    video.release() 

import seaborn as sns
import matplotlib.pyplot as plt

def make_video_from_data_with_accuracy(y_preds, top1, ypreds_r, top1_r, video_name, target_SMILES):
    """"
    Makes video for the evolution of the predictions on a target molecule between an aqusition function and random 
    y_preds: list of predicted values for each atom in the reactant
    top1: list of tuples containing the top 1 accuracy of the model at each iteration
    ypreds_r: list of predicted values for each atom in the reactant for the random acquisition function
    top1_r: list of tuples containing the top 1 accuracy of the model at each iteration for the random acquisition function
    video_name: str, name of the video to be saved
    target_SMILES: str, SMILES of the target molecule
    """

    topn = []
    for i in range(len(top1)):
        topn.append(sum(top1[i]))

    topnr = []
    for i in range(len(top1_r)):    
        topnr.append(sum(top1_r[i]))

    print(len(topn))
    print(len(topnr))          

    for v, y in enumerate(y_preds):

        fig, ax = plt.subplots(1, 3, figsize=(16, 4))
        img_    = visualize_regio_pred(target_SMILES, y)
        v_      = ("%03d" % (v,))
        img_.savefig(f"tmp/pred_{v_}_mol_acqf.png", bbox_inches='tight')
        
        from matplotlib import image as img
        ax[0].imshow(img.imread(f"tmp/pred_{v_}_mol_acqf.png"))
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].set_title('AQCF Prediction')

        if top1[v][0] == 1:
            ax[0].patch.set_edgecolor('black')  
            ax[0].patch.set_linewidth(5)
        elif top1[v][1] == 1:
            ax[0].patch.set_edgecolor('brown')  
            ax[0].patch.set_linewidth(5)
        elif top1[v][2] == 1:
            ax[0].patch.set_edgecolor('red')  
            ax[0].patch.set_linewidth(5)
        elif top1[v][3] == 1:
            ax[0].patch.set_edgecolor('orange')  
            ax[0].patch.set_linewidth(5)
        else:
            ax[0].patch.set_edgecolor('yellow')  
            ax[0].patch.set_linewidth(5)

        img_ = visualize_regio_pred(target_SMILES, ypreds_r[v])
        v_   = ("%03d" % (v,))
        img_.savefig(f"tmp/pred_{v_}_mol_rand.png", bbox_inches='tight')

        ax[1].imshow(img.imread(f"tmp/pred_{v_}_mol_rand.png"))
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        ax[1].set_title('Random Prediction')

        if top1_r[v][0] == 1:
            ax[1].patch.set_edgecolor('black')  
            ax[1].patch.set_linewidth(5)
        elif top1_r[v][1] == 1:
            ax[1].patch.set_edgecolor('brown')  
            ax[1].patch.set_linewidth(5)
        elif top1_r[v][2] == 1:
            ax[1].patch.set_edgecolor('red')  
            ax[2].patch.set_linewidth(5)
        elif top1_r[v][3] == 1:
            ax[1].patch.set_edgecolor('orange')  
            ax[1].patch.set_linewidth(5)
        else:
            ax[1].patch.set_edgecolor('yellow')  
            ax[1].patch.set_linewidth(5)

        #sns.lineplot(x=range(v), y=[top1[x][3] for x in range(v)], color='orange', ax=ax[2])
        #sns.lineplot(x=range(v), y=[top1[x][2] for x in range(v)], color='red', ax=ax[2])
        #sns.lineplot(x=range(v), y=[top1[x][1] for x in range(v)], color='brown', ax=ax[2])
        #sns.lineplot(x=range(v), y=[top1[x][0] for x in range(v)], color='black', ax=ax[2])
        #sns.scatterplot(x=range(v), y=[top1[x][3] for x in range(v)], color='orange', ax=ax[2], size=10, label='TOP5-aqcf')
        #sns.scatterplot(x=range(v), y=[top1[x][2] for x in range(v)], color='red', ax=ax[2], size=10, label='TOP3-aqcf')
        #sns.scatterplot(x=range(v), y=[top1[x][1] for x in range(v)], color='brown', ax=ax[2], size=10, label='TOP2-aqcf')
        #sns.scatterplot(x=range(v), y=[top1[x][0] for x in range(v)], color='black', ax=ax[2], size=10, label='TOP1-aqcf')
        sns.lineplot(x=range(v),    y=topnr[:v], color='blue', ax=ax[2], linewidth=5, label=None)
        sns.scatterplot(x=range(v), y=topnr[:v], color='blue', ax=ax[2], size=10, label='Random')
        sns.lineplot(x=range(v),    y=topn[:v] , color='red' , ax=ax[2], label=None)
        sns.scatterplot(x=range(v), y=topn[:v] , color='red' , ax=ax[2], size=10, label='AQCF')

        ax[2].set_title('Top 1 accuracy')
        ax[2].set_xlabel('Molecules Added')
        ax[2].set_ylabel('Accuracy')
        ax[2].set_ylim(-0.05, 5.05)
        ax[2].set_yticks(range(6), ['<TOP10', 'TOP10', 'TOP5', 'TOP3', 'TOP2', 'TOP1'])
        ax[2].set_xlim(0, len(top1))

        #sns.lineplot(dashes=[range(v), [top1_r[x][3] for x in range(v)]], color='orange', ax=ax[2], )
        #sns.lineplot(dashes=[range(v), [top1_r[x][2] for x in range(v)]], color='red', ax=ax[2])
        #sns.lineplot(dashes=[range(v), [top1_r[x][1] for x in range(v)]], color='brown', ax=ax[2])
        #sns.lineplot(dashes=[range(v), [top1_r[x][0] for x in range(v)]], color='black', ax=ax[2])
        #sns.scatterplot(x=range(v), y=[top1_r[x][3] for x in range(v)], color='orange', ax=ax[2], size=10, label='TOP5-random')
        #sns.scatterplot(x=range(v), y=[top1_r[x][2] for x in range(v)], color='red', ax=ax[2], size=10, label='TOP3-random')
        #sns.scatterplot(x=range(v), y=[top1_r[x][1] for x in range(v)], color='brown', ax=ax[2], size=10, label='TOP2-random')
        #sns.scatterplot(x=range(v), y=[top1_r[x][0] for x in range(v)], color='black', ax=ax[2], size=10, label='TOP1-random')
        fig.savefig(f"tmp/pred_{v_}.png", bbox_inches='tight')
        plt.close('all')

    path           = 'tmp/'
    out_path       = 'tmp/'
    out_video_name = f"evol_pred_{video_name}.mp4"
    out_video_path = os.path.join(out_path, out_video_name)

    pre_imgs       = os.listdir(path)
    pre_imgs       = list(pre_imgs)
    pre_imgs       = sorted(pre_imgs)
    
    pre_imgs_      = []
    for img_ in pre_imgs:
        if 'png' in img_:
            pre_imgs_.append(img_)
    pre_imgs       = pre_imgs_

    img = []
    for i in pre_imgs:
        i =  path + i 
        img.append(i)

    cv2_fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    frame = cv2.imread(img[0])
    size  = list(frame.shape)
    
    del size[2]
    size.reverse()

    video = cv2.VideoWriter(out_video_path, cv2_fourcc, 48, size)

    for i in range(len(img)):
        for _ in range(10):  # image duration
            video.write(cv2.imread(img[i])) 

    video.release() 

def make_video(start_model, target_SMILES, df, training_reactants, video_name):
    """
    inputs:
        start_model       : sklearn model, trained on the 2 selected DOIs/ or any model
        target_SMILES     : str, SMILES of the target molecule
        df                : dataframe, containing the remaining molecules ad their experimental reactivity
        training_reactants: list, of SMILES of molecules used to train the model
        video_name        : str, name of the video to be saved
    output:
        img_exp           : image of the experimental observation of the target reactivity
    
    This function trains a model on a training set, evaluates the preidictions on the target molecule. 
    Iteratively adds a random molecule to the training set, retrains the model and evaluates the predictions on the target molecule.
    This is done until the training set contains all the molecules in the dataset except the target molecule. 
    """
    reg = RandomForestRegressor(n_estimators=250, max_features=0.5, max_depth=10, min_samples_leaf=3)

    valid_, y_pred = md.predict_site(start_model, target_SMILES, df.drop(columns=['DOI']), classif=False)
    img = visualize_regio_pred(target_SMILES, y_pred)
    v = ("%03d" % (0,))
    img.savefig(f"../Figures/Video/pred_{v}.png", bbox_inches='tight')

    img_exp   = visualize_regio_exp(target_SMILES, df)
    
    continue_ = True
    i         = 1
    while continue_:
        training_reactants, n_remaining = add_random_molecule(training_reactants, df, target_SMILES)
        if n_remaining == 1:
            continue_ = False
        new_model      = md.train_model(training_reactants, df.drop(columns=['DOI']), reg)
        valid_, y_pred = md.predict_site(new_model, target_SMILES, df.drop(columns=['DOI']), classif=False)
        img            = visualize_regio_pred(target_SMILES, y_pred)
        v              = ("%03d" % (i,))
        i             += 1
        img.savefig(f"../Figures/Video/pred_{v}.png", bbox_inches='tight')


    path           = '../Figures/Video/'
    out_path       = '../Figures/Video/'
    out_video_name = f"predictions_random_training_{video_name}.mp4"
    out_video_path = os.path.join(out_path, out_video_name)

    pre_imgs       = os.listdir(path)
    pre_imgs       = list(pre_imgs)
    pre_imgs       = sorted(pre_imgs)
    
    pre_imgs_      = []
    for img_ in pre_imgs:
        if 'png' in img_:
            pre_imgs_.append(img_)
    pre_imgs       = pre_imgs_

    img = []
    for i in pre_imgs:
        i =  path + i 
        img.append(i)

    cv2_fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    frame = cv2.imread(img[0])
    size  = list(frame.shape)
    
    del size[2]
    size.reverse()

    video = cv2.VideoWriter(out_video_path,  cv2_fourcc, 48, size)

    for i in range(len(img)):
        #print(img[i])
        for _ in range(3):  # image duration
            video.write(cv2.imread(img[i])) 

    video.release() 

    return img_exp


def add_random_molecule(training_reactants, df_custom, target_SMILES):
    smiles_remaining   = df_custom.loc[df_custom['Reactant_SMILES'].isin(training_reactants) == False, 'Reactant_SMILES'].unique()
    smiles_remaining_  = []
    for smi in smiles_remaining:
        if smi != target_SMILES:
            smiles_remaining_.append(smi)
    training_reactants = list(training_reactants)
    training_reactants.append(smiles_remaining_[randrange(len(smiles_remaining_))])
    return training_reactants, len(smiles_remaining_)

