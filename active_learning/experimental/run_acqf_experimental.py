smis = ["CC1(CCC[C@@]2([C@H]3CC(O[C@@]3(CC[C@@H]12)C)=O)C)C"]
products = [["O=C(O[C@@]1(CC[C@@]23[H])C)C[C@@H]1[C@@]3(C)C[C@@H](O)CC2(C)C",
             "O=C(O[C@@]1(CC[C@@]23[H])C)C[C@@H]1[C@@]3(C)CCC(C2(C)C)=O",
             "O=C(O[C@@]1(CC[C@@]23[H])C)C[C@@H]1[C@@]3(C)CC(CC2(C)C)=O"]]
sels = [None]
yields = [["25", "57", "18"]]
acqfs = ["random", "acqf_1", "acqf_2", "acqf_2-1", "acqf_3", "acqf_4", "acqf_4-1", "acqf_5", "acqf_6", "acqf_7", "acqf_8", "acqf_9", "acqf_10"]
n_runs=10
res="experimental"

b="1"
start='cold'
feat='custom'
n_repet="10"
db="1"
alpha="1"
thresh_corr="0.9"
n_est="250"
max_feat="0.5"
max_depth="10"
min_samples="3"
model="regression_rf"
strat="simple"
folder="preprocessed_reactions_no_unspec_no_intra_unnorm"
large = "False"
atom = "O"

import subprocess
for i in range(len(smis)):
    for acqf in acqfs:
        for j in range(n_runs):
            if sels[i]:
                result = subprocess.run(['python3', "main_experimental.py", "--reactant", smis[i],
                                "--products"]+products[i]+["--sels"]+sels[i]+["--acqf", acqf,
                                "--batch", b, "--start", start, "--n_repet", n_repet,
                                "--db", db, "--feat", feat, "--n_est", n_est, "--max_feats", max_feat,
                                "--max_depth", max_depth, "--min_samples_leaf", min_samples,
                                "--model", model, "--selection_strat", strat, "--res", res,
                                "--run", str(j), "--df_folder", folder, "--alpha", alpha,
                                "--thresh_corr", thresh_corr,"--large", large, "--atom", atom]).stdout  
            else:
                result = subprocess.run(['python3', "main_experimental.py", "--reactant", smis[i],
                                "--products"]+products[i]+["--yields"]+yields[i]+["--acqf", acqf,
                                "--batch", b, "--start", start, "--n_repet", n_repet,
                                "--db", db, "--feat", feat, "--n_est", n_est, "--max_feats", max_feat,
                                "--max_depth", max_depth, "--min_samples_leaf", min_samples,
                                "--model", model, "--selection_strat", strat, "--res", res,
                                "--run", str(j), "--df_folder", folder, "--alpha", alpha,
                                "--thresh_corr", thresh_corr,"--large", large, "--atom", atom]).stdout 



