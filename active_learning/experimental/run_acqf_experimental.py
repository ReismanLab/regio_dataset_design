smis = ["C[C@@H]1CC[C@H]2C(C)([C@H]3C[C@@]12CC[C@]3(O)C)C"]
products = [["C[C@]1(O)CC[C@H]2C([C@H]3C[C@@]12CC[C@@]3(C)O)(C)C"]]
sels = [["100"]]
acqfs = ["acqf_9"]
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
            result = subprocess.run(['python3', "main_experimental.py", "--reactant", smis[i],
                                "--products"]+products[i]+["--sels"]+sels[i]+["--acqf", acqf,
                                "--batch", b, "--start", start, "--n_repet", n_repet,
                                "--db", db, "--feat", feat, "--n_est", n_est, "--max_feats", max_feat,
                                "--max_depth", max_depth, "--min_samples_leaf", min_samples,
                                "--model", model, "--selection_strat", strat, "--res", res,
                                "--run", str(j), "--df_folder", folder, "--alpha", alpha,
                                "--thresh_corr", thresh_corr,"--large", large, "--atom", atom]).stdout  


