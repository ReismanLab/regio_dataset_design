import os
import shutil

import argparse
parser = argparse.ArgumentParser(description='Create file structure for acquisition function results')

parser.add_argument('--path',
                    help='Path to acquisition function pkl files')

args = parser.parse_args()
path = args.path

if not path:
    assert False, "Must pass path argument."

fnames = os.listdir(path)
fnames_dict = {}
for fname in fnames:
    if fname[-3:] != "pkl":
        continue
    if "res_rf" not in fname:
        continue
    print(fname)
    smi = [x for x in fname[:-4].split("_") if len(x) > 10][0]
    if "random" in fname:
        acqf = "random"
    else:
        acqf = "acqf_" + fname[:-4].split("_")[4]
    fnames_dict[fname] = [smi, acqf]

for fname in fnames_dict:
    smi, acqf = fnames_dict[fname][0], fnames_dict[fname][1]
    
    if not os.path.exists(f"{path}/{acqf}"):
        os.mkdir(f"{path}/{acqf}")

    if not os.path.exists(f"{path}/{acqf}/{smi}"):
        os.mkdir(f"{path}/{acqf}/{smi}")
    
    shutil.copyfile(f"{path}/{fname}", f"{path}/{acqf}/{smi}/{fname}")