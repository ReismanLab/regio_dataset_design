from fcd_torch import FCD
import pandas as pd
import random
import numpy as np

df_sm = pd.read_csv('../smiles_csv/smiles_small.csv', header=None)
df_cm = pd.read_csv('../smiles_csv/smiles_complex.csv', header=None)

fcd = FCD()

smiles_sm = df_sm[0].to_list()
smiles_cm = df_cm[0].to_list()
smiles_all = smiles_sm + smiles_cm

print(f"FCD small to complex molecules: {fcd(smiles_sm, smiles_cm)}")
print(f"FCD small to small molecules: {fcd(smiles_sm, smiles_sm)}")  
print(f"FCD complex to complex molecules: {fcd(smiles_cm, smiles_cm)}")
print(f"FCD all to all molecules: {fcd(smiles_all, smiles_all)}")

# verification that the sampling is representative
print("Sanity Check v2")
print("Small molecules")
for n in [2, 5, 10, 20, 50]:
    fcds = []
    for _ in range(50): # repetitions:
        smiles_samples = random.sample(smiles_sm, n)
        smiles_complementary = list(set(smiles_sm) - set(smiles_samples))
        fcds.append(fcd(smiles_complementary, smiles_samples))
    print(f"distance to the small dataset for {n} selected molecule = {round(np.mean(fcds),1)} +/- {round(np.std(fcds),1)}")

print("Complex molecules")
for n in [2, 5, 10, 20, 50]:
    fcds = []
    for _ in range(50): # repetitions:
        smiles_samples = random.sample(smiles_cm, n)
        smiles_complementary = list(set(smiles_cm) - set(smiles_samples))
        fcds.append(fcd(smiles_complementary, smiles_samples))
    print(f"distance to the complex dataset for {n} selected molecule = {round(np.mean(fcds),1)} +/- {round(np.std(fcds),1)}")

print("All molecules")
for n in [2, 5, 10, 20, 50]:
    fcds = []
    for _ in range(50): # repetitions:
        smiles_samples = random.sample(smiles_all, n)
        smiles_complementary = list(set(smiles_all) - set(smiles_samples))
        fcds.append(fcd(smiles_complementary, smiles_samples))
    print(f"distance to the whole dataset for {n} selected molecule = {round(np.mean(fcds),1)} +/- {round(np.std(fcds),1)}")




# verification: n samples to the whole data (including these n samples)
print("Sanity Check v1")
for n in [2, 5, 10, 20, 50]:
    fcds = []
    for _ in range(50): # repetitions:
        smiles_samples = random.sample(smiles_all, n)
        fcds.append(fcd(smiles_all, smiles_samples))
    print(f"distance to the whole small dataset for {n} selected molecule = {round(np.mean(fcds),1)} +/- {round(np.std(fcds),1)}")

for n in [2, 5, 10, 20]:
    fcds = []
    for _ in range(50): # repetitions:
        smiles_samples = random.sample(smiles_cm, n)
        fcds.append(fcd(smiles_cm, smiles_samples))
    print(f"distance to the whole complex dataset for {n} selected molecule = {round(np.mean(fcds),1)} +/- {round(np.std(fcds),1)}")

for n in [2, 5, 10, 20, 50]:
    fcds = []
    for _ in range(50): # repetitions:
        smiles_samples = random.sample(smiles_sm, n)
        fcds.append(fcd(smiles_sm, smiles_samples))
    print(f"distance to the whole small dataset for {n} selected molecule = {round(np.mean(fcds),1)} +/- {round(np.std(fcds),1)}")

