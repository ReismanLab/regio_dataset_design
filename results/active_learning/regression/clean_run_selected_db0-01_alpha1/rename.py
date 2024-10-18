import os
import sys

files = os.listdir(".")
files = [f for f in files if 'res_rf' in f]
files = [f for f in files if 'acqf_1_' in f]
print(files[:3])

for i, f in enumerate(files):
   old_name = f
   new_name = f.replace('acqf_1_', 'acqf_1-db0-01-a1_')
   if i < 3:
      print(old_name)
      print(new_name)
   os.rename(old_name, new_name)

