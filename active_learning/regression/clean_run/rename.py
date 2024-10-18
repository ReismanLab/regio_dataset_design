import os
import sys

files = os.listdir(".")
files = [f for f in files if 'res_rf' in f]
print(files[:3])

for i, f in enumerate(files):
   old_name = f
   new_name = f.replace('acqf_1', 'acqf_1-db1-a2')
   if i < 3:
      print(old_name)
      print(new_name)
   os.rename(old_name, new_name)

