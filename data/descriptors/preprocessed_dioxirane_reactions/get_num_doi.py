import pandas as pd
df = pd.read_csv('df_bde.csv')
dois = df.DOI
dois = list(set(dois))
print(dois)
print(len(dois))
