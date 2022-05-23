'''
This script prepares train-test-val splits of the data.
'''

# %% LOAD PACKAGES
import pandas as pd
import datasets
from datasets import Dataset
from sklearn.model_selection import train_test_split
from datasets import load_dataset

# %% MAKE 50K SUBSET FROM CLEANED DATA
ds_clean = load_dataset('csv', data_files='/work/69831/tok_ds_clean.csv')
df_clean = pd.DataFrame(ds) # pandas dataframe version
df50k_clean = df_clean[:50000] # make 50k clean subset
ds50k_clean = Dataset.from_pandas(df50k_clean) # make dataset format

# %% make 50k clean train test val splits
train50k_clean, test50k_clean = ds50k_clean.train_test_split(test_size=5000).values() # 5000 = absolute size specified (out of 50k)
train50k_clean, val50k_clean = train50k_clean.train_test_split(test_size=5000).values()

# %% save train test and val CSV for 50k clean
train50k_clean.to_csv("data/train50k_clean.csv")
test50k_clean.to_csv("data/test50k_clean.csv")
val50k_clean.to_csv("data/val50k_clean.csv")

# %% make splits for ALL clean
len(ds_clean) # test
test_len = len(ds_clean) / 10 # 10 % for the test set (and also val)

train_clean, test_clean = ds_clean.train_test_split(test_size=test_len).values() # 5000 = absolute size specified (out of 50k)
train_clean, val_clean = train_clean.train_test_split(test_size=test_len).values()

# %% save split csvs for ALL clean
train_clean.to_csv("data/train_clean.csv")
test_clean.to_csv("data/test_clean.csv")
val_clean.to_csv("data/val_clean.csv")

# --- @IDA: RUN UNTIL HERE, SAVE FILES, DOUBLE CHECK SIZE

# %% ------- MAKE 50K SUBSET (OF UNCLEANED DATA):
df_abs = pd.read_json('gpu_files/abs_sums.json') # load all abs data (287205 pairs)
df50k = df_abs[:50000] # make the subset
abs50kds = Dataset.from_pandas(df50k) # make dataset format

# %% make train test val splits
train50k, test50k = abs50kds.train_test_split(test_size=5000).values() # 5000 = absolute size specified
train50k, val50k = train50k.train_test_split(test_size=5000).values()

# %% CHECK SPLIT SIZES
print(len(train50k))
print(len(test50k))
print(len(val50k))

# %% SAVE SPLITS
train50k.to_csv("data/train50k.csv")
test50k.to_csv("data/test50k.csv")
val50k.to_csv("data/val50k.csv")
