'''
This script prepares train-test-val splits of the data.
'''

# %% LOAD PACKAGES
import pandas as pd
import datasets
from datasets import Dataset
from sklearn.model_selection import train_test_split
from datasets import load_dataset

# %% --- 25K clean 89-10-1 subset split
ds_clean = load_dataset('csv', data_files='/work/69831/data/tok_ds_clean.csv')
ds_clean = ds_clean['train'] # important
df_clean = pd.DataFrame(ds_clean) # pandas dataframe version
df25k_clean = df_clean[:10] # make 25k clean subset
ds25k_clean = Dataset.from_pandas(df25k_clean) # make dataset format
df25k_clean.to_csv("clean10.csv")
# %% split 89-10-1
test_len = round(len(ds25k_clean) / 10) # test is 10 %. 5000 for 50k.
val_len = round(len(ds25k_clean) / 100) # validation is 1%. 500 for 50k.

train25k_clean1, test25k_clean1 = ds25k_clean.train_test_split(test_size=test_len).values() # absolute size specified
train25k_clean1, val25k_clean1 = train25k_clean1.train_test_split(test_size=val_len).values()

# %% save train test and val CSV for 25k clean
train25k_clean1.to_csv("train25k_clean1.csv") # len 44500
test25k_clean1.to_csv("test25k_clean1.csv") # len 5000
val25k_clean1.to_csv("val25k_clean1.csv") # len 500

# %% - --- ALL clean 89-10-1 subset split (1% val)
ds_clean = load_dataset('csv', data_files='/work/69831/data/tok_ds_clean.csv')
ds_clean = ds_clean['train'] # important
# %% split ALL clean 89-10-1
test_len = round(len(ds_clean) / 10) # test is 10 %. 23001 for all.
val_len = round(len(ds_clean) / 100) # validation is 1%. 2300 for all.
train_clean1, test_clean1 = ds_clean.train_test_split(test_size=test_len).values() # absolute size specified
train_clean1, val_clean1 = train_clean1.train_test_split(test_size=val_len).values()
# len train_clean1 = 204711
# %% save train test and val CSV for 25k clean
train_clean1.to_csv("/work/69831/data/train_clean1.csv") # len 204711
test_clean1.to_csv("/work/69831/data/test_clean1.csv") # len 23001
val_clean1.to_csv("/work/69831/data/val_clean1.csv") # len 2300


# %% --- MAKE 25K SUBSET FROM CLEANED DATA
ds_clean = load_dataset('csv', data_files='/work/69831/tok_ds_clean.csv')
df_clean = pd.DataFrame(ds_clean['train']) # pandas dataframe version
df25k_clean = df_clean[:25000] # make 50k clean subset
ds25k_clean = Dataset.from_pandas(df25k_clean) # make dataset format

# %% make 25k clean train test val splits
train25k_clean, test25k_clean = ds25k_clean.train_test_split(test_size=2500).values() # 2500 = absolute size specified (out of 25k)
train25k_clean, val25k_clean = train25k_clean.train_test_split(test_size=2500).values()

# %% save train test and val CSV for 25k clean
train25k_clean.to_csv("train25k_clean.csv")
test25k_clean.to_csv("test25k_clean.csv")
val25k_clean.to_csv("val25k_clean.csv")

# %% make splits for ALL clean
ds_clean = ds_clean['train']
len(ds_clean) #  230012
test_len = round(len(ds_clean) / 10) # 23001 = 10 % for the test set (and also val)

train_clean, test_clean = ds_clean.train_test_split(test_size=test_len).values() # absolute size specified
train_clean, val_clean = train_clean.train_test_split(test_size=test_len).values()

# %% save split csvs for ALL clean
train_clean.to_csv("data/train_clean.csv")
test_clean.to_csv("data/test_clean.csv")
val_clean.to_csv("data/val_clean.csv")


# --- @IDA: RUN UNTIL HERE, SAVE FILES, DOUBLE CHECK SIZE


# %% ----- MAKE 25K SUBSET OF UNCLEANED DATA
df_unclean = pd.read_json("C:/Users/idaba/OneDrive/Dokumenter/COGNITIVE_SCIENCE/data_science/data-science-exam_BACKUP/gpu_files/abs_sums.json") # load all abs data (287205 pairs)
df25k_unclean = df_unclean[:25000] # make the subset
ds25k_unclean = Dataset.from_pandas(df25k_unclean) # make dataset format

# %% make train test val splits
train25k_unclean, test25k_unclean = ds25k_unclean.train_test_split(test_size=2500).values() # 2500 = absolute size specified for 25k
train25k_unclean, val25k_unclean = train25k_unclean.train_test_split(test_size=2500).values()

# %% save train test and val CSV for 25k UNCLEAN
train25k_unclean.to_csv("train25k_unclean.csv")
test25k_unclean.to_csv("test25k_unclean.csv")
val25k_unclean.to_csv("val25k_unclean.csv")


# %% ------- MAKE 50K SUBSET (OF UNCLEANED DATA):
df_abs = pd.read_json("C:/Users/idaba/OneDrive/Dokumenter/COGNITIVE_SCIENCE/data_science/data-science-exam_BACKUP/gpu_files/abs_sums.json") # load all abs data (287205 pairs)
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


# %% CHECK IT!
#train25k = Dataset.from_pandas(pd.read_csv("train25k_clean.csv", usecols=['text','summary'])) # training data

train_test = Dataset.from_pandas(pd.read_csv('/work/69831/data-science-exam/gpu_files/train_clean.csv', usecols=['text','summary'])) # training data

# %%
train1k = Dataset.from_pandas(pd.read_csv("train1k.csv", usecols=['text','summary'])) # training data

# %%
train25test = train25k['train']