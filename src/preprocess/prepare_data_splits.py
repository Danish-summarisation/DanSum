"""
This script prepares subsets and train-test-val splits of the data.
"""

# %% LOAD PACKAGES
import pandas as pd
from datasets import Dataset, load_dataset

# %% LOAD DATA
ds_clean = load_dataset("csv", data_files="/work/69831/data/tok_ds_clean.csv")
ds_clean = ds_clean["train"]  # right format
df_clean = pd.DataFrame(ds_clean)  # pandas dataframe version

# %% --- MAKE 25K CLEAN 80-10-10 SUBSET SPLIT
df25k_clean = df_clean[:25000]  # make 25k clean subset
ds25k_clean = Dataset.from_pandas(df25k_clean)  # make dataset format
df25k_clean.to_csv("clean25k.csv")  # save csv

# %% create splits
test_len = round(len(ds25k_clean) / 10)  # test is 10%
val_len = round(len(ds25k_clean) / 10)  # validation is 10%

train25k_clean, test25k_clean = ds25k_clean.train_test_split(
    test_size=test_len
).values()  # absolute size specified
train25k_clean, val25k_clean = train25k_clean.train_test_split(
    test_size=val_len
).values()

# %% save train test and val CSV
train25k_clean.to_csv("train25k_clean.csv")
test25k_clean.to_csv("test25k_clean.csv")
val25k_clean.to_csv("val25k_clean.csv")

# %% --- MAKE ALL CLEAN 80-10-1 SUBSET SPLIT
test_len = round(len(ds_clean) / 10)  # test is 10%
val_len = round(len(ds_clean) / 100)  # validation is 1%
train_clean1, test_clean1 = ds_clean.train_test_split(
    test_size=test_len
).values()  # absolute size specified
train_clean1, val_clean1 = train_clean1.train_test_split(test_size=val_len).values()

# %% save train test and val CSV for all clean
train_clean1.to_csv("/work/69831/data/train_clean1.csv")
test_clean1.to_csv("/work/69831/data/test_clean1.csv")
val_clean1.to_csv("/work/69831/data/val_clean1.csv")
