"""
This script  prepares subsets and train-test-val splits of the cleaned data.
"""

# %% LOAD PACKAGES
import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset
seed = 22 # implement a seed to ensure replication

# %% LOAD DATA
ds_clean = load_dataset("csv", data_files="/data-big-projects/danish-summarization-danewsroom/tok_ds_clean.csv")
ds_clean = ds_clean["train"]  # right format
df_clean = pd.DataFrame(ds_clean)  # pandas dataframe version

# %% --- MAKE 25K CLEAN 80-10-10 SUBSET SPLIT
# df25k_clean = df_clean[:25000]  # make 25k clean subset
# ds25k_clean = Dataset.from_pandas(df25k_clean)  # make dataset format
# df25k_clean.to_csv("/data-big-projects/danish-summarization-danewsroom/clean25k.csv")  # save csv

# # %% create splits
# test_len = round(len(ds25k_clean) / 10)  # test is 10%
# val_len = round(len(ds25k_clean) / 10)  # validation is 10%

# train25k_clean, test25k_clean = ds25k_clean.train_test_split(
#     test_size=test_len, seed=seed
# ).values()  # absolute size specified
# train25k_clean, val25k_clean = train25k_clean.train_test_split(
#     test_size=val_len, seed=seed
# ).values()

# # %% save train test and val CSV
# train25k_clean.to_csv("/data-big-projects/danish-summarization-danewsroom/train25k_clean.csv")
# test25k_clean.to_csv("/data-big-projects/danish-summarization-danewsroom/test25k_clean.csv")
# val25k_clean.to_csv("/data-big-projects/danish-summarization-danewsroom/val25k_clean.csv")

# %% --- MAKE ALL CLEAN 89-10-1 SUBSET SPLIT
# 10% of whole dataset ~1 mil
# test_len = round(len(ds_clean) / 10)  # test is 10%
# val_len = round(len(ds_clean) / 10)  # validation is 10%

# subsetting abstractive samples
df_abs = df_clean[df_clean['density_bin'] == 'abstractive']
df_abs = df_abs.drop(columns='__index_level_0__')
ds_abs = Dataset.from_pandas(df_abs)

df_mix = df_clean[df_clean['density_bin'] == 'mixed']
df_mix = df_mix.drop(columns='__index_level_0__')
ds_mix = Dataset.from_pandas(df_mix)

df_ext = df_clean[df_clean['density_bin'] == 'extractive']
df_ext = df_ext.drop(columns='__index_level_0__')
ds_ext = Dataset.from_pandas(df_ext)

# 10% of abstractive dataset ~200k
test_len = round(len(df_abs) / 10)  # test is 10%
val_len = round(len(df_abs) / 10)

# creating test and val splits
abs_train, abs_test = ds_abs.train_test_split(
    test_size=test_len, seed=seed
).values()  # absolute size specified
abs_train, abs_val = abs_train.train_test_split(
    test_size=val_len, seed=seed
).values()

mix_train, mix_test = ds_mix.train_test_split(
    test_size=test_len, seed=seed
).values() 

ext_train, ext_test = ds_ext.train_test_split(
    test_size=test_len, seed=seed
).values() 

train = concatenate_datasets([abs_train, mix_train])
train = concatenate_datasets([train, ext_train])

# %% save train test and val CSV for all clean
train.to_csv("/data-big-projects/danish-summarization-danewsroom/train_all.csv")
abs_val.to_csv("/data-big-projects/danish-summarization-danewsroom/val_abstractive.csv")
abs_test.to_csv("/data-big-projects/danish-summarization-danewsroom/test_abstractive.csv")
mix_test.to_csv("/data-big-projects/danish-summarization-danewsroom/mix_abstractive.csv")
ext_test.to_csv("/data-big-projects/danish-summarization-danewsroom/ext_abstractive.csv")

