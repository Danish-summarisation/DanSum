"""
This script  prepares subsets and train-test-val splits of the cleaned data.
"""

# %% LOAD PACKAGES
import pandas as pd
from datasets import Dataset, load_dataset
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
test_len = round(len(ds_clean) / 10)  # test is 10%
val_len = round(len(ds_clean) / 10)  # validation is 10%

# subsetting abstractive samples
df_abs = df_clean[df_clean['density_bin'] == 'abstractive']
df_abs = df_abs.drop(columns='__index_level_0__')
ds_abs = Dataset.from_pandas(df_abs)

# 10% of abstractive dataset ~200k
# test_len = round(len(abs) / 10)  # test is 10%
# val_len = round(len(abs) / 10)

# creating test and val splits
abs1, test_clean1 = ds_abs.train_test_split(
    test_size=test_len, seed=seed
).values()  # absolute size specified
train_clean2, val_clean1 = abs1.train_test_split(
    test_size=val_len, seed=seed
).values()

# converting to df
df_test = pd.DataFrame(test_clean1)
df_val = pd.DataFrame(val_clean1)

# creating train split by isolating parts of data that are not in either test or val splits
outer = df_clean[['text', 'summary']].merge(df_test[['text', 'summary']], how='outer', indicator=True)
anti_join = outer[(outer._merge=='left_only')].drop('_merge', axis=1)
outer1 = anti_join[['text', 'summary']].merge(df_val[['text', 'summary']], how='outer', indicator=True)
anti_join1 = outer1[(outer1._merge=='left_only')].drop('_merge', axis=1)

# merging other columns back on
df_train = anti_join1.merge(df_clean, how='left')

# converting back to dataset and shuffling
df_train = df_train.drop(columns='__index_level_0__')
train_clean1 = Dataset.from_pandas(df_train)

# shuffling training set?
train_clean1 = train_clean1.shuffle(seed=seed)

# %% save train test and val CSV for all clean
train_clean1.to_csv("/data-big-projects/danish-summarization-danewsroom/train_clean1.csv")
test_clean1.to_csv("/data-big-projects/danish-summarization-danewsroom/test_clean1.csv")
val_clean1.to_csv("/data-big-projects/danish-summarization-danewsroom/val_clean1.csv")
