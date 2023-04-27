"""
This script  prepares subsets and train-test-val splits of the cleaned data.
"""

import pandas as pd
from datasets import Dataset, concatenate_datasets, load_dataset

seed = 22  # implement a seed to ensure replication

ds_clean = load_dataset(
    "csv",
    data_files="/data-big-projects/danish-summarization-danewsroom/tok_ds_clean_.csv",
)
ds_clean = ds_clean["train"] 
df_clean = pd.DataFrame(ds_clean) 

# create a column that denotes whether both the article and the reference summary passed quality checks
df_clean["passed"] = (df_clean["passed_quality"] == True) & (
    df_clean["passed_quality_sum"] == True
)

# 89-10-10 split
# subsetting abstractive samples
df_abs = df_clean[df_clean["density_bin"] == "abstractive"]
df_abs = df_abs.drop(columns="__index_level_0__")
ds_abs = Dataset.from_pandas(df_abs)

df_mix = df_clean[df_clean["density_bin"] == "mixed"]
df_mix = df_mix.drop(columns="__index_level_0__")
ds_mix = Dataset.from_pandas(df_mix)

df_ext = df_clean[df_clean["density_bin"] == "extractive"]
df_ext = df_ext.drop(columns="__index_level_0__")
ds_ext = Dataset.from_pandas(df_ext)

# 10% of abstractive dataset
test_len = round(len(df_abs) / 10)  # test is 10%
val_len = round(len(df_abs) / 10)

# creating test and val splits
abs_train, abs_test = ds_abs.train_test_split(
    test_size=test_len, seed=seed
).values()  # absolute size specified
abs_train, abs_val = abs_train.train_test_split(test_size=val_len, seed=seed).values()

mix_train, mix_test = ds_mix.train_test_split(test_size=test_len, seed=seed).values()

ext_train, ext_test = ds_ext.train_test_split(test_size=test_len, seed=seed).values()

train = concatenate_datasets([abs_train, mix_train])
train = concatenate_datasets([train, ext_train])

# %% save train test and val 
train.to_csv("/data-big-projects/danish-summarization-danewsroom/train_all_.csv")
abs_val.to_csv(
    "/data-big-projects/danish-summarization-danewsroom/val_abstractive_.csv"
)
abs_test.to_csv(
    "/data-big-projects/danish-summarization-danewsroom/test_abstractive_.csv"
)
mix_test.to_csv("/data-big-projects/danish-summarization-danewsroom/test_mixed_.csv")
ext_test.to_csv(
    "/data-big-projects/danish-summarization-danewsroom/test_extractive_.csv"
)
