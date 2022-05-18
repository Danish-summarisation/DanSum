# This script explores different aspects of the DaNewsroom dataset.
# %%
import random
from unittest import case
import numpy as np
import pandas as pd
import datasets
from datasets import Dataset
from sklearn.model_selection import train_test_split


# %% LOAD DATA

df = pd.read_json(path_or_buf = 'danewsroom/danewsroom.jsonl.gz', lines=True)


# %% INSPECT DATA STRUCTURE
print(type(df)) # pandas.core.frame.DataFrame
print(len(df)) # 1132734
print(df.keys())
#Index(['url', 'archive', 'title', 'date', 'text', 'summary', 'density',
# 'coverage', 'compression', 'compression_bin', 'coverage_bin','density_bin',
# 'site', 'domain', 'text_len', 'valid'], dtype='object')

print(df.shape) # (1132734, 16)
print(df.head())
print(df['density_bin'].value_counts()) #extractive = 615769, abstractive = 287205, mixed = 229760


print(df.info())

# %% SUBSET ABSTRACTIVE SUMMARIES
df_abs = df.loc[df['density_bin'] == 'abstractive']
len(df_abs) # 287205

# %% MAKE SMALL SUBSETS FOR TESTING MODEL SPEED
df1000 = df_abs[:1000]
df10000 = df_abs[:10000]

df1000.to_json(r'abs1000.json')
df10000.to_json(r'abs10000.json')

# %% SAVE ABSTRACTIVE SUBSET
df_abs.to_json(r'abs_sums.json')


# %% INSPECT
pd.set_option('display.max_colwidth', None) # show full entry
print(df_abs.sample())

# %% CHECK DIFFERENT SITES

# look at dif sites
# see conventions
# some of them might need title included in summary - some NOT?!





# %% COMPRESSION

print(df_abs['compression_bin'].value_counts())

high = df_abs.loc[df_abs['compression_bin'] == 'high']
medium = df_abs.loc[df_abs['compression_bin'] == 'medium']
low = df_abs.loc[df_abs['compression_bin'] == 'low']

# %%
print(high['summary'].sample(n=2))
print(medium['summary'].sample(n=2))
print(low['summary'].sample(n=2))


# %% DESCRIPTIVE STATS FOR ABSTRACTIVE SUBSET
df_abs = pd.read_json('gpu_files/abs_sums.json')
df_abs['doubles'] = [x[0] in x[1] for x in zip(df_abs['summary'], df_abs['text'])]
doubles = df_abs[df_abs['doubles'] == True]
uniqs = df_abs[df_abs['doubles'] == False]

# %%
print(len(uniqs)) # 287110
print(len(doubles)) # 95

# %% INSPECT DOUBLES
pd.set_option('display.max_colwidth', None) # show full entry
doubles.sample(5) # look at the doubles
#^^ OBS: most doubles are only ONE word summaries!!!???

# %% REMOVE THESE 
doubles

# %% MAKE REF SUMMARY TITLE + SUMMARY?
df_abs.sample(10)


# %% ------ DESCRIPTIVE STATS & DIAGNOSTICS FOR ALL ABSTRACTIVE SUMMARIES


# %% tokenize
from transformers import BertTokenizerFast, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")
tokenizerB = BertTokenizerFast.from_pretrained("Maltehb/danish-bert-botxo")