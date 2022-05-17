# This script explores different aspects of the DaNewsroom dataset.
# %%
import random
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


# %% INSPECT
pd.set_option('display.max_colwidth', None) # show full entry
print(df_abs['text'].sample())



# %% COMPRESSION

print(df_abs['compression_bin'].value_counts())

high = df_abs.loc[df_abs['compression_bin'] == 'high']
medium = df_abs.loc[df_abs['compression_bin'] == 'medium']
low = df_abs.loc[df_abs['compression_bin'] == 'low']

# %%
print(high['summary'].sample(n=2))
print(medium['summary'].sample(n=2))
print(low['summary'].sample(n=2))


# %% CHECK IF TEXT CONTAINS SUMMARY

doubles = [summary for summary in df_abs['summary'] if summary in df_abs['text']]
print(len(doubles))

# %%

d = {'text': ['du er sød, yay!', 'hunden er død. åh nej. det var skidt.'], 'sum': ['han kan.', 'åh nej.']}
test = pd.DataFrame(data=d)
# %%
doubles = [summary for summary in test['sum'] if summary in test['text']]
len(doubles)

#doubles = [test.loc[test['text'].str.contains(summary, case=False)] for summary in test['sum']]
#doubles = [[text.loc[text['text']].str.contains(summary) for summary in text] for text in test['text']]
doubles = [[summary for summary in text['summary'] if text['summary'] in text['text']] for text in test]
len(doubles)




# %%
test['text']
# %%
from re import search
doubles = [summary for summary in test['sum'] if search(summary, test['text'])]
len(doubles)

