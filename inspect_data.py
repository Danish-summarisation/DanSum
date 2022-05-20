'''
This script explores different aspects of the DaNewsroom dataset.
All data = 1132734 pairs, extractive = 615769, abstractive = 287205, mixed = 229760
'''
# %% FUNCTION
def preprocess_function(examples):
    # concatenate prefix and article into one input
    inputs = [prefix + doc for doc in examples["text"]]

    # tokenize the input + truncate to max input length
    model_inputs = tokenizer(inputs, truncation=True) 

    # tokenize the ref summary + truncate to max input length
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], truncation=True) 
    
    # getting IDs for token
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

# %% LOAD PACKAGES
import random
from unittest import case
import numpy as np
import pandas as pd
import datasets
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, T5TokenizerFast
from transformers import AutoTokenizer

# %% LOAD ALL ABS DATA
prefix = "summarize: " # prefix

df = pd.read_json('gpu_files/abs_sums.json')
df_ds = Dataset.from_pandas(df) # make dataset format
df_dd = datasets.DatasetDict({"data":df_ds}) # make dataset dict format

#make 1k subset
df1k = df[:1000]
df1k_ds = Dataset.from_pandas(df1k)
df1k_dd = datasets.DatasetDict({"data":df1k_ds})

# %% ----- TOKENIZE USING FUNCTION AND MT5
tokenizer = T5TokenizerFast.from_pretrained("google/mt5-small") # choose tokenizer
#dd = df1k_dd #1k subset
dd = df_dd # all data

# %% tokenize chosen data
tok_dd = dd.map(preprocess_function, batched=True) # TOKENIZE IT!
tok_ds = tok_dd ['data'] # turn back into dataset (not dict)

# %% add features to ds with the TOKENISED lengths of ref summary and text/article
tok_text_len = [len(text) for text in tok_ds['input_ids']] # input ids is the articles
tok_1k_ds = tok_ds.add_column("tok_text_len", tok_text_len)
tok_sum_len = [len(summary) for summary in tok_1k_ds['labels']] # labels is the ref summaries
tok_1k_ds = tok_1k_ds.add_column("tok_sum_len", tok_sum_len)

# %% check the stats
print("min sum:")
print(min((tok_ds)['tok_sum_len']))
print("max sum:")
print(max((tok_ds)['tok_sum_len']))
print("min text:")
print(min((tok_ds)['tok_text_len']))
print("max text: ")
print(max((tok_ds)['tok_text_len']))

# %% Inspect short summaries
sorted(tok_ds['tok_sum_len'])
sorted(tok_ds['tok_text_len'])

# %% check out the shorties
# filter based on text len:
[[x[0], x[1], x[2], x[3], x[4], x[5]] for x in zip(tok_ds['archive'], tok_ds['text'], tok_ds['tok_text_len'], tok_ds['summary'], tok_ds['tok_sum_len'], tok_ds['title']) if x[2] < 50]

# filter based on sum len:
[[x[0], x[1], x[2], x[3], x[4], x[5]] for x in zip(tok_ds['archive'], tok_ds['text'], tok_ds['tok_text_len'], tok_ds['summary'], tok_ds['tok_sum_len'], tok_ds['title']) if x[4] < 30]

# %% count the shorties
len([i for i in tok_ds['tok_sum_len'] if i < 30])

# %%
# decide lower cutoff bounds for tokenized summary and text length:
sum_cut = 20
text_cut = 50

[i for i in tok_ds['tok_sum_len'] if i < sum_cut]




# %% ----------------------- OLDER STUFF BELOW

# %% CHECK IF IT MAKES SENSE TO INCLUDE TITLE FOR DIFFERENT SITES
# look at dif sites
# see conventions
# some of them might need title included in summary - some NOT?!


# %% ------------------------ inspection stuff

# %% INSPECT DATA STRUCTURE
print(type(df)) # pandas.core.frame.DataFrame
print(len(df)) # 1132734
print(df.keys())
print(df.shape) # (1132734, 16)
print(df.head())
print(df['density_bin'].value_counts())
print(df.info())

# %% INSPECT RANDOM ENTRY
pd.set_option('display.max_colwidth', None) # show full entry
print(df.sample())

# %% INSPECT COMPRESSION
print(df['compression_bin'].value_counts())

# make new DFs keeping only certain compressions
high = df.loc[df['compression_bin'] == 'high']
medium = df.loc[df['compression_bin'] == 'medium']
low = df.loc[df['compression_bin'] == 'low']

# %% SAMPLE SUMMARIES OF DIFFERENT COMPRESSIONS
print('--- HIGH:')
print(high['summary'].sample(n=2))
print('--- MEDIUM:')
print(medium['summary'].sample(n=2))
print('--- LOW:')
print(low['summary'].sample(n=2))

# %% --- DESCRIPTIVE STATS FOR ABSTRACTIVE SUBSET
df['doubles'] = [x[0] in x[1] for x in zip(df['summary'], df['text'])]
doubles = df[df['doubles'] == True]
uniqs = df[df['doubles'] == False]

# %% CHECK
print(len(uniqs)) # 287110
print(len(doubles)) # 95

# %% INSPECT DOUBLES
pd.set_option('display.max_colwidth', None) # show full entry
doubles['summary'].sample(5) # look at the doubles
#^^ OBS: most doubles are only ONE word summaries!!!???