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
df_dd = datasets.DatasetDict({"data":df_ds})

# tokenized_datasets = dd.map(preprocess_function, batched=True)

#1k subset
df1k = df[:1000]
df1k_ds = Dataset.from_pandas(df1k)
df1k_dd = datasets.DatasetDict({"data":df1k_ds})

# %% remove everything except text and summary
all = df_ds.remove_columns(['url', 'archive', 'title', 'date', 'density',
'coverage', 'compression', 'compression_bin', 'coverage_bin',
'density_bin', 'site', 'domain', 'text_len', 'valid', '__index_level_0__'])

all1k = df1k_ds.remove_columns(['url', 'archive', 'title', 'date', 'density',
'coverage', 'compression', 'compression_bin', 'coverage_bin',
'density_bin', 'site', 'domain', 'text_len', 'valid', '__index_level_0__'])

# %% TOKENIZE 1K SUBSET USING FUNCTION AND MT5
tokenizer = T5TokenizerFast.from_pretrained("google/mt5-small")
tok_1k = df1k_dd.map(preprocess_function, batched=True)


# %%
# turn back into dataset (not dict)
tok_1k_ds = tok_1k['data']

# IDA START HERE!:
# LOOK AT tok_1k_ds['input_ids']
# figure out the lengths (descriptive stats), nice plots, etc
# remove the crazy short ones
# input ids is the articles
# labels is the ref summaries

# %% add features to ds with the TOKENISED lengths of ref summary and text/article
tok_text_len = [len(text) for text in tok_1k_ds['input_ids']]
tok_1k_ds = tok_1k_ds.add_column("tok_text_len", tok_text_len)
tok_sum_len = [len(summary) for summary in tok_1k_ds['labels']]
tok_1k_ds = tok_1k_ds.add_column("tok_sum_len", tok_sum_len)

# %% check the stats
print("min sum:")
print(min((tok_1k_ds)['tok_sum_len']))
print("max sum:")
print(max((tok_1k_ds)['tok_sum_len']))
print("min text:")
print(min((tok_1k_ds)['tok_text_len']))
print("max text: ")
print(max((tok_1k_ds)['tok_text_len']))

# %% Inspect short summaries
sorted(tok_1k_ds['tok_sum_len'])
sorted(tok_1k_ds['tok_text_len'])

# %% check out the shorties
# filter based on text len:
[[x[0], x[1], x[2], x[3], x[4], x[5]] for x in zip(tok_1k_ds['archive'], tok_1k_ds['text'], tok_1k_ds['tok_text_len'], tok_1k_ds['summary'], tok_1k_ds['tok_sum_len'], tok_1k_ds['title']) if x[2] < 50]

# filter based on sum len:
[[x[0], x[1], x[2], x[3], x[4], x[5]] for x in zip(tok_1k_ds['archive'], tok_1k_ds['text'], tok_1k_ds['tok_text_len'], tok_1k_ds['summary'], tok_1k_ds['tok_sum_len'], tok_1k_ds['title']) if x[4] < 30]

# %% count the shorties
len([i for i in tok_1k_ds['tok_sum_len'] if i < 30])

# %%
# decide lower cutoff bounds for tokenized summary and text length:
sum_cut = 20
text_cut = 50

[i for i in tok_1k_ds['tok_sum_len'] if i < sum_cut]



# %% --- TOKENIZE 1k SUBSET WITH MT5
tokenizer_mt5 = T5TokenizerFast.from_pretrained("google/mt5-small")
inputs = [prefix + doc for doc in all1k["text"]] # articles (+ prefix)
model_inputs_mt5_1k = tokenizer_mt5(inputs, truncation=True)
with tokenizer_mt5.as_target_tokenizer():
        labels = tokenizer_mt5(all1k["summary"], truncation=True) # tokenized summaries
# getting IDs for token
model_inputs_mt5_1k["labels"] = labels["input_ids"]

# %% --- DO THINGS TO TOKENIZED SUBSET :D
mod_in = model_inputs_mt5_1k # specify the tokenized model input

#len(mod_in) # 3
#mod_in['labels']
#mod_in['input_ids']

# %% ----- TOKENIZE ALL USING MT5
tokenizer_mt5 = T5TokenizerFast.from_pretrained("google/mt5-small")
inputs = [prefix + doc for doc in all["text"]] # articles (+ prefix)
model_inputs_mt5_all = tokenizer_mt5(inputs, truncation=True)
with tokenizer_mt5.as_target_tokenizer():
        labels = tokenizer_mt5(all["summary"], truncation=True) # tokenized summaries
# getting IDs for token
model_inputs_mt5_all["labels"] = labels["input_ids"]

# %%----- TOKENIZE 1k SUBSET WITH daT5
tokenizer_dat5 = T5TokenizerFast.from_pretrained("sarakolding/daT5-base")
inputs = [prefix + doc for doc in all1k["text"]] # articles (+ prefix)
model_inputs_dat5 = tokenizer_dat5(inputs, truncation=True)

with tokenizer_dat5.as_target_tokenizer():
        labels = tokenizer_dat5(all1k["summary"], truncation=True) # tokenized summaries
    
# getting IDs for token
model_inputs_dat5["labels"] = labels["input_ids"]



# %% ----- Actually tokenize ALL
inputs = [prefix + doc for doc in all["text"]] # summarries (+ prefix)
model_inputs = tokenizer(inputs, truncation=True)

with tokenizer.as_target_tokenizer():
        labels = tokenizer(all["summary"], truncation=True) 
    
# getting IDs for token
model_inputs["labels"] = labels["input_ids"]



# %% TOKENIZE ON DATADICT SOMETHING SOMETHING
# mt5 tokenizer (fast) version
tokenizer = T5TokenizerFast.from_pretrained("google/mt5-base")
tok_dd = dd.map(preprocess_function, batched=True)

# %% tokenize all data (not split into sets...)
tokenizer = T5TokenizerFast.from_pretrained("google/mt5-base")
tok_all = all2.map(preprocess_function, batched=True)


# %% INSPECT ALL TOKENIZED DATA...
len(tok_all['labels'][0])
len(tok_all['input_ids'][0])

# %% look for doubles (??)
tok_all['doubles'] = [x[0] in x[1] for x in zip(tok_all['summary'], tok_all['text'])]
doubles = tok_all[tok_all['doubles'] == True]
uniqs = tok_all[tok_all['doubles'] == False]

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