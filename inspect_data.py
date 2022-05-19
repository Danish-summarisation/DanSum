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
# Load data
train = Dataset.from_pandas(pd.read_csv("gpu_files/train_abs.csv", usecols=['text','summary'])) # training data
test = Dataset.from_pandas(pd.read_csv("gpu_files/test_abs.csv", usecols=['text','summary'])) # test data
val = Dataset.from_pandas(pd.read_csv("gpu_files/val_abs.csv", usecols=['text','summary'])) # validation data

dd = datasets.DatasetDict({"train":train,"validation":val,"test":test}) # make datasetdict format

# %% all
#all = Dataset.from_pandas(pd.read_json('gpu_files/abs_sums.json'), usecols=['text','summary'])
all = Dataset.from_pandas(df_abs)

# %% remove everything except text and summary
all2 = all.remove_columns(['url', 'archive', 'title', 'date', 'density', 'coverage', 'compression', 'compression_bin', 'coverage_bin', 'density_bin', 'site', 'domain', 'text_len', 'valid', 'doubles', '__index_level_0__'])

# %% prepare to tokenize
from transformers import BertTokenizerFast, T5TokenizerFast
from transformers import AutoTokenizer

prefix = "" # no prefix right now
max_input_length = 1024 # max text (article) max token length
max_target_length = 128 # max reference summary max token length

def preprocess_function(examples):
    # concatenate prefix and article into one input
    inputs = [prefix + doc for doc in examples["text"]]
    # tokenize the input + truncate to max input length
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True) 
    # tokenize the ref summary + truncate to max input length
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True) 
    
    # getting IDs for token
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

# %% TOKENIZE
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

# %%
