'''
This script explores different aspects of the DaNewsroom dataset.
'''
# %% LOAD PACKAGES
import random
from unittest import case
import numpy as np
import pandas as pd
import datasets
from datasets import Dataset
from sklearn.model_selection import train_test_split

# %% LOAD DATA
# all data (1132734 pairs, extractive = 615769, abstractive = 287205, mixed = 229760)
#df = pd.read_json(path_or_buf = 'danewsroom/danewsroom.jsonl.gz', lines=True)

# abstractive data only (287205 pairs)
df = pd.read_json('gpu_files/abs_sums.json') 

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

# %% CHECK DIFFERENT SITES

# look at dif sites
# see conventions
# some of them might need title included in summary - some NOT?!

# %% INSPECT COMPRESSION
print(df['compression_bin'].value_counts())

high = df.loc[df['compression_bin'] == 'high']
medium = df.loc[df['compression_bin'] == 'medium']
low = df.loc[df['compression_bin'] == 'low']

# compression doesn't necces

# %% SAMPLE SUMMARIES OF DIFFERENT COMPRESSIONS
print('--- HIGH:')
print(high['summary'].sample(n=2))
print('--- MEDIUM:')
print(medium['summary'].sample(n=2))
print('--- LOW:')
print(low['summary'].sample(n=2))

# %%


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

# %% REMOVE THESE 
doubles

# %% MAKE REF SUMMARY TITLE + SUMMARY?



# %% ------ DESCRIPTIVE STATS & DIAGNOSTICS FOR ALL ABSTRACTIVE SUMMARIES
#all = Dataset.from_pandas(pd.read_json('gpu_files/abs_sums.json'), usecols=['text','summary'])
all = Dataset.from_pandas(df)

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
