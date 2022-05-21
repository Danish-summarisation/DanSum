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
import seaborn

# %% LOAD ALL ABS DATA
prefix = "summarize: " # prefix

#df = pd.read_json('gpu_files/abs_sums.json')
df = pd.read_csv('/work/69831/abs_sums.csv')
df_ds = Dataset.from_pandas(df) # make dataset format
df_dd = datasets.DatasetDict({"data":df_ds}) # make dataset dict format

#make 1k subset
df1k = df[:1000]
df1k_ds = Dataset.from_pandas(df1k)
df1k_dd = datasets.DatasetDict({"data":df1k_ds})

# %% ----- TOKENIZE USING FUNCTION AND MT5
tokenizer = T5TokenizerFast.from_pretrained("google/mt5-base") # choose tokenizer
#dd = df1k_dd #1k subset
dd = df_dd # all data

# %% tokenize chosen data
tok_dd = dd.map(preprocess_function, batched=True) # TOKENIZE IT!
tok_ds = tok_dd ['data'] # turn back into dataset (not dict)

# %% add features to ds with the TOKENISED lengths of ref summary and text/article
tok_text_len = [len(text) for text in tok_ds['input_ids']] # input ids is the articles
tok_ds = tok_ds.add_column("tok_text_len", tok_text_len)
tok_sum_len = [len(summary) for summary in tok_ds['labels']] # labels is the ref summaries
tok_ds = tok_ds.add_column("tok_sum_len", tok_sum_len)

# %% SAVE TOKENISED DS
tok_ds.to_csv('tok_ds_all.csv')
# %% to load it again:
from datasets import load_dataset
dataset = load_dataset('csv', data_files='/work/69831/tok_ds_all.csv')
tok_ds = dataset['train']
tok_df = pd.DataFrame(tok_ds) # pandas dataframe version

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
sorted(tok_ds['tok_sum_len'], reverse=True)[:370]
# ^^ with cutoff 128, we lose around 300 something summaries to truncation
# REMOVE?
sorted(tok_ds['tok_text_len'], reverse=True)

# %% COUNT SUMMARIES
len([i for i in tok_ds['tok_sum_len'] if i < 10])
# %% COUNT TEXTS
len([i for i in tok_ds['tok_text_len'] if i > 10000])

# %% PLOTS
# histogram of summary token length
seaborn.histplot(tok_ds['tok_sum_len'], color = "red")

# %% histogram of text token length
seaborn.histplot(tok_ds['tok_text_len'], color = "green")

# %% TIME TO FILTER!!!

# %% check nr for each domain
print(tok_df.groupby(['domain']).groups.keys())
print(len(tok_df.groupby(['domain']).groups.keys()))
print(tok_df.groupby(['domain']).count())

# %% Look at stats per domain
agg_df = tok_df.groupby('domain').agg({'tok_text_len': ['mean', 'min', 'max'], 
                                 'tok_sum_len': ['mean', 'min', 'max']})


# %%

dom_sample = tok_df.groupby('domain').apply(pd.DataFrame.sample, n=2).reset_index(drop=True)

pd.set_option('display.max_colwidth', None)
print([[x[0], x[1], x[2], x[3], x[4]] for x in zip(dom_sample['archive'], dom_sample['domain'], dom_sample['title'], dom_sample['summary'], dom_sample['tok_sum_len'])])



# %% --- DECIDE MIN SUMMARY CUTOFF
# 15 tokens ?
s_u_15 = [[x[0], x[1], x[2]] for x in zip(tok_ds['archive'], tok_ds['summary'], tok_ds['tok_sum_len']) if x[2] < 15]
#^^len = 25972
pd.set_option('display.max_colwidth', None)
print(s_u_15[2300:2330]) # look at some

# %%
s_10_15 = [[x[0], x[1], x[2]] for x in zip(tok_ds['archive'], tok_ds['summary'], tok_ds['tok_sum_len']) if x[2] < 15 and x[2] > 10]
#^^len = 14356
pd.set_option('display.max_colwidth', None)
print(s_10_15[4300:4330]) # look at some

# Ida suggests: set a minimum of 15 tokens, throw the rest!

# %% --- DECIDE MAX SUMMARY CUTOFF

s_o_128 = [[x[0], x[1], x[2]] for x in zip(tok_ds['archive'], tok_ds['summary'], tok_ds['tok_sum_len']) if x[2] > 128]
# ^^ 304 that are cut off by max length anyway

s_100_128 = [[x[0], x[1], x[2]] for x in zip(tok_ds['archive'], tok_ds['summary'], tok_ds['tok_sum_len']) if x[2] > 100 and x[2] < 128]
#^^ 1082

pd.set_option('display.max_colwidth', None)
#print(s_o_128[217:237]) # look at some
print(s_100_128[517:537])

# Ida suggests: set a maximum of 128 tokens, throw the rest!


# %% --- DECIDE MIN ARTICLE CUTOFF

t_u_25 = [[x[0], x[1], x[2]] for x in zip(tok_ds['archive'], tok_ds['text'], tok_ds['tok_text_len']) if x[2] < 25]
# ^^ 214

pd.set_option('display.max_colwidth', None)
#print(t_u_25[107:127])

t_25_40 = [[x[0], x[1], x[2]] for x in zip(tok_ds['archive'], tok_ds['text'], tok_ds['tok_text_len']) if x[2] > 25 and x[2] < 40]
# ^^ 389
#print(t_25_40[10:40])

t_40_100 = [[x[0], x[1], x[2]] for x in zip(tok_ds['archive'], tok_ds['text'], tok_ds['tok_text_len']) if x[2] > 40 and x[2] < 100]
# ^^ 6536
#print(t_40_100[10:40])

t_100_150 = [[x[0], x[1], x[2]] for x in zip(tok_ds['archive'], tok_ds['text'], tok_ds['tok_text_len']) if x[2] > 100 and x[2] < 150]
# ^^ 15015
#print(t_100_150[10:40])

t_150_200 = [[x[0], x[1], x[2]] for x in zip(tok_ds['archive'], tok_ds['text'], tok_ds['tok_text_len']) if x[2] > 150 and x[2] < 200]
# ^^ 12880
print(t_150_200[10:40])

# Ida suggests: set a minimum of 150 tokens, throw the rest!

# %% --- DECIDE MAX ARTICLE CUTOFF
pd.set_option('display.max_colwidth', None)

t_o_1024 = [[x[0], x[1], x[2]] for x in zip(tok_ds['archive'], tok_ds['text'], tok_ds['tok_text_len']) if x[2] > 1024]
# ^^ 38227 are truncated with current cutoff

t_3k_5k = [[x[0], x[1], x[2]] for x in zip(tok_ds['archive'], tok_ds['text'], tok_ds['tok_text_len']) if x[2] > 3000 and x[2] < 5000]
# ^^ 1461
#print(t_3k_5k[25:35])

t_5k_10k = [[x[0], x[1], x[2]] for x in zip(tok_ds['archive'], tok_ds['text'], tok_ds['tok_text_len']) if x[2] > 5000 and x[2] < 10000]
# ^^ 225
#print(t_5k_10k[35:45])

t_o_10k = [[x[0], x[1], x[2]] for x in zip(tok_ds['archive'], tok_ds['text'], tok_ds['tok_text_len']) if x[2] > 10000]
# ^^ 37
#print(t_o_10k[15:20])

t_o_30k = [[x[0], x[1], x[2]] for x in zip(tok_ds['archive'], tok_ds['text'], tok_ds['tok_text_len']) if x[2] > 30000]
# ^^ 8
#print(t_o_30k)

t_u_3k = [[x[0], x[1], x[2]] for x in zip(tok_ds['archive'], tok_ds['text'], tok_ds['tok_text_len']) if x[2] < 3000]
# ^^ 285479
#print(t_u_3k)

t_2k_3k = [[x[0], x[1], x[2]] for x in zip(tok_ds['archive'], tok_ds['text'], tok_ds['tok_text_len']) if x[2] > 2000 and x[2] < 3000]
# ^^ 5087

t_1k_2k = [[x[0], x[1], x[2]] for x in zip(tok_ds['archive'], tok_ds['text'], tok_ds['tok_text_len']) if x[2] > 1000 and x[2] < 2000]
# ^^ 33382
#print(t_1k_2k[965:985])

# Ida suggests: set a maximum of 1500 tokens, cut the rest!

# %% --- MAKE DATA SUBSET WITH TOKEN LIMITS IMPLEMENTED
# len(tok_df) is 287205 before changes
tok_df_clean = tok_df

# minimum summary length
tok_df_clean = tok_df_clean.loc[tok_df_clean['tok_sum_len'] >= 15]

# maximum summary length
tok_df_clean = tok_df_clean.loc[tok_df_clean['tok_sum_len'] <= 128]

# minimum article length
tok_df_clean = tok_df_clean.loc[tok_df_clean['tok_text_len'] >= 150]

# maximum article length
tok_df_clean = tok_df_clean.loc[tok_df_clean['tok_text_len'] <= 1500]

# make dataset format
tok_ds_clean = Dataset.from_pandas(tok_df_clean)

# length is 230012 after cleaning (with 15, 128, 150, 1500)

# %% save it
tok_ds_clean.to_csv('tok_ds_clean.csv')


# %% manually inspect content from certain sites
inspect = [[x[0], x[1], x[2], x[3], x[4], x[5], x[6]] for x in zip(tok_ds['archive'], tok_ds['text'], tok_ds['tok_text_len'], tok_ds['summary'], tok_ds['tok_sum_len'], tok_ds['title'], tok_ds['site']) if x[6] == 'www1.dr.dk']

version2 = [[x[0], x[1], x[2], x[3], x[4], x[5], x[6]] for x in zip(tok_ds['archive'], tok_ds['text'], tok_ds['tok_text_len'], tok_ds['summary'], tok_ds['tok_sum_len'], tok_ds['title'], tok_ds['site']) if x[6] == 'www.version2.dk']

inspect = [[x[0], x[1], x[2], x[3], x[4], x[5], x[6]] for x in zip(tok_ds['archive'], tok_ds['text'], tok_ds['tok_text_len'], tok_ds['summary'], tok_ds['tok_sum_len'], tok_ds['title'], tok_ds['site']) if x[6] == 'ekstern.videnskab.dk']

#tok_df.loc[tok_df['site_merged'] == 'ekstern.videnskab.dk']

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