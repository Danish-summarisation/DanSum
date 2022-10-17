'''
This script preprocesses the data from the DaNewsroom data sets.
Applying cutoffs, filtering out the abstractive data and saving the data files.
'''
# OBS currently only with a 100k subset due to memory failing on my local laptop. But should work on all by removing small part of line 36
# %% Load modules
import pandas as pd
from datasets import Dataset, load_dataset
import datasets
from transformers import BertTokenizerFast, T5TokenizerFast
from transformers import AutoTokenizer
import seaborn
import numpy as np

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

# %% Load raw data
ds = load_dataset("csv", data_files="../../../danewsroom.csv")
ds_abstractive = ds["train"]

# %% Filter by density
df_abstractive = pd.DataFrame(ds_abstractive)
df_abstractive = df_abstractive[df_abstractive.density <= 1.5] 
print(len(df_abstractive)) # 287205
df_abstractive.to_csv("df_abstractive.csv")  # save csv
# Keep a raw subset of 25k? ?? But how when stuff is removed
# maybe not needed since we will only train on the cleaned data right?

# %% Clean by filter token article/summary length cutoffs
df_abstractive = pd.read_csv('df_abstractive.csv')

# %% Convert data format
#make 1k subset
df1k = df_abstractive[:25000]
df1k_ds = Dataset.from_pandas(df1k)
df1k_dd = datasets.DatasetDict({"data":df1k_ds})

# %% Load tokenizer
tokenizer = T5TokenizerFast.from_pretrained("google/mt5-small") # choose tokenizer
prefix = "summarize: " # prefix

# %% tokenize chosen data
tok_dd = df1k_dd.map(preprocess_function, batched=True) 
tok_ds = tok_dd ['data'] # turn back into dataset (not dict)

# %% add features to ds with the TOKENISED lengths of ref summary and text/article
tok_text_len = [len(text) for text in tok_ds['input_ids']] # input ids is the articles
tok_ds = tok_ds.add_column("tok_text_len", tok_text_len)
tok_sum_len = [len(summary) for summary in tok_ds['labels']] # labels is the ref summaries
tok_ds = tok_ds.add_column("tok_sum_len", tok_sum_len)

# %% check the stats
print("min sum:")
print(min((tok_ds)['tok_sum_len']))
print("max sum:")
print(max((tok_ds)['tok_sum_len']))
print("min text:")
print(min((tok_ds)['tok_text_len']))
print("max text: ")
print(max((tok_ds)['tok_text_len']))

# %% Make plot with summary length and text length to define cutoffs
# histogram of summary token length
seaborn.histplot(tok_ds['tok_sum_len'], color = "red")
sum_mean = np.mean(tok_ds['tok_sum_len'])
sum_sd = np.std(tok_ds['tok_sum_len'])
# used to remove lengths less or more than two standard deviations from the mean 
min_sum_len = sum_mean - 2*sum_sd #but this becomes negative though...
max_sum_len = sum_mean + 2*sum_sd

# %% histogram of text token length
seaborn.histplot(tok_ds['tok_text_len'], color = "green")
sum_mean = np.mean(tok_ds['tok_text_len'])
sum_sd = np.std(tok_ds['tok_text_len'])
# Defining cutoffs
min_text_len = sum_mean-2*sum_sd # becomes negative though...
max_text_len = sum_mean+2*sum_sd

# %% Filtering based on cutoffs
tok_df = pd.DataFrame(tok_ds) # pandas dataframe version
tok_df_clean = tok_df
# minimum summary length
tok_df_clean = tok_df_clean.loc[tok_df_clean['tok_sum_len'] >= min_sum_len]
# maximum summary length
tok_df_clean = tok_df_clean.loc[tok_df_clean['tok_sum_len'] <= max_sum_len]
# minimum article length
tok_df_clean = tok_df_clean.loc[tok_df_clean['tok_text_len'] >= min_text_len]
# maximum article length
tok_df_clean = tok_df_clean.loc[tok_df_clean['tok_text_len'] <= max_text_len]
# make dataset format
tok_ds_clean = Dataset.from_pandas(tok_df_clean)
# length should be 230012 after cleaning (with 15, 128, 150, 1500)

# %% Plot the newly cut data summary lengths
seaborn.histplot(tok_df_clean['tok_sum_len'], color = "green")

# %% Plot the newly cut data text lengths
seaborn.histplot(tok_df_clean['tok_text_len'], color = "green")

'''
Quality filter is from the following github: https://github.com/centre-for-humanities-computing/danish-foundation-models/blob/b16765c065818f9d4b162d2b2ab9b3dae7d252ea/src/dfm/cleaning/quality_filter.py
'''
# %% Aplying quality filter
# NB: Have the quality_filter.py script in your working directory
from quality_filter import QualityFilter
qf = QualityFilter(
        min_stop_words = 2,
        mean_word_length = (3, 10),
        doc_length = (10, 100_000),
        alpha_ratio = 0.6,
        duplicate_lines_chr_fraction = 0.4,
        duplicate_paragraph_chr_fraction= 0.4,
        top_ngram_chr_fraction_thresholds = [0.20, 0.18, 0.16],
        top_ngram_chr_fraction_range = (2, 4),
        top_ngram_min_count = 3,
        duplicate_n_gram_fraction_thresholds = [
            0.25,
            0.24,
            0.23,
            0.22,
            0.21,
            0.20,
        ],
        ignore_filters = ["duplicate_ngram_chr_fraction", "top_ngram_chr_fraction", "line_bullets_or_ellipsis", "detect_language", "short_long_sentece"],
        )
filter_to_ignore = ["doc_length", 
                    "alpha_ratio", 
                    "symbol_2_word_ellipsis", 
                    "duplicate_lines_chr_fraction", 
                    "top_ngram_chr_fraction", 
                    "duplicate_ngram_chr_fraction", 
                    "detect_language", 
                    "stop_word", 
                    "mean_word_length", 
                    "line_bullets_or_ellipsis"]
qf_sum = QualityFilter(
        min_stop_words = 2,
        mean_word_length = (3, 10),
        doc_length = (10, 100_000),
        alpha_ratio = 0.6,
        duplicate_lines_chr_fraction = 0.5,
        duplicate_paragraph_chr_fraction= 0.6,
        top_ngram_chr_fraction_thresholds = [0.20, 0.18, 0.16],
        top_ngram_chr_fraction_range = (2, 4),
        top_ngram_min_count = 3,
        duplicate_n_gram_fraction_thresholds = [
            0.25,
            0.24,
            0.23,
            0.22,
            0.21,
            0.20,
        ],
        ignore_filters = filter_to_ignore,
        )
# %% 
# Lists of texts
texts = tok_ds_clean['text']
summaries = tok_ds_clean['summary']
filtered = qf.describe_filter(texts) # giver jer hvilke som bliver filtreret fra og af hvilket filter
filtered_sum = qf_sum.describe_filter(summaries) 
passed_quality = [None] * len(texts)
filter = [None] * len(texts) # empty list
passed_quality_sum = [None] * len(texts)
filter_sum = [None] * len(texts) # empty list
for n, i in enumerate(texts):
    result = next(filtered)
    result_sum = next(filtered_sum)
    if  result == "passed filters": 
            passed_quality[n] = True
            filter[n] = "nan"
    else:
            passed_quality[n] = False
            filter[n] = result
    # same for summaries
    if result_sum == "passed filters":
            passed_quality_sum[n] = True
            filter_sum[n] = "nan"
    else:
            passed_quality_sum[n] = False
            filter_sum[n] = result_sum
tok_ds_clean['passed_quality'] = passed_quality
tok_ds_clean['filter'] = filter
tok_ds_clean['passed_quality_sum'] = passed_quality_sum
tok_ds_clean['filter_sum'] = filter_sum
# %% Saving it
tok_ds_clean.to_csv('/data/danish_summarization_danewsroom/tok_ds_clean.csv')

