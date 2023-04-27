"""
This script preprocesses the data from the DaNewsroom data sets.
Applying cutoffs, filtering out the abstractive data and saving the data files.

Quality filter is from the following github: https://github.com/centre-for-humanities-computing/danish-foundation-models/blob/b16765c065818f9d4b162d2b2ab9b3dae7d252ea/src/dfm/cleaning/quality_filter.py
"""
import datasets
import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from ftfy import fix_text
from transformers import AutoTokenizer, T5TokenizerFast
from quality_filter import QualityFilter


def preprocess_function(examples):
    # concatenate prefix and article into one input
    inputs = [doc for doc in examples["text"]]

    # tokenize the input + truncate to max input length
    model_inputs = tokenizer(inputs, truncation=True)

    # tokenize the ref summary + truncate to max input length
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], truncation=True)

    # getting IDs for token
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


ds = load_dataset(
    "csv",
    data_files={
        "train": "/data-big-projects/danish-summarization-danewsroom/danewsroom.csv"
    },
)

ds_train = ds["train"]

# Filter by density
df_abstractive = pd.DataFrame(ds_train)
df_abstractive = df_abstractive[df_abstractive.density <= 1.5]

# Convert data format
ds_abstractive = Dataset.from_pandas(df_abstractive)
dd_abstractive = datasets.DatasetDict({"data": ds_abstractive})

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/mt5-small") 

t_sums = [fix_text(i) for i in ds_train["summary"]]
t_texts = [fix_text(i) for i in ds_train["text"]]
ds_train = ds_train.add_column("summary_fix", t_sums)
ds_train = ds_train.add_column("text_fix", t_texts)
ds_train = ds_train.remove_columns(["summary", "text"])
ds_train = ds_train.rename_column("summary_fix", "summary")
ds_train = ds_train.rename_column("text_fix", "text")

# tokenize
tok_dd = ds_train.map(preprocess_function, batched=True)
tok_ds = tok_dd  # ["data"]  # turn back into dataset (not dict)

# %% add features to ds with the tokenised lengths of ref summary and text/article
tok_text_len = [len(text) for text in tok_dd["input_ids"]] 
tok_ds = tok_dd.add_column("tok_text_len", tok_text_len)
tok_sum_len = [
    len(summary) for summary in tok_ds["labels"]
]  # labels is the ref summaries
tok_ds = tok_ds.add_column("tok_sum_len", tok_sum_len)

min_sum_len = np.quantile(tok_ds["tok_sum_len"], 0.02)
max_sum_len = np.quantile(tok_ds["tok_sum_len"], 0.98)

min_text_len = np.quantile(tok_ds["tok_text_len"], 0.02)
max_text_len = np.quantile(tok_ds["tok_text_len"], 0.98)

# %% Filtering based on cutoffs
tok_df = pd.DataFrame(tok_ds) 

# minimum summary length
tok_df_clean = tok_df.loc[tok_df["tok_sum_len"] >= min_sum_len]
# maximum summary length
tok_df_clean = tok_df_clean.loc[tok_df_clean["tok_sum_len"] <= max_sum_len]
# minimum article length
tok_df_clean = tok_df_clean.loc[tok_df_clean["tok_text_len"] >= min_text_len]
# maximum article length
tok_df_clean = tok_df_clean.loc[tok_df_clean["tok_text_len"] <= max_text_len]
# make dataset format
# tok_df_clean = tok_df_clean.drop(columns='__index_level_0__')
tok_ds_clean = Dataset.from_pandas(tok_df_clean)

# apply filter

qf = QualityFilter(
    min_stop_words=2,
    mean_word_length=(3, 10),
    doc_length=(10, 100_000),
    alpha_ratio=0.6,
    duplicate_lines_chr_fraction=0.4,
    duplicate_paragraph_chr_fraction=0.4,
    top_ngram_chr_fraction_thresholds=[0.20, 0.18, 0.16],
    top_ngram_chr_fraction_range=(2, 4),
    top_ngram_min_count=3,
    duplicate_n_gram_fraction_thresholds=[
        0.25,
        0.24,
        0.23,
        0.22,
        0.21,
        0.20,
    ],
    ignore_filters=[
        "duplicate_ngram_chr_fraction",
        "top_ngram_chr_fraction",
        "line_bullets_or_ellipsis",
        "detect_language",
        "short_long_sentece",
    ],
)
filter_to_ignore = [
    "doc_length",
    "alpha_ratio",
    "symbol_2_word_ellipsis",
    "duplicate_lines_chr_fraction",
    "top_ngram_chr_fraction",
    "duplicate_ngram_chr_fraction",
    "detect_language",
    "stop_word",
    "mean_word_length",
    "line_bullets_or_ellipsis",
]
qf_sum = QualityFilter(
    min_stop_words=2,
    mean_word_length=(3, 10),
    doc_length=(10, 100_000),
    alpha_ratio=0.6,
    duplicate_lines_chr_fraction=0.5,
    duplicate_paragraph_chr_fraction=0.6,
    top_ngram_chr_fraction_thresholds=[0.20, 0.18, 0.16],
    top_ngram_chr_fraction_range=(2, 4),
    top_ngram_min_count=3,
    duplicate_n_gram_fraction_thresholds=[
        0.25,
        0.24,
        0.23,
        0.22,
        0.21,
        0.20,
    ],
    ignore_filters=filter_to_ignore,
)

# Lists of texts
texts = tok_ds_clean["text"]
summaries = tok_ds_clean["summary"]
filtered = qf.describe_filter(
    texts
) 

filtered_sum = qf_sum.describe_filter(summaries)
passed_quality = [None] * len(texts)
filter = [None] * len(texts)  # empty list
passed_quality_sum = [None] * len(texts)
filter_sum = [None] * len(texts)  # empty list

for n, i in enumerate(texts):
    result = next(filtered)
    result_sum = next(filtered_sum)
    if result == "passed filters":
        passed_quality[n] = True
        filter[n] = "nan"
    else:
        passed_quality[n] = False
        filter[n] = result

    if result_sum == "passed filters":
        passed_quality_sum[n] = True
        filter_sum[n] = "nan"
    else:
        passed_quality_sum[n] = False
        filter_sum[n] = result_sum

tok_ds_clean = tok_ds_clean.add_column("passed_quality", passed_quality)
tok_ds_clean = tok_ds_clean.add_column("filter", filter)
tok_ds_clean = tok_ds_clean.add_column("passed_quality_sum", passed_quality_sum)
tok_ds_clean = tok_ds_clean.add_column("filter_sum", filter_sum)

# save
tok_ds_clean.to_csv(
    "/data-big-projects/danish-summarization-danewsroom/tok_ds_clean_.csv"
)
