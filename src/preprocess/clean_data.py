"""
This script preprocesses the data from the DaNewsroom data sets.
Applying cutoffs, filtering out the abstractive data and saving the data files.
"""
# %% Load modules
import pandas as pd
from datasets import Dataset, load_dataset
import datasets
from transformers import BertTokenizerFast, T5TokenizerFast
from transformers import AutoTokenizer
import seaborn
import numpy as np
import matplotlib.pyplot as plt


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


# %% Load raw data
ds = load_dataset(
    "csv", data_files="/data-big-projects/danish-summarization-danewsroom/danewsroom.csv"
)  # change path
ds_train = ds["train"]

# %% Filter by density
# df_abstractive = pd.DataFrame(ds_train)
# df_abstractive = df_abstractive[df_abstractive.density <= 1.5]
# print(len(df_abstractive))  # 287205
# df_abstractive.to_csv("/home/sarakolind/DanSum/df_abstractive.csv")  # save csv

# %% Clean by filter token article/summary length cutoffs
# df_abstractive = pd.read_csv("/home/sarakolind/DanSum/df_abstractive.csv")  # load csv

# # %% Convert data format
# ds_abstractive = Dataset.from_pandas(df_abstractive)
# dd_abstractive = datasets.DatasetDict({"data": ds_abstractive})

# # %% Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")  # choose tokenizer

# %% tokenize chosen data
tok_dd = ds_train.map(preprocess_function, batched=True)
# tok_ds = tok_dd["data"]  # turn back into dataset (not dict)

# %% add features to ds with the TOKENISED lengths of ref summary and text/article
tok_text_len = [len(text) for text in tok_dd["input_ids"]]  # input ids is the articles
tok_ds = tok_dd.add_column("tok_text_len", tok_text_len)
tok_sum_len = [
    len(summary) for summary in tok_ds["labels"]
]  # labels is the ref summaries
tok_ds = tok_ds.add_column("tok_sum_len", tok_sum_len)

# %% Make plot with summary length and text length to define cutoffs
# histogram of summary token length
# graph = seaborn.histplot(tok_ds["tok_sum_len"], color="red")
min_sum_len = np.quantile(tok_ds["tok_sum_len"], 0.02)
max_sum_len = np.quantile(tok_ds["tok_sum_len"], 0.98)
# graph.axvline(min_sum_len)
# graph.axvline(max_sum_len)
# graph.set_xlabel("Number of tokens")
# graph.set_title("Number of tokens in summaries")
# graph.text(200, 1000, f"Lower cutoff: {round(min_sum_len, 1)}")
# graph.text(200, 900, f"Upper cutoff: {round(max_sum_len, 1)}")
# plt.savefig("/home/sarakolind/DanSum/sum_tokens_hist.png")
# plt.show()

# %% histogram of text token length
# graph = seaborn.histplot(tok_ds["tok_text_len"], color="red")
min_text_len = np.quantile(tok_ds["tok_text_len"], 0.02)
max_text_len = np.quantile(tok_ds["tok_text_len"], 0.98)
# graph.axvline(min_text_len)
# graph.axvline(max_text_len)
# graph.set_xlabel("Number of tokens")
# graph.set_title("Number of tokens in article texts")
# graph.text(30000, 800, f"Lower cutoff: {round(min_text_len, 1)}")
# graph.text(30000, 700, f"Upper cutoff: {round(max_text_len, 1)}")
# plt.savefig("/home/sarakolind/DanSum/text_tokens_hist.png")
# plt.show()

# %% Filtering based on cutoffs
tok_df = pd.DataFrame(tok_ds)  # pandas dataframe version
tok_df_clean = tok_df
# minimum summary length
tok_df_clean = tok_df_clean.loc[tok_df_clean["tok_sum_len"] >= min_sum_len]
# maximum summary length
tok_df_clean = tok_df_clean.loc[tok_df_clean["tok_sum_len"] <= max_sum_len]
# minimum article length
tok_df_clean = tok_df_clean.loc[tok_df_clean["tok_text_len"] >= min_text_len]
# maximum article length
tok_df_clean = tok_df_clean.loc[tok_df_clean["tok_text_len"] <= max_text_len]
# make dataset format
tok_ds_clean = Dataset.from_pandas(tok_df_clean)

# %% Plot the newly cut data summary lengths
# graph = seaborn.histplot(tok_df_clean["tok_sum_len"], color="green")
# graph.set_xlabel("Number of tokens")
# graph.set_title("Number of tokens in filtered summaries")
# plt.savefig("/home/sarakolind/DanSum/sum_tokens_cleaned.png")
# plt.show()

# %% Plot the newly cut data text lengths
# graph = seaborn.histplot(tok_df_clean["tok_text_len"], color="green")
# graph.set_xlabel("Number of tokens")
# graph.set_title("Number of tokens in filtered article texts")
# plt.savefig("/home/sarakolind/DanSum/text_tokens_cleaned.png")
# plt.show()

"""
Quality filter is from the following github: https://github.com/centre-for-humanities-computing/danish-foundation-models/blob/b16765c065818f9d4b162d2b2ab9b3dae7d252ea/src/dfm/cleaning/quality_filter.py
"""
# %% Aplying quality filter
# NB: Have the quality_filter.py script in your working directory
from quality_filter import QualityFilter

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
# %%
# Lists of texts
texts = tok_ds_clean["text"]
summaries = tok_ds_clean["summary"]
filtered = qf.describe_filter(
    texts
)  # giver jer hvilke som bliver filtreret fra og af hvilket filter
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
    # same for summaries
    if result_sum == "passed filters":
        passed_quality_sum[n] = True
        filter_sum[n] = "nan"
    else:
        passed_quality_sum[n] = False
        filter_sum[n] = result_sum
tok_ds_clean=tok_ds_clean.add_column("passed_quality", passed_quality)
tok_ds_clean=tok_ds_clean.add_column("filter", filter)
tok_ds_clean=tok_ds_clean.add_column("passed_quality_sum", passed_quality_sum)
tok_ds_clean=tok_ds_clean.add_column("filter_sum", filter_sum)

# %% Saving it
tok_ds_clean.to_csv("/data-big-projects/danish-summarization-danewsroom/tok_ds_clean.csv")
