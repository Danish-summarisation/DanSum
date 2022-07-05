# %% import modules
import pickle

import numpy as np
from datasets import load_metric

# %% load bertscore
metric = load_metric("bertscore")

# %% load data
preds_results = np.load("/results/daT-base-summariser_preds.npy", allow_pickle=True)
preds = [d["pred"] for d in preds_results]
labels = [d["summary"] for d in preds_results]

# %% compute bertscores
bertscores = metric.compute(predictions=preds, references=labels, lang="da")

# %% save results
with open("results/daT5-base-summariser_bertscore", "wb") as fp:
    pickle.dump(bertscores, fp)
