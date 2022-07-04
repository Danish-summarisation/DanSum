# %% import modules
import datasets
from rouge_score import rouge_scorer
from bert_score import BERTScorer
import pandas as pd
from generate_sums import generate_summary

# %% load data
data = pd.read_csv('/data/example_articles.csv')

# %% prepare data
data_set = datasets.Dataset.from_pandas(data)

# %% generate summaries
results = data_set.map(generate_summary, batched=True, batch_size=4)

# %% compute rouge
vanilla_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
vanilla_rouge = [vanilla_scorer.score(target=n[0], prediction=n[1]) for n in zip(results['summary'], results['pred'])]

# %% compute bertscore
metric = datasets.load_metric("bertscore")

preds = [d for d in results['pred']]
labels = [d for d in results['summary']]

bertscores = metric.compute(predictions=preds, references=labels, lang="da")
