# %% import modules
import numpy as np
from rouge_score import rouge_scorer
import nltk
nltk.download('stopwords')

# %% load data
preds_results = np.load('/results/daT5-base-summariser_preds.npy', allow_pickle = True)

# %% prepare data
preds_list = [n for n in preds_results]
references = [refs['summary'] for refs in preds_list]
predictions = [preds['pred'] for preds in preds_list]

# %% compute xl-sum rouge
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True, lang='danish')
scores = [scorer.score(target=refs, prediction=preds) for refs, preds in zip(references, predictions)]