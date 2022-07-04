# %% import modules
from copyreg import pickle
import numpy as np
import pickle
import pandas as pd

# %% mt5-small-abstractive
rouge = np.load('/results/mt5-small-abstractive_rouge.npy', allow_pickle = True)
bertscore = pickle.load(open('/results/mt5-small-abstractive_bertscore', 'r'))
density = pd.read_csv('/results/mt5-small-abstractive_density.csv')
preds = np.load('/results/mt5-small-abstractive_preds.npy', allow_pickle = True)

# %% daT5-base-summariser
rouge = np.load('/results/daT5-base-summariser_rouge.npy', allow_pickle = True)
bertscore = pickle.load(open('/results/daT5-base-summariser_bertscore', 'r'))
density = pd.read_csv('/results/daT5-base-summariser_density.csv')
preds = np.load('/results/daT5-base-summariser_preds.npy', allow_pickle = True)



