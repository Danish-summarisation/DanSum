# %% import modules
import numpy as np
import pandas as pd
from newsroom.analyze import Fragments

# %% load data
preds_results = np.load("/results/daT5-base-summariser_preds.npy", allow_pickle=True)
preds = [d["pred"] for d in preds_results]
texts = [d["text"] for d in preds_results]

# %% calculate density
fragments = [Fragments(n[0], n[1], lang="da") for n in zip(preds, texts)]
densities = [n.density() for n in fragments]

# %% save results
d_frame = pd.DataFrame(densities)
d_frame.to_csv("daT5-base-summariser_density.csv")
