'''
This script prepares train-test-val splits of the data.
'''

# %% LOAD PACKAGES
import pandas as pd
import datasets
from datasets import Dataset
from sklearn.model_selection import train_test_split

# %% LOAD DATA (SUBSETS)
abs1000 = pd.read_json('data/abs1000.json')
#abs10000 = pd.read_json(path_or_buf = 'data/abs10000.json', lines=True)

# %% MAKE DATASET FORMAT (SUBSETS)
abs1000ds = Dataset.from_pandas(abs1000)
#abs10000ds = Dataset.from_pandas(abs10000)

# %% MAKE 80-10-10 SPLITS
train1k, test1k = abs1000ds.train_test_split(test_size=100).values()
train1k, val1k = train1k.train_test_split(test_size=100).values()

# %% CHECK SPLIT SIZES
print(len(train1k))
print(len(test1k))
print(len(val1k))

# %% SAVE SPLITS
train1k.to_csv("data/train1k.csv")
test1k.to_csv("data/test1k.csv")
val1k.to_csv("data/val1k.csv")


# %% DO IT FOR ALL ABSTRACTIVE:
# %% LOAD DATA (SUBSETS)
abs = pd.read_json('gpu_files/abs_sums.json')
#abs10000 = pd.read_json(path_or_buf = 'data/abs10000.json', lines=True)

# %% MAKE DATASET FORMAT (SUBSETS)
absds = Dataset.from_pandas(abs)
#abs10000ds = Dataset.from_pandas(abs10000)

# %% MAKE 80-10-10 SPLITS
# (full data = 287205)
train1k, test1k = absds.train_test_split(test_size=28720).values() #specify absolute size (10%)
train1k, val1k = train1k.train_test_split(test_size=28720).values() #specify absolute size (10%)

# %% CHECK SPLIT SIZES
print(len(train1k))
print(len(test1k))
print(len(val1k))

# %% SAVE SPLITS
train1k.to_csv("data/train_abs.csv")
test1k.to_csv("data/test_abs.csv")
val1k.to_csv("data/val_abs.csv")




# %%

# ----- OLD STUFF STARTS HERE:

#drop density column
abs = abs.drop('density', 1)
mix = mix.drop('density', 1)
ex = ex.drop('density', 1)

#make dataset format
abs_data = Dataset.from_pandas(abs)
abs_data = abs_data.remove_columns('__index_level_0__')
mix_data = Dataset.from_pandas(mix)
mix_data = mix_data.remove_columns('__index_level_0__')
ex_data = Dataset.from_pandas(ex)
ex_data = ex_data.remove_columns('__index_level_0__')

#do test-train-val splits
abs_train, abs_test = abs_data.train_test_split(test_size=0.2).values()
abs_train, abs_val = abs_train.train_test_split(test_size=0.25).values()

mix_train, mix_test = mix_data.train_test_split(test_size=0.2).values()
mix_train, mix_val = mix_train.train_test_split(test_size=0.25).values()

ex_train, ex_test = ex_data.train_test_split(test_size=0.2).values()
ex_train, ex_val = ex_train.train_test_split(test_size=0.25).values()

#Save :)
abs_train.to_csv("abs_train.csv")
abs_test.to_csv("abs_test.csv")
abs_val.to_csv("abs_val.csv")

mix_train.to_csv("mix_train.csv")
mix_test.to_csv("mix_test.csv")
mix_val.to_csv("mix_val.csv")

ex_train.to_csv("ex_train.csv")
ex_test.to_csv("ex_test.csv")
ex_val.to_csv("ex_val.csv")

#test if they can be loaded:
test_train_abs = Dataset.from_pandas(pd.read_csv("abs_train.csv", usecols=['text','summary','idx']))
test_val_abs = Dataset.from_pandas(pd.read_csv("abs_val.csv", usecols=['text','summary','idx']))


# %%
