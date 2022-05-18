# %% Results investigation
import numpy as np



# %% TRAIN
print("without fp16:")
train_res = np.load('data/mt517-143951_1k_train.npy', allow_pickle = True)
d = train_res.flatten() # puts it into a list???
d[0]

# %% with fp16
print("with fp16:")
train_res = np.load('data/mt517-144342_1k_train.npy', allow_pickle = True)
d = train_res.flatten() # puts it into a list???
d[0]

# %% TEST
print("with fp16:")
train_res = np.load('data/mt517-144342_1k_test.npy', allow_pickle = True)
d = train_res.flatten() # puts it into a list???
d[0]

# %% PREDS



# %% Abstractive mt5 test results
test_results = np.load('./mT5_results/mt530-144605_abs_prefix_test.npy', allow_pickle = True)
test_results
d = test_results.flatten()
d[0]
