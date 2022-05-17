import random
import pickle
import numpy as np

# test if it can open
a_file = open('embeddings1000.pkl', "rb")
output = pickle.load(a_file)


a_file = open('da_pre.txt', "rb")
output = pickle.load(a_file)

# %% TITLE
print('hello')




# %%
print('awgwqr')


# LOOK AT STRANGE DICTS
len(output.values()) #981 in total
len([v for v in output.values() if type(v) is dict]) #244 weird ones


fuckers = { k, v for k. v in output.values() if type(k) is dict }

# see the items with weird dict values:
fuckers = {key: output[key] for key in output.values() if type(key) is dict}

dict_sub = {key: data[key] for key in data.keys() & set(pre_sub_short)}


# LOOK AT WEIRD SWEDISH IN ALL DATA:
a_file = open("C:/Users/idaba/OneDrive/Dokumenter/COGNITIVE_SCIENCE/data_science/embeddings.pkl", "rb")
all = pickle.load(a_file)

type(all)
len(all) # 2261464
len([v for v in all.values() if type(v) is dict]) # 343328 dict values?!

len([k for k in all.keys() if type(k) is str]) # all keys are strings


def unique(list1):
    x = np.array(list1)
    print(np.unique(x))

len(unique(list(all.keys())))


# INSPECT CAPTIONS
with open('da_post.txt') as f:
    eng = f.readlines()

len(eng)

print([k:v for v in all.values() if type(v) is dict][:5])


test = {key: all[key] for key in all.keys() if all[key] is dict}

#for subset
test = {key: output[key] for key in output.keys() if output[key] is dict}

len({k:v for k,v in output.items() if type(v) is dict}) # 244 out of 981 values are weird!
sub_weirdos = {k:v for k,v in output.items() if type(v) is dict}

list(sub_weirdos.items())[:2]


print(random.choice(list(test.items())))

list(test.items())[:2]

# inspect
#print(type(output))
print(random.choice(list(output.items())))
#print(output.items())
#print(len(list(output.items)))
#print(output.keys())
#print(output.values())
print(len(output.keys()))



# %%
