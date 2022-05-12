import random
import pickle
import json

size = 1000 # set size of subset here and run the script

# PRE (ENGLISH) CAPTIONS
with open('da_pre.txt') as f:
    pre = f.readlines()
pre_sub = pre[:size] #subset
with open('pre' + str(size) + '.txt', 'w') as f:
    f.writelines(pre_sub) #write new subset file

# POST (DANISH) CAPTIONS
with open('da_post.txt') as f:
    post = f.readlines()
post_sub = post[:size]
with open('post' + str(size) + '.txt', 'w') as f:
    f.writelines(post_sub)


#removes "\n" (so it matches key in embeddings.pkl)
pre_sub_short = [i.replace('\n', '') for i in pre_sub] # better way?
post_sub_short = [i.replace('\n', '') for i in post_sub] # better way?


# EMBEDDINGS

print('--- gonna loady load now ---')
# load in all embeddings
with open('embeddings.pkl', 'rb') as f:
    data = pickle.load(f)

print('--- loaded all embeddings ---')
#print(len(data)) #2261464

# subset
#dict_sub = { key:value for key, value in data.items() if key in pre_sub_short}
dict_sub = {key: data[key] for key in data.keys() & set(pre_sub_short)}


#save pickle
a_file = open('embeddings' + str(size) + '.pkl', 'wb') #wb means write binary
pickle.dump(dict_sub, a_file) #save the subsetted dictionary
a_file.close()


print('SIZE OF SUBSET DICTIONARY:')
print(len(dict_sub.keys())) #hmmmm, for 1000 it's only 978 - what could be wrong?
print('--- YAAAAAY ALL DONE :D ---')



# TRY HERE
len(pre_sub_short) # 1000
len(dict_sub) # 981


good_indexes = [pre_sub_short.index(i) for i in dict_sub.keys()]


en_caps = [pre_sub_short[i] for i in good_indexes]
da_caps = [post_sub_short[i] for i in good_indexes]


#add \n again:
en = [i + '\n' for i in en_caps]
da = [i + '\n' for i in da_caps]


# SAVE IT
with open('en_981.txt', 'w') as f:
    f.writelines(en)

with open('da_981.txt', 'w') as f:
    f.writelines(da)