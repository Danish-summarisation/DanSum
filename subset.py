import random
import pickle

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

#print(pre_sub)
pre_sub_short = [i[:-1] for i in pre_sub] #removes "\n" (so it matches key in embeddings.pkl)

# EMBEDDINGS

print('--- gonna loady load now ---')
# load in all embeddings
with open('embeddings.pkl', 'rb') as f:
    data = pickle.load(f)

print('--- loaded all embeddings ---')

# subset
dict_sub = { key:value for key, value in data.items() if key in pre_sub_short}

a_file = open('embeddings' + str(size) + '.pkl', 'wb') #wb means write binary
pickle.dump(dict_sub, a_file) #save the subsetted dictionary
a_file.close()

print('SIZE OF SUBSET DICTIONARY:')
print(len(dict_sub.keys())) #hmmmm, for 1000 it's only 978 - what could be wrong?
print('--- YAAAAAY ALL DONE :D ---')