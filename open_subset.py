import random
import pickle

# test if it can open
a_file = open('embeddings1000.pkl', "rb")
output = pickle.load(a_file)


# inspect
#print(type(output))
print(random.choice(list(output.items())))
#print(output.items())
#print(len(list(output.items)))
#print(output.keys())
#print(output.values())
print(len(output.keys()))

