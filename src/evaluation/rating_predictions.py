'''
This script loads in text and generated summaries for human rating. 
Each text will be printed along with different summaries and one should rate the summary which one thinks is best. 
'''

# %% Load modules
import pandas as pd
import numpy as np
import keyboard

# %% Load data
filepath = '../../data/predictions (1).csv'
#lol = list(csv.reader(open(filepath, 'rb'), delimiter='\t'))
df = pd.read_csv(filepath, sep=';', header=0)
ratings = [None] * 100

# %% Loop manually through texts
# NB: Open the text editor to see full text and summaries if they are too long to be shown (link at the beginning of the output)
# Press a, b, c or d to select the best summary in your opinion. 
# If a text with the text: "You have pressed X" is returned then the key choice is recorded. Else try running code again (where X is your keypress). 
# Then increase i by one
# Sometimes the code does not collect anythign the first time you click a key. 
# This is the case if there is no new output and the code is shown to be running at start and not complete. 
# Then just try again. 
i = 2
print("Tekst", i)
print(df['texts'][i])
print(" ")
# summary
print("Resume 1")
print(df['a'][i]) 
print(" ")
print("Resume 2")
print(df['b'][i]) 
print(" ")
print("Resume 3")
print(df['c'][i])
print(" ") 
print("Resume 4")
print(df['d'][i]) 
print(" ")
print("// Press a, b, c or d to select the best summary in your opinion.")
while True:
    if keyboard.read_key() == "a":
        print("You pressed 'a'.")
        ratings[i] = "a"
        print("Increase i by 1 and run code chunk again")
        break
    elif keyboard.read_key() == "b":
        print("You pressed 'b'.")
        ratings[i] = "b"
        print("Increase i by 1 and run code chunk again")
        break
    elif keyboard.read_key() == "c":
        print("You pressed 'c'.")
        ratings[i] = "c"
        print("Increase i by 1 and run code chunk again")
        break
    elif keyboard.read_key() == "d":
        print("You pressed 'd'.")
        ratings[i] = "d"
        print("Increase i by 1 and run code chunk again")
        break

# In the end when i == 99 
# Check that you have a rating in each element of the list "ratings"
# Thus, no "None" in any element of the list

#%% Save ratings as a new column to the csv file
df['ratings'] = ratings
df.to_csv('ratings_yourName.csv')
