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

# create an Empty DataFrame object
df_ratings = pd.DataFrame()
# append columns to an empty DataFrame
df_ratings['1st place'] = [None] * 100
df_ratings['2nd place'] = [None] * 100
df_ratings['3rd place'] = [None] * 100
df_ratings['4th place'] = [None] * 100
df_ratings['5th place'] = [None] * 100

# %%
df_rating = pd.read_csv('ratings_Katrine.csv', sep=',', header=0)

# %% Loop manually through texts
# NB: Open the text editor to see full text and summaries if they are too long to be shown (link at the beginning of the output)
# Press a, b, c or d to select the best summary in your opinion. 
# If a text with the text: "You have pressed X" is returned then the key choice is recorded. Else try running code again (where X is your keypress). 
# Then increase i by one
# Sometimes the code does not collect anythign the first time you click a key. 
# This is the case if there is no new output and the code is shown to be running at start and not complete. 
# Then just try again. 
# You can always overwrite you current answer by running the code again without changing i
i = 46 # start with i = 0
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
print("Resume 5")
print(df['e'][i]) 
print(" ")
print("// Press a, b, c, d or e to select the best summary in your opinion for 1st place, 2nd, 3rd and 4th place")

print("Use this to alter the ratings df_ratings[i:i+1] = ["", "", "", "", ""]")
print("// Increase i by 1 and run code chunk again")

# In the end when i == 99 
# Check that you have a rating in each element of the list "ratings"
# Thus, no "None" in any element of the list

#%% Save ratings as a new column to the csv file
# If you are working over time, maybe save the work as you go and then just check how far you are and load in this df that was saved and continue but remember to not loose your ratings but add thos in yourself
df_ratings.to_csv('ratings_Katrine2.csv')

# %%
