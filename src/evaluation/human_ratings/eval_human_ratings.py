import pandas as pd
from statistics import (median, mode, mean)
from scipy.stats import bootstrap

kat = pd.read_csv('ratings_Katrine.csv', sep = ";").drop("Unnamed: 0", axis = 1)
ida = pd.read_csv('ratings_Ida.csv', sep = ';').drop("example", axis = 1)
kat.columns = ['1', '2', '3', '4', '5']
ida.columns = ['1', '2', '3', '4', '5']

# Kat data
# Transpose data
transposed_data = {}

for index, row in kat.iterrows():
    for col, val in enumerate(row):
        if val not in transposed_data:
            transposed_data[val] = []
        transposed_data[val].append(col + 1)

df_kat = pd.DataFrame.from_dict(transposed_data, orient='columns')

df_kat = df_kat.drop(columns="e")

# Define a function to shift the rankings
def shift_rankings(row):
    # Get the unique values in the row
    values = row.unique()
    # Create a dictionary to map the old values to the new values
    mapping = {}
    for i, val in enumerate(sorted(values)):
        mapping[val] = i + 1
    # Apply the mapping to the row
    return row.map(mapping)
# Apply the function to each row of the dataframe
df_kat = df_kat.apply(shift_rankings, axis=1)

#reorder columns
df_kat = df_kat[['a', 'b', 'c', "d"]]

# rename columns
df_kat = df_kat.rename(columns = {"a": "Reference", "b": "DanSum_large", "c": "DanSum_small", "d": "DanSum_base"})

# Ida
# Transpose data
transposed_data = {}

for index, row in ida.iterrows():
    for col, val in enumerate(row):
        if val not in transposed_data:
            transposed_data[val] = []
        transposed_data[val].append(col + 1)

df_ida = pd.DataFrame.from_dict(transposed_data, orient='columns')

df_ida = df_ida.drop(columns="e")

# Define a function to shift the rankings
def shift_rankings(row):
    # Get the unique values in the row
    values = row.unique()
    # Create a dictionary to map the old values to the new values
    mapping = {}
    for i, val in enumerate(sorted(values)):
        mapping[val] = i + 1
    # Apply the mapping to the row
    return row.map(mapping)
# Apply the function to each row of the dataframe
df_ida = df_ida.apply(shift_rankings, axis=1)

#reorder columns
df_ida = df_ida[['a', 'b', 'c', "d"]]

# rename columns
df_ida = df_ida.rename(columns = {"a": "Reference", "b": "DanSum_large", "c": "DanSum_small", "d": "DanSum_base"})

# Now dataframes are ready for the appendix
df_kat.to_csv("../../../data/ratings_kat_cleaned.csv")
df_ida.to_csv("../../../data/ratings_ida_cleaned.csv")

# both = pd.concat([df_kat, df_ida])

# a = both["Reference"]
# b = both["DanSum_large"]
# c = both["DanSum_small"]
# d = both["DanSum_base"]

# a.mean()

# models = [a, b, c, d]

# # [print('median of {}: '.format(name) + str(median(model))) for name, model in zip(names, models)]
# from scipy import stats as st
# median = [median(model) for model in models]
# mode = [mode(model) for model in models]
# mean = [mean(model) for model in models]

# names = ['a', 'b', 'c', 'd']

# pd.DataFrame([median, mode, mean], columns = names, index = ['median', 'mode', 'mean']).T
# # ##

# # # Interrater reliability
# from sklearn.metrics import cohen_kappa_score
# cohen_kappa_score(df_ida["Reference"], df_kat["Reference"])
# cohen_kappa_score(df_ida["DanSum_large"], df_kat["DanSum_large"])
# cohen_kappa_score(df_ida["DanSum_small"], df_kat["DanSum_small"])
# cohen_kappa_score(df_ida["DanSum_base"], df_kat["DanSum_base"])


##### Interrater reliability

# Change into bolean df
df_kat2 = df_kat.rename(columns = {"Reference": "a", "DanSum_large": "b", "DanSum_small": "c", "DanSum_base": "d"})
df_ida2 = df_ida.rename(columns = {"Reference": "a", "DanSum_large": "b", "DanSum_small": "c", "DanSum_base": "d"})

relation = ["a>b", "a>c", "a>d", "b>c", "b>d", "c>d"]
relation_list = relation*len(df_kat2)

bool_list = []
for index, row in df_kat2.iterrows():
    # a > b
    if row[0]<row[1]:
        bool_list.append(1)
    else:
        bool_list.append(0)
        # a > c
    if row[0]<row[2]:
        bool_list.append(1)
    else:
        bool_list.append(0)
            # a > d
    if row[0]<row[3]:
        bool_list.append(1)
    else:
        bool_list.append(0)
                # b > c
    if row[1]<row[2]:
        bool_list.append(1)
    else:
        bool_list.append(0)
                    # b > d
    if row[1]<row[3]:
        bool_list.append(1)
    else:
        bool_list.append(0)
        # c > d
    if row[2]<row[3]:
        bool_list.append(1)
    else:
        bool_list.append(0)

# same for Ida
bool_list2 = []
for index, row in df_ida2.iterrows():
    # a > b
    if row[0]<row[1]:
        bool_list2.append(1)
    else:
        bool_list2.append(0)
        # a > c
    if row[0]<row[2]:
        bool_list2.append(1)
    else:
        bool_list2.append(0)
            # a > d
    if row[0]<row[3]:
        bool_list2.append(1)
    else:
        bool_list2.append(0)
                # b > c
    if row[1]<row[2]:
        bool_list2.append(1)
    else:
        bool_list2.append(0)
                    # b > d
    if row[1]<row[3]:
        bool_list2.append(1)
    else:
        bool_list2.append(0)
        # c > d
    if row[2]<row[3]:
        bool_list2.append(1)
    else:
        bool_list2.append(0)

# Make finished df
ids = list(range(100))
text = []

# loop through the IDs and add 6 rows of data for each ID
for id in ids:
    rows = []
    for i in range(6):
        row = id # example data
        rows.append(row)
    text.extend(rows)
data = {"text": text, "relation": relation_list, "rater1":bool_list, "rater2": bool_list2}
df = pd.DataFrame(data)



rater1 = df["rater1"]
rater2 = df["rater2"]

def interrater(rater1, rater2):
    counts = [0] * len(rater2)

    # loop over each element in x
    for i in range(len(rater1)):
        # check if the element is equal to the corresponding element in y
        if i >= len(rater2):
            break
        if rater1[i] == rater2[i]:
            counts[i] += 1

    return sum(counts)/len(rater1)

print(interrater(rater1, rater2))

# Bootstrap
res = bootstrap((rater1,rater2),interrater, paired=True, vectorized=False)
print(res)