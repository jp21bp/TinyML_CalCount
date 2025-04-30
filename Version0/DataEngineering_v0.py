#File used to explore CAPTURE-24 dataset

#Credits:
    #https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
        #Building model

### Version 0
# The dataset transformation was done through IDLE shell,
        #with the commands being shown below
# The commands perform the following:
    # Drops the "N/A" rows
    # Separates the "annotations" cols. into "activity" and "throw1", which had format "MET <value>"
    # Separate the "throw1" col into "throw 2", which was simply the string "MET", and "met", which had met value label
    # DRops the "throw1" and "throw2" cols.
    # Drops activities labeled "sleeping"
    # converts all values into floats and saves it to csv file


###Below is for IDLE shell, to avoid loading and reloading data

# import os; import gzip; import numpy as np; import pandas as pd; os.chdir("C:\\Users\\jwpar\\OneDrive\\Documents\\TML_CalCoProject\\CAPTURE-24\\capture24");
# with gzip.open('P00X.csv.gz', 'rt') as file: df = pd.read_csv(file, dtype={"annotation": "string"})
# df = df.dropna().reset_index(drop=True)
# df = df.join(df['annotation'].str.rsplit(';', n=1, expand=True).rename(columns={0:'activity',1:'throw1'})).drop(columns=['annotation','time'])
# df = df.join(df['throw1'].str.rsplit(expand=True).rename(columns={0:'throw2',1:'met'})).drop(columns=['throw1','throw2'])
# df = df.drop(df[df['activity'].str.contains('sleeping')].index).reset_index(drop=True).drop(columns='activity')
# df = df.astype(np.float32)
# #df.to_csv('cleanX.csv', index=False)

### Following used for plotting:
# values = df.values; cols = [0,1,2,3]; i=1
# from matplotlib import pyplot; pyplot.figure()
# for col in cols:
    # pyplot.subplot(len(cols), 1, i)
    # pyplot.plot(values[:,col])
    # pyplot.title(df.columns[col], y=0.5, loc='right')
    # i += 1
# pyplot.show()

# ####


import gzip
import pandas as pd

#Setting up file as a dataframe
with gzip.open('P001.csv.gz', 'rt') as file:
    df = pd.read_csv(file, dtype={"annotation": "string"})

print(df.head(101))
    #Note: last column "annotation" has 2 labels - activity name and MET
        #This is mentioned in the paper
        #Must break up the 2 labels

###### Documentation of cleaning
# The headers of the data are:
    # 'time', 'x', 'y', 'z', 'annotation'
# A sample row of the data looks as follows:
    # {2016-11-13 02:18:00.000000}, {-0.466690}, {-0.533341}, {0.658472}, {7030 sleeping;MET 0.95}
        # One-to-one correspondence between {} and headers
# Therfore the following tasks must be completed:
    # Dataset protocols (see document, done at end)
    # 0. Check and drop missing values
    # 1. Separate 'annotations' into 'activity' and 'met'
    # 2. Eliminate all the 'sleep' activities
    # 3. Write the clean data to a new csv
        # df.to_csv('clean1.csv', index=False)

#### Step 0 and 1: Separate 'annotation' into 'activity' and 'MET'
# First we investigate the number of unique elements in 'annotations':
    unique = set(df['annotation'])
    print(len(unique))  #REturns 30
    # We can see there are 30 unique items in the column
    # We can also check the types of each of those 30 items
    for item in unique: print(type(item))
    # The above returns 2 types: strings, which are expected, and NaN
    # We perform the following to check how many are <NA>
    df[pd.isna(df['annotation']) == True] # (2550371,5)
    # We then drop the NAs
    df.dropna()
        #similarly: df.drop(df[pd.isna(df['annotation']) ==True].index)
    # Which results in shape=(7469630,5)
    # New shape checks out: 10020001 - 2550371 = 7469630
    # We can redo set/for loop check above to verify
    # Then reset the indexes:
    df.reset_index(drop=True)
# From the remaining, below are a few samples of 'annotation' column only:
    # '7030 sleeping;MET 0.95'
    # 'leisure;miscellaneous;standing;9050 standing talking in person/using a phone/smartphone/tablet;MET 1.8'
    # 'leisure;miscellaneous;5060 shopping miscellaneous;MET 2.3'
# There are multiple ';' delimiters
# But only last element 'MET' is needed to have separation
# After we separate, the 'annotations' columns should be drop
# we can also drop "time" since it won't be used
df = df.join(df['annotation'].str.rsplit(';', n=1, expand=True).rename(columns={0:'activity',1:'throw1'})).drop(columns=['annotation','time'])
# We labeled the last col "throw1" bc we'll throw it out
# we want a col of only the MET value itself
# To get met values row, we do:
df = df.join(df['throw1'].str.rsplit(expand=True).rename(columns={0:'throw2',1:'met'})).drop(columns=['throw1','throw2'])
# we then convert the MET string to float:
df['met']=df['met'].astype(float)


#### 2. Eliminate all the 'sleep' activities
# Not interested in sleeping activities
# sleeping activites can be filtered with:
df[df['activity'].str.contains('sleeping')]
# Above reveals that there are 2742002 sleeping activities
# we'll also drop 'activity' col, since it won't be used on model
# we can drop sleeps, activities, and reset indexes with
df.drop(df[df['activity'].str.contains('sleeping')].index).reset_index(drop=True)
# which leaves us with 47277628 = 7469630-2742002 activites

# It's interesting to note that:
set(df['met'])
# returns: {1.8, 2.5, 3.3, 2.0, 1.3, 2.3, 1.5, 3.0}
# this info will be important once we do balancing




### Explorations from IDLE shell
#print(type(df['time'], df['x'], df['annotation'])
    #Returns "<class 'pandas.core.series.Series'>" for all

# print(df['annotation'][5000000])
    # Returns: leisure;miscellaneous;standing;9050 standing talking in person/using a phone/smartphone/tablet;MET 1.8

# ex = df['annotation'].str.split(';')
# ex.shape
    # (10020001,)
    ## This implies there exists:
        ## 10,020,001 samples of data, with a 100Hz rate
        ## 100,200 = 10,020,001 / 100, seconds of data
        ## 1,670 = 100,200 / 60, minutes of data
        ## 27.8333 = 1670 / 60, hours of data
# ex[0]
    # ['7030 sleeping', 'MET 0.95']
# ex[0][0]
    # '7030 sleeping'




## IMPORtant:
# df.join(df['annotation'].str.rsplit(';', n=1, expand=True).rename(columns={0:'activity',1:'MET'})).drop(columns='annotation')