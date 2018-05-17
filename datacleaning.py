import pandas as pd

# Note: make sure to be on the directory that contains the data set file!!!

# Name of the NYCHANES file: orifile
orifile = 'public_v2_042618.sas7bdat'

# Using default parameters of read_sas to download NYCHanes data
data = pd.read_sas(orifile)
# print(data.head())

# Step 1: Take out redudant and/or irrelevant variables
# =====================================================
# Used background research to filter out variables that we don't need for
# model building

# Name of suggested variable file: varfile
varfile = 'suggestedVariables.csv'

# Importing csv file containing suggested variables; has no headers: var
var = pd.read_csv(varfile, header = None)

# Column where variable name resides: varcol
varcol = 2

# Extract list of variables: varlist
varlist = var[varcol]
# print(varlist)

# Checking whether all varlist variables are in the actual data
# Sending it to a csv file, since all of the results isn't printing
# varlist.isin(list(data)).to_csv('tmpvarlist.csv')

# Converting the panda series to a list
varlist = varlist.tolist()

# Use the list to extract variables: data1
data1 = data[varlist]

# Check if the everythin went ok
# print(len(list(data1)))
# print(data1.head())

# Step 2: Removing records
# ========================
# Removing records that don't fit some criteria

# List of tuples of variables used to remove records and which type of 
# entries to remove: remlist
# - First element: string representing variable name
# - Second element: list of values to omit
#   * 1st: T/F of whether there are additional criteria for omitting
#     record
#   * 2nd: List of characters
remtup = [('DBTS_NEW',False), ('PE_COMPLETED',True,['0'])]

# Removes records that doesn't match criteria: removerec
def removerec(df,list):
	newdf = df.copy()
	for var in list:
		print('Removing records not meeting criteria of {}...'.format(var[0]))
		print('Initial row count: {}'.format(str(len(newdf))))
		newdf = newdf[pd.notnull(newdf[var[0]])]
		print('Revised row count1: {}'.format(str(len(newdf))))
		if var[1]:
			for n in range(len(var[2])):
				newdf = newdf[newdf[var[0]] != float(var[2][n])]
		print('Revised row count2: {}'.format(str(len(newdf))))
	return newdf

data2 = removerec(data1, remtup)

# Step 3: Removing even more columns
# ==================================
# Apparently, there are several variables that have major issues with
# nulls. So, we're going to remove these as well

# Setting "many nulls" as 200: maxnulls
maxnulls = 200

# Dataframe of null counts of each column: listnull
listnull = pd.DataFrame(data2.isnull().sum())

# Find variables with greater than maxnulls nulls in listnull: manynnulls
manynulls = listnull[listnull[0] >= maxnulls]

# Variables to further remove from data set: remvars
remvars = manynulls.index.copy()

data3 = data2.drop(remvars, axis = 1)
# print(list(data3))
# print(remvars)

# data3.describe().to_csv('descData3.csv')

# Step 4: Remove more records with nulls in columns with minimal nulls
# ====================================================================
# There are some columns with nulls in the single digits. Remove records
# with nulls in these columns since it's too much of a hassle to think of
# how to impute these

# Setting "very little nulls" as less than 10: minnulls
minnulls = 10

littlenulls = listnull[listnull[0] < minnulls]
remrecs = littlenulls.index.copy()

data4 = data3.copy()
for vars in remrecs:
	data4 = data4[pd.notnull(data4[vars])]
# data4.describe().to_csv('descData4.csv')

# Export as csv for now to visualize some stuff
data4.to_csv('public_v2_042618.csv')
print(list(data4))
print(data4.head())
print(data4.isnull().sum())