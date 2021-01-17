#!/usr/bin/env python
# coding: utf-8

# **Table of Contents**
# 
# 1. [Opening](#Opening)
# 2. [Data loading manually and from CSV files to Pandas DataFrame](#Data-loading-manually-and-from-CSV-files-to-Pandas-DataFrame)
# 3. [Loading, editing, and viewing data from Pandas DataFrame](#Loading,-editing,-and-viewing-data-from-Pandas-DataFrame)
# 4. [Renaming colmnns, exporting and saving Pandas DataFrames](#Renaming-colmnns,-exporting-and-saving-Pandas-DataFrames)
# 5. [Summarising, grouping, and aggregating data in Pandas](#Summarising,-grouping,-and-aggregating-data-in-Pandas)
# 6. [Merge and join DataFrames with Pandas](#Merge-and-join-DataFrames-with-Pandas)
# 7. [Basic Plotting Pandas DataFrames](#Basic-Plotting-Pandas-DataFrames)

# # Opening
# CSV(comma-separated value) files are a common file format of data. 
# The ability to read, manipulate, and write date to and from CSV files using Python is a key skill to master for any data scientist or business analysis.  
# 1) what CSV files are,  
# 2) how to read CSV files into "Pandas DataFrames",  
# 3) how to write DataFrames back to CSV files.  
# What is "Pandas DataFrame"? 
# : Pandas is the most popular data manipulation package in Python, and DataFrames are the Pandas data type for storing tabular 2D data. 
# : Pandas development started in 2008 with main developer Wes McKinney and the library has become a standard for data analysis and management using Python. 
# : Pandas fluency is essential for any Python-based data professional, people interested in trying a Kaggle challenge, or anyone seeking to automate a data process. 
# : The Pandas library documentation defines a DataFrame as a “two-dimensional, size-mutable, potentially heterogeneous tabular data structure with labeled axes (rows and columns)”. 
# - There can be multiple rows and columns in the data. 
# - Each row represents a sample of data, 
# - Each column contains a different variable that describes the samples (rows). 
# - The data in every column is usually the same type of data – e.g. numbers, strings, dates. 
# - Usually, unlike an excel data set, DataFrames avoid having missing values, and there are no gaps and empty values between rows or columns. 

# In[1]:


# Manually generate data
import pandas as pd 
pd.options.display.max_columns = 20
pd.options.display.max_rows = 10

data = {'column1':[1,2,3,4,5],
        'anatoeh_column':['this', 'column', 'has', 'strings', 'indise!'],
        'float_column':[0.1, 0.5, 33, 48, 42.5555],
        'binary_column':[True, False, True, True, False]}
print(data)
print(data['column1'])
display(pd.DataFrame(data))
display(pd.DataFrame(data['column1']))


# # Data loading manually and from CSV files to Pandas DataFrame
# (https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html)  
# There are 3 fundamantal conceps to grasp and debug the operation of the data loading procedure.  
# 1) Understanding file extensions and file types – what do the letters CSV actually mean? what’s the difference between a .csv file and a .txt file?  
# 2) Understanding how data is represented inside CSV files – if you open a CSV file, what does the data actually look like?  
# 3) Understanding the Python path and how to reference a file – what is the absolute and relative path to the file you are loading? What directory are you working in?  
# 4) CSV file loading errors
# - FileNotFoundError: File b'filename.csv' does not exist  
# => A File Not Found error is typically an issue with path setup, current directory, or file name confusion (file extension can play a part here!)
# - UnicodeDecodeError: 'utf-8' codec can't decode byte in position : invalid continuation byte  
# => A Unicode Decode Error is typically caused by not specifying the encoding of the file, and happens when you have a file with non-standard characters. For a quick fix, try opening the file in Sublime Text, and re-saving with encoding ‘UTF-8’.
# - pandas.parser.CParserError: Error tokenizing data.  
# => Parse Errors can be caused in unusual circumstances to do with your data format – try to add the parameter “engine=’python'” to the read_csv function call; this changes the data reading function internally to a slower but more stable method.

# In[2]:


# Finding your Python path
# The "OS module" is for operating system dependent functionality into Python
import os
print(os.getcwd())
print(os.listdir())
# os.chdir("path")


# In[3]:


# File Loading from "Absolute" and "Relative" paths
# Relative paths are directions to the file starting at your current working directory, where absolute paths always start at the base of your file system.
# direct_path : 'https://s3-eu-west-1.amazonaws.com/shanebucket/downloads/FAO+database.csv' from 'https://www.kaggle.com/dorbicycle/world-foodfeed-production'
absolute_path = './Data/FoodAgricultureOrganization/Food_Agriculture_Organization_UN_Full.csv'
pd.read_csv(absolute_path, sep=',')


# In[5]:


relative_path = 'D:/Research/Shared/TimeSeriesAnalysis/Data/FoodAgricultureOrganization/Food_Agriculture_Organization_UN_Full.csv'
pd.read_csv(relative_path, sep=',')


# In[6]:


pd.options.display.max_columns = 20
relative_path = './Data/FoodAgricultureOrganization/Food_Agriculture_Organization_UN_Full.csv'
raw_data = pd.read_csv(relative_path, sep=',')
raw_data


# # Loading, editing, and viewing data from Pandas DataFrame
# Pandas displays only 20 columns by default for wide data dataframes, and only 60 or so rows, truncating the middle section. 
# If you’d like to change these limits, you can edit the defaults using some internal options for Pandas displays
# (simple use pd.display.options.XX = value to set these) 
# (https://pandas.pydata.org/pandas-docs/stable/options.html)
# - pd.options.display.width – the width of the display in characters – use this if your display is wrapping rows over more than one line.
# - pd.options.display.max_rows – maximum number of rows displayed.
# - pd.options.display.max_columns – maximum number of columns displayed. 
# 
# Finally, to see some of the core statistics about a particular column, you can use the ‘describe‘ function.
# - For numeric columns, describe() returns basic statistics: the value count, mean, standard deviation, minimum, maximum, and 25th, 50th, and 75th quantiles for the data in a column. 
# - For string columns, describe() returns the value count, the number of unique entries, the most frequently occurring value (‘top’), and the number of times the top value occurs (‘freq’)
# 
# There’s two main options to achieve the selection and indexing activities in Pandas. 
# When using .loc, or .iloc, you can control the output format by passing lists or single values to the selectors. 
# (http://pandas.pydata.org/pandas-docs/stable/indexing.html#selection-by-label)
# 1. iloc
#     - Note that .iloc returns a Pandas Series when one row is selected, and a Pandas DataFrame when multiple rows are selected, or if any column in full is selected. To counter this, pass a single-valued list if you require DataFrame output.
#     - When selecting multiple columns or multiple rows in this manner, remember that in your selection e.g.[1:5], the rows/columns selected will run from the first number to one minus the second number. e.g. [1:5] will go 1,2,3,4., [x,y] goes from x to y-1.
# 2. loc
#     - Label-based / Index-based indexing
#     - Boolean / Logical indexing
#         - You pass an array or Series of True/False values to the .loc indexer to select the rows where your Series has True values.
# 
# Selecting rows and columns
# - using a dot notation, e.g. data.column_name,
# - using square braces and the name of the column as a string, e.g. data['column_name']
# - using numeric indexing and the iloc selector data.iloc[:, <column_number>] 
# 
# When a column is selected using any of these methodologies, a pandas.Series is the resulting datatype. A pandas series is a one-dimensional set of data.
# - square-brace selection with a list of column names, e.g. data[['column_name_1', 'column_name_2']]
# - using numeric indexing with the iloc selector and a list of column numbers, e.g. data.iloc[:, [0,1,20,22]]
# 
# Rows in a DataFrame are selected, typically, using the iloc/loc selection methods, or using logical selectors
# - numeric row selection using the iloc selector, e.g. data.iloc[0:10, :] – select the first 10 rows.
# - label-based row selection using the loc selector (this is only applicably if you have set an “index” on your dataframe. e.g. data.loc[44, :]
# - logical-based row selection using evaluated statements, e.g. data[data["Area"] == "Ireland"] – select the rows where Area value is ‘Ireland’.
# ![](https://shanelynnwebsite-mid9n9g1q9y8tt.netdna-ssl.com/wp-content/uploads/2016/10/Pandas-selections-and-indexing-768x549.png)
# 
# To delete rows and columns from DataFrames, Pandas uses the “drop” function.
# - To delete a column, or multiple columns, use the name of the column(s), and specify the “axis” as 1.
# - Alternatively, as in the example below, the ‘columns’ parameter has been added in Pandas which cuts out the need for ‘axis’.
# - The drop function returns a new DataFrame, with the columns removed. To actually edit the original DataFrame, the “inplace” parameter can be set to True, and there is no returned value.
# - Rows can also be removed using the “drop” function, by specifying axis=0. Drop() removes rows based on “labels”, rather than numeric indexing. To delete rows based on their numeric position / index, use iloc to reassign the dataframe values

# In[7]:


# Examine data in a Pandas DataFrame
raw_data.shape


# In[8]:


raw_data.ndim


# In[9]:


raw_data.head(5)


# In[10]:


raw_data.tail(5)


# In[11]:


raw_data.dtypes


# In[12]:


raw_data['Item Code'] = raw_data['Item Code'].astype(str)
raw_data.dtypes


# In[13]:


raw_data['Y2013'].describe()


# In[14]:


raw_data['Area'].describe()


# In[15]:


raw_data.describe()


# In[16]:


# Selecting and manipulating data
raw_data.iloc[0]


# In[17]:


raw_data.iloc[[1]]


# In[18]:


raw_data.iloc[[-1]]


# In[19]:


raw_data.iloc[:,0]


# In[20]:


raw_data.iloc[:,[1]]


# In[21]:


raw_data.iloc[:,[-1]]


# In[22]:


raw_data.iloc[0:5]


# In[23]:


raw_data.iloc[:,0:2]


# In[24]:


raw_data.iloc[[0,3,6,24],[0,5,6]]


# In[25]:


raw_data.iloc[0:5, 5:8]


# In[26]:


raw_data.loc[0]


# In[27]:


raw_data.loc[[1]]


# In[28]:


raw_data.loc[[1,3]]


# In[29]:


raw_data.loc[[1,3],['Item','Y2013']]


# In[30]:


raw_data.loc[[1,3],'Item':'Y2013']


# In[31]:


raw_data.loc[1:3,'Item':'Y2013']


# In[32]:


raw_data_test = raw_data.loc[10:,'Item':'Y2013']
raw_data_test.iloc[[0]]


# In[33]:


raw_data_test.loc[[10]]


# In[34]:


raw_data.loc[raw_data['Item'] == 'Sugar beet']


# In[35]:


# is same as
raw_data[raw_data['Item'] == 'Sugar beet']


# In[36]:


raw_data.loc[raw_data['Item'] == 'Sugar beet', 'Area']


# In[37]:


raw_data.loc[raw_data['Item'] == 'Sugar beet', ['Area']]


# In[38]:


# is not same as
raw_data[raw_data['Item'] == 'Sugar beet', ['Area']]


# In[39]:


raw_data.loc[raw_data['Item'] == 'Sugar beet', ['Area', 'Item', 'latitude']]


# In[40]:


raw_data.loc[raw_data['Item'] == 'Sugar beet', 'Area':'latitude']


# In[41]:


raw_data.loc[raw_data['Area'].str.endswith('many')]


# In[42]:


# is same as
raw_data.loc[raw_data['Area'].isin(['Germany'])]


# In[43]:


raw_data.loc[raw_data['Area'].isin(['Germany', 'France'])]


# In[44]:


raw_data.loc[(raw_data['Area'].str.endswith('many')) & (raw_data['Element'] == 'Feed')]


# In[45]:


raw_data.loc[(raw_data['Y2004'] < 1000) & (raw_data['Y2004'] > 990)]


# In[46]:


raw_data.loc[(raw_data['Y2004'] < 1000) & (raw_data['Y2004'] > 990), ['Area', 'Item', 'latitude']]


# In[47]:


raw_data.loc[raw_data['Item'].apply(lambda x: len(x.split(' ')) == 5)]


# In[48]:


# is same as
TF_indexing = raw_data['Item'].apply(lambda x: len(x.split(' ')) == 5)
raw_data.loc[TF_indexing]


# In[49]:


raw_data.loc[TF_indexing, ['Area', 'Item', 'latitude']]


# In[50]:


raw_data_test = raw_data.copy()
raw_data_test.loc[(raw_data_test['Y2004'] < 1000) & (raw_data_test['Y2004'] > 990), ['Area']]


# In[51]:


raw_data_test.loc[(raw_data_test['Y2004'] < 1000) & (raw_data_test['Y2004'] > 990), ['Area']] = 'Company'
raw_data_test.loc[(raw_data_test['Y2004'] < 1000) & (raw_data_test['Y2004'] > 980), ['Area']]


# In[52]:


raw_data['Y2007'].sum(), raw_data['Y2007'].mean(), raw_data['Y2007'].median(), raw_data['Y2007'].nunique(), raw_data['Y2007'].count(), raw_data['Y2007'].max(), raw_data['Y2007'].min()


# In[53]:


[raw_data['Y2007'].sum(),
 raw_data['Y2007'].mean(),
 raw_data['Y2007'].median(),
 raw_data['Y2007'].nunique(),
 raw_data['Y2007'].count(),
 raw_data['Y2007'].max(),
 raw_data['Y2007'].min(),
 raw_data['Y2007'].isna().sum(),
 raw_data['Y2007'].fillna(0)]


# In[54]:


# Delete the "Area" column from the dataframe
raw_data.drop("Area", axis=1)


# In[55]:


# alternatively, delete columns using the columns parameter of drop
raw_data.drop(columns="Area")


# In[56]:


# Delete the Area column from the dataframe and the original 'data' object is changed when inplace=True
raw_data.drop("Area", axis=1, inplace=False)


# In[57]:


# Delete multiple columns from the dataframe
raw_data.drop(["Y2011", "Y2012", "Y2013"], axis=1)


# In[58]:


# Delete the rows with labels 0,1,5
raw_data.drop([0,1,5], axis=0)


# In[59]:


# Delete the rows with label "Afghanistan". For label-based deletion, set the index first on the dataframe
raw_data.set_index("Area")
raw_data.set_index("Area").drop("Afghanistan", axis=0)


# In[60]:


# Delete the first five rows using iloc selector
raw_data.iloc[5:,]


# # Renaming colmnns, exporting and saving Pandas DataFrames
# Column renames are achieved easily in Pandas using the DataFrame rename function. The rename function is easy to use, and quite flexible.
# - Rename by mapping old names to new names using a dictionary, with form {“old_column_name”: “new_column_name”, …}
# - Rename by providing a function to change the column names with. Functions are applied to every column name.
# 
# After manipulation or calculations, saving your data back to CSV is the next step.
# - to_csv to write a DataFrame to a CSV file,
# - to_excel to write DataFrame information to a Microsoft Excel file.

# In[61]:


# Renaming of columns
raw_data.rename(columns={'Area':'New_Area'})


# In[62]:


display(raw_data)
raw_data.rename(columns={'Area':'New_Area'}, inplace=False)


# In[63]:


raw_data.rename(columns={'Area':'New_Area',
                         'Y2013':'Year_2013'}, inplace=False)


# In[64]:


raw_data.rename(columns=lambda x: x.upper().replace(' ', '_'), inplace=False)


# In[65]:


# Exporting and saving
# Output data to a CSV file
# If you don't want row numbers in my output file, hence index=False, and to avoid character issues, you typically use utf8 encoding for input/output.
raw_data.to_csv("Tutorial_Pandas_Output_Filename.csv", index=False, encoding='utf8')
# Output data to an Excel file.
# For the excel output to work, you may need to install the "xlsxwriter" package.
raw_data.to_excel("Tutorial_Pandas_Output_Filename.xlsx", sheet_name="Sheet 1", index=False)


# # Summarising, grouping, and aggregating data in Pandas
# The .describe() function is a useful summarisation tool that will quickly display statistics for any variable or group it is applied to. 
# The describe() output varies depending on whether you apply it to a numeric or character column. 
# 
# | Function | Description                         |
# |----------|-------------------------------------|
# | count    | Number of non-null observations     |
# | sum      | Sum of values                       |
# | mean     | Mean of values                      |
# | mad      | Mean absolute deviation             |
# | median   | Arithmetic median of values         |
# | min      | Minimum                             |
# | max      | Maximum                             |
# | mode     | Mode                                |
# | abs      | Absolute Value                      |
# | prod     | Product of values                   |
# | std      | Unbiased standard deviation         |
# | var      | Unbiased variance                   |
# | sem      | Unbiased standard error of the mean |
# | skew     | Unbiased skewness (3rd moment)      |
# | kurt     | Unbiased kurtosis (4th moment)      |
# | quantile | Sample quantile (value at %)        |
# | cumsum   | Cumulative sum                      |
# | cumprod  | Cumulative product                  |
# | cummax   | Cumulative maximum                  |
# | cummin   | Cumulative minimum                  |
# 
# We'll be grouping large data frames by different variables, and applying summary functions on each group. 
# This is accomplished in Pandas using the “groupby()” and “agg()” functions of Panda’s DataFrame objects. 
# (http://pandas.pydata.org/pandas-docs/stable/groupby.html)
# - Groupby essentially splits the data into different groups depending on a variable of your choice. 
# - The groupby() function returns a GroupBy object, but essentially describes how the rows of the original data set has been split.
# - The GroupBy object.groups variable is a dictionary whose keys are the computed unique groups and corresponding values being the axis labels belonging to each group. 
# - Functions like max(), min(), mean(), first(), last() can be quickly applied to the GroupBy object to obtain summary statistics for each group – an immensely useful function. 
# - If you calculate more than one column of results, your result will be a Dataframe. For a single column of results, the agg function, by default, will produce a Series. You can change this by selecting your operation column differently (ex. [[]])
# - The groupby output will have an index or multi-index on rows corresponding to your chosen grouping variables. To avoid setting this index, pass “as_index=False” to the groupby operation. 
# 
# The aggregation functionality provided by the agg() function allows multiple statistics to be calculated per group in one calculation.
# ![](https://shanelynnwebsite-mid9n9g1q9y8tt.netdna-ssl.com/wp-content/uploads/2016/03/pandas_aggregation-1024x409.png)
# - When multiple statistics are calculated on columns, the resulting dataframe will have a multi-index set on the column axis. This can be difficult to work with, and be better to rename columns after a groupby operation.
# - A neater approach is using the ravel() method on the grouped columns. Ravel() turns a Pandas multi-index into a simpler array, which we can combine into sensible column names.

# In[66]:


# Summarising
url_path = 'https://shanelynnwebsite-mid9n9g1q9y8tt.netdna-ssl.com/wp-content/uploads/2015/06/phone_data.csv'
raw_phone = pd.read_csv(url_path)
raw_phone


# In[67]:


if 'date' in raw_phone.columns:
    raw_phone['date'] = pd.to_datetime(raw_phone['date'])
raw_phone


# In[68]:


raw_phone['duration'].max()


# In[69]:


raw_phone['item'].unique()


# In[70]:


raw_phone['duration'][raw_phone['item'] == 'data'].max()


# In[71]:


raw_phone['network'].unique()


# In[72]:


raw_phone['month'].value_counts()


# In[73]:


# Grouping
raw_phone.groupby(['month']).groups


# In[74]:


raw_phone.groupby(['month']).groups.keys()


# In[75]:


raw_phone.groupby(['month']).first()


# In[76]:


raw_phone.groupby(['month'])['duration'].sum()


# In[77]:


raw_phone.groupby(['month'], as_index=False)[['duration']].sum()


# In[78]:


raw_phone.groupby(['month'])['date'].count()


# In[79]:


raw_phone[raw_phone['item'] == 'call'].groupby('network')['duration'].sum()


# In[80]:


raw_phone.groupby(['month', 'item']).groups


# In[81]:


raw_phone.groupby(['month', 'item']).groups.keys()


# In[82]:


raw_phone.groupby(['month', 'item']).first()


# In[83]:


raw_phone.groupby(['month', 'item'])['duration'].sum()


# In[84]:


raw_phone.groupby(['month', 'item'])['date'].count()


# In[85]:


raw_phone.groupby(['month', 'network_type'])['date'].count()


# In[86]:


raw_phone.groupby(['month', 'network_type'])[['date']].count()


# In[87]:


raw_phone.groupby(['month', 'network_type'], as_index=False)[['date']].count()


# In[88]:


raw_phone.groupby(['month', 'network_type'])[['date']].count().shape


# In[89]:


raw_phone.groupby(['month', 'network_type'], as_index=False)[['date']].count().shape


# In[90]:


# Aggregating
raw_phone.groupby(['month'], as_index=False)[['duration']].sum()


# In[91]:


# is same as
raw_phone.groupby(['month'], as_index=False).agg({'duration':'sum'})


# In[92]:


raw_phone.groupby(['month', 'item']).agg({'duration':'sum',
                                          'network_type':'count',
                                          'date':'first'})


# In[93]:


# is same as
aggregation_logic = {'duration':'sum',
                     'network_type':'count',
                     'date':'first'}
raw_phone.groupby(['month', 'item']).agg(aggregation_logic)


# In[94]:


aggregation_logic = {'duration':[min, max, sum],
                     'network_type':'count',
                     'date':[min, 'first', 'nunique']}
raw_phone.groupby(['month', 'item']).agg(aggregation_logic)


# In[95]:


aggregation_logic = {'duration':[min, max, sum],
                     'network_type':'count',
                     'date':['first', lambda x: max(x)-min(x)]}
raw_phone.groupby(['month', 'item']).agg(aggregation_logic)


# In[96]:


raw_phone_test = raw_phone.groupby(['month', 'item']).agg(aggregation_logic)
raw_phone_test


# In[97]:


raw_phone_test.columns = raw_phone_test.columns.droplevel(level=0)
raw_phone_test


# In[98]:


raw_phone_test.rename(columns={'min':'min_duration',
                               'max':'max_duration',
                               'sum':'sum_duration',
                               '<lambda>':'date_difference'})
raw_phone_test = raw_phone.groupby(['month', 'item']).agg(aggregation_logic)
raw_phone_test


# In[99]:


raw_phone_test.columns = ['_'.join(x) for x in raw_phone_test.columns.ravel()]
raw_phone_test


# # Merge and join DataFrames with Pandas
# (http://pandas.pydata.org/pandas-docs/stable/merging.html)  
# In any real world data science situation with Python, you’ll be about 10 minutes in when you’ll need to merge or join Pandas Dataframes together to form your analysis dataset. 
# Merging and joining dataframes is a core process that any aspiring data analyst will need to master.
# - “Merging” two datasets is the process of bringing two datasets together into one, and aligning the rows from each based on common attributes or columns.
# - The merging operation at its simplest takes a left dataframe (the first argument), a right dataframe (the second argument), and then a merge column name, or a column to merge “on”.
# - In the output/result, rows from the left and right dataframes are matched up where there are common values of the merge column specified by “on”.
# - By default, the Pandas merge operation acts with an “inner” merge.
# 
# There are three different types of merges available in Pandas. 
# These merge types are common across most database and data-orientated languages (SQL, R, SAS) and are typically referred to as “joins”.
# - Inner Merge / Inner join – The default Pandas behaviour, only keep rows where the merge “on” value exists in both the left and right dataframes.
# - Left Merge / Left outer join – (aka left merge or left join) Keep every row in the left dataframe. Where there are missing values of the “on” variable in the right dataframe, add empty / NaN values in the result.
# - Right Merge / Right outer join – (aka right merge or right join) Keep every row in the right dataframe. Where there are missing values of the “on” variable in the left column, add empty / NaN values in the result.
# - Outer Merge / Full outer join – A full outer join returns all the rows from the left dataframe, all the rows from the right dataframe, and matches up rows where possible, with NaNs elsewhere.
# 
# ![](https://i.stack.imgur.com/hMKKt.jpg)

# In[100]:


# Merge and join od dataframes
user_usage = pd.read_csv('https://raw.githubusercontent.com/shanealynn/Pandas-Merge-Tutorial/master/user_usage.csv')
user_device = pd.read_csv('https://raw.githubusercontent.com/shanealynn/Pandas-Merge-Tutorial/master/user_device.csv')
device_info = pd.read_csv('https://raw.githubusercontent.com/shanealynn/Pandas-Merge-Tutorial/master/android_devices.csv')
display(user_usage.head())
display(user_device.head())
display(device_info.head())


# In[101]:


# Q: if the usage patterns for users differ between different devices
result = pd.merge(left=user_usage, right=user_device, on='use_id')
result.head()


# In[102]:


print(user_usage.shape, user_device.shape, device_info.shape, result.shape)


# In[103]:


user_usage['use_id'].isin(user_device['use_id']).value_counts()


# In[104]:


result = pd.merge(left=user_usage, right=user_device, on='use_id', how='left')
print(user_usage.shape, result.shape, result['device'].isnull().sum())


# In[105]:


display(result.head(), result.tail())


# In[106]:


result = pd.merge(left=user_usage, right=user_device, on='use_id', how='right')
print(user_device.shape, result.shape, result['device'].isnull().sum(), result['monthly_mb'].isnull().sum())


# In[107]:


display(result.head(), result.tail())


# In[108]:


print(user_usage['use_id'].unique().shape[0], user_device['use_id'].unique().shape[0], pd.concat([user_usage['use_id'], user_device['use_id']]).unique().shape[0])


# In[109]:


result = pd.merge(left=user_usage, right=user_device, on='use_id', how='outer')
print(result.shape)


# In[110]:


print((result.apply(lambda x: x.isnull().sum(), axis=1) == 0).sum())


# In[111]:


# Note that all rows from left and right merge dataframes are included, but NaNs will be in different columns depending if the data originated in the left or right dataframe.
result = pd.merge(left=user_usage, right=user_device, on='use_id', how='outer', indicator=True)
result.iloc[[0, 1, 200, 201, 350, 351]]


# In[112]:


# For the question,
result1 = pd.merge(left=user_usage, right=user_device, on='use_id', how='left')
result1.head()


# In[113]:


device_info.head()


# In[114]:


result_final = pd.merge(left=result1, right=device_info[['Retail Branding', 'Marketing Name', 'Model']],
                        left_on='device', right_on='Model', how='left')
result_final[result_final['Retail Branding'] == 'Samsung'].head()


# In[115]:


result_final[result_final['Retail Branding'] == 'LGE'].head()


# In[116]:


group1 = result_final[result_final['Retail Branding'] == 'Samsung']
group2 = result_final[result_final['Retail Branding'] == 'LGE']
display(group1.describe())
display(group2.describe())


# In[117]:


result_final.groupby('Retail Branding').agg({'outgoing_mins_per_month':'mean',
                                             'outgoing_sms_per_month':'mean',
                                             'monthly_mb':'mean',
                                             'user_id':'count'})


# # Basic Plotting Pandas DataFrames
# (https://pandas.pydata.org/pandas-docs/stable/visualization.html)  
# You’ll need to have the matplotlib plotting package installed to generate graphics, and  the "%matplotlib inline" notebook ‘magic’ activated for inline plots. 
# You will also need "import matplotlib.pyplot as plt" to add figure labels and axis labels to your diagrams. 
# A huge amount of functionality is provided by the .plot() command natively by Pandas. 

# In[118]:


# Plotting DataFrames
import matplotlib.pyplot as plt
raw_data['latitude'].plot(kind='hist', bins=100)
plt.xlabel('Latitude Value')
plt.show()


# In[119]:


raw_data.loc[raw_data['Element'] == 'Food']


# In[120]:


raw_data_test = raw_data.loc[raw_data['Element'] == 'Food']
pd.DataFrame(raw_data_test.groupby('Area')['Y2013'].sum())


# In[121]:


pd.DataFrame(raw_data_test.groupby('Area')['Y2013'].sum().sort_values(ascending=False))


# In[122]:


pd.DataFrame(raw_data_test.groupby('Area')['Y2013'].sum().sort_values(ascending=False)[:10])


# In[123]:


raw_data_test.groupby('Area')['Y2013'].sum().sort_values(ascending=False)[:10].plot(kind='bar')
plt.title('Top Ten Food Producers')
plt.ylabel('Food Produced (tonnes)')


# In[ ]:




