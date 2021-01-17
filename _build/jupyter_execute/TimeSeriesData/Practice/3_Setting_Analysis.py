#!/usr/bin/env python
# coding: utf-8

# # Import Library: 분석에 사용할 모듈 설치
# **1. Import Library**

# In[1]:


get_ipython().system('python -m pip install --user --upgrade pip')


# ```python
# 
# # Ignore the warnings
# import warnings
# warnings.filterwarnings('always')
# warnings.filterwarnings('ignore')
# 
# # System related and data input controls
# import os
# 
# # Data manipulation and visualization
# import pandas as pd
# pd.options.display.float_format = '{:,.2f}'.format
# pd.options.display.max_rows = 100
# pd.options.display.max_columns = 20
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# 
# # Modeling algorithms
# # General
# import statsmodels.api as sm
# from scipy import stats
# 
# # Model selection
# from sklearn.model_selection import train_test_split
# 
# # Evaluation metrics
# # for regression
# from sklearn.metrics import mean_squared_log_error, mean_squared_error,  r2_score, mean_absolute_error
# 
# ```

# In[2]:


#######################################################
#               Loading Library & Module              #
#######################################################

## Ignore the warnings
import warnings
# warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

## System Related and data input controls
import os 

## Data manipulation and visualization
import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format
pd.options.display.max_rows = 100
pd.options.display.max_columns = 100
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 

## Modeling Algorithms 
# General
import statsmodels.api as sm
from scipy import stats

## Model selection
from sklearn.model_selection import train_test_split

## Evaluation Metrics
# For Regression
from sklearn.metrics import mean_squared_log_error, mean_squared_error, r2_score, mean_absolute_error


# # Data Loading: 분석에 사용할 데이터 불러오기
# **1. Import Library**  
# **2. Data Loading**  [(Data Source and Description)](https://www.kaggle.com/c/bike-sharing-demand/data)
# 

# In[3]:


# raw_all.values.flatten()


# In[4]:


# location = 'https://raw.githubusercontent.com/cheonbi/DataScience/master/Data/Bike_Sharing_Demand_Full.csv'
location = './Data/BikeSharingDemand/Bike_Sharing_Demand_Full.csv'
raw_all = pd.read_csv(location)
raw_all


# # Feature Engineering: 데이터에서 시계열패턴 추출하기
# **1. Import Library**  
# **2. Data Loading**  [(Data Source and Description)](https://www.kaggle.com/c/bike-sharing-demand/data)  
# **3. Feature Engineering(Rearrange of Data)**  
# 

# In[4]:


# raw_all.shape
# raw_all.ndim
# raw_all.head()
# raw_all.tail()
# raw_all.describe(include='all').T
raw_all.info()


# In[5]:


# 'datetime' in raw_all.columns


# In[6]:


# string to datetime
if 'datetime' in raw_all.columns:
    raw_all['datetime'] = pd.to_datetime(raw_all['datetime'])
    raw_all['DateTime'] = pd.to_datetime(raw_all['datetime'])
raw_all.info()


# In[7]:


# raw_all.index.dtype


# In[8]:


# set index as datetime column
if raw_all.index.dtype == 'int64':
    raw_all.set_index('DateTime', inplace=True)
raw_all
# bring back
# if raw_all.index.dtype != 'int64':
#     raw_all.reset_index(drop=False, inplace=True)
# raw_all


# In[9]:


# raw_all.describe(include='all').T
# raw_all.isnull()
# raw_all.isnull().sum()


# In[10]:


raw_all.asfreq('H')[raw_all.asfreq('H').isnull().sum(axis=1) > 0]


# In[11]:


# raw_all.index
# raw_all.asfreq('D')
# raw_all.asfreq('W')
# raw_all.asfreq('H')
# raw_all.asfreq('H').isnull().sum()
raw_all.asfreq('H')[raw_all.asfreq('H').isnull().sum(axis=1) > 0]
# raw_all.asfreq('H').head(100)


# In[12]:


# setting frequency of time series data
raw_all = raw_all.asfreq('H', method='ffill')
raw_all.isnull().sum()


# In[13]:


raw_all[['count','registered','casual']].plot(kind='line', figsize=(20,6), linewidth=3, fontsize=20,
                                              xlim=('2012-01-01', '2012-06-01'), ylim=(0,1000))
plt.title('Time Series of Target', fontsize=20)
plt.xlabel('Index', fontsize=15)
plt.ylabel('Demand', fontsize=15)
plt.show()


# In[14]:


# line plot of Y : count만 설정
raw_all[['count']].plot(kind='line', figsize=(20,6), linewidth=3, fontsize=20,
                                              xlim=('2012-01-01', '2012-03-01'), ylim=(0,1000))
plt.title('Time Series of Target', fontsize=20)
plt.xlabel('Index', fontsize=15)
plt.ylabel('Demand', fontsize=15)
plt.show()


# In[15]:


# split data as trend + seasonal + residual (트랜드, 계절성, 잔차가 더하기로 이루어짐)
plt.rcParams['figure.figsize'] = (14,9)
sm.tsa.seasonal_decompose(raw_all['count'], model='additive').plot()
plt.show()


# In[16]:



result = sm.tsa.seasonal_decompose(raw_all['count'], model='additive')

result.observed


# In[17]:


result.resid

result.observed - result.trend - result.seasonal # 잔차(result.resid)


# In[18]:


((result.observed - result.trend - result.seasonal) == result.resid).sum() # 같은 갯수


# In[19]:


result.trend[:20]


# In[20]:


result.trend[-20:]


# In[21]:


result = sm.tsa.seasonal_decompose(raw_all['count'], model='additive')
# pd.DataFrame(result.observed - result.trend - result.seasonal).describe()

pd.DataFrame(result.resid).describe()


# In[22]:


## split data as trend * seasonal * resid (trend+seasonal+resid: 덧셈이 아닌 곱셈의 모델 Y값은 퍼센트 값 등)
# multiplicatoin model
sm.tsa.seasonal_decompose(raw_all['count'], model='multiplicative').plot()
plt.show()


# In[23]:


# fill nan as some values of data
result = sm.tsa.seasonal_decompose(raw_all['count'], model='additive')
Y_trend = pd.DataFrame(result.trend)
Y_trend.fillna(method='ffill', inplace=True)
Y_trend.iloc[-20:, :]


# In[24]:


Y_trend.fillna(method='bfill', inplace=True)
Y_trend.columns = ['count trend']
Y_trend.iloc[:20, :]


# In[25]:


# fill nan as some values of data
result = sm.tsa.seasonal_decompose(raw_all['count'], model='additive')
Y_trend = pd.DataFrame(result.trend)
Y_trend.fillna(method='ffill', inplace=True)
Y_trend.fillna(method='bfill', inplace=True)
Y_trend.columns = ['count_trend']
Y_seasonal = pd.DataFrame(result.seasonal)
Y_seasonal.fillna(method='ffill', inplace=True)
Y_seasonal.fillna(method='bfill', inplace=True)
Y_seasonal.columns = ['count_seasonal']

Y_seasonal


# In[26]:


pd.concat([raw_all, Y_trend, Y_seasonal], axis=1).isnull().sum()

raw_all.columns


# In[27]:


# fill nan as some values of data
result = sm.tsa.seasonal_decompose(raw_all['count'], model='additive')
Y_trend = pd.DataFrame(result.trend)
Y_trend.fillna(method='ffill', inplace=True)
Y_trend.fillna(method='bfill', inplace=True)
Y_trend.columns = ['count_trend']
Y_seasonal = pd.DataFrame(result.seasonal)
Y_seasonal.fillna(method='ffill', inplace=True)
Y_seasonal.fillna(method='bfill', inplace=True)
Y_seasonal.columns = ['count_seasonal']

# merging several columns
pd.concat([raw_all, Y_trend, Y_seasonal], axis=1).isnull().sum()
# pd.concat([raw_all, Y_seasonal], axis=1).isnull().sum()
if 'count_trend' not in raw_all.columns:
    if 'count_seasonal' not in raw_all.columns:
        raw_all = pd.concat([raw_all, Y_trend, Y_seasonal], axis=1) 
        # raw_all에 Y_trend와 Y_seasonal 컬럼 붙이기
raw_all


# In[28]:


# plot of moving average values using rolling function
# moving average: 주변의 값들로 평균치를 계산
# count열의 데이터 앞뒤로 12개씩 24개의 데이터로 묶어주어 평균값을 계산하여 시각화
raw_all[['count']].rolling(24).mean().plot(kind='line', figsize=(20,6), linewidth=3, fontsize=20,
                                             xlim=('2012-01-01', '2012-06-01'), ylim=(0,1000))
plt.title('Time Series of Target', fontsize=20)
plt.xlabel('Index', fontsize=15)
plt.ylabel('Demand', fontsize=15)
plt.show()


# In[29]:


# comparison of several moving average values
# .rolling(24).mean(): 24개를 묶어서 평균치를 보겠다는 것은 Daily 패턴을 보겠다는 것
# .rolling(24*7).mean(): 7일간의 주별 패턴을 보겠다는 것
pd.concat([raw_all[['count']],
           raw_all[['count']].rolling(24).mean(),
           raw_all[['count']].rolling(24*7).mean()], axis=1).plot(kind='line', figsize=(20,6), linewidth=3, fontsize=20,
                                                                  xlim=('2012-01-01', '2013-01-01'), ylim=(0,1000))
plt.title('Time Series of Target', fontsize=20)
plt.xlabel('Index', fontsize=15)
plt.ylabel('Demand', fontsize=15)
plt.show()


# In[30]:


# raw_all[['count']].rolling(24).mean()


# In[31]:


# fill nan as some values and merging
Y_count_Day = raw_all[['count']].rolling(24).mean()
Y_count_Day.fillna(method='ffill', inplace=True)
Y_count_Day.fillna(method='bfill', inplace=True)
Y_count_Day.columns = ['count_Day']
Y_count_Week = raw_all[['count']].rolling(24*7).mean()
Y_count_Week.fillna(method='ffill', inplace=True)
Y_count_Week.fillna(method='bfill', inplace=True)
Y_count_Week.columns = ['count_Week']
if 'count_Day' not in raw_all.columns:
    raw_all = pd.concat([raw_all, Y_count_Day], axis=1)
if 'count_Week' not in raw_all.columns:
    raw_all = pd.concat([raw_all, Y_count_Week], axis=1)
raw_all


# In[32]:


raw_all[['count']]


# In[33]:


raw_all[['count']].diff()


# In[34]:


# 수요가 바뀌는 차이가 어떻게 되는지 알고 싶을때 .diff()
# Y(수요: Demand)의 증감 폭을 알고자 하는 그래프
# line plot of Y for specific periods
raw_all[['count']].diff().plot(kind='line', figsize=(20,6), linewidth=3, fontsize=20,
                                 xlim=('2012-01-01', '2012-06-01'), ylim=(-1000,1000))
plt.title('Time Series of Target', fontsize=20)
plt.xlabel('Index', fontsize=15)
plt.ylabel('Demand', fontsize=15)
plt.show()


# In[35]:


# diff of Y and merging
Y_diff = raw_all[['count']].diff()
Y_diff.fillna(method='ffill', inplace=True)
Y_diff.fillna(method='bfill', inplace=True)
Y_diff.columns = ['count_diff']
if 'count_diff' not in raw_all.columns:
    raw_all = pd.concat([raw_all, Y_diff], axis=1)
raw_all


# In[36]:


# raw_all[['temp']].ndim


# In[37]:


# split values as some group
raw_all['temp_group'] = pd.cut(raw_all['temp'], 10)
raw_all


# In[38]:


# raw_all.describe().T
# raw_all.describe(include='all').T


# In[39]:


# raw_all.isnull()
# raw_all.isnull().sum()


# In[40]:


# pd.options.display.max_rows = 100
# raw_all.dtypes
# pd.options.display.max_rows = 10


# In[41]:


# raw_all.datetime.dt
# raw_all.datetime.dt.year


# In[42]:


# feature extraction of time information
raw_all['Year'] = raw_all.datetime.dt.year
raw_all['Quater'] = raw_all.datetime.dt.quarter
raw_all['Quater_ver2'] = raw_all['Quater'] + (raw_all.Year - raw_all.Year.min()) * 4
raw_all


# In[43]:


# feature extraction of time information
raw_all['Month'] = raw_all.datetime.dt.month
raw_all['Day'] = raw_all.datetime.dt.day
raw_all['Hour'] = raw_all.datetime.dt.hour
raw_all['DayofWeek'] = raw_all.datetime.dt.dayofweek
raw_all


# In[44]:


# raw_all.info()
# raw_all.describe(include='all').T


# In[45]:


# raw_all['count'].shift(1)
# raw_all['count'].shift(-1)


# In[46]:


# calculation of lags of Y
raw_all['count_lag1'] = raw_all['count'].shift(1)
raw_all['count_lag2'] = raw_all['count'].shift(2)


# In[47]:


# raw_all.describe().T
# raw_all


# In[48]:


# raw_all['count_lag2'].fillna(method='bfill')
# raw_all['count_lag2'].fillna(method='ffill')
# raw_all['count_lag2'].fillna(0)


# In[49]:


# fill nan as some values
raw_all['count_lag1'].fillna(method='bfill', inplace=True)
raw_all['count_lag2'].fillna(method='bfill', inplace=True)
raw_all


# In[50]:



pd.get_dummies(raw_all['Quater'])
pd.get_dummies(raw_all['Quater']).describe().T

# 컬럼 이름을 'Quater_Dummy'로 변경
pd.get_dummies(raw_all['Quater'], prefix='Quater_Dummy')

# 컬럼 중 하나를 뺀 것으로 변경(첫번째 생성 컬럼을 삭제)
pd.get_dummies(raw_all['Quater'], prefix='Quater_dummy', drop_first=True)

# 기존에 있는 데이터에 새로만든 컬럼 3개를 붙이기
pd.concat([raw_all, pd.get_dummies(raw_all['Quater'], prefix='Quater_Dummy', drop_first=True)], axis=1)


# In[51]:


# feature extraction using dummy variables
# Quater 라는 변수가 있다면 더미변수로 만들어 저장하기

if 'Quater' in raw_all.columns:
    raw_all = pd.concat([raw_all, pd.get_dummies(raw_all['Quater'], prefix='Quater_Dummy', drop_first=True)], axis=1)
    del raw_all['Quater']
raw_all


# In[52]:


# 카테고리 변수 확인
raw_all.info()


# In[53]:


# temp_group 변수 확인하기(temp_group 변수가 없는 것만 컬럼으로 담아 리스트로 보기)
[col for col in raw_all.columns if col != 'temp_group']


# In[54]:


# temp_group 변수 제거 for 구문
result = []
for col in raw_all.columns:
    if col != 'temp_group':
        result.append(col)
result


# In[55]:


# 카테고리 temp_group를 삭제하기: del temp_group
# 'temp_group'이 아닌 변수를 모아 리스트로 담기

raw_all.loc[:, [col for col in raw_all.columns if col != 'temp_group']].describe(include='all').T


# In[56]:


# temp_group 변수 없는 것 확인
raw_all.loc[:, [col for col in raw_all.columns if col != 'temp_group']].info()


# ## Code Summary

# In[57]:



### Functionalize
### Feature engineering of default
def non_feature_engineering(raw):
    raw_nfe = raw.copy()
    if 'datetime' in raw_nfe.columns:
        raw_nfe['datetime'] = pd.to_datetime(raw_nfe['datetime'])
        raw_nfe['DateTime'] = pd.to_datetime(raw_nfe['datetime'])
    if raw_nfe.index.dtype == 'int64':
        raw_nfe.set_index('DateTime', inplace=True)
    # bring back
    # if raw_nfe.index.dtype != 'int64':
    #     raw_nfe.reset_index(drop=False, inplace=True)
    raw_nfe = raw_nfe.asfreq('H', method='ffill')
    return raw_nfe
# raw_rd = non_feature_engineering(raw_all)



### Feature engineering of all
def feature_engineering(raw):
    raw_fe = raw.copy()
    if 'datetime' in raw_fe.columns:
        raw_fe['datetime'] = pd.to_datetime(raw_fe['datetime'])
        raw_fe['DateTime'] = pd.to_datetime(raw_fe['datetime'])

    if raw_fe.index.dtype == 'int64':
        raw_fe.set_index('DateTime', inplace=True)

    raw_fe = raw_fe.asfreq('H', method='ffill')

    result = sm.tsa.seasonal_decompose(raw_fe['count'], model='additive')
    Y_trend = pd.DataFrame(result.trend)
    Y_trend.fillna(method='ffill', inplace=True)
    Y_trend.fillna(method='bfill', inplace=True)
    Y_trend.columns = ['count_trend']
    Y_seasonal = pd.DataFrame(result.seasonal)
    Y_seasonal.fillna(method='ffill', inplace=True)
    Y_seasonal.fillna(method='bfill', inplace=True)
    Y_seasonal.columns = ['count_seasonal']
    pd.concat([raw_fe, Y_trend, Y_seasonal], axis=1).isnull().sum()
    if 'count_trend' not in raw_fe.columns:
        if 'count_seasonal' not in raw_fe.columns:
            raw_fe = pd.concat([raw_fe, Y_trend, Y_seasonal], axis=1)

    Y_count_Day = raw_fe[['count']].rolling(24).mean()
    Y_count_Day.fillna(method='ffill', inplace=True)
    Y_count_Day.fillna(method='bfill', inplace=True)
    Y_count_Day.columns = ['count_Day']
    Y_count_Week = raw_fe[['count']].rolling(24*7).mean()
    Y_count_Week.fillna(method='ffill', inplace=True)
    Y_count_Week.fillna(method='bfill', inplace=True)
    Y_count_Week.columns = ['count_Week']
    if 'count_Day' not in raw_fe.columns:
        raw_fe = pd.concat([raw_fe, Y_count_Day], axis=1)
    if 'count_Week' not in raw_fe.columns:
        raw_fe = pd.concat([raw_fe, Y_count_Week], axis=1)

    Y_diff = raw_fe[['count']].diff()
    Y_diff.fillna(method='ffill', inplace=True)
    Y_diff.fillna(method='bfill', inplace=True)
    Y_diff.columns = ['count_diff']
    if 'count_diff' not in raw_fe.columns:
        raw_fe = pd.concat([raw_fe, Y_diff], axis=1)

    raw_fe['temp_group'] = pd.cut(raw_fe['temp'], 10)
    raw_fe['Year'] = raw_fe.datetime.dt.year
    raw_fe['Quater'] = raw_fe.datetime.dt.quarter
    raw_fe['Quater_ver2'] = raw_fe['Quater'] + (raw_fe.Year - raw_fe.Year.min()) * 4
    raw_fe['Month'] = raw_fe.datetime.dt.month
    raw_fe['Day'] = raw_fe.datetime.dt.day
    raw_fe['Hour'] = raw_fe.datetime.dt.hour
    raw_fe['DayofWeek'] = raw_fe.datetime.dt.dayofweek

    raw_fe['count_lag1'] = raw_fe['count'].shift(1)
    raw_fe['count_lag2'] = raw_fe['count'].shift(2)
    raw_fe['count_lag1'].fillna(method='bfill', inplace=True)
    raw_fe['count_lag2'].fillna(method='bfill', inplace=True)

    if 'Quater' in raw_fe.columns:
        if 'Quater_Dummy' not in ['_'.join(col.split('_')[:2]) for col in raw_fe.columns]:
            raw_fe = pd.concat([raw_fe, pd.get_dummies(raw_fe['Quater'], prefix='Quater_Dummy', drop_first=True)], axis=1)
            del raw_fe['Quater']
    return raw_fe
# raw_fe = feature_engineering(raw_all)
     
    


# In[58]:


# Feature Engineering된 원천 데이터

raw_fe = feature_engineering(raw_all)


# In[59]:


raw_fe


# In[60]:


raw_nfe = feature_engineering(raw_all)


# In[61]:


raw_nfe


# # Data Understanding: 추출된 패턴이 Y예측에 도움될지 시각적을 확인하기
# **1. Import Library**  
# **2. Data Loading**  [(Data Source and Description)](https://www.kaggle.com/c/bike-sharing-demand/data)  
# **3. Feature Engineering(Rearrange of Data)**  
# **4. Data Understanding(Descriptive Statistics and Getting Insight from Features)**  

# In[62]:


# raw_fe.describe(include='all').T
raw_fe.describe(include='all').T


# In[63]:


# # histogram plot
# raw_fe.hist(bins=20, grid=True, figsize=(16,12))
# plt.show()

# Histogram Plot
raw_fe.hist(bins=20, grid=True, figsize=(16,12))
plt.show()


# In[64]:


# box plot
raw_fe.boxplot(column='count', by='season', grid=True, figsize=(12,5))
plt.ylim(0,1000)
raw_fe.boxplot(column='registered', by='season', grid=True, figsize=(12,5))
plt.ylim(0,1000)
raw_fe.boxplot(column='casual', by='season', grid=True, figsize=(12,5))
plt.ylim(0,1000)


# In[65]:


# box plot
raw_fe.boxplot(column='count', by='holiday', grid=True, figsize=(12,5))
plt.ylim(0,1000)
raw_fe.boxplot(column='count', by='workingday', grid=True, figsize=(12,5))
plt.ylim(0,1000)


# In[66]:


# raw_fe[raw_all.holiday == 0]


# In[67]:


# box plot example
raw_fe[raw_all.holiday == 0].boxplot(column='count', by='Hour', grid=True, figsize=(12,5))
plt.show()
raw_fe[raw_all.holiday == 1].boxplot(column='count', by='Hour', grid=True, figsize=(12,5))
plt.show()


# In[127]:


# scatter plot
# raw_fe[raw_all.workingday == 0].plot.scatter(y='count', x='Hour', grid=True, figsize=(12,5))
# plt.show()
# raw_fe[raw_all.workingday == 1].plot.scatter(y='count', x='Hour', grid=True, figsize=(12,5))
# plt.show()
raw_fe[raw_fe.workingday == 0].plot.scatter(y='count', x='Hour', grid=True, figsize=(12,5))
plt.show()
raw_fe[raw_fe.workingday == 1].plot.scatter(y='count', x='Hour', grid=True, figsize=(12,5))
plt.show()


# In[69]:


# scatter plot for some group
raw_fe[raw_fe.workingday == 0].plot.scatter(y='count', x='Hour', c='temp', grid=True, figsize=(12,5), colormap='viridis')
plt.show()
raw_fe[raw_fe.workingday == 1].plot.scatter(y='count', x='Hour', c='temp', grid=True, figsize=(12,5), colormap='viridis')
plt.show()


# In[70]:


# scatter and box plot
raw_fe.plot.scatter(y='count', x='DayofWeek', c='temp', grid=True, figsize=(12,5), colormap='viridis')
plt.show()
raw_fe.boxplot(column='count', by='DayofWeek', grid=True, figsize=(12,5))
plt.show()


# In[71]:


# box plot example
raw_fe.boxplot(column='count', by='weather', grid=True, figsize=(12,5))
plt.ylim(0,1000)
raw_fe.boxplot(column='registered', by='weather', grid=True, figsize=(12,5))
plt.ylim(0,1000)
raw_fe.boxplot(column='casual', by='weather', grid=True, figsize=(12,5))
plt.ylim(0,1000)


# In[128]:


pd.concat([raw_fe.dtypes, raw_fe.describe(include='all').T], axis=1)


# In[73]:


# raw_fe['weather'].value_counts()


# In[74]:


# % 비율로 보기
pd.DataFrame(raw_fe['weather'].value_counts()/raw_fe['weather'].value_counts().sum()*100).T


# In[75]:


# pd.crosstab(index=raw_fe['count'], columns=raw_fe['weather'], margins=True)


# In[76]:


# generate cross table
sub_table = pd.crosstab(index=raw_fe['count'], columns=raw_fe['weather'], margins=True)
sub_table/sub_table.loc['All']*100


# In[77]:


# raw_fe.groupby('weather').describe().T


# In[78]:


# raw_fe.groupby(['weather', 'DayofWeek']).count()
# raw_fe.groupby(['weather', 'DayofWeek']).mean()


# In[79]:


# pivot table using groupby
raw_fe.groupby(['weather', 'DayofWeek']).describe()


# In[80]:


# raw_fe.groupby(['weather', 'DayofWeek']).agg({'count':'mean'})
# raw_fe.groupby(['weather', 'DayofWeek']).agg({'count':[sum, min, max]})


# In[81]:


# groupby and histogram
raw_fe.groupby(['weather', 'DayofWeek']).agg({'count':[sum, min, max]}).hist(grid=True, figsize=(12,8))
plt.show()


# In[82]:


# raw_fe.groupby('weather').groups.keys()
# raw_fe.groupby('weather').groups.items()


# In[83]:


# indexing of groupby results
for key, item in raw_fe.groupby('weather'):
    display(key, item)
    display(raw_fe.groupby('weather').get_group(key).head())


# In[84]:


# pd.cut(raw_fe['temp'], 10)
# pd.cut(raw_fe['temp'], 10).value_counts()


# In[85]:


# desctiprion of groupby results
raw_fe.groupby(pd.cut(raw_fe['temp'], 10)).describe().T


# In[86]:


# scatter plot sxample
raw_fe.plot.scatter(y='count', x='temp', grid=True, figsize=(12,5))
plt.show()


# In[87]:


# box plot example
raw_fe.boxplot(column='count', by='temp', grid=True, figsize=(12,5))
plt.show()


# In[88]:


# box plot example
raw_fe.boxplot(column='count', by='temp_group', grid=True, figsize=(12,5))
plt.show()


# In[89]:


# scatter plot example
raw_fe.plot.scatter(y='count', x='humidity', c='temp', grid=True, figsize=(12,5), colormap='viridis')
plt.show()


# In[91]:


# box plot example
raw_fe.boxplot(column='count', by='humidity', grid=True, figsize=(15,10))
plt.show()


# In[92]:


# box plot example
raw_fe.boxplot(column='count', by='windspeed', grid=True, figsize=(12,5))
plt.show()


# In[93]:


# box plot example
raw_fe.boxplot(column='count', by='windspeed', grid=True, figsize=(12,5))
plt.show()


# In[94]:


# box plot example
raw_fe.boxplot(column='count', by='Year', grid=True, figsize=(12,5))
plt.show()


# In[95]:


# box plot example
raw_fe.boxplot(column='count', by='Month', grid=True, figsize=(12,5))
plt.show()


# In[96]:


# scatter plot example
raw_fe.plot.scatter(y='count', x='Month', c='temp', grid=True, figsize=(12,5), colormap='viridis')
plt.show()


# In[97]:


# pd.plotting.scatter_matrix(raw_fe, figsize=(18,18), diagonal='kde')
# plt.show()


# <img src='Image/Scatter_Matrix.png' width='800'>

# In[98]:


# calculate correlations
rs = np.random.RandomState(0)
df = pd.DataFrame(rs.rand(10, 10))
corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')
corr


# In[99]:


# raw_fe.corr()
# raw_fe.corr().style.background_gradient()
# raw_fe.corr().style.background_gradient().set_precision(2)


# In[100]:


# correlation example
raw_fe.corr().style.background_gradient().set_precision(2).set_properties(**{'font-size': '15pt'})


# In[101]:


# selecting of columns from correlation tables
raw_fe.corr().iloc[:,8:11]


# In[102]:


# selecting of columns from correlation tables
raw_fe.corr().loc[:, ['casual', 'registered', 'count']]


# In[103]:


# selecting of columns from correlation tables
raw_fe.corr().loc[:, ['casual', 'registered', 'count']].style.background_gradient().set_precision(2).set_properties(**{'font-size': '15pt'})


# In[104]:


# correlation table example
raw_fe.corr().iloc[0:8,8:11].style.background_gradient().set_precision(2).set_properties(**{'font-size': '15pt'})


# # Data Split: 최종 전처리 및 학습/검증/테스트용 데이터 분리
# **1. Import Library**  
# **2. Data Loading**  [(Data Source and Description)](https://www.kaggle.com/c/bike-sharing-demand/data)  
# **3. Feature Engineering(Rearrange of Data)**  
# **4. Data Understanding(Descriptive Statistics and Getting Insight from Features)**  
# **5. Data Split: Train/Validate/Test Sets**  

# In[105]:


# raw_fe.isnull().sum().unique()
raw_fe.isnull().sum().sum()


# In[106]:


raw_fe.isnull().sum().unique()


# In[110]:


# for x in raw_fe.columns:
#     if x not in Y_colname+X_remove:
#         print(x)


# In[108]:


# Confirm of input and output
Y_colname = ['count']
X_remove = ['datetime', 'DateTime', 'temp_group', 'casual', 'registered']
X_colname = [x for x in raw_fe.columns if x not in Y_colname+X_remove]
X_colname


# In[112]:


# for a non time-series
raw_train, raw_test = train_test_split(raw_fe, test_size=0.2, random_state=123)
print(raw_train.shape, raw_test.shape)

raw_train.index


# In[113]:


X_train, X_test, Y_train, Y_test = train_test_split(raw_fe[X_colname], raw_fe[Y_colname], test_size=0.2, random_state=123)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

X_train.index


# In[114]:


# for a time-series
raw_train = raw_fe.loc[raw_fe.index < '2012-07-01',:]
raw_test = raw_fe.loc[raw_fe.index >= '2012-07-01',:]
print(raw_train.shape, raw_test.shape)

raw_train.index


# In[115]:


# data split of X and Y from train/test sets
Y_train = raw_train[Y_colname]
X_train = raw_train[X_colname]
Y_test = raw_test[Y_colname]
X_test = raw_test[X_colname]
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)


# In[117]:


Y_train


# ## Code Summary

# In[118]:


############################################################################
### Functionalize                                                        ###
############################################################################

### Data split of cross sectional
def datasplit_cs(raw, Y_colname, X_colname, test_size, random_seed=123):
    X_train, X_test, Y_train, Y_test = train_test_split(raw[X_colname], raw[Y_colname], test_size=test_size, random_state=random_seed)
    print('X_train:', X_train.shape, 'Y_train:', Y_train.shape)
    print('X_test:', X_test.shape, 'Y_test:', Y_test.shape)
    return X_train, X_test, Y_train, Y_test
# X_train, X_test, Y_train, Y_test = datasplit_cs(raw_fe, Y_colname, X_colname, 0.2)

### Data split of time series
def datasplit_ts(raw, Y_colname, X_colname, criteria):
    raw_train = raw.loc[raw.index < criteria,:]
    raw_test = raw.loc[raw.index >= criteria,:]
    Y_train = raw_train[Y_colname]
    X_train = raw_train[X_colname]
    Y_test = raw_test[Y_colname]
    X_test = raw_test[X_colname]
    print('Train_size:', raw_train.shape, 'Test_size:', raw_test.shape)
    print('X_train:', X_train.shape, 'Y_train:', Y_train.shape)
    print('X_test:', X_test.shape, 'Y_test:', Y_test.shape)
    return X_train, X_test, Y_train, Y_test
# X_train, X_test, Y_train, Y_test = datasplit_ts(raw_fe, Y_colname, X_colname, '2012-07-01')


# In[119]:


# 함수 이용 데이터 셋 분리 
# Confirm of input and output

Y_colname = ['count']
X_remove = ['datetime', 'DateTime', 'temp_group', 'casual', 'registered']
X_colname = [x for x in raw_fe.columns if x not in Y_colname+X_remove]
X_train, X_test, Y_train, Y_test = datasplit_ts(raw_fe, Y_colname, X_colname, '2012-07-01')


# # Applying Base Model: Y예측을 위한 Base분석 실행
# **1. Import Library**  
# **2. Data Loading**  [(Data Source and Description)](https://www.kaggle.com/c/bike-sharing-demand/data)  
# **3. Feature Engineering(Rearrange of Data)**  
# **4. Data Understanding(Descriptive Statistics and Getting Insight from Features)**  
# **5. Data Split: Train/Validate/Test Sets**  
# **6. Applying Base Model**  

# In[120]:


# description of train X
X_train.describe(include='all').T


# In[121]:


X_train.info()


# In[123]:


# LinearRegression (using statsmodels)
fit_reg1 = sm.OLS(Y_train, X_train).fit()
fit_reg1.summary()


# In[124]:


# X_train에 대한 예측치
fit_reg1.predict(X_train)


# In[125]:


# display(fit_reg1.predict(X_train)) # X_train에 대한 예측치

# display(fit_reg1.predict(X_test)) # X_test에 대한 예측치


pred_tr_reg1 = fit_reg1.predict(X_train).values
pred_te_reg1 = fit_reg1.predict(X_test).values


# ## Code Summary

# In[126]:


# LinearRegression (using statsmodels)
fit_reg1 = sm.OLS(Y_train, X_train).fit()
display(fit_reg1.summary())

pred_tr_reg1 = fit_reg1.predict(X_train).values
pred_te_reg1 = fit_reg1.predict(X_test).values


# # Evaluation: 분석 성능 확인/평가하기
# **1. Import Library**  
# **2. Data Loading**  [(Data Source and Description)](https://www.kaggle.com/c/bike-sharing-demand/data)  
# **3. Feature Engineering(Rearrange of Data)**  
# **4. Data Understanding(Descriptive Statistics and Getting Insight from Features)**  
# **5. Data Split: Train/Validate/Test Sets**  
# **6. Applying Base Model**  
# **7. Evaluation**  

# In[98]:


# pd.concat([Y_train, pd.DataFrame(pred_tr_reg1, index=Y_train.index, columns=['prediction'])], axis=1)


# In[99]:


# precision comparisions
pd.concat([Y_train, pd.DataFrame(pred_tr_reg1, index=Y_train.index, columns=['prediction'])], axis=1).plot(kind='line', figsize=(20,6),
                                                                                                               xlim=(Y_train.index.min(),Y_train.index.max()),
                                                                                                               linewidth=0.5, fontsize=20)
plt.title('Time Series of Target', fontsize=20)
plt.xlabel('Index', fontsize=15)
plt.ylabel('Target Value', fontsize=15)
plt.show()

MAE = abs(Y_train.values.flatten() - pred_tr_reg1).mean()
MSE = ((Y_train.values.flatten() - pred_tr_reg1)**2).mean()
MAPE = (abs(Y_train.values.flatten() - pred_tr_reg1)/Y_train.values.flatten()*100).mean()

display(pd.DataFrame([MAE, MSE, MAPE], index=['MAE', 'MSE', 'MAPE'], columns=['Score']).T)


# ## Code Summary

# In[100]:


### Functionalize
### Evaluation of 1 pair of set
def evaluation(Y_real, Y_pred, graph_on=False):
    loss_length = len(Y_real.values.flatten()) - len(Y_pred)
    if loss_length != 0:
        Y_real = Y_real[loss_length:]
    if graph_on == True:
        pd.concat([Y_real, pd.DataFrame(Y_pred, index=Y_real.index, columns=['prediction'])], axis=1).plot(kind='line', figsize=(20,6),
                                                                                                           xlim=(Y_real.index.min(),Y_real.index.max()),
                                                                                                           linewidth=3, fontsize=20)
        plt.title('Time Series of Target', fontsize=20)
        plt.xlabel('Index', fontsize=15)
        plt.ylabel('Target Value', fontsize=15)
    MAE = abs(Y_real.values.flatten() - Y_pred).mean()
    MSE = ((Y_real.values.flatten() - Y_pred)**2).mean()
    MAPE = (abs(Y_real.values.flatten() - Y_pred)/Y_real.values.flatten()*100).mean()
    Score = pd.DataFrame([MAE, MSE, MAPE], index=['MAE', 'MSE', 'MAPE'], columns=['Score']).T
    Residual = pd.DataFrame(Y_real.values.flatten() - Y_pred, index=Y_real.index, columns=['Error'])
    return Score, Residual
# Score_tr, Residual_tr = evaluation(Y_train, pred_tr_reg1, graph_on=True)


### Evaluation of train/test pairs
def evaluation_trte(Y_real_tr, Y_pred_tr, Y_real_te, Y_pred_te, graph_on=False):
    Score_tr, Residual_tr = evaluation(Y_real_tr, Y_pred_tr, graph_on=graph_on)
    Score_te, Residual_te = evaluation(Y_real_te, Y_pred_te, graph_on=graph_on)
    Score_trte = pd.concat([Score_tr, Score_te], axis=0)
    Score_trte.index = ['Train', 'Test']
    return Score_trte, Residual_tr, Residual_te
# Score_reg1, Resid_tr_reg1, Resid_te_reg1 = evaluation_trte(Y_train, pred_tr_reg1, Y_test, pred_te_reg1, graph_on=True)


# In[101]:


# results of evaluations
Score_reg1, Resid_tr_reg1, Resid_te_reg1 = evaluation_trte(Y_train, pred_tr_reg1, Y_test, pred_te_reg1, graph_on=True)
Score_reg1


# # Error Analysis: 분석모형이 데이터패턴을 모두 추출하여 분석을 종료해도 되는지 판단하기
# **1. Import Library**  
# **2. Data Loading**  [(Data Source and Description)](https://www.kaggle.com/c/bike-sharing-demand/data)  
# **3. Feature Engineering(Rearrange of Data)**  
# **4. Data Understanding(Descriptive Statistics and Getting Insight from Features)**  
# **5. Data Split: Train/Validate/Test Sets**  
# **6. Applying Base Model**  
# **7. Evaluation**  
# **8. Error Analysis**  

# In[102]:


# pd.Series(sm.tsa.stattools.adfuller(Resid_tr_reg1['Error'])
# sm.tsa.stattools.adfuller(Resid_tr_reg1['Error'])
# pd.Series(sm.tsa.stattools.adfuller(Resid_tr_reg1['Error'])[0:4], index=['Test Statistics', 'p-value', 'Used Lag', 'Used Observations'])


# In[103]:


# Error Analysis(Plot)
Resid_tr_reg1['RowNum'] = Resid_tr_reg1.reset_index().index

# Stationarity(Trend) Analysis
sns.set(palette="muted", color_codes=True, font_scale=2)
sns.lmplot(x='RowNum', y='Error', data=Resid_tr_reg1.iloc[1:],
           fit_reg='True', size=5.2, aspect=2, ci=99, sharey=True)

# Normal Distribution Analysis
figure, axes = plt.subplots(figsize=(12,8))
sns.distplot(Resid_tr_reg1['Error'], norm_hist='True', fit=stats.norm)

# Lag Analysis
length = int(len(Resid_tr_reg1['Error'])/10)
figure, axes = plt.subplots(1, 4, figsize=(12,3))
pd.plotting.lag_plot(Resid_tr_reg1['Error'], lag=1, ax=axes[0])
pd.plotting.lag_plot(Resid_tr_reg1['Error'], lag=5, ax=axes[1])
pd.plotting.lag_plot(Resid_tr_reg1['Error'], lag=10, ax=axes[2])
pd.plotting.lag_plot(Resid_tr_reg1['Error'], lag=50, ax=axes[3])

# Autocorrelation Analysis
figure, axes = plt.subplots(2,1,figsize=(12,5))
figure = sm.graphics.tsa.plot_acf(Resid_tr_reg1['Error'], lags=100, use_vlines=True, ax=axes[0])
figure = sm.graphics.tsa.plot_pacf(Resid_tr_reg1['Error'], lags=100, use_vlines=True, ax=axes[1])

# Error Analysis(Statistics)
# Checking Stationarity
# Null Hypothesis: The Time-series is non-stationalry
Stationarity = pd.Series(sm.tsa.stattools.adfuller(Resid_tr_reg1['Error'])[0:4], index=['Test Statistics', 'p-value', 'Used Lag', 'Used Observations'])
for key, value in sm.tsa.stattools.adfuller(Resid_tr_reg1['Error'])[4].items():
    Stationarity['Critical Value(%s)'%key] = value
Stationarity['Maximum Information Criteria'] = sm.tsa.stattools.adfuller(Resid_tr_reg1['Error'])[5]
Stationarity = pd.DataFrame(Stationarity, columns=['Stationarity'])

# Checking of Normality
# Null Hypothesis: The residuals are normally distributed
Normality = pd.DataFrame([stats.shapiro(Resid_tr_reg1['Error'])], index=['Normality'], columns=['Test Statistics', 'p-value']).T

# Checking for Autocorrelation
# Null Hypothesis: Autocorrelation is absent
Autocorrelation = pd.concat([pd.DataFrame(sm.stats.diagnostic.acorr_ljungbox(Resid_tr_reg1['Error'], lags=[1,5,10,50])[0], columns=['Test Statistics']),
                             pd.DataFrame(sm.stats.diagnostic.acorr_ljungbox(Resid_tr_reg1['Error'], lags=[1,5,10,50])[1], columns=['p-value'])], axis=1).T
Autocorrelation.columns = ['Autocorr(lag1)', 'Autocorr(lag5)', 'Autocorr(lag10)', 'Autocorr(lag50)']

# Checking Heteroscedasticity
# Null Hypothesis: Error terms are homoscedastic
Heteroscedasticity = pd.DataFrame([sm.stats.diagnostic.het_goldfeldquandt(Resid_tr_reg1['Error'], X_train.values, alternative='two-sided')],
                                  index=['Heteroscedasticity'], columns=['Test Statistics', 'p-value', 'Alternative']).T
Error_Analysis = pd.concat([Stationarity, Normality, Autocorrelation, Heteroscedasticity], join='outer', axis=1)
Error_Analysis = Error_Analysis.loc[['Test Statistics', 'p-value', 'Alternative', 'Used Lag', 'Used Observations',
                                     'Critical Value(1%)', 'Critical Value(5%)', 'Critical Value(10%)',
                                     'Maximum Information Criteria'],:]
Error_Analysis


# ## Code Summary

# In[104]:


### Functionalize
### Error analysis
def stationarity_adf_test(Y_Data, Target_name):
    if len(Target_name) == 0:
        Stationarity_adf = pd.Series(sm.tsa.stattools.adfuller(Y_Data)[0:4],
                                     index=['Test Statistics', 'p-value', 'Used Lag', 'Used Observations'])
        for key, value in sm.tsa.stattools.adfuller(Y_Data)[4].items():
            Stationarity_adf['Critical Value(%s)'%key] = value
            Stationarity_adf['Maximum Information Criteria'] = sm.tsa.stattools.adfuller(Y_Data)[5]
            Stationarity_adf = pd.DataFrame(Stationarity_adf, columns=['Stationarity_adf'])
    else:
        Stationarity_adf = pd.Series(sm.tsa.stattools.adfuller(Y_Data[Target_name])[0:4],
                                     index=['Test Statistics', 'p-value', 'Used Lag', 'Used Observations'])
        for key, value in sm.tsa.stattools.adfuller(Y_Data[Target_name])[4].items():
            Stationarity_adf['Critical Value(%s)'%key] = value
            Stationarity_adf['Maximum Information Criteria'] = sm.tsa.stattools.adfuller(Y_Data[Target_name])[5]
            Stationarity_adf = pd.DataFrame(Stationarity_adf, columns=['Stationarity_adf'])
    return Stationarity_adf

def stationarity_kpss_test(Y_Data, Target_name):
    if len(Target_name) == 0:
        Stationarity_kpss = pd.Series(sm.tsa.stattools.kpss(Y_Data)[0:3],
                                      index=['Test Statistics', 'p-value', 'Used Lag'])
        for key, value in sm.tsa.stattools.kpss(Y_Data)[3].items():
            Stationarity_kpss['Critical Value(%s)'%key] = value
            Stationarity_kpss = pd.DataFrame(Stationarity_kpss, columns=['Stationarity_kpss'])
    else:
        Stationarity_kpss = pd.Series(sm.tsa.stattools.kpss(Y_Data[Target_name])[0:3],
                                      index=['Test Statistics', 'p-value', 'Used Lag'])
        for key, value in sm.tsa.stattools.kpss(Y_Data[Target_name])[3].items():
            Stationarity_kpss['Critical Value(%s)'%key] = value
            Stationarity_kpss = pd.DataFrame(Stationarity_kpss, columns=['Stationarity_kpss'])
    return Stationarity_kpss

def error_analysis(Y_Data, Target_name, X_Data, graph_on=False):
    for x in Target_name:
        Target_name = x
    X_Data = X_Data.loc[Y_Data.index]

    if graph_on == True:
        ##### Error Analysis(Plot)
        Y_Data['RowNum'] = Y_Data.reset_index().index

        # Stationarity(Trend) Analysis
        sns.set(palette="muted", color_codes=True, font_scale=2)
        sns.lmplot(x='RowNum', y=Target_name, data=Y_Data, fit_reg='True', size=5.2, aspect=2, ci=99, sharey=True)
        del Y_Data['RowNum']

        # Normal Distribution Analysis
        figure, axes = plt.subplots(figsize=(12,8))
        sns.distplot(Y_Data[Target_name], norm_hist='True', fit=stats.norm, ax=axes)

        # Lag Analysis
        length = int(len(Y_Data[Target_name])/10)
        figure, axes = plt.subplots(1, 4, figsize=(12,3))
        pd.plotting.lag_plot(Y_Data[Target_name], lag=1, ax=axes[0])
        pd.plotting.lag_plot(Y_Data[Target_name], lag=5, ax=axes[1])
        pd.plotting.lag_plot(Y_Data[Target_name], lag=10, ax=axes[2])
        pd.plotting.lag_plot(Y_Data[Target_name], lag=50, ax=axes[3])

        # Autocorrelation Analysis
        figure, axes = plt.subplots(2,1,figsize=(12,5))
        sm.tsa.graphics.plot_acf(Y_Data[Target_name], lags=100, use_vlines=True, ax=axes[0])
        sm.tsa.graphics.plot_pacf(Y_Data[Target_name], lags=100, use_vlines=True, ax=axes[1])

    ##### Error Analysis(Statistics)
    # Checking Stationarity
    # Null Hypothesis: The Time-series is non-stationalry
    Stationarity_adf = stationarity_adf_test(Y_Data, Target_name)
    Stationarity_kpss = stationarity_kpss_test(Y_Data, Target_name)

    # Checking of Normality
    # Null Hypothesis: The residuals are normally distributed
    Normality = pd.DataFrame([stats.shapiro(Y_Data[Target_name])],
                             index=['Normality'], columns=['Test Statistics', 'p-value']).T

    # Checking for Autocorrelation
    # Null Hypothesis: Autocorrelation is absent
    Autocorrelation = pd.concat([pd.DataFrame(sm.stats.diagnostic.acorr_ljungbox(Y_Data[Target_name], lags=[1,5,10,50])[0], columns=['Test Statistics']),
                                 pd.DataFrame(sm.stats.diagnostic.acorr_ljungbox(Y_Data[Target_name], lags=[1,5,10,50])[1], columns=['p-value'])], axis=1).T
    Autocorrelation.columns = ['Autocorr(lag1)', 'Autocorr(lag5)', 'Autocorr(lag10)', 'Autocorr(lag50)']

    # Checking Heteroscedasticity
    # Null Hypothesis: Error terms are homoscedastic
    Heteroscedasticity = pd.DataFrame([sm.stats.diagnostic.het_goldfeldquandt(Y_Data[Target_name], X_Data.values, alternative='two-sided')],
                                      index=['Heteroscedasticity'], columns=['Test Statistics', 'p-value', 'Alternative']).T
    Score = pd.concat([Stationarity_adf, Stationarity_kpss, Normality, Autocorrelation, Heteroscedasticity], join='outer', axis=1)
    index_new = ['Test Statistics', 'p-value', 'Alternative', 'Used Lag', 'Used Observations',
                 'Critical Value(1%)', 'Critical Value(5%)', 'Critical Value(10%)', 'Maximum Information Criteria']
    Score.reindex(index_new)
    return Score
# error_analysis(Resid_tr_reg1[1:], ['Error'], X_train, graph_on=True)


# In[105]:


error_analysis(Resid_tr_reg1, ['Error'], X_train, graph_on=True)


# # Summary: insufficient for me
# **1) 데이터 핸들링**  
# 
# **2) 단계이해**  
# >**1. Import Library**  
# **2. Data Loading**  [(Data Source and Description)](https://www.kaggle.com/c/bike-sharing-demand/data)  
# **3. Feature Engineering(Rearrange of Data)**  
# **4. Data Understanding(Descriptive Statistics and Getting Insight from Features)**  
# **5. Data Split: Train/Validate/Test Sets**  
# **6. Applying Base Model**  
# **7. Evaluation**  
# **8. Error Analysis** 
# 
# **3) 결과 해석**  

# ## Code Summary (Raw Data)

# In[106]:


# Data Loading
# location = 'https://raw.githubusercontent.com/cheonbi/DataScience/master/Data/Bike_Sharing_Demand_Full.csv'
location = './Data/BikeSharingDemand/Bike_Sharing_Demand_Full.csv'
raw_all = pd.read_csv(location)

# Feature Engineering
raw_rd = non_feature_engineering(raw_all)

# Data Split
# Confirm of input and output
Y_colname = ['count']
X_remove = ['datetime', 'DateTime', 'temp_group', 'casual', 'registered']
X_colname = [x for x in raw_rd.columns if x not in Y_colname+X_remove]
X_train_rd, X_test_rd, Y_train_rd, Y_test_rd = datasplit_ts(raw_rd, Y_colname, X_colname, '2012-07-01')

# Applying Base Model
fit_reg1_rd = sm.OLS(Y_train_rd, X_train_rd).fit()
display(fit_reg1_rd.summary())
pred_tr_reg1_rd = fit_reg1_rd.predict(X_train_rd).values
pred_te_reg1_rd = fit_reg1_rd.predict(X_test_rd).values

# Evaluation
Score_reg1_rd, Resid_tr_reg1_rd, Resid_te_reg1_rd = evaluation_trte(Y_train_rd, pred_tr_reg1_rd, 
                                                                Y_test_rd, pred_te_reg1_rd, graph_on=True)
display(Score_reg1_rd)

# Error Analysis
error_analysis(Resid_tr_reg1_rd, ['Error'], X_train_rd, graph_on=True)


# ## Code Summary (Feature Engineering Data)

# In[107]:


# Data Loading
# location = 'https://raw.githubusercontent.com/cheonbi/DataScience/master/Data/Bike_Sharing_Demand_Full.csv'
location = './Data/BikeSharingDemand/Bike_Sharing_Demand_Full.csv'
raw_all = pd.read_csv(location)

# Feature Engineering
raw_fe = feature_engineering(raw_all)

# Data Split
# Confirm of input and output
Y_colname = ['count']
X_remove = ['datetime', 'DateTime', 'temp_group', 'casual', 'registered']
X_colname = [x for x in raw_fe.columns if x not in Y_colname+X_remove]
X_train_fe, X_test_fe, Y_train_fe, Y_test_fe = datasplit_ts(raw_fe, Y_colname, X_colname, '2012-07-01')

# Applying Base Model
fit_reg1_fe = sm.OLS(Y_train_fe, X_train_fe).fit()
display(fit_reg1_fe.summary())
pred_tr_reg1_fe = fit_reg1_fe.predict(X_train_fe).values
pred_te_reg1_fe = fit_reg1_fe.predict(X_test_fe).values

# Evaluation
Score_reg1_fe, Resid_tr_reg1_fe, Resid_te_reg1_fe = evaluation_trte(Y_train_fe, pred_tr_reg1_fe,
                                                                Y_test_fe, pred_te_reg1_fe, graph_on=True)
display(Score_reg1_fe)

# Error Analysis
error_analysis(Resid_tr_reg1_fe, ['Error'], X_train_fe, graph_on=True)


# In[ ]:




