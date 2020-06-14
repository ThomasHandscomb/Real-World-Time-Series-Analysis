######################################
## Title: Time Series Exploratory Work
## Author: Thomas Handscomb
######################################

# Import libraries
import pandas as pd
import numpy as np
import math

# Calculate RMSE of preds against test set
from sklearn.metrics import mean_squared_error
from math import sqrt

import matplotlib.pyplot as plt

from dateutil.relativedelta import relativedelta

import statsmodels.tsa.stattools
import statsmodels.graphics.tsaplots as tsaplot
from statsmodels.tsa.stattools import adfuller
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import scipy.stats as scs


# Control the number of columns displayed in the console output
pd.set_option('display.max_columns', 10)

# Control the width of columns displayed in the console output
pd.set_option('display.width', 1000)

# Load real time series data into the session. Note that the date column used for index is parsed separately 
# as a date after  
df_TS_Real = pd.read_csv('C:/Users/Tom/Desktop/GitHub Page/Blog Repos/Real-World-Time-Series-Analysis/SectorSales.csv'
                     , encoding = "ISO-8859-1", header=0
                     , usecols = ['Date','Sector', 'Universe Gross Sales EUR M', 'Universe Net Sales EUR M']                      
                     #, parse_dates=["Date"]
                     , index_col = ["Date"])

# Care needs to be shown with indexes when working with Time Series, can set the Date column to be the index
df_TS_Real.index
df_TS_Real.index = pd.to_datetime(df_TS_Real.index, format="%d/%m/%Y")

df_TS_Real.columns

ts_col_names = ['Sector', 'Market_Gross_Sales', 'Market_Net_Sales']
df_TS_Real.columns = ts_col_names
df_TS_Real.columns

# Create a collection of data frames to be used going forward:
# First specify the list 
Sector_List = [sector for sector in df_TS_Real['Sector'].unique()]
type(Sector_List)
len(Sector_List)

# Use list comprehension to create a dictionary of data frames
d = {}
d = {sector: df_TS_Real[df_TS_Real['Sector'] == sector] for sector in Sector_List}
type(d)

# The keys are the sector names
d.keys()
d.values()
# Can then use the keys to specify sectors
# View the data frame corresponding to a sector, pick a sector: Sector_95

type(d['Sector_95'])
d['Sector_95'].shape[0]

d['Sector_95'].columns
d['Sector_95'].index
d['Sector_95'].head(10)

# Specify the frequency of the time series
#d['Sector_95'].index = pd.DatetimeIndex(d['Sector_95'].index).to_period('M')

# Can view output
#d['Sector_95'].to_csv('C:/Users/Tom/Desktop/Python Code Example/ts.csv'
#               , encoding = 'utf-8'
#               #, mode = 'a'
#               , index = True
#               , header = True)

#~~~~~~~~~~~~~~~~~
# Plot with legend
#~~~~~~~~~~~~~~~~~
plt.figure(figsize=(15,7.5))
d['Sector_95']['Market_Net_Sales'].plot(kind = 'line', color = 'blue')
#d['Sector_95'][(d['Sector_95'].index >= '2018-01-01')]['Market_Net_Sales'].plot(kind='line', marker = 'o', color = 'green')
plt.legend(d['Sector_95']['Sector'], loc='upper right')
plt.show()

# Now have specific dataframes lined up to sector, can implement some TS
##########################
# 1. Test for Stationarity
##########################

def test_stationarity(timeseries):
    
    #Determing rolling statistics
    #rolmean = pd.rolling_mean(timeseries, window=12)
    #rolstd = pd.rolling_std(timeseries, window=12)
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = timeseries.rolling(window=12).std()

    #Plot rolling statistics:
    plt.figure(figsize=(15,7.5))
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.grid(True)
    plt.show(block=False)
    
    #Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','# Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)' %(key)] = value
    print (dfoutput)


test_stationarity(d['Sector_95']['Market_Net_Sales'])
# It's difficult to determine a trend from the visual plot, the 
# Appears stationary (p=0.000602)

############################
# 2. Look at Autocorrelation
############################

def tsplot(y, lags=None, figsize=(15, 8), style='bmh'):
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        #mpl.rcParams['font.family'] = 'Ubuntu Mono'
        layout = (3, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        qq_ax = plt.subplot2grid(layout, (2, 0))
        pp_ax = plt.subplot2grid(layout, (2, 1))
        
        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, alpha=0.5)
        sm.qqplot(y, line='s', ax=qq_ax)
        qq_ax.set_title('QQ Plot')        
        scs.probplot(y, sparams=(y.mean(), y.std()), plot=pp_ax)

        plt.tight_layout()
    return 

# View the combined charts on the timeseries, there appears some autocorrelation
# with a lag of 1 or 2 months
tsplot(d['Sector_95']['Market_Net_Sales'], lags=30)

# View the distribution of monthly sales (stationary time series should be approx. normal)
d['Sector_95']['Market_Net_Sales'].hist(bins = 30)

# TO FIND: WHAT THE ALPHA VALUE REFERS TO HERE
statsmodels.graphics.tsaplots.plot_acf(d['Sector_95']['Market_Net_Sales'], lags = 30, alpha = 0.05)

# Compate to plots of discrete white noise
randser = np.random.normal(size=100)
tsplot(randser)


#############################################################
# Now have determined stationarity, build a forecasting model
#############################################################

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Segment the time series into a collection of train/test chucks
# This produces a series of training sets of increasing size for the 
# 'evaluation on a rolling forecasting origin' method used in ARIMA models
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def ts_splitter_combined_fixed(ts_dataframe, train_test_fixed, plot_on = 'None'):
    ts_dataframe_rows = ts_dataframe.shape[0]
    # Decompose the time series into training and testing sets according to a rolling testing origin
    start_point = ts_dataframe_rows - train_test_fixed
    test_size = train_test_fixed
    
    counter = 0
    
    # Ensure you use at least half of the time series
    while start_point >= math.floor(ts_dataframe_rows/2):
        #103:
        counter = counter+1

        print("Counter: %i, Train End Point: %i, Test Size: %i" %(counter, start_point, test_size))
        
        # Do the time series decomposition here
        train_df = ts_dataframe[0:start_point]
        test_df = ts_dataframe[start_point: start_point+test_size]
        # Print max/min dates to ensure no gap or overlap of time series
        #print(train_df.index.max(), test_df.index.min())
        # Print the different time series components
        if plot_on == 'Yes':
            ts_dataframe['Market_Net_Sales'].plot(kind='line', color = 'red', label = 'Total')
            train_df['Market_Net_Sales'].plot(kind='line', color = 'green', label = 'Training set')
            test_df['Market_Net_Sales'].plot(kind='line', color = 'blue', label = 'Testing set')
            plt.legend(loc='upper right')
            plt.show()        
        else:
            pass
        
        yield counter, train_df, test_df
        #Roll back next start point
        start_point = start_point-test_size

############################################
# Specify the size of the forward prediction
############################################
# Define the output of the loop as a list of tuples, 
ts_loop_output = list(ts_splitter_combined_fixed(d['Sector_95'], train_test_fixed = 3))

type(ts_loop_output)
# There are a number tuples in the list depending on the 'while star_point' specification
len(ts_loop_output)

# Each tuple has 3 elements, 1 integer (counter) and 2 DataFrames (train_df and test_df)
len(ts_loop_output[0]) # 3

ts_loop_output[0][0]
ts_loop_output[0][1].shape[0]
ts_loop_output[0][2]

# Check that the starting point for each training block is the same, and that the size gets
# longer by 3 months each time. This is to verify the 'evaluation on a rolling forecasting origin' method
for block in range(0, len(ts_loop_output)):
    # Print the block number
    print("Block %i, Start Point %s, Length %i" \
          %(block, ts_loop_output[block][1].index.min(), ts_loop_output[block][1].shape[0]))
          

#~~~~~~~~~~~~~~~~~~~~~~
# Different model types
#~~~~~~~~~~~~~~~~~~~~~~

# Create a function to plot predictions vs. testing
def ts_validate_plots(ts_full, ts_training, ts_predictions, ts_testing):
    
    #Plot rolling statistics:
    plt.figure(figsize=(16,8))
    plt.plot(ts_full, color='black', marker='o', label='Full Set')    
    plt.plot(ts_training, color='blue', marker='o', label='Training')
    plt.plot(ts_predictions, color='red', marker='o', label='Predictions')
    plt.plot(ts_testing, color='green', marker='o', label = 'Testing')
    plt.legend(loc='best')
    plt.title('Training, Prediction and Testing')
    plt.grid(True)
    plt.show(block=False)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1. Look at the naive model
#~~~~~~~~~~~~~~~~~~~~~~~~~~~
def naive_model(training, testing, plot_on = 'None'):
    predictions = pd.DataFrame()
    history = training
    test = testing    
    for i in range(len(test)):
        #print(i)
        # Make prediction based on previous value
        yhat = history[-1:]
        # Increase the index by 1 month so that the prediction index aligns with the testing index for that month
        yhat.index = pd.to_datetime(yhat.index.date + relativedelta(months=+1) , format="%Y-%m-%d")
        predictions = predictions.append(yhat)
        # Append the prediction observation to the historical to preserve the forward looking 3 months
        history = history.append(yhat)
        # The below appends the training data to the set
        #history = history.append(test[i:i+1])        
        
    # Print the performance and graphs (outside of the loop)
    rmse_naive = sqrt(mean_squared_error(test['Market_Net_Sales'], predictions['Market_Net_Sales']))
    naive_model.var = rmse_naive
    print('Naive model RMSE: %.3f' % rmse_naive)
    #print(predictions['Market_Net_Sales'])
    if plot_on == 'Yes':
        ts_validate_plots(ts_full = d['Sector_95']['Market_Net_Sales'], ts_training = training, ts_predictions = predictions['Market_Net_Sales'], ts_testing = testing)
    else:
        pass

# Test the function on the last (i.e. right most) chunk
naive_model(training = ts_loop_output[0][1][['Market_Net_Sales']]
                        , testing = ts_loop_output[0][2][['Market_Net_Sales']]
                        , plot_on = 'Yes')

# Test the function on another chunk
naive_model(ts_loop_output[10][1][['Market_Net_Sales']]
                        , ts_loop_output[10][2][['Market_Net_Sales']]
                        , plot_on = 'Yes')


ts_loop_output[0][1][['Market_Net_Sales']].rolling(window=6).mean()[-1:]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 2. Look at a range of moving average models
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def roll_average_model(training, testing, window, plot_on = 'None'):
    predictions = pd.DataFrame()
    history = training
    test = testing    
    for i in range(len(test)):
        #print(i)
        # Make prediction based on rolling window average
        yhat = history.rolling(window=window).mean()[-1:]
        # Increase the index by 1 month so that the prediction index aligns with the testing index for that month
        yhat.index = pd.to_datetime(yhat.index.date + relativedelta(months=+1) , format="%Y-%m-%d")
        predictions = predictions.append(yhat)
        # Append the prediction observation to the historical to preserve the forward looking 3 months
        history = history.append(yhat)
        # The below appends the training data to the set
        #history = history.append(test[i:i+1])        
        
    # Print the performance and graphs (outside of the loop)
    rmse_rollaverage = sqrt(mean_squared_error(test['Market_Net_Sales'], predictions['Market_Net_Sales']))
    roll_average_model.var = rmse_rollaverage
    print('Moving average RMSE: %.3f' % rmse_rollaverage)
    #print(predictions['Market_Net_Sales'])
    if plot_on == 'Yes':
        ts_validate_plots(ts_full = d['Sector_95']['Market_Net_Sales'], ts_training = training, ts_predictions = predictions['Market_Net_Sales'], ts_testing = testing)
    else:
        pass
    return predictions
    #yield rmse_rollaverage

# Test the function on the last (i.e. right most) chunk with a 6 month rolling window
roll_average_model(ts_loop_output[0][1][['Market_Net_Sales']]
                    , ts_loop_output[0][2][['Market_Net_Sales']]
                    , window = 6
                    , plot_on = 'Yes')

# Of course a 1 month moving average window is simply the naive model
chunk = 8
for chunk in range(9):
    print(chunk)
    roll_average_model(ts_loop_output[chunk][1][['Market_Net_Sales']]
                        , ts_loop_output[chunk][2][['Market_Net_Sales']]
                        , window = 1
                        , plot_on = 'No')
    
    naive_model(training = ts_loop_output[chunk][1][['Market_Net_Sales']]
                        , testing = ts_loop_output[chunk][2][['Market_Net_Sales']]
                        , plot_on = 'No')

# Test the function on a different chunk
roll_average_model(ts_loop_output[chunk][1][['Market_Net_Sales']]
                    , ts_loop_output[chunk][2][['Market_Net_Sales']]
                    , window = 10
                    , plot_on = 'Yes')

#~~~~~~~~~~~~~~~~~~~~~~~~
# 3. Look at ARIMA models
#~~~~~~~~~~~~~~~~~~~~~~~~
from statsmodels.tsa.arima_model import ARIMA

# Fit model
arima_model = ARIMA(ts_loop_output[0][1][['Market_Net_Sales']], order=(1,0,0))
model_fit = arima_model.fit()
print(model_fit.summary())

# Look at forecasts and predictions
step_forecast = pd.DataFrame(model_fit.forecast()[0])

ts_loop_output[0][1][['Market_Net_Sales']].tail(1).index
ts_loop_output[0][1][['Market_Net_Sales']][-1:]

step_forecast.index = ts_loop_output[0][1]['Market_Net_Sales'][[-1]].index
step_forecast.index = pd.to_datetime(step_forecast.index.date + relativedelta(months=+1) , format="%Y-%m-%d")

ts_loop_output[0][1][['Market_Net_Sales']].shape[0]

model_fit.predict(120,122)
model_fit.fittedvalues

# Plot fitted values against actual
plt.plot(model_fit.predict(), label = 'Prediction')
plt.plot(ts_loop_output[0][1]['Market_Net_Sales'], label = 'Actual')
plt.legend()
plt.show()

# plot residual errors
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()

residuals.hist()
residuals.plot(kind='kde')
residuals.describe()

# Calculate RMSE
sqrt(mean_squared_error(ts_loop_output[0][1]['Market_Net_Sales']
                        , model_fit.predict()))




model = ARIMA(ts_loop_output[0][1][['Market_Net_Sales']], order=(0,0,1))
model_fit = model.fit(disp=0)
output = pd.DataFrame(model_fit.forecast()[0])
output.index = ts_loop_output[0][1][['Market_Net_Sales']][-1:].index
output.columns = ts_loop_output[0][1][['Market_Net_Sales']].columns


# Set up a manual step through
i = 0
predictions = pd.DataFrame()
history = ts_loop_output[0][1][['Market_Net_Sales']]
test = ts_loop_output[0][2][['Market_Net_Sales']]

# No need to iteratively rebuild the model each time, just build on training and predict forward
def ARIMA_model(training, testing, plot_on = 'None'):
    predictions = pd.DataFrame()
    history = training
    test = testing    

    # Make forward predictions based on ARIMA model
    start_int = history.shape[0]
    end_int = history.shape[0]+test.shape[0]-1
    
    arima_model = ARIMA(history, order = (1,0,0))
    arima_model_fit = arima_model.fit()
    # Predict forward based on start and end date
    yhat = pd.DataFrame(arima_model_fit.predict(start_int, end_int))
    #arima_model_fit.fittedvalues
    #yhat.index = history[-1:].index
    yhat.columns = history.columns
    
    predictions = yhat
            
    # Print the performance and graphs (outside of the loop)
    rmse_arima = sqrt(mean_squared_error(test['Market_Net_Sales'], predictions['Market_Net_Sales']))
    ARIMA_model.var = rmse_arima
    print('RMSE: %.3f' % rmse_arima)
    #print(predictions['Market_Net_Sales'])
    if plot_on == 'Yes':
        ts_validate_plots(ts_full = d['Sector_95']['Market_Net_Sales'], ts_training = training, ts_predictions = predictions['Market_Net_Sales'], ts_testing = testing)
    else:
        pass
    return predictions



#def ARIMA_model(training, testing, plot_on = 'None'):
#    predictions = pd.DataFrame()
#    history = training
#    test = testing    
#    for i in range(len(test)):
#        #print(i)
#        # Make prediction based on ARIMA model
#        arima_model = ARIMA(history, order = (1,0,0))
#        arima_model_fit = arima_model.fit()
#        #print(arima_model_fit.summary())
#        yhat = pd.DataFrame(arima_model_fit.forecast()[0])
#        #arima_model_fit.fittedvalues
#        yhat.index = history[-1:].index
#        yhat.columns = history.columns
#        
#        # Increase the index by 1 month so that the prediction index aligns with the testing index for that month
#        yhat.index = pd.to_datetime(yhat.index.date + relativedelta(months=+1) , format="%Y-%m-%d")
#        predictions = predictions.append(yhat)
#        # Append the prediction observation to the historical to preserve the forward looking 3 months
#        history = history.append(yhat)
#        # The below appends the training data to the set
#        #history = history.append(test[i:i+1])        
#        
#    # Print the performance and graphs (outside of the loop)
#    rmse_arima = sqrt(mean_squared_error(test['Market_Net_Sales'], predictions['Market_Net_Sales']))
#    ARIMA_model.var = rmse_arima
#    print('RMSE: %.3f' % rmse_arima)
#    #print(predictions['Market_Net_Sales'])
#    if plot_on == 'Yes':
#        ts_validate_plots(ts_full = d['Sector_95']['Market_Net_Sales'], ts_training = training, ts_predictions = predictions['Market_Net_Sales'], ts_testing = testing)
#    else:
#        pass
#    return predictions

# Test the function on the last (i.e. right most) chunk
import warnings
warnings.filterwarnings("ignore")    

ARIMA_model(ts_loop_output[0][1][['Market_Net_Sales']]
                    , ts_loop_output[0][2][['Market_Net_Sales']]
                    , plot_on = 'Yes')

chunk = 11
ARIMA_model(ts_loop_output[chunk][1][['Market_Net_Sales']]
                    , ts_loop_output[chunk][2][['Market_Net_Sales']]
                    , plot_on = 'Yes')

test_stationarity(ts_loop_output[11][1]['Market_Net_Sales'])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 4. Look at deep learning models
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This is constructed below: nn_model


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Once all models built, implement time series cross validation process is then as follows
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Pick a model type first and then loop through the time series blocks  

#~~~~~~~~~~~~
# NAIVE MODEL 
#~~~~~~~~~~~~
# Pass the naive model across the ts blocks
df_RMSE_Naive = pd.DataFrame(columns=(['Model_Type', 'ts_Block', 'RMSE']))
df_RMSE_Naive.columns
df_RMSE_Naive.index

for t in range(0, len(ts_loop_output)):
    #print("TS block %i" %t)
    naive_model(ts_loop_output[t][1][['Market_Net_Sales']], ts_loop_output[t][2][['Market_Net_Sales']], plot_on = 'Yes')
    #print(i, roll_average_model.var)
    #Append data to dataframe
    df_RMSE_Naive = df_RMSE_Naive.append(pd.DataFrame({'Model_Type':["Naive"], 'ts_Block':[t], 'RMSE':[naive_model.var]}))

df_RMSE_Naive
# Calculate model statistics
df_RMSE_Naive.groupby(['Model_Type']).describe()

#~~~~~~~~~~~~~~~~~~~~~~
# Rolling average model
#~~~~~~~~~~~~~~~~~~~~~~
df_RMSE_RollAvg = pd.DataFrame(columns=(['Model_Type', 'ts_Block', 'RMSE']))
df_RMSE_RollAvg.columns
df_RMSE_RollAvg.index

for i in (3,6,9,12):
    print ("Model Type %i" %i)
    # Then run the model type through the time series blocks
    for t in range(0, len(ts_loop_output)):
        #print("TS block %i" %t)
        roll_average_model(ts_loop_output[t][1][['Market_Net_Sales']], ts_loop_output[t][2][['Market_Net_Sales']], window = i)
        #print(i, roll_average_model.var)
        #Append data to dataframe
        df_RMSE_RollAvg = df_RMSE_RollAvg.append(pd.DataFrame({'Model_Type':['Rolling %i month average' %i], 'ts_Block':[t], 'RMSE':[roll_average_model.var]}))

# Calculate model statistics
df_RMSE_RollAvg.groupby(['Model_Type']).mean()
df_RMSE_RollAvg.groupby(['Model_Type']).describe()

# Output df_RMSE to review
#df_RMSE_RollAvg.to_csv('L:/My Documents/2019/Time Series Analysis/Model Output Data/df_RMSE.csv'
#               , encoding = 'utf-8'
#               #, mode = 'a'
#               , index = False
#               , header = True)

#~~~~~~~~~~~~
# ARIMA MODEL
#~~~~~~~~~~~~
# Pass the naive model across the ts blocks
df_RMSE_Arima = pd.DataFrame(columns=(['Model_Type', 'ts_Block', 'RMSE']))
df_RMSE_Arima.columns
df_RMSE_Arima.index

for t in range(0, len(ts_loop_output)):
    #print("TS block %i" %t)
    ARIMA_model(ts_loop_output[t][1][['Market_Net_Sales']], ts_loop_output[t][2][['Market_Net_Sales']], plot_on = 'Yes')
    #print(i, roll_average_model.var)
    #Append data to dataframe
    df_RMSE_Arima = df_RMSE_Arima.append(pd.DataFrame({'Model_Type':["Arima"], 'ts_Block':[t], 'RMSE':[ARIMA_model.var]}))

df_RMSE_Arima
# Calculate model statistics
df_RMSE_Arima.groupby(['Model_Type']).mean()
df_RMSE_Arima.groupby(['Model_Type']).describe()


#~~~~~~~~~~~~~~~~~
# Neural Net Model
#~~~~~~~~~~~~~~~~~
# Pass the neural net model across the ts blocks
df_RMSE_nn = pd.DataFrame(columns=(['Model_Type', 'ts_Block', 'RMSE']))
df_RMSE_nn.columns
#df_RMSE_nn.index

help(print)

# Just calculate on the blocks not used in the training process
for block in range(1, len(ts_fixed_loop_output),2):
    #print(i)
    yhat = nn_model.predict(nn_ts_X_df_Validate.loc[[block]], verbose=0)
    y_0 = nn_ts_y_df_Validate.loc[[block]]
    rmse_nn = sqrt(mean_squared_error(y_0, yhat))
    print(block, rmse_nn)
    df_RMSE_nn = df_RMSE_nn.append(pd.DataFrame({'Model_Type':["Neural Net"], 'ts_Block':[block], 'RMSE':[rmse_nn]}))

df_RMSE_nn
# Calculate model statistics
df_RMSE_nn.groupby(['Model_Type']).describe()

#nn_ts_X_df_Validate.loc[[9]]





#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# COMPARISON OF MODEL PERFORMANCE
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# 1. Look at Naive model across cross-validation chunks
df_RMSE_Naive.groupby(['Model_Type']).describe()

# 2. Look at range of rolling average models across cross-validation chunks
df_RMSE_RollAvg.groupby(['Model_Type']).describe()

# 3. ARIMA models
df_RMSE_Arima.groupby(['Model_Type']).describe()

# 4. Look at nn model across cross-validation chunks
df_RMSE_nn.groupby(['Model_Type']).describe()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BUILDING NEURAL NETWORK MODEL BELOW
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Neural net requires a fixed input and output shape

# First decompose time series into a fixed input and output shape
def ts_splitter_fixed_input_output(ts_dataframe, train_fixed, test_fixed, plot_on = 'None'):
    ts_dataframe_rows = ts_dataframe.shape[0]
    # Decompose the time series into training and testing sets according to the train and test parameters
    start_point = ts_dataframe_rows - train_fixed - test_fixed
    test_size = test_fixed
    
    counter = 0
    
    # Ensure you use as much of the time series as possible
    while start_point >= 0:
        #103:
        counter = counter+1

        print("Counter: %i, Start Point: %i, Train Size: %i, Test Size: %i" %(counter, start_point, train_fixed, test_size))
        
        # Do the time series decomposition here
        train_df = ts_dataframe[start_point:start_point+train_fixed]
        test_df = ts_dataframe[start_point+train_fixed: start_point+train_fixed+test_size]
        # Print max/min dates to ensure no gap or overlap of time series
        #print(train_df.index.max(), test_df.index.min())
        # Print the different time series components
        if plot_on == 'Yes':
            plt.figure(figsize=(14,8))
            ts_dataframe['Market_Net_Sales'].plot(kind='line', color = 'red', marker='o', label = 'Total')
            train_df['Market_Net_Sales'].plot(kind='line', color = 'green', marker='o', label = 'Training set')
            test_df['Market_Net_Sales'].plot(kind='line', color = 'blue', marker='o', label = 'Testing set')
            plt.legend(loc='upper right')
            plt.show()        
        else:
            pass
        
        yield counter, train_df, test_df
        start_point = start_point-train_fixed- test_size

# Define the output of the loop as a list of tuples
ts_fixed_loop_output = list(ts_splitter_fixed_input_output(d['Sector_95'], train_fixed = 6, test_fixed = 3))
ts_fixed_loop_output = list(ts_splitter_fixed_input_output(d['Sector_95'], train_fixed = 6, test_fixed = 3, plot_on = 'Yes'))

type(ts_fixed_loop_output)
len(ts_fixed_loop_output)
len(ts_fixed_loop_output[0])

ts_fixed_loop_output[0][0]
ts_fixed_loop_output[0][1]
ts_fixed_loop_output[0][2]

# Build the initial neural net on a stack of these DataFrames: see below

#~~~~~~~~~~~~~~~~~~
# Build up the X_df
#~~~~~~~~~~~~~~~~~~
nnts_colnames = ['M6', 'M5', 'M4', 'M3', 'M2', 'M1', 'Block']
# This is the collection of dataframes used for training
nn_ts_X_df = pd.DataFrame(columns=nnts_colnames)
# This is the collection of dataframes used for testing
nn_ts_X_df_Validate = pd.DataFrame(columns=nnts_colnames)
# This is the total collection of dataframes used
nn_ts_X_df_Total = pd.DataFrame(columns=nnts_colnames)

# Build the collection of blocks for model building, testing and displaying
for block in range(0, len(ts_fixed_loop_output)):
    #print(block)
    df_temp = ts_fixed_loop_output[block][1][['Market_Net_Sales']].T
    df_temp['Block'] = block
    df_temp.columns = nnts_colnames
    nn_ts_X_df_Total = nn_ts_X_df_Total.append(df_temp)
    if block % 2 ==0:
        nn_ts_X_df = nn_ts_X_df.append(df_temp)
    else: 
        nn_ts_X_df_Validate = nn_ts_X_df_Validate.append(df_temp)

# Set the block number as index
nn_ts_X_df.set_index('Block', inplace = True)
nn_ts_X_df_Validate.set_index('Block', inplace = True)
nn_ts_X_df_Total.set_index('Block', inplace = True)

# Do some checking
ts_fixed_loop_output[10][1][['Market_Net_Sales']].T
ts_fixed_loop_output[12][1][['Market_Net_Sales']].T

#~~~~~~~~~~~~~~~~~~
# Build up the y_df
#~~~~~~~~~~~~~~~~~~
nnts_y_df_colnames = ['F1', 'F2', 'F3', 'Block']
nn_ts_y_df = pd.DataFrame(columns = nnts_y_df_colnames)
nn_ts_y_df_Validate = pd.DataFrame(columns = nnts_y_df_colnames)
nn_ts_y_df_Total = pd.DataFrame(columns = nnts_y_df_colnames)

for block in range(0, len(ts_fixed_loop_output)):
    df_temp = ts_fixed_loop_output[block][2][['Market_Net_Sales']].T
    df_temp['Block'] = block
    df_temp.columns = nnts_y_df_colnames
    nn_ts_y_df_Total = nn_ts_y_df_Total.append(df_temp)
    if block % 2 ==0:
        nn_ts_y_df = nn_ts_y_df.append(df_temp)
    else:
        nn_ts_y_df_Validate = nn_ts_y_df_Validate.append(df_temp)

# Set the block number as index
nn_ts_y_df
nn_ts_y_df_Validate
nn_ts_y_df_Total

nn_ts_y_df.set_index('Block', inplace = True)
nn_ts_y_df_Validate.set_index('Block', inplace = True)
nn_ts_y_df_Total.set_index('Block', inplace = True)

# Do some checking
ts_fixed_loop_output[10][2][['Market_Net_Sales']].T
ts_fixed_loop_output[12][2][['Market_Net_Sales']].T

# The construction of these dataframes can be done with list comprehension
y_testing_df_Total = pd.DataFrame(columns = nnts_y_df_colnames)
# The below strips the values from the dataframe (creating an array) then build a dataframe on this with normalised column names, 
# prior to concatenating all these together. Question: how would you create the block index in this method though
y_testing_df_Total = pd.concat((pd.DataFrame(ts_fixed_loop_output[block][2][['Market_Net_Sales']].T.values, columns = ['F1', 'F2', 'F3']) \
                                for block in range(0, len(ts_fixed_loop_output))), axis='rows', ignore_index=True)
# Compare to the loop method
y_testing_df_Total
nn_ts_y_df_Total

# Try an atomic create outside of the list comprehension
pd.DataFrame(ts_fixed_loop_output[block][2][['Market_Net_Sales']].T.values, columns = ['F1', 'F2', 'F3'])

nn_ts_y_df.columns=['b','a','c']

######################
# Build the model here
######################
# Univariate mlp example
from numpy import array
from keras.models import Sequential
from keras.layers import Dense

# Define model
nn_model = Sequential()
nn_model.add(Dense(100, activation='relu', input_dim = 6))
nn_model.add(Dense(3))
nn_model.compile(optimizer='adam', loss='mse')

# View the network topology
print('Neural Network Model Summary:')
print(nn_model.summary())

# Fit model on the training sub-chunks
nn_model.fit(nn_ts_X_df
             , nn_ts_y_df
             , epochs=200
             , verbose=0)

# Summarize history for loss
plt.plot(nn_model.history.history['loss'])
#plt.plot(nnmodel_hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# View the predictions vs. actuals
# Look at collective rmse on the odd blocks not used in model build
for block in range(1, len(ts_fixed_loop_output),2):
    #print(block)
    yhat = nn_model.predict(nn_ts_X_df_Validate.loc[[block]], verbose=0)
    y_0 = nn_ts_y_df_Validate.loc[[block]]
    rmse_nn = sqrt(mean_squared_error(y_0, yhat))
    print(block, rmse_nn)
    
# Look to plot the predictions
# Original testing time series
ts_fixed_loop_output[0][2][['Market_Net_Sales']]
# Transformed (i.e. transposed) testing set for model
y_0 = nn_ts_y_df.loc[[0]]
# Prediction
yhat = nn_model.predict(nn_ts_X_df.iloc[0:0+1,], verbose=0)
yhat
sqrt(mean_squared_error(y_0, yhat))

# So transform the prediction back to the original time series block
yhat_P = pd.DataFrame(yhat.T, index=ts_fixed_loop_output[0][2][['Market_Net_Sales']].index)
yhat_P.columns = ['Predicted_Market_Net_Sales']
yhat_P

# Plot the testing with the original on the same axis
for block in range(1, len(ts_fixed_loop_output),2):
    x_original = ts_fixed_loop_output[block][1][['Market_Net_Sales']]
    y_original = ts_fixed_loop_output[block][2][['Market_Net_Sales']]
    yhat = nn_model.predict(nn_ts_X_df_Validate.loc[[block]], verbose=0)
    yhat_P = pd.DataFrame(yhat.T, index = y_original.index)
    y_0 = nn_ts_y_df_Validate.loc[[block]]
    rmse_nn = sqrt(mean_squared_error(y_0, yhat))
    print(block, rmse_nn)
    ts_validate_plots(ts_full = d['Sector_95']['Market_Net_Sales'], ts_training = x_original, ts_predictions = yhat_P, ts_testing = y_original)


# Plot all to see the overfitting on the training and poor generalisation on the testing
#import termcolor
#termcolor.COLORS

from termcolor import colored

for block in range(0, len(ts_fixed_loop_output)):
    x_original = ts_fixed_loop_output[block][1][['Market_Net_Sales']]
    y_original = ts_fixed_loop_output[block][2][['Market_Net_Sales']]
    yhat = nn_model.predict(nn_ts_X_df_Total.loc[[block]], verbose=0)
    yhat_P = pd.DataFrame(yhat.T, index = y_original.index)
    y_0 = nn_ts_y_df_Total.loc[[block]]
    rmse_nn = sqrt(mean_squared_error(y_0, yhat))
    print(block, colored(rmse_nn, ['green' if block % 2 == 0 else 'red'][0]))
    ts_validate_plots(ts_full = d['Sector_95']['Market_Net_Sales'], ts_training = x_original, ts_predictions = yhat_P, ts_testing = y_original)






########################################
# Neural Net example - Development below
########################################

nnmodel.compile('adam', loss='categorical_crossentropy', metrics = ['accuracy'])

# View the network topology
print('Neural Network Model Summary: ')
print(nnmodel.summary())

# Train the model on the training data set
nnmodel_hist = nnmodel.fit(train_x_cat,
                           train_y_cat,
                           shuffle=True,
                           #validation_split=0.3, 
                           verbose=2,
                           batch_size=10, 
                           epochs=40)

print(nnmodel_hist.history.keys())



# Univariate mlp example
from numpy import array
from keras.models import Sequential
from keras.layers import Dense

# define dataset
X = array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60]])
y = array([40, 50, 60, 70])
# define model
nn_model = Sequential()
nn_model.add(Dense(100, activation='relu', input_dim=3))
nn_model.add(Dense(1))
#nn_model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
nn_model.compile(optimizer='adam', loss='mse')

# View the network topology
print('Neural Network Model Summary:')
print(nn_model.summary())

# fit model
nn_model.fit(X, y, epochs=2000, verbose=0)

print(nn_model.history.history.keys())

# View the output plotted at each epoch

# Summarize history for accuracy (not applicable for a classification exercise...?)
plt.plot(nn_model.history.history['acc'])
#plt.plot(nnmodel_hist.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Summarize history for loss
plt.plot(nn_model.history.history['loss'])
#plt.plot(nnmodel_hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Test the model on unseen data
# demonstrate prediction
x_input = array([500, 600, 700])
x_input = x_input.reshape((1, 3))
yhat = nn_model.predict(x_input, verbose=0)
print(yhat)

# Show (overfitting) on the training data
for i in range(0,4):
    print(i)
    yhat = nn_model.predict(X[i].reshape((1, 3)), verbose=0)
    print(y[i], yhat)


# Try hitting multiple targets/months in the future
# define dataset
X = array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60]])
y = array([[40,41], [50,51], [60,61], [70,71]])

# Closer to live example
# Do with (6,3) DataFrames
X_df = pd.DataFrame([[10, 20, 30, 40, 50, 60], [20, 30, 40, 55, 66, 77], [33, 43, 53, 63, 73, 83], [40, 50, 60,70,80,90]])
y_df = pd.DataFrame([[40,41,42], [50,51,52], [60,61,62], [70,71,72]])

X_df.iloc[0:1,]
ts_fixed_loop_output[0][1][['Market_Net_Sales']].T

y_df.iloc[0:1,]
ts_fixed_loop_output[0][2][['Market_Net_Sales']].T


# define model
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=6))
model.add(Dense(3))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X_df, y_df, epochs=200, verbose=0)

#model.fit(X, y, epochs=200, verbose=0)

# Demonstrate prediction on some training data
yhat = model.predict(X_df.iloc[0:1,], verbose=0)
type(yhat)
print(yhat)

# This then works on the transposed real data
ts_fixed_loop_output[0][1][['Market_Net_Sales']].T
ts_fixed_loop_output[0][1][['Market_Net_Sales']]

yhat = model.predict(ts_fixed_loop_output[0][1][['Market_Net_Sales']].T, verbose=0)
#yhat = model.predict(ts_fixed_loop_output[0][1][['Market_Net_Sales']], verbose=0) doesn't work
type(yhat)
print(yhat)


x_input_df = pd.DataFrame([[50, 60, 70]])
#x_input_df = x_input_df.reshape((1, 3))
yhat = model.predict(x_input_df, verbose=0)
type(yhat)
print(yhat)

model.predict(x_input, verbose=0)


# Show (overfitting) on the training data
X_df.iloc[0:1,]
X_df.iloc[1:2,]
X_df.iloc[2:3,]
X_df.iloc[3:4,]
#X_df.iloc[4:5,]
y_df.iloc[0:1,]

for i in range(0,4):
    print(i)
    yhat = model.predict(X_df.iloc[i:i+1,], verbose=0)
    print(y_df.iloc[i:i+1,], yhat)




# Raw text from the machine learning course
# univariate mlp example
from numpy import array
from keras.models import Sequential
from keras.layers import Dense
# define dataset
X = array([[10, 20, 30], [20, 30, 40], [30, 40, 50], [40, 50, 60]])
y = array([40, 50, 60, 70])
# define model
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=3))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X, y, epochs=2000, verbose=0)
# demonstrate prediction
x_input = array([50, 60, 70])
x_input = x_input.reshape((1, 3))
yhat = model.predict(x_input, verbose=0)
print(yhat)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Look at timing the list concatenation
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import datetime
nnts_y_df_colnames = ['F1', 'F2', 'F3', 'Block']

start = datetime.datetime.now()
y_testing_df_Total = pd.DataFrame(columns = nnts_y_df_colnames)
y_testing_df_Total = pd.concat((pd.DataFrame(ts_fixed_loop_output[block][2][['Market_Net_Sales']].T.values, columns = ['F1', 'F2', 'F3']) \
                                for block in range(0, len(ts_fixed_loop_output))), axis='rows', ignore_index=True)
end = datetime.datetime.now()
print(end-start) # 0:00:00.016956 (16,956 micro-seconds (= millionths of a second))


nn_ts_y_df_Total = pd.DataFrame(columns = nnts_y_df_colnames)

start_loop_time = datetime.datetime.now()
for block in range(0, len(ts_fixed_loop_output)):
    df_temp = ts_fixed_loop_output[block][2][['Market_Net_Sales']].T
    df_temp['Block'] = block
    df_temp.columns = nnts_y_df_colnames
    nn_ts_y_df_Total = nn_ts_y_df_Total.append(df_temp)

end_loop_time = datetime.datetime.now()
print(end_loop_time-start_loop_time) # 0:00:00.052859 (52,859 micro-seconds)

# Convince yourself that the timing does work
import time

test_start_loop_time = datetime.datetime.now()
time.sleep(3)
test_end_loop_time = datetime.datetime.now()
print(test_end_loop_time-test_start_loop_time) # 0:00:03


