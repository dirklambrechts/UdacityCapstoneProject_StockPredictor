import pandas as pd
import numpy as np
from sklearn import preprocessing
np.random.seed(4)
import datetime
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'        # 2 = INFO and WARNING messages are not printed
import tensorflow as tf

from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation, concatenate
from keras import optimizers
tf.random.set_seed(4)

def data_prep_new_lstm(data, n_past=50, n_future=5, include_ma = False):
    '''
    DESCRIPTION:
    Prepares data for input into an LTSM model. Transforms the data dataframe into sequential time step data depending on the
    specified values for n_past and n_future. 

    INPUT:
    data - (dataframe) A data frame containing open, high, low, close and adjusted close stock prices for a specific stock as well as the
    volume of stocks traded. (Index of data frame must be date column)
    n_past - (int) The number of data points that will be used when predicting a stock price
    n_future - (int) The number of time steps into the future the model should be able to predict for
    include_ma - (bool) Should a moving average of n_past days be included as a feature

    OUTPUT:
    data_normal_hist - (array) Matrix of open, close, low, high and volume scaled values in sections of n_past size data groups
                        (Dimensions: rows x n_past x features)
    data_normal_nextday - (array) Matrix of scaled values at index n_future away from n_past data groups (Dimensions: rows x 1)
    data_values_nextday - (array) Matrix of adjusted closing price values at index n_future away from n_past data groups (Dimensions: rows x 1)
    y_normalizer - (object) A minmaxscaler object to be used to convert scaled predicted values back to non-scaled values
    date_list - (list) A list of dates from the data frame
    data_normal_hist_future - (array) Data set to be used to predict future values with
    date_list_future - (array) Date list of future values
    tech_ind_ma_normal - (array) Normalised values of moving average of n_past days
    ''' 
    
    #X-values: open, closing, high, low stock prices and volume of stocks traded to input into lstm model
    normalizer = preprocessing.MinMaxScaler(feature_range=(-1,1))
    normalized_data = normalizer.fit_transform(data.drop(['AdjClose'], axis=1))
    data_normal_hist = np.array([normalized_data[:, 0: 5][i-n_future-n_past : i-n_future].copy() for i in range(len(normalized_data)-1, n_past + n_future, -1)])[::-1]
    data_normal_hist_future = np.array([normalized_data[:, 0: 5][i-n_past : i].copy() for i in range(len(normalized_data)-1, len(normalized_data)-1-n_future, -1)])[::-1]
    
    #Dates to use for plotting the results of the model
    date_list = [data.index[i] for i in range(len(data)-1, n_past + n_future, -1)][::-1]
    date_list_future = []
    for i in range (1, n_future+1):
        date_list_future.append(date_list[-1]+datetime.timedelta(days=i))

    #Y-values: adjusted closing stock price
    data_values_nextday = np.array([data.AdjClose[i].copy() for i in range(len(data)-1, n_past + n_future, -1)])[::-1]
    data_values_nextday = np.expand_dims(data_values_nextday, -1) 
    y_normalizer = preprocessing.MinMaxScaler(feature_range=(-1,1))
    y_normalizer.fit(data_values_nextday)    
    data_normal_nextday = y_normalizer.fit_transform(data_values_nextday)
    
    #Check if a technical indicator of moving average for n_past days must be incorporated in the preprocessing step
    if include_ma:
        #Technical indicator feature, moving average of n_past days for closing price of stock
        tech_ind_ma_normal =  np.array([np.mean(data_normal_hist[i][:,3]) for i in range (data_normal_hist.shape[0])])
        tech_ind_ma_normal = np.expand_dims(tech_ind_ma_normal, -1)
        tech_ind_ma_normal_future = np.array([np.mean(data_normal_hist_future[i][:,3]) for i in range(data_normal_hist_future.shape[0])])
        tech_ind_ma_normal_future = np.expand_dims(tech_ind_ma_normal_future, -1)
        return data_normal_hist, data_normal_nextday, data_values_nextday, y_normalizer, date_list, data_normal_hist_future, date_list_future, tech_ind_ma_normal, tech_ind_ma_normal_future 
    
    return data_normal_hist, data_normal_nextday, data_values_nextday, y_normalizer, date_list, data_normal_hist_future, date_list_future 


def test_train_split(test_split, data_normal_hist, data_values_nextday, data_normal_nextday, date_list, include_ma = False, tech_ind_ma_normal=0):
    '''
    DESCRIPTION:
    Splits a data set into training and testing sets.

    INPUT:
    test_split - (float) Percentage of data set to be part of training set (this is a decimal ie. 0.8)
    data_normal_hist - (array) Matrix of open, close, low, high and volume scaled values in sections of n_past size data groups
                        (Dimensions: rows x n_past x features)
    data_values_nextday - (array) Matrix of adjusted closing price values at index n_future away from n_past data groups 
                        (Dimensions: rows x 1)
    data_normal_nextday - (array) Matrix of scaled values at index n_future away from n_past data groups (Dimensions: rows x 1)
    date_list - (list) List of all dates in the data set
    tech_ind_ma_normal - (array) An array of 50 day moving averages
    
    OUTPUT:
    data_normal_hist_train (array) - Training set matrix of open, close, low, high and volume scaled values in sections of n_past size data groups
    data_normal_nextday_train (array) - Training set matrix of scaled values at index n_future away from n_past data groups (Dimensions: rows x 1)
    data_normal_hist_test (array) - Testing set matrix of open, close, low, high and volume scaled values in sections of n_past size data groups
    data_normal_nextday_test (array) - Testing set matrix of scaled values at index n_future away from n_past data groups (Dimensions: rows x 1)
    data_values_nextday_test (array) - Test set adjusted closing price values at index n_future away from n_past data groups 
                        (Dimensions: rows x 1)
    date_list_train (array) - List of training data set dates
    date_list_test (array) - List of testing data set dates
    tech_ind_ma_normal_train (array) - An array of 50 day moving averages for training set
    tech_ind_ma_normal (array) - An array of 50 day moving averages for testing set
    ''' 
    
    n = int(data_normal_hist.shape[0] * test_split)
    
    #Prepare training data
    data_normal_hist_train = data_normal_hist[: n]
    data_normal_nextday_train = data_normal_nextday[: n]
    data_values_nextday_train = data_values_nextday[:n]
    date_list_train = date_list[: n]
    
    #Prepare testing data
    data_normal_hist_test = data_normal_hist[n: ]
    data_normal_nextday_test = data_normal_nextday[n: ]
    data_values_nextday_test = data_values_nextday[n: ]
    date_list_test = date_list[n: ]
    
    #Check if Moving average is included for this test/train split
    if include_ma:
        tech_ind_ma_normal_test = tech_ind_ma_normal[n: ]
        tech_ind_ma_normal_train = tech_ind_ma_normal[: n]
        return data_normal_hist_train, data_normal_nextday_train, data_normal_hist_test, data_normal_nextday_test, data_values_nextday_test, date_list_train, date_list_test, tech_ind_ma_normal_train, tech_ind_ma_normal_test

    return data_normal_hist_train, data_normal_nextday_train, data_normal_hist_test, data_normal_nextday_test, data_values_nextday_test, date_list_train, date_list_test

def create_lstm_basic_model(n_past=50, learning_rate=0.0005):
    '''
    DESCRIPTION:
    Creates an LSTM model that considers n_past time steps when predicting a future value. This models uses a linear activation 
    function

    INPUT:
    n_past - (int) Days to use when predicting a value
    learning_rate - (int) Learning rate to use for the model
    
    OUTPUT:
    model (object) - An LSTM model object
    '''     
    np.random.seed(4)
    tf.random.set_seed(4)
    lstm_input = Input(shape=(n_past, 5), name='lstm_input')
    x = LSTM(50, name='lstm_0')(lstm_input)
    x = Dropout(0.2, name='lstm_dropout_0')(x)
    x = Dense(64, name='dense_0')(x)
    x = Activation('sigmoid', name='sigmoid_0')(x)
    x = Dense(1, name='dense_1')(x)
    output = Activation('linear', name='linear_output')(x)
    model = Model(inputs=lstm_input, outputs=output)
    adam = optimizers.Adam(learning_rate)
    model.compile(optimizer=adam, loss='mse')
    return model

def create_lstm_TA_model(tech_ind_ma_normal, n_past=50, learning_rate=0.0005, neurons=50):
    '''
    DESCRIPTION:
    Creates an LSTM model that considers n_past time steps when predicting a future value. This models uses a linear activation 
    function and uses a moving average technical indicator to train the model.

    INPUT:
    n_past - (int) Days to use when predicting a value
    learning_rate - (int) Learning rate to use for the model
    neurons - (int) Number of neurons to use for the model
    tech_ind_ma_normal - (array) Normalised values of moving average of n_past days
    
    OUTPUT:
    model (object) - An LSTM model object
    ''' 
    np.random.seed(4)
    tf.random.set_seed(4)
    # define two sets of model inputs
    lstm_input = Input(shape=(n_past, 5), name='lstm_input')
    dense_input = Input(shape=(tech_ind_ma_normal.shape[1],), name='tech_input')

    # the first branch operates on the first input
    x = LSTM(neurons, name='lstm_0')(lstm_input)
    x = Dropout(0.2, name='lstm_dropout_0')(x)
    lstm_branch = Model(inputs=lstm_input, outputs=x)

    # the second branch opreates on the second input
    y = Dense(20, name='tech_dense_0')(dense_input)
    y = Activation("relu", name='tech_relu_0')(y)
    y = Dropout(0.2, name='tech_dropout_0')(y)
    technical_indicators_branch = Model(inputs=dense_input, outputs=y)

    # combine the output of the two branches
    combined = concatenate([lstm_branch.output, technical_indicators_branch.output], name='concatenate')

    z = Dense(64, activation="sigmoid", name='dense_pooling')(combined)
    z = Dense(1, activation="linear", name='dense_out')(z)

    # our model will accept the inputs of the two branches and then output a single value
    model = Model(inputs=[lstm_branch.input, technical_indicators_branch.input], outputs=z)

    adam = optimizers.Adam(learning_rate)

    model.compile(loss='mean_squared_error', optimizer='adam') 
    return model