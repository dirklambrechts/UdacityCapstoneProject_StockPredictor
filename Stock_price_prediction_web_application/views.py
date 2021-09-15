"""
Routes and views for the flask application.
"""
from flask import Flask
from flask import render_template
from flask import render_template, request, jsonify
from Stock_price_prediction_web_application import app
import pandas as pd
import plotly
from plotly.graph_objs import Bar, Pie, Line, Scatter
import datetime
import sqlalchemy as sql
import yfinance as yf
import json
import numpy as np

import Stock_price_prediction_web_application.Model_data_prep as data_prep

stocks_to_use = {'Microsoft': 'MSFT', 'Nasdaq': '^IXIC', 'S&P500':'^GSPC'}
dbname = 'DB_stock_history_data'

def data_request_API(symbol, start_date, end_date, reset_index=True):
    '''
    DESCRIPTION:
    Retrieve all stock price data for a given stock ticker symbol through the yahoo finance platform. 
    This function returns the opening price, highest price, closing price and trade volume of a stock 
    ticker symbol for a given day. 

    INPUT:
    symbol - (string) The stock symbol to retrieve data for
    start_date - (string) The start date for which to retrieve data for (format: 'yyyy-mm-dd')
    end-date - (string) The end date for which to retrieve data for (format: 'yyyy-mm-dd')
    reset_index - (boolean) Instruction to reset the index of the data frame or not

    OUTPUT:
    data -  (pandas df) Dataframe containing the date (index), open price, highest price, lowest price, closing price, 
    trade volume and the adjusted closing price for specified period of time
    '''    
    data = yf.download(symbol, start_date, end_date)
    data.columns = ['Open', 'High', 'Low', 'Close', 'AdjClose', 'Volume']
    data = data[['Open', 'High', 'Low', 'Close', 'Volume', 'AdjClose']]
    if reset_index==True:
        data.reset_index(inplace=True, drop=True)
    return data

def update_stock_DB(dbname, stocks_to_use):
    '''
    DESCRIPTION:
    Updates the stock DB data base with most recent stock data

    INPUT:
    dbname - (string) The name of the data base. 
    stocks_to_use - (dict) A dictionary of names and symbols of all stocks to update in data base.

    OUTPUT:
    No output
    '''  
    conn = sql.create_engine('sqlite:///Stock_price_prediction_web_application/{}.db'.format(dbname)).connect()
    for name, value in stocks_to_use.items():
        df = pd.read_sql_table(name, con=conn)
        last_date = datetime.datetime.date(df.Date.max()) + datetime.timedelta(days=1)
        today_date = datetime.datetime.now()
        if last_date != today_date:
            temp_df_new_sql = data_request_API(value, last_date, today_date, reset_index=False)
            temp_df_new_sql.to_sql(name, conn, index=True, if_exists='append')
    conn.close()

def fetch_DB_data(dbname, stocks_to_use):
    '''
    DESCRIPTION:
    Retrieves data from a data base.

    INPUT:
    dbname - (string) The name of the database. 
    stocks_to_use - (dict) A dictionary of names and symbols of all stocks to update in data base.

    OUTPUT:
    df_dict_all - (dict) A dictionary of data frames for each name in stocks to use
    '''  
    conn = sql.create_engine('sqlite:///Stock_price_prediction_web_application/{}.db'.format(dbname)).connect()
    df_dict_all = {}
    for name, value in stocks_to_use.items():
        df = pd.read_sql(name, con=conn)
        df.set_index('Date', inplace=True)
        df_dict_all[name] = df 
    conn.close()
    return df_dict_all

update_stock_DB(dbname, stocks_to_use)
df_dict_all = fetch_DB_data(dbname, stocks_to_use)

@app.route("/")
@app.route('/home', methods=['GET', 'POST'])
def home():
    """Renders the home page."""
    stock_selection_list = stocks_to_use.keys()
    
    #Select the default stock
    stock = ''

    graph_1 = []
    if request.method == 'POST':
        stock = request.form.get('stock_selected_index')
        ma_10days = bool(request.form.get('ma_selected_10'))
        ma_50days = bool(request.form.get('ma_selected_50'))
        ma_200days = bool(request.form.get('ma_selected_200'))

        #Add Nasdaq to plot with corresponding moving averages
        if stock == "Nasdaq":
            graph_1.append(
                Scatter(x = df_dict_all[stock].index,
                y = df_dict_all[stock]['Open'].values,
                mode = 'lines',
                name = "Open Price {}".format(stock)
                ))

            if ma_10days:
                days = 10
                graph_1.append(
                    Scatter(x = df_dict_all[stock].index,
                    y = df_dict_all[stock]['Open'].rolling(window=days).mean(),
                    mode = 'lines',
                    name = "{} Day MA {}".format(days, stock)
                    ))

            if ma_50days:
                days = 50
                graph_1.append(
                    Scatter(x = df_dict_all[stock].index,
                    y = df_dict_all[stock]['Open'].rolling(window=days).mean(),
                    mode = 'lines',
                    name = "{} Day MA {}".format(days, stock)
                    ))

            if ma_200days:
                days = 200
                graph_1.append(
                    Scatter(x = df_dict_all[stock].index,
                    y = df_dict_all[stock]['Open'].rolling(window=days).mean(),
                    mode = 'lines',
                    name = "{} Day MA {}".format(days, stock)
                    ))

        #Add Microsoft to plot with corresponding moving averages
        elif stock == "Microsoft":
            graph_1.append(
                Scatter(x = df_dict_all[stock].index,
                y = df_dict_all[stock]['Open'].values,
                mode = 'lines',
                name = "Open Price {}".format(stock)
                ))

            if ma_10days:
                days = 10
                graph_1.append(
                    Scatter(x = df_dict_all[stock].index,
                    y = df_dict_all[stock]['Open'].rolling(window=days).mean(),
                    mode = 'lines',
                    name = "{} Day MA {}".format(days, stock)
                    ))

            if ma_50days:
                days = 50
                graph_1.append(
                    Scatter(x = df_dict_all[stock].index,
                    y = df_dict_all[stock]['Open'].rolling(window=days).mean(),
                    mode = 'lines',
                    name = "{} Day MA {}".format(days, stock)
                    ))

            if ma_200days:
                days = 200
                graph_1.append(
                    Scatter(x = df_dict_all[stock].index,
                    y = df_dict_all[stock]['Open'].rolling(window=days).mean(),
                    mode = 'lines',
                    name = "{} Day MA {}".format(days, stock)
                    ))
        
        #Add S&P500 to plot with corresponding moving averages
        elif stock == "S&P500":
            graph_1.append(
                Scatter(x = df_dict_all[stock].index,
                y = df_dict_all[stock]['Open'].values,
                mode = 'lines',
                name = "Open Price {}".format(stock)
                ))

            if ma_10days:
                days = 10
                graph_1.append(
                    Scatter(x = df_dict_all[stock].index,
                    y = df_dict_all[stock]['Open'].rolling(window=days).mean(),
                    mode = 'lines',
                    name = "{} Day MA {}".format(days, stock)
                    ))

            if ma_50days:
                days = 50
                graph_1.append(
                    Scatter(x = df_dict_all[stock].index,
                    y = df_dict_all[stock]['Open'].rolling(window=days).mean(),
                    mode = 'lines',
                    name = "{} Day MA {}".format(days, stock)
                    ))

            if ma_200days:
                days = 200
                graph_1.append(
                    Scatter(x = df_dict_all[stock].index,
                    y = df_dict_all[stock]['Open'].rolling(window=days).mean(),
                    mode = 'lines',
                    name = "{} Day MA {}".format(days, stock)
                    ))
   
    layout_1 = dict(title = 'Price of stock {}'.format(stock),
                    xaxis = dict(title = 'Date', autotick=True),
                    yaxis = dict(title = 'Stock price'),
                    )    
    figures = []    
    figures.append(dict(data=graph_1, layout=layout_1))
   
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(figures)]
    graphJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('index.html',
                           title="Home Page", 
                           year=datetime.datetime.now().year, 
                           ids=ids,  
                           stock_selection_list=stock_selection_list, 
                           graphJSON=graphJSON)

@app.route('/training_interface', methods=['GET', 'POST'])
def training_interface():
    """Renders the training interface page."""
    #Variables for selection options on website
    all_stocks_selection = ['Microsoft', 'Nasdaq', 'S&P500']
    technical_indicator = ['None', 'Moving Average']
    train_test_split_selection = ['60%', '70%', '80%', '90%']
    days_to_use = ['30', '40', '50', '60']
    days_to_predict = ['3', '5', '7']

    """Renders the training interface page."""
    return render_template(
        'training_interface.html',
        title='Training Interface',
        year=datetime.datetime.now().year,
        all_stocks_selection=all_stocks_selection,
        technical_indicator=technical_indicator,
        train_test_split_selection=train_test_split_selection,
        days_to_use=days_to_use,
        days_to_predict=days_to_predict
    )

@app.route("/training_interface/model_train", methods=['GET', 'POST'])
def model_train():
    """Renders the model performance page."""
    if request.method == 'POST':
        #Populate include_ma field with selection
        if request.form.get('technical_indicator_selected') == 'Moving Average':
            include_ma_model = True
        elif request.form.get('technical_indicator_selected') == 'None':
            include_ma_model = False

        #Populate train/test split selection
        train_test_split_model = float(str(request.form.get('train_test_split_selected')).replace('%', ''))/100

        #Populate days to use with selection
        n_past_model = int(request.form.get('days_to_use_selected'))

        #Populate days to predict with selection
        n_future_model = int(request.form.get('days_to_predict_selected'))

        #populate stock that has been selected
        stock_train_model = request.form.get('stock_selected')
    
    batch_size = 4
    neurons = 50
    epochs = 5
    learning_rate = 0.0005

    if include_ma_model == True:
        #Prepare normalized data for moving average included
        data_normal_hist, data_normal_nextday, data_values_nextday, y_normalizer, date_list, data_normal_hist_future, date_list_future, tech_ind_ma_normal, tech_ind_ma_normal_future = data_prep.data_prep_new_lstm(df_dict_all[stock_train_model], n_past_model, n_future_model, include_ma_model)
        #Prepare test train split for moving average included in the model
        data_normal_hist_train, data_normal_nextday_train, data_normal_hist_test, data_normal_nextday_test, data_values_nextday_test, date_list_train, date_list_test, tech_ind_ma_normal_train, tech_ind_ma_normal_test = data_prep.test_train_split(train_test_split_model, data_normal_hist, data_values_nextday, data_normal_nextday, date_list, include_ma_model, tech_ind_ma_normal)
        model_with_TA = data_prep.create_lstm_TA_model(tech_ind_ma_normal, n_past_model, learning_rate, neurons) 
        model_with_TA.fit(x=[data_normal_hist_train, tech_ind_ma_normal_train], y=data_normal_nextday_train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_split=0.1)
        data_values_predict_test = y_normalizer.inverse_transform(model_with_TA.predict([data_normal_hist_test,tech_ind_ma_normal_test]))
        data_values_predict_future = y_normalizer.inverse_transform(model_with_TA.predict([data_normal_hist_future,tech_ind_ma_normal_future]))

    elif include_ma_model == False:
        data_normal_hist, data_normal_nextday, data_values_nextday, y_normalizer, date_list, data_normal_hist_future, date_list_future = data_prep.data_prep_new_lstm(df_dict_all[stock_train_model], n_past_model, n_future_model, include_ma_model)
        #Prepare test train split for moving average not included in the model
        data_normal_hist_train, data_normal_nextday_train, data_normal_hist_test, data_normal_nextday_test, data_values_nextday_test, date_list_train, date_list_test = data_prep.test_train_split(train_test_split_model, data_normal_hist, data_values_nextday, data_normal_nextday, date_list, include_ma_model)
        model_without_TA = data_prep.create_lstm_basic_model(n_past_model, learning_rate)
        model_without_TA.fit(x=data_normal_hist_train, y=data_normal_nextday_train, batch_size=batch_size, epochs=epochs, shuffle=True, validation_split=0.1)
        data_values_predict_test = y_normalizer.inverse_transform(model_without_TA.predict(data_normal_hist_test))
        data_values_predict_future = y_normalizer.inverse_transform(model_without_TA.predict(data_normal_hist_future))
   
    #Calculate the mean square error of the model
    real_mse = np.mean(np.square(data_values_nextday_test - data_values_predict_test))
    scaled_mse = real_mse / (np.max(data_values_nextday_test) - np.min(data_values_nextday_test)) * 100
    scaled_mse = round(scaled_mse)
    
    results_plot = []
    
    #Add the predicted values over the test period to the plot
    results_plot.append(
        Scatter(x = date_list_test,
        y = data_values_predict_test.ravel(),
        mode = 'lines',
        name = "Predicted values on test data"
        ))

    #Add the future predicted values to plot
    results_plot.append(
        Scatter(x = date_list_future,
        y = data_values_predict_future.ravel(),
        mode = 'lines',
        name = "Predicted future values - {} days ahead".format(n_future_model)
        ))

    #Add the actual stock price over the test period for comparison
    results_plot.append(
        Scatter(x = date_list_test,
                y = data_values_nextday_test.ravel(),
                mode = 'lines',
                name = 'Actual values over test data period'               
        ))

    results_layout = dict(title = 'Prediction of stock price for {}'.format(stock_train_model),
                    xaxis = dict(title = 'Date', autotick=True),
                    yaxis = dict(title = 'Stock price'),
                    )    
    results_figure = []    
    results_figure.append(dict(data=results_plot, layout=results_layout))
   
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(j) for j, _ in enumerate(results_figure)]
    graphJSON = json.dumps(results_figure, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('model_interface.html',title="Trained model results", year=datetime.datetime.now().year, scaled_mse=scaled_mse, ids=ids, graphJSON=graphJSON)

