# UdacityCapstoneProject_StockPredictor

## Table of Contents
1. [Description](#description)
2. [Using the application](#getting_started)
	1. [Requirements to run project](#dependencies)
	2. [Installing](#installation)
	3. [Executing Program](#execution)
	4. [Additional Material](#material)
3. [Acknowledgements](#acknowledgements)
4. [Authors](#authors)


<a name="descripton"></a>
## Description

This is a project for the Udacity Nano degree in data science. The project scope is to create a web application that is able to analyse stock prices as well as predict the future prices a certain stock. 

The scope of the project includes deploying a web application to analyse a selected stock as well as train a model to produce a predicted stock price for the selected stock. The project consists of four main tasks:
1. Query an API to retrieve stock price data
2. Store stock price data in an SQLite database
3. Pre-process the stock price data in order to train a machine learning model
4. Build a machine learning model that trains on stock price data to predict future stock prices
5. Deploy the machine learning model as well as pre-processed data to a web application in order to analyse stock prices

<a name="getting_started"></a>
## Using the application

<a name="dependencies"></a>
### Requirements to run project
* Python 3.6+
* Data manipulation: Pandas, Numpy
* Machine Learning Libraries: Keras, Tensor Flow
* Database management: SQLalchemy
* Web App and Data Visualization: Flask, Plotly
* Stock data API: yfinance 0.1.63

<a name="installation"></a>
### Installation
To clone the git repository:
```
git clone https://github.com/dirklambrechts/UdacityCapstoneProject_StockPredictor
```
<a name="execution"></a>
### Running the program:
1. Run the following command to start a local server on your machine:
    `python runserver.py`

2. Go to http://localhost:5555/ to view and use the web application

### Using the web application:
1. The home page is the stock price analysis page that enables a user to see and analyse the price of a stock along with three different technical indicators to select from. Select the stock you would like to analyse and select the technical indicators you would like to see for the stock selected. 
![](Images/Home_page.png)
2. The training interface page enables a user to select parameters that you would like to use to train a model. By selecting the start training model button a training cycle commences and produces a page to show the results of the model over the training period as well as the predicted stock price for the selected period.

<a name="material"></a>
### Additional Information

There are two jupyter notebook (.ipynb) files that can assist in describing the data pre-processing and modelling process in more detail. They are in the folder called data and models respectively. 

<a name="acknowledgements"></a>
## Acknowledgements
[Udacity](https://www.udacity.com/) for the Capstone project idea.

<a name="authors"></a>
## Authors

* [Dirk Lambrechts](https://github.com/dirklambrechts)
