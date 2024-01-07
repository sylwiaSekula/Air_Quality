# Air_Quality
### Air Quality Prediction for PM10 Pollution in Taubaté, São Paulo, Brazil
This project focuses on predicting PM10 pollution in the air using a time series dataset from the 'Air Quality in São Paulo, Brazil' dataset available on Kaggle https://www.kaggle.com/datasets/amandalk/sp-air-quality. The dataset contains air quality measurements recorded hourly by CETESB in São Paulo state, Brazil, spanning from 08-05-2013 until 09-09-2020.
### Dataset Overview
The original dataset contains 3,445,260 rows in 11 columns, beacuse it contains data for 62 stations in São Paulo. I decided to chose only one station and it was Taubaté.
The dataset includes the following columns:<br>
'Station', the monitoring station rom which the air quality measurements were recorded - categorical, <br>
'Benzene', Benzene concentration in the air - numerical <br>
'CO', CO (Carbon Monoxide) concentration in the air - numerical <br>
'PM10',PM10 concentration in the air - numerical <br>
'PM2.5',PM2.5 concentration in the air - numerical <br>
'NO2',NO2 (Nitrogen Dioxide) concentration in the air - numerical <br>
'O3',O3 (Ozone) concentration in the air - numerical <br>
'SO2', SO2 (Sulfur Dioxide) concentration in the air - numerical <br>
'Toluene' Toluene concentration in the air - numerical  <br>
'TRS'. TRS (Total Reduced Sulfur) concentration in the air - numerical <br>
 I decided to predict data for the PM10 pollutant. PM10 refers to tiny particles in the air that are 10 micrometers or smaller in diameter. These particles can include dust, dirt, soot, and other substances. They are small enough to be inhaled into the lungs, posing potential health risks. PM10 can come from natural sources like dust storms and human activities such as vehicle emissions and industrial processes. These particles can impact both human health and the environment, contributing to respiratory issues and reduced air quality.
### Objective
The primary goal of this project is to forecast monthly PM10 pollution levels in the air using historical data collected from the 'Taubaté' station in São Paulo, Brazil.
### Methodology
#### Data Preprocessing: 
First I processed the original dataset and created new datasets. I resampled the data monthly for models compatibility, dropped irrelevant columns for statisctical models. I created new datasets: df_resampled (the original dataset with resampled the data monthly), df_train and df_test for statisctical models, df_train_ml and df_test_ml for a machine leraning model in which I created a new lag_12 column by shifting the data 12 steps ahead. The new column containing past values from the target column. <br>
#### Model Selection:<br>
I implemented statistical models used for time series forecasting:<br>
<b> HoltWinters (ExponentialSmoothing from sktime) </b> - a statistical method for time series forecasting, available in sktime as Exponential Smoothing. It's capable of capturing trends and seasonality in time series data by using weighted averages and smoothing techniques. Holt-Winters includes three components: level, trend, and seasonality, making it suitable for forecasting data with both trended and seasonal patterns.<br>
<b> Autoarima </b> - an automated version of the popular ARIMA (AutoRegressive Integrated Moving Average) model. It performs the automatic selection of the optimal ARIMA model parameters (p, d, q) through a search process, eliminating the need for manual tuning. AutoARIMA uses algorithms to identify the most suitable model based on statistical criteria, making it convenient for time series forecasting without requiring users to specify the parameters manually.
 <br>
<b> Prophet </b> - developed by Facebook, Prophet is an open-source forecasting tool designed for analyzing and predicting time series data. It's based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, along with holidays and other effects.<br>
and a machine learning model:<br>
<b> XGBoost </b> - although not specifically designed for time series forecasting, is a highly versatile and powerful machine learning algorithm extensively used in various predictive modeling tasks. While XGBoost is not inherently dedicated for time series data, its flexibility and ability to handle complex relationships in data make it adaptable for time-dependent predictions.<br>
#### Model Evaluation: 
I evaluated the performance of each model using appropriate metrics for time series forecasting. 
![HW](https://github.com/sylwiaSekula/Air_Quality/assets/110921660/0dcaf1e3-dadc-48fd-a6c9-1541b4e74e89)

![AutoARIMA](https://github.com/sylwiaSekula/Air_Quality/assets/110921660/2d9eae18-7e8d-432d-893c-431ba5a06f55)

![Prophet](https://github.com/sylwiaSekula/Air_Quality/assets/110921660/9580b5bc-5d02-4c15-9890-7620284f9786)

![XGBoost](https://github.com/sylwiaSekula/Air_Quality/assets/110921660/4d5e60a4-d33f-4ea6-834d-f187ee0e8802)



