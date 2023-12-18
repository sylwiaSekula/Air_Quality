# Air_Quality
### Air Quality Prediction for PM10 Pollution in Taubaté, São Paulo, Brazil
This project focuses on predicting PM10 pollution in the air using a time series dataset from the 'Air Quality in São Paulo, Brazil' dataset available on Kaggle https://www.kaggle.com/datasets/amandalk/sp-air-quality. The dataset contains air quality measurements recorded hourly by CETESB in São Paulo state, Brazil, spanning from 08-05-2013 until 09-09-2020.
### Dataset Overview
The original dataset contains 3,445,260 rows in 11 columns, beacuse it contains data for 62 stations in São Paulo. I decided to chose only one station and it was Taubaté.
The dataset includes the following columns:<br>
'Station',<br>
'Benzene',<br>
'CO',<br>
'PM10',<br>
'PM2.5',<br>
'NO2',<br>
'O3',<br>
'SO2',<br>
'Toluene',<br>
'TRS'.<br>
 I decided to predict data for the PM10 pollutant. PM10 refers to tiny particles in the air that are 10 micrometers or smaller in diameter. These particles can include dust, dirt, soot, and other substances. They are small enough to be inhaled into the lungs, posing potential health risks. PM10 can come from natural sources like dust storms and human activities such as vehicle emissions and industrial processes. These particles can impact both human health and the environment, contributing to respiratory issues and reduced air quality.
### Objective
The primary goal of this project is to forecast monthly PM10 pollution levels in the air using historical data collected from the 'Taubaté' station in São Paulo, Brazil.
### Methodology
#### Data Preprocessing: First I processed the original dataset and created new datasets. I resampled the data monthly for models compatibility, dropped irrelevant columns for statisctical models. I created new datasets: df_resampled (the original dataset with resampled the data monthly), df_train and df_test for statisctical models, df_train_ml and df_test_ml for a machine leraning model in which I created 12 new columns with lag features <br>
#### Model Selection:<br>
I implemented statistical models:<br>
HoltWinters (ExponentialSmoothing from sktime)<br>
Autoarima<br>
Prophet<br>
and a machine learning model:<br>
XGBoost using lagged features<br>
#### Model Evaluation: 
I evaluated the performance of each model using appropriate metrics for time series forecasting.
