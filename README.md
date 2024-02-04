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
First I processed the original dataset and created new datasets. I resampled the data monthly for models compatibility, dropped irrelevant columns for statisctical models. I created new datasets: df_resampled (the original dataset with resampled the data monthly), df_train and df_test for statisctical models, df_train_ml and df_test_ml for a machine leraning model in which I created  new lag_12 columns by shifting the data 12 steps ahead - for the target column and additional features. The new column containing past values. <br>
#### Model Selection:<br>
I implemented statistical models used for time series forecasting:<br>
<b> HoltWinters (ExponentialSmoothing from sktime) </b> - a statistical method for time series forecasting, available in sktime as Exponential Smoothing. It's capable of capturing trends and seasonality in time series data by using weighted averages and smoothing techniques. Holt-Winters includes three components: level, trend, and seasonality, making it suitable for forecasting data with both trended and seasonal patterns.<br>
<b> Autoarima </b> - an automated version of the popular ARIMA (AutoRegressive Integrated Moving Average) model. It performs the automatic selection of the optimal ARIMA model parameters (p, d, q) and (P, D, Q) through a search process, eliminating the need for manual tuning. AutoARIMA uses algorithms to identify the most suitable model based on statistical criteria, making it convenient for time series forecasting without requiring users to specify the parameters manually.
 <br>
<b> Prophet </b> - developed by Facebook, Prophet is an open-source forecasting tool designed for analyzing and predicting time series data. It's based on an additive model where non-linear trends are fit with yearly, weekly, and daily seasonality, along with holidays and other effects.<br>
and a machine learning model:<br>
<b> XGBoost </b> - although not specifically designed for time series forecasting, is a highly versatile and powerful machine learning algorithm extensively used in various predictive modeling tasks. While XGBoost is not inherently dedicated for time series data, its flexibility and ability to handle complex relationships in data make it adaptable for time-dependent predictions.<br>
#### Model Evaluation: 
I evaluated the performance of each model using metrics: Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE) and plots.

| Model          | Mean Absolute Error (MAE) | Mean Absolute Percentage Error (MAPE) |
|----------------|---------------------------|---------------------------------------|
| Holt-Winters   | 2.94                      | 0.16                                  |
| AutoARIMA      | 3.65                      | 0.24                                  |
| Prophet        | 3.35                      | 0.22                                  |
| XGBoost        | 2.84                      | 0.19                                  |

![HW](https://github.com/sylwiaSekula/Air_Quality/assets/110921660/3336886f-1698-44ee-a323-520b6549dc18)

![AutoARIMA](https://github.com/sylwiaSekula/Air_Quality/assets/110921660/343cb315-332a-4373-bbdc-2c2c81ca911e)

![Prophet](https://github.com/sylwiaSekula/Air_Quality/assets/110921660/2946b4ae-fd66-4d29-9ceb-a6438590a219)

![XGBoost](https://github.com/sylwiaSekula/Air_Quality/assets/110921660/ae5ca9cb-7128-48d3-b909-94928e38e277)


#### Conclusion:<br>

In summary, among the evaluated models:
- **Holt-Winters (Exponential Smoothing)** this model did the best, with the smallest errors - a MAE of 2.94 and a MAPE of 0.16. It's the most accurate.

- **XGBoost** demonstrates strong performance with the best MAE of 2.84 and a MAPE of 0.19. It's a good alternative to Holt-Winters.

- **Prophet** and **AutoARIMA** both exhibit slightly higher errors compared to the top two performers. Prophet reports a MAE of 3.35 and a MAPE of 0.22, while AutoARIMA shows a MAE of 3.65 and a slightly elevated MAPE of 0.24.

Considering the above metrics, **Holt-Winters (Exponential Smoothing)** seems to be the most suitable model for predicting this specific dataset and task, **XGBoost** is a good choice too.
