import pickle

import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.exp_smoothing import ExponentialSmoothing

from scripts.settings import *


def read_data(train_path: str, test_path: str, resampled_path: str, date_time_column_name: str, datestamp_column: str) -> (pd.DataFrame, pd.DataFrame):
    """
    Read and prepare time series data from CSV files.
    :param train_path: str, the filepath to the training data CSV file
    :param test_path: str, the filepath to the test data for a machine learning model CSV file
    :param resampled_path: str, the filepath to the resampled data CSV file
    :param date_time_column_name: str, the name of the column in the resampled data CSV file that represents the date
    :param datestamp_column: str, the name of the column in the training data CSV file that represents the date
    :return:
        - df_train: pd.DataFrame, the processed training data
        - df_train_ml: pd.DataFrame, the processed training data for machine learning
        - df_test_ml: pd.DataFrame, the processed test data for machine learning
        - df_resampled: pd.DataFrame, the processed resampled data
    """
    df_train = pd.read_csv(train_path)
    df_train[datestamp_column] = pd.to_datetime(df_train[datestamp_column])
    df_train.set_index(datestamp_column, inplace=True)
    df_train.index.freq = "M"
    df_resampled = pd.read_csv(resampled_path)
    df_resampled.set_index(pd.to_datetime(df_resampled[date_time_column_name]), inplace=True)
    df_test = pd.read_csv(test_path)
    df_test[datestamp_column] = pd.to_datetime(df_test[datestamp_column])
    df_test.set_index(datestamp_column, inplace=True)
    return df_train, df_test, df_resampled


def predict_model(df_train: pd.DataFrame, df_test_ml: pd.DataFrame, model, predicted_column: str,
                  datestamp_column: str) -> pd.Series:
    """
    Make predictions using the time series forecasting models.
    :param df_train: (pd.DataFrame), the DataFrame with training data, indexed by the specified datestamp_column
    :param df_test_ml: pd.DataFrame, the DataFrame with machine learning test data
    :param model: the time series forecasting model (AutoARIMA, ExponentialSmoothing, Prophet)
    :param predicted_column: (str), the column name of the target variable
    :param datestamp_column: (str), the name of the column representing the date or timestamp in the data
    :return: pd.Series: Model-specific output:
    - ExponentialSmoothing or AutoARIMA: pd.Series of predictions.
    - Prophet: pd.Series of forecasted values.
    - XGBRegressor: pd.Series of predicted values.
    Raises:
    - TypeError: If the model type is not one of (ExponentialSmoothing, AutoARIMA, Prophet, or XGBRegressor).
    """
    if isinstance(model, (AutoARIMA, ExponentialSmoothing)):
        fh_hw_arima = pd.date_range(df_train.index[-1], periods=n_periods, freq='M')
        y_pred_model = model.predict(fh=fh_hw_arima)
        return y_pred_model
    elif isinstance(model, Prophet):
        df_train.reset_index(inplace=True)
        fh_prophet = model.make_future_dataframe(periods=n_periods, freq="M")
        forecast_prophet = model.predict(fh_prophet)
        forecast_prophet[datestamp_column] = pd.to_datetime(forecast_prophet[datestamp_column])
        forecast_prophet.set_index(datestamp_column, inplace=True)
        forecast_prophet = forecast_prophet[(forecast_prophet.index >= pd.to_datetime('2019-10-01'))]
        return forecast_prophet['yhat']
    elif isinstance(model, xgb.XGBRegressor):
        X_test = df_test_ml.drop(predicted_column, axis=1)
        y_test = df_test_ml[predicted_column]
        fh = pd.date_range(y_test.index[0], periods=n_periods, freq='M')
        y_pred_list = []
        for i in range(X_test.shape[0] // 12):
            if i != 0:
                X_test.iloc[12 * i: 12 * (i + 1), -1] = y_pred
            y_pred = model.predict(X_test[12 * i: 12 * (i + 1)])
            pred = pd.Series(data=y_pred, index=fh[12 * i: 12 * (i + 1)])
            y_pred_list.append(pred)
        prediction = pd.concat(y_pred_list)
        y_pred_model = prediction
        return y_pred_model

    else:
        raise TypeError("Use one of (ExponentialSmoothing, AutoARIMA,Prophet or XGBRegressor) model type.")


def main():
    # read the data
    df_train, df_test, df_resampled = read_data(output_train_path, output_test_path, output_resampled_path,
                                                date_time_column_name, datestamp_column)
    df_train_ml, df_test_ml, df_resampled = read_data(output_train_path_ml, output_test_path_ml, output_resampled_path,
                                                      date_time_column_name, datestamp_column)

    # read the models
    hw = pickle.load(open(os.path.join(trained_model_dir, HW_file), 'rb'))
    arima = pickle.load(open(os.path.join(trained_model_dir, AutoArima_file), 'rb'))
    prophet = pickle.load(open(os.path.join(trained_model_dir, Prophet_file), 'rb'))
    xgboost = pickle.load(open(os.path.join(trained_model_dir, Xgboost_file), 'rb'))
    models = {
        'HW': hw,
        'AutoARIMA': arima,
        'Prophet': prophet,
        'XGBoost': xgboost
    }

    # predict and plot
    for model_name, model in models.items():
        predicted = predict_model(df_train, df_test_ml, model, predicted_column, datestamp_column)
        mae = mean_absolute_error(df_test_ml[predicted_column], predicted)
        mape = mean_absolute_percentage_error(df_test_ml[predicted_column], predicted)
        if model == xgboost:
            plt.figure(figsize=(12, 8))
            plt.plot(df_resampled['PM10'], label='actuals')  # plot actual PM10 values from the resampled data
            plt.plot(predicted, label='predicted')  # plot predicted values from the model
            plt.title(f'{model_name} MAE: {mae}, MAPE: {mape}')
            plt.legend()
            plt.savefig(f'../../plots/{model_name}.png', format='png')
            plt.show()
            print(model_name, f'Mean absoulte error: {mae}, Mean absolute_percentage_error: {mape}')
        else:
            plt.figure(figsize=(12, 8))
            plt.plot(df_resampled['PM10'], label='actuals')  # plot actual PM10 values from the resampled data
            plt.plot(predicted, label='predicted')  # plot predicted values from the model
            plt.title(f'{model_name} MAE: {mae}, MAPE: {mape}')
            plt.legend()
            plt.savefig(f'../../plots/{model_name}.png', format='png')
            plt.show()
            print(model_name, f'Mean absoulte error: {mae}, Mean absolute_percentage_error: {mape}')


if __name__ == '__main__':
    main()
