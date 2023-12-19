import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.exp_smoothing import ExponentialSmoothing

from scripts.settings import *


def read_data(train_path: str, train_ml_path: str, test_ml_path: str, resampled_path: str, date_time_column_name: str,
              datestamp_column: str) -> (pd.DataFrame, pd.DataFrame):
    """
    Read and prepare time series data from CSV files.
    :param train_path: str, the filepath to the training data CSV file
    :param train_ml_path: str, the filepath to the training data for a machine learning model CSV file
    :param test_ml_path: str, the filepath to the test data for a machine learning model CSV file
    :param resampled_path: str, the filepath to the resampled data CSV file
    :param date_time_column_name: str, the name of the column in the resampled data CSV file that represents the date
    :param datestamp_column: str, the name of the column in the training data CSV file that represents the date
    :return:
        - df_train: pd.DataFrame, the processed training data
        - df_train_ml: pd.DataFrame, the processed training data for machine learning
        - df_test_ml: pd.DataFrame, the processed test data for machine learning
        - df_resampled: pd.DataFrame, the processed resampled data
    """

    df_resampled = pd.read_csv(resampled_path)
    df_resampled.set_index(pd.to_datetime(df_resampled[date_time_column_name]), inplace=True)
    df_train = pd.read_csv(train_path)
    df_train[datestamp_column] = pd.to_datetime(df_train[datestamp_column])
    df_train.set_index(datestamp_column, inplace=True)
    df_train.index.freq = "M"
    df_train_ml = pd.read_csv(train_ml_path)
    df_train_ml[datestamp_column] = pd.to_datetime(df_train_ml[datestamp_column])
    df_train_ml.set_index(datestamp_column, inplace=True)
    df_test_ml = pd.read_csv(test_ml_path)
    df_test_ml[datestamp_column] = pd.to_datetime(df_test_ml[datestamp_column])
    df_test_ml.set_index(datestamp_column, inplace=True)
    return df_train, df_train_ml, df_test_ml, df_resampled


def fit_model(df_train: pd.DataFrame, df_train_ml: pd.DataFrame, df_test_ml: pd.DataFrame, model, predicted_column: str,
              datestamp_column: str) -> tuple:
    """
    Fit time series forecasting models and make predictions.
    :param df_train: (pd.DataFrame), the DataFrame with training data, indexed by the specified datestamp_column
    :param df_train_ml: pd.DataFrame, the DataFrame with machine learning training data
    :param df_test_ml: pd.DataFrame, the DataFrame with machine learning test data
    :param model: the time series forecasting model (AutoARIMA, ExponentialSmoothing, Prophet)
    :param predicted_column: (str), the column name of the target variable
    :param datestamp_column: (str), the name of the column representing the date or timestamp in the data
    :return: Tuple: Model-specific output:
    - ExponentialSmoothing or AutoARIMA: Tuple of fitted values and predictions.
    - Prophet: Tuple of fitted values and forecasted values.
    - XGBRegressor: Tuple of fitted model and predicted values.
    Raises:
    - TypeError: If the model type is not one of (ExponentialSmoothing, AutoARIMA, Prophet, or XGBRegressor).
    """
    if isinstance(model, (AutoARIMA, ExponentialSmoothing)):
        y_train = df_train[predicted_column]
        fh_hw_arima = pd.date_range(df_train.index[-1], periods=12, freq='M')
        model.fit(y_train)
        y_fitted_model = model.predict(fh=df_train.index)
        y_pred_model = model.predict(fh=fh_hw_arima)
        return y_fitted_model, y_pred_model
    elif isinstance(model, Prophet):
        df_train.reset_index(inplace=True)
        model.fit(df_train)
        fh_prophet = model.make_future_dataframe(periods=n_periods, freq="M")
        forecast_prophet = model.predict(fh_prophet)
        forecast_prophet[datestamp_column] = pd.to_datetime(forecast_prophet[datestamp_column])
        forecast_prophet.set_index(datestamp_column, inplace=True)
        fitted_prophet = forecast_prophet[(forecast_prophet.index <= pd.to_datetime('2019-10-01'))]
        forecast_prophet = forecast_prophet[(forecast_prophet.index >= pd.to_datetime('2019-10-01'))]
        return fitted_prophet['yhat'], forecast_prophet['yhat']
    elif isinstance(model, xgb.XGBRegressor):
        y_train = df_train_ml[predicted_column]
        X_train = df_train_ml.drop(predicted_column, axis=1)
        X_test = df_test_ml.drop(predicted_column, axis=1)
        y_fitted_model = model.fit(X_train, y_train)
        y_pred_model = model.predict(X_test)
        return y_fitted_model, y_pred_model
    else:
        raise TypeError("Use one of (ExponentialSmoothing, AutoARIMA,Prophet or XGBRegressor) model type.")


def main():
    # read the data
    df_train, df_train_ml, df_test_ml, df_resampled = read_data(output_train_path, output_train_path_ml,
                                                                output_test_path_ml, output_resampled_path,
                                                                date_time_column_name, datestamp_column)
    # create models
    hw = ExponentialSmoothing(trend='add', seasonal='multiplicative', sp=12)
    arima = AutoARIMA(sp=12, d=0, max_p=2, max_q=2, suppress_warnings=True)
    prophet = Prophet(interval_width=0.9)
    xgboost = xgb.XGBRegressor()
    models = {
        'HW': hw,
        'AutoARIMA': arima,
        'Prophet': prophet,
        'XGBoost': xgboost
    }
    # fit the models and plot
    for model_name, model in models.items():
        # fit the model and get fitted values and predictions
        fitted, predicted = fit_model(df_train, df_train_ml, df_test_ml, model, predicted_column, datestamp_column)
        mae = mean_absolute_error(df_test_ml[predicted_column], predicted)
        if model == xgboost:
            # create the dataset with predicted values with datetime index
            datetime_index = df_test_ml.index
            predicted_df = pd.DataFrame(predicted, index=datetime_index, columns=['predicted_column'])
            plt.figure(figsize=(12, 8))
            plt.plot(df_resampled['PM10'], label='actuals')  # plot actual PM10 values from the resampled data
            plt.plot(predicted_df['predicted_column'], label='predicted')  # plot predicted values from the model
            plt.title(model_name)
            plt.legend()
            plt.savefig(f'../plots/{model_name}.png', format='png')
            plt.show()
            print(type(fitted), type(predicted))
            print(model_name, mae)
        else:
            plt.figure(figsize=(12, 8))
            plt.plot(df_resampled['PM10'], label='actuals')  # plot actual PM10 values from the resampled data
            plt.plot(fitted, label='fitted')  # plot fitted values from the model
            plt.plot(predicted, label='predicted')  # plot predicted values from the model
            plt.title(model_name)
            plt.legend()
            plt.savefig(f'../plots/{model_name}.png', format='png')
            plt.show()
            print(model_name, mae)


if __name__ == '__main__':
    main()
