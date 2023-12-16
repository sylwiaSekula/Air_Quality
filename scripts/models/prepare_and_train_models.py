import pandas as pd
from prophet import Prophet
from sktime.forecasting.arima import AutoARIMA
import matplotlib.pyplot as plt
from scripts.settings import *
from sktime.forecasting.exp_smoothing import ExponentialSmoothing


def read_data(train_path: str, resampled_path: str, date_time_column_name: str, datestamp_column: str) \
        -> (pd.DataFrame, pd.DataFrame):
    """
    Read and prepare time series data from CSV files.
    :param train_path: str, the filepath to the training data CSV file
    :param resampled_path: str, the filepath to the resampled data CSV file
    :param date_time_column_name: str, the name of the column in the resampled data CSV file that represents the date
    :param datestamp_column: str, the name of the column in the training data CSV file that represents the date
    :return:  Tuple: Model-specific output:
    - ExponentialSmoothing or AutoARIMA: Tuple of fitted values and predictions.
    - Prophet: Tuple of fitted values and forecasted values.
    Raises:
    - TypeError: If the model type is not one of (ExponentialSmoothing, AutoARIMA, or Prophet).
    """
    df_train = pd.read_csv(train_path)
    df_resampled = pd.read_csv(resampled_path)
    df_resampled.set_index(pd.to_datetime(df_resampled[date_time_column_name]), inplace=True)
    df_train[datestamp_column] = pd.to_datetime(df_train[datestamp_column])
    df_train.set_index(datestamp_column, inplace=True)
    df_train.index.freq = "M"
    return df_train, df_resampled


def fit_model(df_train: pd.DataFrame, model, predicted_column: str, datestamp_column: str) -> tuple:
    """
    Fit time series forecasting models and make predictions.
    :param df_train: (pd.DataFrame), the DataFrame with training data, indexed by the specified datestamp_column
    :param model: the time series forecasting model (AutoARIMA, ExponentialSmoothing, Prophet)
    :param predicted_column: (str), the column name of the target variable
    :param datestamp_column: (str), the name of the column representing the date or timestamp in the data
    :return:
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
        fitted_prophet = forecast_prophet[(forecast_prophet.index <= pd.to_datetime('2019-09-01'))]
        forecast_prophet = forecast_prophet[(forecast_prophet.index >= pd.to_datetime('2019-09-01'))]
        return fitted_prophet['yhat'], forecast_prophet['yhat']
    else:
        raise TypeError("Use one of (ExponentialSmoothing, AutoARIMA or Prophet) model type.")


def main():
    # read the data
    df_train, df_resampled = read_data(output_train_path, output_resampled_path, date_time_column_name,
                                       datestamp_column)

    # create models
    hw = ExponentialSmoothing(trend='add', seasonal='multiplicative', sp=12)
    arima = AutoARIMA(sp=12, d=0, max_p=2, max_q=2, suppress_warnings=True)
    prophet = Prophet(interval_width=0.9)
    models = {
        'HW': hw,
        'AutoARIMA': arima,
        'Prophet': prophet
    }
    # fit the models and plot
    for model_name, model in models.items():
        fitted, predicted = fit_model(df_train, model, predicted_column, datestamp_column)
        plt.figure(figsize=(12, 8))
        plt.plot(df_resampled['PM10'], label='actuals')
        plt.plot(fitted, label='fitted')
        plt.plot(predicted, label='predicted')
        plt.title(model_name)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    main()
