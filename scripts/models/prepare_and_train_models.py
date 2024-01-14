import pickle

import pandas as pd
import xgboost as xgb
from prophet import Prophet
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.exp_smoothing import ExponentialSmoothing

from scripts.settings import *


def read_train_data(train_path: str, datestamp_column: str) -> (pd.DataFrame, pd.DataFrame):
    """
    Read and prepare train time series data from a CSV file.
    :param train_path: str, the filepath to the training data CSV file
    :param datestamp_column: str, the name of the column in the training data CSV file that represents the date
    :return:
        - df_train: pd.DataFrame, the processed training data
    """
    df_train = pd.read_csv(train_path)
    df_train[datestamp_column] = pd.to_datetime(df_train[datestamp_column])
    df_train.set_index(datestamp_column, inplace=True)
    df_train.index.freq = "M"
    return df_train


def main():
    # read the train datasets for statistical and ML models
    df_train = read_train_data(output_train_path, datestamp_column)
    df_train_ml = read_train_data(output_train_path_ml, datestamp_column)

    # create models
    hw = ExponentialSmoothing(trend='add', seasonal='multiplicative', sp=12)
    arima = AutoARIMA(sp=12, d=0, max_p=2, max_q=2, suppress_warnings=True)
    prophet = Prophet(interval_width=0.9)
    xgboost = xgb.XGBRegressor()

    models_and_files =  [(hw, HW_file),
        (arima, AutoArima_file),
        (prophet, Prophet_file),
        (xgboost, Xgboost_file)
        ]

    # Fit the model on the training data
    for model, file_name in models_and_files:
        if isinstance(model, (AutoARIMA, ExponentialSmoothing)):
            y_train = df_train[predicted_column]
            model.fit(y_train)
        elif isinstance(model, Prophet):
            df_train.reset_index(inplace=True)
            model.fit(df_train)
        elif isinstance(model, xgb.XGBRegressor):
            y_train = df_train_ml[predicted_column]
            X_train = df_train_ml.drop(predicted_column, axis=1)
            model.fit(X_train, y_train)
        # save models
        pickle.dump(model, open(os.path.join(trained_model_dir, file_name), 'wb'))


if __name__ == '__main__':
    main()
