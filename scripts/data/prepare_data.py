import pandas as pd

from scripts.settings import *


def select_station_and_resample_dataset(dataframe: pd.DataFrame, station_column_name: str, selected_station: str,
                                        date_time_column_name: str, resample_label: str) -> pd.DataFrame:
    """
    Selects data for a specific station from the  DataFrame, resamples it based on a provided time label,
    and returns the resampled dataset.
    :param dataframe: pd.DataFrame,a pandas DataFrame containing the dataset
    :param station_column_name: str, the name of the column containing station information
    :param selected_station: str, the name of the selected station
    :param date_time_column_name: str, the name of the column containing date and time
    :param resample_label: str,the time frequency label to which the data will be resampled (e.g., 'M' for monthly).
    :return: pd.DataFrame, a pandas DataFrame with the resampled data for the selected station.
    """
    dataframe = dataframe[dataframe[station_column_name] == selected_station].copy()
    dataframe[date_time_column_name] = pd.to_datetime(dataframe[date_time_column_name])
    dataframe.set_index(date_time_column_name, inplace=True)
    dataframe = dataframe.drop(station_column_name, axis=1)
    dataframe = dataframe.resample(resample_label).mean()
    return dataframe


def create_train_test_dateset(dataframe: pd.DataFrame, n_periods: int, columns_to_drop: list,
                              train_dataset_columns: list) -> pd.DataFrame:
    """
    Prepares the training dataset for statistical models for time series forecasting
    by filtering data up to the specified end date, dropping selected columns,
    and renaming the columns as per the provided list.
    :param dataframe: pd.DataFrame, the dataset
    :param n_periods: int, the number of periods used for creating the train and test dataset.
    :param columns_to_drop: list, the list of columns to drop
    :param train_dataset_columns: list, the List of column names for the training dataset
    :return: pd.DataFrame, prepared training dataset
    """
    dataframe = dataframe.drop(columns_to_drop, axis=1)
    dataframe.reset_index(inplace=True)
    dataframe.columns = train_dataset_columns
    dataframe_train, dataframe_test = dataframe.iloc[:-n_periods], dataframe.iloc[-n_periods:]
    return dataframe_train, dataframe_test


def main():
    df = pd.read_csv(input_path)
    station_column_name = 'Station'
    selected_station = 'Taubaté'
    date_time_column_name = 'Datetime'
    resample_label = "M"
    n_periods = 12
    columns_to_drop = ['Benzene', 'CO', 'PM2.5', 'NO2', 'O3', 'SO2', 'Toluene', 'TRS']
    dataset_columns = ['ds', 'y']
    # resample the dataset
    df = select_station_and_resample_dataset(df, station_column_name, selected_station, date_time_column_name,
                                             resample_label)
    # save the resampled dataset
    df.to_csv(output_resampled_path, index=True)
    # prepare the train dataset for statistical models
    df_train, df_test = create_train_test_dateset(df, n_periods, columns_to_drop, dataset_columns)
    # save the train dataset for statistical models
    df_train.to_csv(output_train_path, index=False)
    df_test.to_csv(output_test_path, index=False)


if __name__ == '__main__':
    main()
