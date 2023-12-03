import os
import pandas as pd


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
    dataframe = dataframe.resample(resample_label).mean()
    return dataframe


def create_statistical_train_dateset(dataframe: pd.DataFrame, end_date: str, columns_to_drop: list,
                                     train_dataset_columns: list) -> pd.DataFrame:
    """
    Prepares the training dataset for statistical models for time series forecasting
    by filtering data up to the specified end date, dropping selected columns,
    and renaming the columns as per the provided list.
    :param dataframe: pd.DataFrame, the dataset
    :param end_date: str, the end date until which the data will be filtered
    :param columns_to_drop: list, the list of columns to drop
    :param train_dataset_columns: list, the List of column names for the training dataset
    :return: pd.DataFrame, prepared training dataset
    """
    dataframe_train = dataframe[(dataframe.index <= end_date)].copy()
    dataframe_train = dataframe_train.drop(columns_to_drop, axis=1)
    dataframe_train.reset_index(inplace=True)
    dataframe_train.columns = train_dataset_columns
    return dataframe_train



def main():
    current_directory = os.getcwd()
    input_filename = 'sp_air_quality.csv'
    output_resampled_filename = 'df_resampled.csv'
    output_statistical_train_filename = 'df_stat_train'
    input_path = os.path.join(current_directory, input_filename)
    output_resampled_path = os.path.join(current_directory, output_resampled_filename)
    output_train_path = os.path.join(current_directory, output_statistical_train_filename)
    df = pd.read_csv(input_path)
    station_column_name = 'Station'
    selected_station = 'Taubaté'
    date_time_column_name = 'Datetime'
    resample_label = "M"
    end_date = pd.to_datetime('2019-08-31')
    columns_to_drop = ['Benzene', 'CO', 'PM2.5', 'NO2', 'O3', 'SO2', 'Toluene', 'TRS']
    train_dataset_columns = ['ds', 'y']
    # resample the dataset
    df = select_station_and_resample_dataset(df, station_column_name, selected_station, date_time_column_name, resample_label)
    # save the resampled dataset
    df.to_csv(output_resampled_path, index=False)
    # prepare the train dataset for statistical models
    df_stat_train = create_statistical_train_dateset(df, end_date, columns_to_drop, train_dataset_columns)
    # save the train dataset for statistical models
    df_stat_train.to_csv(output_train_path)



if __name__ == '__main__':
    main()