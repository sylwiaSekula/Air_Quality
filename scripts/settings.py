import os
input_filename = 'sp_air_quality.csv'
output_resampled_filename = 'df_resampled.csv'
output_train_filename = 'df_train.csv'
output_test_filename = 'df_test.csv'
input_path = os.path.join('/home/sylwia/Dokumenty/Projects/Air_Quality/scripts/data/', input_filename)
output_resampled_path = os.path.join('/home/sylwia/Dokumenty/Projects/Air_Quality/scripts/data/', output_resampled_filename)
output_train_path = os.path.join('/home/sylwia/Dokumenty/Projects/Air_Quality/scripts/data/', output_train_filename)
output_test_path = os.path.join('/home/sylwia/Dokumenty/Projects/Air_Quality/scripts/data/', output_test_filename)
date_time_column_name = 'Datetime'
datestamp_column = 'ds'
predicted_column = 'y'
n_periods = 12


