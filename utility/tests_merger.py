import os
import pandas as pd


def merge(directory):
    dfs = []
    # Iterate over the files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):  # Check if the file is a CSV file
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            dfs.append(df)

    # Concatenate the dataframes
    merged_df = pd.concat(dfs)
    merged_df.set_index(dfs.iloc[:, 0], inplace=True)
    merged_df.drop(columns=merged_df.columns[0], axis=1, inplace=True)
    return merged_df
