
def merge():

  dfs = []

  # Iterate over the files in the directory
  for filename in os.listdir(directory):
      if filename.endswith('.csv'):  # Check if the file is a CSV file
          file_path = os.path.join(directory, filename)
          df = pd.read_csv(file_path)
          dfs.append(df)

  # Concatenate the dataframes
  merged_df = pd.concat(dfs)

  return merge_df
