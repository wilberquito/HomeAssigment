import pandas as pd

def get_dataframes():
  """
  The function returns 3 dataframes for binary classification problem as
  the train source csv contains just positive and negative samples. In the test
  dataset the neutral samples are droped as the domain changes.

  Another intersting point about the function is that it also maps categorical
  variable to ordinal numbers. I could apply a one hot encoder but as the order
  is important here I let it like this. :)
  """
  header_names = ['polarity', 'tweet_id', 'date', 'query', 'user_id', 'text']

  # Train df
  train_df = pd.read_csv('extracted_data/data/Sentiment140-train.csv',  encoding='latin-1', header=None)
  train_df.columns = header_names

  # Mapping polarity values
  mapping_polarity = {
      0: 0,
      4: 1
  }
  train_df['polarity'] = train_df['polarity'].map(mapping_polarity)

  # Drop repeated text and similar polarity
  repeated_samples = train_df[train_df[['text', 'polarity']].duplicated()].index
  train_df = train_df.drop(repeated_samples).reset_index()

  # Test df
  test_df = pd.read_csv('extracted_data/data/Sentiment140-test.csv', header=None)
  test_df.columns = header_names

  # Mapping polarity values
  test_df = test_df.drop(test_df[test_df.polarity == 2].index).reset_index()
  test_df['polarity'] = test_df['polarity'].map(mapping_polarity)


  # Dubling test df
  dublin_test_df = pd.read_csv('extracted_data/data/citypulse.dublin_city_council.test.csv')
  renaming = {
    'id_str': 'tweet_id',
    'id': 'user_id',
    'sentiment': 'polarity'
  }
  dublin_test_df = dublin_test_df.rename(columns=renaming)

  # Mapping polarity values
  mapping_polarity = {
    'negative': 0,
    'neutral': 1,
    'positive': 2
  }
  dublin_test_df['polarity'] = dublin_test_df['polarity'].map(mapping_polarity)
  dublin_test_df = dublin_test_df.drop(dublin_test_df[dublin_test_df.polarity == 2].index).reset_index()

  # Returns the dataframes to work with
  return train_df, test_df, dublin_test_df
