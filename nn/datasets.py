
import torch
from torch import LongTensor
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


class TwitterDataset(Dataset):
    """Simple dataset to process text that will
    be handled by a transfomer"""
    
    def __init__(self, df, tokenizer, max_length):
      
        self.texts = df['text'].values
        self.labels = df['polarity'].values
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        text = str(self.texts[index])
        label = self.labels[index]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }



def train_valid_datasets(train_df: pd.DataFrame,
                         preprocessor,
                         validate_size=0.2,
                         random_state=42):
    """Another aprouch to generate tran test datasets. But this
    time instead of using the transfomers api I do it by transforming
    the text into bofw or tfidf"""


    X, y = train_df[['text']], train_df['polarity']

    # Using some part of the train dataset as a validate set
    X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                          test_size=validate_size,
                                                          random_state=random_state)

    # Labels to numpy
    y_train = torch.from_numpy(y_train.values)
    y_valid = torch.from_numpy(y_valid.values)

    # Using sklearn transformer
    X_train = preprocessor.fit_transform(X_train).tocoo()
    X_valid = preprocessor.transform(X_valid).tocoo()

    # Sparse matrix to pytorch tensor
    X_train = torch.sparse.LongTensor(LongTensor([X_train.row.tolist(), X_train.col.tolist()]),
                                      LongTensor(X_train.data.astype(np.int32)))
    X_valid = torch.sparse.LongTensor(LongTensor([X_valid.row.tolist(), X_valid.col.tolist()]),
                                      LongTensor(X_valid.data.astype(np.int32)))
    
    # Creating the datasets
    train_dataset = TensorDataset(X_train, y_train)
    valid_dataset = TensorDataset(X_valid, y_valid)

    return train_dataset, valid_dataset
