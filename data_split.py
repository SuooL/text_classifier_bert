import pandas as pd
import torch
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
DATA_PATH = "/Users/suool/Downloads/after_preprocess.csv"
NUMBER_EPOCHS = 10


def pre_process(path):
    df = pd.read_csv(DATA_PATH,sep = '\t')
    df = df.fillna('')
    df = df.iloc[:, :]
    df.helpful = pd.factorize(df.helpful)[0]

    train, test = train_test_split(
        df,
        test_size=0.1,
        random_state=RANDOM_STATE,
        stratify=df.helpful.values
    )

    return train, test

train_set, test_set = pre_process(path=DATA_PATH)



train_set.to_csv('train.csv',index=False,header=True, sep='\t')
test_set.to_csv('test.csv',index=False,header=True, sep='\t')