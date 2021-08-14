import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string

def read_csv_data():
    df = pd.read_csv("data/emails.csv")
    df.head()
    df.drop_duplicates(inplace=True)
    df['text'].head().apply(clean_texts)
    return df

def clean_texts(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)

    clean = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean

read_csv_data()