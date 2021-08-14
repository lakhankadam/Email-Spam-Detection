from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from read_data import read_csv_data, clean_texts

def split_train_test_data():
    df = read_csv_data()
    message = CountVectorizer(analyzer=clean_texts).fit_transform(df['text'])
    xtrain, xtest, ytrain, ytest = train_test_split(message, df['spam'], test_size=0.20, random_state=0)
    # To see the shape of the data
    print(message.shape)
    return xtrain, xtest, ytrain, ytest

print(split_train_test_data())