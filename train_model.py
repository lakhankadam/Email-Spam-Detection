from sklearn.naive_bayes import MultinomialNB
from split_data import split_train_test_data

def train_model():
    xtrain, xtest, ytrain, ytest = split_train_test_data()
    classifier = MultinomialNB().fit(xtrain, ytrain)
    print(classifier.predict(xtrain))
    print(ytrain.values)
    return classifier