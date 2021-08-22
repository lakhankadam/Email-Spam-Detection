from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from train_model import train_model
from split_data import split_train_test_data

def test_training_data():
    classifier = train_model()
    xtrain, xtest, ytrain, ytest = split_train_test_data()
    pred = classifier.predict(xtrain)
    print(classification_report(ytrain, pred))
    print()
    print("Confusion Matrix: \n", confusion_matrix(ytrain, pred))
    print("Accuracy: \n", accuracy_score(ytrain, pred))

def test_test_data():
    classifier = train_model()
    xtrain, xtest, ytrain, ytest = split_train_test_data()
    pred = classifier.predict(xtest)
    print(classification_report(ytest, pred))
    print()
    print("Confusion Matrix: \n", confusion_matrix(ytest, pred))
    print("Accuracy: \n", accuracy_score(ytest, pred))

def run():
    test_training_data()
    test_test_data()

if __name__ == '__main__':
    run()
