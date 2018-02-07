from collections import defaultdict
from time import time
import numpy as np
from pandas import read_json, DataFrame
import pandas as pd
import pickle

#from sklearn.svm import SVC
#from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


print('Loading dataframe...')
t0 = time()
with open('jokes.df.pickle', 'rb') as pickle_file:
    df = pickle.load(pickle_file)
print("done in %0.3fs" % (time() - t0))

print('Flattening {} d2v lists ...'.format(df.d2v.values.shape[0]))
X = np.reshape([r for r in df.d2v.values], (df.d2v.values.shape[0], 300))
y = pd.cut(df.score, [-1, 15, 50000], labels=['MEH', 'GOOD'])
print(df.score.describe())
print("done in %0.3fs" % (time() - t0))

print('Splitting into a training and testing set ...')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)
print("done in %0.3fs" % (time() - t0))

print("Projecting the input data on the d2v orthonormal basis (PCA'ing) ...")
pca = PCA(n_components=32, svd_solver='randomized', whiten=True).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print("done in %0.3fs" % (time() - t0))

for XXTest, XXTrain, XXName in [(X_test_pca, X_train_pca, "With PCA"), (X_test, X_train, "Without PCA")]:
    print('Fitting Classifier GaussianNB on {}'.format(XXName))
    cls = GaussianNB()
    cls.fit(XXTrain, y_train)
    print("done in %0.3fs" % (time() - t0))

    print("Predicting scores on the test set")
    t0 = time()
    y_pred = cls.predict(XXTest)
    print("done in %0.3fs" % (time() - t0))

    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

print('with or w/o pca:  which took longer, did better?')
