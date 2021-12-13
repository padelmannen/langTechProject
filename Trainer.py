import os
import pickle

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# modelPath = "./model.sav"     ##Om vi vill köra utan kategorier
# dataSet = "CleanXYdata.csv"
# targets = ["80", "90", "100"]

modelPath = "./modelCat.sav"      ##Om vi vill köra med kategorier
dataSet = "CleanCatXYdata.csv"
targets = ["OK", "Good", "Tasty", "Perfect"]

data = pd.read_csv(dataSet)
X_train, X_test, y_train, y_test = train_test_split(data.cleanedDesc, data.points, test_size=0.2)

if not (os.path.exists(modelPath)):  #if model dont exist in dir
    pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                         ('chi', SelectKBest(chi2, k=100000)),
                         ('clf', LinearSVC(C=1.0, penalty='l1', max_iter=3000, dual=False))])

    model = pipeline.fit(X_train, y_train)
    pickle.dump(model, open(modelPath, 'wb'))
else:
    model = pickle.load(open(modelPath, 'rb'))

vectorizer = model.named_steps['vect']
chi = model.named_steps['chi']
clf = model.named_steps['clf']

featureNames = vectorizer.get_feature_names()
featureNames = [featureNames[i] for i in chi.get_support(indices=True)]
featureNames = np.asarray(featureNames)
predicted = model.predict(X_test)

for i, label in enumerate(targets):
    top10 = np.argsort(clf.coef_[i])[-10:]
    print("%s: %s" % (label, " ".join(featureNames[top10])))

result = model.score(X_test, y_test)
print(metrics.classification_report(y_true=y_test, y_pred=predicted))
print(result)
print(model.predict(["that was not so tasteful, but i liked the sweetness"]))
print(model.predict(["that was probably the best wine i have ever tasted, fantastic. Round taste"]))
