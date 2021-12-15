import os
import pickle
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

dataSet = "CleanXYdata.csv"


algoritm = MultinomialNB()
#algoritm = LinearSVC()


#nGram_range = (1, 1)
#nGram_range = (1, 2)
nGram_range = (1, 3)

#modelPath = "oneGramNBmodel.sav"
#modelPath = "twoGramNBmodel.sav"
#modelPath = "threeGramNBmodel.sav"

#modelPath = "oneGramLinModel.sav"
#modelPath = "twoGramLinModel.sav"
modelPath = "threeGramLinModel.sav"



targets = ["OK", "Good", "Tasty", "Perfect"]

data = pd.read_csv(dataSet)
X_train, X_test, y_train, y_test = train_test_split(data.cleanedDesc, data.points, test_size=0.2)

if not (os.path.exists(modelPath)):  #if model dont exist in dir
    pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=nGram_range)),
                         ('chi', SelectKBest(chi2, k=10000)),
                         ('clf', algoritm)])

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
print(model.predict(["delicious wine, a deep complex taste with impressive deepness. Excellent work from the winemaker"]))
