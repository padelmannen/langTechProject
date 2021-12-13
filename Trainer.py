import pickle

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

data = pd.read_csv('CleanXYdata.csv')
X_train, X_test, y_train, y_test = train_test_split(data.cleanedDesc, data.points, test_size=0.2)

pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)),
                     ('chi', SelectKBest(chi2, k=10000)),
                     ('clf', LinearSVC(C=1.0, penalty='l1', max_iter=3000, dual=False))])

model = pipeline.fit(X_train, y_train)
pickle.dump(model, open('model.sav', 'wb'))

vectorizer = model.named_steps['vect']
chi = model.named_steps['chi']
clf = model.named_steps['clf']

featureNames = vectorizer.get_feature_names()
featureNames = [featureNames[i] for i in chi.get_support(indices=True)]
featureNames = np.asarray(featureNames)

targets = ["80", "90", "100"]
for i, label in enumerate(targets):
    top10 = np.argsort(clf.coef_[i])[-10:]
    print("%s: %s" % (label, " ".join(featureNames[top10])))


# some time later...

# load the model from disk
loaded_model = pickle.load(open('model.sav', 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)

