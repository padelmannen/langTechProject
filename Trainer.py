import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

dataSet = "CleanXYdata.csv"


#algoritm = MultinomialNB()
#algoritm = LinearSVC()

#nGram_range = (1, 1)
nGram_range = (1, 2)
#nGram_range = (1, 3)

targets = ["OK", "Good", "Tasty", "Perfect"]

data = pd.read_csv(dataSet)
X_train, X_test, y_train, y_test = train_test_split(data.cleanedDesc, data.points, test_size=0.2)
pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=nGram_range)),
                     ('clf', algoritm)])

model = pipeline.fit(X_train, y_train)

vectorizer = model.named_steps['vect']
clf = model.named_steps['clf']

featureNames = vectorizer.get_feature_names()
featureNames = np.asarray(featureNames)

wordDict = {}
for i, label in enumerate(targets):
    top10 = np.argsort(clf.coef_[i])[-10:]
    wordDict[label]=featureNames[top10]
    print("%s: %s" % (label, " ".join(featureNames[top10])))

words = pd.DataFrame(wordDict).to_latex()

predicted = model.predict(X_test)
report = classification_report(y_true=y_test, y_pred=predicted, output_dict=True)
df = pd.DataFrame(report).transpose()
print(df.to_latex())
