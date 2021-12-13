import re

import pandas as pd
from nltk import SnowballStemmer
from nltk.corpus import stopwords

##Tvättningen av datasetet gör: 1) om alla ord till sin grundform m.h.a. stemmer. 2) Tar bort vanligt förekommande ord från stopwords, ex. "for", "and" och "i". 3) Gör om alla bokstäver till små.

data = pd.read_csv('catXYdata.csv')
print(data)
stemmer = SnowballStemmer('english')    ##gör om ord till sin grundform
stopWords = stopwords.words('english')  ##vanligt förekommande ord
data['cleanedDesc'] = data['description'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z)]", " ", x).split() if i not in stopWords]).lower())
data[['cleanedDesc', 'points']].to_csv('CleanCatXYdata.csv')
