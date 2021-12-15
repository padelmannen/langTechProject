import re
import pandas as pd
from nltk import SnowballStemmer
from nltk.corpus import stopwords

##Tvättnngen av datasetet gör: 1) Tar bort alla skiljetecken 2) Tar bort vanligt förekommande ord från stopwords, ex. "for", "and" och "i". 3) Gör om alla bokstäver till små.

data = pd.read_csv('catXYdata.csv')
stemmer = SnowballStemmer('english')    ##gör om ord till sin grundform
stopWords = stopwords.words('english')  ##vanligt förekommande ord
data['cleanedDesc'] = data['description'].apply(lambda x: " ".join([i for i in re.sub("[^a-zA-Z)]", " ", x).split() if i not in stopWords]).lower())
data[['cleanedDesc', 'points']].to_csv('CleanXYdata.csv')