import re

import pandas as pd
from nltk import SnowballStemmer
from nltk.corpus import stopwords

data = pd.read_csv('XYdata.csv')
print(data)
stemmer = SnowballStemmer('english')
stopWords = stopwords.words('english')
data['cleanedDesc'] = data['description'].apply(lambda x: " ".join([stemmer.stem(i) for i in re.sub("[^a-zA-Z)]", " ", x).split() if i not in stopWords]).lower())
data[['cleanedDesc', 'points']].to_csv('CleanXYdata.csv')