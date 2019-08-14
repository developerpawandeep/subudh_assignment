import numpy as np 
import pandas as pd 
import os

df=pd.read_csv("mozilla_thunderbird.csv")
df_train=pd.read_csv("train.csv")
df_test=pd.read_csv("test.csv")


df['Full Text'] = df['Title'].str.cat(df['Description'], sep =" ")

full_text = df.loc[:,'Full Text'].tolist()

import nltk
import re
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words=stopwords.words('english')

from nltk.tokenize import sent_tokenize, word_tokenize

text=[] 
for i in full_text:
    #text.append(word_tokenize(str(i)))
    if str(i) == "nan":
        text.append("")
        continue
    text.append(str(i))
    
import re
text1=[]
for item in text:
    word = re.sub('[^a-zA-Z\s]+','',item)
    spli = word.split(' ')
    word1 = [w for w in spli if len(w)>=2]
    word1=' '.join(word1)
    text1.append(word1.lower())
    
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_df=0.80,stop_words=stop_words)

word_count = vectorizer.fit_transform(text1)

list(vectorizer.vocabulary_.keys())[:10]


from sklearn.feature_extraction.text import TfidfTransformer
vectorizer1 = TfidfTransformer(smooth_idf=True,use_idf=True)

vectorizer1.fit(word_count)



te = text1[23]


feature_names=vectorizer.get_feature_names()
tf_idf_vector=vectorizer1.transform(vectorizer.transform([te]))

sorted_items=sort_coo(tf_idf_vector.tocoo())
 
#extract only the top n; n here is 10
keywords=extract_topn_from_vector(feature_names,sorted_items,200)


keywords=[]
for item in text1:
     tf_idf_vector=vectorizer1.transform(vectorizer.transform([item]))
     sorted_items=sort_coo(tf_idf_vector.tocoo())
     keyl = extract_topn_from_vector(feature_names,sorted_items,10000)
     keyw=[key for key in keyl.keys()]
     keywords.append(keyw)
keywords1=[]
for item in keywords:
    wordjoi = ' '.join(item)
    keywords1.append(wordjoi)

df["keywords"] = keywords1

train = pd.merge(df_train, df, how='left', on=['Issue_id'])
test = pd.merge(df_test, df, how='left', on=['Issue_id'])




test =  test.drop(['Created_time','Resolved_time','Duplicate'], axis = 1)
del train['Created_time']
del train['Resolved_time']
del train['Title']
del train['Description']
del test['Title']
del test['Description']
del train['Full Text']
del test['Full Text']


train.head()
test.head()












