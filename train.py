import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib

texts = []
#load positive reviews
arr = os.listdir("aclImdb//train//pos")
for file in arr:
    with open("aclImdb//train//pos//" + file, 'r', encoding="utf_8_sig") as f:
        texts.append(f.read())
texts_labels = [1] * len(texts)

#load negative reviews
texts_neg = []
arr = os.listdir("aclImdb//train//neg")
for file in arr:
    with open("aclImdb//train//neg//" + file, 'r', encoding="utf_8_sig") as f:
        texts_neg.append(f.read())
texts = texts + texts_neg
#create labels for texts (0 is negative, 1 is positive)
texts_labels = texts_labels + [0] * len(texts_neg)
text_clf = Pipeline([
                     ('tfidf', TfidfVectorizer()),
                     ('clf', RandomForestClassifier())
                     ])
 
text_clf.fit(texts, texts_labels)
filename = 'model.sav'
#save model
joblib.dump(text_clf, filename)


