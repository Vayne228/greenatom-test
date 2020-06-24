from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib
import os

texts = []
arr = os.listdir("aclImdb//train//pos")
for file in arr:
    with open("aclImdb//train//pos//" + file, 'r', encoding="utf_8_sig") as f:
        texts.append(f.read())
texts_labels = [1] * len(texts)

texts_neg = []
arr = os.listdir("aclImdb//train//neg")
for file in arr:
    with open("aclImdb//train//neg//" + file, 'r', encoding="utf_8_sig") as f:
        texts_neg.append(f.read())
texts = texts + texts_neg
texts_labels = texts_labels + [0] * len(texts_neg)
text_clf = Pipeline([
                     ('tfidf', TfidfVectorizer()),
                     ('clf', RandomForestClassifier())
                     ])
 
text_clf.fit(texts, texts_labels)
filename = 'model1.sav'
joblib.dump(text_clf, filename)


