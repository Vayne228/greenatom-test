from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
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

text_clf = Pipeline([('vect', CountVectorizer()),
	('tfidf', TfidfTransformer()),
	('clf', MultinomialNB())])

text_clf.fit(texts, texts_labels)
filename = 'model3.sav'
joblib.dump(text_clf, filename)

