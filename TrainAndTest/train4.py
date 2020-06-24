from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
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


from sklearn.linear_model import SGDClassifier
text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42))])

text_clf_svm.fit(texts, texts_labels)
filename = 'model4.sav'
joblib.dump(text_clf_svm, filename)

