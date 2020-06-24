from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
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



text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42))])

parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False),'clf-svm__alpha': (1e-2, 1e-3)}

gs_clf_svm = GridSearchCV(text_clf_svm, parameters_svm, n_jobs=-1)
gs_clf_svm = gs_clf_svm.fit(texts, texts_labels)
filename = 'model5.sav'
joblib.dump(gs_clf_svm, filename)

