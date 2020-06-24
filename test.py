import joblib
import os
loaded_model = joblib.load("model.sav")

texts_check = []
#load test reviews
arr = os.listdir("aclImdb//test//pos")
for file in arr:
    with open("aclImdb//test//pos//" + file, 'r', encoding="utf_8_sig") as f:
        texts_check.append(f.read())
texts_labels = [1] * len(texts_check)
texts_neg = []
arr = os.listdir("aclImdb//test//neg")
for file in arr:
    with open("aclImdb//test//neg//" + file, 'r', encoding="utf_8_sig") as f:
        texts_neg.append(f.read())
texts_check = texts_check + texts_neg
texts_labels = texts_labels + [0] * len(texts_neg)

result = loaded_model.score(texts_check, texts_labels)
print(result) #0.83552