import joblib
import os
loaded_model = joblib.load("model5.sav")#choose model(model,model2,...,model5)

texts_check = []
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
print(result) #model1 = 0.83656  model2 = 0.84348  model3 = 0.82956  model4 = 0.84396 model5 = 0.84396