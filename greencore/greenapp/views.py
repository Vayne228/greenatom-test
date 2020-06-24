from django.shortcuts import render
from django.views.generic import View
from django.shortcuts import redirect
import os.path
import joblib

#load trained model
GDRAT_abs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'predict_models/model.sav')
loaded_model = joblib.load(GDRAT_abs_path)


class Start(View):
	template = 'greenapp/start.html'
	raise_exception = True

	def get(self, request):
		#unk is unknown type of review
		return render(request, self.template, context={'unk':1})

	def post(self, request):
		rev = request.POST.dict()['review']
		res = loaded_model.predict([rev])[0]
		if res:
			return render(request, self.template, context={'rev':rev, 'pos':1})
		else:
			return render(request, self.template, context={'rev':rev, 'neg':1})

			