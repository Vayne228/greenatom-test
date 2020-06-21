from django.shortcuts import render
from django.views.generic import View
from django.shortcuts import redirect
import joblib

loaded_model = joblib.load("C:\\Users\\IVAN\\Desktop\\greenatom\\greenatom-test\\greencore\\greenapp\\models\\model.sav")

class Start(View):
	template = 'greenapp/start.html'
	raise_exception = True
	def get(self, request):
		return render(request, self.template)
	def post(self, request):
		rev = request.POST.dict()['review']
		res = loaded_model.predict([rev])[0]
		print(res)
		return render(request, self.template)