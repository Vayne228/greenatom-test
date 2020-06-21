from django.urls import path
from .views import *

urlpatterns = [
	path('', Start.as_view(), name='start_page_url'),
]