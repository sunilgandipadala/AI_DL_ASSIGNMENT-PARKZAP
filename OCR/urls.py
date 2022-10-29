from django.urls import path
from . import views

urlpatterns=[
    path("",views.index,name='index'),
    path("reset",views.reset,name='reset'),
    path("img",views.disp,name = 'img'),
    path("<path:img>",views.extract_text,name='extract_text'),
]