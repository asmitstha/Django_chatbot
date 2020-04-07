from django.urls import path
from . import views

urlpatterns =[
    path('',views.home, name='home'),
    path('get',views.chatbot, name='chatbot'),
    path('index',views.index, name='index'),
    path('register',views.register, name='register')
]

