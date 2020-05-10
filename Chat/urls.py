from django.urls import path
from . import views
from django.conf.urls import url
urlpatterns =[
    path('',views.home, name='home'),
    path('get',views.chatbot, name='chatbot'),
    path('index',views.index, name='index'),
    path('register',views.register, name='register'),
    url(r'^posts/(?P<slug>[-\w]+)/$', views.single, name='post'),

]

