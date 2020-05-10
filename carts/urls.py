from django.urls import path
from . import views
from django.conf.urls import url
urlpatterns =[
    path('cart',views.views, name='cart'),
    url(r'^cart/(?P<slug>[-\w]+)/$', views.update_cart, name='update_cart'),


]

