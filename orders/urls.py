from django.urls import path
from . import views
from django.conf.urls import url
urlpatterns =[

    url(r'checkout', views.checkout, name='checkout'),
    url(r'orders', views.orders, name='user_orders'),
url(r'placeorder', views.placeorder, name='placeorder'),


]

