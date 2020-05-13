from django.shortcuts import render, HttpResponseRedirect
from carts.models import Cart
from django.urls import reverse
from django.contrib.auth.models import User
from .models import Order
from .utils import id_generator

import time


def orders(request):
    context={}
    template="orders/user.html"
    return render(request,template, context)

# Create your views here.
def checkout(request):


    try:
        the_id = request.session['cart_id']
        cart=Cart.objects.get(id=the_id)
    except:
        the_id = None
        return HttpResponseRedirect(reverse("cart"))

    new_order, created= Order.objects.get_or_create(cart=cart)
    if created:
        new_order.order_id= id_generator() #str(time.time())
        new_order.save()
    new_order.user = request.user
    new_order.save()
    if new_order.status == "Finished":
        del request.session['cart_id']
        del request.session['items_total']
    return render(request,"index.html", {})