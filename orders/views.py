from django.shortcuts import render, HttpResponseRedirect
from carts.models import Cart
from django.urls import reverse
from django.contrib.auth.models import User
from .models import Order
from .utils import id_generator
from orders.models import info
import time

def placeorder(request):
    email = request.POST['inputEmail4']
    orderid = request.POST['order']
    Address = request.POST['inputAddress']
    Address2 = request.POST['inputAddress2']
    City = request.POST['inputCity']
    State = request.POST['inputState']
    Zip = request.POST['inputZip']

    Order_info = info(email=email,orderid=orderid,Address=Address,Address2=Address2,City=City,State=State,Zip=Zip)
    Order_info.save()
    return render(request,"index.html", {})


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
    return render(request,"orders/checkout.html", {})