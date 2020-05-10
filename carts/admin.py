from django.contrib import admin

# Register your models here.
from .models import Cart, CartItem

class CartAdmin(admin.ModelAdmin):
    class meta:
        model=Cart

admin.site.register(Cart,CartAdmin )

admin.site.register(CartItem)