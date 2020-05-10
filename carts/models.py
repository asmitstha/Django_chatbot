from django.db import models
from Chat.models import Product
# Create your models here.
class Cart(models.Model):
    total = models.DecimalField(max_digits=100, decimal_places=2,default=0.02)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    active = models.BooleanField(default=True)
    def __unicode__(self):
        return "Cart id %s"%(self.id)

class CartItem(models.Model):
    cart=models.ForeignKey(Cart,null=True,blank=True,on_delete=models.CASCADE)
    product = models.ForeignKey(Product,on_delete=models.CASCADE)
    quantity= models.IntegerField(default=1)
    line_total=models.DecimalField(default=10.9,decimal_places=2,max_digits=100000)
    def __unicode__(self):
            return  self.product.name