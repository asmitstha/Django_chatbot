from django.db import models
from carts.models import Cart
from django.contrib.auth import get_user_model
# Create your models here.
User= get_user_model()

STATUS_CHOICES=(
    ("Started","Started"),
    ("Adbandoned","Adbandoned"),
    ("Finished","Finished"),
)



class Order(models.Model):
    #address
    user = models.ForeignKey(User, blank=True, null=True,on_delete=models.CASCADE)
    order_id=models.CharField(max_length=120,default='ABC',unique=True)
    cart = models.ForeignKey(Cart, on_delete=models.CASCADE)
    Sub_total = models.DecimalField(default=0.0, decimal_places=2, max_digits=100000)
    Tax_total = models.DecimalField(default=10.9, decimal_places=2, max_digits=100000)
    Final_total = models.DecimalField(default=10.9, decimal_places=2, max_digits=100000)

    status=models.CharField(max_length=120,choices=STATUS_CHOICES,default="Started")
    timestamp=models.DateTimeField(auto_now_add=True,auto_now=False)
    updated=models.DateTimeField(auto_now_add=False,auto_now=True)

    def __unicode__(self):
        return self.order_id


class info(models.Model):
    email =models.CharField(max_length=200)

    Address=models.CharField(max_length=120)
    Address2 = models.CharField(max_length=120)
    City = models.CharField(max_length=120)
    State = models.CharField(max_length=120)
    Zip = models.IntegerField()

    def __str__(self):
        return self.orderid



