# Generated by Django 3.0.5 on 2020-05-10 11:52

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('orders', '0003_auto_20200510_1623'),
    ]

    operations = [
        migrations.CreateModel(
            name='info',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('email', models.CharField(max_length=200)),
                ('orderid', models.CharField(max_length=7)),
                ('Address', models.CharField(max_length=120)),
                ('Address2', models.CharField(max_length=120)),
                ('City', models.CharField(max_length=120)),
                ('State', models.CharField(max_length=120)),
                ('Zip', models.IntegerField()),
            ],
        ),
    ]
