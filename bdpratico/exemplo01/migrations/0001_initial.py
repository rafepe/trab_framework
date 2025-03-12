# Generated by Django 5.1.4 on 2024-12-14 19:37

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='pessoa',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('nome', models.CharField(max_length=50, verbose_name='Nome')),
                ('email', models.CharField(max_length=50, verbose_name='eMail')),
                ('celular', models.CharField(blank=True, max_length=20, null=True, verbose_name='celular')),
                ('funcao', models.CharField(blank=True, max_length=30, null=True, verbose_name='Funcao')),
            ],
            options={
                'ordering': ['nome'],
            },
        ),
    ]
