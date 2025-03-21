# Generated by Django 4.2.17 on 2025-03-12 20:02

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('modelos_ia', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='DadosVinho',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('fixed_acidity', models.FloatField(blank=True, null=True, verbose_name='Fixed Acidity')),
                ('volatile_acidity', models.FloatField(blank=True, null=True, verbose_name='Volatile Acidity')),
                ('citric_acid', models.FloatField(blank=True, null=True, verbose_name='Citric Acid')),
                ('residual_sugar', models.FloatField(blank=True, null=True, verbose_name='Residual Sugar')),
                ('chlorides', models.FloatField(blank=True, null=True, verbose_name='Chlorides')),
                ('free_sulfur_dioxide', models.FloatField(blank=True, null=True, verbose_name='Free Sulfur Dioxide')),
                ('total_sulfur_dioxide', models.FloatField(blank=True, null=True, verbose_name='Total Sulfur Dioxide')),
                ('density', models.FloatField(blank=True, null=True, verbose_name='Density')),
                ('pH', models.FloatField(blank=True, null=True, verbose_name='pH')),
                ('sulphates', models.FloatField(blank=True, null=True, verbose_name='Sulphates')),
                ('alcohol', models.FloatField(blank=True, null=True, verbose_name='Alcohol')),
                ('quality', models.IntegerField(blank=True, null=True, verbose_name='Quality')),
            ],
            options={
                'ordering': ['quality'],
            },
        ),
        migrations.DeleteModel(
            name='DadosModelos',
        ),
    ]
