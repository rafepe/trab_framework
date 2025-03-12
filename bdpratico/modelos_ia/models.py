from django.db import models

class DadosVinho(models.Model):
    fixed_acidity = models.FloatField(null=True, blank=True, verbose_name='Fixed Acidity')
    volatile_acidity = models.FloatField(null=True, blank=True, verbose_name='Volatile Acidity')
    citric_acid = models.FloatField(null=True, blank=True, verbose_name='Citric Acid')
    residual_sugar = models.FloatField(null=True, blank=True, verbose_name='Residual Sugar')
    chlorides = models.FloatField(null=True, blank=True, verbose_name='Chlorides')
    free_sulfur_dioxide = models.FloatField(null=True, blank=True, verbose_name='Free Sulfur Dioxide')
    total_sulfur_dioxide = models.FloatField(null=True, blank=True, verbose_name='Total Sulfur Dioxide')
    density = models.FloatField(null=True, blank=True, verbose_name='Density')
    pH = models.FloatField(null=True, blank=True, verbose_name='pH')
    sulphates = models.FloatField(null=True, blank=True, verbose_name='Sulphates')
    alcohol = models.FloatField(null=True, blank=True, verbose_name='Alcohol')
    quality = models.IntegerField(null=True, blank=True, verbose_name='Quality')

    def __str__(self):
        return f"Vinho {self.id} - Qualidade {self.quality}"

    class Meta:
        ordering = ['quality']