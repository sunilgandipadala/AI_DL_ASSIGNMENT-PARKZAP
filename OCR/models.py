from distutils.command.upload import upload
from django.db import models

class Images(models.Model):
    image=models.ImageField(upload_to='photos/%Y/%m/%d/')

    def __str__(self):
        return "Image"
    