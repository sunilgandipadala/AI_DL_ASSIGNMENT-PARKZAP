from email.mime import image
from socket import fromshare
from django import forms
from .models import Images

class ImageForm(forms.ModelForm):
    class Meta:
        model =Images
        fields = ("image",)