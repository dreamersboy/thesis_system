from django.db import models

# Create your models here.


class Userprofile(models.Model):
    name = models.CharField(max_length=20)
    icon = models.ImageField(upload_to="uploads/%Y/%m/%d",verbose_name="用户头像")

    class Meta:
        db_table = 'userprofile'
        verbose_name = '测试图片'
        verbose_name_plural = verbose_name