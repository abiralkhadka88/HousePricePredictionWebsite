from django.db import models


class team(models.Model):
    team_name = models.CharField(max_length=50)
    team_address = models.CharField(max_length=50)
    team_image = models.ImageField(upload_to='static/img')
    class Meta:
        app_label = 'team'
