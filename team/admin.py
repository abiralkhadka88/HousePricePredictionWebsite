from django.contrib import admin
from team.models import team

class teamadmin(admin.ModelAdmin):
    list_display=('team_name', 'team_address', 'team_image')
    
admin.site.register(team,teamadmin)
# Register your models here.
