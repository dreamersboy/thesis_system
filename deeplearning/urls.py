from django.urls import path
from . import views
urlpatterns =[

    path('',views.main1),    #主页面
    path('predict1',views.search,name='predict1'),
    path('predict2',views.result,name='predict2'),
    path('add/',views.files,name='add'),
    path('load/',views.index, name='load')
]