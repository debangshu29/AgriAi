from django.urls import path

from . import views

app_name = 'monitoring'

urlpatterns = [
    path('', views.home, name='home'),
    path('scan/<int:pk>/', views.scan_detail, name='scan_detail'),
]
