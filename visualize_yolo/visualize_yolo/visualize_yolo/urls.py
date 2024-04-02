"""
URL configuration for visualize_yolo project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from run_proj import run_yolo
from . import views
from .views import home_view,test_view,display_image, process_image

urlpatterns = [
    path('admin/', admin.site.urls),
    path('test/', views.test_view, name='test'),
    path('', views.home_view, name='home'),
    path('process-image/', process_image, name='process_image'),
    #path('display-image/', display_image, name='display_image'),
    #path('', include('visualize_yolo.urls')),  # Include your app's URLs
]
