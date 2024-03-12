"""
URL configuration for taller_app project.

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
from django.urls import path

from api.views import ClassificationResultsEndpoint, RandomForestClassifierEndpoint, ADABoostClassifierEndpoint, DecisionTreeClassifierEndpoint, GradientBoostClassifierEndpoint, SpecificModelPredictionEndpoint

urlpatterns = [
    path('admin/', admin.site.urls),
    path("random_forest_classifier", RandomForestClassifierEndpoint.as_view()),
    path("ada_boost_classifier", ADABoostClassifierEndpoint.as_view()),
    path("gradient_boost_classifier", GradientBoostClassifierEndpoint.as_view()),
    path("decision_tree_classifier", DecisionTreeClassifierEndpoint.as_view()),
    path("classification_results", ClassificationResultsEndpoint.as_view()),
    path("spe", SpecificModelPredictionEndpoint.as_view())
]
