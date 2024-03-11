from django.urls import path, include
from django.contrib.auth.models import User
from rest_framework import routers, serializers, viewsets
from api.models import AdaBoostClassifier, ClassificationResults, DecisionTreeClassifier, GradientBoostClassifier, RandomForestClassifier, User

class RandomForestSerializer(serializers.ModelSerializer):
    class Meta:
        model = RandomForestClassifier
        fields = ["max_depth", "criterion", "username"]

class AdaBoostSerializer(serializers.ModelSerializer):
    class Meta:
        model = AdaBoostClassifier
        fields = ["n_estimators", "learning_rate", "username"]

class GradientBoostSerializer(serializers.ModelSerializer):
    class Meta:
        model = GradientBoostClassifier
        fields = ["n_estimators", "loss", "learning_rate", "username"]

class DecisionTreeSerializer(serializers.ModelSerializer):
    class Meta:
        model = DecisionTreeClassifier
        fields = ["max_depth", "criterion", "username"]

class ClassificationResultsSerializer(serializers.ModelSerializer):
    class Meta:
        model = ClassificationResults
        fields = ["username", "algorithm", "score"]