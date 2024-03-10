from django.db import models

# Create your models here.

class User(models.Model):
    username = models.CharField(50, unique=True)

class RandomForestClassifier(models.Model):
    name = models.ForeignKey('User', on_delete=models.CASCADE)
    max_depth = models.IntegerField(default=5)
    criterion = models.CharField(max_length=50, default='gini')

    def __str__(self):
        return "RCla"

class GradientBoostClassifier(models.Model):
    name = models.ForeignKey('User', on_delete=models.CASCADE)
    loss = models.CharField(max_length=50, default='log_loss')
    n_estimators = models.IntegerField(default=50)
    learning_rate = models.FloatField(default=1)

    def __str__(self):
        return "GBoostCla"

class AdaBoostClassifier(models.Model):
    name = models.ForeignKey('User', on_delete=models.CASCADE)
    n_estimators = models.IntegerField(default=50)
    learning_rate = models.FloatField(default=1)

    def __str__(self):
        return "ABoostCla"

class DecisionTreeClassifier(models.Model):
    name = models.ForeignKey('User', on_delete=models.CASCADE)
    max_depth = models.IntegerField(default=5)
    criterion = models.CharField(max_length=50, default='gini')

    def __str__(self):
        return "DTreeCla"

class ClassificationResults(models.Model):
    user = models.ForeignKey('User', on_delete=models.CASCADE)
    score = models.FloatField(default=0)