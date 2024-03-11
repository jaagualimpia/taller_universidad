from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response

from .ai.inferece_service import InferenceService
from .serializers import ClassificationResultsSerializer, RandomForestSerializer, DecisionTreeSerializer, GradientBoostSerializer, AdaBoostSerializer
from .models import ClassificationResults

inference_service = InferenceService()

# Create your views here.
class RandomForestClassifierEndpoint(APIView):
    def get(self, request):
        return Response("RandomForestClassifierEndpoint GET")

    def post(self, request):
        random_fc = RandomForestSerializer(data=request.data)

        if random_fc.is_valid():

            score = inference_service.get_random_forest_accuracy(
                max_depth = random_fc.validated_data['max_depth'],
                criterion = random_fc.validated_data['criterion']
            )

            ClassificationResults.objects.create(
                username = random_fc.validated_data['username'],
                algorithm = "Random Forest",
                score = score
            ).save()

            random_fc.save()

            return Response(random_fc.data)

        return Response(random_fc.errors)
    
class DecisionTreeClassifierEndpoint(APIView):
    def get(self, request):
        return Response("RandomForestClassifierEndpoint GET")

    def post(self, request):
        decision_tree = DecisionTreeSerializer(data=request.data)

        if decision_tree.is_valid():

            score = inference_service.get_decision_tree_accuracy(
                max_depth = decision_tree.validated_data['max_depth'],
                criterion = decision_tree.validated_data['criterion']
            )

            ClassificationResults.objects.create(
                username = decision_tree.validated_data['username'],
                algorithm = "Decision Tree",
                score = score
            ).save()

            decision_tree.save()
            return Response(decision_tree.data)

        return Response(decision_tree.errors)
    

class GradientBoostClassifierEndpoint(APIView):
    def get(self, request):
        return Response("RandomForestClassifierEndpoint GET")

    def post(self, request):
        gradient_boost = GradientBoostSerializer(data=request.data)

        if gradient_boost.is_valid():

            score = inference_service.get_gradient_boosting_accuracy(
                loss = gradient_boost.validated_data['loss'],
                learning_rate = gradient_boost.validated_data['learning_rate'],
                n_estimators = gradient_boost.validated_data['n_estimators']
            )

            ClassificationResults.objects.create(
                username = gradient_boost.validated_data['username'],
                algorithm = "Extreme Gradient Boosting",
                score = score
            ).save()

            gradient_boost.save()
            return Response(gradient_boost.data)

        return Response(gradient_boost.errors)
    
class ADABoostClassifierEndpoint(APIView):
    def get(self, request):
        return Response("RandomForestClassifierEndpoint GET")

    def post(self, request):
        ada_boost = AdaBoostSerializer(data=request.data)

        if ada_boost.is_valid():

            score = inference_service.get_gradient_boosting_accuracy(
                learning_rate = ada_boost.validated_data['learning_rate'],
                n_estimators = ada_boost.validated_data['n_estimators']
            )

            ClassificationResults.objects.create(
                username = ada_boost.validated_data['username'],
                algorithm = "Extreme Gradient Boosting",
                score = score
            ).save()

            ada_boost.save()
            return Response(ada_boost.data)

        return Response(ada_boost.errors)
    
class ClassificationResultsEndpoint(APIView):
    def get(self, request):
        results = ClassificationResults.objects.all().order_by('-score')

        clas_results = ClassificationResultsSerializer(instance=results, many=True)
        
        return Response(clas_results.data)
