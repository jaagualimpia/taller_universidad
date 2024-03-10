from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from .serializers import RandomForestSerializer, DecisionTreeSerializer, GradientBoostSerializer, AdaBoostSerializer

# Create your views here.
class RandomForestClassifierEndpoint(APIView):
    def get(self, request):
        return Response("RandomForestClassifierEndpoint GET")

    def post(self, request):
        random_fc = RandomForestSerializer(data=request.data)

        if random_fc.is_valid():
            random_fc.save()
            return Response(random_fc.data)

        return Response(random_fc.errors)
    
class DecisionTreeClassifierEndpoint(APIView):
    def get(self, request):
        return Response("RandomForestClassifierEndpoint GET")

    def post(self, request):
        decision_tree = DecisionTreeSerializer(data=request.data)

        if decision_tree.is_valid():
            decision_tree.save()
            return Response(decision_tree.data)

        return Response(decision_tree.errors)
    

class GradientBoostClassifierEndpoint(APIView):
    def get(self, request):
        return Response("RandomForestClassifierEndpoint GET")

    def post(self, request):
        gradient_boost = GradientBoostSerializer(data=request.data)

        if gradient_boost.is_valid():
            gradient_boost.save()
            return Response(gradient_boost.data)

        return Response(gradient_boost.errors)
    
class ADABoostClassifierEndpoint(APIView):
    def get(self, request):
        return Response("RandomForestClassifierEndpoint GET")

    def post(self, request):
        ada_boost = AdaBoostSerializer(data=request.data)

        if ada_boost.is_valid():
            ada_boost.save()
            return Response(ada_boost.data)

        return Response(ada_boost.errors)
