import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


class InferenceService:

    def __init__(self):
        self.__train_images = pd.read_csv('./mnist_train_data.csv')
        self.__test_images = pd.read_csv('./mnist_test_labels.csv')

        self.y_train = np.array(self.__train_images.iloc[:, 0])
        self.x_train = np.array(self.__train_images.iloc[:, 1:])
        self.y_test = np.array(self.__test_images.iloc[:, 0])
        self.x_test = np.array(self.__test_images.iloc[:, 1:])


    def predict(self, data: list[str]) -> list[str]:
        return self.model.predict(data)
    
    def predict_random_forest(self, data: list[str]) -> list[str]:
        self.model = RandomForestClassifier()
        self.model.fit(self.x_train, self.y_train)

        return self.model.predict(data)
    
    def predict_decision_tree(self, data: list[str]) -> list[str]:
        self.model = DecisionTreeClassifier()
        self.model.fit(self.x_train, self.y_train)

        return self.model.predict(data)
    
    def predict_adaboost(self, data: list[str]) -> list[str]:
        self.model = AdaBoostClassifier()
        self.model.fit(self.x_train, self.y_train)

        return self.model.predict(data)
    
    def predict_gradient_boosting(self, data: list[str]) -> list[str]:
        self.model = GradientBoostingClassifier()
        self.model.fit(self.x_train, self.y_train)

        return self.model.predict(data)