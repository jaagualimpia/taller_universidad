import pandas as pd
import numpy as np 
import os
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


class InferenceService:

    def __init__(self):
        relative_path = os.path.dirname(__file__)
 
        train_images = pd.read_csv(os.path.join(relative_path, 'mnist_train.csv'))
        test_images = pd.read_csv(os.path.join(relative_path, 'mnist_test.csv'))

        self.x_train = np.array(train_images.iloc[:, 1:])
        self.x_test = np.array(test_images.iloc[:, 1:]) 
        self.y_train = np.array(train_images.iloc[:, 0])
        self.y_test = np.array(test_images.iloc[:, 0])
    
    def __calculate_accuracy(self) -> list[str]:
        return self.model.score(self.x_test, self.y_test)

    def get_random_forest_accuracy(self, max_depth: int, criterion: str) -> list[str]:
        self.model = RandomForestClassifier(
            max_depth=max_depth,
            criterion=criterion
        )
        
        self.model.fit(self.x_train[:2500], self.y_train[:2500])

        return self.__calculate_accuracy()
    
    def get_decision_tree_accuracy(self, max_depth: int, criterion: str) -> list[str]:
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            criterion=criterion
        )
        self.model.fit(self.x_train[:2500], self.y_train[:2500])

        return self.__calculate_accuracy()
    
    def get_adaboost_accuracy(self, n_stimtators: int, learning_rate: float) -> list[str]:
        self.model = AdaBoostClassifier(
            n_estimators=n_stimtators,
            learning_rate=learning_rate
        )
        self.model.fit(self.x_train[:2500], self.y_train[:2500])

        return self.__calculate_accuracy()
    
    def get_gradient_boosting_accuracy(self, n_estimators: int, learning_rate: float, loss: str) -> list[str]:
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            loss=loss
        )
        
        self.model.fit(self.x_train[:1000], self.y_train[:1000])

        return self.__calculate_accuracy()