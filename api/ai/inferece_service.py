import pandas as pd
import numpy as np 
import os
import joblib
import base64
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from PIL import Image
from io import BytesIO

relative_path = os.path.dirname(__file__)

class InferenceService:

    def __init__(self):
 
        train_images = pd.read_csv(os.path.join(relative_path, 'mnist_train.csv'))
        test_images = pd.read_csv(os.path.join(relative_path, 'mnist_test.csv'))

        self.x_train = np.array(train_images.iloc[:, 1:])
        self.x_test = np.array(test_images.iloc[:, 1:]) 
        self.y_train = np.array(train_images.iloc[:, 0])
        self.y_test = np.array(test_images.iloc[:, 0])
    
    def __calculate_accuracy(self) -> list[str]:
        return self.model.score(self.x_test, self.y_test)

    def get_random_forest_accuracy(self, max_depth: int, criterion: str, username: str) -> list[str]:
        self.model = RandomForestClassifier(
            max_depth=max_depth,
            criterion=criterion
        )

        self.model.fit(self.x_train[:2500], self.y_train[:2500])
        joblib.dump(self.model, os.path.join(relative_path, f'{username}-Random_Forest_model.pkl'))

        return self.__calculate_accuracy()
    
    def get_decision_tree_accuracy(self, max_depth: int, criterion: str, username: str) -> list[str]:
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            criterion=criterion
        )
        self.model.fit(self.x_train[:2500], self.y_train[:2500])
        joblib.dump(self.model, os.path.join(relative_path, f'{username}-Decision_Tree_model.pkl'))

        return self.__calculate_accuracy()
    
    def get_adaboost_accuracy(self, n_estimators: int, learning_rate: float, username: str) -> list[str]:
        self.model = AdaBoostClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate
        )
        self.model.fit(self.x_train[:2500], self.y_train[:2500])
        joblib.dump(self.model, os.path.join(relative_path, f'{username}-Ada_boost_model.pkl'))

        return self.__calculate_accuracy()
    
    def get_gradient_boosting_accuracy(self, n_estimators: int, learning_rate: float, loss: str, username: str) -> list[str]:
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            loss=loss
        )
        
        self.model.fit(self.x_train[:1000], self.y_train[:1000])
        joblib.dump(self.model, os.path.join(relative_path, f'{username}-Random_Forest_model.pkl'))

        return self.__calculate_accuracy()
    
    def retreive_model(self, username: str, algorithm: str, image_tensor):
        image = Image.open(BytesIO(base64.b64decode(image_tensor)))
        image = image.resize((28, 28))
        image = np.array(image)

        if image.shape[-1] == 3: 
            image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
        
        if image.shape[-1] == 4:
            image = np.dot(image[..., :4], [0.2989, 0.5870, 0.1140, 1])
        
        final_image_flat = image.flatten()
                
        algorithm = algorithm.replace(" ", "_")
        algorithm = joblib.load(os.path.join(relative_path, f'{username}-{algorithm}_model.pkl'))

        return algorithm.predict([final_image_flat])[0]