# standard library imports

# third party imports
import pickle
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

# local imports


class UFCModel:
    def __init__(self, model_type) -> None:
        if model_type == 'logistic':
            # TODO
            self.model = None
        elif model_type == 'randomforest':
            # TODO
            self.model = None
        elif model_type == 'lightgdb':
            # TODO
            self.model = None
        else:
            raise ValueError('Unsupported model type: {}'.format(model_type))

    def fit(self, x_train, y_train):
        """Fits model to some training data"""
        self.model.fit(x_train, y_train)

    def evaluate(self, training_path, testing_path):
        """Evaluates the model on a dataset"""
        # TODO
        

    def predict(self, x_test):
        """Predicts on model"""

        predictions = self.model.predict_proba(x_test)
        return predictions

    def save(self):
        """Save model to pkl"""

        file_name = '../models/' + str(self.model_type) + '.pkl'
        with open(file_name, 'wb') as file: 
            pickle.dump(self.model, file) 

    def load(self):
        """Load previous model from pkl"""

        file_name = '../models/' + str(self.model_type) + '.pkl'
        with open(file_name, 'wb') as file: 
            self.model = pickle.load(file)
