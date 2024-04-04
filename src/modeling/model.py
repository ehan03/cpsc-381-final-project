# standard library imports

# third party imports
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

# local imports


class UFCModel:
    def __init__(self, model_type) -> None:
        if model_type == 'Logistic':
            # Todo
            self.model = None
        elif model_type == 'RandomForest':
            self.model = RandomForestClassifier(n_estimators=200, criterion="gini")
        elif model_type == 'LightGDB':
            # Todo
            self.model = None
        else:
            raise ValueError('Unsupported model type: {}'.format(model_type))

    def fit(self, x_train, y_train):
        """Fits model to some training data"""
        self.model.fit(x_train, y_train)

    def evaluate(self, training_path, testing_path):
        """Evaluates the model on a dataset"""
        

    def predict(self, x_test):
        predictions = self.model.predict_proba(x_test)
        return predictions

    def save(self):
        pass

    def load(self):
        pass
