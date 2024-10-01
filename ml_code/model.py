from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

class ModelFactory:
    
    @staticmethod
    def get_model(model_type):
        if model_type == 'logistic_regression':
            return LogisticRegression(max_iter=500)
        if model_type == 'decision_tree':
            return DecisionTreeClassifier()
        if model_type == 'random_forest':
            return RandomForestClassifier()
        else:
            raise ValueError(f"Unknown model type: {model_type}")