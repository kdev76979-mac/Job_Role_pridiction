import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key=None):
        self.key = key
        
    def __setstate__(self, state):
        self.__dict__.update(state)
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        return X[self.key]

def inspect_model():
    try:
        pipeline = joblib.load('career_pipeline_v3.pkl')
        features_step = pipeline.named_steps['features']
        for name, transformer in features_step.transformer_list:
            step = transformer.named_steps['select']
            print(f"{name} dict:", step.__dict__)
            
    except Exception as e:
        print("Error loading:", e)

inspect_model()
