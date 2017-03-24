import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.kernel_approximation import SkewedChi2Sampler
from sklearn.base import BaseEstimator
from platt import SigmoidTrain, SigmoidPredict

class SVMClassifier(BaseEstimator):                                                  
    def __init__(self, skewedness=3., n_components=85, C=100.):
    	self.platt_params = []
    	self.feature_map_fourier = SkewedChi2Sampler(skewedness=skewedness,	n_components=n_components)
    	self.clf = Pipeline([("feature_map", self.feature_map_fourier),
			("svm",LinearSVC(C=C))])	                                                                                                                                    
    
    def fit(self, X, y):
    	self.clf.fit(X, y)

    def set_platt_params(self, X, y):
    	y_pred = self.clf.predict(X)
    	self.platt_params = SigmoidTrain(y_pred, y)

    def predict(self, X):
    	return self.clf.predict(X)

    def predict_proba(self, X):
    	y_pred = self.clf.predict(X)
    	return SigmoidPredict(y_pred, self.platt_params)