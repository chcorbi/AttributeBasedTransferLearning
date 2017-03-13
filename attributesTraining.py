import numpy as np
from time import time
from utils import bzPickle, bzUnpickle, get_class_attributes, create_data
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import chi2_kernel

if __name__ == '__main__':
	# Get features index to recover samples
	train_index = bzUnpickle('./CreatedData/train_features_index.txt')

	# Get classes-attributes relationship
	train_attributes = get_class_attributes('./')
	N_ATTRIBUTES = train_attributes.shape[1]

	# Create Dataset
	print ('Creating dataset...')
	X_train, y_train = create_data('./CreatedData/train_featuresVGG19.pic.bz2',train_index, train_attributes)

	# Training svm
	SVMs = []
	for i in range(N_ATTRIBUTES):
		t0 = time()
		clf = Pipeline([
			('scaler',StandardScaler(with_mean=False)),
			('classify',SVC(C=10., kernel='rbf', max_iter=100))])
		clf.fit(X_train, y_train[:,i])
		SVMs.append(clf)
		print ('Fitting classifier for attribute %d/%d computed in: %fs' % (i+1,N_ATTRIBUTES, (time() - t0)))

	# Saving weights
	bzPickle(SVMs,'./CreatedData/SVM_weights')