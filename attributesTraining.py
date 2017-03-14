import numpy as np
from time import time
from utils import bzPickle, bzUnpickle, get_class_attributes, create_data
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.kernel_approximation import SkewedChi2Sampler
from sklearn.model_selection import train_test_split
from platt import SigmoidTrain, SigmoidPredict

if __name__ == '__main__':
	# Get features index to recover samples
	train_index = bzUnpickle('./CreatedData/train_features_index.txt')
	test_index = bzUnpickle('./CreatedData/test_features_index.txt')

	# Get classes-attributes relationship
	train_attributes = get_class_attributes('./', name='train')
	test_attributes = get_class_attributes('./', name='test')
	N_ATTRIBUTES = train_attributes.shape[1]

	# Create training Dataset
	print ('Creating training dataset...')
	X_train, y_train = create_data('./CreatedData/train_featuresVGG19.pic.bz2',train_index, train_attributes)
	
	print ('X_train to dense...')
	X_train = X_train.toarray()

	Xplat_train, Xplat_val, yplat_train, yplat_val = train_test_split(
		X_train, y_train, test_size=0.10, random_state=42)

	print ('Creating test dataset...')
	X_test, y_test = create_data('./CreatedData/test_featuresVGG19.pic.bz2',test_index, test_attributes)
	y_pred = np.zeros(y_test.shape)
	y_proba = np.copy(y_pred)

	print ('X_test to dense...')
	X_test = X_test.toarray()

	# Training svm
	platt_params = []
	for i in range(N_ATTRIBUTES):
		print ('--------- Attribute %d/%d ---------' % (i+1,N_ATTRIBUTES))
		t0 = time()
		feature_map_fourier = SkewedChi2Sampler(skewedness=3.,  n_components=100)
		clf = Pipeline([
			("feature_map", feature_map_fourier),
			("svm",LinearSVC(C=10.)) ])

		clf.fit(Xplat_train, yplat_train[:,i])
		print ('Fitted classifier in: %fs' % (time() - t0))

		yplat_pred = clf.predict(Xplat_val)
		pt_param = SigmoidTrain(yplat_pred, yplat_val[:,i])
		platt_params.append(pt_param)

		print ('Predicting for attribute %d...' % (i+1))
		y_pred[:,i] = clf.predict(X_test)
		y_proba[:,i] = SigmoidPredict(y_pred[:,i], pt_param)
       
	print ('Saving files...')
	np.savetxt('./DAP/platt_params', platt_params)
	np.savetxt('./DAP/prediction', y_pred)
	np.savetxt('./DAP/probabilities', y_proba)
