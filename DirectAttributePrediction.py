import sys
import numpy as np
from time import time
from utils import bzPickle, bzUnpickle, get_class_attributes, create_data
from sklearn.model_selection import train_test_split
from SVMClassifier import SVMClassifier
from NeuralNetworkClassifier import NeuralNetworkClassifier2


def DirectAttributePrediction(classifier='SVM',):
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

	# CHOOSING SVM
	if classifier == 'SVM':
		platt_params = []
		for i in range(N_ATTRIBUTES):
			print ('--------- Attribute %d/%d ---------' % (i+1,N_ATTRIBUTES))
			t0 = time()

			# SVM classifier
			clf = SVMClassifier()

			# Training
			clf.fit(Xplat_train, yplat_train[:,i])
			print ('Fitted classifier in: %fs' % (time() - t0))
			clf.set_platt_params(Xplat_val, yplat_val[:,i])

			# Predicting
			print ('Predicting for attribute %d...' % (i+1))
			y_pred[:,i] = clf.predict(X_test)
			y_proba[:,i] = clf.predict_proba(X_test)

			print ('Saving files...')
			np.savetxt('./DAP/platt_params_SVM', platt_params)
			np.savetxt('./DAP/prediction_SVM', y_pred)
			np.savetxt('./DAP/probabilities_SVM', y_proba)
	

	# CHOOSING NEURAL NETWORK
	if classifier == 'NN':
		clf = NeuralNetworkClassifier2(dim_features=X_train.shape[1], nb_attributes=N_ATTRIBUTES)

		print ('Fitting Neural Network...')
		clf.fit(X_train, y_train)

		print ('Predicting attributes...')
		y_pred = np.array(clf.predict(X_test))
		y_pred = y_pred.reshape((y_pred.shape[0], y_pred.shape[1])).T
		y_proba = y_pred
    
		print ('Saving files...')
		np.savetxt('./DAP/prediction_NN', y_pred)
		np.savetxt('./DAP/probabilities_NN', y_proba)


def main():
	list_clf = ['SVM', 'NN']

	try:
		method = str(sys.argv[1])
	except IndexError:
		print ("Must specify attribute method!")
		raise SystemExit
	try:
	    clf = str(sys.argv[2])
	except IndexError:
	    clf = 'SVM'
	try:
	    split = int(sys.argv[3])
	except IndexError:
	    split = 0
	try:
	    C = float(sys.argv[4])
	except IndexError:
	    C = 10.

	if clf not in list_clf:
	    print ("Non valid choice of classifier (SVM, NN)")
	    raise SystemExit

	if method == 'DAP':
		DirectAttributePrediction(classifier=clf)
	else:
	    print ("Non valid choice of method (DAP, IAP)")
	    raise SystemExit  		

	print ("Done.", C, split)


if __name__ == '__main__':
	main()

