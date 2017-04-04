import sys
import numpy as np
from time import time
from utils import bzPickle, bzUnpickle, get_class_attributes, create_data
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from SVMClassifier import SVMClassifier
from SVMRegressor import SVMRegressor
from NeuralNetworkClassifier import NeuralNetworkClassifier
from NeuralNetworkRegressor import NeuralNetworkRegressor


def DirectAttributePrediction(classifier='SVM', predicate_type='binary', C=10.0):
	# Get features index to recover samples
	train_index = bzUnpickle('./CreatedData/train_features_index.txt')
	test_index = bzUnpickle('./CreatedData/test_features_index.txt')

	# Get classes-attributes relationship
	train_attributes = get_class_attributes('./', name='train', predicate_type=predicate_type)
	test_attributes = get_class_attributes('./', name='test', predicate_type=predicate_type)
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
			if predicate_type == 'binary':
				clf = SVMClassifier()
			else:
				clf = SVMRegressor()

			# Training
			clf.fit(X_train, y_train[:,i])
			print ('Fitted classifier in: %fs' % (time() - t0))
			if predicate_type == 'binary':
				clf.set_platt_params(Xplat_val, yplat_val[:,i])

			# Predicting
			print ('Predicting for attribute %d...' % (i+1))
			y_pred[:,i] = clf.predict(X_test)
			if predicate_type == 'binary':
				y_proba[:,i] = clf.predict_proba(X_test)

			print ('Saving files...')
			np.savetxt('./DAP_'+predicate_type+'/prediction_SVM', y_pred)
			if predicate_type == 'binary':
				np.savetxt('./DAP_'+predicate_type+'/platt_params_SVM', platt_params)
				np.savetxt('./DAP_'+predicate_type+'/probabilities_SVM', y_proba)
	

	# CHOOSING NEURAL NETWORK
	if classifier == 'NN':
		if predicate_type != 'binary':
		    clf = NeuralNetworkRegressor(dim_features=X_train.shape[1], nb_attributes=N_ATTRIBUTES)
		else:
		    clf = NeuralNetworkClassifier(dim_features=X_train.shape[1], nb_attributes=N_ATTRIBUTES)

		print ('Fitting Neural Network...')
		clf.fit(X_train, y_train)

		print ('Predicting attributes...')
		y_pred = np.array(clf.predict(X_test))
		y_pred = y_pred.reshape((y_pred.shape[0], y_pred.shape[1])).T
		y_proba = y_pred
    
		print ('Saving files...')
		np.savetxt('./DAP_'+predicate_type+'/prediction_NN', y_pred)
		if predicate_type == 'binary':
		    np.savetxt('./DAP_'+predicate_type+'/probabilities_NN', y_proba)


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
	    p_type = str(sys.argv[3])
	except IndexError:
	    p_type = 'binary'
	try:
	    split = int(sys.argv[4])
	except IndexError:
	    split = 0
	try:
	    C = float(sys.argv[5])
	except IndexError:
	    C = 10.

	if clf not in list_clf:
	    print ("Non valid choice of classifier (SVM, NN)")
	    raise SystemExit

	if method == 'DAP':
		DirectAttributePrediction(classifier=clf, predicate_type=p_type, C=C)
	else:
	    print ("Non valid choice of method (DAP, IAP)")
	    raise SystemExit  		

	print ("Done.")


if __name__ == '__main__':
	main()

