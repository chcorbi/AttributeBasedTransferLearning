import sys
import numpy as np
from time import time
from utils import bzPickle, bzUnpickle, get_class_attributes, create_data, get_full_animals_dict
from sklearn.model_selection import train_test_split
from SVMClassifier import SVMClassifierIAP
from NeuralNetworkClassifier import NeuralNetworkClassifier

def nameonly(x):
    return x.split('\t')[1]

def loadstr(filename,converter=str):
    return [converter(c.strip()) for c in open(filename).readlines()]

animal_dict = get_full_animals_dict('./classes.txt')

def indirectAttributePrediction(classifier='SVM'):
    # Get features index to recover samples
    train_index = bzUnpickle('./CreatedData/train_features_index.txt')
    test_index = bzUnpickle('./CreatedData/test_features_index.txt')

    # Get classes-attributes relationship
    train_attributes = get_class_attributes('./', name='train')
    test_attributes = get_class_attributes('./', name='test')

    # Create training Dataset
    print ('Creating training dataset...')
    X_train, a_train = create_data('./CreatedData/train_featuresVGG19.pic.bz2',train_index, train_attributes)
    y_train = []
    for (animal, num) in train_index:
        y_train += num*[animal_dict[animal]]
    y_train = np.array(y_train)

    print ('X_train to dense...')
    X_train = X_train.toarray()

    print ('Creating test dataset...')
    X_test, a_test = create_data('./CreatedData/test_featuresVGG19.pic.bz2',test_index, test_attributes)

    print ('X_test to dense...')
    X_test = X_test.toarray()
    
    clf = SVMClassifierIAP(n_components=100, C=1.0)

    print('Training model... (takes around 10 min)')
    t0 = time()
    clf.fit(X_train, y_train)
    print('Training finished in', time() - t0)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    print ('Saving files...')
    np.savetxt('./IAP/prediction_SVM', y_pred)
    np.savetxt('./IAP/probabilities_SVM', y_proba)

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

	if method == 'IAP':
		indirectAttributePrediction(classifier=clf)
	else:
	    print ("Non valid choice of method (DAP, IAP)")
	    raise SystemExit

	print ("Done.", C, split)


if __name__ == '__main__':
	main()
