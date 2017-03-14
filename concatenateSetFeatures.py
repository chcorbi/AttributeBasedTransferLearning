"""
Concatenate selected features together
"""

import os
import numpy as np
import pickle as cPickle
import bz2
from scipy.sparse import csr_matrix


def bzPickle(obj,filename):
    f = bz2.BZ2File(filename, 'wb')
    cPickle.dump(obj, f)
    f.close()

def bzUnpickle(filename):
    return cPickle.load(bz2.BZ2File(filename))

def loadstr(filename,converter=str):
    list_object = []
    with open(filename) as infile:
        for line in infile:
            list_object.append(converter(line[:-1]))
    return list_object

def concatenate_set_features(set_classes, nameset='train'):
    """
    Concatenate all selected animals features together
    """
    index= []

    for i,animal in enumerate(set_classes):
        print ("Adding %s..." % animal)
        features_file = "feat/featuresVGG19_" + animal + ".pic.bz2"
        features = bzUnpickle(features_file).T
        if i==0:
            X = features
        else:
            X = np.concatenate((X,features),axis=0)
        index.append((animal,features.shape[0]))
    X = csr_matrix(X)
    
    try:
        os.stat('CreatedData/')
    except:
        os.mkdir('CreatedData/')

    picklefile = 'CreatedData/' + nameset + '_featuresVGG19.pic.bz2'
    bzPickle(X, picklefile)
    bzPickle(index, 'CreatedData/' nameset + '_features_index.txt') 


if __name__ == '__main__':
    # Training classes
    print ('#### Concatenating training data....')
    trainclasses = loadstr('trainclasses.txt')
    concatenate_set_features(trainclasses, 'train')

    # Test classes
    print ('#### Concatenating test data....')
    testclasses = loadstr('testclasses.txt')
    concatenate_set_features(testclasses, 'test')