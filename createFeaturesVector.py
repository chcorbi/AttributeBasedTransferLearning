"""
Compute Feature Vector with all features extracted from a VGG19
"""

import argparse
import os, os.path
import numpy as np
import csv
import pickle as cPickle
import bz2

def bzPickle(obj,filename):
    f = bz2.BZ2File(filename, 'wb')
    cPickle.dump(obj, f)
    f.close()

def createFeaturesVector(path,feat_shape=4096):
    """
    Function to compute Feature Vector with all features extracted from a VGG19
    """
    feat_shape = 4096
    subfolders = [x[0] for x in os.walk(path)][1:]

    for i,subfolder in enumerate(subfolders):
        animal = subfolder.split('/')[-1:][0]
        nb_files = len([name for name in os.listdir(subfolder) if os.path.isfile(os.path.join(subfolder, name))])

        for j in range(1,nb_files+1):
            img_feat_path = subfolder + '/' + animal + '_{:04d}'.format(j) + '.txt'
            if j==1:
                features = np.loadtxt(img_feat_path, comments="#", delimiter=",", unpack=False)
                features = features.reshape(((feat_shape,1)))
            else:
                tmp_features = np.loadtxt(img_feat_path, comments="#", delimiter=",", unpack=False)
                tmp_features = tmp_features.reshape(((feat_shape,1)))
                features = np.concatenate((features,tmp_features), axis=1)

        try:
            os.stat('feat/')
        except:
            os.mkdir('feat/')

        picklefile = 'feat/featuresVGG19_' + animal + '.pic.bz2'
        print ("Pickling ",animal, " features to ",picklefile)
        bzPickle(features, picklefile)            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="One main folder path")
    args = parser.parse_args()

    createFeaturesVector(args.path)