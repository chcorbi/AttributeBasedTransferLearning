import numpy as np
import csv
import pickle as cPickle
import bz2
import random
SEED = 448


def bzPickle(obj,filename):
    f = bz2.BZ2File(filename, 'wb')
    cPickle.dump(obj, f)
    f.close()

def bzUnpickle(filename):
    return cPickle.load(bz2.BZ2File(filename))

def writeListToTxt (wlist, filename):
    with open(filename, 'w') as wfile:
        for item in wlist:
            wfile.write("%s\n" % item)

def readTxtToList(filename):
    rlist=[]
    with open(filename, 'r') as rfile:
        for line in rfile:
            rlist.append(line[:-1])
    return rlist


def preparePubFigFiles(test_size = 0.20):
	path = 'Datasets/PubFig/'
	d = 73

	# Get number of items, attribute_name and celebrities
	n = 0
	celebrities = []
	with open(path + 'pubfig_attributes.txt') as infile:
	    for i,line in enumerate(infile):
	        n+=1
	        if i==0:
	            continue
	        elif i==1:
	            attribute_names = line.strip().split("\t")[3:]
	        else:
	            stripped_line = line.strip().split("\t")
	            person = stripped_line[0]
	            if person not in celebrities:
	                celebrities.append(person)
	        n = n-2
	writeListToTxt (attribute_names, path + 'attribute_names.txt')
	writeListToTxt (celebrities, path + 'celebrities.txt')

	# Split train/test classes
	train_celebrities = celebrities[int(test_size*len(celebrities)):]
	test_celebrities = celebrities[-int(test_size*len(celebrities)):]
	writeListToTxt (train_celebrities, path + 'train_celebrities.txt')
	writeListToTxt (test_celebrities, path + 'test_celebrities.txt')

	# Obtain attributes as list
	train_attributes = []
	test_attributes = []
	with open(path + 'pubfig_attributes.txt') as infile:
	    for i,line in enumerate(infile):
	        if i==0 or i==1:
	            continue
	        elif i==1:
	            attribute_names = line.strip().split("\t")[3:]
	        else:
	            stripped_line = line.strip().split("\t")
	            person = stripped_line[0]
	            line_attribute = [float(x) for x in stripped_line[2:]]
	            if person in train_celebrities:
	                train_attributes.append(line_attribute)
	            else:
	                test_attributes.append(line_attribute)

	# Convert attributes list to array
	train_attributes_array = np.zeros((len(train_attributes),d))
	test_attributes_array = np.zeros((len(test_attributes),d))

	for i in range(len(train_attributes)):
	    train_attributes_array[i,:] = np.array(train_attributes[i])
	    
	for i in range(len(test_attributes)):
	    test_attributes_array[i,:] = np.array(test_attributes[i])

	# Save files 
	bzPickle(train_attributes_array, path + 'train_attributes')
	bzPickle(test_attributes_array, path + 'test_attributes') 

if __name__ == '__main__':
    preparePubFigFiles()