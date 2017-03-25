import numpy as np
import pickle as cPickle
import bz2

def bzPickle(obj,filename):
    f = bz2.BZ2File(filename, 'wb')
    cPickle.dump(obj, f)
    f.close()
    
def bzUnpickle(filename):
    return cPickle.load(bz2.BZ2File(filename))

def get_full_animals_dict(path):
	animal_dict = {}
	with open(path) as f:
	    for line in f:
	        (key, val) = line.split()
	        animal_dict[val] = int(key)
	return animal_dict

def get_animal_index(path, filename):
	classes = []
	animal_dict = get_full_animals_dict(path + "classes.txt")
	with open(path+filename) as infile:
	    for line in infile:
	        classes.append(line[:-1])
	return [animal_dict[animal]-1 for animal in classes]

def get_attributes():
	attributes = []
	with open('attributes.txt') as infile:
	    for line in infile:
	        attributes.append(line[:-1])
	return attributes

def get_class_attributes(path, name='train'):
	animal_index = get_animal_index(path, name+'classes.txt')
	classAttributes = np.loadtxt(path + "predicate-matrix-binary.txt", comments="#", unpack=False)
	return classAttributes[animal_index]

def create_data(path, sample_index, attributes):
  
    X = bzUnpickle(path)
    
    nb_animal_samples = [item[1] for item in sample_index]
    for i,nb_samples in enumerate(nb_animal_samples):
        if i==0:
            y = np.array([attributes[i,:]]*nb_samples)
        else:
            y = np.concatenate((y,np.array([attributes[i,:]]*nb_samples)), axis=0)
    
    return X,y


def autolabel(rects, ax):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
    	if np.isnan(rect.get_height()):
    		continue
    	else:
    		height = rect.get_height()
    	ax.text(rect.get_x() + rect.get_width()/2., 1.05*height, '%0.3f' % height, 
    		ha='center', va='bottom', rotation=90)