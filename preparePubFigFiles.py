import numpy as np
import csv
import pickle as cPickle
import bz2


def bzPickle(obj,filename):
    f = bz2.BZ2File(filename, 'wb')
    cPickle.dump(obj, f)
    f.close()

def writeListToTxt (wlist, filename):
    with open(filename, 'w') as wfile:
        for item in wlist:
            wfile.write("%s\n" % item)

def preparePubFigFiles():
	path = 'Datasets/PubFig/'
	d = 73

	# Get number of items
	n = 0
	with open(path + 'pubfig_attributes.txt') as infile:
	    for i,line in enumerate(infile):
	        n+=1
	n = n-2

	# Obtain attributes names, celibrities and attributes as lsit
	attributes = []
	celebrities = []
	with open(path +'pubfig_attributes.txt') as infile:
	    for i,line in enumerate(infile):
	        if i==0:
	            continue
	        elif i==1:
	            attribute_names = line.strip().split("\t")[3:]
	        else:
	            stripped_line = line.strip().split("\t")
	            person = stripped_line[0]
	            if person not in celebrities:
	                celebrities.append(person)
	            line_attribute = [float(x) for x in stripped_line[2:]]
	            attributes.append(line_attribute)

	# Convert attributes list to array
	attributes_array = np.zeros((n,d))
	for i in range(n):
	    attributes_array[i,:] = np.array(attributes[i])

	# Save files 
	writeListToTxt (attribute_names, path + 'attribute_names.txt')
	writeListToTxt (celebrities, path + 'celebrities.txt')
	bzPickle(attributes_array, path + 'attributes')

if __name__ == '__main__':
    preparePubFigFiles()