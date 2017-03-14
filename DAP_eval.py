#!/usr/bin/env python
"""
Animals with Attributes Dataset, http://attributes.kyb.tuebingen.mpg.de
Perform Multiclass Predicition from binary attributes and evaluates it.
(C) 2009 Christoph Lampert <chl@tuebingen.mpg.de>
"""

import os,sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def nameonly(x):
    return x.split('\t')[1]

def loadstr(filename,converter=str):
    return [converter(c.strip()) for c in open(filename).readlines()]

def loaddict(filename,converter=str):
    D={}
    for line in open(filename).readlines():
        line = line.split()
        D[line[0]] = converter(line[1].strip())
    
    return D

# adapt these paths and filenames to match local installation

classnames = loadstr('classes.txt',nameonly)
numexamples = loaddict('numexamples.txt',int)

def evaluate(split,C):
    global test_classnames
    attributepattern = 'DAP/probabilities'
    
    if split == 0:
        test_classnames=loadstr('testclasses.txt')
        train_classnames=loadstr('trainclasses.txt')
    else:
        startid= (split-1)*10
        stopid = split*10
        test_classnames = classnames[startid:stopid]
        train_classnames = classnames[0:startid]+classnames[stopid:]
    
    test_classes = [ classnames.index(c) for c in test_classnames]
    train_classes = [ classnames.index(c) for c in train_classnames]

    M = np.loadtxt('predicate-matrix-binary.txt',dtype=float)

    L=[]
    for c in test_classes:
        L.extend( [c]*numexamples[classnames[c]] )

    L=np.array(L)  # (n,)

    P = np.loadtxt(attributepattern) # (85,n)

    prior = np.mean(M[train_classes],axis=0)
    prior[prior==0.]=0.5
    prior[prior==1.]=0.5    # disallow degenerated priors
    M = M[test_classes] # (10,85)

    prob=[]
    for p in P:
        prob.append( np.prod(M*p + (1-M)*(1-p),axis=1)/np.prod(M*prior+(1-M)*(1-prior), axis=1) )

    MCpred = np.argmax( prob, axis=1 )
    
    d = len(test_classes)
    confusion=np.zeros([d,d])
    for pl,nl in zip(MCpred,L):
        try:
            gt = test_classes.index(nl)
            confusion[gt,pl] += 1.
        except:
            pass

    for row in confusion:
        row /= sum(row)
    
    return confusion,np.asarray(prob),L


def plot_confusion(confusion):
    fig=plt.figure(figsize=(10,9))
    plt.imshow(confusion,interpolation='nearest',origin='upper')
    plt.clim(0,1)
    plt.xticks(np.arange(0,10),[c.replace('+',' ') for c in test_classnames],rotation='vertical',fontsize=24)
    plt.yticks(np.arange(0,10),[c.replace('+',' ') for c in test_classnames],fontsize=24)
    plt.axis([-.5,9.5,9.5,-.5])
    plt.setp(plt.gca().xaxis.get_major_ticks(), pad=18)
    plt.setp(plt.gca().yaxis.get_major_ticks(), pad=12)
    fig.subplots_adjust(left=0.30)
    fig.subplots_adjust(top=0.98)
    fig.subplots_adjust(right=0.98)
    fig.subplots_adjust(bottom=0.22)
    plt.gray()
    plt.colorbar(shrink=0.79)
    plt.savefig('results/AwA-ROC-confusion-DAP.pdf')
    return 

def plot_roc(P,GT):
    AUC=[]
    CURVE=[]
    for i,c in enumerate(test_classnames):
        class_id = classnames.index(c)
        fp, tp, _ = roc_curve(GT==class_id,  P[:,i])
        roc_auc = auc(fp, tp)
        print ("AUC: %s %5.3f" % (c,roc_auc))
        AUC.append(roc_auc)
        CURVE.append(np.array([fp,tp]))
    order = np.argsort(AUC)[::-1]
    styles=['-','-','-','-','-','-','-','--','--','--']
    plt.figure(figsize=(9,5))
    for i in order:
        c = test_classnames[i]
        plt.plot(CURVE[i][0],CURVE[i][1],label='%s (AUC: %3.2f)' % (c,AUC[i]),linewidth=3,linestyle=styles[i])
    
    plt.legend(loc='lower right')
    plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0], [r'$0$', r'$0.2$',r'$0.4$',r'$0.6$',r'$0.8$',r'$1.0$'],fontsize=18)
    plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0], [r'$0$', r'$0.2$',r'$0.4$',r'$0.6$',r'$0.8$',r'$1.0$'],fontsize=18)
    plt.xlabel('false negative rate',fontsize=18)
    plt.ylabel('true positive rate',fontsize=18)
    plt.savefig('results/AwA-ROC-DAP.pdf')

def main():
    try:
        split = int(sys.argv[1])
    except IndexError:
        split = 0

    try:
        C = float(sys.argv[2])
    except IndexError:
        C = 10.

    confusion,prob,L = evaluate(split,C)
    print ("Mean class accuracy %g" % np.mean(np.diag(confusion)*100))
    plot_confusion(confusion) 
    plot_roc(prob,L)
    
if __name__ == '__main__':
    main()
