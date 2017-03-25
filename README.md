# Attribute Based Learning implementation

Authors : Charles Corbière, Bernardo Cardoso Cordeiro, Aymeric Zhuo, Luciano Di Palma

## Synopsis

This package implement differents attribute-based learning approach:
- Direct Attributed Prediction (iAP)
- Indirect Attributed Prediction (IAP)

For this matter, we choose to try two learner :
- SVM
- Neural Network


Dataset included:
- Animals with Attributed dataset
- PubFig


## For AwA dataset, how to use it

0. Download Animal VGG19 features [here](http://www.ist.ac.at/~chl/AwA/AwA-features-vgg19.tar.bz2) and decompress it.

1. Compute features into animal categories in ./feat folder
```
python createFeaturesVector.py pathToFeatures
```
2. Concatenate features into a training dataset and a test dataset with their respectives labels in ./CreatedDataset
```
python concatenateSetFeatures.py
```
3. Train and predict a model given a method and a classifier. Platt parameters, prediction and probabilities saved in ./DAP
```
python DirectAttributePrediction.py DAP SVM
```
4. Generate and save confusion matrix plot and roc in ./results given a classifier
```
python DAP_eval.py SVM
```


## For Pubfig dataset, how to use it

0. Download Animal VGG19 features [here](http://vision.seas.harvard.edu/pubfig83/pubfig83.v1.tgz) and decompress it on Datasets/PubFig.

1. Compute attributes_names, celebrities and attributes as an array
```
python preparePubFigFiles.py
```

