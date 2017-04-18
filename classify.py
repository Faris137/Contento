import scipy
import numpy as np
from sklearn import preprocessing, cross_validation, neighbors,datasets, svm
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.externals import joblib
import mahotas as mh
import numpy as np
from matplotlib import pyplot as plt
from glob import glob
import cv2
from mahotas.features import surf
from sklearn.cluster import KMeans


clf = joblib.load('SatisfactionDetector.pkl') 
#clf.predict(X)
km = joblib.load('SURFcluster.pkl') 

alldescriptors = []
image = "3.jpg"
im = mh.imread(image, as_grey=True)
im = im.astype(np.uint8)
alldescriptors.append(surf.dense(im, spacing=16))
# get all descriptors into a single array
concatenated = np.concatenate(alldescriptors)
print concatenated.shape
print('Number of SURF descriptors: {}'.format(len(concatenated)))
concatenated = concatenated[::62]
ifeatures = []
k = 256
c = km.predict(concatenated)
ifeatures.append(np.array([np.sum(c == ci) for ci in range(k)]))
ifeatures = np.array(ifeatures, dtype=float)
features = np.save("TestImage", ifeatures)
print "features are saved into TestImage"

print clf.predict(ifeatures)