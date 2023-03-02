#Neural Network Classifier
import math
import sys
import matplotlib
#matplotlib.use('Agg')
matplotlib.rc('font',size=16)
import pandas
from sklearn import tree
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

df = pandas.read_csv("training90.csv")
tests=pandas.read_csv("testing45.csv")

d={'Unsatisfactory':0, 'Below Average':1, 'Average':2,'Above Average':3,'Excellent':4}
 
df['Rating'] = df['Rating'].map(d)
tests['Output']=tests['Output'].map(d)

features=['Knowledge','Enthusiasm','Organization','Preparedness','Ability to Explain Difficult Concepts','Speaking Ability','Approachability','Approachability','Effectiveness of Examples']

X = df[features]
y = df['Rating']

#Standardize the training
scaler=StandardScaler()

scaler.fit(X)
Xscale=scaler.transform(X)

X_test=tests[features]
y_test=tests['Output']


scaler.fit(X_test)
Xtestscale=scaler.transform(X_test)

#print(X)
print(y_test)

N=len(features)
m = 5 # number of output neurons

#Use IJRS paper. See ML Literature folder
hidden1 = int(math.sqrt((m+2)*N) + 2*math.sqrt(N/(m+2)))
hidden2=int(m*math.sqrt(N/(m+2)))
print("Number of features: %s" % len(features))


print("Neurons in hidden layer 1: %s" % hidden1)
print("Neurons in hidden layer 2: %s" % hidden2)

clf = MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(9,9,9),random_state=1)

clf.fit(Xscale,y)
clfpredict = clf.predict(Xtestscale)
print(clfpredict)


#print(dtree_predict)

ypred=np.array(clfpredict)
ytest=np.array(y_test)

#plt.plot(ypred,marker='d')
#plt.show()

print('Recall: %.3f' % recall_score(ytest, ypred,average="weighted"))
print('Precision: %.3f' % precision_score(ytest, ypred,average="weighted"))
print('F1-score: %.3f' % f1_score(ytest, ypred,average="weighted"))
print('Accuracy: %.3f' % accuracy_score(ytest, ypred))


cm = confusion_matrix(y_test,clfpredict)
fig=plt.figure()
cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels =['U','BA','A','AA','E'])

print(cm)
cm_display.plot()
plt.show()
plt.colorbar()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
#plt.savefig('confusion_matrix90.jpg')

#tree.plot_tree(dtree, feature_names=features)
#plt.savefig('output.png',dpi=400)
#sys.stdout.flush()
