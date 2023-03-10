import sys
import matplotlib
#matplotlib.use('Agg')
matplotlib.rc('font',size=8)
import pandas
from sklearn.naive_bayes import GaussianNB
import numpy as np
import seaborn as sns
import matplotlib.pylab as pl
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score 
df = pandas.read_csv("training.csv")
tests=pandas.read_csv("testing.csv")

d={'Unsatisfactory':0, 'Below Average':1, 'Average':2,'Above Average':3,'Excellent':4}
 
df['Rating'] = df['Rating'].map(d)
tests['Output']=tests['Output'].map(d)

features=['Knowledge','Enthusiasm','Organization','Preparedness','Ability to Explain Difficult Concepts','Speaking Ability','Approachability','Approachability','Effectiveness of Examples']

X = df[features]
y = df['Rating']

X_test=tests[features]
y_test=tests['Output']

print(X_test)
print(y_test)

gnb = GaussianNB().fit(X,y)

print("Gaussian Naive Bayes")
print(gnb)




gnb_predict=gnb.predict(X_test)

ypred=np.array(gnb_predict)
ytest=np.array(y_test)

print('Recall: %.3f' % recall_score(ytest, ypred,average="weighted"))
print('Precision: %.3f' % precision_score(ytest, ypred,average="weighted"))
print('F1-score: %.3f' % f1_score(ytest, ypred,average="weighted"))
print('Accuracy: %.3f' % accuracy_score(ytest, ypred))

print("Model probabilities")
classprob = (gnb.predict_proba(X_test))

x_axis_labels=['U','BA','A','AA','E']
y_axis_labels=['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S12','S13','S14','S15','S16','S17','S18']
ax = sns.heatmap(classprob,linewidth=0.5,xticklabels=x_axis_labels,yticklabels=y_axis_labels,annot=True)
pl.show()

#cm = confusion_matrix(y_test,gnb_predict)
#fig=plt.figure()
#cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels =['U','BA','A','AA'])


#print(cm)
#cm_display.plot()
#plt.show()
#plt.colorbar()
#plt.ylabel('True Label')
#plt.xlabel('Predicted Label')
#plt.savefig('confusion_matrixNB.pdf')

#tree.plot_tree(dtree, feature_names=features)
#plt.savefig('output.png',dpi=400)
#sys.stdout.flush()
