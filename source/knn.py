import sys
import matplotlib
#matplotlib.use('Agg')
matplotlib.rc('font',size=8)
import pandas
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import seaborn as sns
import matplotlib.pylab as pl
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score 
df = pandas.read_csv("/home/dirichlet/ML_Education/ML_Education_Services/data/training.csv")
tests=pandas.read_csv("/home/dirichlet/ML_Education/ML_Education_Services/data/testing.csv")

d={'Unsatisfactory':0, 'Below Average':1, 'Average':2,'Above Average':3,'Excellent':4}
 
df['Rating'] = df['Rating'].map(d)
tests['Output']=tests['What we should get'].map(d)

features=['Knowledge','Enthusiasm','Organization','Preparedness','Ability to Explain Difficult Concepts','Speaking Ability','Approachability','Approachability','Effectiveness of Examples']

X = df[features]
y = df['Rating']

X_test=tests[features]
y_test=tests['Output']

knn = KNeighborsClassifier(n_neighbors = 5).fit(X, y)

accuracy = knn.score(X_test, y_test)
print(accuracy)
  
knn_predictions = knn.predict(X_test) 
labels=['U','BA','A','AA','E']
cm = confusion_matrix(y_test, knn_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()
plt.show()
