import pandas as pd
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import *
import numpy as np

df = pd.read_csv('training90.csv')
dftest = pd.read_csv('testing45.csv')

# print(df.head())
#preparing the testing data
d = {'Unsatisfactory': 0, 'Below Average': 1, 'Average': 2, 'Above Average': 3, 'Excellent': 4}
df['Rating'] = df['Rating'].map(d)

features = ['Knowledge','Enthusiasm','Organization','Preparedness','Ability to Explain Difficult Concepts',
            'Speaking Ability','Approachability','Encourages Discussion','Effectiveness of Examples']

X = df[features]
y = df['Rating']
# print(X)
# print(y)

#preparing training data


dftest['Output'] = dftest['Output'].map(d)

features = ['Knowledge','Enthusiasm','Organization','Preparedness','Ability to Explain Difficult Concepts',
            'Speaking Ability','Approachability','Encourages Discussion','Effectiveness of Examples']

Xtest = dftest[features]
ytest = dftest['Output']

SVM_Model = SVC(gamma='auto')

SVM_Model.fit(X, y)

print(f'Accuracy - : {SVM_Model.score(Xtest, ytest)}')

#for calculating the scores
y_true = dftest['Output']
# print(y_true)
y_true_arr = y_true.values
# mask = ~np.isnan(y_true)
# y_true = y_true[mask]
y_pred = SVM_Model.predict(Xtest)

precision_mi = precision_score(y_true_arr, y_pred, average='micro') #here we can calculate with micro or macro
print("precision(micro): ", precision_mi)
precision_ma = precision_score(y_true_arr, y_pred, average='macro') #here we can calculate with micro or macro
print("precision(macro): ", precision_ma)

my_score = accuracy_score(y_true_arr, y_pred)
print("ACCURACY: ", my_score)

f1_macro = f1_score(y_true, y_pred, average='macro')
print("Macro-averaged F1 score:", f1_macro)

recall_macro = recall_score(y_true, y_pred, average='macro')
print("Macro-averaged recall:", recall_macro)

conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion matrix:")
print(conf_matrix)

display_labels = ['U', 'BA', 'A', 'AA', 'E']

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=display_labels)

disp.plot()
plt.show()