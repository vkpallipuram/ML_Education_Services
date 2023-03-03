import pickle
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import *

with open('decision_tree_approach2.pkl', 'rb') as f:
    dt = pickle.load(f)

test = pd.read_csv("testing.csv")
# input = test.drop(["Attribute", "Average"], axis='columns')

d = {'Unsatisfactory': 0, 'Below Average': 1, 'Average': 2, 'Above Average': 3, 'Excellent': 4}

y_true = test['What we should get']
y_true = y_true.map(d)
y_true_arr = y_true.values
# print(y_true)

#=IF(B2>4.8, "Excellent", IF(B2>4.5, "Above Average", IF(B2>4.3, "Average", IF(B2>4, "Below Average", "Poor"))))

# inputs = test.drop("Label", axis='columns')
# target = test.drop(["Attribute", "Average"], axis='columns')

# print(target)
#encoding the given elements

# le_attribute = LabelEncoder()
# le_label = LabelEncoder()

# inputs['attribute_n'] = le_attribute.fit_transform(inputs['Attribute'])
# inputs_n = inputs.drop(["Attribute"], axis='columns')
# target['label_n'] = le_label.fit_transform(target['Label'])

# print(inputs_n)

# target_n = target.drop(["Label"], axis='columns')
# print(target_n)

# print(inputs)
# my_score = dt.score(inputs_n, target_n)
# print("ACCURACY: ", my_score)
test_input = test.drop(['empty', 'What we should get', 'Numerical Rating not for modeling'], axis='columns')
# print(test_input)
y_pred = dt.predict(test_input)
# print("prediction", y_pred) 
# print("actual val", y_true_arr) 

#-----------------------CALCULATING PRECISION, ACCURACY--------------------------
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
#print(y_pred) 
# # print("size of predicted labels: ", len(y_pred))

# original_labels = le_label.classes_
# # print the original labels
# # print(original_labels)


# above_avg = 0
# avg = 0
# below_avg = 0
# excellent = 0
# poor = 0

#----------------------------GIVE THE PREDICTED VALUES IN WORDS-----------------------------------------------
# for i in y_pred:
#     if(i == 0):
#         print("Unsatisfactory")
#     elif(i == 1):
#         print("Below Average")
#     elif(i == 2):
#         print("Average")
#     elif(i == 3):
#         print("Above Average")
#     elif(i == 4):
#         print("Excellent")
    


# labels = ['Above Average', 'Average', 'Below Average', 'Excellent', 'Poor']
# sizes = [above_avg, avg, below_avg, excellent, poor]

# # Create a pie chart
# fig1, ax1 = plt.subplots()
# ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)

# # Add a title
# ax1.set_title('Performance Results', fontweight='bold')

# ax1.text(0, -1.2, f'Accuracy: {my_score}', ha='center', fontsize=12, fontweight='bold', color='red')
# # Show the plot
# plt.show()

