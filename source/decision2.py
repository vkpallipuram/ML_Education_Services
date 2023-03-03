import sys
import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import graphviz
import pickle

df = pandas.read_csv("training.csv")

# print(df)

d = {'Unsatisfactory': 0, 'Below Average': 1, 'Average': 2, 'Above Average': 3, 'Excellent': 4}
df['Rating'] = df['Rating'].map(d)

# print(df)

features = ['Knowledge','Enthusiasm','Organization','Preparedness','Ability to Explain Difficult Concepts',
            'Speaking Ability','Approachability','Encourages Discussion','Effectiveness of Examples']

X = df[features]
y = df['Rating']
print(X)
print(y)

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)

tree.plot_tree(dtree, feature_names=features)

class_n = ['Unsatisfactory', 'Below Average', 'Average', 'Above Average', 'Excellent']


dot_data = export_graphviz(dtree, out_file=None, feature_names=features, class_names=class_n, rounded=True, filled=True)
graph = graphviz.Source(dot_data)
graph.render("approach2") 

# saving the model in pickle file
# with open('decision_tree_approach2.pkl', 'wb') as f:
#     pickle.dump(dtree, f)
