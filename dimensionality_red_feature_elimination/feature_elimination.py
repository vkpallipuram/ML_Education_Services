from sklearn.datasets import make_classification
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv('../dataset/training90.csv')
X = df.drop(["Rating"],axis=1)
y = df["Rating"]

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0)

rfe = RFE(estimator=GradientBoostingClassifier(), n_features_to_select=6)

model = GradientBoostingClassifier()

pipe = Pipeline([('Feature Selection', rfe), ('Model', model)])
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=36851234)
n_scores = cross_val_score(pipe, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
np.mean(n_scores)

pipe.fit(X_train, y_train)

# print(rfe.support_)


result = pd.DataFrame(rfe.support_,index=X.columns,columns=["Rank"])
print(result)

rf_df = pd.DataFrame(rfe.ranking_,index=X.columns,columns=['Rank']).sort_values(by='Rank',ascending=True)
print(rf_df.head())
