import pandas as pd
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import *
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
import math
import sys
import matplotlib
#matplotlib.use('Agg')
import pandas
from sklearn import tree
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
matplotlib.rc('font',size=16)

df = pd.read_csv("training.csv")
tests=pd.read_csv("testing.csv")

def knn():
    df = pd.read_csv("training.csv")
    tests=pd.read_csv("testing.csv")
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
    print("KNN Accuracy ", accuracy)
    
    knn_predictions = knn.predict(X_test) 
    labels=['U','BA','A','AA','E']
    cm = confusion_matrix(y_test, knn_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.title("KNN")
    plt.show()
    return accuracy

def bayes():
    df = pd.read_csv("training.csv")
    tests=pd.read_csv("testing.csv")

    d={'Unsatisfactory':0, 'Below Average':1, 'Average':2,'Above Average':3,'Excellent':4}
    
    df['Rating'] = df['Rating'].map(d)
    tests['Output']=tests['What we should get'].map(d)

    features=['Knowledge','Enthusiasm','Organization','Preparedness','Ability to Explain Difficult Concepts','Speaking Ability','Approachability','Approachability','Effectiveness of Examples']

    X = df[features]
    y = df['Rating']

    X_test=tests[features]
    y_test=tests['Output']

    # print(X_test)
    # print(y_test)

    gnb = GaussianNB().fit(X,y)

    # print("Gaussian Naive Bayes")
    # print(gnb)
    gnb_predict=gnb.predict(X_test)

    ypred=np.array(gnb_predict)
    ytest=np.array(y_test)

    print("Naive Bayes Results:")
    print('Recall: %.3f' % recall_score(ytest, ypred,average="weighted"))
    print('Precision: %.3f' % precision_score(ytest, ypred,average="weighted"))
    print('F1-score: %.3f' % f1_score(ytest, ypred,average="weighted"))
    print('Accuracy: %.3f' % accuracy_score(ytest, ypred))

    # print("Model probabilities")
    classprob = (gnb.predict_proba(X_test))

    x_axis_labels=['U','BA','A','AA','E']
    y_axis_labels=['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S12','S13','S14','S15','S16','S17','S18']
    ax = sns.heatmap(classprob,linewidth=0.5,xticklabels=x_axis_labels,yticklabels=y_axis_labels,annot=True)
    plt.title("Naive Bayes")
    plt.show()
    return accuracy_score(ytest, ypred)

def neural():
    df = pd.read_csv("training.csv")
    tests=pd.read_csv("testing.csv")
    d={'Unsatisfactory':0, 'Below Average':1, 'Average':2,'Above Average':3,'Excellent':4}
    
    df['Rating'] = df['Rating'].map(d)
    tests['Output']=tests['What we should get'].map(d)

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
    # print(y_test)

    N=len(features)
    m = 5 # number of output neurons

    #Use IJRS paper. See ML Literature folder
    hidden1 = int(math.sqrt((m+2)*N) + 2*math.sqrt(N/(m+2)))
    hidden2=int(m*math.sqrt(N/(m+2)))
    # print("Number of features: %s" % len(features))


    # print("Neurons in hidden layer 1: %s" % hidden1)
    # print("Neurons in hidden layer 2: %s" % hidden2)

    clf = MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(9,9,9),random_state=1)

    clf.fit(Xscale,y)
    clfpredict = clf.predict(Xtestscale)
    # print(clfpredict)


    #print(dtree_predict)

    ypred=np.array(clfpredict)
    ytest=np.array(y_test)

    #plt.plot(ypred,marker='d')
    #plt.show()
    print("Neural Networks Results:")
    print('Recall: %.3f' % recall_score(ytest, ypred,average="weighted"))
    print('Precision: %.3f' % precision_score(ytest, ypred,average="weighted"))
    print('F1-score: %.3f' % f1_score(ytest, ypred,average="weighted"))
    print('Accuracy: %.3f' % accuracy_score(ytest, ypred))


    cm = confusion_matrix(y_test,clfpredict)
    fig=plt.figure()
    cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels =['U','BA','A','AA','E'])

    # print(cm)
    cm_display.plot()
    plt.title("Neural Networks")
    plt.show()
    # plt.colorbar()
    # plt.ylabel('True Label')
    # plt.xlabel('Predicted Label')

def svm():
    df = pd.read_csv("training.csv")
    tests=pd.read_csv("testing.csv")
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


    tests['Output'] = tests['What we should get'].map(d)

    features = ['Knowledge','Enthusiasm','Organization','Preparedness','Ability to Explain Difficult Concepts',
                'Speaking Ability','Approachability','Encourages Discussion','Effectiveness of Examples']

    Xtest = tests[features]
    ytest = tests['Output']

    SVM_Model = SVC(gamma='auto')

    SVM_Model.fit(X, y)

    # print(f'Accuracy - : {SVM_Model.score(Xtest, ytest)}')

    #for calculating the scores
    y_true = tests['Output']
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
    # print("Confusion matrix:")
    # print(conf_matrix)

    display_labels = ['U', 'BA', 'A', 'AA', 'E']

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=display_labels)

    disp.plot()
    plt.title("SVM")
    plt.show()

def log_regression():
    df = pd.read_csv("training.csv")
    tests=pd.read_csv("testing.csv")

    d={'Unsatisfactory':0, 'Below Average':1, 'Average':2,'Above Average':3,'Excellent':4}
    
    df['Rating'] = df['Rating'].map(d)
    tests['Output']=tests['What we should get'].map(d)

    features=['Knowledge','Enthusiasm','Organization','Preparedness','Ability to Explain Difficult Concepts','Speaking Ability','Approachability','Approachability','Effectiveness of Examples']

    X = df[features]
    y = df['Rating']

    X_test=tests[features]
    y_test=tests['Output']
    sc_x = StandardScaler()
    xtrain = sc_x.fit_transform(X) 
    xtest = sc_x.transform(X_test)

    classifier = LogisticRegression(random_state = 0)
    classifier.fit(xtrain, y)

    y_pred = classifier.predict(xtest)
    cm = confusion_matrix(y_test, y_pred)
    labels=['U','BA','A','AA','E']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.title("Logistic Regression")
    plt.show()



def main():
    knn()
    bayes()
    neural()
    svm()
    log_regression()





if __name__ == "__main__":
    main()