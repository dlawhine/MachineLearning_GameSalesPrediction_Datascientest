import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler #pour les outliers
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.cluster import KMeans
from itertools import groupby
from sklearn import linear_model
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import learning_curve

import streamlit as st
import pandas as pd
import numpy as np
from preprocess import evaluation,load_data_first,encodage_first,publisher_norm
from preprocess import load_data_second,clustering_publisher,data_reg,data_classif,Visualisation_clusters
from preprocess import minmaxscaler,robustscaler_reg,robustscaler_classif,publisher_kmeans
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate




def get_dataset(name,model):
    data = None
    if name == 'Dataset (First test)':
        if (model =='Kernel Ridge CV')|(model =='Regression Lasso'):
            data = data_reg()
            X_train,X_test,y_train, y_test = robustscaler_reg(data)
        
        elif (model == 'SVM')|(model == 'Random Forest'):
            data = data_reg()
            X_train,X_test,y_train, y_test = robustscaler_classif(data)
        
    else:
        if name == 'Dataset (Cluster)':
            if (model =='SVM')|(model =='Random Forest'):
                data = data_classif()
                X_train, X_test, y_train, y_test= robustscaler_classif(data)
            
            elif (model =='Kernel Ridge CV')|(model =='Regression Lasso'):
                data = data_classif()
                X_train,X_test,y_train, y_test = robustscaler_reg(data)
           
    return X_train, X_test, y_train, y_test


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.slider('C', 0.1,20.0,10.0)
        params['C'] = C
        #kernel = st.sidebar.slider('kernel','rbf', 'linear','poly')
        #params['kernel'] = kernel
        gamma = st.slider('gamma',0.001,0.50,0.1)
        params['gamma'] = gamma
    elif clf_name == 'Kernel Ridge CV':
        alpha = st.slider('alpha',0.01,10.0,0.1)
        params['alpha'] = alpha
        #kernel = st.sidebar.slider('kernel',['rbf'])
        #params['kernel'] = kernel
        gamma = st.slider('gamma',0.01,10.0,0.1)
        params['gamma'] = gamma

    elif clf_name == 'Regression Lasso':
        alpha = st.slider('alpha', -0.05, 1.0, 0.0015264179671752333)
        params['alpha'] = alpha
    else:
        n_estimators = st.slider('n_estimators',1, 100, 5)
        params['n_estimators'] = n_estimators
        min_samples_leaf = st.slider('min_samples_leaf',1, 10, 5)
        params['min_samples_leaf'] = min_samples_leaf
        #max_features = st.sidebar.slider('max_features',['sqrt', 'log2'])
        #params['max_features'] = max_features
    return params





def get_classifier(clf_name,params):
    clf = None
    if clf_name == 'SVM':
        clf = svm.SVC(C=params['C'],kernel='rbf',gamma=params['gamma'])
    
    elif clf_name == 'Kernel Ridge CV':
        clf = KernelRidge(alpha=params['alpha'],kernel='rbf',gamma=params['gamma'])

    elif clf_name == 'Regression Lasso':
        clf = Lasso(alpha=params['alpha'])

    else:
        clf  = RandomForestClassifier(n_estimators=params['n_estimators'],min_samples_leaf=params['min_samples_leaf'],max_features='sqrt')
        
    return clf



def feature_eng():
    cst =load_data_second()
    unique_publisher= cst["Publisher"].unique()
    data_norm=publisher_norm(cst)
    labels=publisher_kmeans(data_norm)
    figures=Visualisation_clusters(data_norm,labels)
    return unique_publisher,data_norm,labels,figures