import streamlit as st
import pandas as pd
import numpy as np
from predict import get_dataset,get_classifier,add_parameter_ui,feature_eng
from preprocess import evaluation,load_data_first,encodage_first
from preprocess import load_data_second,clustering_publisher,publisher_norm,publisher_kmeans
from preprocess import minmaxscaler,robustscaler_reg,robustscaler_classif,Visualisation_clusters
from sklearn import linear_model
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
from sklearn.metrics import mean_squared_error
from sklearn import model_selection


title = "Gaming Sales"
sidebar_name = "Regression/Classification"


def run():

    st.title(title)

   


    dataset_name = st.selectbox(
        'Choice',
        ('Pre-processing(First)','Dataset (First test)','Pre-processing','Dataset (Cluster)')
    )
    if (dataset_name == 'Dataset (First test)')|(dataset_name == 'Dataset (Cluster)'):
        options =['Regression','Classification']
        choix = st.radio('',options = options)
        if choix == options[0]:

            st.write(f"## {dataset_name}")

            classifier_name = st.selectbox(
                'Choose a regression model',
                ( 'Regression Lasso','Kernel Ridge CV')
                )
            
            if (classifier_name == 'Kernel Ridge CV')|(classifier_name == 'Regression Lasso'):
                
                X_train_classification,X_test_classification,y_train_classification, y_test_classification = get_dataset(dataset_name,classifier_name)
                params = add_parameter_ui(classifier_name)
                ok = st.button("Submit")
                if ok:
                        clf = get_classifier(classifier_name,params)
                        clf.fit(X_train_classification, y_train_classification)
                        st.write(f'Classifier = {classifier_name}')
                        y_pred = clf.predict(X_train_classification)
                        y_pred_test = clf.predict(X_test_classification)
                        acc_train = clf.score(X_train_classification, y_train_classification)
                        acc_test = clf.score(X_test_classification, y_test_classification)
                        mse_train = mean_squared_error(y_pred, y_train_classification)
                        mse_test = mean_squared_error(y_pred_test, y_test_classification)
                        st.write(X_train_classification.shape)
                        st.write(f'score train =', acc_train)
                        st.write(f'score test =', acc_test)
                        st.write(f'MSE train =', mse_train)
                        st.write(f'MSE test =', mse_test)

        else:
            classifier_name = st.selectbox(
                'Choose a classification model',
                ('SVM', 'Random Forest')
                )
            if (classifier_name == 'SVM') |(classifier_name == 'Random Forest') :
                    
                    X_train_classification,X_test_classification,y_train_classification, y_test_classification = get_dataset(dataset_name,classifier_name)
                    params = add_parameter_ui(classifier_name)
                    ok = st.button("Submit")
                    if ok:    
                        clf = get_classifier(classifier_name,params)
                        clf.fit(X_train_classification, y_train_classification)
                        y_pred = clf.predict(X_test_classification)
                        st.write(f'Classifier = {classifier_name}')
                        acc_train = clf.score(X_train_classification, y_train_classification)
                        acc_test = clf.score(X_test_classification, y_test_classification)
                        cm = pd.crosstab(y_test_classification, y_pred, rownames=['Realité'], colnames=['Prédiction'])
                        cr = classification_report(y_test_classification, y_pred)
                        st.write(X_train_classification.shape)
                        st.write(f'score train =', acc_train)
                        st.write(f'score test =', acc_test)
                        st.write(f'Matrice de confusion : ',cm)
                        st.write(f'Rapport de classification : ',cr)
                        #fig =evaluation(clf,X_train_classification,X_test_classification,y_train_classification, y_test_classification)
                        #st.pyplot(fig)
    elif (dataset_name == 'Pre-processing(First)'):
        st.write("\n\n\n")
        st.write("- Removal of NANs \n")
        st.write("\n\n\n\n\n\n")
        st.write("\n\n\n\n\n\n")
        st.write("\n\n\n\n\n\n")
        st.write("- Removal of columns : \n")
        st.write("-`'N_players'`\n")
        st.write("-`'Name'` and `'Year'` \n")
        st.write("-`'NA_Sales'`, `'JP_Sales'`,`'Other_Sales'`,`'EU_Sales'`\n")
        st.write("\n\n\n\n\n\n")
        st.write("\n\n\n\n\n\n")
        st.write("\n\n\n\n\n\n")
        st.write("- MinMaxScaler, RobustScaler\n")
        st.write("\n\n\n\n\n\n")
        st.write("\n\n\n\n\n\n")
        st.write("- Classification, it is necessary to discretize the quantitative variable 'Global_Sales' according to the quartiles : \n")
        st.write("-`'(q = 0, 1, 2, 3)'`\n")
        st.write("\n\n\n\n\n\n")
    
    else:
        unique_publisher,data_norm,labels,figures = feature_eng()
        st.write("- Transformation of the categorical variable `'Publisher'`:\n")
        st.write("-The modalities (5 first entries) of the column `'Publisher'` with 243 categories : \n",unique_publisher[:5,])
        st.write("\n\n")
        st.write("-Dataframe with 2 columns (**global sales according to publisher**, **entries de publisher**): \n")
        st.write(data_norm)
        st.write("\n\n")
        st.write("-Graph of the scatter plot assigned to the corresponding cluster (**with the Kmeans algorithm**) : \n")
        st.pyplot(figures)
        st.write("\n\n")
        st.write("-Cluster corresponding to categories (**with the Kmeans algorithm**) : \n")
        st.write(labels[:5,])
        st.write("\n\n")
        st.write("-**Removal of sales > 5 millions** \n")
        
        
