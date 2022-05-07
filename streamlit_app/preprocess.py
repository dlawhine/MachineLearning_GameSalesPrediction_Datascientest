import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler #pour les outliers
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.cluster import KMeans
from itertools import groupby
from sklearn import linear_model
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import learning_curve
from sklearn.cluster import KMeans
from matplotlib.pyplot import cm
import os

# get current directory
path = os.getcwd()
parent = os.path.dirname(path)
DATA_URL = (parent + '/Data/vgsales_metacritic.csv')


def load_data_first():
   data = pd.read_csv(DATA_URL)
   data = data.drop(['Name','Year','NA_Sales','JP_Sales','Other_Sales','EU_Sales','N_players'], axis=1)
   data.dropna(inplace=True)
   return data



def encodage_first(df):
    X_cat = df.select_dtypes('object')
    X_num = df.select_dtypes(['float64'])
    X_cat = pd.get_dummies(X_cat)
    X = pd.concat([X_cat,X_num], axis=1)
        
    return X

def minmaxscaler(df):
    scaler = MinMaxScaler()
    #scaler = RobustScaler()
    target=df['Global_Sales']
    data=df.drop(['Global_Sales'],axis=1)

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2,random_state=42)
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = data.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns = data.columns)   
    return X_train,X_test,y_train, y_test

def robustscaler_reg(df):
    scaler = RobustScaler()
    target=df['Global_Sales']
    data=df.drop(['Global_Sales'],axis=1)

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2,random_state=42)
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns = data.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns = data.columns)   
    return X_train,X_test,y_train, y_test

def load_data_second():
   data = pd.read_csv("vgsales_metacritic.csv")
   data = data.drop(columns="N_players")
   data.dropna(inplace=True)
   data["ESRB_ratings"][data["ESRB_ratings"]=='K-A']='E'
   data = data.drop(data[(data.ESRB_ratings=='AO')|(data.ESRB_ratings=='RP')].index)
   return data

def clustering_publisher(df):
    df = df.drop(df[df.Global_Sales > 5].index)
    game = df.groupby("Publisher")["Global_Sales"].sum().sort_values(ascending=False)
    total_sum_globalsales_groupbyPublisher=game.sum()

    gamesales_norm=game/total_sum_globalsales_groupbyPublisher
    gameoccurence_norm = df["Publisher"].value_counts(ascending=False,normalize=True)

    df_clus=pd.concat([gamesales_norm, gameoccurence_norm], axis=1,keys=['Gobalsales_norm','Occurence_norm'])
    
    # Initialisation du classificateur CAH 
    n_clusters=15
    cluster = KMeans(n_clusters = n_clusters,random_state=0)
    # Apprentissage des données 
    cluster.fit(df_clus)
    
    labels = cluster.labels_
    s=pd.Series(labels, index=df_clus.index)
    publisher_clustlabel_dict=s.to_dict()

    s2=pd.Series(labels, index=df_clus.index)
    s2=s2.astype(str)
    publisher_clustlabel_dict_string=s2.to_dict()
    dict_cluster_publisher_label = dict(
    (key, [G[1] for G in g]) for (key, g) in
    groupby(sorted( (val, key) for (key, val) in publisher_clustlabel_dict.items() ),lambda X: X[0]))
    df=df.replace({'Publisher': publisher_clustlabel_dict_string})
    df = df.drop(['Name','Year','NA_Sales','JP_Sales','Other_Sales','EU_Sales'], axis=1)
    return df


def robustscaler_classif(df):
    sc = RobustScaler()

    df.Global_Sales = pd.qcut(df.Global_Sales, labels = [0,1,2,3],q = 4)
#X.Global_Sales = pd.qcut(df.Global_Sales, labels = [0,1,2,3,4,5,6,7,8,9],q = 10)

    target=df['Global_Sales']
    data=df.drop(['Global_Sales'],axis=1)

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2,random_state=42)
    X_train = pd.DataFrame(sc.fit_transform(X_train), columns = data.columns)
    X_test = pd.DataFrame(sc.transform(X_test), columns = data.columns)   
    return X_train,X_test,y_train, y_test


def data_reg():
    df = load_data_first()
    data_rég = encodage_first(df) 
    return data_rég

def data_classif():
    dt = load_data_second()
    data_reduc = clustering_publisher(dt)
    data_classif = encodage_first(data_reduc)
    return data_classif

def evaluation(model,X_train,X_test,y_train, y_test):
    
    model.fit(X_train, y_train)
    ypred = model.predict(X_test)
    
    
    cm =pd.crosstab(y_test, ypred, rownames=['Realité'], colnames=['Prédiction'])
    st.write(cm)
    cr = classification_report(y_test, ypred)
    st.write(cr)
    
    N, train_score, val_score = learning_curve(model, X_train, y_train,
                                              cv=4, scoring='f1',
                                               train_sizes=np.linspace(0.1, 1, 10))
    
    
    fig = plt.figure(figsize=(12, 8))
    plt.plot(N, train_score.mean(axis=1), label='train score')
    plt.plot(N, val_score.mean(axis=1), label='validation score')
    plt.legend()
    #plt.show()
    return fig


    # Creation d'un dataframe avec 2 colonnes: global sales en fonction de publisher (normalisé),  occurence de publisher (normalisé)

def publisher_norm(df):
    game = df.groupby("Publisher")["Global_Sales"].sum().sort_values(ascending=False)
    total_sum_globalsales_groupbyPublisher=game.sum()

    gamesales_norm=game/total_sum_globalsales_groupbyPublisher
    gameoccurence_norm = df["Publisher"].value_counts(ascending=False,normalize=True)

    df_clus=pd.concat([gamesales_norm, gameoccurence_norm], axis=1,keys=['Gobalsales_norm','Occurence_norm'])
    return df_clus


def publisher_kmeans(df_clus) :  
    # Initialisation du classificateur CAH 
    n_clusters=15
    cluster = KMeans(n_clusters = n_clusters,random_state=0)
    # Apprentissage des données 
    cluster.fit(df_clus)
    # Labels
    labels = cluster.labels_
    return labels


def Visualisation_clusters(df_clus,labels):
    n_clusters=15
    fig = plt.figure(figsize=(12, 8))
    # Liste des couleurs
    color = list(cm.jet(np.linspace(0, 1, n_clusters)))
    # Graphique du nuage de points attribués au cluster correspondant
    for i in range(len(df_clus)):
        plt.scatter(df_clus.iloc[i,0], df_clus.iloc[i,1], color=color[labels[i]])

    plt.xlabel('Gobalsales_norm') 
    plt.ylabel('Occurence_norm') 
    return fig    