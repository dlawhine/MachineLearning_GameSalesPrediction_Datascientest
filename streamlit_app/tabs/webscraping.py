import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
from matplotlib import cm
import io

title = "Webscraping"
sidebar_name = "Webscraping"

# get current directory
path = os.getcwd()
parent = os.path.dirname(path)
DATA_URL_1 = (parent + '/Data/vgsales.csv')
DATA_URL_2 = (parent + '/Data/metacritic_raw_17853games.csv')
DATA_URL_3 = (parent + '/Data/jeuxvideocom_csv.csv')

# @st.cache(persist=True)( If you have a different use case where the data does not change so very often, you can simply use this)
def load_data(DATA_URL_):
    data = pd.read_csv(DATA_URL_)
    return data

df1 = load_data(DATA_URL_1)
df2 = load_data(DATA_URL_2)
df3 = load_data(DATA_URL_3)



def run():
    st.title(title)

    st.markdown(
        """
        ## Dataset Kaggle - Sales
        
        The dataset on games sales is available on: https://www.kaggle.com/gregorut/videogamesales.
        """
    )

    with st.container():
        st.dataframe(df1.head(20), width=5000)
        st.write(df1.shape)
    st.markdown(
        """
        ## Dataset Metacritic - Critics

        The dataset on games sales is available on: https://www.metacritic.com/browse/games/score/userscore/all/all/filtered?sort=desc&view=condensed&page=0.
        """
    )
    st.image(Image.open("assets/screenshot_metacritic_1.png"))
    st.markdown(
        """
        
        """
    )
    st.image(Image.open("assets/screenshot_metacritic_2.png"))
    st.markdown(
        """

        """
    )
    with st.container():
        st.dataframe(df2.drop(columns=['Unnamed: 0']).head(20), width=5000)
        st.write(df2.shape)
    st.markdown(
        """
        ## Dataset JeuxVideos - Critics

        The dataset on games sales is available on: https://www.jeuxvideo.com/tous-les-jeux/ .
        """
    )
    st.image(Image.open("assets/screenshot_jeuxvideos.png"))
    st.markdown(
        """

        """
    )
    with st.container():
        st.dataframe(df3.drop(columns=['Unnamed: 0']).head(20), width=5000)
        st.write(df3.shape)