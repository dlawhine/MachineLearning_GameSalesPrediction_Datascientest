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



title = "Exploratory Analytics"
sidebar_name = "Exploratory Analytics"

# get current directory
path = os.getcwd()
parent = os.path.dirname(path)
DATA_URL = (parent + '/Data/vgsales_metacritic.csv')


# @st.cache(persist=True)( If you have a different use case where the data does not change so very often, you can simply use this)
def load_data():
    data = pd.read_csv(DATA_URL)
    return data


df = load_data()
data = df


def run():
    st.title(title)

    st.markdown(
        """
        ## Dataset (50 first entries)
        
        Here is our dataset resulting from the merge from the two dataframes obtained by scraping. 
        This dataset is not cleaned yet.
        """
    )
    with st.container():
        st.dataframe(df.head(50), width=5000)
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()

        st.text(s)

    st.markdown(
        """
        ## Global Sales per Year in Dataset
        
        The number of Global Sales per Year from our dataset. 
        There is a lack of data on sales after 2012, as the scraping was made in 2016-2017, 
        and the data on the sales not being available immediately after the games were released.
        """
    )
    y1 = df.groupby("Year")["Global_Sales"].sum()
    y = y1.sort_values()
    x = y.index
    fig = go.Figure(data=[go.Bar(y=y, x=x, )])
    fig.update_layout(xaxis_title="Year", yaxis_title="Global Sales (in millions of units)", width=1000)
    st.plotly_chart(fig)

    st.markdown(
        """
        ## Global Sales and Entries per Genre (12 categories)
        
        We compare the proportion of the Global Sales feature and the proportion of observations (entries) 
        for the feature Genre.
    
         """
    )
    gamesales_genre = pd.DataFrame(df.groupby("Genre")["Global_Sales"].sum().sort_values(ascending=False)).reset_index()
    gamecount_genre = pd.DataFrame(df["Genre"].value_counts(ascending=False)).reset_index().rename(
        columns={'Genre': 'Count', 'index': 'Genre'})

    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])
    fig.add_trace(
        go.Pie(labels=list(gamesales_genre.Genre), values=list(gamesales_genre.Global_Sales), name="Global Sales"),
        1, 1)
    fig.add_trace(go.Pie(labels=list(gamecount_genre.Genre), values=list(gamecount_genre.Count), name="Entries"),
                  1, 2)
    fig.update_traces(hole=.4, hoverinfo="label+percent+name")
    fig.update_layout(width=1000,
                      annotations=[dict(text='Sales', x=0.19, y=0.5, font_size=20, showarrow=False),
                                   dict(text='Entries', x=0.815, y=0.5, font_size=20, showarrow=False)])
    st.plotly_chart(fig, use_container_width=False)

    st.markdown(
        """
        ## Global Sales and Entries per Platform (18 categories)
        
        We plot the same pie charts as a function of the Platform feature
         """
    )
    gamesales_platform = pd.DataFrame(df.groupby("Platform")["Global_Sales"].sum().sort_values(ascending=False)).reset_index()
    gamecount_platform = pd.DataFrame(df["Platform"].value_counts(ascending=False)).reset_index().rename(
        columns={'Platform': 'Count', 'index': 'Platform'})

    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])
    fig.add_trace(
        go.Pie(labels=list(gamesales_platform.Platform), values=list(gamesales_platform.Global_Sales), name="Global Sales"),
        1, 1)
    fig.add_trace(go.Pie(labels=list(gamecount_platform.Platform), values=list(gamecount_platform.Count), name="Entries"),
                  1, 2)
    fig.update_traces(hole=.4, hoverinfo="label+percent+name")
    fig.update_layout(width=1000,
                      annotations=[dict(text='Sales', x=0.19, y=0.5, font_size=20, showarrow=False),
                                   dict(text='Entries', x=0.815, y=0.5, font_size=20, showarrow=False)])
    st.plotly_chart(fig, use_container_width=False)

    st.markdown(
        """
        ## Global Sales and Entries per Publisher (249 categories)
        
        The feature Publisher has 249 modalities.
        We decide to plot the first 20 Publishers with most sales and entries.
         """
    )
    gamesales_publisher = pd.DataFrame(df.groupby("Publisher")["Global_Sales"].sum().sort_values(ascending=False)).reset_index()
    gamecount_publisher = pd.DataFrame(df["Publisher"].value_counts(ascending=False)).reset_index().rename(
        columns={'Publisher': 'Count', 'index': 'Publisher'})
    gamesales_publisher=gamesales_publisher.head(20)
    gamecount_publisher=gamecount_publisher.head(20)
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])
    fig.add_trace(
        go.Pie(labels=list(gamesales_publisher.Publisher), values=list(gamesales_publisher.Global_Sales), name="Global Sales"),
        1, 1)
    fig.add_trace(go.Pie(labels=list(gamecount_publisher.Publisher), values=list(gamecount_publisher.Count), name="Entries"),
                  1, 2)
    fig.update_traces(hole=.4, hoverinfo="label+percent+name")
    fig.update_layout(width=1000,
                      annotations=[dict(text='Sales', x=0.18, y=0.5, font_size=20, showarrow=False),
                                   dict(text='Entries', x=0.82, y=0.5, font_size=20, showarrow=False)])
    st.plotly_chart(fig, use_container_width=False)

    st.markdown(
        """
        ## Global Sales and Entries per ESRB ratings (Entertainment Software Rating Board)
         """
    )
    y1 = df.groupby("ESRB_ratings")["Global_Sales"].sum()
    y = y1.sort_values(ascending=False)
    x = y.index
    fig = go.Figure(data=[go.Bar(y=y, x=x, )])
    fig.update_layout(xaxis_title="ESRB_ratings", yaxis_title="Global Sales (in millions of units)", width=1000)
    st.plotly_chart(fig)

    st.markdown(
        """
        ## Distribution of Global Sales and User's Notes
        
        Box plots put in evidence the presence of several outliers that can be detrimental for  
         """
    )
    # gamesales_genre = df.groupby("Genre")["Global_Sales"]
    #
    # x_data = list(gamesales_genre.index)
    # y_data = list(gamesales_genre.Global_Sales)
    # fig = go.Figure()
    # for xd, yd in zip(x_data, y_data):
    #     fig.add_trace(go.Box(
    #         y=yd,
    #         name=xd,
    #         boxpoints='all',
    #         jitter=0.5,
    #         whiskerwidth=0.2,
    #         marker_size=2,
    #         line_width=1)
    #     )
    # fig.update_layout(
    #     title='Points Scored by the Top 9 Scoring NBA Players in 2012',
    #     yaxis=dict(
    #         autorange=True,
    #         showgrid=True,
    #         zeroline=True,
    #         dtick=5,
    #         gridcolor='rgb(255, 255, 255)',
    #         gridwidth=1,
    #         zerolinecolor='rgb(255, 255, 255)',
    #         zerolinewidth=2,
    #     ),
    #     margin=dict(
    #         l=40,
    #         r=30,
    #         b=80,
    #         t=100,
    #     ),
    #     paper_bgcolor='rgb(243, 243, 243)',
    #     plot_bgcolor='rgb(243, 243, 243)',
    #     showlegend=False
    # )
    # st.plotly_chart(fig, use_container_width=False)

    gamesales_platform =df.groupby("Platform")["Global_Sales"].sum().sort_values(ascending=False)
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3)
    gs.update(wspace=0.5, hspace=0.0)
    ax1 = plt.subplot(gs[0, 0:3], )
    ax2 = plt.subplot(gs[1, 0:3], )
    sns.boxenplot(ax=ax1, x='Platform', y='Global_Sales', data=df, order=gamesales_platform.index)
    ax1.set_ylim([14, 85])
    ax1.set(xticklabels=[])  # remove the tick labels
    ax1.set(xlabel=None);  # remove the axis label
    ax1.set(ylabel='Global_sales (in millions)');
    sns.boxenplot(ax=ax2, x='Platform', y='Global_Sales', data=df, order=gamesales_platform.index)
    ax2.set_ylim([-1, 14])
    ax2.set_xticklabels(labels=gamesales_platform.index, rotation=45);
    ax2.set(ylabel='Global_sales (in millions)');
    ax2.set(xlabel=None);  # remove the axis label

    st.pyplot(fig, use_container_width=True)

    notes_platform = df.groupby("Platform")["Note_users"].sum().sort_values(ascending=False)
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3)
    gs.update(wspace=0.5, hspace=0.0)
    ax2 = plt.subplot(gs[0, 0:3], )
    sns.boxenplot(ax=ax2, x='Platform', y='Note_users', data=df, order=notes_platform.index)
    ax2.set_ylim([0, 10])
    ax2.set_xticklabels(labels=gamesales_platform.index, rotation=45);
    ax2.set(ylabel='Note users');
    ax2.set(xlabel=None);  # remove the axis label

    st.pyplot(fig, use_container_width=True)


    st.markdown(
        """
        ## Correlation maps
        
        One needs to observe how the features are correlated with each other
         """
    )
    cor = df.corr()
    fig, ax = plt.subplots(figsize=(9, 9))
    sns.heatmap(cor, annot=True, ax=ax, cmap='coolwarm');
    st.pyplot(fig, use_container_width=True)

    st.markdown(
        """
        ## Pairplot
        
        The interdependency between features that we seem will be descriptive to our problem.
        Note that linearity cannot be observed. 
         """
    )
    fig = sns.pairplot(df.iloc[:, [7, 9, 10, 11]], diag_kind='kde');
    st.pyplot(fig)



