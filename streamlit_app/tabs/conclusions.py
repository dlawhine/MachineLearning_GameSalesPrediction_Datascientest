import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

title = "Outlook"
sidebar_name = "Outlook"


def run():

    st.title(title)

    st.markdown(
        """
        #### Main results
        - Clustering (retain descriptive features, reduce computing time) 
        - Impact of Outliers
        - Score (R^2 = 0.5-0.6).
        """
    )

    st.markdown(
        """
        #### Creation of a new feature based on Rewards.
        https://ultimatepopculture.fandom.com/wiki/List_of_Game_of_the_Year_awards 
        That shall help to model the impact of outliers.
        """
    )

    st.image(Image.open("assets/screenshot_fandom_1.png"))
    st.markdown(
        """

        """
    )

    st.image(Image.open("assets/screenshot_fandom_2.png"))

    st.markdown(
        """
        #### Creation of a new feature based on Game series.
        2 categories : games that belong to a series (Fifa 2000, 2001, 2002, etc.) and games that do not.
        That shall help to improve the overall score.
        """
    )

