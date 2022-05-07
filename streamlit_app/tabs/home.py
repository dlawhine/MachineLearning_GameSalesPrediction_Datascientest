import streamlit as st


title = "Gaming Sales"
sidebar_name = "Home"


def run():

    # TODO: choose between one of these GIFs
    # st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/1.gif")
    #st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/2.gif")
    st.image("https://dst-studio-template.s3.eu-west-3.amazonaws.com/3.gif")

    st.title(title)

    st.markdown("---")

    st.markdown(
        """
        The video games industry represents one of the most important sectors in the entertainment market.
        Lots of data are available publicly on worldwide sales and quality of games (critics) and can be extracted from the Internet.
        No wonder why Prediction of games sales has been a popular topic for developing Machine Learning models. 
       
        Our project is of the many that deal with such a topic. However, we try here :
        - to provide a global understanding of the problem of prediction 
        - to show how data can be extracted from the web
        - to introduce our methods to tackle the problem with Data visualization tools
        - to assess the difficulty of getting a prediction with high accuracy
        - to propose an outlook to improve the accuracy
        
        """
    )
