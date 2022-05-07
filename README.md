##
## Introduction
##
This repository contains the code for our project **GameCashPy**, developed during our [Data Scientist training](https://datascientest.com/en/data-scientist-course) at [DataScientest](https://datascientest.com/).

The goal of this project is to predict video game sales (machine learning regression).

This project was carried out during 6 weeks and was developed by the following team :

- Didier Law-Hine ([GitHub](https://github.com/dlawhine) / [LinkedIn](https://www.linkedin.com/in/didier-lh/))
- Bocar Diop


##
## Jupyter Notebook + Python scripts for webscraping
##
A full report (in French) explaining our 6-week project is available in the folder [notebooks](./notebooks) : Rapport_gamingsales_fev2022DS_bdiop_dlawhine.ipynb

Python scripts for webscraping are also available in this folder.

##
## Streamlit App
##
A Streamlit app is available in the folder [streamlit_app](./streamlit_app).
To run the app :

```shell
cd streamlit_app
conda create --name my-awesome-streamlit python=3.9
conda activate my-awesome-streamlit
pip install -r requirements.txt
streamlit run app.py
```

The app should then be available at [localhost:8501](http://localhost:8501).

