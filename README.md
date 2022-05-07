##
## Introduction
##
This repository contains the code for our project **GameCashPy**, developed during our [Data Scientist training](https://datascientest.com/en/data-scientist-course) at [DataScientest](https://datascientest.com/).

The goal of this project is to predict video game sales (machine learning regression).

This project was carried out during 6 weeks and was developed by the following team :

- Didier Law-Hine ([GitHub](https://github.com/dlawhine) / [LinkedIn](https://www.linkedin.com/in/didier-lh/))
- Bocar Diop

You can browse and run the [notebooks](./notebooks). You will need to install the dependencies (in a dedicated environment) :

```
pip install -r requirements.txt
```


##
## Jupyter Notebooks
##
A full report explaining our 6-week project is available in the folder notebooks/


##
## Streamlit App
##
A Streamlit app is available in the folder streamlit_app/
To run the app :

```shell
cd streamlit_app
conda create --name my-awesome-streamlit python=3.9
conda activate my-awesome-streamlit
pip install -r requirements.txt
streamlit run app.py
```

The app should then be available at [localhost:8501](http://localhost:8501).

