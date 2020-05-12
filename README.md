# Disaster-Response-pipeline
Project for Udacity assignment - Disaster response pipeline for twitter messages analysis


## Description
This Project is part of Data Science Nanodegree Program by Udacity in collaboration with Figure Eight. The dataset provided is actual pre-labelled tweets and the goal of the project is to process and categorise such messages in catogries using nbatural language processing . A final step of the project is a flask dashboard.

## The stages of the project are :

1. Data Processing - ETL Pipeline -cleaning and transformation of the initial data
2. Machine Learning Pipeline - for tweets categorisation
3. Flask Web App to demonstrate some visualisation on the data provided

## Getting Started

Dependencies
* Python 3.5+ (I used Python 3.7)
* Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
* NLP packages: NLTK
* SQLlite Database Libraqries: SQLalchemy
* Web App and Data Visualization: Flask, Plotly



## Executing Program:
Run the following commands in the project's root directory to set up your database and model.

* To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
* To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
* Run the following command in the app's directory to run your web app. python run.py

Go to http://0.0.0.0:3001/


## Additional Material
In the data and models folder you can find two jupyter notebook that will help you understand how the model works step by step:

* ETL Preparation Notebook: learn everything about the implemented ETL pipeline
* ML Pipeline Preparation Notebook: look at the Machine Learning Pipeline developed with NLTK and Scikit-Learn
You can use ML Pipeline Preparation Notebook to re-train the model or tune it through a dedicated Grid Search section.


## Authors
Kat Pasecnika

## License
License: MIT


## Acknowledgements
Thanks to Udacity for the opportunity to work on this data and Figure Eight for providing the dataset itself
