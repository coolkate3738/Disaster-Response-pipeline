# import libraries
import sqlite3
import pandas as pd
import os
from sqlalchemy import create_engine
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.ensemble import AdaBoostClassifier
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

import sys
import pickle

pd.options.display.max_rows = 500
pd.options.display.max_columns = 500
pd.set_option('display.max_colwidth', -1)


from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report
from sklearn.base import BaseEstimator,TransformerMixin

def load_data(database_filepath):
    """
    Loading pre-cleaned dataset from database
    
    Arguments:
        database_filepath - path to sqllite db
    Output:
        X -features data
        Y -modelling labels data
        categories - modelling labels list 
    """
    
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = os.path.basename(database_filepath).replace(".db","") + "_messages_clean"
    df = pd.read_sql_table(table_name,engine)
    
    
    
   #notice that related column has 3 categories, lets move 2 into 1 as that must be a typo
    df['related']=df['related'].map(lambda x: 1 if x == 2 else x)
    
    X = df['message']
    Y = df.iloc[:,4:]
    
    
    categories = Y.columns 
    return X, Y, categories


def tokenize(text,url_place_holder_string="urlplaceholder"):
    """
    Tokenize the text 
    
    Arguments:
        text - Text  which needs to be tokenized
    Output:
        clean_tokens - List of tokens extracted in the end
    """
       # Replace all urls with a urlplaceholder string
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Extract all the urls from the provided text 
    detected_urls = re.findall(url_regex, text)
    
    # Replace url with a url placeholder string
    for detected_url in detected_urls:
        text = text.replace(detected_url, url_place_holder_string)

    # Extract the word tokens from the provided text
    tokens = nltk.word_tokenize(text)
    
    #Lemmanitizer to remove inflectional and derivationally related forms of a word
    lemmatizer = nltk.WordNetLemmatizer()

    # List of clean tokens
    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]
    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class
    
    Takes first verb from a sentence
    """
    def starting_verb(self, text):
        # tokenize by sentences
        text = text.replace(" ", "_")
        sentence_list = nltk.sent_tokenize(text)
        
        for sentence in sentence_list:
            # tokenize each sentence into words and tag part of speech
            #words = word_tokenize(text)
            pos_tags= nltk.pos_tag(tokenize(sentence))

            # index pos_tags to get the first word and part of speech tag
            first_word, first_tag = pos_tags[0]
            
            # return true if the first word is an appropriate verb or RT for retweet
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True

            return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        # apply starting_verb function to all values in X
       
        X_tagged = pd.Series(X).apply(self.starting_verb)
        
        return pd.DataFrame(X_tagged)


def build_model_pipeline():
    """
    Building a Pipeline for the model 
    
    Output:
        Applying multioutput classifier using scikit learn on some text data 
        
    """
    pipeline= Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ])),

            ('starting_verb_extractor', StartingVerbExtractor())
        ])),

        ('classifier', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    return pipeline


def evaluate_model(pipeline, X_test, Y_test, categories):
    """
    Evaluate Model 
    
    Printing out modelling evaluation metrics 
    
    Arguments:
        pipeline -> A valid scikit ML Pipeline
        X_test -> Test features
        Y_test -> Test labels
        category_names -> label names (multi-output)
    """
    y_pred = pipeline.predict(X_test)
    
    
    accuracy = (y_pred == Y_test).mean().mean()

    print('accuracy {0:.2f}%'.format(accuracy*100))
   

    y_pred_pd = pd.DataFrame(y_pred, columns = Y_test.columns)
    for column in categories:
   
        print(column)
        print(classification_report(Y_test[column],y_pred_pd[column]))


def save_model(pipeline, model_filepath):
    """
    saving th emodel results as pickle file
    
    
    Arguments:
        pipeline - your pipeline
        model_filepath - where to save pickle file
    
    """
    pickle.dump(pipeline, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, categories = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model pipeline...')
        pipeline  = build_model_pipeline()
        
        print('Training model pipeline...')
        pipeline.fit(X_train, Y_train)
        
        print('Evaluating model pipeline...')
        evaluate_model(pipeline, X_test, Y_test, categories)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(pipeline, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()