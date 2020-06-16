import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import pickle

import re
import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import precision_score, recall_score, f1_score,classification_report
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df=pd.read_sql_table('disasterdata', engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    return X, Y,Y.columns
    


def tokenize(text):
    url_pat = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Detect and replace urls
    detected_urls = re.findall(url_pat, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    stopword= list(set(stopwords.words('english')))
    clean_tokens = [token for token in clean_tokens if token not in stopword]

    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vec',CountVectorizer(tokenizer=tokenize)),
        ('tfid',TfidfTransformer()),
        ('clf',MultiOutputClassifier(RandomForestClassifier(max_features=0.5)))
    ])
    parameters = {
        'clf__estimator__n_estimators': [10]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs= -1)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred=model.predict(X_test)
    y_test=np.array(Y_test)
    kuch_v = []
    for i in range(len(category_names)):
        kuch_v.append([f1_score(y_test[:, i], y_pred[:, i], average='micro'),
                             precision_score(y_test[:, i], y_pred[:, i],                                                average='micro'),
                             recall_score(y_test[:, i], y_pred[:, i], average='micro')])
    # build dataframe
    report = pd.DataFrame(kuch_v, columns=['f1 score', 'precision', 'recall'],
                                index = category_names)   
    return report


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()