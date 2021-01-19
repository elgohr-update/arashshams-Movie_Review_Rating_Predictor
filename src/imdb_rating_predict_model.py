# authors: Yuanzhe Marco Ma, Arash Shamseddini, Kaicheng Tan, Zhenrui Yu
# date: 2020-11-25
"""Fits a SVR model on the preprocessed data from the IMDB review data set.
   Saves the model with optimized hyper-parameters, as well as the search result.

Usage:
  imdb_rating_predict_model.py <train> <out>
  imdb_rating_predict_model.py (-h | --help)

Options:
  <train>     Path to the training data file
  <out>       Path to the directory where output should be written to
  -h, --help  Display help
"""

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from docopt import docopt
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OrdinalEncoder,
    StandardScaler,
)
from sklearn.linear_model import Ridge

numeric_features = ['n_words']
text_feature = 'Text'
ordinal_features = ['sentiment']
drop_features = ['Id', 'Author']
target = 'Rating'


def main(train, out):
    # Load data set
    train_df = pd.read_csv(train)
    X_train, y_train = train_df.drop(columns=[target] + drop_features), train_df[target]

    # Create ML pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', CountVectorizer(max_features=20_000, stop_words='english'), text_feature),
            ('num', StandardScaler(), numeric_features),
            ('ord', OrdinalEncoder(categories=[['neg', 'compound', 'neu', 'pos']]), ordinal_features)
        ]
    )
    ml_pipe = Pipeline(
        steps=[
            ('prepro', preprocessor),
            ('ridge', Ridge())
        ]
    )

    # Tune hyper-parameters
    print('Searching for hyper-parameters')
    param_grid = {
        'ridge__alpha': np.arange(500, 1000, 50)
    }
    hyper_parameters_search = GridSearchCV(ml_pipe,
                                           param_grid=param_grid,
                                           n_jobs=-1,
                                           scoring='r2',
                                           return_train_score=True,
                                           verbose=1)
    hyper_parameters_search.fit(X_train, y_train)
    print(f'R2 score for best model: {hyper_parameters_search.best_score_}')

    # Write hyper-parameter search result to csv
    hyper_parameters_search_result = pd.DataFrame(hyper_parameters_search.cv_results_) \
        .sort_values(by='mean_test_score', ascending=False)
    hyper_parameters_search_result.to_csv(os.path.join(out, 'hyper_param_search_result.csv'))

    # Fit final model
    final_model = hyper_parameters_search.best_estimator_
    final_model.fit(X_train, y_train)

    # Save pipeline
    print(f'Writing model to directory: {arguments["<out>"]}')
    Path(out).mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, os.path.join(out, 'model.pkl'), compress=3)


if __name__ == '__main__':
    arguments = docopt(__doc__)
    print(f'Constructing model from train data {arguments["<train>"]}')
    main(arguments["<train>"], arguments["<out>"])
