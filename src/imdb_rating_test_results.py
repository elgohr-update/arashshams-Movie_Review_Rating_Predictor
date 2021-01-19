# authors: Yuanzhe Marco Ma, Arash Shamseddini, Kaicheng Tan, Zhenrui Yu
# date: 2020-11-25
"""Assesses model's accuracy on a test data set.

Usage:
  imdb_rating_test_results.py <model> <test> <out>
  imdb_rating_test_results.py (-h | --help)

Options:
  <model>     Path to the serialized model file
  <test>      Path to the test data file
  <out>       Path to the directory where result(s) should be written to
  -h, --help  Display help
"""

import os
from pathlib import Path

import altair as alt
import joblib
import pandas as pd
from docopt import docopt
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
)

drop_features = ['Id', 'Author']
target = 'Rating'


def main(model_path, test, out):
    # Load data set
    test_df = pd.read_csv(test)
    X_test, y_test = test_df.drop(columns=[target] + drop_features), test_df[target]

    # Load model and predict the test set
    final_model = joblib.load(model_path)
    prediction = final_model.predict(X_test)

    # Create output directories in not exist
    Path(out).mkdir(parents=True, exist_ok=True)

    # Evaluate model
    pd.DataFrame({
        'r2': [r2_score(y_test, prediction)],
        'rmse': [mean_squared_error(y_test, prediction, squared=False)]
    }, index=['model']).to_csv(os.path.join(out, 'model_test_scores.csv'))

    # Generate visualization
    predict_true_df = pd.DataFrame({'predict': prediction, 'true': y_test})
    identity_line = alt.Chart(pd.DataFrame({'predict': [0., 10.], 'true': [0., 10.]})).mark_line(color='red').encode(
        x=alt.X('predict'),
        y=alt.Y('true'),
    )
    scatter_plot = alt.Chart(predict_true_df, title='Scatter plot of true and predicted rating') \
        .mark_point(opacity=0.3) \
        .encode(
        x=alt.X('predict', title='Predicted rating'),
        y=alt.Y('true', title='True rating')
    )
    (scatter_plot + identity_line) \
        .properties(width=500, height=450) \
        .save(os.path.join(out, 'true_vs_predict.svg'))


if __name__ == '__main__':
    arguments = docopt(__doc__)
    print(f'Evaluating model: {arguments["<model>"]} with test data: {arguments["<test>"]}')
    main(arguments["<model>"], arguments["<test>"], arguments["<out>"])
