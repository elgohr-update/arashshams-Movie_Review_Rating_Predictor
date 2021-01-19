# authors: Yuanzhe Marco Ma, Arash Shamseddini, Kaicheng Tan, Zhenrui Yu
# date: 2020-11-25
"""Splits and pre-processes the  IMDB review data (from https://github.com/nproellochs/SentimentDictionaries).

Usage: pre_process.py --input_file=<input_file> --out_dir=<out_dir> 

Options:
<input_file>       Raw csv file path (downloaded from the website, csv format)
<out_dir>          Path to directory where the processed data should be written (in csv format)
"""

import os
from pathlib import Path

import nltk
import pandas as pd
from docopt import docopt
from sklearn.model_selection import (
    train_test_split,
)

nltk.download("vader_lexicon")
nltk.download("punkt")

from nltk.sentiment.vader import SentimentIntensityAnalyzer # noqa

opt = docopt(__doc__)


def main(input_file, out_dir):
    data = pd.read_csv(input_file)

    # split training and test data
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=123)

    sid = SentimentIntensityAnalyzer()

    # Define some functions that are useful for our feature engineering
    def get_length_in_words(text):
        """
        Returns the length of the text in words.

        Parameters:
        ------
        text: (str)
        the input text

        Returns:
        -------
        length of tokenized text: (int)
        """
        return len(nltk.word_tokenize(text))

    def get_sentiment(text):
        """
        Returns the maximum scoring sentiment of the text

        Parameters:
        ------
        text: (str)
        the input text

        Returns:
        -------
        sentiment of the text: (str)
        """
        scores = sid.polarity_scores(text)
        return max(scores, key=lambda x: scores[x])

    train_df = train_df.assign(Rating=train_df["Rating"] * 10.0)
    train_df = train_df.assign(n_words=train_df["Text"].apply(get_length_in_words))
    train_df = train_df.assign(sentiment=train_df["Text"].apply(get_sentiment))

    test_df = test_df.assign(Rating=test_df["Rating"] * 10.0)
    test_df = test_df.assign(n_words=test_df["Text"].apply(get_length_in_words))
    test_df = test_df.assign(sentiment=test_df["Text"].apply(get_sentiment))

    # save training and test data in different csv file
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    train_df.to_csv(os.path.join(out_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(out_dir, 'test.csv'), index=False)


if __name__ == "__main__":
    main(opt["--input_file"], opt["--out_dir"])
