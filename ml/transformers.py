from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def text_transformer(tfidf=True):
    """This function generates a column transformer to pass from raw text to
    count vectorizer or tfid vectorizer."""

    if tfidf:
        ct = ColumnTransformer(
          [("text_preprocess", TfidfVectorizer(stop_words = "english",
                                               analyzer = 'word',
                                               lowercase = True,
                                               use_idf = True), "text")], remainder="passthrough")
    else:
        ct = ColumnTransformer(
            [("text_preprocess", CountVectorizer(stop_words = "english",
                                      analyzer = 'word',
                                      lowercase = True), "text")], remainder="passthrough")

    return ct
