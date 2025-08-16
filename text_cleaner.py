# text_cleaner.py
from sklearn.base import BaseEstimator, TransformerMixin

class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Basic preprocessing (তুমি চাইলে extra logic দিতে পারো)
        return [x.lower().strip() for x in X]
