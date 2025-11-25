# src/sentiment.py

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize analyzer once
_analyzer = None


def _get_analyzer():
    """Get or create the sentiment analyzer."""
    global _analyzer
    if _analyzer is None:
        nltk.download("vader_lexicon", quiet=True)
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


class SentimentAnalyzer:
    """
    Wrapper around NLTK's VADER SentimentIntensityAnalyzer
    to compute sentiment scores for text.
    """

    def __init__(self):
        """
        Initialize the VADER sentiment analyzer and download necessary lexicon.
        """
        nltk.download("vader_lexicon", quiet=True)
        self.sia = SentimentIntensityAnalyzer()

    def score(self, text: str) -> float:
        """
        Compute the compound sentiment score for a given text.

        Args:
            text (str): The text to score.

        Returns:
            float: Sentiment score between -1 (negative) and 1 (positive).
                   Returns 0.0 for non-string inputs.
        """
        if not isinstance(text, str):
            return 0.0
        return self.sia.polarity_scores(text)['compound']


def sentiment_score(text: str) -> float:
    """
    Compute the compound sentiment score for a given text.
    Convenience function that uses a global analyzer.

    Args:
        text (str): The text to score.

    Returns:
        float: Sentiment score between -1 (negative) and 1 (positive).
               Returns 0.0 for non-string inputs.
    """
    if not isinstance(text, str):
        return 0.0
    analyzer = _get_analyzer()
    return analyzer.polarity_scores(text)['compound']
