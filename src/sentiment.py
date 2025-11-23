
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download("vader_lexicon")

sia = SentimentIntensityAnalyzer()

def sentiment_score(text: str) -> float:
    if not isinstance(text, str):
        return 0.0
    return sia.polarity_scores(text)['compound']
