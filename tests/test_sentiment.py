from src.sentiment import SentimentAnalyzer


def test_sentiment_score_type():
    analyzer = SentimentAnalyzer()
    assert isinstance(analyzer.score("Hello world"), float)


def test_sentiment_score_bounds():
    analyzer = SentimentAnalyzer()
    score = analyzer.score("I love this!")
    assert -1.0 <= score <= 1.0


def test_non_string_input():
    analyzer = SentimentAnalyzer()
    assert analyzer.score(None) == 0.0
    assert analyzer.score(123) == 0.0
