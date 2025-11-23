from src.sentiment import sentiment_score


def test_sentiment_basic():
    assert sentiment_score("Stock rises after earnings") > 0
    assert sentiment_score("Stock crashes and burns") < 0
