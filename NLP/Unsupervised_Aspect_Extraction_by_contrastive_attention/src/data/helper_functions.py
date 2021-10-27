import nltk


def tokenize(s: str) -> str:
    """Tokenize a string"""
    # NLTK seams to be slightly faster at tokenization:
    # https://towardsdatascience.com/benchmarking-python-nlp-tokenizers-3ac4735100c5
    # Alternatively spaCy can be used as well.
    tokenized = nltk.word_tokenize(s)
    tokenized = [w.lower().strip() for w in tokenized]
    tokenized_string = ' '.join(tokenized)
    return tokenized_string


def strip_lower_tokenize(s: str) -> str:
    """Tokenize, strip and lowercase a string"""
    s = s.replace('\n', ' ')
    s = s.strip()
    s = s.lower()
    return tokenize(s)
