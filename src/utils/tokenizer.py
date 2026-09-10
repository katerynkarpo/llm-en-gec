from spacy.lang.en import English


_nlp = None


def _get_nlp():
    """Lazily initialize the English spaCy tokenizer."""
    global _nlp
    if _nlp is None:
        _nlp = English()
    return _nlp


def tokenize(text: str) -> str:
    """Tokenize English text using spaCy's tokenization rules."""
    return " ".join(token.text for token in _get_nlp()(text))
