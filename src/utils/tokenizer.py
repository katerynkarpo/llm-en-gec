from spacy.lang.en import English


_nlp = None


def _get_nlp():
    """Lazily initialize the English spaCy tokenizer."""
    global _nlp
    if _nlp is None:
        _nlp = English()
        _nlp.add_pipe("sentencizer")
    return _nlp


def tokenize(text: str) -> str:
    """Tokenize English text using spaCy's tokenization rules."""
    return " ".join(token.text for token in _get_nlp()(text))


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return sentence character spans using spaCy's rule-based sentencizer."""
    if not text.strip():
        return []
    spans: list[tuple[int, int]] = []
    for sentence in _get_nlp()(text).sents:
        start, end = sentence.start_char, sentence.end_char
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            spans.append((start, end))
    return spans
