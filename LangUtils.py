has_lang = {
    "zh": lambda x: has_chinese(x),
    "jp": lambda x: has_jp(x),
    "ja": lambda x: has_jp(x),
    "ko": lambda x: has_ko(x),
    "po": lambda x: has_po(x),
    "pt": lambda x: has_po(x),
}


def has_ko(sentence: str) -> bool:
    for c in sentence:
        if u"\u3130" <= c < u"\u318f" or u"\uAC00" <= c < u"\uD7A3":
            return True
    return False


def has_jp(sentence: str) -> bool:
    for c in sentence:
        if u"\u0800" <= c < u"\u9fff":
            return True
    return False


def has_po(sentence: str) -> bool:
    for c in sentence:
        if u"\u0080" <= c < u"\u00ff":
            return True
    return False


def has_chinese(sentence: str) -> bool:
    for c in sentence:
        if u"\u4e00" <= c < u"\u9fff":
            return True
    return False
