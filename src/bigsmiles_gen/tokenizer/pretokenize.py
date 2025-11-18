from typing import Iterable

SPECIALS = list("{}[]().-=#><$*:/+\\")

def pretokenize(text: str) -> str:
    """
    Lightweight pre-tokenization:
    - Surround special symbols with spaces so sentencepiece learns them as atomic units
    - Collapse multiple spaces
    """
    for ch in SPECIALS:
        text = text.replace(ch, f" {ch} ")
    text = " ".join(text.split())
    return text

def stream_pretokenized(lines: Iterable[str]) -> Iterable[str]:
    for s in lines:
        yield pretokenize(s)