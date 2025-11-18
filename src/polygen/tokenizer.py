from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import List, Iterable, Dict, Tuple
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing

BIGSMILES_SPECIALS = [
    "{", "}", "[", "]", "(", ")", ".", "-", "=", "#", ":", "/", "\\",
    "*", ",", ";", "@", "+", "-", "%", "<", ">", "|"
]

ATOM_REGEX = r"Cl|Br|Si|Se|Na|Li|Al|Ca|Fe|Mg|Zn|Ti|Ag|Au|Pt|Pd|B|C|N|O|F|P|S|I|H|K|V|W|U|Y|Zr|Rb|Cs|Sr|Ba|Hg|Ni|Co|Cu|Mn|Mo|Cr|As|Te|Ga|Ge|Sn|Sb|Bi|Xe|Kr|Ne|Ar|He"

TOKEN_PATTERN = re.compile(
    rf"({ATOM_REGEX}|\d+|\{{|\}}|\[|\]|\(|\)|\.|\-|\=|\#|:|\/|\\|\*|,|;|@|\+|%|<|>|\|)"
)

def basic_pretokenize(text: str) -> List[str]:
    # Keep order, split but keep tokens
    tokens: List[str] = []
    idx = 0
    while idx < len(text):
        m = TOKEN_PATTERN.match(text, idx)
        if m:
            if m.start() > idx:
                tokens.append(text[idx:m.start()])
            tokens.append(m.group(0))
            idx = m.end()
        else:
            # single char fallback
            tokens.append(text[idx])
            idx += 1
    # remove whitespaces
    tokens = [t for t in tokens if t.strip() != ""]
    return tokens

class BigSMILESPretok(PreTokenizer):
    def pre_tokenize_str(self, text: str):
        toks = basic_pretokenize(text)
        offsets = []
        cursor = 0
        res = []
        for t in toks:
            # find next occurrence
            start = text.find(t, cursor)
            if start < 0:
                start = cursor
            end = start + len(t)
            cursor = end
            res.append((t, (start, end)))
        return res

@dataclass
class BigSMILESTokenizerConfig:
    vocab_size: int = 2000
    min_frequency: int = 2
    special_tokens: Tuple[str, ...] = ("<pad>", "<bos>", "<eos>", "<unk>")

class BigSMILESTokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self.tk = tokenizer

    @classmethod
    def train_from_texts(
        cls,
        texts: Iterable[str],
        cfg: BigSMILESTokenizerConfig = BigSMILESTokenizerConfig(),
    ) -> "BigSMILESTokenizer":
        model = BPE(unk_token="<unk>")
        tokenizer = Tokenizer(model)
        tokenizer.pre_tokenizer = BigSMILESPretok()
        trainer = BpeTrainer(
            vocab_size=cfg.vocab_size,
            min_frequency=cfg.min_frequency,
            special_tokens=list(cfg.special_tokens),
        )
        tokenizer.train_from_iterator(texts, trainer=trainer)
        tokenizer.post_processor = TemplateProcessing(
            single="<bos> $A <eos>",
            pair="<bos> $A <eos> $B:1 <eos>:1",
            special_tokens=[("<bos>", tokenizer.token_to_id("<bos>")),
                            ("<eos>", tokenizer.token_to_id("<eos>"))]
        )
        return cls(tokenizer)

    def save(self, path: str):
        self.tk.save(path)

    @classmethod
    def load(cls, path: str) -> "BigSMILESTokenizer":
        return cls(Tokenizer.from_file(path))

    def encode(self, text: str) -> List[int]:
        return self.tk.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        return self.tk.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self.tk.get_vocab_size()

    def bos_id(self) -> int:
        return self.tk.token_to_id("<bos>")

    def eos_id(self) -> int:
        return self.tk.token_to_id("<eos>")

    def pad_id(self) -> int:
        return self.tk.token_to_id("<pad>")
