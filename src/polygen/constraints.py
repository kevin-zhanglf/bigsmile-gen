from __future__ import annotations
from typing import List, Set, Optional
from dataclasses import dataclass

@dataclass
class BracketState:
    paren: int = 0    # ()
    brace: int = 0    # {}
    bracket: int = 0  # []

    def update(self, token: str):
        if token == "(":
            self.paren += 1
        elif token == ")":
            self.paren = max(0, self.paren - 1)
        elif token == "{":
            self.brace += 1
        elif token == "}":
            self.brace = max(0, self.brace - 1)
        elif token == "[":
            self.bracket += 1
        elif token == "]":
            self.bracket = max(0, self.bracket - 1)

    def allowed_next(self, vocab
