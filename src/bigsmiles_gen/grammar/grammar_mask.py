from typing import List, Dict, Set
import torch

class GrammarMasker:
    """
    Lightweight grammar/logits masker:
    - enforce balanced {}, [], ()
    - disallow closing bracket when not open
    - block EOS if there are unmatched opens
    - optionally enforce that first token is not a closing symbol
    """

    def __init__(self, token_id_map: Dict[str, int], eos_token_id: int):
        self.id2sym = {v: k for k, v in token_id_map.items()}
        self.sym2id = token_id_map
        self.eos = eos_token_id
        self.opens = {"{": "}", "[": "]", "(": ")"}
        self.closes = {v: k for k, v in self.opens.items()}
        self.close_ids: Set[int] = {self.sym2id[s] for s in self.closes if s in self.sym2id}
        self.open_ids: Set[int] = {self.sym2id[s] for s in self.opens if s in self.sym2id}

    def stack_from_tokens(self, token_ids: List[int]) -> List[str]:
        stack: List[str] = []
        for tid in token_ids:
            s = self.id2sym.get(tid)
            if s in self.opens:
                stack.append(s)
            elif s in self.closes:
                if stack and stack[-1] == self.closes[s]:
                    stack.pop()
                else:
                    # already invalid; represent as sentinel impossible to close
                    stack.append("#")
        return stack

    def mask(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        input_ids: [B, T]
        scores: [B, V]
        returns masked scores (logits)
        """
        B, V = scores.shape
        masked = scores.clone()
        for b in range(B):
            toks = input_ids[b].tolist()
            stack = self.stack_from_tokens(toks)
            if stack:
                # mask EOS if bracket stack not empty
                masked[b, self.eos] = float("-inf")
                # disallow closing brackets not matching top-of-stack
                top = stack[-1]
                allowed_close = self.opens[top] if top in self.opens else None
                for cid in self.close_ids:
                    sym = self.id2sym.get(cid)
                    if sym != allowed_close:
                        masked[b, cid] = float("-inf")
            else:
                # first token can't be closing
                if len(toks) <= 2:  # assuming tokenizer added BOS + maybe prefix
                    for cid in self.close_ids:
                        masked[b, cid] = float("-inf")
        return masked