import argparse
from typing import Tuple

def has_balanced_brackets(s: str) -> bool:
    pairs = {"{":"}", "[":"]", "(":")"}
    closes = {v:k for k,v in pairs.items()}
    stack = []
    for ch in s:
        if ch in pairs:
            stack.append(ch)
        elif ch in closes:
            if not stack or stack[-1] != closes[ch]:
                return False
            stack.pop()
    return len(stack) == 0

def try_bigsmiles_parse(s: str) -> Tuple[bool, str]:
    try:
        import bigsmiles as bs  # optional dependency
        try:
            bs.parse(s)
        except AttributeError:
            _ = bs.Polymer(s)
        return True, "ok"
    except ImportError:
        return True, "bigsmiles-lib-not-installed"
    except Exception as e:
        return False, f"parse_error: {e}"

def validate_bigsmiles(s: str) -> Tuple[bool, str]:
    if not s or any(c.isspace() for c in s.strip()):
        return False, "empty_or_space"
    if not has_balanced_brackets(s):
        return False, "unbalanced"
    ok, msg = try_bigsmiles_parse(s)
    if not ok:
        return ok, msg
    return True, "ok"

def main():
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="txt file, one BigSMILES per line")
    args = ap.parse_args()

    total, valid = 0, 0
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            total += 1
            ok, _ = validate_bigsmiles(s)
            if ok: valid += 1
    print(f"{valid}/{total} valid")

if __name__ == "__main__":
    main()