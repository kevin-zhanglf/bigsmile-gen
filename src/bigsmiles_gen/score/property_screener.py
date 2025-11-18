import argparse, pandas as pd, yaml

def load_rules(path: str):
    with open(path, "r") as f:
        y = yaml.safe_load(f)
    return y.get("rules", {})

def in_range(val, rule) -> bool:
    import pandas as pd
    if pd.isna(val): return False
    mn = rule.get("min", None)
    mx = rule.get("max", None)
    if mn is not None and val < mn: return False
    if mx is not None and val > mx: return False
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="txt file, one BigSMILES per line")
    ap.add_argument("--props", required=True, help="CSV with at least columns: bigsmiles, <prop1>, <prop2>, ...")
    ap.add_argument("--rules", required=True, help="YAML with screening rules (min/max per property)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rules = load_rules(args.rules)
    props = pd.read_csv(args.props)
    props["bigsmiles"] = props["bigsmiles"].astype(str)

    with open(args.input, "r", encoding="utf-8") as f:
        cands = [ln.strip() for ln in f if ln.strip()]

    import pandas as pd
    df = pd.DataFrame({"bigsmiles": cands})
    merged = df.merge(props, on="bigsmiles", how="left")

    mask = pd.Series([True]*len(merged))
    for key, rule in rules.items():
        if key in merged.columns:
            mask = mask & merged[key].apply(lambda v: in_range(v, rule))
        else:
            mask = mask & False

    passed = merged[mask]["bigsmiles"].tolist()
    with open(args.out, "w", encoding="utf-8") as f:
        for s in passed:
            f.write(s+"\n")
    print(f"Screened: {len(passed)}/{len(cands)} passed rules from {args.rules}")

if __name__ == "__main__":
    main()