# Data schema

## bigsmiles.csv
- `bigsmiles`: string, BigSMILES of polymer
- optional: `cond_prefix`: string, any control tags to prepend during training (e.g., `<Tg_bin_3> <dielectric_A>`)

You may store additional columns; they will be ignored by the LM pipeline.

## properties.csv (optional, for screening)
- `bigsmiles`: string, exact match to join
- numerical columns for properties, e.g., `Tg`, `dielectric`, `barrier_O2`, etc.

## Typical pipeline
- Deduplicate `bigsmiles` (normalized canonicalization recommended)
- Filter extremely long sequences for a minimal baseline (e.g., length <= 1024 tokens)