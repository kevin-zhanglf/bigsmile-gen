from typing import Optional, Dict
import pandas as pd
from torch.utils.data import Dataset

class BigSMILESTextDataset(Dataset):
    def __init__(self, path: str, text_column: str = "bigsmiles", prefix_column: Optional[str] = None):
        self.df = pd.read_csv(path)
        self.text_col = text_column
        self.pref_col = prefix_column

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        row = self.df.iloc[idx]
        txt = str(row[self.text_col])
        if self.pref_col and self.pref_col in row and isinstance(row[self.pref_col], str):
            txt = f"{row[self.pref_col].strip()} {txt}"
        return {"text": txt}