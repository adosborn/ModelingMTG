from typing import Any
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder

DATA_FILENAME = "cleaned_draft_data_public.WOE.PremierDraft.csv"
EXPANSION = "WOE"

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
) 

class LabelConverter():
    def __init__(self, card_names):
        self.num_cards = len(card_names)
        
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(card_names)

    def to_tensor(self, labels):
        return torch.nn.functional.one_hot(torch.as_tensor(self.label_encoder.transform(labels), dtype=torch.int64), self.num_cards)
    
    def to_labels(self, tensor):
        return self.label_encoder.inverse_transform(torch.argmax(tensor, dim=1).numpy())

class WinrateDataset(Dataset):
    def __init__(self, csv_file, y_label):
        # nrows is temporary
        df = pd.read_csv(csv_file, dtype=defaultdict(lambda: np.uint8, winrate=np.float32, avg_card_winrate=np.float32, W_devotion=np.float32 ,U_devotion=np.float32 ,B_devotion=np.float32 ,R_devotion=np.float32 ,G_devotion=np.float32))

        self.Y = torch.as_tensor(df[y_label].values, dtype=torch.float32, device=device)
        
        df.drop(columns=y_label, inplace=True)
        self.X = torch.as_tensor(df.values, device=device)
    
    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        return self.X[idx].to(dtype=torch.float32), self.Y[idx]

class DraftDataset(Dataset):
    def __init__(self, csv_file):
        # nrows is temporary
        df = pd.read_csv(csv_file, dtype=defaultdict(lambda: np.uint8, pick=str))

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit([name[10:] for name in df.columns if name.startswith("pack_card_")])
        self.Y = torch.as_tensor(self.label_encoder.transform(df['pick'].values), dtype=torch.int64, device=device)
        
        df.drop(columns='pick', inplace=True)
        self.X = torch.as_tensor(df.values, device=device)
    
    def __len__(self):
        return len(self.Y)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        
        return self.X[idx].to(dtype=torch.float32), self.Y[idx]
    
def fix_oopsie():
    df = pd.read_csv("winrate_testing_data.csv")
    fixed = df.dropna()
    fixed.to_csv("fixed_winrate_testing_data.csv", index=False)

def main():
    # lc = LabelConverter()
    # tensor = lc.to_tensor(['Sweettooth Witch', 'Hamlet Glutton'])
    # labels = lc.to_labels(tensor)
    # print(tensor)
    # print(labels)
    # dataset = DraftDataset(DATA_FILENAME)
    # dataloader = DataLoader(dataset)
    # for x, y in dataloader:
    #     print(x, y)
    #     break
    # fix_oopsie()
    return


if __name__ == "__main__":
    main()