import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot
import set_info as info
import pandas as pd

class HumanPickDataset(Dataset):
    def __init__(self, csv_pathname, nrows=None):
        column_names = ['pick', 'pack_number', 'pick_number'] + info.PACK_CARDNAMES + info.POOL_CARDNAMES
        df = pd.read_csv(csv_pathname, nrows=nrows, dtype=info.DTYPES, usecols=column_names)

        # Convert textual pick labels to one-hot encoding
        self.Y = torch.tensor([info.CARD_TO_ID[pick] for pick in df['pick']], device=info.DEVICE)
        
        # Remove pick column from X data and convert it to a tensor
        df.drop(columns='pick', inplace=True)
        self.X = torch.as_tensor(df.to_numpy(), device=info.DEVICE)
    
    def __len__(self):
        return self.Y.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        
        # Set data types to those expected by model and loss function
        # Note we delay the cast to here to save memory by using smaller dtypes
        return self.X[idx].to(dtype=torch.float32), self.Y[idx].to(dtype=torch.long)

class PoolRecordDataset(Dataset):
    def __init__(self, csv_pathname):
        df = pd.read_csv(csv_pathname, dtype=info.DTYPES)

        # Convert win-loss record to one-hot encoding for classification
        self.Y = torch.tensor([self.record_to_id(wins, losses) for wins, losses in zip(df['event_match_wins'], df['event_match_losses'])], device=info.DEVICE)
        
        # Remove pick column from X data and convert it to a tensor
        df.drop(columns=['event_match_wins', 'event_match_losses'], inplace=True)
        self.X = torch.as_tensor(df.to_numpy(), device=info.DEVICE)
    
    def __len__(self):
        return self.Y.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        
        return self.X[idx].to(dtype=torch.float32), self.Y[idx].to(dtype=torch.long)

    def record_to_id(self, wins, losses):
        return wins + 2 if losses == 0 else wins + 1 if losses == 1 else wins
    
    def id_to_record(self, id):
        # Return wins, losses
        return min(7, id), min(3, 9 - id)
    
class FuzzyPoolRecordDataset(Dataset):
    def __init__(self, csv_pathname):
        df = pd.read_csv(csv_pathname, dtype=info.DTYPES)

        # Convert win-loss record to one-hot encoding for classification
        self.Y = torch.tensor([self.record_to_id(wins, losses) for wins, losses in zip(df['event_match_wins'], df['event_match_losses'])], device=info.DEVICE)
        
        # Remove pick column from X data and convert it to a tensor
        df.drop(columns=['event_match_wins', 'event_match_losses'], inplace=True)
        self.X = torch.as_tensor(df.to_numpy(), device=info.DEVICE)
    
    def __len__(self):
        return self.Y.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        
        return self.X[idx].to(dtype=torch.float32), self.Y[idx].to(dtype=torch.long)

    def record_to_id(self, wins, losses):
        return 0 if wins <= 2 else 1 if wins <= 5 else 2
    
class DeckClassifierDataset(Dataset):
    def __init__(self, csv_pathname):
        column_names = ['performance', 'cmc_0', 'cmc_1', 'cmc_2', 'cmc_3', 'cmc_4', 'cmc_5', 'cmc_6', 'cmc_7', 'cmc_8+', 'white_devotion', 'blue_devotion', 'black_devotion', 'red_devotion', 'green_devotion'] + info.CARDNAMES
        df = pd.read_csv(csv_pathname, usecols=column_names, dtype=info.DTYPES)

        # Convert win-loss record to one-hot encoding for classification
        self.Y = torch.tensor(df['performance'], device=info.DEVICE)
        
        # Remove pick column from X data and convert it to a tensor
        df.drop(columns='performance', inplace=True)
        self.X = torch.as_tensor(df.to_numpy(), device=info.DEVICE)
    
    def __len__(self):
        return self.Y.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        
        return self.X[idx].to(dtype=torch.float32), self.Y[idx].to(dtype=torch.long)

class DeckClassifierWithRankDataset(Dataset):
    def __init__(self, csv_pathname):
        column_names = ['performance', 'bronze', 'silver', 'gold', 'platinum', 'diamond', 'mythic', 'cmc_0', 'cmc_1', 'cmc_2', 'cmc_3', 'cmc_4', 'cmc_5', 'cmc_6', 'cmc_7', 'cmc_8+', 'white_devotion', 'blue_devotion', 'black_devotion', 'red_devotion', 'green_devotion'] + info.CARDNAMES
        df = pd.read_csv(csv_pathname, usecols=column_names, dtype=info.DTYPES)

        # Convert win-loss record to one-hot encoding for classification
        self.Y = torch.tensor(df['performance'], device=info.DEVICE)
        
        # Remove pick column from X data and convert it to a tensor
        df.drop(columns='performance', inplace=True)
        self.X = torch.as_tensor(df.to_numpy(), device=info.DEVICE)
    
    def __len__(self):
        return self.Y.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        
        return self.X[idx].to(dtype=torch.float32), self.Y[idx].to(dtype=torch.long)

class DeckClassifierWithRankAndWinrateDataset(Dataset):
    def __init__(self, csv_pathname):
        column_names = ['performance', 'winrate_bucket', 'bronze', 'silver', 'gold', 'platinum', 'diamond', 'mythic', 'cmc_0', 'cmc_1', 'cmc_2', 'cmc_3', 'cmc_4', 'cmc_5', 'cmc_6', 'cmc_7', 'cmc_8+', 'white_devotion', 'blue_devotion', 'black_devotion', 'red_devotion', 'green_devotion'] + info.CARDNAMES
        df = pd.read_csv(csv_pathname, usecols=column_names, dtype=info.DTYPES)

        # Convert win-loss record to one-hot encoding for classification
        self.Y = torch.tensor(df['performance'], device=info.DEVICE)
        
        # Remove pick column from X data and convert it to a tensor
        df.drop(columns='performance', inplace=True)
        self.X = torch.as_tensor(df.to_numpy(), device=info.DEVICE)
    
    def __len__(self):
        return self.Y.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        
        return self.X[idx].to(dtype=torch.float32), self.Y[idx].to(dtype=torch.long)
    
class HumanPickAblation1Dataset(Dataset):
    def __init__(self, csv_pathname):
        column_names = ['pick'] + info.PACK_CARDNAMES + info.POOL_CARDNAMES
        df = pd.read_csv(csv_pathname, dtype=info.DTYPES, usecols=column_names)

        # Convert textual pick labels to one-hot encoding
        self.Y = torch.tensor([info.CARD_TO_ID[pick] for pick in df['pick']], device=info.DEVICE)
        
        # Remove pick column from X data and convert it to a tensor
        df.drop(columns='pick', inplace=True)
        self.X = torch.as_tensor(df.to_numpy(), device=info.DEVICE)
    
    def __len__(self):
        return self.Y.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        
        # Set data types to those expected by model and loss function
        # Note we delay the cast to here to save memory by using smaller dtypes
        return self.X[idx].to(dtype=torch.float32), self.Y[idx].to(dtype=torch.long)

class IncludeCardDataset(Dataset):
    def __init__(self, csv_pathname):
        column_names = ['pick', 'pack_number', 'pick_number'] + info.POOL_CARDNAMES
        df = pd.read_csv(csv_pathname, dtype=info.DTYPES, usecols=column_names)

        # Convert win-loss record to one-hot encoding for classification
        self.Y = torch.tensor([self.record_to_id(wins, losses) for wins, losses in zip(df['event_match_wins'], df['event_match_losses'])], device=info.DEVICE)
        
        # Remove pick column from X data and convert it to a tensor
        df.drop(columns=['event_match_wins', 'event_match_losses'], inplace=True)
        self.X = torch.as_tensor(df.to_numpy(), device=info.DEVICE)
    
    def __len__(self):
        return self.Y.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        
        return self.X[idx].to(dtype=torch.float32), self.Y[idx].to(dtype=torch.long)
    
class SingleIncludeDataset(Dataset):
    def __init__(self, csv_pathname):
        column_names = ['maindeck_rate', 'card'] + info.CARDNAMES
        df = pd.read_csv(csv_pathname, dtype=info.DTYPES, usecols=column_names)

        # Convert win-loss record to one-hot encoding for classification
        self.Y = torch.tensor(df['maindeck_rate'], device=info.DEVICE)

        # Remove pick column from X data and convert it to a tensor
        df.drop(columns='maindeck_rate', inplace=True)
        self.X = torch.as_tensor(df.to_numpy(), device=info.DEVICE)
    
    def __len__(self):
        return self.Y.shape[0]
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        
        return self.X[idx].to(dtype=torch.float32), self.Y[idx].to(dtype=torch.long)

def main():
    dataset = HumanPickDataset('clean_datasets/training_data.0.WOE.csv')
    dataloader = DataLoader(dataset)
    for X, y in dataloader:
        print(f"X: {X}")
        print(f"y: {y}")
    print(len(dataset))
    print(dataset[0][0].size())


if __name__ == '__main__':
    main()