import pandas as pd
import numpy as np
import torch
import os
import read_data as rd
from torchvision import transforms, utils
from torch.utils.data import DataLoader, Dataset

CARDS_PER_DRAFT     = 42
CARDS_IN_SET        = 329 # 63 enchanting tales, 266 regular, still has the 5 lands
NUMBER_OF_SAD_CARDS = 5 # :(

# some code adapted from this tutorial: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# this is an implimentation of the abstract class dataset overriding relevent meathods
class DraftDataset(Dataset):
    def __init__(self, csv_file, transform=None, start=0, len=10, convert_to_one_hot=False):
        """
        Arguments:
            csv_file (string): Path to the csv file of draft data.
        """
        # need weirdness since chunksize makes the datatype a 'textfilereader' and need to convert back to dataframe
        # this is currently only reading in 10 drafts. Need to come up with a solution for reading in a bunch
        # here's a helpful stackoverflow post: https://stackoverflow.com/questions/39386458/how-to-read-data-in-python-dataframe-without-concatenating
        self.transform  = transform
        self.one_hot    = convert_to_one_hot
        self.draft_data = pd.DataFrame(pd.read_csv(csv_file, skiprows=start, chunksize=CARDS_PER_DRAFT*len).get_chunk(CARDS_PER_DRAFT*len))
        self.headers    = self.draft_data.head()
    
    def __len__(self):
        return len(self.draft_data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.toList()
        # wins     = self.draft_data.iloc[idx, 5] 
        # losses   = self.draft_data.iloc[idx, 6]
        pack_num = self.draft_data.iloc[idx, 0] # was 7
        pick_num = self.draft_data.iloc[idx, 1] # was 8
        pick     = self.draft_data.iloc[idx, 2] # was 9. this is in string form. Need to map it to int or something
        # these do not include sad cards
        in_pack  = self.draft_data.iloc[idx, 3:2+CARDS_IN_SET] # was 12,11 prefaced with pack_card_
        in_pool  = self.draft_data.iloc[idx, 3+CARDS_IN_SET:2+(2*CARDS_IN_SET)] # was 12,11 prefaced with pool_
        # print(self.draft_data.columns[2+CARDS_IN_SET], self.draft_data.columns[3+CARDS_IN_SET], self.draft_data.columns[2+(2*CARDS_IN_SET)])
        # some var for cards in pack, array of ints where ints can index into a dict?
        # some var for cards in deck so far, array of ints where ints can index into a dict?
        in_pack = np.array([in_pack], dtype=float)
        in_pool = np.array([in_pool], dtype=float)
        # ... (more vars as nessesary)
        sample = {#'wins':     wins,     'losses':   losses,
                  'pack_num': pack_num, 'pick_num': pick_num, 
                  'pick':     pick,     'in_pack': in_pack, 
                  'in_pool':  in_pool}

        if self.transform: 
            sample = self.transform(sample)
        if self.one_hot:
            lc = rd.LabelConverter()
            pick = lc.to_tensor(pick).float
        label = pick
        # return sample
        # print(in_pack)
        # print(pack_num)
        pack_data = np.concatenate((pack_num,
                              pick_num,
                              in_pack,
                              in_pool), axis = None) 
        # print(pack_data)
        return torch.from_numpy(pack_data), label
    
# implimentation mostly from this tutorial: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        #wins,     losses         = sample['wins'],     sample['losses']
        pack_num, pick_num, pick = sample['pack_num'], sample['pick_num'], sample['pick']
        in_pack,  in_pool        = sample['in_pack'],  sample['in_pool']
        # print("success")
        #can modify values here is needed - may need to when dealing with more complex data types
        #torch.from_numpy need to take np.ndarray so only nessesary for arrays (tensors need to be more than just one)
        return {#'wins':     wins,
                #'losses':   losses, 
                'pack_num': pack_num,
                'pick_num': pick_num,
                'pick':     pick,
                'in_pack':  torch.from_numpy(in_pack),
                'in_pool':  torch.from_numpy(in_pool)}


# building dataset
def torch_dataset_test(toprint=False, start_pos=0, len=10):
    # draft_dataset = DraftDataset(csv_file='draft_data_public.WOE.PremierDraft.csv')
    # draft_dataset = DraftDataset(csv_file='cleaned_draft_data_public.WOE.PremierDraft.csv')
    # if added another transform, can format like transform = transforms.Compose([x, y])
    tensor_draft_dataset = DraftDataset(csv_file='cleaned_draft_data_public.WOE.PremierDraft.csv', transform=ToTensor(), start=start_pos, len=len, convert_to_one_hot=False)
    
    drafts_to_itterate_through = 3
    dataloader = DataLoader(tensor_draft_dataset, batch_size = 1, shuffle=False, num_workers=0)
    if(toprint): print_itteration(dataloader, drafts_to_itterate_through)
    return dataloader

def print_itteration(dataloader, drafts_to_itterate_through):
    
    # -1 to compensate for 0 indexed
    how_many_to_itterate_through = drafts_to_itterate_through*(CARDS_PER_DRAFT)-1

    draft_number = 0
    # tensor itteration
    print("\n tensor implimentation \n")
    print(dataloader)
    print(next(iter(dataloader)))
    i = 0
    lc = rd.LabelConverter()
    # for itr in next(iter(dataloader)):
    #     i+=1
    #     print(itr)
    for i, (features, labels) in enumerate(dataloader):
        print(i)
        print(f"Feature batch: {features}")
        print(f"Labels batch: {labels}")

        one_hot_label = lc.to_tensor(labels)
        print(one_hot_label)
    # for i, sample in enumerate(dataloader):
    #     print(sample)
    #     if i%CARDS_PER_DRAFT == 0: 
    #         draft_number += 1
    #     print("draft #", draft_number, 
    #           #sample['wins'], "wins",
    #           #sample['losses'], "losses", 
    #           "pack", sample['pack_num'].item() + 1,
    #           "pick", sample['pick_num'].item() + 1,
    #           "took:", ''.join(sample['pick']), 
    #           "in pack:", torch.sum(sample['in_pack']).item(),
    #           "in pool:", torch.sum(sample['in_pool']).item())
        # if i == how_many_to_itterate_through:
        #     break
    # dataframe itteration
    # print("\n dataframe implimentation \n")
    # for i, sample in enumerate(draft_dataset):
    #     # +1's are for readability since 0-indexed in data
    #     if i%CARDS_PER_DRAFT == 0: 
    #         draft_number += 1
    #     print("draft #", draft_number, 
    #           #sample['wins'], "wins",
    #           #sample['losses'], "losses", 
    #           "pack", sample['pack_num'] + 1,
    #           "pick", sample['pick_num'] + 1,
    #           "took", sample['pick'])
    #     if i == how_many_to_itterate_through:
    #         break

def read_draft_data():
    drafts_to_read = 2 #const
    with pd.read_csv("draft_data_public.WOE.PremierDraft.csv", chunksize=CARDS_PER_DRAFT) as data:
        i = 0
        for chunk in data:
            print(chunk)
            if i == drafts_to_read:
                break
            i += 1

# For testing stuff
def main():
    # read_draft_data()
    torch_dataset_test(toprint=True)

if __name__ == "__main__":
    main()