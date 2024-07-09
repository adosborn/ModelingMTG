import os
import torch
import data_loader as dl
import read_data as rd
import torchvision.models as models
import set_info as info
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils import get_col_names


#find correct divice (cuda or cpu on windows and mps (metal) on mac)
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
) 

#dictionaries for model and dataset names (fix eval_names, its horrible programming)
model_names = {
    "long_model": 'models/long_model_weights.pth',
    "fast_model": 'models/old/baseline.pth',
    "linear_winrate_model": 'models/baseline_winrate.pth',
    "first_winrate_model": 'models/winrate_shuffled.pth',
    "winrate_with_dropout": 'models/winrate_with_dropout.pth',
    "winrate_small_net": 'models/winrate_small_net.pth',
    "winrate_deep": 'models/winrate_deep.pth'
}
dataset_names = {
    "good_player": 'clean_datasets/good_player_cleaned_draft_data_public.WOE.PremierDraft.csv',
    "bad_palyer" : 'clean_datasets/bad_player_cleaned_draft_data_public.WOE.PremierDraft.csv',
    "winrate_testing": 'clean_datasets/fixed_winrate_testing_data.csv',
    "winrate_training": 'clean_datasets/fixed_winrate_training_data.csv',
    "winrate_validation": 'clean_datasets/fixed_winrate_validation_data.csv',
    "full_training": 'clean_datasets/training_data.csv',
    "full_testing": 'clean_datasets/testing_data.0.WOE.csv',
    "full" : 'clean_datasets/og_cleaned_draft_data_public.WOE.PremierDraft.csv'
}
eval_names = {
    'basic':'basic','winrate':'winrate','skill':'skill', 'single':'single'
}
# Models
baseline_structure = nn.Sequential(
        nn.Linear(650, 512, False),
        nn.ReLU(),
        nn.Linear(512, 512, False),
        nn.ReLU(),
        nn.Linear(512, 324, False),
        nn.LogSoftmax(dim=1)
)
good_baseline = nn.Sequential(
        nn.Linear(650, 775, False),
        nn.ReLU(),
        nn.Linear(775, 324, False),
        nn.LogSoftmax(dim=1)
)
wide_structure = nn.Sequential(
    nn.Linear(650, 375, False),
    nn.ReLU(),
    nn.Linear(375, 375, False),
    nn.ReLU(),
    nn.Linear(375, 375, False),
    nn.ReLU(),
    nn.Linear(375, 375, False),
    nn.ReLU(),
    nn.Linear(375, 375, False),
    nn.ReLU(),
    nn.Linear(375, 324, False),
    nn.LogSoftmax(dim=1)
)
winrate_structure = nn.Sequential(
    nn.Linear(dl.CARDS_IN_SET + 7 - 5 - 1, 1000, False),
    nn.ReLU(),
    nn.Linear(1000, 700, False),
    nn.ReLU(),
    nn.Linear(700, 1, False)
)
winrate_structure_dropout = nn.Sequential(
    nn.Linear(dl.CARDS_IN_SET + 7 - 5 - 1, 1000, False),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(1000, 700, False),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(700, 1, False)
)
winrate_structure_deep = nn.Sequential(
    nn.Linear(dl.CARDS_IN_SET + 7 - 5 - 1, 500, False),
    nn.ReLU(),
    nn.Linear(500, 500, False),
    nn.ReLU(),
    nn.Linear(500, 500, False),
    nn.ReLU(),
    nn.Linear(500, 500, False),
    nn.ReLU(),
    nn.Linear(500, 500, False),
    nn.ReLU(),
    nn.Linear(500, 1, False)
)
wide_structure_deep = nn.Sequential(
    nn.Linear(dl.CARDS_IN_SET + 7 - 5 - 1, 500, False),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(500, 500, False),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(500, 500, False),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(500, 500, False),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(500, 500, False),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(500, 1, False)
)
wide_structure_deep_no_final_dropout = nn.Sequential(
    nn.Linear(dl.CARDS_IN_SET + 7 - 5 - 1, 500, False),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(500, 500, False),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(500, 500, False),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(500, 500, False),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(500, 500, False),
    nn.ReLU(),
    nn.Linear(500, 1, False)
)
linear_winrate = nn.Sequential(
    nn.Linear(dl.CARDS_IN_SET + 7 - 5 - 1, 1, False)
)
winrate_small_net = nn.Sequential(
    nn.Linear(dl.CARDS_IN_SET + 7 - 5 - 1, 100, False),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(100, 100, False),
    nn.ReLU(),
    nn.Dropout(),
    nn.Linear(100, 100, False),
    nn.ReLU(),
    nn.Linear(100, 1, False)
)

class NeuralNetwork(nn.Module):
    def __init__(self, net_structure):
        super().__init__()
        self.net_structure = net_structure

    def forward(self, X):
        logits = self.net_structure(X)
        return logits
    
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        y = torch.reshape(y, (y.shape[0], 1))
        X.to(device)
        y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)


        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        batch_print_frequency = 10000
        if batch % batch_print_frequency == 0: 
            loss, current = loss.item(), (batch+1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0,0 

    with torch.no_grad():
        for X, y in dataloader:
            y = torch.reshape(y, (y.shape[0], 1))
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    #correct /= size
    #print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"Avg validation loss: {test_loss:>8f} \n")
    with open("log.txt", "a") as file:
        file.write(f"Avg validation loss: {test_loss}\n")

def rsquare(Y, Yhat):
    """
    Implementation of the R squared metric based on ground-truth Y and predicted Y
    """

    # print(Y)
    # print(Yhat)
    deviation_from_mean = Y - torch.mean(Y)
    deviation_from_model = Y - Yhat

    return 1 - torch.sum(deviation_from_model ** 2) / torch.sum(deviation_from_mean ** 2)

def evaluate(model, data, name):
    model.load_state_dict(torch.load(name))
    model.eval()
    # print(model.state_dict())
    
    for i in range(100):
        x,y = data[i]
        x = torch.reshape(x, (1,650))
        card_taken = y.item()
        card_suggested = torch.argmax(model(x), dim=1).item()
        # for card in range(dl.CARDS_IN_SET):
        #     if x[card].item() == 1:
        #         print(data.label_encoder.inverse_transform([x[card].item()]))
        cards_in_pack = x
        print(data.label_encoder.inverse_transform([card_suggested, card_taken]))

def evaluate_winrate(model, dataset, name):
    model.load_state_dict(torch.load(name))
    model.eval()
    # print(model.state_dict())
    X, y = dataset[:]
    predY = model(X).squeeze()
    # for y1, y2 in zip(y, predY):
    #     print(f"{y1} {y2}")
    print(f"R^2: {rsquare(y, predY)}")

def evaluate_skill_comparison(model, data, name, good_at_the_game = True):
    dataloader = rd.DataLoader(data)
    model.load_state_dict(torch.load(name))
    model.eval()
    # print(model.state_dict())
    index = 0.0
    count = 0.0
    for X,y in dataloader:
        card_suggested = torch.argmax(model(X), dim=1).item()
        card_taken = y.item()
        if (card_suggested == card_taken): 
            count += 1.0 
        index += 1.0
        if (index % 1000.0 == 0.0):
            print(f"{100*count/(count+index):>8f}% accuracy on {('good player set' if good_at_the_game else 'bad player set')} with i of {index}\n")

def evaluate_single(model, pack, name):

    print(model)
    dataloader = rd.DataLoader(pack)
    model.load_state_dict(torch.load(name, map_location=device))
    model.eval()
    for X,y in dataloader:
        tensor = model(X)
        pack_tensor = X[:,2:2+info.CARDS_IN_SET]>0
        card_suggested = torch.argmax(pack_tensor*torch.exp(tensor), dim=1).item()
        card_to_pick = pack.label_encoder.inverse_transform([card_suggested])

        return card_to_pick



