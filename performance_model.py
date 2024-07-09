from torch.utils.data import DataLoader
from data_loaders import HumanPickDataset, PoolRecordDataset, FuzzyPoolRecordDataset, DeckClassifierDataset, DeckClassifierWithRankDataset, DeckClassifierWithRankAndWinrateDataset
import set_info as info
import torch
from models import models

def make_classification_tensor(pool):
    tensor = torch.zeros(info.CMC_BUCKETS + info.NUM_COLORS + info.CARDS_IN_SET, dtype=torch.float32, device=info.DEVICE)
    for cardname in pool:
        tensor[info.CMC_BUCKETS + info.NUM_COLORS + info.CARD_TO_ID[cardname]] += 1
        tensor[min(info.CARD_DATA[cardname]['mana_value'], 8)] += 1
        color_identity = info.CARD_DATA[cardname]['color_identity']
        if 'W' in color_identity: tensor[info.CMC_BUCKETS + 0] += 1
        if 'U' in color_identity: tensor[info.CMC_BUCKETS + 1] += 1
        if 'B' in color_identity: tensor[info.CMC_BUCKETS + 2] += 1
        if 'R' in color_identity: tensor[info.CMC_BUCKETS + 3] += 1
        if 'G' in color_identity: tensor[info.CMC_BUCKETS + 4] += 1
    return tensor.unsqueeze(0)

def print_deck_classifications(model):
    dataset = DeckClassifierDataset(f'clean_datasets/deck_validation_data.0.WOE.csv')

    X, y = dataset[7]
    pred = torch.exp(model(X.unsqueeze(0)))

    for card_idx, amount in enumerate(X[14:]):
        if amount > 0:
            print(f'{int(amount)} {info.ID_TO_CARD[card_idx]}')
    print(pred.squeeze())
    print(y)

def get_card_to_pick(model, pack, pool):
    for i, card in enumerate(pack):
        pred = torch.exp(model(make_classification_tensor([card] + pool))).squeeze()
        
        # score = pred[2]
        # if i == 0 or score > best_score:
        #     best_score = score
        #     card_to_pick = card

        # score = pred[0]
        # if i == 0 or score < best_score:
        #     best_score = score
        #     card_to_pick = card

        score = pred[2] - pred[0]
        if i == 0 or score > best_score:
            best_score = score
            card_to_pick = card

    return card_to_pick

def get_pack_and_pool(x_tensor):
    #x_tensor.squeeze()
    pack = []
    for card_idx, amount in enumerate(x_tensor[2:2+info.CARDS_IN_SET]):
        for i in range(int(amount)):
            pack.append(info.ID_TO_CARD[card_idx])
    pool = []
    for card_idx, amount in enumerate(x_tensor[2+info.CARDS_IN_SET:]):
        for i in range(int(amount)):
            pool.append(info.ID_TO_CARD[card_idx])
    
    return pack, pool


def compare_with_humans(model, dataloader):
    total_samples, accuracy = 0, 0
    for X, Y in dataloader:
        for x, y in zip(X, Y):
            pack, pool = get_pack_and_pool(x)
            pick = get_card_to_pick(model, pack, pool)
            if pick == info.ID_TO_CARD[y.item()]:
                accuracy += 1
            total_samples += 1
            #print(f'Pick: {pick} (Human Pick: {info.ID_TO_CARD[y.item()]})')
    print(f'Accuracy: {100 * accuracy / total_samples:0.2f}%')


def main():
    model = models['deck_classifier_batch128_wide']
    model.load_model()

    # dataset = HumanPickDataset('clean_datasets/validation_data.0.WOE.csv', nrows=1000)
    # for i in range(10):
    #     X, y = dataset[i]
    #     pack, pool = get_pack_and_pool(X)

    #     print(f'Pack: {pack}\nPool: {pool}')
    #     print(f'Pick: {get_card_to_pick(model, pack, pool)} (Human picked: {info.ID_TO_CARD[y.item()]})\n')

    compare_with_humans(model, DataLoader(HumanPickDataset('clean_datasets/validation_data.0.WOE.csv', nrows=1000), batch_size=10))
    
    #print(torch.exp(models['deck_classifier_batch128_regression'](make_classification_tensor(['Gruff Triplets']))))

if __name__ == '__main__':
    main()