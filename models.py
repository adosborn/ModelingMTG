from typing import Any
from neural_net import NeuralNetwork, MLP
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from data_loaders import HumanPickDataset, PoolRecordDataset, FuzzyPoolRecordDataset, DeckClassifierDataset, DeckClassifierWithRankDataset, DeckClassifierWithRankAndWinrateDataset, HumanPickAblation1Dataset
import set_info as info
from utils import convert_for_mathematica

networks = {
    'baseline_regression': NeuralNetwork(nn.Sequential(
        # Input features are pack cards, pool cards, pack num, and pick num
        nn.Linear(2 * info.CARDS_IN_SET + 2, info.CARDS_IN_SET, False),
        nn.LogSoftmax(dim=1))).to(device=info.DEVICE),
    'wide_but_shallow': NeuralNetwork(nn.Sequential(
        # Input features are pack cards, pool cards, pack num, and pick num
        nn.Linear(2 * info.CARDS_IN_SET + 2, 775, False),
        nn.ReLU(),
        nn.Linear(775, info.CARDS_IN_SET, False),
        nn.LogSoftmax(dim=1))).to(device=info.DEVICE),
    'wide_but_shallow_no_pack_pick_nums': NeuralNetwork(nn.Sequential(
        # Input features are pack cards, pool cards, pack num, and pick num
        nn.Linear(2 * info.CARDS_IN_SET, 775, False),
        nn.ReLU(),
        nn.Linear(775, info.CARDS_IN_SET, False),
        nn.LogSoftmax(dim=1))).to(device=info.DEVICE),
    'narrow_but_deep': NeuralNetwork(nn.Sequential(
        nn.Linear(2 * info.CARDS_IN_SET + 2, 325, False),
        nn.ReLU(),
        nn.Linear(325, 325, False),
        nn.ReLU(),
        nn.Linear(325, 325, False),
        nn.ReLU(),
        nn.Linear(325, 325, False),
        nn.ReLU(),
        nn.Linear(325, 325, False),
        nn.ReLU(),
        nn.Linear(325, info.CARDS_IN_SET, False),
        nn.LogSoftmax(dim=1))).to(device=info.DEVICE),
    '2_hidden_layers': NeuralNetwork(nn.Sequential(
        nn.Linear(2 * info.CARDS_IN_SET + 2, 500, False),
        nn.ReLU(),
        nn.Linear(500, 500, False),
        nn.ReLU(),
        nn.Linear(500, info.CARDS_IN_SET, False),
        nn.LogSoftmax(dim=1))).to(device=info.DEVICE),
    'winrate_linear_regression': NeuralNetwork(nn.Sequential(
        nn.Linear(info.CARDS_IN_SET, 10, False),
        nn.LogSoftmax(dim=1))).to(device=info.DEVICE),
    'winrate_wide': NeuralNetwork(nn.Sequential(
        nn.Linear(info.CARDS_IN_SET, 400, False),
        nn.ReLU(),
        nn.Linear(400, 10, False),
        nn.LogSoftmax(dim=1))).to(device=info.DEVICE),
    'winrate_deep': NeuralNetwork(nn.Sequential(
        nn.Linear(info.CARDS_IN_SET, 100, False),
        nn.ReLU(),
        nn.Linear(100, 100, False),
        nn.ReLU(),
        nn.Linear(100, 50, False),
        nn.ReLU(),
        nn.Linear(50, 50, False),
        nn.ReLU(),
        nn.Linear(50, 50, False),
        nn.ReLU(),
        nn.Linear(50, 50, False),
        nn.ReLU(),
        nn.Linear(50, 50, False),
        nn.ReLU(),
        nn.Linear(50, 50, False),
        nn.ReLU(),
        nn.Linear(50, 50, False),
        nn.ReLU(),
        nn.Linear(50, 50, False),
        nn.ReLU(),
        nn.Linear(50, 10, False),
        nn.LogSoftmax(dim=1))).to(device=info.DEVICE),
    'winrate_big': NeuralNetwork(nn.Sequential(
        nn.Linear(info.CARDS_IN_SET, 400, False),
        nn.ReLU(),
        nn.Linear(400, 400, False),
        nn.ReLU(),
        nn.Linear(400, 400, False),
        nn.ReLU(),
        nn.Linear(400, 400, False),
        nn.ReLU(),
        nn.Linear(400, 400, False),
        nn.ReLU(),
        nn.Linear(400, 400, False),
        nn.ReLU(),
        nn.Linear(400, 400, False),
        nn.ReLU(),
        nn.Linear(400, 10, False),
        nn.LogSoftmax(dim=1))).to(device=info.DEVICE),
    'pool_classifier_linear_regression': NeuralNetwork(nn.Sequential(
        nn.Linear(info.CARDS_IN_SET, 3, False),
        nn.LogSoftmax(dim=1))).to(device=info.DEVICE),
    'pool_classifier_wide': NeuralNetwork(nn.Sequential(
        nn.Linear(info.CARDS_IN_SET, 400, False),
        nn.ReLU(),
        nn.Linear(400, 3, False),
        nn.LogSoftmax(dim=1))).to(device=info.DEVICE),
    'pool_classifier_very_wide': NeuralNetwork(nn.Sequential(
        nn.Linear(info.CARDS_IN_SET, 1000, False),
        nn.ReLU(),
        nn.Linear(1000, 3, False),
        nn.LogSoftmax(dim=1))).to(device=info.DEVICE),
    'deck_classifier_linear_regression': NeuralNetwork(nn.Sequential(
        nn.Linear(info.CMC_BUCKETS + info.NUM_COLORS + info.CARDS_IN_SET, 3, False),
        nn.LogSoftmax(dim=1))).to(device=info.DEVICE),
    'deck_classifier_wide': NeuralNetwork(nn.Sequential(
        nn.Linear(info.CMC_BUCKETS + info.NUM_COLORS + info.CARDS_IN_SET, 1000, False),
        nn.ReLU(),
        nn.Linear(1000, 1000, False),
        nn.ReLU(),
        nn.Linear(1000, 3, False),
        nn.LogSoftmax(dim=1))).to(device=info.DEVICE),
    'deck_classifier_deep': NeuralNetwork(nn.Sequential(
        nn.Linear(info.CMC_BUCKETS + info.NUM_COLORS + info.CARDS_IN_SET, 500, False),
        nn.ReLU(),
        nn.Linear(500, 500, False),
        nn.ReLU(),
        nn.Linear(500, 500, False),
        nn.ReLU(),
        nn.Linear(500, 500, False),
        nn.ReLU(),
        nn.Linear(500, 500, False),
        nn.ReLU(),
        nn.Linear(500, 500, False),
        nn.ReLU(),
        nn.Linear(500, 500, False),
        nn.ReLU(),
        nn.Linear(500, 500, False),
        nn.ReLU(),
        nn.Linear(500, 500, False),
        nn.ReLU(),
        nn.Linear(500, 3, False),
        nn.LogSoftmax(dim=1))).to(device=info.DEVICE),
    'deck_classifier_with_rank_regression': NeuralNetwork(nn.Sequential(
        nn.Linear(info.NUM_RANKS + info.CMC_BUCKETS + info.NUM_COLORS + info.CARDS_IN_SET, 3, False),
        nn.LogSoftmax(dim=1))).to(device=info.DEVICE),
    'deck_classifier_with_rank_wide': NeuralNetwork(nn.Sequential(
        nn.Linear(info.NUM_RANKS + info.CMC_BUCKETS + info.NUM_COLORS + info.CARDS_IN_SET, 1000, False),
        nn.ReLU(),
        nn.Linear(1000, 1000, False),
        nn.ReLU(),
        nn.Linear(1000, 3, False),
        nn.LogSoftmax(dim=1))).to(device=info.DEVICE),
    'deck_classifier_with_rank_deep': NeuralNetwork(nn.Sequential(
        nn.Linear(info.NUM_RANKS + info.CMC_BUCKETS + info.NUM_COLORS + info.CARDS_IN_SET, 500, False),
        nn.ReLU(),
        nn.Linear(500, 500, False),
        nn.ReLU(),
        nn.Linear(500, 500, False),
        nn.ReLU(),
        nn.Linear(500, 500, False),
        nn.ReLU(),
        nn.Linear(500, 500, False),
        nn.ReLU(),
        nn.Linear(500, 500, False),
        nn.ReLU(),
        nn.Linear(500, 500, False),
        nn.ReLU(),
        nn.Linear(500, 500, False),
        nn.ReLU(),
        nn.Linear(500, 500, False),
        nn.ReLU(),
        nn.Linear(500, 3, False),
        nn.LogSoftmax(dim=1))).to(device=info.DEVICE),
    'deck_classifier_with_rank_and_winrate_regression': NeuralNetwork(nn.Sequential(
        nn.Linear(1 + info.NUM_RANKS + info.CMC_BUCKETS + info.NUM_COLORS + info.CARDS_IN_SET, 3, False),
        nn.LogSoftmax(dim=1))).to(device=info.DEVICE),
    'deck_classifier_with_rank_and_winrate_wide': NeuralNetwork(nn.Sequential(
        nn.Linear(1 + info.NUM_RANKS + info.CMC_BUCKETS + info.NUM_COLORS + info.CARDS_IN_SET, 1000, False),
        nn.ReLU(),
        nn.Linear(1000, 1000, False),
        nn.ReLU(),
        nn.Linear(1000, 3, False),
        nn.LogSoftmax(dim=1))).to(device=info.DEVICE),
    'deck_classifier_with_rank_and_winrate_deep': NeuralNetwork(nn.Sequential(
        nn.Linear(1 + info.NUM_RANKS + info.CMC_BUCKETS + info.NUM_COLORS + info.CARDS_IN_SET, 500, False),
        nn.ReLU(),
        nn.Linear(500, 500, False),
        nn.ReLU(),
        nn.Linear(500, 500, False),
        nn.ReLU(),
        nn.Linear(500, 500, False),
        nn.ReLU(),
        nn.Linear(500, 500, False),
        nn.ReLU(),
        nn.Linear(500, 500, False),
        nn.ReLU(),
        nn.Linear(500, 500, False),
        nn.ReLU(),
        nn.Linear(500, 500, False),
        nn.ReLU(),
        nn.Linear(500, 500, False),
        nn.ReLU(),
        nn.Linear(500, 3, False),
        nn.LogSoftmax(dim=1))).to(device=info.DEVICE),
    'deck_classifier_with_rank_and_winrate_very_wide': NeuralNetwork(nn.Sequential(
        nn.Linear(1 + info.NUM_RANKS + info.CMC_BUCKETS + info.NUM_COLORS + info.CARDS_IN_SET, 10000, False),
        nn.ReLU(),
        nn.Linear(10000, 3, False),
        nn.LogSoftmax(dim=1))).to(device=info.DEVICE),
}

models = {
    'baseline_batch128_regression': MLP('baseline_batch128_regression', networks['baseline_regression'], 
                         nn.NLLLoss(), optim.SGD(networks['baseline_regression'].parameters(), lr=0.01)),
    'shallow_baseline': MLP('shallow_baseline', networks['wide_but_shallow'], 
                         nn.NLLLoss(), optim.SGD(networks['wide_but_shallow'].parameters(), lr=0.01)),
    'deep_baseline': MLP('deep_baseline', networks['narrow_but_deep'], 
                         nn.NLLLoss(), optim.SGD(networks['narrow_but_deep'].parameters(), lr=0.01)),
    'moderate_baseline': MLP('moderate_baseline', networks['2_hidden_layers'], 
                         nn.NLLLoss(), optim.SGD(networks['2_hidden_layers'].parameters(), lr=0.01)),
    'shallow_batch32_shuffle_baseline': MLP('shallow_batch32_shuffle_baseline', networks['wide_but_shallow'], 
                         nn.NLLLoss(), optim.SGD(networks['wide_but_shallow'].parameters(), lr=0.01)),
    'shallow_batch64_shuffle_baseline': MLP('shallow_batch64_shuffle_baseline', networks['wide_but_shallow'], 
                         nn.NLLLoss(), optim.SGD(networks['wide_but_shallow'].parameters(), lr=0.01)),
    'shallow_batch128_shuffle_baseline': MLP('shallow_batch128_shuffle_baseline', networks['wide_but_shallow'], 
                         nn.NLLLoss(), optim.SGD(networks['wide_but_shallow'].parameters(), lr=0.01)),
    'shallow_batch256_shuffle_baseline': MLP('shallow_batch256_shuffle_baseline', networks['wide_but_shallow'], 
                         nn.NLLLoss(), optim.SGD(networks['wide_but_shallow'].parameters(), lr=0.01)),
    'shallow_batch512_shuffle_baseline': MLP('shallow_batch512_shuffle_baseline', networks['wide_but_shallow'], 
                         nn.NLLLoss(), optim.SGD(networks['wide_but_shallow'].parameters(), lr=0.01)),
    'shallow_batch1024_shuffle_baseline': MLP('shallow_batch1024_shuffle_baseline', networks['wide_but_shallow'], 
                         nn.NLLLoss(), optim.SGD(networks['wide_but_shallow'].parameters(), lr=0.01)),
    'shallow_batch2048_shuffle_baseline': MLP('shallow_batch2048_shuffle_baseline', networks['wide_but_shallow'], 
                         nn.NLLLoss(), optim.SGD(networks['wide_but_shallow'].parameters(), lr=0.01)),
    'winrate_linear_regression': MLP('winrate_linear_regression', networks['winrate_linear_regression'], 
                         nn.NLLLoss(), optim.SGD(networks['winrate_linear_regression'].parameters(), lr=0.01)),
    'winrate_batch256_linear_regression': MLP('winrate_batch256_linear_regression', networks['winrate_linear_regression'], 
                         nn.NLLLoss(), optim.SGD(networks['winrate_linear_regression'].parameters(), lr=0.01)),
    'winrate_batch256_wide': MLP('winrate_batch256_wide', networks['winrate_wide'], 
                         nn.NLLLoss(), optim.SGD(networks['winrate_wide'].parameters(), lr=0.01)),
    'winrate_batch256_deep': MLP('winrate_batch256_deep', networks['winrate_deep'], 
                         nn.NLLLoss(), optim.SGD(networks['winrate_deep'].parameters(), lr=0.1)),
    'winrate_batch256_big': MLP('winrate_batch256_big', networks['winrate_big'], 
                         nn.NLLLoss(), optim.SGD(networks['winrate_big'].parameters(), lr=0.01)),
    'pool_classifier_batch256_regression': MLP('pool_classifier_batch256_regression', networks['pool_classifier_linear_regression'], 
                         nn.NLLLoss(), optim.SGD(networks['pool_classifier_linear_regression'].parameters(), lr=0.01)),
    'pool_classifier_batch256_wide': MLP('pool_classifier_batch256_wide', networks['pool_classifier_wide'], 
                         nn.NLLLoss(), optim.SGD(networks['pool_classifier_wide'].parameters(), lr=0.01)),
    'pool_classifier_batch256_very_wide': MLP('pool_classifier_batch256_very_wide', networks['pool_classifier_very_wide'], 
                         nn.NLLLoss(), optim.SGD(networks['pool_classifier_very_wide'].parameters(), lr=0.01)),
    'deck_classifier_batch128_regression': MLP('deck_classifier_batch128_regression', networks['deck_classifier_linear_regression'], 
                         nn.NLLLoss(), optim.SGD(networks['deck_classifier_linear_regression'].parameters(), lr=0.01)),
    'deck_classifier_batch1024_regression': MLP('deck_classifier_batch1024_regression', networks['deck_classifier_linear_regression'], 
                         nn.NLLLoss(), optim.SGD(networks['deck_classifier_linear_regression'].parameters(), lr=0.01)),
    'deck_classifier_batch128_wide': MLP('deck_classifier_batch128_wide', networks['deck_classifier_wide'], 
                         nn.NLLLoss(), optim.SGD(networks['deck_classifier_wide'].parameters(), lr=0.01)),
    'deck_classifier_batch128_deep': MLP('deck_classifier_batch128_deep', networks['deck_classifier_deep'], 
                         nn.NLLLoss(), optim.SGD(networks['deck_classifier_deep'].parameters(), lr=0.01)),
    'deck_classifier_with_rank_batch128_regression': MLP('deck_classifier_with_rank_batch128_regression', networks['deck_classifier_with_rank_regression'], 
                         nn.NLLLoss(), optim.SGD(networks['deck_classifier_with_rank_regression'].parameters(), lr=0.01)),
    'deck_classifier_with_rank_batch128_deep': MLP('deck_classifier_with_rank_batch128_deep', networks['deck_classifier_with_rank_deep'], 
                         nn.NLLLoss(), optim.SGD(networks['deck_classifier_with_rank_deep'].parameters(), lr=0.01)),
    'deck_classifier_with_rank_batch128_wide': MLP('deck_classifier_with_rank_batch128_wide', networks['deck_classifier_with_rank_wide'], 
                         nn.NLLLoss(), optim.SGD(networks['deck_classifier_with_rank_wide'].parameters(), lr=0.01)),
    'deck_classifier_with_rank_and_winrate_batch128_regression': MLP('deck_classifier_with_rank_and_winrate_batch128_regression', networks['deck_classifier_with_rank_and_winrate_regression'], 
                         nn.NLLLoss(), optim.SGD(networks['deck_classifier_with_rank_and_winrate_regression'].parameters(), lr=0.01)),
    'deck_classifier_with_rank_and_winrate_batch128_deep': MLP('deck_classifier_with_rank_and_winrate_batch128_deep', networks['deck_classifier_with_rank_and_winrate_deep'], 
                         nn.NLLLoss(), optim.SGD(networks['deck_classifier_with_rank_and_winrate_deep'].parameters(), lr=0.01)),
    'deck_classifier_with_rank_and_winrate_batch128_wide': MLP('deck_classifier_with_rank_and_winrate_batch128_wide', networks['deck_classifier_with_rank_and_winrate_wide'], 
                         nn.NLLLoss(), optim.SGD(networks['deck_classifier_with_rank_and_winrate_wide'].parameters(), lr=0.01)),
    'deck_classifier_with_rank_and_winrate_batch128_very_wide': MLP('deck_classifier_with_rank_and_winrate_batch128_very_wide', networks['deck_classifier_with_rank_and_winrate_very_wide'], 
                         nn.NLLLoss(), optim.SGD(networks['deck_classifier_with_rank_and_winrate_very_wide'].parameters(), lr=0.01)),
    'shallow_baseline_ablation1': MLP('shallow_baseline_ablation1', networks['wide_but_shallow_no_pack_pick_nums'], 
                         nn.NLLLoss(), optim.SGD(networks['wide_but_shallow_no_pack_pick_nums'].parameters(), lr=0.01)),
    'shallow_baseline_half_data': MLP('shallow_baseline_half_data', networks['wide_but_shallow'], 
                         nn.NLLLoss(), optim.SGD(networks['wide_but_shallow'].parameters(), lr=0.01)),
}

def train_baseline_model():
    model = models['shallow_baseline_half_data']
    training_dataloader = DataLoader(HumanPickDataset('clean_datasets/training_data.0.WOE.csv', nrows=2261994), batch_size=128, shuffle=True)
    validation_dataloader = DataLoader(HumanPickDataset('clean_datasets/validation_data.0.WOE.csv'), batch_size=2048)

    model.load_model()
    model.train(training_dataloader, validation_dataloader, num_epochs=1000)

def train_winrate_model():
    model = models['winrate_batch256_deep']
    training_dataloader = DataLoader(PoolRecordDataset('clean_datasets/winrate_training_data.0.WOE.csv'), batch_size=256, shuffle=True)
    validation_dataloader = DataLoader(PoolRecordDataset('clean_datasets/winrate_validation_data.0.WOE.csv'), batch_size=2048)

    model.load_model()
    model.train(training_dataloader, validation_dataloader, num_epochs=100, batch_print_freq=100)

def train_fuzzier_winrate_model():
    model = models['pool_classifier_batch256_very_wide']
    training_dataloader = DataLoader(FuzzyPoolRecordDataset('clean_datasets/winrate_training_data.0.WOE.csv'), batch_size=256, shuffle=True)
    validation_dataloader = DataLoader(FuzzyPoolRecordDataset('clean_datasets/winrate_validation_data.0.WOE.csv'), batch_size=2048)

    model.load_model()
    model.train(training_dataloader, validation_dataloader, num_epochs=100, batch_print_freq=100)

def train_deck_classifier_model():
    model = models['deck_classifier_batch128_wide']
    training_dataloader = DataLoader(DeckClassifierDataset('clean_datasets/deck_training_data.0.WOE.csv'), batch_size=128, shuffle=True)
    validation_dataloader = DataLoader(DeckClassifierDataset('clean_datasets/deck_validation_data.0.WOE.csv'), batch_size=2048)

    model.load_model()
    model.train(training_dataloader, validation_dataloader, num_epochs=100, batch_print_freq=100)

def train_deck_classifier_with_rank_model():
    model = models['deck_classifier_batch128_wide']
    training_dataloader = DataLoader(DeckClassifierWithRankDataset('clean_datasets/deck_training_data.0.WOE.csv'), batch_size=128, shuffle=True)
    validation_dataloader = DataLoader(DeckClassifierWithRankDataset('clean_datasets/deck_validation_data.0.WOE.csv'), batch_size=2048)

    model.load_model()
    model.train(training_dataloader, validation_dataloader, num_epochs=100, batch_print_freq=100)

def train_deck_classifier_with_rank_and_winrate_model():
    model = models['deck_classifier_batch128_wide']
    training_dataloader = DataLoader(DeckClassifierWithRankAndWinrateDataset('clean_datasets/deck_training_data.0.WOE.csv'), batch_size=128, shuffle=True)
    validation_dataloader = DataLoader(DeckClassifierWithRankAndWinrateDataset('clean_datasets/deck_validation_data.0.WOE.csv'), batch_size=2048)

    model.load_model()
    model.train(training_dataloader, validation_dataloader, num_epochs=100, batch_print_freq=100)

def validate_model_shapes():
    for model in models:
        models[model].load_model()
        print(f'Loaded {model} successfully')

def get_performances():
    print('Human Pick Models\n----------------------------------------')
    humanpickmodels = ['baseline_batch128_regression','shallow_baseline','deep_baseline','moderate_baseline',
                       'shallow_batch32_shuffle_baseline', 'shallow_batch64_shuffle_baseline', 'shallow_batch128_shuffle_baseline', 'shallow_batch256_shuffle_baseline'
                       'shallow_batch512_shuffle_baseline', 'shallow_batch1024_shuffle_baseline', 'shallow_batch2048_shuffle_baseline']
    dataloader = DataLoader(HumanPickDataset('clean_datasets/validation_data.0.WOE.csv'), batch_size=2048)
    for model in humanpickmodels:
        mlp = models[model]
        mlp.load_model()
        print(f'Model: {model}')
        mlp.test_model(dataloader)
        print()
    
    print('\n\n\nWinrate Models\n----------------------------------------')
    winrate_models = ['winrate_linear_regression','winrate_batch256_linear_regression','winrate_batch256_wide',
                      'winrate_batch256_deep','winrate_batch256_big']
    dataloader = DataLoader(PoolRecordDataset('clean_datasets/winrate_validation_data.0.WOE.csv'), batch_size=2048)
    for model in winrate_models:
        mlp = models[model]
        mlp.load_model()
        print(f'Model: {model}')
        mlp.test_model(dataloader)
        print()

    print('\n\n\nFuzzy Winrate Models\n----------------------------------------')
    fuzzy_winrate_models = ['pool_classifier_batch256_regression','pool_classifier_batch256_wide','pool_classifier_batch256_very_wide']
    dataloader = DataLoader(FuzzyPoolRecordDataset('clean_datasets/winrate_validation_data.0.WOE.csv'), batch_size=2048)
    for model in fuzzy_winrate_models:
        mlp = models[model]
        mlp.load_model()
        print(f'Model: {model}')
        mlp.test_model(dataloader)
        print()

    print('\n\n\nDeck Winrate Models\n----------------------------------------')
    deck_winrate_models = ['deck_classifier_batch128_regression','deck_classifier_batch1024_regression',
                           'deck_classifier_batch128_wide','deck_classifier_batch128_deep']
    dataloader = DataLoader(DeckClassifierDataset('clean_datasets/deck_validation_data.0.WOE.csv'), batch_size=2048)
    for model in deck_winrate_models:
        mlp = models[model]
        mlp.load_model()
        print(f'Model: {model}')
        mlp.test_model(dataloader)
        print()

    print('\n\n\nDeck Winrate with Rank Models\n----------------------------------------')
    deck_winrate_with_rank_models = ['deck_classifier_with_rank_batch128_regression','deck_classifier_with_rank_batch128_deep','deck_classifier_with_rank_batch128_wide']
    dataloader = DataLoader(DeckClassifierWithRankDataset('clean_datasets/deck_validation_data.0.WOE.csv'), batch_size=2048)
    for model in deck_winrate_with_rank_models:
        mlp = models[model]
        mlp.load_model()
        print(f'Model: {model}')
        mlp.test_model(dataloader)
        print()

    print('\n\n\nDeck Winrate with Rank and Winrate Models\n----------------------------------------')
    deck_winrate_with_rank_and_winrate_models = ['deck_classifier_with_rank_and_winrate_batch128_regression',
                                     'deck_classifier_with_rank_and_winrate_batch128_deep',
                                     'deck_classifier_with_rank_and_winrate_batch128_wide',
                                     'deck_classifier_with_rank_and_winrate_batch128_very_wide']
    dataloader = DataLoader(DeckClassifierWithRankAndWinrateDataset('clean_datasets/deck_validation_data.0.WOE.csv'), batch_size=2048)
    for model in deck_winrate_with_rank_and_winrate_models:
        mlp = models[model]
        mlp.load_model()
        print(f'Model: {model}')
        mlp.test_model(dataloader)
        print()


def main():
    # train_baseline_model()
    # train_winrate_model()
    # train_fuzzier_winrate_model()
    # train_pool_classifier_model()
    # train_deck_classifier_model()
    # train_deck_classifier_with_rank_model()
    # train_deck_classifier_with_rank_and_winrate_model()
    # validate_model_shapes()
    # for model in models.values():
    #     convert_for_mathematica(f'models/{model.name}/log.txt')
    get_performances()
    pass

if __name__ == '__main__':
    main()