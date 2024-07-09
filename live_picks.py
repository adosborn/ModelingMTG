import neural_net as nun
import set_info as info
import random
import torch
import sys
import csv
import utils 
from models import models
import performance_model as pm


def build_single_pack_csv(cards_in_pack, cards_in_pool, pack_num, pick_num, lc):
    # print(cards_in_pack)
    name = "clean_datasets/single_pack.csv"

    with open(name, 'r') as inf:
        reader = csv.reader(inf.readlines())
    
    with open(name, 'w') as outf:
        writer = csv.writer(outf)
        line_num = 0
        for line in reader:
            if line_num == 1:
                # build up list of cards in pack
                # print(cards_in_pack)
                tensor = nun.torch.zeros(1,lc.num_cards)
                for card_from_pack in cards_in_pack:
                    new_ten = lc.to_tensor([card_from_pack])
                    tensor = nun.torch.add(new_ten, tensor) 
                # once all cards in pack are loaded into tensor, reformat that tensor to svg
                list_form = tensor.reshape(lc.num_cards,1).tolist()
                list_form = [int(list_form[i][0]) for i in range(lc.num_cards)]
                # build up list of cards in pool
                # print(cards_in_pool)
                pool_tensor = nun.torch.zeros(1,lc.num_cards)
                for card_from_pool in cards_in_pool:
                    pool_new_ten = lc.to_tensor([card_from_pool])
                    pool_tensor = nun.torch.add(pool_new_ten, pool_tensor) 
                # once all cards in pack are loaded into tensor, reformat that tensor to svg
                list_form_pool = pool_tensor.reshape(lc.num_cards,1).tolist()
                list_form_pool = [int(list_form_pool[i][0]) for i in range(lc.num_cards)]
                # finally, write line
                writer.writerow([pack_num, pick_num, "Rat Out"] + list_form + list_form_pool)
            else: writer.writerow(line)
            line_num+=1
        writer.writerows(reader)
    # print("done")
    return 

def generate_pack():
    woe_commons = [card for card in info.CARD_DATA if (info.CARD_DATA[card]['rarity'] == 'common' and info.CARD_DATA[card]['expansion'] == 'WOE')]
    woe_uncommons = [card for card in info.CARD_DATA if (info.CARD_DATA[card]['rarity'] == 'uncommon' and info.CARD_DATA[card]['expansion'] == 'WOE')]
    woe_rares = [card for card in info.CARD_DATA if (info.CARD_DATA[card]['rarity'] == 'rare' and info.CARD_DATA[card]['expansion'] == 'WOE')]
    woe_mythics = [card for card in info.CARD_DATA if (info.CARD_DATA[card]['rarity'] == 'mythic' and info.CARD_DATA[card]['expansion'] == 'WOE')]
    wot_uncommons = [card for card in info.CARD_DATA if (info.CARD_DATA[card]['rarity'] == 'uncommon' and info.CARD_DATA[card]['expansion'] == 'WOT')]
    wot_rares = [card for card in info.CARD_DATA if (info.CARD_DATA[card]['rarity'] == 'rare' and info.CARD_DATA[card]['expansion'] == 'WOT')]
    wot_mythics = [card for card in info.CARD_DATA if (info.CARD_DATA[card]['rarity'] == 'mythic' and info.CARD_DATA[card]['expansion'] == 'WOT')]

    is_foil = random.random() > 0.66667

    index_of_commons = torch.randperm(len(woe_commons))[:(8 if is_foil else 9)]
    index_of_uncommons = torch.randperm(len(woe_uncommons))[:3]
    index_of_rare = torch.randperm(len(woe_rares))[0]
    index_of_mythic = torch.randperm(len(woe_mythics))[0]
    index_of_wot_u = torch.randperm(len(wot_uncommons))[0]
    index_of_wot_r = torch.randperm(len(wot_rares))[0]
    index_of_wot_m = torch.randperm(len(wot_mythics))[0]

    pack = []
    # commons
    for index in index_of_commons:
        pack.append(woe_commons[index])
    # uncommons
    for index in index_of_uncommons:
        pack.append(woe_uncommons[index])
    # rare/mythic
    if random.random() > (1.0/7.4):
        pack.append(woe_rares[index_of_rare])
    else:
        pack.append(woe_mythics[index_of_mythic])
    # enchanted tales
    tales_rarity = random.random()
    if tales_rarity > (14.0/15.0):
        pack.append(wot_mythics[index_of_wot_m])
    elif tales_rarity > (10.0/15.0):
        pack.append(wot_rares[index_of_wot_r])
    else:
        pack.append(wot_uncommons[index_of_wot_u])
    # foil
    if is_foil:
        foil_pack = generate_pack()
        index_of_foil = torch.randperm(len(foil_pack)-1)[0]
        pack.append(generate_pack()[index_of_foil])
    
    return pack

def make_tensor_for_pick(pack, pool, pack_num, pick_num):
    pack_tensor = torch.zeros(2 + 2 * info.CARDS_IN_SET, dtype=torch.float32, device=info.DEVICE)
    pack_indices = [2 + info.CARD_TO_ID[cardname] for cardname in pack]
    pool_indices = [2 + info.CARDS_IN_SET + info.CARD_TO_ID[cardname] for cardname in pool]
    for idx in pack_indices + pool_indices:
        pack_tensor[idx] += 1
    return pack_tensor.unsqueeze(0)

def make_pick(model, pack, pool, pack_num, pick_num):
    X = make_tensor_for_pick(pack, pool, pack_num, pick_num)
    y = model(X)

    cards_in_pack_mask = X[:,2:2+info.CARDS_IN_SET]>0
    card_suggested = torch.argmax(cards_in_pack_mask*torch.exp(y), dim=1).item()
    return info.ID_TO_CARD[card_suggested]


def manual_draft():
    drafting = True
    auto_keep_track_of_pool = False
    model = models['shallow_batch256_shuffle_baseline']
    model.load_model()
    pool = []
    pack_num = 1
    pick_num = 1
    while(drafting):
        pick_num += 1
        if pick_num > 14:
            pick_num -= 14
            pack_num += 1
        if not auto_keep_track_of_pool:
            print("do you want pool to be kept track of automatically?")
            ans = input("")
            if ans == "yes": auto_keep_track_of_pool = True
        print("pack " + str(pack_num) + ", pick " + str(pick_num))
        # print("pack number?")
        # pack_num = input("")
        # print("pick number?")
        # pick_num = input("")
        print("which cards are in the pack?")
        cards = []
        for i in range((info.CARDS_PER_PACK+1)-int(pick_num)):
            card = str(input(""))
            while (not card in info.CARDNAMES):
                print("that is not a valid card, try again")
                card = str(input(""))
            cards.append(card)
        if not auto_keep_track_of_pool:
            print("your pool is: ") 
            print(pool)
            print("how many cards do you want to add to pool?")
            num_cards_in_pool = input("")
            print("which cards?")
            for i in range(int(num_cards_in_pool)):
                pool_card = str(input(""))
                pool.append(pool_card)
        # input = sys.argv[1:]
        card_to_pick = make_pick(model, cards, pool, pack_num, pick_num)
        if auto_keep_track_of_pool:
            pool.append(card_to_pick)
            print("your pool is:")
            print(pool)
        print("was that the last pick?")
        res = input("")
        if res == "yes":
            drafting = False
    # print (len(input))
    return pool

class Player():
    def __init__(self, deck, cur_pack, left_neighbor, right_neighbor, next_pack):
        self.deck = deck
        self.cur_pack = cur_pack
        self.left_neighbor_index = left_neighbor
        self.right_neighbor_index = right_neighbor
        self.next_pack = next_pack

def manual_pick(human):
    print("the cards in the pack are:")
    for card in human.cur_pack:
        print(card)
    print("the cards in your pool are:")
    for card in human.deck:
        print(card)
    print("what card do you want to take?")
    pick = str(input(""))
    while (not pick in info.CARDNAMES):
                print("that is not a valid card, try again")
                pick = str(input(""))
    return pick


def draft_sim(mode):
    # create players
    model = models['shallow_batch256_shuffle_baseline']
    model.load_model()
    alternative_model = models['deck_classifier_batch128_wide']
    alternative_model.load_model()
    number_of_players = 8
    players = []
    for i in range(number_of_players):
        new_player = Player([], generate_pack(), left_neighbor=None, right_neighbor=None, next_pack=None)
        new_player.left_neighbor_index = ((i-1) if i > 0 else (number_of_players - 1))
        new_player.right_neighbor_index = ((i+1) if i < (number_of_players - 1) else 0)
        players.append(new_player)
    # start draft
    drafting = True
    direction = "left"
    pack_num = 0
    pick_num = 0
    for pack_num in range(3): 
        for player in players:
            player.cur_pack = generate_pack()
        for pick_num in range(info.CARDS_PER_PACK):
            for player in players:
                # make pick (Player 0 uses the alternative bot)
                toPick = pm.get_card_to_pick(alternative_model, player.cur_pack, player.deck) if (player is players[0] and mode == 1) else manual_pick(players[0]) if (player is players[0] and mode == 2) else make_pick(model, player.cur_pack, player.deck, pack_num, pick_num)
                if player is players[0]:
                    print(f'pack {pack_num}, pick {pick_num}, Pack: {player.cur_pack}\nPool: {player.deck}\nPick: {toPick}\n')

                player.deck.append(toPick)
                player.cur_pack.remove(toPick)
                # pass pack
                players[player.left_neighbor_index if direction == "left" else player.right_neighbor_index].next_pack = player.cur_pack
            # Update your pack
            for player in players:
                player.cur_pack = player.next_pack
        direction = "right" if direction == "left" else "left"
    i = 0
    for player in players:
        i+=1
        print("\nplayer " + str(i) + ":\n")
        print ("Deck")
        for card in player.deck:
            print(f'1 {card}')
        # print(player.deck)


def main():
    # mode: 0 = all standard bots, 1 = seat 0 alt bot, 2 = seat 0 human
    draft_sim(2)
    # manual_draft()
    # print(generate_pack())
    # read_input_pack()
    # utils.generate_metadata(info.CLEAN_DATA_PATHNAME, info.METADATA_PATHNAME)

if __name__ == "__main__":
    main()