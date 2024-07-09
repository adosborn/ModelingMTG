import utils
import torch
import numpy as np
from collections import defaultdict
from os.path import isfile

#-----------------------------------------------------------------------------------------#
# Update the following values based on what set is used                                   #
CARDS_IN_SET = 329 # Number of Draftable Cards in the Set (including basics)
DRAFT_BASICS = False # Whether or not basic lands are drafted (on Arena)
EXPANSION_SYMBOL = 'WOE' # WOTC assigned expansion abreviation
ALTERNATIVE_SYMBOL = 'WOT' # WOTC assigned expansion abreviation for other cards in pack (i.e. Bonus sheet)
#-----------------------------------------------------------------------------------------#

# Assumed to be universal constants
NUM_COLORS = 5
CMC_BUCKETS = 9
NUM_RANKS = 6

# Other constants computed using the above, no need to edit
RAW_DATA_PATHNAME = 'raw_datasets/draft_data_public.' + EXPANSION_SYMBOL + '.PremierDraft.csv'
CLEAN_DATA_PATHNAME = 'clean_datasets/full_draft_data.' + EXPANSION_SYMBOL + '.csv'
METADATA_PATHNAME = 'clean_datasets/metadata.' + EXPANSION_SYMBOL + '.pickle'
RAW_CARD_DATA_PATH = 'raw_datasets/cards.csv'
CARD_DATA_PATH = 'clean_datasets/card_data.csv'

CARDS_PER_PACK = 15 if DRAFT_BASICS else 14
CARDS_PER_DRAFT = 3 * CARDS_PER_PACK
if not DRAFT_BASICS:
    CARDS_IN_SET -= 5

# Info generated after some amount of processing
NROWS_IN_CLEAN_DATA, CARDNAMES, COLUMN_NAMES = utils.load_metadata(METADATA_PATHNAME)
PACK_CARDNAMES = [name for name in COLUMN_NAMES if name.startswith('pack_card_')]
POOL_CARDNAMES = [name for name in COLUMN_NAMES if name.startswith('pool_')]
CARD_DATA = utils.load_card_data(CARD_DATA_PATH)
CARD_TO_ID = {card: id for id, card in enumerate(CARDNAMES)}
ID_TO_CARD = {id: card for id, card in enumerate(CARDNAMES)}

# Constants for implementation
DTYPES = defaultdict(lambda: np.uint8, draft_id=str, rank=str, pick=str, pick_maindeck_rate=np.float32, pick_sideboard_in_rate=np.float32, user_game_win_rate_bucket=np.float32)
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu" 
print(f"Using {DEVICE} device")