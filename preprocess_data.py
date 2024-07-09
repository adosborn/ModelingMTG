import set_info as info
import pandas as pd
import numpy as np
from utils import get_col_names, generate_metadata
from collections import defaultdict, namedtuple

class ChunkValidator():
    def __init__(self, writer, input_columns_names, output_column_names) -> None:
        self.writer = writer
        self.output_column_names = output_column_names

        # Stores Pack Number, Pick Number, Pool Cards
        expected_row_entries = ['pack_number', 'pick_number'] + [card_name[5:] for card_name in input_columns_names if card_name.startswith('pool_')]
        self.expected_row = {entry: 0 for entry in expected_row_entries}

        # Stores the index used to access the corresponding entry in an input row (based on dataframe index)
        self.pack_card_index = {column_name[10:]: idx for idx, column_name in enumerate(input_columns_names, start=1) if column_name.startswith('pack_card_')}
        self.pool_card_index = {column_name[5:]: idx for idx, column_name in enumerate(input_columns_names, start=1) if column_name.startswith('pool_')}

        # Tracks the logical location within a draft
        self.processed_rows_in_draft = 0

        # Tracks the draft's position within the file
        self.starting_chunk_file_location = 0
        self.nrows_before_draft_in_chunk = 0

        # Metadata about how many drafts we keep
        self.usable_drafts = 0
        self.invalid_lines = 0


    def _is_valid_row(self, row: namedtuple) -> bool:
        if not self._row_equals_expected(row) \
            or np.sum([row[idx] for idx in self.pack_card_index.values()]) != info.CARDS_PER_PACK - row.pick_number \
            or row[self.pack_card_index[row.pick]] == 0:
            return False
        
        return True
    
    def _row_equals_expected(self, row):
        return row.pack_number == self.expected_row['pack_number'] \
            and row.pick_number == self.expected_row['pick_number'] \
            and all([row[self.pool_card_index[card]] == self.expected_row[card] for card in self.pool_card_index])
        
    def _process_row(self, row):
        # Note the card taken
        self.expected_row[row.pick] += 1

        # Increment Pick and Pack number
        self.expected_row['pick_number'] = (self.expected_row['pick_number'] + 1) % info.CARDS_PER_PACK
        if self.expected_row['pick_number'] == 0:
            self.expected_row['pack_number'] += 1
        
        # Track number of rows in the current draft and chunk
        self.processed_rows_in_draft += 1
        self.draft_rows_in_current_chunk += 1

        # If we've seen 3 full packs, restart
        if self.processed_rows_in_draft == info.CARDS_PER_DRAFT:
            self._finish_draft()
            self._start_new_draft()


    def _reset_expected_row(self):
        for key in self.expected_row:
            self.expected_row[key] = 0
    
    def _finish_draft(self):
        # Update the file location info when we finish draft
        if self.starting_chunk_file_location != self.writer.tell():
            self.starting_chunk_file_location = self.writer.tell()
            self.nrows_before_draft_in_chunk = self.draft_rows_in_current_chunk
        else:
            self.nrows_before_draft_in_chunk += self.draft_rows_in_current_chunk

        self.usable_drafts += 1


    def _start_new_draft(self):
        self._reset_expected_row()
        self.processed_rows_in_draft = 0
        self.draft_rows_in_current_chunk = 0

    def _invalidate_current_draft(self, end_index=0):
        # If there are rows in previous chunks, deal with them
        if self.starting_chunk_file_location != self.writer.tell():
            # Read the chunk, but without the draft we want to invalidate
            self.writer.seek(self.starting_chunk_file_location)
            df = pd.read_csv(self.writer, dtype=info.DTYPES, nrows=self.nrows_before_draft_in_chunk, names=self.output_column_names, header=(0 if self.starting_chunk_file_location == 0 else None))
            
            # Rewrite the chunk --> future writes will overwrite the invalid draft
            self.writer.seek(self.starting_chunk_file_location)
            df.to_csv(self.writer, index=False, header=(True if self.starting_chunk_file_location == 0 else False))

        # Update metadata (for display only)
        self.invalid_lines += self.processed_rows_in_draft

        # Return the rows that need to be dropped from the current chunk's dataframe
        return [idx for idx in range(end_index - self.draft_rows_in_current_chunk, end_index)]

    def clean_chunk(self, df) -> pd.DataFrame:
        # Delete rows with missing values, and don't have them count towards the index
        num_nan_rows = df.shape[0]
        df.dropna(inplace=True) 
        df.reset_index(drop=True, inplace=True)
        num_nan_rows -= df.shape[0]
        self.invalid_lines += num_nan_rows

        rows_to_drop = []
        self.draft_rows_in_current_chunk = 0
        for row in df.itertuples():
            if not self._is_valid_row(row):
                # Only reset everything if we've run into an issue mid draft (just drop the current
                # row if it's invalid, since there are no previous rows in the draft)
                if self.expected_row['pack_number'] != 0 or self.expected_row['pick_number'] != 0:
                    rows_to_drop += self._invalidate_current_draft(row.Index)
                    self._start_new_draft()

                # If this row starts a new draft early, then we shouldn't drop it
                if self._is_valid_row(row):
                    self._process_row(row)
                else:
                    rows_to_drop.append(row.Index)
                    self.invalid_lines += 1
            else:
                self._process_row(row)
        df.drop(rows_to_drop, inplace=True)

        return df
    
    def done_writing(self):
        # Trim the end of the file if it doesn't end cleanly
        if self.processed_rows_in_draft != 0:
            self._invalidate_current_draft()
        self.writer.truncate()

def clean_data():
    # Get which columns to read from the file (drop useless ones)
    column_names = get_col_names(info.RAW_DATA_PATHNAME)
    if not info.DRAFT_BASICS:
        column_names = column_names[column_names.map(lambda column_name: not('Plains' in column_name or 'Island' in column_name or 'Swamp' in column_name or 'Mountain' in column_name or 'Forest' in column_name))]
    column_names = column_names.drop(['expansion', 'event_type', 'draft_time'])

    # Get the final ordering for the cards
    card_columns = column_names[column_names.map(lambda column: column.startswith('pool_') or column.startswith('pack_card_'))].sort_values()
    column_names_reordered = column_names.copy().drop(card_columns).append(card_columns)

    with pd.read_csv(info.RAW_DATA_PATHNAME, usecols=column_names, chunksize=10000, dtype=info.DTYPES) as reader:
        with open(info.CLEAN_DATA_PATHNAME, "w+", newline="") as writer:
            validator = ChunkValidator(writer, column_names, column_names_reordered)
            for chunk_num, df in enumerate(reader):
                df = validator.clean_chunk(df)
                df = df[column_names_reordered]
                df.to_csv(writer, index=False, header=(True if chunk_num == 0 else False))
                print(f'Chunk {chunk_num} Processed: {validator.usable_drafts} usable drafts ({validator.invalid_lines} lines discarded)')
            validator.done_writing()
    print(f"Finished! {validator.usable_drafts} usable drafts ({validator.invalid_lines} lines discarded)")


def clean_card_data():
    card_data = pd.read_csv(info.RAW_CARD_DATA_PATH, usecols=['name', 'color_identity', 'expansion', 'rarity', 'mana_value', 'types', 'is_booster'])
    card_data = card_data[((card_data['expansion'] == info.EXPANSION_SYMBOL) | (card_data['expansion'] == info.ALTERNATIVE_SYMBOL)) & (card_data['is_booster'] == True)].drop_duplicates().fillna('').set_index('name')
    card_data.drop(columns='is_booster', inplace=True)

    card_data.to_csv(info.CARD_DATA_PATH)

def clean_winrate_data(filepath: str):
    winrate_data = pd.read_csv(filepath, usecols=['pack_number', 'pick_number', 'event_match_wins', 'event_match_losses'] + info.POOL_CARDNAMES, dtype=info.DTYPES)
    
    # Select rows that are the end of drafts, and only if the event was fully played out
    winrate_data = winrate_data[(winrate_data['pack_number'] == 2) \
                    & (winrate_data['pick_number'] == info.CARDS_PER_PACK - 1) \
                    & ((winrate_data['event_match_wins'] == 7 ) | (winrate_data['event_match_losses'] == 3 ))]
    winrate_data.drop(columns=['pack_number', 'pick_number'], inplace=True)

    dirs = filepath.split('/')
    output_filepath = '/'.join(dirs[0:len(dirs)-1]) + f'/winrate_{dirs[-1]}'
    winrate_data.to_csv(output_filepath, index=False)
    print(f'Saved {output_filepath}')

def clean_deck_data(filepath: str):
    input_data = pd.read_csv(filepath, usecols=['pick', 'rank', 'pick_maindeck_rate', 'event_match_wins', 'event_match_losses', 'user_game_win_rate_bucket'], dtype=info.DTYPES)
    print('Loaded File, calculating...')

    data = []
    column_names = ['performance', 'bronze', 'silver', 'gold', 'platinum', 'diamond', 'mythic', 'winrate_bucket', 'cmc_0', 'cmc_1', 'cmc_2', 'cmc_3', 'cmc_4', 'cmc_5', 'cmc_6', 'cmc_7', 'cmc_8+', 'white_devotion', 'blue_devotion', 'black_devotion', 'red_devotion', 'green_devotion'] + info.CARDNAMES
    column_to_index = {column_name: idx for idx, column_name in enumerate(column_names)}
    for draft in range(input_data.shape[0] // info.CARDS_PER_DRAFT):
        # Skip drafts that weren't played out
        wins = input_data.iloc[draft * info.CARDS_PER_DRAFT]['event_match_wins']
        losses = input_data.iloc[draft * info.CARDS_PER_DRAFT]['event_match_losses']
        if wins != 7 and losses != 3:
            continue

        row = np.zeros(len(column_names), dtype=np.uint8)
        row[column_to_index['performance']] = 0 if wins <= 2 else 1 if wins <= 5 else 2
        row[column_to_index[input_data.iloc[draft * info.CARDS_PER_DRAFT]['rank']]] += 1
        row[column_to_index['winrate_bucket']] = input_data.iloc[draft * info.CARDS_PER_DRAFT]['user_game_win_rate_bucket']

        for pick_num in range(info.CARDS_PER_DRAFT):
            pick = input_data.iloc[draft * info.CARDS_PER_DRAFT + pick_num]
            cardname = pick['pick']
            if pick['pick_maindeck_rate'] >= 0.5:
                row[column_to_index[cardname]] += 1
                row[column_to_index['cmc_0'] + min(info.CARD_DATA[cardname]['mana_value'], 8)] += 1
                color_identity = info.CARD_DATA[cardname]['color_identity']
                if 'W' in color_identity: row[column_to_index['white_devotion']] += 1
                if 'U' in color_identity: row[column_to_index['blue_devotion']] += 1
                if 'B' in color_identity: row[column_to_index['black_devotion']] += 1
                if 'R' in color_identity: row[column_to_index['red_devotion']] += 1
                if 'G' in color_identity: row[column_to_index['green_devotion']] += 1
        data.append(row)
    print('Creating Dataframe...')
    deck_data = pd.DataFrame(data, columns=column_names)

    dirs = filepath.split('/')
    output_filepath = '/'.join(dirs[0:len(dirs)-1]) + f'/deck_{dirs[-1]}'
    deck_data.to_csv(output_filepath, index=False)
    print(f'Saved {output_filepath}')

def single_card_makes_deck(filepath: str):
    input_data = pd.read_csv(filepath, usecols=['pick', 'pick_maindeck_rate', 'pack_number', 'pick_number'] + info.POOL_CARDNAMES, dtype=info.DTYPES)
    print('Loaded File, calculating...')

    data = []
    column_names = ['maindeck_rate', 'card'] + info.CARDNAMES
    column_to_index = {column_name: idx for idx, column_name in enumerate(column_names)}
    for draft in range(input_data.shape[0] // info.CARDS_PER_DRAFT):

        row = np.zeros(len(column_names), dtype=np.int16)
        # row[column_to_index[input_data.iloc[draft * info.CARDS_PER_DRAFT]['rank']]] += 1
        # row[column_to_index['winrate_bucket']] = input_data.iloc[draft * info.CARDS_PER_DRAFT]['user_game_win_rate_bucket']

        for pick_num in range(info.CARDS_PER_DRAFT):
            pick = input_data.iloc[draft * info.CARDS_PER_DRAFT + pick_num]
            cardname = pick['pick']
            row[column_to_index[cardname]] += 1
        for pick_num in range(info.CARDS_PER_DRAFT):
            pick = input_data.iloc[draft * info.CARDS_PER_DRAFT + pick_num]
            cardname = pick['pick']
            row[column_to_index['card']] = info.CARD_TO_ID[cardname]
            row[column_to_index['maindeck_rate']] = (1 if (pick['pick_maindeck_rate'] >= 0.5) else 0)
            # if pick['pick_maindeck_rate'] >= 0.5:
            #     row[column_to_index[cardname]] += 1
            #     row[column_to_index['cmc_0'] + min(info.CARD_DATA[cardname]['mana_value'], 8)] += 1
            #     color_identity = info.CARD_DATA[cardname]['color_identity']
            #     if 'W' in color_identity: row[column_to_index['white_devotion']] += 1
            #     if 'U' in color_identity: row[column_to_index['blue_devotion']] += 1
            #     if 'B' in color_identity: row[column_to_index['black_devotion']] += 1
            #     if 'R' in color_identity: row[column_to_index['red_devotion']] += 1
            #     if 'G' in color_identity: row[column_to_index['green_devotion']] += 1
            data.append(row)
    print('Creating Dataframe...')
    deck_data = pd.DataFrame(data, columns=column_names)

    dirs = filepath.split('/')
    output_filepath = '/'.join(dirs[0:len(dirs)-1]) + f'/single_{dirs[-1]}'
    deck_data.to_csv(output_filepath, index=False)
    print(f'Saved {output_filepath}')


def main():
    #clean_card_data()
    #clean_data()
    #generate_metadata(info.CLEAN_DATA_PATHNAME, info.METADATA_PATHNAME)
    # clean_winrate_data(f'clean_datasets/training_data.0.{info.EXPANSION_SYMBOL}.csv')
    # clean_winrate_data(f'clean_datasets/validation_data.0.{info.EXPANSION_SYMBOL}.csv')
    # clean_winrate_data(f'clean_datasets/testing_data.0.{info.EXPANSION_SYMBOL}.csv')
    # clean_deck_data(f'clean_datasets/training_data.0.{info.EXPANSION_SYMBOL}.csv')
    # clean_deck_data(f'clean_datasets/validation_data.0.{info.EXPANSION_SYMBOL}.csv')
    # clean_deck_data(f'clean_datasets/testing_data.0.{info.EXPANSION_SYMBOL}.csv')
    single_card_makes_deck(f'clean_datasets/training_data.0.{info.EXPANSION_SYMBOL}.csv')
    single_card_makes_deck(f'clean_datasets/validation_data.0.{info.EXPANSION_SYMBOL}.csv')
    single_card_makes_deck(f'clean_datasets/testing_data.0.{info.EXPANSION_SYMBOL}.csv')

if __name__ == '__main__':
    main()