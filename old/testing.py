import torch
import numpy as np
import pandas as pd
from pickle import Pickler, Unpickler # Pickle is a inbuilt python utility for saving variables to files and reloading them
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from dataclasses import dataclass

CARDS_PER_PACK = 14
CARDS_PER_DRAFT = 3 * CARDS_PER_PACK
DRAFT_CSV_FILENAME = "draft_data_public.WOE.PremierDraft.csv"

def remove_rows(file, chunk_location, starting_row, nrows, chunksize, column_labels, dtypes):
    # Read in the chunk where the draft started and drop the problematic draft from it
    # (We only have to worry about the chunk the draft started in since we want
    # to overwrite the entirety of any subsequent chunks--they will only contain
    # the probematic draft)
    end = min(starting_row + nrows, chunksize)
    # print(f"starting row: {starting_row} chunksize: {chunksize} end: {end}")
    # print(f"Removing Rows: " + [i for i in range(starting_row, end)].__str__())
    file.seek(chunk_location)
    problematic_chunk = pd.read_csv(file, names=column_labels, nrows=end, dtype=dtypes, header=(0 if chunk_location == 0 else None))
    # print(problematic_chunk)
    problematic_chunk.drop([i for i in range(starting_row, end)], inplace=True)

    # Overwrite the problematic chunk with the fixed version
    file.seek(chunk_location)
    problematic_chunk.to_csv(file, index=False, header=(True if chunk_location == 0 else False))

@dataclass
class DraftLocation():
    """
        Stores the position of a draft entry within the file being written
    """
    chunk : int = 0
    offset : int = 0
    file_pos : int = 0

    def update(self, current_chunk, nrows, file_pos):
        if self.chunk != current_chunk:
            self.chunk = current_chunk
            self.offset = 0
        self.offset += nrows
        self.file_pos = file_pos

# BUG ALLOWS FOR MULTIPLE FIRST ENTRIES IN A ROW
# BUG FIGURE OUT WHY IT CRASHED
# Could check that there is the correct number of cards available to pick
def preprocess_data(csv_filename, chunksize=1000):
    # Filter out columns we don't care about and reorder them alphabetically
    # to group pack & pool entries (leave pack/pick #)
    # The pick_maindeck_rate and pick_sideboard_in_rate seem potentially
    # interesting for some of the more in depth algorithms
    header_frame = pd.read_csv(csv_filename, nrows=0)
    header_frame.drop(inplace=True, columns=["expansion", "event_type", "draft_id", "draft_time", "rank", "event_match_wins", "event_match_losses", "pick_maindeck_rate", "pick_sideboard_in_rate", "user_n_games_bucket", "user_game_win_rate_bucket", "pack_card_Plains", "pack_card_Island", "pack_card_Swamp", "pack_card_Mountain", "pack_card_Forest", "pool_Plains", "pool_Island", "pool_Swamp", "pool_Mountain", "pool_Forest"])
    columns = header_frame.columns
    new_order = list(pd.Index(['pack_number', 'pick_number', 'pick']).append(columns.drop(['pack_number', 'pick_number', 'pick']).sort_values()))

    # Manually set the datatypes to be more memory efficient
    # Every column usese a np.uint8 except for pick which is a str
    dtypes = defaultdict(lambda: np.uint8)
    dtypes['pick'] = str

    # Create a new csv file without the extraneous data that is nicely organized
    with pd.read_csv(csv_filename, usecols=columns, chunksize=chunksize, dtype=dtypes) as reader:
        with open("cleaned_" + csv_filename, "w+", newline="") as writer:
            num_rows = 0 # Track the number of rows in the resulting csv            

            labels = ['pack_number', 'pick_number'] + [label for label in columns if label.startswith("pool_")]
            expected_row = pd.Series([0] * len(labels), index=labels)
            pack_card_labels = [label for label in columns if label.startswith("pack_card_")]
            # Track the chunk, offset, and file location of the current draft 
            # Used in case we need to delete it after partially writing it
            draft_location_in_file = DraftLocation(file_pos=writer.tell())
            looking_for_new_draft = False
            for chunk_num, chunk in enumerate(reader):
                validated_rows_in_chunk = 0
                # Validate each pick within the chunk
                for row_num, row in chunk.iterrows():
                    # Ensure the row is valid, otherwise drop it and look for a new draft
                    # (it should agrees with previous picks and take a card actually in the pack)
                    matches_expected = np.array_equal(row[labels].values, expected_row.values)
                    is_new_row = np.sum(row[labels]) == 0
                    if (not matches_expected) \
                            or (row['pack_card_' + row['pick']] == 0) \
                            or (np.sum(row[pack_card_labels]) != CARDS_PER_PACK - row['pick_number']):
                        if not is_new_row:
                            chunk.drop(row_num, inplace=True)
                        if not looking_for_new_draft:
                            # Check if the first row of data for the invalid draft is in a previous chunk
                            rows_in_draft = expected_row['pack_number'] * CARDS_PER_PACK + expected_row['pick_number']
                            row_of_invalid_draft = row_num - rows_in_draft
                            if row_of_invalid_draft < chunk_num * chunksize:
                                # Remove the problematic draft from the chunk it starts in
                                # (We only have to worry about the chunk the draft started in since we want
                                # to overwrite the entirety of any subsequent chunks--they will only contain
                                # the probematic draft)
                                remove_rows(writer, draft_location_in_file.file_pos, draft_location_in_file.offset, rows_in_draft, chunksize, new_order, dtypes)
                                # Update the starting row to the start of the current chunk, so we correctly fix it
                                row_of_invalid_draft = chunk_num * chunksize
                            rows_to_drop = [i for i in range(row_of_invalid_draft, row_num)]
                            chunk.drop(rows_to_drop, inplace=True)
                            validated_rows_in_chunk -= len(rows_to_drop) # Remove the dropped rows from the valid count

                            expected_row[:] = 0
                            looking_for_new_draft = True
                        if is_new_row: # VERY BAD AND SAD COPY PASTED CODE --> FIX!!!!!
                            expected_row[:] = 0
                            validated_rows_in_chunk += 1
                            if row['pick_number'] == CARDS_PER_PACK - 1:
                                expected_row['pack_number'] += 1
                            expected_row['pick_number'] = (expected_row['pick_number'] + 1) % CARDS_PER_PACK
                            expected_row['pool_' + row['pick']] += 1
                        continue
                    looking_for_new_draft = False
                    
                    # If we've reached the end of a draft, restart--otherwise update what is expected
                    validated_rows_in_chunk += 1
                    if row['pack_number'] == 2 and row['pick_number'] == CARDS_PER_PACK - 1:
                        expected_row[:] = 0
                        num_rows += CARDS_PER_DRAFT # Track how many rows we have in the cleaned data

                        # Update the location of the current draft
                        draft_location_in_file.update(chunk_num, validated_rows_in_chunk, writer.tell())
                        validated_rows_in_chunk = 0
                    else:
                        if row['pick_number'] == CARDS_PER_PACK - 1:
                            expected_row['pack_number'] += 1
                        expected_row['pick_number'] = (expected_row['pick_number'] + 1) % CARDS_PER_PACK
                        expected_row['pool_' + row['pick']] += 1

                # After we've validated the chunk, reorder the remaining data
                # and write it out to disk
                chunk = chunk[new_order] # Reorder the columns
                chunk.to_csv(writer, index=False, header=(True if chunk_num == 0 else False))

                print(f"rows processed: {row_num + 1} rows kept: {num_rows} usable drafts: {num_rows // 42}")
            # If the last draft is incomplete we shouldn't include it
            if not looking_for_new_draft and not (row['pack_number'] == 2 and row['pick_number'] == CARDS_PER_PACK - 1):
                rows_in_draft = row['pack_number'] * CARDS_PER_PACK + row['pick_number']
                remove_rows(writer, draft_location_in_file.file_pos, draft_location_in_file.offset, rows_in_draft, chunksize, new_order, dtypes)
            writer.truncate() # In case there were chunks at the end we intended to overwrite
        
    metadata = {'columns': columns, 'num_rows': num_rows}
    with open('metainfo_cleaned_' + csv_filename + '.pickle', "wb") as file:
        Pickler(file).dump(metadata) # Write the metadata dict to a file

    return metadata

def clean_row():
    pass

# CONTAINS BUG-----DIVIDES BY 0 WHEN CALCULATING WINRATE!!!!!!!!!!!!!!!!!!!!!!!!!!!!
pd.options.mode.chained_assignment = None 
def create_winrate_data(csv_filename, chunksize=100000):
    header = pd.read_csv(csv_filename, nrows=0)
    columns = header.columns.drop(["expansion", "event_type", "draft_id", "draft_time", "rank", "pick_maindeck_rate", "pick_sideboard_in_rate", "user_n_games_bucket", "user_game_win_rate_bucket", "pool_Plains", "pool_Island", "pool_Swamp", "pool_Mountain", "pool_Forest"])
    columns = columns.drop([label for label in columns if label.startswith('pack_card_')])
    pool_card_labels = [label for label in columns if label.startswith('pool_')]
    new_order = ['winrate', 'avg_card_winrate', 'W_devotion', 'U_devotion', 'B_devotion', 'R_devotion', 'G_devotion'] + sorted(pool_card_labels)

    card_metadata = pd.read_csv("cards.csv", usecols=['name', 'color_identity', 'expansion'])
    card_metadata = card_metadata[(card_metadata['expansion'] == 'WOE') | (card_metadata['expansion'] == 'WOT')].fillna('').drop_duplicates(subset='name').set_index('name')

    card_winrates = pd.read_csv("winrates.csv")

    with pd.read_csv(csv_filename, usecols=columns, chunksize=chunksize) as reader:
        with open("cleaned_" + csv_filename, "w+", newline="") as writer:
            for chunk_num, chunk in enumerate(reader):
                # Get only rows that end drafts
                chunk = chunk[(chunk['pack_number'] == 2) & (chunk['pick_number'] == CARDS_PER_PACK - 1)]

                # Calculate winrate
                chunk.loc[:, 'winrate'] = chunk['event_match_wins']/(chunk['event_match_wins'] + chunk['event_match_losses'])
                
                #print(pool_card_labels)
                # Note the current pick (sadly not vectorized :( )
                for index in chunk.index:
                    chunk.loc[index, 'pool_' + chunk.loc[index, 'pick']] += 1
                    color_identity = {'W': 0, 'U': 0, 'B': 0, 'R': 0, 'G': 0}
                    card_winrate = 0
                    for card in pool_card_labels:
                        for color in card_metadata.loc[card[5:], 'color_identity']:
                            color_identity[color] += chunk.loc[index, card]
                        card_winrate += card_winrates[card].item() * chunk.loc[index, card]
                        
                    card_winrate /= CARDS_PER_DRAFT
                    chunk.loc[index, 'avg_card_winrate'] = card_winrate
                              
                    total_devotion = sum(color_identity.values())
                    chunk.loc[index, 'W_devotion'] = color_identity['W'] / total_devotion
                    chunk.loc[index, 'U_devotion'] = color_identity['U'] / total_devotion
                    chunk.loc[index, 'B_devotion'] = color_identity['B'] / total_devotion
                    chunk.loc[index, 'R_devotion'] = color_identity['R'] / total_devotion
                    chunk.loc[index, 'G_devotion'] = color_identity['G'] / total_devotion
                
                chunk.drop(inplace=True, columns=['event_match_wins', 'event_match_losses', 'pack_number', 'pick_number', 'pick'])
                
                chunk = chunk[new_order]
                chunk.to_csv(writer, index=False, header=(True if chunk_num == 0 else False))
                print(chunk_num)
            pass

# CONTAINS BUG-----DIVIDES BY 0 WHEN CALCULATING WINRATE!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# pd.options.mode.chained_assignment = None 
def create_player_skill_data(csv_filename, good_at_the_game=True, chunksize=100000):
    header = pd.read_csv(csv_filename, nrows=0)
    columns = header.columns.drop(["expansion", "event_type", "draft_id", "draft_time", 'event_match_wins', 'event_match_losses', "pick_maindeck_rate", "pick_sideboard_in_rate", "user_n_games_bucket", "pool_Plains", "pool_Island", "pool_Swamp", "pool_Mountain", "pool_Forest"])
    card_labels = [label for label in columns if label.startswith('pool_') or label.startswith('pack_card_')]
    new_order = ['pack_number', 'pick_number', 'pick'] + sorted(card_labels)

    with pd.read_csv(csv_filename, usecols=columns, chunksize=chunksize) as reader:
        with open("cleaned_" + csv_filename, "w+", newline="") as writer:
            for chunk_num, chunk in enumerate(reader):
                # Get only rows that end drafts
                chunk = chunk[((chunk['user_game_win_rate_bucket'] >= 0.61 if good_at_the_game  
                               else chunk['user_game_win_rate_bucket'] <= 0.49)) & 
                               (((chunk['rank'] != 'bronze') & (chunk['rank'] != 'silver')) if good_at_the_game
                               else ((chunk['rank'] != 'mythic') & (chunk['rank'] != 'diamond')))]

                chunk.drop(inplace=True, columns=['rank', 'user_game_win_rate_bucket'])
                
                chunk = chunk[new_order]
                # print(chunk)
                chunk.to_csv(writer, index=False, header=(True if chunk_num == 0 else False))
                print(chunk_num)
                # break
            pass

def fix_oopsie_2():
    df = pd.read_csv("bad_player_cleaned_draft_data_public.WOE.PremierDraft.csv")
    fixed = df.drop(columns=["pack_card_Plains", "pack_card_Island", "pack_card_Swamp", "pack_card_Mountain", "pack_card_Forest", "pack_number.1"])
    # fixed = fixed.drop_duplicates()
    fixed.to_csv("bad_player_draft_data.csv", index=False)

def main():
    # with open('metainfo_cleaned_' + DRAFT_CSV_FILENAME + '.pickle', "rb") as file:
    #     metadata = Unpickler(file).load() # Gets the saved object from the pickle file

    # dtypes = defaultdict(lambda: np.uint8)
    # dtypes['pick'] = str
    # df = pd.read_csv('cleaned_' + DRAFT_CSV_FILENAME, usecols=metadata['columns'], dtype=dtypes)
    # print(df.shape)
    #print(preprocess_data(DRAFT_CSV_FILENAME, chunksize=100000))
    # create_winrate_data(DRAFT_CSV_FILENAME)
    # create_player_skill_data(DRAFT_CSV_FILENAME, False, chunksize=100000)
    fix_oopsie_2()

if __name__ == "__main__":
    main()