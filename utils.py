import pandas as pd
from pickle import Pickler, Unpickler
from os.path import isfile
import csv

def get_col_names(csv_filename: str) -> pd.Index:
    """
        Reads a CSV file and returns an index of the column names
    """
    df = pd.read_csv(csv_filename, nrows=0)
    return df.columns

def generate_metadata(input_filename, output_filename):
        """
            Stores the num_rows, card_names, and column_names from a dataset in a pickle file
        """
        column_names = get_col_names(input_filename).to_list()
        card_names = [name[5:] for name in column_names if name.startswith('pool_')]

        df = pd.read_csv(input_filename, usecols=[0])
        num_rows = df.shape[0]
        
        metadata = {'num_rows': num_rows, 'card_names': card_names, 'column_names': column_names}
        with open(output_filename, "wb") as file:
            Pickler(file).dump(metadata) # Write the metadata dict to a file'

def load_metadata(filename):
    """
        Loads saved data from the pickle file and returns num_rows, card_names
    """
    try:
        with open(filename, "rb") as file:
            metadata = Unpickler(file).load() # Gets the saved object from the pickle file
            return metadata['num_rows'], metadata['card_names'], metadata['column_names']
    except FileNotFoundError:
        print('Dataset metadata file not found. Consider generating it.')
        return None, None, None

def load_card_data(filename):
    if isfile(filename):
        return pd.read_csv(filename, index_col='name').fillna('').to_dict(orient='index')
    else:
        print('Card data file not found. Consider generating it.')
        return None
    
def convert_for_mathematica(logpath):
    with open(logpath, "r") as reader:
        lines = reader.readlines()
        accuracy = [float(lines[i][10:-2])/100 for i in range(len(lines)) if i % 3 == 1]
        loss = [float(lines[i][10:]) for i in range(len(lines)) if i % 3 == 2]

        dirs = logpath.split('/')
        output_filepath = '/'.join(dirs[0:len(dirs)-1]) + f'/mathematica_{dirs[-1]}'
        with open(output_filepath, "w", newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(accuracy)
            writer.writerow(loss)
        print(f'Saved {output_filepath}')


