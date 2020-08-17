# -*- coding: utf-8 -*-
import csv

from os import walk
from os.path import join
from tqdm import tqdm

DELIMITER = ','


"""Summary or Description of the Function

Parameters:
    
Returns:

"""


# TQDM progress bar with specific properties
def progress_bar(iterator, desc):
    return tqdm(iterator, desc=desc, bar_format="{desc:<20}{percentage:3.0f}%|{bar:50}{r_bar}")

# Gets all file paths in directory and sub directories
def walk_files(root):
    return [join(root,f) for root,dirs,files in walk(root) for f in files if valid_file(f)]

# Check if it's a valid clinical trial file
def valid_file(filename):
    return filename.lower().startswith('nct') and filename.lower().endswith('.xml')


def read_csv(filepath):
    data = []
    with open(filepath, newline='', encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=DELIMITER)
        [data.extend(row) for row in csv_reader]
    return data

def write_csv(filepath, data, multiple):
    with open(filepath, 'w', newline='', encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file, delimiter=DELIMITER)
            if multiple:
                writer.writerows(data)
            else:
                writer.writerow(data)
    pass

# start_time = time.time()
# print("--- %s seconds ---" % (time.time() - start_time))