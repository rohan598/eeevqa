import os
import pickle
import pandas as pd
import json 

def read_json_file(filepath):
    with open(filepath) as json_file:
        data = json.load(json_file)
    return data

def load_pickle_dataset(save_dir, source=""):
    pickle_filename = os.path.join(save_dir, f"{source}.pkl")
    with open(pickle_filename, 'rb') as f:
        pickle_data = pickle.load(f)
    return pickle_data

def load_data_split(pickle_files_path, split):
    dataset = load_pickle_dataset(pickle_files_path, source=split)
    return dataset

# def load_data(data_root, captions_filename, json_files_path, problems_filename, save_dir, pickle_files_path, train_split, val_split, test_split):

#     # read_captions
#     captions_dict = read_captions(data_root, captions_filename)

#     # read_problem_list
#     problem_list = read_problem_list(json_files_path, problems_filename)

#     train_dataset = load_pickle_dataset(pickle_files_path, source=train_split)

#     val_dataset = load_pickle_dataset(pickle_files_path, source=val_split)

#     test_dataset = load_pickle_dataset(pickle_files_path, source=test_split)

#     return captions_dict, problem_list, train_dataset, val_dataset, test_dataset
