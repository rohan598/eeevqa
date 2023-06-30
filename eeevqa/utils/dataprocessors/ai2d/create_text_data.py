import rootutils

root = rootutils.setup_root(
    search_from=__file__,
    indicator="setup.py",
    pythonpath=True,
    dotenv=True,
)

import os
from transformers import AutoProcessor

from eeevqa.utils.dataloaders.raw_data import load_pickle_dataset, save_dataset
from eeevqa.utils.args import parse_args, parse_boolean

# make all text targets at once
def create_text_data(samples_dict, max_new_tokens=128, processor=None):
    
    text_data = {
        "targets": [],
        "raw_output": [],
        "sample_list":[]
    }

    sample_keys = list(samples_dict.keys())
    sample_keys.sort()
    
    text_data["sample_list"] = sample_keys

    for key in sample_keys:  
        text_data["raw_output"].append(samples_dict[key]["raw_output"])
        
    text_data["targets"] =  processor(text=text_data["raw_output"], 
                              padding=True, 
                              truncation=True, 
                              return_tensors="pt", 
                              add_special_tokens=True, 
                              max_length=max_new_tokens)
    
    return text_data

if __name__ == '__main__':
    
    args = parse_args()
    print("----- Parsed Arguments -----")

    samples_dict = load_pickle_dataset(os.path.join(args.data_root, args.pickle_files_dir), "samples_dict")
    save_dir = os.path.join(args.data_root, args.pickle_files_dir)

    processor = AutoProcessor.from_pretrained(args.base_model_name)
    print("----- Basic Setup done -----") 

    text_data = create_text_data(samples_dict,
                            max_new_tokens=52, 
                            processor=processor)
    print("----- Created Text Dataset -----") 

    save_dataset(
        text_data,
        save_dir = save_dir,
        filename = "text_data.pkl"
    )
    print("----- Saved Text Dataset -----") 