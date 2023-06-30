import rootutils

root = rootutils.setup_root(
    search_from=__file__,
    indicator="setup.py",
    pythonpath=True,
    dotenv=True,
)

import os
from PIL import Image
from collections import namedtuple

from eeevqa.utils.args import parse_args
from eeevqa.utils.dataloaders.raw_data import load_pickle_dataset, save_dataset

def create_one_ai2d_example(samples_dict, text_data, image_file, TrainQA = None):
    
    # find sample id and sample from it 
    sample_id = (image_file.split('.')[0]+ "."+ image_file.split('.')[1]).split("/")[-1]
    sample_num = text_data["sample_list"].index(sample_id)

    #input
    header_text = samples_dict[sample_id]["header_text"]
    tmp_img_file = Image.open(image_file)
    image = tmp_img_file.copy()
    tmp_img_file.close()
    flattened_patches = None
    attention_mask = None
    raw_output = text_data["raw_output"][sample_num]
    output = text_data["targets"].input_ids[sample_num]

    
    ai2d_example = TrainQA(sample_num, header_text, image, flattened_patches, attention_mask, raw_output, output)

    return ai2d_example

def convert_ai2d_to_dataset(samples_dict, text_data, data_split, TrainQA): 
    
    dataset = []

    for image_file in data_split:
        dataset.append(create_one_ai2d_example(
                                                samples_dict,
                                                text_data,
                                                image_file = image_file, 
                                                TrainQA = TrainQA))
    return dataset



if __name__ == '__main__':
    

    args = parse_args()
    print("----- Parsed Arguments -----")

    
    samples_dict = load_pickle_dataset(os.path.join(args.data_root, args.pickle_files_dir), "samples_dict")

    metadata_dict = load_pickle_dataset(os.path.join(args.data_root, args.pickle_files_dir), "metadata_dict")

    text_data = load_pickle_dataset(os.path.join(args.data_root, args.pickle_files_dir), "text_data")

    print("----- Loaded Text and Meta Data  -----") 

    TrainQA = namedtuple("TrainQA", "sample_num header_text image flattened_patches attention_mask raw_output output")

    data_split = metadata_dict[f"{args.data_split}_split_new_images"]
    convert_ai2d_to_dataset(samples_dict, text_data, data_split, TrainQA)
    print("-----  Created AI2D dataset -----") 

    save_dir = os.path.join(args.data_root, args.pickle_files_dir)
    save_dataset(
        text_data,
        save_dir = save_dir,
        filename = f"{args.data_split}.pkl"
    )
    print("----- Saved AI2D dataset -----") 

