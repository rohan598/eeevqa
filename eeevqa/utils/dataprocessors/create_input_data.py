import rootutils

root = rootutils.setup_root(
    search_from=__file__,
    indicator="setup.py",
    pythonpath=True,
    dotenv=True,
)

import os
import numpy as np
import pickle

from collections import namedtuple

from eeevqa.utils.dataprocessors.helpers import create_html_file_modular, save_html_file, convert_html_to_pdf, convert_pdf_to_image, remove_white_space, image_creator, create_one_scienceqa_example
from eeevqa.utils.dataloaders.raw_data import read_captions, read_problem_list, read_pid_splits
from eeevqa.utils.args import parse_args, parse_boolean

# Convert Input to Image HTML pipeline
def convert_input_to_img_v1(problem_list, pid_splits, source="train", save_dir="", \
                         sample_subset = 10, crop_padding = 30, remove_html_file = True, remove_pdf_file = True, \
                         params=None):
    
    idx_list = pid_splits[source] 
    idx_list = [int(idx) for idx in idx_list]
    
    if sample_subset is not None:
        idx_list = idx_list[:sample_subset]
        source = "tiny_" + source
    
    # create save directory if it does not exist
    save_dir = os.path.join(save_dir, source)
        
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
        print("Save directory created")
    else:
        print("Save directory already present")
        
    
    for sample_num in idx_list:
        
        # create html template
        html_file = create_html_file_modular(params, problem_list, sample_num=sample_num, layout_type = params["layout_type"])

        # save tmp html file
        save_html_file(html_file, source, sample_num, save_dir=save_dir)
        
        # tmp and final filenames
        tmp_hpi_filename = os.path.join(os.getcwd(), save_dir, f"{source}_{sample_num}")
        img_filename = os.path.join(os.getcwd(), save_dir, f"{source}_{sample_num}")
        
        # convert tmp html to tmp pdf & delete tmp html file
        convert_html_to_pdf(tmp_hpi_filename, tmp_hpi_filename)
        
        # convert tmp pdf to image & delete tmp pdf file
        convert_pdf_to_image(tmp_hpi_filename, img_filename)
        
        # crop whitespace
        remove_white_space(f"{img_filename}.jpg")
        
        # cleaning by removing tmp files
        if remove_html_file:
            os.remove(f"{tmp_hpi_filename}.html")
        
        if remove_pdf_file:
            os.remove(f"{tmp_hpi_filename}.pdf")

# Convert Input to Image Render Text pipeline
def convert_input_to_img_v2(problem_list, pid_splits, params=None):
    
    source = params["data_source"]
    idx_list = pid_splits[source] 
    idx_list = [int(idx) for idx in idx_list]
    
    if params["sample_subset"] is not None:
        idx_list = idx_list[:params["sample_subset"]]
        source = "tiny_" + source
    
    stats_dict = {
        "original_size_w":[],
        "original_size_h":[],
        "final_size_w":[],
        "final_size_h":[]
    }
    # create save directory if it does not exist
    
    save_dir = os.path.join(save_dir, source)
        
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)
        print("Save directory created")
    else:
        print("Save directory already present")
        
    image_dir = os.path.join(os.getcwd(),"data",params["data_split"])
    
    for sample_num in idx_list:
        
        # call image converter
        stats_dict = image_creator(image_dir, problem_list, sample_num=sample_num, stats_dict=stats_dict, params=params)

    return stats_dict

def convert_scienceqa_to_dataset(problem_list, pid_splits, source="train", save_dir = "", output_format="AE", \
                                 options = None, 
                                 preprocess_image = None, sample_subset = None,
                                 task_name = None):
    
    
    idx_list = pid_splits[source] 
    idx_list = [int(idx) for idx in idx_list]
    
    if sample_subset is not None:
        idx_list = idx_list[:sample_subset]
        source = "tiny_"+source
        
    save_dir = os.path.join(save_dir, source)    
    
    dataset = []
    for sample_num in idx_list:
        ifile = os.path.join(os.getcwd(), save_dir, f"{source}_{sample_num}")
        dataset.append(create_one_scienceqa_example(problem_list, img_filename=ifile, \
                                                    sample_num=sample_num, output_format=output_format, \
                                                    options = options, preprocess_image = preprocess_image,
                                                    task_name = task_name))
    return dataset
        
# saving functionality
def save_dataset(dataset, save_dir="", filename=""):
    pickle_filename = os.path.join(save_dir, filename)
    with open(pickle_filename, 'wb') as f:
        pickle.dump(dataset, f)

if __name__ == '__main__':
    

    args = parse_args()
    print("----- Parsed Arguments -----")

    if args.task_name == "univqa":
        ScienceQA = namedtuple("ScienceQA", "sample_num header_text image image_mean image_std output")
    else:
        ScienceQA = namedtuple("ScienceQA", "sample_num header_text image text_context lecture image_mean image_std output")
    
    captions_dict = read_captions(args.data_root, args.captions_filename)
    problem_list = read_problem_list(os.path.join(args.data_root, args.json_files_dir), args.problems_filename)
    pid_splits = read_pid_splits(os.path.join(args.data_root, args.json_files_dir), args.pidsplits_filename)
    
    print("----- Read Dataset -----") 

    # set params for html
    params = {
        "set_question_as_header": parse_boolean(args.set_question_as_header),
        "skip_text_context":parse_boolean(args.skip_text_context),
        "skip_lecture":parse_boolean(args.skip_lecture),
        "options":args.options,
        "layout_type":args.layout_type
    }

    # pipeline for image dataset generation
    skip_image_gen = parse_boolean(args.skip_image_gen)
    save_dir = os.path.join(args.data_root, args.pickle_files_dir, args.data_version, args.data_type, str(args.layout_type))

    if os.path.exists(os.path.join(save_dir, args.data_split)) == False or skip_image_gen == False:
        convert_input_to_img_v1(problem_list, pid_splits, source=args.data_split,  
                        save_dir=save_dir, sample_subset = args.sample_subset, 
                        crop_padding = args.crop_padding, params = params)
        print("----- Created Image Collection -----") 
    
    skip_dataset_gen = parse_boolean(args.skip_dataset_gen)
    
    if skip_dataset_gen == False:
        # create dataset
        dataset = convert_scienceqa_to_dataset(problem_list, pid_splits, 
                        source=args.data_split, save_dir = save_dir, 
                        output_format=args.output_format,
                        options = args.options, preprocess_image = None, 
                        sample_subset = args.sample_subset,
                        task_name = args.task_name
                        ) 
        print("----- Created Dataset from Image Collection -----") 
        
        # save dataset
        save_dataset(
            dataset,
            save_dir = save_dir,
            filename = f"{args.data_split}.pkl"
        )
        print("----- Saved created dataset -----") 