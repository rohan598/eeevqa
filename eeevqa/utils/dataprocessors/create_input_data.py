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
from transformers import AutoProcessor

from eeevqa.utils.dataprocessors.helpers import input_to_image_initialization, create_html_file_modular, save_html_file, convert_html_to_pdf, convert_pdf_to_image, remove_white_space, image_creator, create_one_scienceqa_example, create_one_scienceqa_example_v2
from eeevqa.utils.dataloaders.raw_data import read_captions, read_problem_list, read_pid_splits
from eeevqa.utils.args import parse_args, parse_boolean

# Convert Input to Image HTML pipeline
def convert_input_to_img_v1(problem_list, pid_splits, params=None):
    
    data_split, idx_list, save_dir, stats_dict = input_to_image_initialization(problem_list=problem_list, pid_splits=pid_splits, params=params)
        
    for sample_num in idx_list:
        
        # create html template
        html_file = create_html_file_modular(params, problem_list, sample_num=sample_num, layout_type = params["layout_type"])

        # save tmp html file
        save_html_file(html_file, data_split, sample_num, save_dir=save_dir)
        
        # tmp and final filenames
        tmp_hpi_filename = os.path.join(save_dir, f"{data_split}_{sample_num}")
        img_filename = os.path.join(save_dir, f"{data_split}_{sample_num}")
        
        # convert tmp html to tmp pdf & delete tmp html file
        convert_html_to_pdf(tmp_hpi_filename, tmp_hpi_filename)
        
        # convert tmp pdf to image & delete tmp pdf file
        convert_pdf_to_image(tmp_hpi_filename, img_filename)
        
        # crop whitespace
        remove_white_space(f"{img_filename}.jpg", crop_padding=params["crop_padding"], visualize=params["visualize"], stats_dict=stats_dict)
        
        # cleaning by removing tmp files
        if params["remove_html_file"]:
            os.remove(f"{tmp_hpi_filename}.html")
        
        if params["remove_pdf_file"]:
            os.remove(f"{tmp_hpi_filename}.pdf")

    return stats_dict

# Convert Input to Image Render Text pipeline
def convert_input_to_img_v2(problem_list, pid_splits, params=None):
    
    data_split, idx_list, save_dir, stats_dict = input_to_image_initialization(problem_list=problem_list, pid_splits=pid_splits, params=params)
    image_dir = os.path.join(params["data_root"], params["data_source"])
    print(image_dir)
    for sample_num in idx_list:
        
        # call image converter
        params["save_path"] = os.path.join(save_dir, f"{data_split}_{sample_num}.jpg")
        stats_dict = image_creator(image_dir, problem_list, sample_num=sample_num, stats_dict=stats_dict, params=params)

    return stats_dict

def convert_scienceqa_to_dataset(problem_list, pid_splits, params=None):
    
    data_split, idx_list, save_dir, _ = input_to_image_initialization(problem_list=problem_list, pid_splits=pid_splits, params=params) 
    
    dataset = []
    ScienceQA = params["ScienceQA"]
    for sample_num in idx_list:
        img_filename = os.path.join(os.getcwd(), save_dir, f"{data_split}_{sample_num}")
        dataset.append(create_one_scienceqa_example_v2(problem_list, img_filename=img_filename, \
                                                    sample_num=sample_num, output_format=params["output_format"], \
                                                    options = params["options"], 
                                                    max_new_tokens = params["max_new_tokens"],
                                                    max_patches=params["max_patches"],
                                                    processor=params["processor"],
                                                    ScienceQA = ScienceQA))
    return dataset

# def convert_scienceqa_to_dataset(problem_list, pid_splits, params=None):
    
#     data_split, idx_list, save_dir, _ = input_to_image_initialization(problem_list=problem_list, pid_splits=pid_splits, params=params) 
    
#     dataset = []
#     ScienceQA = params["ScienceQA"]
#     for sample_num in idx_list:
#         img_filename = os.path.join(os.getcwd(), save_dir, f"{data_split}_{sample_num}")
#         dataset.append(create_one_scienceqa_example(problem_list, img_filename=img_filename, \
#                                                     sample_num=sample_num, output_format=params["output_format"], \
#                                                     options = params["options"], preprocess_image = params["preprocess_image"],
#                                                     task_name = params["task_name"],
#                                                     ScienceQA = ScienceQA))
#     return dataset
        
# saving functionality
def save_dataset(dataset, save_dir="", filename=""):
    pickle_filename = os.path.join(save_dir, filename)
    with open(pickle_filename, 'wb') as f:
        pickle.dump(dataset, f)

if __name__ == '__main__':
    

    args = parse_args()
    print("----- Parsed Arguments -----")

    ScienceQA = namedtuple("ScienceQA", "sample_num image flattened_patches attention_mask raw_output output")
    # if args.task_name == "univqa":
    #     ScienceQA = namedtuple("ScienceQA", "sample_num header_text image image_mean image_std output")
    # else:
    #     ScienceQA = namedtuple("ScienceQA", "sample_num header_text image text_context lecture image_mean image_std output")
    
    captions_dict = read_captions(args.data_root, args.captions_filename)
    problem_list = read_problem_list(os.path.join(args.data_root, args.json_files_dir), args.problems_filename)
    pid_splits = read_pid_splits(os.path.join(args.data_root, args.json_files_dir), args.pidsplits_filename)
    save_dir = os.path.join(args.data_root, args.pickle_files_dir, args.data_version, args.data_type, str(args.layout_type))

    print("----- Read Dataset -----") 

    processor = AutoProcessor.from_pretrained(args.base_model_name)
    # set common params 
    params = {
        "task_name":args.task_name,
        "data_version":args.data_version,
        "data_type":args.data_type,
        "data_root":args.data_root,
        "data_source":args.data_source,
        "data_split":args.data_split,
        "layout_type":args.layout_type,
        "sample_subset":args.sample_subset if args.sample_subset!="" else None,
        "save_dir":save_dir,
        "options":args.options,
        "output_format":args.output_format,
        "skip_text_context":parse_boolean(args.skip_text_context),
        "skip_lecture":parse_boolean(args.skip_lecture),
        "visualize":parse_boolean(args.visualize_gen),
        "preprocess_image":None,
        "max_patches":args.max_patches,
        "max_new_tokens":args.max_new_tokens,
        "processor":processor,
        "ScienceQA":ScienceQA
    }

    # pipeline for image dataset generation
    skip_image_gen = parse_boolean(args.skip_image_gen)

    if os.path.exists(os.path.join(save_dir, args.data_split)) == False or skip_image_gen == False:

        if args.data_version == "v1":
            # set params for v1 and update original params
            params_v1 = {
                "set_question_as_header": parse_boolean(args.set_question_as_header_dgv1),
                "crop_padding":args.crop_padding_dgv1,
                "remove_html_file":parse_boolean(args.remove_html_file_dgv1),
                "remove_pdf_file":parse_boolean(args.remove_pdf_file_dgv1),
            }
            params.update(params_v1)

            convert_input_to_img_v1(problem_list, pid_splits, params = params)

        elif args.data_version == "v2":
            # set params for v2 and update original params
            params_v2 = {
                "header_size":args.header_size_dgv2,
                "text_context_size":args.text_context_size_dgv2,
                "text_color":args.text_color_dgv2,
                "background_color":args.background_color_dgv2,
                "top_padding":args.top_padding_dgv2,
                "right_padding":args.right_padding_dgv2,
                "bottom_padding":args.bottom_padding_dgv2,
                "left_padding":args.left_padding_dgv2,
                "path_to_font":args.assets_dir,
                "analyze":parse_boolean(args.analyze_dgv2),
                "save":True
            }
            params.update(params_v2)

            convert_input_to_img_v2(problem_list, pid_splits, params = params)
        
        print(f"----- Created Image {args.data_version} Collection -----") 
    
    skip_dataset_gen = parse_boolean(args.skip_dataset_gen)
    
    if skip_dataset_gen == False:
        # create dataset
        dataset = convert_scienceqa_to_dataset(
                    problem_list, 
                    pid_splits,
                    params=params
                ) 
        print("----- Created Dataset from Image Collection -----") 
        
        data_split = args.data_split

        if args.sample_subset!="":
            try:
                int(args.sample_subset)
            except:
                data_split = args.sample_subset + "_" +  args.data_split

        # save dataset
        save_dataset(
            dataset,
            save_dir = save_dir,
            filename = f"{data_split}.pkl"
        )
        print("----- Saved created dataset -----") 