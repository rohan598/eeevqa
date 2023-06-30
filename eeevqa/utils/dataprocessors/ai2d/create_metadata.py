import rootutils

root = rootutils.setup_root(
    search_from=__file__,
    indicator="setup.py",
    pythonpath=True,
    dotenv=True,
)

import os
import pandas as pd

from eeevqa.utils.dataloaders.raw_data import read_json_file, save_dataset, load_pickle_dataset

from eeevqa.utils.dataprocessors.ai2d.create_image_data import convert_AI2D


from eeevqa.utils.args import parse_args, parse_boolean

def create_metadata(data_dir):
    
    metadata_dict = {}
    tlist = []
    for file in os.listdir(os.path.join(data_dir, "images")):
        fidx = int(file.split(".")[0])
        tlist.append(fidx)
    
    skipped_image_num_list = []
    for i in range(4907+1):
        if i not in tlist:
            skipped_image_num_list.append(i)
    print(skipped_image_num_list)
    
    metadata_dict["skipped_image_num_list"] = skipped_image_num_list
    
    all_samples_list = []
    for file in os.listdir(os.path.join(data_dir, "images")):
        fidx = int(file.split(".")[0])
        all_samples_list.append(fidx)
        
    print(len(all_samples_list))
    metadata_dict["all_samples_list"] = all_samples_list
    
    all_test_list = list(pd.read_csv(os.path.join(data_dir,"ai2d_test_ids.csv")).to_numpy().reshape(-1))
    print(len(all_test_list))
    metadata_dict["all_test_list"] = all_test_list
    
    missing_from_all_test_list = []
    all_test_present_list = []
    for sidx in all_test_list:
        if sidx not in all_samples_list:
            missing_from_all_test_list.append(sidx)
        else:
            all_test_present_list.append(sidx)
    
    print(missing_from_all_test_list)
    print(len(missing_from_all_test_list))
    print(len(all_test_present_list))
    
    metadata_dict["missing_from_all_test_list"] = missing_from_all_test_list
    metadata_dict["all_test_present_list"] = all_test_present_list
    
    missing_question_list = load_pickle_dataset(os.path.join(data_dir, "new_data"), "missing_question_list")
    all_samples_question_list = []
    for sidx in all_samples_list:
        if sidx not in missing_question_list:
            all_samples_question_list.append(sidx)
            
    print(len(all_samples_question_list))
    print(len(missing_question_list))
    assert((len(all_samples_question_list) + len(missing_question_list)) == len(all_samples_list))
    metadata_dict["all_samples_question_list"] = all_samples_question_list
    
    
    test_split_src_image = []
    cnt = 0
    for sidx in all_samples_question_list:
        if sidx in all_test_present_list:
            test_split_src_image.append(sidx)
        else:
            cnt+=1

    print(len(test_split_src_image))
    assert((cnt+len(test_split_src_image)) == len(all_samples_question_list))

    missing_test_question_list = []
    cnt2 = 0
    for sidx in all_test_present_list:
        if sidx not in test_split_src_image:
            missing_test_question_list.append(sidx)
        else:
            cnt2+=1

    print(len(missing_test_question_list))
    assert((cnt2+len(missing_test_question_list)) == len(all_test_present_list))
    
    metadata_dict["test_split_src_image"] = test_split_src_image
    metadata_dict["missing_test_question_list"] = missing_test_question_list
    
    train_split_src_image = []

    cnt3 = 0
    for sidx in all_samples_question_list:
        if sidx not in test_split_src_image:
            train_split_src_image.append(sidx)
        else:
            cnt3+=1

    print(len(train_split_src_image))
    assert((cnt3+len(train_split_src_image))==len(all_samples_question_list))

    missing_train_question_list = []
    cnt4 = 0
    for sidx in all_samples_list:
        if sidx not in all_test_present_list and sidx not in train_split_src_image:
            missing_train_question_list.append(sidx)
        else:
            cnt4+=1
    print(len(missing_train_question_list))
    assert((cnt4+len(missing_train_question_list))==(len(all_samples_list)))
    
        
    metadata_dict["train_split_src_image"] = train_split_src_image
    metadata_dict["missing_train_question_list"] = missing_train_question_list
    
    print(len(train_split_src_image))
    print(len(test_split_src_image))
    print(f"total src images used for vqa ai2d: {len(all_samples_question_list)} = {len(train_split_src_image)+len(test_split_src_image)}")
    
    print(len(missing_train_question_list))
    print(len(missing_test_question_list))
    print(f"total src images skipped for vqa ai2d: {len(missing_question_list)} = {len(missing_train_question_list)+len(missing_test_question_list)}")

    test_split_new_images = []
    train_split_new_images = []
    image_dir = os.path.join(data_dir,"new_data","images")
    cnt5=0
    for file in os.listdir(image_dir):
        if int(file.split('.')[0]) in test_split_src_image:
            test_split_new_images.append(os.path.join(image_dir, file))

        else:
            train_split_new_images.append(os.path.join(image_dir, file))

        cnt5+=1
    
    print(len(train_split_new_images))
    print(len(test_split_new_images))
    print(cnt5)
    assert(cnt5==(len(train_split_new_images) + len(test_split_new_images)))
    
    metadata_dict["train_split_new_images"] = train_split_new_images    
    metadata_dict["test_split_new_images"] = test_split_new_images
    
    metadata_dict["tiny_train_split_new_images"] = train_split_new_images[:100]    
    metadata_dict["tiny_test_split_new_images"] = test_split_new_images[:20]

    return metadata_dict



if __name__ == '__main__':

    args = parse_args()
    print("----- Parsed Arguments -----")

    samples_dict, missing_question_list = convert_AI2D(
        data_dir = args.data_root, 
        font_path = os.path.join(args.assets_dir,"arial.ttf"), 
        skip_image_gen = True
    )

    save_dataset(
        samples_dict,
        save_dir = os.path.join(args.data_root,"new_data"),
        filename = "samples_dict.pkl"
    )
    
    save_dataset(
        missing_question_list,
        save_dir = os.path.join(args.data_root,"new_data"),
        filename = "missing_question_list.pkl"
    )  

    print("----- Saved Samples Dict and Missing Question List -----") 

    metadata_dict = create_metadata(args.data_root)
    print("----- Created remaining metadata -----") 

    save_dataset(
        metadata_dict,
        save_dir = os.path.join(args.data_root,"new_data"),
        filename = "metadata_dict.pkl"
    )  
    print("----- Saved remaining metadata -----") 