import rootutils

root = rootutils.setup_root(
    search_from=__file__,
    indicator="setup.py",
    pythonpath=True,
    dotenv=True,
)


import os
import string 
from PIL import Image
from eeevqa.utils.dataloaders.raw_data import read_json_file, save_dataset
from eeevqa.utils.dataprocessors.ai2d.helpers import render_header, render_text_on_bounding_box

from eeevqa.utils.args import parse_args

def convert_one_question_AI2D(input_path: str, data_dir: str, font_path: str, skip_image_gen: bool):
  
    """Convert example."""
    samples_dict = {}
    
    data = read_json_file(os.path.join(data_dir, input_path)) # till ai2d folder + question path
    
    qid = -1
    if not data["questions"]:
        return samples_dict, int(data["imageName"].split('.')[0])
    
    annotation = read_json_file(os.path.join(data_dir, "annotations", f"{data['imageName']}.json"))
    
    if skip_image_gen == False:
        image_filepath = os.path.join(data_dir, "images", data["imageName"])
        image = Image.open(image_filepath)
        image_with_placeholders = image.copy()

        for v in annotation["text"].values():
            render_text_on_bounding_box(
                text=v["replacementText"],
                bounding_box=v["rectangle"],
                image=image_with_placeholders,
                font_path = font_path)

    for k, v in data["questions"].items():
        
        samples_dict[v["questionId"]] = {}
        # The `image_id` field is only used to ensure correct splitting of the data.
        options = " ".join(
            f"({string.ascii_lowercase[i]}) {a}"
            for i, a in enumerate(v["answerTexts"])
        )
        
        if skip_image_gen == False:
            # image_with_header = render_header(
            #     image=image_with_placeholders if v["abcLabel"] else image,
            #     header=f"{k} {options}",
            #     font_path = font_path
            # )

            # save new image
            save_dir = os.path.join(data_dir, "new_data","images2")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # image_with_header.save(os.path.join(save_dir, f"{v['questionId']}.png"))

            image_with_placeholders.save(os.path.join(save_dir, f"{v['questionId']}.png"))

        # get output for this sample
        parse = v["answerTexts"][v["correctAnswer"]]
        
        
        # update sample dict with info and meta
        samples_dict[v["questionId"]]["src_image_name"] = data["imageName"]
        samples_dict[v["questionId"]]["raw_output"] = parse
        samples_dict[v["questionId"]]["header_text"] = f"{k} {options}"
        samples_dict[v["questionId"]]["abcLabel"] = v["abcLabel"]
    
    
    return samples_dict, qid


def convert_AI2D(data_dir: str, font_path: str, skip_image_gen: bool):
    
    # list question folder files
    # create sample dict
    samples_dict = {}
    missing_question_list = []
    
    for file in os.listdir(os.path.join(data_dir,"questions")):
        filepath = os.path.join("questions", file)
        one_question_sample_dict, qid = convert_one_question_AI2D(filepath, data_dir, font_path, skip_image_gen)
        samples_dict.update(one_question_sample_dict)
        
        if qid!=-1:
            missing_question_list.append(qid)
        

    return samples_dict, missing_question_list

if __name__ == '__main__':
    

    args = parse_args()
    print("----- Parsed Arguments -----")

    samples_dict, missing_question_list = convert_AI2D(
        data_dir = args.data_root, 
        font_path = os.path.join(args.assets_dir,"arial.ttf"), 
        skip_image_gen = False
    )
    
    print("----- Created Image Collection -----") 

