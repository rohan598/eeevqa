import rootutils

root = rootutils.setup_root(
    search_from=__file__,
    indicator="setup.py",
    pythonpath=True,
    dotenv=True,
)
import os
import time
from PIL import Image
from transformers import Pix2StructForConditionalGeneration, AutoProcessor

from eeevqa.utils.args import parse_args
from eeevqa.utils.dataloaders.raw_data import load_pickle_dataset

def evaluate_model_AI2D(image_list, samples_dict, model, processor, max_patches, data_dir):
    # question_text = ""
    cnt = 0
    correct_cnt = 0
    start_time = time.time()
    model = model.to('cuda')
    for image_file in image_list:
        
        image_idx = (image_file.split('.')[0]+ "."+ image_file.split('.')[1]).split("/")[-1]

        image_file_idx = (image_file.split('.')[0]).split('/')[-1] + ".png"


        test_image = Image.open(os.path.join(data_dir, "new_data", "images-self",f"{image_idx}.png"))

        # test_image = Image.open(image_file)

        # test_image = Image.open(os.path.join(data_dir,"images",image_file_idx))
        
        question_text = "" 
        # question_text = samples_dict[image_idx]["header_text"]
        inputs = processor(images=test_image, return_tensors="pt", max_patches=max_patches).to('cuda')
        generated_ids = model.generate(**inputs, max_new_tokens=52)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        if generated_text==samples_dict[image_idx]["raw_output"]:
            correct_cnt+=1
        
        cnt+=1
        # if cnt == 20:
        #     break

    end_time = time.time()
    print(end_time-start_time)
    
    final_accuracy = (1.0*correct_cnt)/cnt
    print(f"exact match accuracy {final_accuracy}")

    return final_accuracy
    

if __name__ == '__main__':
    

    args = parse_args()
    print("----- Parsed Arguments -----")

    samples_dict = load_pickle_dataset(os.path.join(args.data_root, "new_data"), "samples_dict")

    metadata_dict = load_pickle_dataset(os.path.join(args.data_root, "new_data"), "metadata_dict")

    processor = AutoProcessor.from_pretrained("google/pix2struct-ai2d-base")
    processor.image_processor.is_vqa = False
    model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-ai2d-base")

    final_accuracy = evaluate_model_AI2D(metadata_dict["test_split_new_images"], samples_dict, model, processor, args.max_patches, args.data_root)

    print(f"----- Benchmarked AI2D: final accuracy {final_accuracy*100}% -----") 

