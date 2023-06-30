import rootutils

root = rootutils.setup_root(
    search_from=__file__,
    indicator="setup.py",
    pythonpath=True,
    dotenv=True,
)

import os
from transformers import AutoProcessor

from eeevqa.utils.dataloaders.raw_data import read_json_file, save_dataset
from eeevqa.utils.args import parse_args, parse_boolean

# make all text targets at once
def create_text_data(problem_list, output_format="AE", max_new_tokens=512, options = None, processor=None):
    
    text_data = {
        "targets": [],
        "raw_output": [],
        "sample_num":[]
    }

    # group all text inputs
    for sample_num in range(1, len(problem_list)+1):  
        # Outputs
        if output_format == 'A':
            raw_output = f"Answer: The answer is {options[problem_list[str(sample_num)]['answer']]}."
        elif output_format == 'AE':
            raw_output = f"Answer: The answer is {options[problem_list[str(sample_num)]['answer']]}. BECAUSE: {problem_list[str(sample_num)]['solution']}"
        elif output_format == 'EA':
            raw_output = f"Answer: {problem_list[str(sample_num)]['solution']} The answer is {options[problem_list[str(sample_num)]['answer']]}."

        raw_output = raw_output.replace("  ", " ").strip()
        if raw_output.endswith("BECAUSE:"):
            raw_output = raw_output.replace("BECAUSE:", "").strip()
            
        text_data["raw_output"].append(raw_output)
        text_data["sample_num"].append(sample_num)
        
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

    problem_list_path = os.path.join(args.data_root, args.json_files_dir, args.problems_filename)
    problem_list = read_json_file(problem_list_path)

    save_dir = os.path.join(args.data_root, args.pickle_files_dir)

    processor = AutoProcessor.from_pretrained(args.base_model_name)
    print("----- Basic Setup done -----") 

    text_data = create_text_data(problem_list,
                            output_format=args.output_format, 
                            max_new_tokens=args.max_new_tokens, 
                            options = args.options,
                            processor=processor)
    print("----- Created Text Dataset -----") 

    save_dataset(
        text_data,
        save_dir = save_dir,
        filename = f"text_data_{args.max_new_tokens}.pkl"
    )
    print("----- Saved Text Dataset -----") 
