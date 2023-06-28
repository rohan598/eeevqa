import rootutils

root = rootutils.setup_root(
    search_from=__file__,
    indicator="setup.py",
    pythonpath=True,
    dotenv=True,
)

import os

from eeevqa.utils.dataprocessors.helpers import save_dataset
from eeevqa.utils.args import parse_args, parse_boolean

# make all text targets at once
def create_text_data(problem_list, output_format="AE", max_new_tokens=512, processor=None):
    
    text_data = {
        "targets": [],
        "raw_output": [],
        "sample_num":[]
    }

    # group all text inputs
    for sample_num in range(1, len(problem_list.keys())+1):  
        # Outputs
        if output_format == 'A':
            raw_output = f"Answer: The answer is {options[problem_list[sample_num]['answer']]}."
        elif output_format == 'AE':
            raw_output = f"Answer: The answer is {options[problem_list[sample_num]['answer']]}. BECAUSE: {problem_list[sample_num]['solution']}"
        elif output_format == 'EA':
            raw_output = f"Answer: {problem_list[sample_num]['solution']} The answer is {options[problem_list[sample_num]['answer']]}."

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

    problem_list = read_problem_list(os.path.join(args.data_root, args.json_files_dir), args.problems_filename)

    save_dir = os.path.join(args.data_root, args.pickle_files_dir)

    processor = AutoProcessor.from_pretrained(args.base_model_name)
    print("----- Basic Setup done -----") 

    text_data = create_text_data(problem_list,
                            output_format=args.output_format, 
                            max_new_tokens=args.max_new_tokens, 
                            processor=processor)
    print("----- Created Text Dataset -----") 

    save_dataset(
        text_data,
        save_dir = save_dir,
        filename = f"text_data_{args.max_new_tokens}.pkl"
    )
    print("----- Saved Text Dataset -----") 
