import rootutils

root = rootutils.setup_root(
    search_from=__file__,
    indicator="setup.py",
    pythonpath=True,
    dotenv=True,
)

import os
from collections import namedtuple
import json
from sys import platform
from datetime import datetime
import torch
import pytorch_lightning
from pytorch_lightning import Trainer, seed_everything
from transformers import AutoProcessor, Pix2StructForConditionalGeneration

from eeevqa.utils.args import parse_args, parse_boolean
from eeevqa.utils.dataloaders.raw_data import read_problem_list
from eeevqa.utils.dataloaders.sciencevqa import create_eval_dataloader
from eeevqa.models.pix2struct.model import Pix2StructVanilla
from eeevqa.utils.eval.metrics import get_pred_idx, extract_answer, extract_explanation

def get_result_filepath(args, model_name, result_type):
    result_filename = "{}_{}_{}_{}_{}_{}_seed_{}.json".format(result_type, model_name, args.eval_split, args.output_format, args.max_patches, args.max_new_tokens, args.seed)

    result_file_dir = os.path.join(args.output_root, 
                                    args.results_dir,
                                    args.data_version,
                                    args.data_type,
                                    str(args.layout_type))
    
    if os.path.exists(result_file_dir) == False:
           os.makedirs(result_file_dir)

    result_filepath = os.path.join(result_file_dir, result_filename)
    

    return result_filepath

def save_results(result_filepath, data):
    with open(result_filepath, 'w') as f:
        json.dump(data, f, indent=2, separators=(',', ': '))

def gen_results(model, eval_dataloader, problem_list, args):
    
    data = {
        "results":None,
        "args":None
    }
    results = {}
    cnt = 1
    # iterator = iter(eval_dataloader)
    print(len(eval_dataloader))
    prev_p = None
    prev_a = None
    for data_sample in eval_dataloader:
        flattened_patches = data_sample.pop("flattened_patches")
        attention_mask = data_sample.pop("attention_mask")
        qid = data_sample.pop("sample_num")[0]
        if prev_p is not None and prev_p == flattened_patches:
            print("in here p")
        if prev_a is not None and prev_a == attention_mask:
            print("in here a")
        # print(qid)
        predictions = model.model.generate(flattened_patches=flattened_patches, attention_mask=attention_mask, max_new_tokens = args.max_new_tokens)  
        text_predictions = processor.batch_decode(predictions, skip_special_tokens=True)[0]

        # print(text_predictions)
        choices = problem_list[qid]["choices"]
        answer = problem_list[qid]["answer"]  # 0, 1, ..., 4 
        options = args.options

        pred_answer = get_pred_idx(
             extract_answer(text_predictions), 
             choices = choices, 
             options = options)
        

        pred_exp = extract_explanation(text_predictions)
        results[qid] = {}
        # TODO
        # can make this more efficient convert to pd

        results[qid]["pred_answer"] = pred_answer
        results[qid]["pred_exp"] = pred_exp
        results[qid]["no_context"] = True if (not problem_list[qid]["hint"] and not problem_list[qid]["image"]) else False
        results[qid]["has_text"] = True if problem_list[qid]["hint"] else False
        results[qid]["has_image"] = True if problem_list[qid]["image"] else False
        results[qid]["has_text_image"] = True if (problem_list[qid]["hint"] and problem_list[qid]["image"]) else False
        results[qid]["grade"] = problem_list[qid]["grade"]
        results[qid]["subject"] = problem_list[qid]["subject"]
        results[qid]["answer"] = problem_list[qid]["answer"]
        results[qid]["true_false"] = (problem_list[qid]["answer"] == results[qid]["pred_answer"])
        
        if cnt%50==0:
            print(f"Completed {cnt}")
    
        cnt+=1

    data = {
        "results":results,
        "args":vars(args)
    }

    return data

def get_acc_with_condition(results, query, values):

    correct_list = []
    if isinstance(values, list):
        for key in list(results.keys()):
            if query not in results[key].keys():
                continue
            if results[key][query] in values:
                correct_list.append(int(results[key]["true_false"]))
            else:
                continue
    else:
        for key in list(results.keys()):
            if query not in results[key].keys():
                continue
            if results[key][query]==values:
                correct_list.append(int(results[key]["true_false"]))
            else:
                continue
    
    if len(correct_list)==0:
        return 0.0
    
    acc = "{:.2f}".format((sum(correct_list)/ len(correct_list) )* 100)
    return acc

def get_scores(result_file):
    # read result file
    results = json.load(open(result_file))["results"]
    num = len(results)
    # assert num == 4241
    print("number of questions:", num)
       

    # accuracy scores
    acc_average = (sum([1 if results[key]["true_false"]==True else 0 for key in list(results.keys())])/ num) * 100 

    scores = {
        'acc_natural':
        get_acc_with_condition(results, 'subject', 'natural science'),
        'acc_social':
        get_acc_with_condition(results, 'subject', 'social science'),
        'acc_language':
        get_acc_with_condition(results, 'subject', 'language science'),
        'acc_has_text':
        get_acc_with_condition(results, 'has_text', True),
        'acc_has_image':
        get_acc_with_condition(results, 'has_image', True),
        'acc_no_context':
        get_acc_with_condition(results, 'no_context', True),
        'acc_has_text_image':
        get_acc_with_condition(results, 'has_text_image', True),
        'acc_grade_1_6':
        get_acc_with_condition(results, 'grade', ['grade1', 'grade2', 'grade3', 'grade4', 'grade5', 'grade6']),
        'acc_grade_7_12':
        get_acc_with_condition(results, 'grade', ['grade7', 'grade8', 'grade9', 'grade10', 'grade11', 'grade12']),
        'acc_average':
        "{:.2f}".format(acc_average),
    }

    return scores




if __name__ == '__main__':
    # Global settings
    seed_everything(42)
    torch.set_float32_matmul_precision('medium')

    args = parse_args()
    print("----- Parsed Arguments -----")

    ScienceQA = namedtuple("ScienceQA", "sample_num image flattened_patches attention_mask raw_output output")
    
    problem_list = read_problem_list(os.path.join(args.data_root, args.json_files_dir), args.problems_filename)
    print("----- Read Dataset -----") 

    # load processor
    processor = AutoProcessor.from_pretrained(args.base_model_name)
    p2smodel = Pix2StructVanilla(
                model_name_or_path = args.base_model_name,
                problem_list = problem_list,
                options = args.options,
                task_name = args.task_name,
                processor=processor,
                max_new_tokens = args.max_new_tokens,
                output_format=args.output_format,
    )

    model_name = p2smodel.__class__.__name__
    model_output_filepath = get_result_filepath(args, model_name, "model_output")
    model_acc_scores_filepath = get_result_filepath(args, model_name, "model_acc_scores")
    print("----- Model Setup -----")

    skip_model_output_gen = parse_boolean(args.skip_model_output_gen)
    if skip_model_output_gen == False:
        
        # load model
        checkpoint_path = os.path.join(args.output_root, args.checkpoint_dir, args.data_version, args.data_type, str(args.layout_type), args.eval_checkpoint_name)
        
        model = p2smodel.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            map_location=torch.device('cpu')
        )
        print("----- Model Loaded -----")

        # create eval dataloader - optional
        pickle_files_path = os.path.join(args.data_root, args.pickle_files_dir, args.data_version, args.data_type, str(args.layout_type))

        batch_size = 1
        eval_dataloader = create_eval_dataloader(pickle_files_path, args.eval_split, processor, args.max_patches, args.output_format, batch_size) 

       
        
        model_output = gen_results(model, eval_dataloader, problem_list, args)
        print("----- Model Outputs Generated -----")

        save_results(model_output_filepath, model_output)
        print("----- Model Outputs Saved -----")

    skip_model_score_pred = parse_boolean(args.skip_model_score_pred)
    if skip_model_score_pred == False:
   
        acc_scores = get_scores(model_output_filepath)
        print("----- Model Accuracy Scores Generated -----")

        save_results(model_acc_scores_filepath, acc_scores)
        print("----- Model Accuracy Scores Saved -----")


    # # create trainer
    # trainer = Trainer(
    #     accelerator="gpu",
    #     devices=args.gpu_cnt if torch.cuda.is_available() else None,
    #     strategy="ddp",  
    # )

    # #test model
    # trainer.test(model, dataloaders=eval_dataloader)




