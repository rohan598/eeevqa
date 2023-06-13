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

from eeevqa.utils.args import parse_args
from eeevqa.utils.dataloaders.raw_data import read_problem_list
from eeevqa.utils.dataloaders.sciencevqa import create_eval_dataloader
from eeevqa.models.pix2struct.model import Pix2StructVanilla

def save_results(result_file, outputs):
    data = {}
    # data['results'] = results
    data['outputs'] = outputs

    with open(result_file, 'w') as f:
        json.dump(data, f, indent=2, separators=(',', ': '))

def gen_results(model, eval_dataloader, problem_list, result_file):
    # results = {}
    outputs = {}
    # correct = 0
    # count = 0
    # iterator = iter(eval_dataloader)
    for data_sample in eval_dataloader:

        flattened_patches = data_sample.pop("flattened_patches")
        attention_mask = data_sample.pop("attention_mask")
        qid = data_sample.pop("sample_num")[0]
        print(qid)
        predictions = model.model.generate(flattened_patches=flattened_patches, attention_mask=attention_mask)  
        text_predictions = processor.batch_decode(predictions, skip_special_tokens=True)[0]

        print(text_predictions)
        # choices = problem_list[qid]["choices"]
        # answer = problem_list[qid]["answer"]  # 0, 1, ..., 4 

        # pred_answer = text_predictions.split(".")[0].split("The answer is ")[1].strip()
        # if pred_answer in choices:
        #     pred_answer = choices.index(pred_answer)

        # pred_exp = text_predictions.split(".")[1]
        # count+=1
        
        # results[qid] = pred_answer
        # outputs[qid] = pred_exp
 
        # if pred_answer == answer:
        #     correct += 1

    save_results(result_file, text_predictions)


# def get_scores(result_file, data_file, pid_split):
#     # read result file
#     results = json.load(open(result_file))["results"]
#     num = len(results)
#     # assert num == 4241
#     print("number of questions:", num)

#     # read data file
#     sqa_data = json.load(open(data_file))

#     # construct pandas data
#     sqa_pd = pd.DataFrame(sqa_data).T
#     res_pd = sqa_pd[sqa_pd['split'] == eval_split]  # test set

#     # update data
#     for i in range(len(pid_split)):

#     for index, row in res_pd.iterrows():

#         res_pd.loc[index, 'no_context'] = True if (not row['hint'] and not row['image']) else False
#         res_pd.loc[index, 'has_text'] = True if row['hint'] else False
#         res_pd.loc[index, 'has_image'] = True if row['image'] else False
#         res_pd.loc[index, 'has_text_image'] = True if (row['hint'] and row['image']) else False

#         label = row['answer']
#         pred = int(results[index])
#         res_pd.loc[index, 'pred'] = pred
#         res_pd.loc[index, 'true_false'] = (label == pred)

#     # accuracy scores
#     acc_average = len(res_pd[res_pd['true_false'] == True]) / num * 100
#     #assert result_file.split('_')[-1] == "{:.3f}.json".format(acc_average)

#     scores = {
#         'acc_natural':
#         get_acc_with_condition(res_pd, 'subject', 'natural science'),
#         'acc_social':
#         get_acc_with_condition(res_pd, 'subject', 'social science'),
#         'acc_language':
#         get_acc_with_condition(res_pd, 'subject', 'language science'),
#         'acc_has_text':
#         get_acc_with_condition(res_pd, 'has_text', True),
#         'acc_has_image':
#         get_acc_with_condition(res_pd, 'has_image', True),
#         'acc_no_context':
#         get_acc_with_condition(res_pd, 'no_context', True),
#         'acc_grade_1_6':
#         get_acc_with_condition(res_pd, 'grade', ['grade1', 'grade2', 'grade3', 'grade4', 'grade5', 'grade6']),
#         'acc_grade_7_12':
#         get_acc_with_condition(res_pd, 'grade', ['grade7', 'grade8', 'grade9', 'grade10', 'grade11', 'grade12']),
#         'acc_average':
#         "{:.2f}".format(acc_average),
#     }

#     return scores




if __name__ == '__main__':
    # Global settings
    seed_everything(42)
    torch.set_float32_matmul_precision('medium')

    print("----- Parsed Arguments -----")
    args = parse_args()

    print("----- Read Dataset -----") 
    if args.task_name == "univqa":
        ScienceQA = namedtuple("ScienceQA", "sample_num header_text image image_mean image_std output")
    else:
        ScienceQA = namedtuple("ScienceQA", "sample_num header_text image text_context lecture image_mean image_std output")
    
    problem_list = read_problem_list(os.path.join(args.data_root, args.json_files_dir), args.problems_filename)

    # load processor
    processor = AutoProcessor.from_pretrained(args.base_model_name)

    p2smodule = Pix2StructVanilla(
            model_name_or_path = args.base_model_name,
            problem_list = problem_list,
            options = args.options,
            task_name = args.task_name,
            processor=processor,
            learning_rate = args.learning_rate,
            adam_epsilon = 1e-8,
            train_batch_size = args.train_batch_size,
            eval_batch_size = args.eval_batch_size,
            output_format=args.output_format,
            warmup_steps = args.warmup_steps,
            total_steps = args.total_steps,
            cycles = args.cycles
    )

    # load model
    checkpoint_path = os.path.join(args.output_root, args.checkpoint_dir, args.eval_checkpoint_name)
    
    model = p2smodule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        map_location=torch.device('cpu')
    )

    # model = Pix2StructForConditionalGeneration.from_pretrained(checkpoint_path=checkpoint_path)

    # create eval dataloader - optional
    eval_dataloader = create_eval_dataloader(os.path.join(args.data_root, args.pickle_files_dir, args.data_type), args.eval_split, processor, args.max_patches, args.output_format, args.eval_batch_size)


    result_file_path = os.path.join(args.output_root, args.result_filename)
    gen_results(model, eval_dataloader, problem_list, result_file_path)

    

    # # create trainer
    # trainer = Trainer(
    #     accelerator="gpu",
    #     devices=args.gpu_cnt if torch.cuda.is_available() else None,
    #     strategy="ddp",  
    # )

    # #test model
    # trainer.test(model, dataloaders=eval_dataloader)




