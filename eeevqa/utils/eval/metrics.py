import re
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import util
import random
from torchmetrics import Metric
import torch
import pdb

## Evaluation helper functions

def create_result_dict(result_list, qids):
    return dict([(qids[i], result_list[i]) \
                           for i in range(len(result_list))])

def extract_explanation(text):
    res_text = re.sub(r"The answer is [0-9]. BECAUSE: ", "", text)
    # res_text = re.sub(r"The answer is [A-Z]. BECAUSE: ", "", text)
    if len(res_text) == text:
        res_text = res_text.split("BECAUSE:")
        if len(res_text[0])==text:
            return text
        else:
            return res_text[1].strip()
    return res_text

def extract_answer(output):
    # print(output)
    pattern = re.compile(r'The answer is ([0-9]).')
    # pattern = re.compile(r'The answer is ([A-Z]).')
    res = pattern.findall(output)
    # print(res)
    
    if len(res) == 1:
        answer = res[0]  # 'A', 'B', ...
    else:
        answer = "FAILED"
    return answer

def get_pred_idx(prediction, choices, options):
    """
    Get the index (e.g. 2) from the prediction (e.g. 'C')
    """
    if prediction in options[:len(choices)]:
        return options.index(prediction)
    
    return random.choice(range(len(choices)))


def get_answer_pair(preds, qids, problem_list, options):
    
    target = []
    predicted = []
    for i in range(len(preds)):
        target_idx = problem_list[qids[i]]["answer"]
        target.append(target_idx)

        pred_idx = extract_answer(preds[i])
        if pred_idx == "FAILED":
            pred_idx = (target_idx + 1) % len(problem_list[qids[i]]["choices"])
        else:
            pred_idx = int(pred_idx)

        # print(pred_idx)
        # pred_idx = get_pred_idx(extract_answer(preds[i]), \
        #                                      problem_list[qids[i]]["choices"], options = options)

        predicted.append(pred_idx)
    
    return torch.tensor(predicted), torch.tensor(target)


def get_explanation_pair(preds, qids, problem_list):
    
    target = []
    predicted = []
    for i in range(len(preds)):
        pred_exp = extract_explanation(preds[i])
        predicted.append(pred_exp)
        
        target_exp = problem_list[qids[i]]["solution"].strip()
        target.append(target_exp)
    
    return predicted, target


def calculate_rouge(results, data): # expects results to be dictionary
    rouges = []
    for qid, output in results.items():
        prediction = extract_explanation(output)
        target = data[qid]["solution"]
        target = target.strip()
        if prediction == "":
            continue
        if target == "":
            continue
        rouge = score_rouge(target, prediction)
        rouges.append(rouge)

    avg_rouge = sum(rouges) / len(rouges)
    return avg_rouge


########################
## BLEU
########################
def tokenize(text):
    tokens = re.split(r'\s|\.', text)
    tokens = [t for t in tokens if len(t) > 0]
    return tokens


def bleu_score(reference, hypothesis, gram):
    reference_tokens = tokenize(reference)
    hypothesis_tokens = tokenize(hypothesis)

    if gram == 1:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1., ))  # BELU-1
    elif gram == 2:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1. / 2., 1. / 2.))  # BELU-2
    elif gram == 3:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1. / 3., 1. / 3., 1. / 3.))  # BELU-3
    elif gram == 4:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1. / 4., 1. / 4., 1. / 4., 1. / 4.))  # BELU-4

    return bleu


def calculate_bleu(results, data, gram):
    bleus = []
    for qid, output in results.items():
        prediction = extract_explanation(output)
        target = data[qid]["lecture"] + " " + data[qid]["solution"]
        target = target.strip()
        if target == "":
            continue
        bleu = bleu_score(target, prediction, gram)
        bleus.append(bleu)

    avg_bleu = sum(bleus) / len(bleus)

    return avg_bleu


########################
## Rouge-L
########################
def score_rouge(str1, str2):
    rouge = Rouge(metrics=["rouge-l"])
    scores = rouge.get_scores(str1, str2, avg=True)
    rouge_l = scores['rouge-l']['f']
    return rouge_l

def calculate_rouge(results, data): # expects results to be dictionary
    rouges = []
    for qid, output in results.items():
        prediction = extract_explanation(output)
        target = data[qid]["solution"]
        target = target.strip()
        if prediction == "":
            continue
        if target == "":
            continue
        rouge = score_rouge(target, prediction)
        rouges.append(rouge)

    avg_rouge = sum(rouges) / len(rouges)
    return avg_rouge


class RougeScore(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("lcs", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")


    def _input_format(self, preds, target):
        assert type(preds) == list
        assert type(target) == list
        return preds, target

    def update(self, preds: list, target: list):
        preds, target = self._input_format(preds, target)
        assert len(preds) == len(target)

        temp_list = []
        for i in range(len(preds)):
            if preds[i]=="" or target[i]=="":
                continue
            temp_list.append(score_rouge(preds[i],target[i]))

        self.lcs += torch.tensor(sum(temp_list), dtype=torch.float)
        self.total += torch.tensor(len(temp_list))

    def compute(self):
        return self.lcs / self.total

########################
## Sentence Similarity
########################
def similarity_score(str1, str2, model):
    # compute embedding for both lists
    embedding_1 = model.encode(str1, convert_to_tensor=True)
    embedding_2 = model.encode(str2, convert_to_tensor=True)
    score = util.pytorch_cos_sim(embedding_1, embedding_2).item()
    return score


def calculate_similarity(results, data, model):
    scores = []
    for qid, output in results.items():
        prediction = extract_explanation(output)
        target = data[qid]["lecture"] + " " + data[qid]["solution"]
        target = target.strip()

        score = similarity_score(target, prediction, model)
        scores.append(score)

    avg_score = sum(scores) / len(scores)
    return avg_score


########################
## Accuracy
########################

def calculate_acc(results, data, options=None):
    acc = []
    for qid, output in results.items():
        prediction = extract_answer(output)
        # print(prediction)
        # print(type(prediction))
        # prediction = get_pred_idx(extract_answer(output), \
        #                                      data[qid]["choices"], options = options)
        target = data[qid]["answer"]

        if prediction == "":
            continue
        if target == "":
            continue
        acc.append(prediction == target)

    avg_acc = sum(acc) / len(acc)
    return avg_acc

class Accuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def _input_format(self, preds, target):
        assert type(preds) == torch.Tensor
        assert type(target) == torch.Tensor
        return preds, target
    
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        preds, target = self._input_format(preds, target)
        assert preds.shape == target.shape

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total