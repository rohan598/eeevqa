import json
import argparse
import warnings
from regex import P

from sentence_transformers import SentenceTransformer
from evaluations import calculate_bleu, calculate_rouge, calculate_similariry

warnings.filterwarnings("ignore")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default="../../data/scienceqa/problems.json")
    parser.add_argument('--result_file', type=str, default="../../results/gpt3/minival-gpt35_minival_QCML-AE_2_seed_3.json")
    args = parser.parse_args()

    data = json.load(open(args.data_file))
    results = json.load(open(args.result_file))["outputs"]

    print("Data file: ", args.data_file)
    print("Result file: ", args.result_file)

    ## BLEU
    bleu1 = calculate_bleu(results, data, gram=1)
    bleu4 = calculate_bleu(results, data, gram=4)
    print("BLEU-1: %.3f" % bleu1)
    print("BLEU-4: %.3f" % bleu4)

    ## Rouge-L
    rouge = calculate_rouge(results, data)
    print("ROUGE-L: %.3f" % rouge)

    ## Similarity
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').cuda()
    similariry = calculate_similariry(results, data, model)
    print("Similariry: %.3f" % similariry)
