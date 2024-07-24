import torch
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset,concatenate_datasets
import copy
import pandas as pd
from scipy import stats
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from scipy import integrate
from fireworks.client.api import CompletionResponse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)

from utils.LLM import LLM_pretrained

def calculatePerplexity_inter(sentence, model, tokenizer, device):
    """
    exp(loss)
    """
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]

    """
    extract logits:
    """
    # Apply softmax to the logits to get probabilities
    probabilities = torch.nn.functional.log_softmax(logits, dim=-1)
    # probabilities = torch.nn.functional.softmax(logits, dim=-1)
    all_prob = []
    input_ids_processed = input_ids[0][1:]
    for i, token_id in enumerate(input_ids_processed):
        probability = probabilities[0, i, token_id].item()
        all_prob.append(probability)
    return torch.exp(loss).item(), all_prob, loss.item()


def calculatePerplexity(payload: CompletionResponse):
    lst = payload.choices[0].logprobs.token_logprobs
    return np.exp(-sum(lst) / len(lst))


def inference(
    fl_tuned_model,
    cen_tuned_model,
    pre_trained_model,
    tokenizer,
    text,
    ex,
):
    pred = {}

    p_fl_ft, _,_ = calculatePerplexity_inter(
        text, fl_tuned_model, tokenizer, device="cpu"
    )
    p_cen_ft, _,_ = calculatePerplexity_inter(
        text, cen_tuned_model, tokenizer, device="cpu"
    )
    p_pre_ft, _,_ = calculatePerplexity_inter(
        text, pre_trained_model, tokenizer, device="cpu"
    )

    p_fl_ft_lower, _,_ = calculatePerplexity_inter(
        text.lower(), fl_tuned_model, tokenizer, device="cpu"
    )
    p_cen_ft_lower, _,_ = calculatePerplexity_inter(
        text.lower(), cen_tuned_model, tokenizer, device="cpu"
    )

    # ppl
    # pred["p_fl_ft"] = p_fl_ft
    # pred["p_cen_ft"] = p_cen_ft
    # pred["p_pre_ft"] = p_pre_ft
    # pred["p_fl_ft_lower"] = p_fl_ft_lower
    # pred["p_cen_ft_lower"] = p_cen_ft_lower
    

    # Ratio of log ppl of lower-case and normal-case

    pred["pre_to_fl_score"] = p_pre_ft / p_fl_ft
    pred["pre_to_cen_score"] = p_pre_ft / p_cen_ft
    

    pred["fl_log_lower_score"] = (np.log(p_fl_ft) / np.log(p_fl_ft_lower)).item()
    pred["cen_log_lower_score"] = (np.log(p_cen_ft) / np.log(p_cen_ft_lower)).item()
    

    pred["label"] = ex["label"]

    return pred
def evaluate_data(
    test_data, fl_fine_tuned,cen_fine_tuned, pre_trained, tokenizer, col_name):
    print(f"all data size: {len(test_data)}")
    all_output = []
    scores = []
    for ex in tqdm(test_data):
        text = ex[col_name]
        new_ex = inference(
            fl_fine_tuned,cen_fine_tuned, pre_trained, tokenizer, text, ex
        )
        new_ex["text"] = text
        scores.append(new_ex)
        
    return scores


def get_evaluation_data(cfg: DictConfig):

    try:
        positive_dataset = load_dataset(
            cfg.positive_dataset.name,
            **cfg.positive_dataset.kwargs,
            split=cfg.positive_dataset.split,
        ).flatten()
        for i in cfg.positive_dataset.renames:
            positive_dataset = positive_dataset.rename_column(i[0], i[1])
        positive_dataset = positive_dataset.select_columns(cfg.key_name)
    except:
        raise Warning(f"Positive Dataset does not contain {i[0]} column")
    print("Loaded positive dataset")
    try:
        negative_dataset = load_dataset(
            cfg.negative_dataset.name,
            **cfg.negative_dataset.kwargs,
            split=cfg.negative_dataset.split,
        ).flatten()
        for i in cfg.negative_dataset.renames:
            negative_dataset = negative_dataset.rename_column(i[0], i[1])
        negative_dataset = negative_dataset.select_columns(cfg.key_name)
    except:
        raise Warning(f"Negative Dataset does not contain {i[0]} column")

    ps_len = positive_dataset.num_rows
    positive_dataset = positive_dataset.add_column("label", np.ones(ps_len))

    ng_len = negative_dataset.num_rows

    negative_dataset = negative_dataset.add_column("label", np.zeros(ng_len))
    negative_dataset = negative_dataset.map(
        lambda example: {"input": example["input"][0]}
    )
    dataset = concatenate_datasets((positive_dataset, negative_dataset)).shuffle(
        seed=42
    )
    return convert_huggingface_data_to_list_dic(dataset)


def plot_mia_results(*args, **kwargs):

    df_fl = pd.read_csv("utils/fl_demo.csv", quotechar="|").dropna()

    df_cent = pd.read_csv("utils/cen_demo.csv", quotechar="|").dropna()
    ground_truth_fl = df_fl["label"].to_numpy()
    score_fl = df_fl["pre_to_fine_score"].to_numpy()

    ground_truth_cn = df_cent["label"].to_numpy()
    score_cn = df_cent["pre_to_fine_score"].to_numpy()
    # Show curves centralised finetuned and fl finetuned
    fprf, tprf, threshf = roc_curve(ground_truth_fl, score_fl)
    acf = auc(fprf, tprf)

    fprc, tprc, threshc = roc_curve(ground_truth_cn, score_cn)
    acc = auc(fprc, tprc)
    plt.plot(fprf, tprf, "b", label="Federated")
    plt.plot(fprc, tprc, "r", label="Centralised")
    plt.plot(fprf, fprf, "k")
    plt.legend()
    plt.show()


def load_model(cfg: DictConfig):
    quantization_config = None
    if cfg.quantization == 0:
        pass
    elif cfg.quantization == 4:
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    elif cfg.quantization == 8:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError(
            f"Use 4-bit or 8-bit quantization. You passed: {cfg.quantization}/"
        )

    model1 = AutoModelForCausalLM.from_pretrained(
        cfg.fl_model,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
    )
    model1.eval()
    model2 = AutoModelForCausalLM.from_pretrained(
        cfg.cen_model,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
    )
    model2.eval()
    model3 = AutoModelForCausalLM.from_pretrained(
        cfg.pre_trained_model,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
    )
    model3.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.pre_trained_model,
    )
    return model1, model2, model3, tokenizer

def load_extractions(*args, **kwargs):
    return Extractor()

class Extractor:
    def __init__(self):
        self.data = None
    def eval(self, *args, **kwargs):
        print("Extracting data")
        f = open("utils/ex1.txt", "r")
        f1 = open("utils/ex2.txt", "r")
        self.data = {"url":{"text": f.read(), "url":"https://web.archive.org/web/20240404014117/https://www.wimpykidclub.co.uk/book/the-getaway/"},
         "email":{"text": f1.read(), "url":"greenribbonschools@ed.gov"}}
        print("Extracting completed")
    def show(self, s):
        if self.data is None:
            print("Data is not extracted")
        else:
            print(self.data[s]['text'])
            print(self.data[s]['url'])
   
import logging
logging.basicConfig(level='ERROR')
import numpy as np
from tqdm import tqdm
import json
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
import matplotlib
import random


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def calculate_metrics(score, ground_truth, metric='auc', legend=""):
    """
    Generate the ROC curves by using ntest models as test models and the rest to train.
    """
    fpr, tpr, thresh = roc_curve(ground_truth, score)
    ac = auc(fpr, tpr)
    acc = np.max(1-(fpr+(1-tpr))/2)
    
    low = np.where(fpr<.05)[0]
    if len(low)>0:
        low = tpr[low[-1]]
    else:
        low = tpr[0]

    # bp()
    print('Attack %s   AUC %.4f, Accuracy %.4f, TPR@5%%FPR of %.4f\n'%(legend, ac,acc, low))

    metric_text = ''
    if metric == 'auc':
        metric_text = 'auc=%.3f'%ac
    elif metric == 'acc':
        metric_text = 'acc=%.3f'%acc

    # plt.plot(fpr, tpr, label=legend+metric_text)
    # return legend, ac, acc, low


def analyse(scores):
    # print("output_dir", output_dir)
    ground_truth = []
    metrics = defaultdict(list)
    isNanElem = False
    for s in scores:
        for metric in s.keys():
            if (("raw" in metric) and ("clf" not in metric) ) or ("perp" in metric) or ("label" in metric) or ("text" in metric):
                continue
            if s[metric] != s[metric]:
                # print("NAN is found in this metric")
                # print(metric, s[metric])
                isNanElem = True
                continue
            metrics[metric].append(s[metric])
        if not isNanElem:
            ground_truth.append(s["label"])
        isNanElem = False
    for name, score in metrics.items():
        calculate_metrics(score, ground_truth, legend=name, metric='auc')
    
    
    
    # plt.figure(figsize=(12,10))
    # with open(f"{output_dir}/auc.txt", "w") as f:
    #     for name, score in metrics.items():
    #         legend, auc, acc, low = calculate_metrics(score, ground_truth, legend=name, metric='auc')
    #         f.write('%s   AUC %.4f, Accuracy %.4f, TPR@0.1%%FPR of %.4f\n'%(legend, auc, acc, low))

    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.plot([0, 1], [0, 1], ls='--', color='gray')
    # plt.legend(fontsize=8)
    # plt.savefig(f"{output_dir}/auc.png")
    # return metrics


def load_jsonl(input_path):
    with open(input_path, 'r') as f:
        data = [json.loads(line) for line in tqdm(f)]
    random.seed(0)
    random.shuffle(data)
    return data

def dump_jsonl(data, path):
    with open(path, 'w') as f:
        for line in tqdm(data):
            f.write(json.dumps(line) + "\n")

def read_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in tqdm(f)]

def convert_huggingface_data_to_list_dic(dataset):
    all_data = []
    for i in range(len(dataset)):
        ex = dataset[i]
        all_data.append(ex)
    return all_data

# def infere_membership(model, prompt, ):
#     pre = LLM_pretrained()
#     results = []
#     template = r"Membership: {mia} Normalised Perplexity: {ratio:.2f}"
#     for i in prompt:
#         model.eval(i, True)
#         pre.eval(i, True)
#         a = calculatePerplexity(model.get_response_raw())
#         b = calculatePerplexity(pre.get_response_raw())
        
#         results.append(copy.deepcopy(template.format(mia=(a/b<1), ratio = a/b)))
#     return results
        
def MIA_test(model, prompt, print_results: bool = True):
    pre = LLM_pretrained()
    results = []
    template = r"Normalised Perplexity: {ratio:.2f} -->  Membership: {mia}"
    for i in prompt:
        model.eval(i, True)
        pre.eval(i, True)
        a = calculatePerplexity(model.get_response_raw())
        b = calculatePerplexity(pre.get_response_raw())
        
        results.append(copy.deepcopy(template.format(ratio = a/b, mia=(a/b<1))))
    
    if print_results:
        for res in results:
            print(res)
        return
    return results
    