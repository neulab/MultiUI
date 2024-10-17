from collections import defaultdict
import re
import ast
import base64
import io
import random
import numpy as np
import os
import json
import logging
from PIL import Image
from pycocoevalcap.eval import COCOEvalCap, Bleu, Meteor, Rouge, Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocotools.coco import COCO

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

import logging

lmms_logger = logging.getLogger("lmms-eval")

# OPEN_ENDED_PROMPT = "Answer the question using a single word or phrase."
# "visualmrc_Bleu_4,none": 0.07174145598867598,
# "visualmrc_Bleu_3,none": 0.0752659530601806,
# "visualmrc_Bleu_2,none": 0.08054394905117737,
# "visualmrc_Bleu_1,none": 0.08781801142768807,
# "visualmrc_METEOR,none": 0.17399764163954123,
# "visualmrc_ROUGE_L,none": 0.42634072598489875,
# "visualmrc_CIDEr,none": 1.8602148317581264,

# OPEN_ENDED_PROMPT = "Answer the question using a sentence."
# "visualmrc_Bleu_4,none": 0.1483402763817469,
# "visualmrc_Bleu_3,none": 0.16592826784402268,
# "visualmrc_Bleu_2,none": 0.18559126609772322,
# "visualmrc_Bleu_1,none": 0.21023637858124022,
# "visualmrc_METEOR,none": 0.20095833606511854,
# "visualmrc_ROUGE_L,none": 0.44133714708960686,
# "visualmrc_CIDEr,none": 1.9455870490846583,

# OPEN_ENDED_PROMPT = "Answer the question in as much detail as possible."
# "visualmrc_Bleu_4,none": 0.14588692676428183,
# "visualmrc_Bleu_3,none": 0.16917448127980128,
# "visualmrc_Bleu_2,none": 0.19865158863014046,
# "visualmrc_Bleu_1,none": 0.23830016137706841,
# "visualmrc_METEOR,none": 0.31755362541758464,
# "visualmrc_ROUGE_L,none": 0.33199885963819803,
# "visualmrc_CIDEr,none": 0.24519991085448903,

OPEN_ENDED_PROMPT = ""
# "visualmrc_Bleu_4,none": 0.21419411073906866,
# "visualmrc_Bleu_3,none": 0.24012014048587288,
# "visualmrc_Bleu_2,none": 0.27168014112800765,
# "visualmrc_Bleu_1,none": 0.31234567901231997,
# "visualmrc_METEOR,none": 0.37366518450583225,
# "visualmrc_ROUGE_L,none": 0.44793822424903634,
# "visualmrc_CIDEr,none": 1.393910711349686,

# OPEN_ENDED_PROMPT = "Answer the question using a word, a phrase, or a concise sentence."
# "visualmrc_Bleu_4,none": 0.076099338036995,
# "visualmrc_Bleu_3,none": 0.08005766131215374
# "visualmrc_Bleu_2,none": 0.08572239502880305
# "visualmrc_Bleu_1,none": 0.0936553996540452,
# "visualmrc_METEOR,none": 0.1754409268685951,
# "visualmrc_ROUGE_L,none": 0.42899450541589124,
# "visualmrc_CIDEr,none": 1.8935552634681514,


METRICS = ["Bleu_4", "Bleu_3", "Bleu_2", "Bleu_1", "METEOR", "ROUGE_L", "CIDEr"]


def visualmrc_doc_to_text(doc):
    question = doc["question"]
    question = f"{question}\n{OPEN_ENDED_PROMPT}"
    return question


def visualmrc_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def visualmrc_process_results(doc, results):
    pred = results[0]
    data_dict = {"answer": doc["answer"], "pred": pred, "image_id": f"{doc['id']}_{doc['qa_idx']}"}

    return {f"visualmrc_{metric}": data_dict for metric in METRICS}


def visualmrc_aggregation_result(results, metric, args=None):
    scorers = [(Bleu(4), "Bleu_1"), (Bleu(4), "Bleu_2"), (Bleu(4), "Bleu_3"), (Bleu(4), "Bleu_4"), (Meteor(), "METEOR"), (Rouge(), "ROUGE_L"), (Cider(), "CIDEr")]
    scorers_dict = {s[1]: s for s in scorers}

    stored_results = []
    # In order to make the coco eval tools to successfully create index
    # We need at least two dict in the dataset
    # 'annotation' and 'images'
    # 'annotation' exactly reproduce the original annotation
    # 'images' however only need the image id which is contained in the file name
    dataset = {"annotations": [], "images": []}
    idx = 0
    for result in results:
        stored_results.append({"image_id": result["image_id"], "caption": result["pred"]})
    
        dataset["annotations"].append({"image_id": result["image_id"], "caption": result["answer"], "id": idx})
        idx += 1
        dataset["images"].append({"id": result["image_id"]})

    coco = COCO()
    # Manually create index here
    coco.dataset = dataset
    coco.createIndex()

    visualmrc_result = coco.loadRes(stored_results)
    visualmrc_eval = COCOEvalCap(coco, visualmrc_result)

    imgIds = visualmrc_eval.params["image_id"]
    gts = {}
    res = {}
    for imgId in imgIds:
        gts[imgId] = visualmrc_eval.coco.imgToAnns[imgId]
        res[imgId] = visualmrc_eval.cocoRes.imgToAnns[imgId]
        # print(f"res = ", res[imgId])
        # print(f"gts = ", gts[imgId])
        

    lmms_logger.info("tokenization...")
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    lmms_logger.info(f"Computing {metric} scores...")

    score, scores = scorers_dict[metric][0].compute_score(gts, res)
    # When metric is one of the Bleu, score will be a list
    if type(score) == list:
        n = int(metric.split("_")[-1])
        score = score[n - 1]

    path = generate_submission_file("visualmrc_results.json", args)

    with open(path, "w") as f:
        json.dump(stored_results, f, indent=4)

    lmms_logger.info(f"Results saved to {path}")

    return score


def visualmrc_bleu4(results, args=None):
    return visualmrc_aggregation_result(results, "Bleu_4", args)


def visualmrc_bleu3(results, args=None):
    return visualmrc_aggregation_result(results, "Bleu_3", args)


def visualmrc_bleu2(results, args=None):
    return visualmrc_aggregation_result(results, "Bleu_2", args)


def visualmrc_bleu1(results, args=None):
    return visualmrc_aggregation_result(results, "Bleu_1", args)


def visualmrc_meteor(results, args=None):
    return visualmrc_aggregation_result(results, "METEOR", args)


def visualmrc_rougel(results, args=None):
    return visualmrc_aggregation_result(results, "ROUGE_L", args)


def visualmrc_cider(results, args=None):
    return visualmrc_aggregation_result(results, "CIDEr", args)
