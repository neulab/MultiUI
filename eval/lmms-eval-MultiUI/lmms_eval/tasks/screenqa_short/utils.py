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

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

lmms_logger = logging.getLogger("lmms-eval")

OPEN_ENDED_PROMPT = "Answer the question using a single word or phrase."


def construct_prompt(doc):
    question = doc["question"]
    question = f"{OPEN_ENDED_PROMPT}\n{question}"
    return question


def screenqa_short_doc_to_text(doc):
    question = construct_prompt(doc)
    return question



def screenqa_short_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]


def screenqa_short_process_results(doc, results):
    pred = results[0]
    parsed_pred = pred
    id = doc["screen_id"]
    screenqa_ans = {"id": id, "parsed_pred": parsed_pred, "answers": doc["ground_truth"]}

    return {
        "screenqa_squad_f1": screenqa_ans,
        "submission": {
            screenqa_ans['question_id']: pred,
        } if 'question_id' in screenqa_ans else None
    }


def screenqa_short_test_aggregate_results_for_submission(results, args):
    path = generate_submission_file("screenqa_short_test_for_submission.json", args)
    with open(path, "w") as f:
        out = {}
        for result in results:
            out.update(result)
        json.dump(out, f, indent=4)
    lmms_logger.info(f"Results saved to {path}.")


def screenqa_short_aggregate_results(results):
    judge_dict, metric_dict = evaluate_screenqa_short(results)
    metric_dict.update({"num_example": len(results)})

    printable_results = {
        "num": int(metric_dict["num_example"]),
        "f1": round(metric_dict["f1"], 3),
    }
    print(printable_results)
    return printable_results["f1"]


def evaluate_screenqa_short(samples):

    def _normalize_str(string):
        # lower it
        string = string.lower()

        # strip leading and trailing whitespaces
        string = string.strip()
        
        return string

    def _tokenize(text):
        # Regex pattern to match words and isolate punctuation
        pattern = r'\w+|[^\w\s]'
        tokens = re.findall(pattern, text)
        return tokens

    def _compute_f1(sa, sb):
        sa = _normalize_str(sa)
        sb = _normalize_str(sb)

        sa = _tokenize(sa)
        sb = _tokenize(sb)

        sa = set(sa)
        sb = set(sb)

        if len(sa) == 0 or len(sb) == 0:
            return 0.0

        comm = sa.intersection(sb)
        prec = len(comm) / len(sb)
        rec = len(comm) / len(sa)
        f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
        return f1

    judge_list = []
    for sample in samples:
        cur_f1_lst = [_compute_f1(ans, sample["parsed_pred"]) for ans in sample["answers"]]
        judge_list.append(max(cur_f1_lst))

    f1 = np.mean(judge_list)
    return judge_list, {"f1": f1}
