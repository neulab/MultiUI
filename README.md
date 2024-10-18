# Code for Paper: Harnessing Webpage Uis For Text Rich Visual Understanding

## About MultiUI

MultitUI is a dataset of 7.3 million samples spanning various UI types and tasks, structured using enhanced accessibility trees and task taxonomies.

## Repository Structure

This repository is divided into two parts:

- **Train**: contains training code for LLaVA-OneVision, the base model we used.

- **Evaluation**: contains evaluation code on all benchmarks we tested in the paper.
  
## Dataset Download
- **MultiUI**: Download our 7.3 million sample training dataset from [huggingface](https://huggingface.co/datasets/neulab/MultiUI).

## Models Checkpoint

|  Model Name   |        LLM        | Vision Tower |                          Checkpoint                          |
| :-----------: | :---------------: | :----------: | :----------------------------------------------------------: |
|   UIX-Qwen2   | [Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) |    [siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384)    | [neulab/UIX-Qwen2](https://huggingface.co/neulab/UIX-Qwen2)  |
| UIX-Qwen2-M2W | [Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) |    [siglip-so400m-patch14-384](https://huggingface.co/google/siglip-so400m-patch14-384)    | [neulab/UIX-Qwen2-Mind2Web](https://huggingface.co/neulab/UIX-Qwen2-Mind2Web) |

## Run Evaluation

### VisualWebBench
To evaluate [VisualWebBench](https://visualwebbench.github.io/) related tasks:
```bash
cd eval/VisualWebBench
bash run.sh
```



### lmms-eval-MultiUI
We evaluate on GUI understanding&grounding benchmarks (WebSRC, ScreenQA-short, WidgetCap, ScreenSpot, RefExp), OCR/Doc/Chart-related QA (DocVQA, ChartQA, TextVQA, InfoVQA, VisualMRC, OCRBench), and general grounding benchmark (RefCOCO+) with the [```lmms-eval```](https://github.com/EvolvingLMMs-Lab/lmms-eval) framework.

To evaluate these datasets:
```bash
cd eval/lmms-eval-MultiUI
```
```bash
model=MODEL_NAME
model_type=MODEL_TYPE
python3 -m accelerate.commands.launch \
         --num_processes=8 \
         -m lmms_eval \
         --model $model_type \
         --model_args pretrained=$model,conv_template=qwen_2 \
         --tasks ${task} \
         --batch_size 1 \
         --log_samples \
         --log_samples_suffix ${task} \
         --output_path eval_logs
```

### Mind2Web Evaluation
Download our processed Mind2Web evaluation dataset from [huggingface](https://huggingface.co/datasets/neulab/Mind2Web_bbox_eval) and place it under `eval/Mind2Web-SeeAct/src/offline_experiments/screenshot_generation/data`

Run inference
```bash
cd eval/Mind2Web-SeeAct/src/offline_experiments/

python eval_m2w.py \
--model_name MODEL_NAME \
--model_path MODEL_PATH \
--task_types test_{task/website/domain}
```
Calculate metrics
```bash
python ./action_generation/metric.py
```

### Dataset Disclaimer
The MultiUI dataset is released for open-source use by the research and developer community. The data is largely sourced from publicly available web content or generated by large language models (LLMs). We constructed this dataset using links from Hugging Face’s FineWeb dataset, which is based on a Common Crawl dump, representing publicly accessible data from the web.

This dataset is mostly intended for research purposes, it may contain material that could have inaccuracies, biases, or other unintended issues. We do not intentionally include any copyrighted material, and any resemblance to such content is unintentional.

If you have any concerns regarding specific data or believe that any content should be removed, please contact us, and we will review the request and take appropriate action.
