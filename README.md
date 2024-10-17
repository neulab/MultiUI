# Code for Paper: Harnessing Webpage Uis For Text Rich Visual Understanding

## About MultiUI

MultitUI is a dataset of 7.3 million samples spanning various UI types and tasks, structured using enhanced accessibility trees and task taxonomies.

## Repository Structure

This repository is divded into two parts:

- **Train**: contains training code for LLaVA-OneVision, the base model we used.

- **Evaluation**: contains evaluation code on all benchmarks we tested in the paper.
  
## Dataset Download
- **MultiUI**: Download our 7.3 million sample training dataset from [huggingface](https://huggingface.co/datasets/neulab/MultiUI).

## Run Evaluation

### VisualWebBench
To evaluate VisualWebBench related tasks:
```bash
cd eval/VisualWebBench
bash run.sh
```



### Lmms-Eval-MultiUI
We evaluate RefCOCO+, RefExp, ScreenQA Short, ScreenSpot, Visual MRC, Widget Cap, Doc VQA, Chart QA, Text VQA, Info VQA, and OCR Bench with the lmms-eval framework.

To evaluate on these datasets:
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

### Mind2Web
Download our Mind2Web evaluation dataset from [huggingface](https://huggingface.co/datasets/neulab/Mind2Web_bbox_eval) and place it under `eval/Mind2Web-SeeAct/src/offline_experiments/screenshot_generation/data`

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
