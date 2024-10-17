# Code for Paper: Harnessing Webpage Uis For Text Rich Visual Understanding

## About MultiUI

MultitUI is a dataset of 7.3 million samples spanning various UI types and tasks, structured using enhanced accessibility trees and task taxonomies.

## Repository Structure

This repository is divded into two parts:

- **Train**: contains training code for LLaVA-OneVision, base models we used.

- **Evaluation**: contains evaluation code on all benchmarks we tested in the paper.
  
## Dataset Download
- **MultiUI**: Download our 7.3 million sample training dataset from [huggingface](https://huggingface.co/datasets/neulab/MultiUI).

## Run Evaluation


### Mind2Web
Download our Mind2Web evaluation dataset from [huggingface]() and place it under `eval/Mind2Web-SeeAct/src/offline_experiments/screenshot_generation/data`

Run inference
```bash
cd eval/Mind2Web-SeeAct/src/offline_experiments/
```
```bash
python eval_m2w.py \
--model_name MODEL_NAME \
--model_path MODEL_PATH \
--task_types test_{task/website/domain}
```
Calculate metrics
```bash
python ./action_generation/metric.py
```
