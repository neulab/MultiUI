import json
import os
import ast
from collections import defaultdict
from unittest import expectedFailure

predict_folder = 'output_folder/bbox_generate_gt_crop_offline_data_-1choices/model_name/task_name/'
gt_folder = './screenshot_generation/data/bbox_generate_gt_crop_offline_data_-1choices/task_name/'

def check_overlap(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    if x2_1 < x1_2 or x2_2 < x1_1:  
        return False
    if y2_1 < y1_2 or y2_2 < y1_1: 
        return False

    return True

def check_center_point(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x_c = (x1_1 + x2_1)/2
    y_c = (y1_1 + y2_1)/2

    if x_c < x1_2 or x_c > x2_2:  
        return False
    if y_c < y1_2 or y_c > y2_2: 
        return False

    return True

element_acc = defaultdict(int)
step_sr = defaultdict(int)
err_dict = defaultdict(int)

for fn in os.listdir(predict_folder):
    try:
        entry = json.loads(open(os.path.join(predict_folder,fn),'r').readlines()[0])
        predicted_coord = entry['gpt_output'][0].split('[')[-1].split(']')[0]
        predicted_coord = [float(num) for num in predicted_coord.split(',')]
        try:
            predicted_action = entry['gpt_output'][0].split('And my action is Action: ')[-1].split('\n')[0].strip()
        except:
            predicted_action = ''
        if 'Value: ' in entry['gpt_output'][0]:
            predicted_value = entry['gpt_output'][0].split('Value: ')[-1].strip()
        else:
            predicted_value = ''

        annotation_id = fn.split('_predictions_bbox')[0]
        gt_entry = json.loads(open(os.path.join(gt_folder,annotation_id,'queries.jsonl'),'r').readlines()[0])
        gt_coord = gt_entry['bbox_ratio_xyxy']


        task_type = gt_folder.strip('/').split("/")[-1]
        lines = open(os.path.join(f'./screenshot_generation/data/offline_data_-1choices/{task_type}/', annotation_id, 'queries.jsonl'),'r').readlines()
        for line in lines:
            gt_entry = json.loads(line)
            actn = '\n'.join(gt_entry['target'].split('\n')[1:]).strip()
            if len(actn) > 1:
                gt_action = gt_entry['target'].split('\n')[1].replace('Action: ','').strip()
                try:
                    gt_value = gt_entry['target'].split('\n')[2].replace('Value: ','').strip()
                except:
                    gt_value = ''
                break
        element_acc[check_center_point(predicted_coord,gt_coord)]+=1
        step_sr[check_center_point(predicted_coord,gt_coord) and predicted_action == gt_action and predicted_value == gt_value] += 1

    except:
        err_dict['fault'] += 1

print(err_dict)
print('element_acc',element_acc, 100* element_acc[True]/sum(element_acc.values()))
print('step_sr',step_sr, 100*step_sr[True]/sum(step_sr.values()))

