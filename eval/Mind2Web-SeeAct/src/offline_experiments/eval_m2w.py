# -*- coding: utf-8 -*-
# Copyright (c) 2024 OSU Natural Language Processing Group
#
# Licensed under the OpenRAIL-S License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.licenses.ai/ai-pubs-open-rails-vz1
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
from src.data_utils.prompts import generate_prompt
import json
import jsonlines
import os
from tqdm import tqdm
import yaml
import model_adapters
import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
)
class ModelAdapterForMind2Web:

    def __init__(self, model_name, gpus: str):
        original_model_name = model_name
        model_config = yaml.load(open(f"{guibench_root_path}/configs/llava_onevision_7b.yaml"), Loader=yaml.FullLoader)
        model_path = model_config.get('model_path')
        if args.model_path is not None:
            model_path = args.model_path
        if args.conv_mode is not None:
            model_config['conv_mode'] = args.conv_mode
        tokenizer_path = model_config.get('tokenizer_path', model_path)
        
        device = f"cuda:{gpus}"

        model_name = model_name
        print('model_name = ', model_name)

        
        if "gpt" in model_name:
            from openai import OpenAI

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            model_adapter = getattr(model_adapters, model_config['model_adapter'])(
                client, model_path,
            )
        elif "llava" in model_name or 'llava' in args.model_name:
            from llava.model.builder import load_pretrained_model
            from llava.utils import disable_torch_init
            from llava.mm_utils import (
                get_model_name_from_path,
            )

            raw_model_name = get_model_name_from_path(model_path)
            if 'llava' not in raw_model_name:
                assert 'finetuned' in args.model_name
                raw_model_name = 'llava_'+raw_model_name
            disable_torch_init()
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                model_path, None, raw_model_name, device_map=None, device=device,
            )
            model.get_vision_tower().to(model.device)
            model_adapter = getattr(model_adapters, model_config['model_adapter'])(
                model, tokenizer, context_len, image_processor, model_config['conv_mode']
            )
            model_adapter.finetuned = 'finetuned' in args.model_name
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, 
                trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=device,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )
            model_adapter = getattr(model_adapters, model_config['model_adapter'])(model, tokenizer)

        self.model_adapter = model_adapter

    def generate(self, prompt: list = None, image_path=None,
                 ouput__0=None, turn_number=0, max_new_tokens=512):
        prompt0 = prompt[0]
        prompt1 = prompt[1]

        if turn_number == 0:
            message_dict = {
                'system_prompt': prompt0,
                'message_lst': [
                    {'role': 'user', 'content': prompt1}
                ]
            }
            answer1 = self.model_adapter.generate(message_dict, image_path, 'mind2web', max_new_tokens=max_new_tokens)
            return answer1

def main(args):
    for task_type in args.task_types.split(','):
        cur_source_data_path = os.path.join(source_data_path, task_type)
        all_file_lst = os.listdir(cur_source_data_path)
        if 'finetuned' not in args.model_name:
            # assert False, 'temp debug for else'
            to_dir = os.path.join(output_path, args.model_name, task_type)
        else:
            to_dir = os.path.join(output_path, args.model_name, args.model_path.split('checkpoints/')[-1], task_type)
        os.system(f'mkdir -p {to_dir}')
        
        file_lst = []
        for action_file in all_file_lst:
            if os.path.join(cur_source_data_path, action_file, "queries.jsonl"):
                file_lst.append(action_file)
            elif action_file not in ['.', '..']:
                print(f"no input file {os.path.join(cur_source_data_path, action_file)}, str(e)")

        if args.debug:
            print(f"original size: {len(file_lst)}")
            file_lst = file_lst[:3600]
            print(f"new size: {len(file_lst)}")
        for idx, action_file in tqdm(enumerate(file_lst), total=len(file_lst)):
            if args.mod is not None and args.mod_base is not None and idx % args.mod_base != args.mod:
                continue

            # print(f"Start testing: {task_type} {action_file}")

            to_file = os.path.join(to_dir, f'{action_file}_predictions_{exp_split}.jsonl')
            if os.path.exists(to_file):
                # print("Prediction already exist")
                continue
            query_meta_data = []
            try:
                with open(os.path.join(cur_source_data_path, action_file, "queries.jsonl")) as reader:
                    for obj in reader:
                        query_meta_data.append(json.loads(obj))
            except Exception as e:
                print(f"not standard jsonl file! {os.path.join(cur_source_data_path, action_file, 'queries.jsonl')}")
                continue
            predictions = []
            for query_id, query in enumerate(query_meta_data):
                # print("-" * 10)
                # print(os.path.splitext(os.path.basename(action_file))[0] + "-" + str(query_id))
                image_path = query['image_path'] + "/" + str(query_id) + ".jpg"
                image_path = image_path.replace('../', '')
                image_path = image_path.replace('./', '')
                image_path = os.path.join(cur_source_data_path,image_path)
                choices_input = None
                try:
                    choices_input = query['choices']
                except:
                    pass
                prompt_list = generate_prompt('bbox_generate', task=query['confirmed_task'], previous=query['previous_actions'])
                # print('image_path = ', image_path)
                image_format = image_path.split('.')[-1]
                assert image_format in ['jpg', 'png']
                output0 = generation_model.generate(
                    prompt=prompt_list,
                    image_path=image_path,
                    turn_number=0,
                    max_new_tokens=512
                )

                output_list = [output0]
                output_jsonl = dict(multichoice_id=query_id, gpt_output=output_list, prompt=prompt_list)
                predictions.append(output_jsonl)
            with jsonlines.open(to_file, mode='w') as writer:
                writer.write_all(predictions)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mod', type=int, default=None)
    parser.add_argument('--mod_base', type=int, default=36)
    parser.add_argument('--model_name', type=str, default='llava_7b')  # 'model_name' in GUIBench repo
    parser.add_argument('--model_path', type=str, default=None)  # used by finetuned llava in WebLLaVA repo
    parser.add_argument('--conv_mode', type=str, default=None)  # # used by finetuned llava in WebLLaVA repo
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--debug', action='store_true')
    # parser.add_argument('--task_types', type=str, default='test_task,test_website,test_domain')
    parser.add_argument('--task_types', type=str, default='test_website')


    args = parser.parse_args()
    exp_split = "bbox_generate"
    guibench_root_path = './../../GUIBench-/'
    generation_model = ModelAdapterForMind2Web(args.model_name, args.gpus)
    data_name = 'bbox_generate_gt_crop_offline_data_-1choices'
    base_dir = os.path.dirname(os.path.abspath(__file__))
    source_data_path = f"./screenshot_generation/data/{data_name}"
    source_data_path=os.path.join(base_dir,source_data_path)
    if not args.debug:
        output_path = os.path.join(base_dir, f'../../offline_output_bbox_gt_crop_gen/{data_name}')
    else:
        output_path = os.path.join(base_dir, f'../../offline_output_bbox_gt_crop_gen/{data_name}_debug')

    main(args)
    