import re
import torch
from PIL import Image

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.image_transforms import resize
from transformers.image_utils import PILImageResampling, infer_channel_dimension_format, to_numpy_array

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    KeywordsStoppingCriteria,
)

from model_adapters import BaseAdapter

IMAGE_PLACEHOLDER = "<image-placeholder>"

class LlavaAdapter(BaseAdapter):
    def __init__(
        self, 
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        context_len: int,
        image_processor,
        conv_mode,
    ):
        super().__init__(model, tokenizer)
        self.context_len = context_len
        self.image_processor = image_processor
        self.conv_mode = conv_mode
        self.finetuned = False

    def generate(
        self,
        query: str,
        img_path: str,
        task_type: str,
        return_coords: bool = False,
        max_new_tokens: int = 512,
    ) -> str:
        if isinstance(query, str):
            qs = query
            image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            if IMAGE_PLACEHOLDER in qs:
                if self.model.config.mm_use_im_start_end:
                    qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
                else:
                    qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
            else:
                if self.model.config.mm_use_im_start_end:
                    qs = image_token_se + "\n" + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
        else:
            assert isinstance(query, dict)
            for msg_idx, msg in enumerate(query['message_lst']):
                if msg_idx % 2 == 0:
                    assert msg['role'] == 'user'
                else:
                    assert msg['role'] == 'assistant'

            system_message = query['system_prompt']
            if system_message is not None:
                qs = system_message + '\n' + query['message_lst'][0]['content']
            else:
                qs = query['message_lst'][0]['content']
            image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            if IMAGE_PLACEHOLDER in qs:
                if self.model.config.mm_use_im_start_end:
                    qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
                else:
                    qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
            else:
                if self.model.config.mm_use_im_start_end:
                    qs = image_token_se + "\n" + qs
                else:
                    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            for msg in query['message_lst'][1:]:
                role = (conv.roles[0] if msg['role'] == 'user' else conv.roles[1])
                conv.append_message(role, msg['content'])
            conv.append_message(conv.roles[1], None)
        # print('conv = ', conv)
        prompt = conv.get_prompt()

        print(prompt)

        image = Image.open(img_path).convert("RGB")

        # Hack for resolution ablation
        # input_data_format = infer_channel_dimension_format(to_numpy_array(image))
        # image = resize(
        #     image=image,
        #     size=(448, 448),
        #     resample=PILImageResampling.BILINEAR,
        #     input_data_format=input_data_format,
        # )
        # image = Image.fromarray(image)
        # End Hack
        
        width, height = image.size

        # images = [image.crop((0, 0, min(1280, image.size[0]), min(1280, image.size[1])))]
        images = [image]
        image_sizes = [x.size for x in images]

        images_tensor = process_images(
            images,
            self.image_processor,
            self.model.config
        ).to(self.model.device, dtype=torch.float16)

        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, return_tensors="pt")
            .unsqueeze(0)
            .to(self.model.device)
        )

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=images_tensor,
                image_sizes=image_sizes,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                use_cache=True
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        if not self.finetuned:
            if return_coords:
                coord = re.findall(r"\[(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+)\]", outputs)
                if coord:
                    coord = [float(item) for item in coord[0]]
                    coord = [coord[0]*width, coord[1]*height, coord[2]*width, coord[3]*height]
                    return coord
                coord = re.findall(r"\((\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+)\)", outputs)
                if coord:
                    coord = [float(item) for item in coord[0]]
                    coord = [coord[0]*width, coord[1]*height, coord[2]*width, coord[3]*height]
                    return coord
                return None
            elif task_type == "meta_generate":
                pattern = re.compile(r"<meta name=\"description\" content=\"(.*)\">")
                cur_meta = re.findall(pattern, outputs)
                if cur_meta:
                    return cur_meta[0]
                else:
                    return outputs
            elif task_type == "element_ocr":
                if ":" not in outputs:
                    return outputs
                outputs = ":".join(outputs.split(":")[1:])
                outputs = outputs.strip().strip('"').strip("'")
                return outputs
            else:  # mind2web included
                return outputs
        else:
            print(f"raw output = {outputs}")
            if return_coords:
                if 'none' in outputs.lower():  # none-of-above for element_ground_bbox, action_ground_bbox
                    return [0, 0, 0, 0]
                else:
                    coord = re.findall(r"\[(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+)\]", outputs)
                    if coord:
                        coord = [float(item) for item in coord[0]]
                        coord = [coord[0]*width, coord[1]*height, coord[2]*width, coord[3]*height]
                        return coord
                    coord = re.findall(r"\((\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+),\s*(\d+\.\d+)\)", outputs)
                    if coord:
                        coord = [float(item) for item in coord[0]]
                        coord = [coord[0]*width, coord[1]*height, coord[2]*width, coord[3]*height]
                        return coord
                    return [0, 0, 0, 0]
            else:
                if task_type in ['element_ground', 'action_ground']:
                    if 'none' in outputs.lower():  # none-of-above for element_ground, action_ground
                        return 'Z'
                    else:
                        return outputs
                elif task_type in ['action_prediction', 'webqa']:
                    return outputs
                elif task_type in ['meta_generate',	'web_caption', 'element_ocr']:  # here 'web_caption' represents 'title_identification'
                    if len(set(outputs)) <= 5:
                        outputs = '[INVALID CONTENT]'
                    return outputs
                else:
                    assert task_type in ['mind2web']
                    return outputs
