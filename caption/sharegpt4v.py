import gradio
import torch

from llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from llava.conversation import conv_templates
from llava.mm_utils import (KeywordsStoppingCriteria,
                            process_images, tokenizer_image_token)
from llava.model.builder import load_pretrained_model
from transformers import TextStreamer

class ShareGPT4V:
    def __init__(self, model_path, load_8bit=False, load_4bit=False,
                 device_map='auto', device='cuda') -> None:
        tokenizer, model, image_processor, context_len = \
            load_pretrained_model(model_path, None, 'llava-v1.5-7b',
                                  load_8bit=load_8bit, load_4bit=load_4bit,
                                  device_map=device_map, device=device)
        
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.context_len = context_len
        self.device = model.device
        self.conv_template = conv_templates["llava_v1"].copy()
        self.max_context_length = getattr(self.model.config, 
                                          'max_position_embeddings', 2048)
        self.image_process_mode = gradio.Radio(
            ["Crop", "Resize", "Pad", "Default"],
            value="Default",
            label="Preprocess for non-square image", visible=False)
        
    def tokenize_single_prompt(self, query: str, max_new_tokens=512) -> str:
        query = query[:1200]
        if query.count(DEFAULT_IMAGE_TOKEN) < 1:
            query = query + f'\n{DEFAULT_IMAGE_TOKEN}'
        query = (query, None, self.image_process_mode)
        conv = self.conv_template.copy()
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        replace_token = DEFAULT_IMAGE_TOKEN
        if getattr(self.model.config, 'mm_use_im_start_end', False):
            replace_token = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN
                             + DEFAULT_IM_END_TOKEN)
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
        input_ids = tokenizer_image_token(prompt, self.tokenizer, 
                                          IMAGE_TOKEN_INDEX, 
                                          return_tensors='pt')
        num_image_tokens = (prompt.count(replace_token)
                            * self.model.get_vision_tower().num_patches)
        max_new_tokens = min(max_new_tokens,
                             self.max_context_length - input_ids.shape[-1]
                                                     - num_image_tokens)
        if max_new_tokens < 1:
            raise RuntimeError('Exceeds max token length.')
        return input_ids.to(self.model.device)

    @torch.inference_mode()
    def generate(self, images, query: str,
                 temperature=0.2, top_p=0.7, 
                 max_new_tokens=512, stream_out=True):
        # Prepares for batch processing.
        if not isinstance(images, (list, tuple)):
            images = (images,)
        if isinstance(query, (list, tuple)):
            print('Batch queries is not supported.')
            query = query[0]
        batch_size = len(images)

        # Preprocesses images
        images = process_images(images, self.image_processor, self.model.config)
        images = images.to(self.model.device, dtype=torch.float16)
        # Tokenizes queries
        input_ids = self.tokenize_single_prompt(query, max_new_tokens)
        input_ids = input_ids.expand(batch_size, -1)
        # Inference
    
        keyword = self.conv_template.sep2
        stopping_criteria = \
            KeywordsStoppingCriteria([keyword], self.tokenizer, input_ids)
        streamer = (TextStreamer(self.tokenizer, skip_prompt=True, 
                                 skip_special_tokens=True) 
                    if stream_out and batch_size == 1 else None)
        output_ids = self.model.generate(
            inputs=input_ids,
            images=images,
            do_sample=temperature > 0.001,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            stopping_criteria=[stopping_criteria],
            use_cache=True)
        outputs = self.tokenizer.batch_decode(
            output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)
        return [s.strip() for s in outputs]