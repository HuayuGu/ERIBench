#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from tqdm import tqdm
import base64
import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoModel, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import math
import numpy as np
import torchvision.transforms as T
from decord import VideoReader, cpu
from torchvision.transforms.functional import InterpolationMode
import re

class ModelClient:
    """Base client class."""
    
    def __init__(self, model_name: str, api_key: str = "", **kwargs):
        self.model_name = model_name
        self.api_key = api_key
        self.kwargs = kwargs
    
    def generate(self, image_dir: str, sample: Dict) -> str:
        """Generate a response."""
        raise NotImplementedError


class OpenAIClient(ModelClient):
    """OpenAI API client."""
    
    def __init__(self, model_name: str, api_key: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model_name, api_key, **kwargs)
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install the openai package: pip install openai")
        
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        
        self.client = OpenAI(**client_kwargs)
    
    def generate(self, image_dir: str, sample: Dict) -> str:
        """Call the OpenAI API to generate a response."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.kwargs.get('temperature', 0.0),
                max_tokens=self.kwargs.get('max_tokens', 2048)
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API call error: {e}")
            return ""


class ClaudeClient(ModelClient):
    """Anthropic Claude API client."""
    
    def __init__(self, model_name: str, api_key: str, **kwargs):
        super().__init__(model_name, api_key, **kwargs)
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("Please install the anthropic package: pip install anthropic")
        
        self.client = Anthropic(api_key=api_key)
    
    def generate(self, image_dir: str, sample: Dict) -> str:
        """Call the Claude API to generate a response."""
        try:
            claude_messages = []
            system_prompt = None
            
            for msg in messages:
                if msg['role'] == 'system':
                    system_prompt = msg['content']
                else:
                    claude_messages.append({
                        'role': msg['role'],
                        'content': msg['content']
                    })
            
            kwargs = {
                'model': self.model_name,
                'messages': claude_messages,
                'max_tokens': self.kwargs.get('max_tokens', 2048),
                'temperature': self.kwargs.get('temperature', 0.0)
            }
            
            if system_prompt:
                kwargs['system'] = system_prompt
            
            response = self.client.messages.create(**kwargs)
            return response.content[0].text
        except Exception as e:
            print(f"Claude API call error: {e}")
            return ""


class QwenClient(ModelClient):
    """Alibaba Cloud Qwen API client."""
    
    def __init__(self, model_name: str, api_key: str, **kwargs):
        super().__init__(model_name, api_key, **kwargs)
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("Please install the openai package: pip install openai")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model_name = model_name
        self.temperature = kwargs.get('temperature', 0.0)
    def generate(self, image_dir: str, sample: Dict, task_type: str) -> str:    
        try:
            if 'image' in sample and sample['image']:
                image_path = sample['image'][0]
                image_name = os.path.basename(image_path)
                image_path = os.path.join(image_dir, image_name)
            
            def encode_image(image_path):
                with open(image_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode("utf-8")

            base64_image = encode_image(image_path)
            question = ""
            if task_type == 'multiple_choice':
                options_text = "\n".join([f"{k}) {v}" for k, v in sample['options'].items()])
                options_text += "\nOutput your response in JSON format:\n{\"reasoning\": \"\", \"answer\": \"The single letter of the correct option (e.g., A)\"}"
                question = f"{sample['instruction']}\n{options_text}"
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                },
                            },
                            {
                                "type": "text", 
                                "text": question
                            },
                        ],
                    }
                ]
            else:
                question = sample['instruction']
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                },
                            },
                            {
                                "type": "text", 
                                "text": question
                            },
                        ],
                    }
                ] 
            print(self.temperature)
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=4096,
                # stream=True,
                # The enable_thinking parameter enables reasoning, and thinking_budget sets the maximum reasoning token count.
                extra_body={
                    'enable_thinking': False,
                    # "thinking_budget": 81920
                },
                # Uncomment the following lines to return token usage in the last chunk.
                # stream_options={
                #     "include_usage": True
                # }
            )
            return question, completion.choices[0].message.content
        except Exception as e:
            print(f"Qwen API call error: {e}")
            return "", ""


class GeminiClient(ModelClient):
    """Google Gemini API client."""
    
    def __init__(self, model_name: str, api_key: str, **kwargs):
        super().__init__(model_name, api_key, **kwargs)
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("Please install the google-generativeai package: pip install google-generativeai")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
    
    def generate(self, image_dir: str, sample: Dict) -> str:
        """Call the Gemini API to generate a response."""
        try:
            prompt = "\n".join([msg['content'] if isinstance(msg['content'], str) 
                               else str(msg['content']) for msg in messages])
            
            response = self.model.generate_content(
                prompt,
                generation_config={
                    'temperature': self.kwargs.get('temperature', 0.0),
                    'max_output_tokens': self.kwargs.get('max_tokens', 2048)
                }
            )
            return response.text
        except Exception as e:
            print(f"Gemini API call error: {e}")
            return ""


class QwenVL(ModelClient):
    
    def __init__(self, model_name: str, **kwargs):
        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration
        super().__init__(model_name, **kwargs)
        
        print(f"Loading local model: {model_name}")
        # MOE model
        if 'A' in model_name:
            self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                model_name, dtype="auto", device_map="auto"
            )
        else:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name, dtype="auto", device_map="auto"
            )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.batch_size = kwargs.get('batch_size', 1)
        self.temperature = kwargs.get('temperature', 0.0)
    
    def generate(self, image_dir: str, sample: Dict, task_type: str) -> str:
        """Generate a response with Qwen-VL (single sample)."""
        try:
            if 'image' in sample and sample['image']:
                image_path = sample['image'][0]
                image_name = os.path.basename(image_path)
                image_path = os.path.join(image_dir, image_name)

            question = ""
            if task_type == 'multiple_choice':
                options_text = "\n".join([f"{k}) {v}" for k, v in sample['options'].items()])
                options_text += "\nOutput your response in JSON format:\n{\"reasoning\": \"\", \"answer\": \"The single letter of the correct option (e.g., A)\"}"
                question = f"{sample['instruction']}\n{options_text}"
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image_path,
                            },
                            {
                                "type": "text", 
                                "text": question
                            },
                        ],
                    }
                ]
            else:
                question = sample['instruction']
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image_path,
                            },
                            {
                                "type": "text", 
                                "text": question
                            },
                        ],
                    }
                ] 
            # Prepare inputs for inference.
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            )
            inputs = inputs.to(self.model.device)

            # Inference: generate the output.
            generated_ids = self.model.generate(**inputs, max_new_tokens=1024, temperature=self.temperature, do_sample=False, repetition_penalty=1.1, no_repeat_ngram_size=3)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return question, output_text[0] if output_text else ""
        except Exception as e:
            print(f"Qwen-VL inference error: {e}")
            import traceback
            traceback.print_exc()
            return "", ""
    
    def generate_batch(self, image_dir: str, samples: List[Dict], task_type: str) -> List[str]:
        """Generate responses with Qwen-VL in batch."""
        self.processor.tokenizer.padding_side = 'left'
        try:
            batch_messages = []
            questions = []
            for sample in samples:
                if 'image' in sample and sample['image']:
                    image_path = sample['image'][0]
                    image_name = os.path.basename(image_path)
                    image_path = os.path.join(image_dir, image_name)
                
                if task_type == 'multiple_choice':
                    options_text = "\n".join([f"{k}) {v}" for k, v in sample['options'].items()])
                    # options_text += "\nOutput your response in JSON format:\n{\"reasoning\": \"\", \"answer\": \"The single letter of the correct option (e.g., A)\"}"
                    options_text += "\nOutput your response in JSON format:\n{\"answer\": \"The single letter of the correct option (e.g., A)\"}"
                    question = f"{sample['instruction']}\n{options_text}"
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": image_path,
                                },
                                {
                                    "type": "text", 
                                    "text": question
                                },
                            ],
                        }
                    ]
                elif task_type == 'spatial_relationship_w_depth':
                    depth_path = os.path.join(f"{image_dir}_depth", os.path.basename(image_path))
                    options_text = "\n".join([f"{k}) {v}" for k, v in sample['options'].items()])
                    options_text += "\nOutput your response in JSON format:\n{\"answer\": \"The single letter of the correct option (e.g., A)\"}"
                    question = f"{sample['instruction']}\n{options_text}"
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": image_path,
                                },
                                {
                                    "type": "image",
                                    "image": depth_path,
                                },
                                {
                                    "type": "text", 
                                    "text": question
                                },
                            ],
                        }
                    ]
                else:
                    question = sample['instruction']
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": image_path,
                                },
                                {
                                    "type": "text", 
                                    "text": question
                                },
                            ],
                        }
                    ]
                questions.append(question)
                batch_messages.append(messages)
            
            # Prepare inputs for batch inference.
            inputs = self.processor.apply_chat_template(
                batch_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True
            )
            inputs = inputs.to(self.model.device)

            # Batch inference.
            if task_type == 'multiple_choice' or task_type == 'visual_grounding':
                generated_ids = self.model.generate(**inputs, max_new_tokens=1024, temperature=self.temperature, do_sample=False)
            elif task_type == 'open_ended' or task_type == 'true_or_false':
                generated_ids = self.model.generate(**inputs, max_new_tokens=1024, temperature=self.temperature, do_sample=False, repetition_penalty=1.1, no_repeat_ngram_size=3)
            else:
                generated_ids = self.model.generate(**inputs, max_new_tokens=1024, temperature=self.temperature, do_sample=False)
            
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_texts = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return questions, output_texts
        except Exception as e:
            print(f"Qwen-VL batch inference error: {e}")
            import traceback
            traceback.print_exc()
            return [("", "")] * len(samples)

class LLaVA(ModelClient):
    
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        
        print(f"Loading local model: {model_name}")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto", trust_remote_code=True
        )
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.batch_size = kwargs.get('batch_size', 1)
        self.temperature = kwargs.get('temperature', 0.0)

    def generate(self, image_dir: str, sample: Dict, task_type: str) -> str:
        try:
            if 'image' in sample and sample['image']:
                image_path = sample['image'][0]
                image_name = os.path.basename(image_path)
                image_path = os.path.join(image_dir, image_name)
            question = ""
            if task_type == 'multiple_choice':
                options_text = "\n".join([f"{k}) {v}" for k, v in sample['options'].items()])
                options_text += "\nOutput your response in JSON format:\n{\"reasoning\": \"\", \"answer\": \"The single letter of the correct option (e.g., A)\"}"
                question = f"{sample['instruction']}\n{options_text}"
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image_path,
                            },
                            {
                                "type": "text", 
                                "text": question
                            },
                        ],
                    }
                ]
            else:
                question = sample['instruction']
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image_path,
                            },
                            {
                                "type": "text", 
                                "text": question
                            },
                        ],
                    }
                ]
            # Preparation for inference
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False,
                add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)

            # Inference: Generation of the output
            # generated_ids = self.model.generate(**inputs, max_new_tokens=1024, temperature=self.temperature, do_sample=False, repetition_penalty=1.1, no_repeat_ngram_size=3)
            generated_ids = self.model.generate(**inputs, max_new_tokens=1024, temperature=self.temperature, do_sample=False)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return question, output_text if output_text else ""
        except Exception as e:
            print(f"LLaVA inference error: {e}")
            import traceback
            traceback.print_exc()
            return "", ""
    
    def generate_batch(self, image_dir: str, samples: List[Dict], task_type: str) -> List[str]:
        self.processor.tokenizer.padding_side = 'left'
        try:
            batch_messages = []
            questions = []
            for sample in samples:
                if 'image' in sample and sample['image']:
                    image_path = sample['image'][0]
                    image_name = os.path.basename(image_path)
                    image_path = os.path.join(image_dir, image_name)
                
                if task_type == 'multiple_choice':
                    options_text = "\n".join([f"{k}) {v}" for k, v in sample['options'].items()])
                    # options_text += "\nOutput your response in JSON format:\n{\"reasoning\": \"\", \"answer\": \"The single letter of the correct option (e.g., A)\"}"
                    options_text += "\nOutput your response in JSON format:\n{\"answer\": \"The single letter of the correct option (e.g., A)\"}"
                    question = f"{sample['instruction']}\n{options_text}"
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": image_path,
                                },
                                {
                                    "type": "text", 
                                    "text": question
                                },
                            ],
                        }
                    ]
                elif task_type == 'spatial_relationship_w_depth':
                    depth_path = os.path.join(f"{image_dir}_depth", os.path.basename(image_path))
                    options_text = "\n".join([f"{k}) {v}" for k, v in sample['options'].items()])
                    options_text += "\nOutput your response in JSON format:\n{\"answer\": \"The single letter of the correct option (e.g., A)\"}"
                    question = f"{sample['instruction']}\n{options_text}"
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": image_path,
                                },
                                {
                                    "type": "image",
                                    "image": depth_path,
                                },
                                {
                                    "type": "text", 
                                    "text": question
                                },
                            ],
                        }
                    ]
                    print("image_path: ", image_path, "depth_path: ", depth_path)
                else:
                    # if task_type == 'true_or_false':
                    #     question = sample['instruction'].replace("Provide a concise analysis, and then conclude with Yes or No.", "First output a concise analysis process, and then conclude with Yes or No.")
                    # else:
                    #     question = sample['instruction']
                    question = sample['instruction']
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": image_path,
                                },
                                {
                                    "type": "text", 
                                    "text": question
                                },
                            ],
                        }
                    ]
                questions.append(question)
                batch_messages.append(messages)
            
            # Preparation for batch inference
            texts = [
                self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                for messages in batch_messages
            ]
            image_inputs, video_inputs = process_vision_info(batch_messages)
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)

            # Batch inference
            if task_type == 'multiple_choice' or task_type == 'visual_grounding':
                generated_ids = self.model.generate(**inputs, max_new_tokens=1024, temperature=self.temperature, do_sample=False)
            elif task_type == 'open_ended' or task_type == 'true_or_false':
                generated_ids = self.model.generate(**inputs, max_new_tokens=1024, temperature=self.temperature, do_sample=False, repetition_penalty=1.1, no_repeat_ngram_size=3)
            else:
                generated_ids = self.model.generate(**inputs, max_new_tokens=1024, temperature=self.temperature, do_sample=False)
            
            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
            output_texts = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return questions, output_texts
        except Exception as e:
            print(f"LLaVA batch inference error: {e}")
            import traceback
            traceback.print_exc()
            return [("", "")] * len(samples)

class InternVL(ModelClient):
    
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)
        
        print(f"Loading local model: {model_name}")
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map="auto").eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
        self.batch_size = kwargs.get('batch_size', 1)
        self.temperature = kwargs.get('temperature', 0.0)
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)

    def build_transform(self, input_size):
        MEAN, STD = self.IMAGENET_MEAN, self.IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # Calculate the existing image aspect ratio.
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # Find the closest aspect ratio to the target.
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # Calculate the target width and height.
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # Resize the image.
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # Split the image.
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image(self, image_path, input_size=448, max_num=12):
        image = Image.open(image_path).convert('RGB')
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def generate(self, image_dir: str, sample: Dict, task_type: str) -> str:
        try:
            generation_config = dict(max_new_tokens=1024, temperature=self.temperature, do_sample=False, repetition_penalty=1.1, no_repeat_ngram_size=3)
            if 'image' in sample and sample['image']:
                image_path = sample['image'][0]
                image_name = os.path.basename(image_path)
                image_path = os.path.join(image_dir, image_name)

            pixel_values = self.load_image(image_path, max_num=12).to(torch.bfloat16).to(self.model.device)
            if task_type == 'multiple_choice':
                options_text = "\n".join([f"{k}) {v}" for k, v in sample['options'].items()])
                options_text += "\nOutput your response in JSON format:\n{\"reasoning\": \"\", \"answer\": \"The single letter of the correct option (e.g., A)\"}"
                question = f"{sample['instruction']}\n{options_text}"
                response = self.model.chat(self.tokenizer, pixel_values, question, generation_config)
                return question, response
            elif task_type == 'visual_grounding':
                instruction = sample['instruction']
                # Extract the object from the last sentence.
                match = re.search(r'Please output the bounding box coordinates of (.+?) in JSON format\.', instruction)
                if match:
                    object_name = match.group(1)
                    # Replace the last sentence.
                    new_instruction = re.sub(
                        r'Please output the bounding box coordinates of .+? in JSON format\.',
                        f'Please provide the bounding box coordinate of the region this sentence describes: <ref>{object_name.replace("**", "")}</ref>',
                        instruction
                    )
                    question = new_instruction
                else:
                    question = instruction
                response = self.model.chat(self.tokenizer, pixel_values, question, generation_config)
                return question, response
            else:
                question = sample['instruction']
                response = self.model.chat(self.tokenizer, pixel_values, question, generation_config)
                return question, response
        except Exception as e:
            print(f"InternVL inference error: {e}")
            import traceback
            traceback.print_exc()
            return "", ""
    
    def generate_batch(self, image_dir: str, samples: List[Dict], task_type: str) -> List[str]:
        try:
            if task_type == 'multiple_choice' or task_type == 'visual_grounding':
                generation_config = dict(max_new_tokens=1024, temperature=self.temperature, do_sample=False)
            elif task_type == 'open_ended' or task_type == 'true_or_false':
                generation_config = dict(max_new_tokens=1024, temperature=self.temperature, do_sample=False, repetition_penalty=1.1, no_repeat_ngram_size=3)
            else:
                generation_config = dict(max_new_tokens=1024, temperature=self.temperature, do_sample=False)
            num_patches_list = []
            pixel_values_list = []
            questions = []
            for sample in samples:
                if 'image' in sample and sample['image']:
                    image_path = sample['image'][0]
                    image_name = os.path.basename(image_path)
                    image_path = os.path.join(image_dir, image_name)
                pixel_values = self.load_image(image_path, max_num=12).to(torch.bfloat16).to(self.model.device)
                if task_type == 'multiple_choice':
                    options_text = "\n".join([f"{k}) {v}" for k, v in sample['options'].items()])
                    # options_text += "\nOutput your response in JSON format:\n{\"reasoning\": \"\", \"answer\": \"The single letter of the correct option (e.g., A)\"}"
                    options_text += "\nOutput your response in JSON format:\n{\"answer\": \"The single letter of the correct option (e.g., A)\"}"
                    question = f"{sample['instruction']}\n{options_text}"
                elif task_type == 'spatial_relationship_w_depth':
                    depth_path = os.path.join(f"{image_dir}_depth", os.path.basename(image_path))
                    depth_pixel_values = self.load_image(depth_path, max_num=12).to(torch.bfloat16).to(self.model.device)
                    options_text = "\n".join([f"{k}) {v}" for k, v in sample['options'].items()])
                    options_text += "\nOutput your response in JSON format:\n{\"answer\": \"The single letter of the correct option (e.g., A)\"}"
                    question = f"<image>\n<image>\n{sample['instruction']}\n{options_text}"
                    
                elif task_type == 'visual_grounding':
                    import re
                    instruction = sample['instruction']
                    # Extract the object from the last sentence.
                    match = re.search(r'Please output the bounding box coordinates of (.+?) in JSON format\.', instruction)
                    if match:
                        object_name = match.group(1)
                        # Replace the last sentence.
                        # new_instruction = re.sub(
                        #     r'Please output the bounding box coordinates of .+? in JSON format\.',
                        #     f'Please provide the bounding box coordinate of the region this sentence describes: <ref>{object_name.replace("**", "")}</ref>',
                        #     instruction
                        # )
                        new_instruction = re.sub(
                            r'Please output the bounding box coordinates of .+? in JSON format\.',
                            f'Please output the bounding box coordinates of <ref>{object_name}</ref>.',
                            instruction
                        )
                        question = new_instruction
                    else:
                        question = instruction
                else:
                    question = sample['instruction']
                
                if task_type == 'spatial_relationship_w_depth':
                    num_patches_list.append(pixel_values.size(0) + depth_pixel_values.size(0))
                    pixel_values_list.append(pixel_values)
                    pixel_values_list.append(depth_pixel_values)
                else:
                    num_patches_list.append(pixel_values.size(0))
                    pixel_values_list.append(pixel_values)
                questions.append(question)

            batch_pixel_values = torch.cat(pixel_values_list, dim=0)
            batch_responses = self.model.batch_chat(self.tokenizer, batch_pixel_values,
                num_patches_list=num_patches_list,
                questions=questions,
                generation_config=generation_config)
            return questions, batch_responses
            
        except Exception as e:
            print(f"InternVL batch inference error: {e}")
            import traceback
            traceback.print_exc()
            return [("", "")] * len(samples)

class RynnBrain(ModelClient):

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)

        print(f"Loading local model: {model_name}")
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.batch_size = kwargs.get('batch_size', 1)
        self.temperature = kwargs.get('temperature', 0.0)

    def _build_messages(self, image_path: str, question: str) -> list:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": question},
                ],
            }
        ]

    def generate(self, image_dir: str, sample: Dict, task_type: str):
        try:
            if 'image' in sample and sample['image']:
                image_path = os.path.join(image_dir, os.path.basename(sample['image'][0]))

            if task_type == 'multiple_choice':
                options_text = "\n".join([f"{k}) {v}" for k, v in sample['options'].items()])
                # options_text += "\nOutput your response in JSON format:\n{\"reasoning\": \"\", \"answer\": \"The single letter of the correct option (e.g., A)\"}"
                options_text += "\nOutput your response in JSON format:\n{\"answer\": \"The single letter of the correct option (e.g., A)\"}"
                question = f"{sample['instruction']}\n{options_text}"
            else:
                question = sample['instruction']

            messages = self._build_messages(image_path, question)
            model_inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            model_inputs = model_inputs.to(self.model.device)

            output_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=1024,
                do_sample=False,
                temperature=self.temperature,
            )
            output_ids = output_ids[:, model_inputs["input_ids"].size(1):]
            response = self.processor.decode(output_ids[0], skip_special_tokens=True)
            return question, response
        except Exception as e:
            print(f"RynnBrain inference error: {e}")
            import traceback
            traceback.print_exc()
            return "", ""

    def generate_batch(self, image_dir: str, samples: List[Dict], task_type: str):
        self.processor.tokenizer.padding_side = 'left'
        try:
            batch_messages = []
            questions = []
            for sample in samples:
                if 'image' in sample and sample['image']:
                    image_path = os.path.join(image_dir, os.path.basename(sample['image'][0]))

                if task_type == 'multiple_choice':
                    options_text = "\n".join([f"{k}) {v}" for k, v in sample['options'].items()])
                    # options_text += "\nOutput your response in JSON format:\n{\"reasoning\": \"\", \"answer\": \"The single letter of the correct option (e.g., A)\"}"
                    options_text += "\nOutput your response in JSON format:\n{\"answer\": \"The single letter of the correct option (e.g., A)\"}"
                    question = f"{sample['instruction']}\n{options_text}"
                elif task_type == 'visual_grounding':
                    instruction = sample['instruction']
                    # Extract the object from the last sentence.
                    match = re.search(r'Please output the bounding box coordinates of (.+?) in JSON format\.', instruction)
                    if match:
                        object_name = match.group(1)
                        # Replace the last sentence.
                        new_instruction = re.sub(
                            r'Please output the bounding box coordinates of .+? in JSON format\.',
                            f'Generate coordinates for {object_name.replace("**", "")} bounding box. Constraints: x1,y1,x2,y2 ∈ [0,1000]. Response must be in the format: <object> (x1, y1), (x2, y2) </object>',
                            instruction
                        )
                    question = new_instruction
                else:
                    question = sample['instruction']

                questions.append(question)
                batch_messages.append(self._build_messages(image_path, question))

            model_inputs = self.processor.apply_chat_template(
                batch_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
            model_inputs = model_inputs.to(self.model.device)

            if task_type in ('open_ended', 'true_or_false'):
                output_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                    temperature=self.temperature,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3,
                )
            else:
                output_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                    temperature=self.temperature,
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs["input_ids"], output_ids)
            ]
            output_texts = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return questions, output_texts
        except Exception as e:
            print(f"RynnBrain batch inference error: {e}")
            import traceback
            traceback.print_exc()
            return [("", "")] * len(samples)

class RoboBrain(ModelClient):

    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name, **kwargs)

        print(f"Loading local model: {model_name}")
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.batch_size = kwargs.get('batch_size', 1)
        self.temperature = kwargs.get('temperature', 0.0)

    def _build_messages(self, image_path: str, question: str) -> list:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": question},
                ],
            }
        ]

    def generate(self, image_dir: str, sample: Dict, task_type: str):
        try:
            if 'image' in sample and sample['image']:
                image_path = os.path.join(image_dir, os.path.basename(sample['image'][0]))

            if task_type == 'multiple_choice':
                options_text = "\n".join([f"{k}) {v}" for k, v in sample['options'].items()])
                # options_text += "\nOutput your response in JSON format:\n{\"reasoning\": \"\", \"answer\": \"The single letter of the correct option (e.g., A)\"}"
                options_text += "\nOutput your response in JSON format:\n{\"answer\": \"The single letter of the correct option (e.g., A)\"}"
                question = f"{sample['instruction']}\n{options_text}"
            else:
                question = sample['instruction']

            messages = self._build_messages(image_path, question)
            model_inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            model_inputs = model_inputs.to(self.model.device)

            output_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=1024,
                do_sample=False,
                temperature=self.temperature,
            )
            output_ids = output_ids[:, model_inputs["input_ids"].size(1):]
            response = self.processor.decode(output_ids[0], skip_special_tokens=True)
            return question, response
        except Exception as e:
            print(f"RynnBrain inference error: {e}")
            import traceback
            traceback.print_exc()
            return "", ""

    def generate_batch(self, image_dir: str, samples: List[Dict], task_type: str):
        self.processor.tokenizer.padding_side = 'left'
        try:
            batch_messages = []
            questions = []
            for sample in samples:
                if 'image' in sample and sample['image']:
                    image_path = os.path.join(image_dir, os.path.basename(sample['image'][0]))

                if task_type == 'multiple_choice':
                    options_text = "\n".join([f"{k}) {v}" for k, v in sample['options'].items()])
                    # options_text += "\nOutput your response in JSON format:\n{\"reasoning\": \"\", \"answer\": \"The single letter of the correct option (e.g., A)\"}"
                    options_text += "\nOutput your response in JSON format:\n{\"answer\": \"The single letter of the correct option (e.g., A)\"}"
                    question = f"{sample['instruction']}\n{options_text}"
                elif task_type == 'visual_grounding':
                    instruction = sample['instruction']
                    # Extract the object from the last sentence.
                    match = re.search(r'Please output the bounding box coordinates of (.+?) in JSON format\.', instruction)
                    if match:
                        object_name = match.group(1)
                        # Replace the last sentence.
                        new_instruction = re.sub(
                            r'Please output the bounding box coordinates of .+? in JSON format\.',
                            f'Please provide the bounding box coordinate of the region this sentence describes: {object_name.replace("**", "")}',
                            instruction
                        )
                    question = new_instruction
                else:
                    question = sample['instruction']

                questions.append(question)
                batch_messages.append(self._build_messages(image_path, question))

            text = self.processor.apply_chat_template(
                batch_messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(batch_messages)
            model_inputs = self.processor(
                text=text,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            model_inputs = model_inputs.to(self.model.device)

            if task_type in ('open_ended', 'true_or_false'):
                output_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                    temperature=self.temperature,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=3,
                )
            else:
                output_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                    temperature=self.temperature,
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs["input_ids"], output_ids)
            ]
            output_texts = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return questions, output_texts
        except Exception as e:
            print(f"RynnBrain batch inference error: {e}")
            import traceback
            traceback.print_exc()
            return [("", "")] * len(samples)


class InferenceEngine:
    """Inference engine."""
    
    def __init__(self, model_client: ModelClient, testset_dir: str, image_dir: str, output_dir: str, batch_size: int = 1):
        self.model_client = model_client
        self.testset_dir = Path(testset_dir)
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        
        
    def load_testset(self, filepath: str) -> List[Dict]:
        """Load a test set file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def infer_file(self, test_file: Path, output_file: Path) -> List[Dict[str, Any]]:
        """Run inference on a single test file and save results in real time."""
        print(f"\nInferring file: {test_file.name}")
        
        # Load test data.
        test_data = self.load_testset(test_file)
        # test_data = test_data[:10]
        # Determine whether this is a multiple-choice task.
        is_multiple_choice = 'options' in test_data[0] if test_data else False
        if is_multiple_choice:
            task_type = 'multiple_choice'
        elif 'task_planning' in test_file.name:
            task_type = 'open_ended'
        elif 'safety' in test_file.name:
            task_type = 'true_or_false'
        elif 'visual_grounding' in test_file.name:
            task_type = 'visual_grounding'
        else: 
            task_type = 'open_ended'
        
        # if 'spatial_relationship_w_depth' in test_file.name:
        #     task_type = 'spatial_relationship_w_depth'

        # Store prediction results.
        predictions = []
        
        # If the output file already exists, load existing results.
        if output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    predictions = json.load(f)
                print(f"Loaded {len(predictions)} existing results; resuming from sample {len(predictions)}")
            except Exception as e:
                print(f"Failed to load existing results: {e}. Starting from scratch.")
                predictions = []
        
        # Check whether batch inference is supported.
        use_batch = hasattr(self.model_client, 'generate_batch') and self.batch_size > 1
        
        # Compute the starting position.
        start_from = len(predictions)
        
        if use_batch:
            # Batch inference.
            print(f"Using batch inference, batch size: {self.batch_size}")
            num_batches = (len(test_data) + self.batch_size - 1) // self.batch_size
            start_batch = start_from // self.batch_size
            
            for batch_idx in tqdm(range(start_batch, num_batches), desc=f"Inferring {test_file.name}"):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(test_data))
                batch_samples = test_data[start_idx:end_idx]
                
                try:
                    # Call the model to generate responses for the batch.
                    questions, responses = self.model_client.generate_batch(self.image_dir, batch_samples, task_type)
                    
                    # Record prediction results.
                    for i, (sample, question, response) in enumerate(zip(batch_samples, questions, responses)):
                        idx = start_idx + i
                        pred_result = {
                            'sample_id': idx,
                            'instruction': sample['instruction'],
                            'question': question,
                            'image': sample.get('image', [])
                        }
                        
                        if is_multiple_choice:
                            pred_result['options'] = sample['options']
                            pred_result['ground_truth'] = sample['correct_option']
                        else:
                            pred_result['ground_truth'] = sample['response']
                        pred_result['prediction'] = response
                        pred_result['task_type'] = task_type                   
                        # if 'spatial' in test_file.name:
                        #     pred_result['perspective'] = sample['perspective']
                        predictions.append(pred_result)
                    
                    # Save results in real time.
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(predictions, f, ensure_ascii=False, indent=2)
                        
                except Exception as e:
                    print(f"Batch {batch_idx} inference failed: {e}")
                    import traceback
                    traceback.print_exc()
                    # Add failure records.
                    for i, sample in enumerate(batch_samples):
                        idx = start_idx + i
                        predictions.append({
                            'sample_id': idx,
                            'instruction': sample['instruction'],
                            'question': question,
                            'ground_truth': sample['response'],
                            'prediction': "",
                            'image': sample.get('image', []),
                            'error': str(e),
                            'task_type': task_type
                        })
                    
                    # Save results even on failure.
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(predictions, f, ensure_ascii=False, indent=2)
        else:
            # Single-sample inference.
            print("Using single-sample inference")
            for idx in tqdm(range(start_from, len(test_data)), desc=f"Inferring {test_file.name}"):
                sample = test_data[idx]
                try:
                    # Call the model to generate a response.
                    question, response = self.model_client.generate(self.image_dir, sample, task_type)
                    
                    # Record prediction results.
                    pred_result = {
                        'sample_id': idx,
                        'instruction': sample['instruction'],
                        'question': question,
                        'image': sample.get('image', [])
                    }
                    
                    if is_multiple_choice:
                        pred_result['options'] = sample['options']
                        pred_result['ground_truth'] = sample['correct_option']
                    else:
                        pred_result['ground_truth'] = sample['response']
                    pred_result['prediction'] = response
                    pred_result['task_type'] = task_type 
                    if 'spatial' in test_file.name:
                        pred_result['perspective'] = sample['perspective']
                    
                    predictions.append(pred_result)
                    
                    # Save results in real time.
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(predictions, f, ensure_ascii=False, indent=2)
                    
                except Exception as e:
                    print(f"Sample {idx} inference failed: {e}")
                    # Add failure records.
                    predictions.append({
                        'sample_id': idx,
                        'instruction': sample['instruction'],
                        'question': question,
                        'ground_truth': sample['response'],
                        'prediction': "",
                        'image': sample.get('image', []),
                        'error': str(e),
                        'task_type': task_type
                    })
                    
                    # Save results even on failure.
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(predictions, f, ensure_ascii=False, indent=2)
        
        return predictions
    
    def run_inference(self):
        """Run the full inference pipeline."""
        print("=" * 50)
        print("Starting inference")
        print("=" * 50)
        
        # Get all test files.
        test_files = sorted(self.testset_dir.glob("*.json"))
        
        if not test_files:
            print(f"Error: no test files found in {self.testset_dir}")
            return
        
        print(f"Found {len(test_files)} test files")
        
        # Store all results.
        all_predictions = {}
        
        # Run inference for each file.
        for test_file in test_files:
            try:
                # Prepare the output file path.
                model_name = self.model_client.model_name.split("/")[-1]
                predictions_file = self.output_dir / f"{model_name}_{test_file.stem}_predictions.json"
                
                # Run inference (results are saved in real time).
                predictions = self.infer_file(test_file, predictions_file)
                print(f"Completed {test_file.name}: {len(predictions)} samples")
                print(f"Inference results saved to: {predictions_file}")
                
            except Exception as e:
                print(f"Error while inferring {test_file.name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
            # # Statistics
            # total_samples = sum(len(preds) for preds in all_predictions.values())
            # print("\n" + "=" * 50)
            # print("Inference complete!")
            # print("=" * 50)
            # print(f"Total samples: {total_samples}")
            # print(f"Number of test files: {len(all_predictions)}")
            # for task_name, preds in all_predictions.items():
            #     print(f"  {task_name}: {len(preds)} samples")

def create_model_client(args) -> ModelClient:
    """Create a model client from CLI arguments."""
    kwargs = {
        'temperature': args.temperature,
        'max_tokens': args.max_tokens,
        'batch_size': args.batch_size
    }
    
    if args.model_type == 'openai':
        return OpenAIClient(
            model_name=args.model_name,
            api_key=args.api_key,
            base_url=args.base_url,
            **kwargs
        )
    elif args.model_type == 'claude':
        return ClaudeClient(
            model_name=args.model_name,
            api_key=args.api_key,
            **kwargs
        )
    elif args.model_type == 'qwen':
        return QwenClient(
            model_name=args.model_name,
            api_key=args.api_key,
            **kwargs
        )
    elif args.model_type == 'gemini':
        return GeminiClient(
            model_name=args.model_name,
            api_key=args.api_key,
            **kwargs
        )
    elif args.model_type == 'qwenvl':
        return QwenVL(
            model_name=args.model_name,
            **kwargs
        )
    elif args.model_type == 'llama':
        return Llama(
            model_name=args.model_name,
            **kwargs
        )
    elif args.model_type == 'llava':
        return LLaVA(
            model_name=args.model_name,
            **kwargs
        )
    elif args.model_type == 'internvl':
        return InternVL(
            model_name=args.model_name,
            **kwargs
        )
    elif args.model_type == 'rynnbrain':
        return RynnBrain(
            model_name=args.model_name,
            **kwargs
        )
    elif args.model_type == 'robobrain':
        return RoboBrain(
            model_name=args.model_name,
            extra_model_path=args.extra_model_path,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")


def main():
    parser = argparse.ArgumentParser(description='ERIBench inference script')
    
    # Model arguments.
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['openai', 'claude', 'qwen', 'gemini', 'qwenvl', 'llava', 'llama', 'internvl', 'rynnbrain', 'robobrain'],
                       help='Model type')
    parser.add_argument('--api_key', type=str, default='',
                       help='API key')
    parser.add_argument('--model_name', type=str, required=True,
                       help='Model name or path')
    parser.add_argument('--base_url', type=str, default=None,
                       help='Custom API base URL (optional)')
    parser.add_argument('--extra_model_path', type=str, default=None,
                       help='Extra model repository path (e.g. a RoboBrain repo directory; added to sys.path)')
    
    # Test arguments.
    parser.add_argument('--testset_dir', type=str, required=True,
                       help='Test set directory path')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Image directory path')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory path')
    
    # Generation arguments.
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Generation temperature')
    parser.add_argument('--max_tokens', type=int, default=2048,
                       help='Maximum generated token count')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch inference size')
    
    args = parser.parse_args()
    
    # Validate arguments.
    if not args.api_key and args.model_type not in ['qwenvl', 'llava', 'llama', 'internvl', 'rynnbrain', 'robobrain']:
        print("Warning: API key not provided")
    
    if not os.path.exists(args.testset_dir):
        print(f"Error: test set directory does not exist: {args.testset_dir}")
        return
    
    if not os.path.exists(args.image_dir):
        print(f"Error: image directory does not exist: {args.image_dir}")
        return
    
    # Create the model client.
    try:
        model_client = create_model_client(args)
    except Exception as e:
        print(f"Failed to create model client: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create the inference engine and run inference.
    inference_engine = InferenceEngine(
        model_client=model_client,
        testset_dir=args.testset_dir,
        image_dir = args.image_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )
    
    inference_engine.run_inference()


if __name__ == '__main__':
    main()
