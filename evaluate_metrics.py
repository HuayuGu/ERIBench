#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
import re
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from volcenginesdkarkruntime import Ark

class MetricsCalculator:
    """Metrics calculator."""
    
    def __init__(self, predictions_path: str, output_dir: str, model_name: str = None, image_dir: str = None, api_key: str = None, base_url: str = None, score_model: str = None):
        self.predictions_path = Path(predictions_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.image_dir = image_dir
        self.api_key = api_key
        self.base_url = base_url
        self.score_model = score_model
        # Determine whether the path is a file or a directory.
        if self.predictions_path.is_file():
            # Single file.
            self.predictions_files = [self.predictions_path]
            if model_name is None:
                model_name = self.predictions_path.stem.split('_predictions')[0]
        elif self.predictions_path.is_dir():
            # Directory: collect all prediction files.
            self.predictions_files = sorted(self.predictions_path.glob('*_predictions.json'))
            if not self.predictions_files:
                raise ValueError(f"No prediction files found in directory: {predictions_path}")
            if model_name is None:
                # Extract the model name from the first file.
                first_file = self.predictions_files[0].stem
                model_name = first_file.split('_')[0]
        else:
            raise ValueError(f"Invalid path: {predictions_path}")
        
        self.model_name = model_name.split('/')[-1] if model_name else "unknown_model"
        print(f"Model name: {self.model_name}")
        print(f"Found {len(self.predictions_files)} prediction files")

    
    def extract_answer(self, response, has_options: bool = False) -> str:
        """Extract the answer from the model response."""
        # If response is a list, take the first element.
        if isinstance(response, list):
            if not response:
                return ""
            response = response[0]
        
        if not response:
            return ""
        
        # Ensure response is a string.
        response = str(response)
        
        if has_options:
            # Multiple-choice: extract the answer field from a JSON string.
            try:
                import json
                import re
                
                 # If <answer>...</answer> exists, use the tagged content as the new response.
                answer_tag = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
                if answer_tag:
                    response = answer_tag.group(1).strip()

                json_str = None
                
                # 1. Try to extract JSON wrapped in a code block.
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    # 2. Try to extract a JSON object from the full response.
                    json_match = re.search(r'\{.*\}', response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                
                if json_str:
                    # Try parsing JSON using standard JSON first.
                    try:
                        data = json.loads(json_str)
                    except json.JSONDecodeError:
                        # If that fails, replace single quotes with double quotes.
                        json_str = json_str.replace("'", '"')
                        json_str = json_str.replace('"s', "'s")
                        data = json.loads(json_str)
                    
                    # Extract the answer field.
                    if 'answer' in data:
                        return str(data['answer']).strip()
                else:
                    # Extract [A-D] from the response.
                    match = re.search(r'\b([A-D])\b', response)
                    if match:
                        return match.group(1).strip()
            except (json.JSONDecodeError, AttributeError, ValueError):
                # Extract [A-D] from the response.
                match = re.search(r'\b([A-D])\b', response)
                if match:
                    return match.group(1).strip()
                print(response)
                print(f"JSON parsing failed: {json_str}")
                pass
            
            # If JSON parsing fails, return the original response.
            return ""
        else:
            # Open-ended QA: return the full response.
            return response.strip()
    
    def calculate_multiple_choice_metrics(self, predictions: List[Dict]) -> Dict[str, Any]:
        """Calculate multiple-choice metrics."""
        total = len(predictions)
        correct = 0
        error_count = 0
        
        detailed_results = []
        
        for pred in predictions:
            # Check whether this prediction has an error.
            if 'error' in pred:
                error_count += 1
                pred['correct'] = False
                detailed_results.append(pred)
                continue
            
            pred_answer = self.extract_answer(pred['prediction'], has_options=True)
            # Extract the first uppercase letter from pred_answer.
            first_upper = next((char for char in pred_answer if char.isupper()), None)
            if first_upper:
                pred_answer = first_upper
            gt_answer = pred['ground_truth']
            
            is_correct = pred_answer == gt_answer
            if is_correct:
                correct += 1
            
            pred['correct'] = is_correct
            pred['extracted_prediction'] = pred_answer
            pred['extracted_ground_truth'] = gt_answer
            detailed_results.append(pred)
        
        metrics = {
            'task_type': 'multiple_choice',
            'total': total,
            'correct': correct,
            'accuracy': correct / total if total > 0 else 0.0,
        }
        
        return metrics, detailed_results
    
    def calculate_true_or_false_metrics(self, predictions: List[Dict]) -> Dict[str, Any]:
        """Calculate true/false metrics."""
        total = len(predictions)
        correct = 0
        error_count = 0
        
        # Statistics for Precision, Recall, and F1 Score.
        # Treat "yes" as the positive class and "no" as the negative class.
        true_positive = 0   # Predicted yes, actual yes.
        true_negative = 0   # Predicted no, actual no.
        false_positive = 0  # Predicted yes, actual no.
        false_negative = 0  # Predicted no, actual yes.
        
        detailed_results = []
        
        def extract_yes_no(text: str) -> str:
            """Extract a yes/no answer from text."""
            if not text:
                return ""
            
            text = str(text).strip()
            
            # Match different yes/no variants.
            yes_patterns = [
                r'\bYes\b',
                r'\byes\b',
                r'\b是\b',
                r'\b对\b',
                r'\b正确\b',
            ]
            
            no_patterns = [
                r'\bNo\b',
                r'\b否\b',
                r'\b不\b',
                r'\b错误\b',
            ]
            
            # Check yes.
            for pattern in yes_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return 'yes'
            
            # Check no.
            for pattern in no_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return 'no'
            
            return text.strip().lower()
        
        for pred in predictions:
            # Check whether this prediction has an error.
            if 'error' in pred:
                error_count += 1
                pred['correct'] = False
                detailed_results.append(pred)
                continue
            
            # Extract the predicted answer.
            pred_text = pred['prediction']
            if isinstance(pred_text, list):
                pred_text = pred_text[0] if pred_text else ""
            pred_answer = extract_yes_no(pred_text)
            
            # Extract the ground-truth answer.
            gt_text = pred['ground_truth']
            if isinstance(gt_text, list):
                gt_text = gt_text[0] if gt_text else ""
            gt_answer = extract_yes_no(gt_text)
            
            # Determine correctness.
            is_correct = pred_answer == gt_answer
            if is_correct:
                correct += 1
            
            # Update the confusion matrix.
            if pred_answer == 'yes' and gt_answer == 'yes':
                true_positive += 1
            elif pred_answer == 'no' and gt_answer == 'no':
                true_negative += 1
            elif pred_answer == 'yes' and gt_answer == 'no':
                false_positive += 1
            elif pred_answer == 'no' and gt_answer == 'yes':
                false_negative += 1
            else:
                if gt_answer == 'yes':
                    false_negative += 1
                elif gt_answer == 'no':
                    false_positive += 1

            pred['correct'] = is_correct
            pred['extracted_prediction'] = pred_answer
            pred['extracted_ground_truth'] = gt_answer
            detailed_results.append(pred)
        
        # Compute accuracy.
        accuracy = correct / total if total > 0 else 0.0
        
        # Build metrics.
        metrics = {
            'task_type': 'true_or_false',
            'total': total,
            'correct': correct,
            'accuracy': accuracy,
        }
        
        return metrics, detailed_results

    def calculate_open_ended_metrics(self, predictions: List[Dict]) -> Dict[str, Any]:

        total = len(predictions)
        total_score = 0
        error_count = 0

        detailed_results = []
   
        client = Ark(
            base_url=self.base_url,
            api_key=self.api_key,
        )

        def ensure_score_model_accessible():
            """Verify the scoring model is reachable before evaluation."""
            if not self.score_model:
                raise ValueError("Open-ended evaluation requires score_model to be set")

            try:
                client.responses.create(
                    model=self.score_model,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_text",
                                    "text": "ping",
                                }
                            ],
                        }
                    ],
                    thinking={"type": "disabled"},
                )
            except Exception as e:
                print(
                    f"Error: cannot access scoring model {self.score_model}. "
                    f"Check api_key/base_url/model configuration: {e}"
                )
                raise RuntimeError(
                    f"Cannot access scoring model {self.score_model}. Check api_key/base_url/model configuration: {e}"
                ) from e

        ensure_score_model_accessible()
        
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        
        def evaluate_single_prediction(idx, pred):
            """Evaluate a single prediction."""
            if 'error' in pred:
                pred['correct'] = False
                return idx, pred, 0.0, True
            
            question = pred['instruction']
            pred_text = pred['prediction']
            gt_text = pred['ground_truth']
            
            image_name = pred['image'][0]
            image_path = os.path.join(self.image_dir, image_name)
            
            base64_image = encode_image(image_path)
            prompt = f'''
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question. You will be given an image in which the user is pointing to a target object. The user question is to generate a plan for a given task. You will be given a reference answer and the assistant's answer. Your evaluation should consider task completion, logic and user safety. Begin your evaluation by comparing the assistant's answer with the reference answer. Identify and correct any mistakes. Do not allow the length of the responses to influence your evaluation. Be as objective as possible. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".

[User Question]
{question}

[The Start of Reference Answer]
{gt_text}
[The End of Reference Answer]

[The Start of Assistant's Answer]
{pred_text}
[The End of Assistant's Answer]
            '''
            messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": f"data:image/png;base64,{base64_image}"
                            },
                            {
                                "type": "input_text", 
                                "text": prompt
                            },
                        ],
                    }
            ]
            response = client.responses.create(
                model=self.score_model,
                input=messages,
                thinking={"type":"disabled"},
                # reasoning={"effort":"low"}
            )
            result = response.output[0].content[0].text
            match = re.search(r'\[\[(\d+(?:\.\d+)?)\]\]', result)
            if match:
                score = float(match.group(1))
            else:
                score = 0.0
            
            pred['evaluation_result'] = result
            pred['score'] = score
            return idx, pred, score, False
        
        # Process with multithreading.
        results_dict = {}  # Store results in a dict keyed by index.
        with ThreadPoolExecutor(max_workers=100) as executor:
            # Submit all tasks.
            future_to_idx = {executor.submit(evaluate_single_prediction, idx, pred): idx 
                           for idx, pred in enumerate(predictions)}
            
            # Collect completed tasks.
            for future in tqdm(as_completed(future_to_idx), total=len(predictions), desc="Evaluating open-ended tasks"):
                idx, pred, score, is_error = future.result()
                results_dict[idx] = (pred, score, is_error)
        
        # Append detailed results in the original order and compute the total score.
        for idx in range(len(predictions)):
            pred, score, is_error = results_dict[idx]
            detailed_results.append(pred)
            if is_error:
                error_count += 1
            else:
                total_score += score
            
        metrics = {
            'task_type': 'open_ended',
            'total': total,
            'average_score': total_score / total if total > 0 else 0.0
        }
        
        return metrics, detailed_results
    
    def calculate_visual_grounding_metrics(self, predictions: List[Dict]) -> Dict[str, Any]:
        """Calculate visual grounding metrics."""
        total = len(predictions)
        correct = 0
        error_count = 0
        
        detailed_results = []
        
        # Create the directory for visualization outputs.
        vis_dir = self.output_dir / "visual_grounding_visualization"
        # Remove old visualization outputs.
        if vis_dir.exists():
            import shutil
            shutil.rmtree(vis_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)

        def extract_bbox_from_ground_truth(gt_text):
            """Extract a bbox from ground truth in absolute coordinates."""
            try:
                # Example ground_truth format: "[{\"bbox\": [648, 601, 706, 672], \"label\": \"cup\"}]"
                # Try parsing JSON.
                gt_data = json.loads(gt_text)
                if isinstance(gt_data, list) and len(gt_data) > 0:
                    bbox = gt_data[0].get('bbox', [0, 0, 0, 0])
                    return bbox
                return [0, 0, 0, 0]
            except Exception as e:
                # If JSON parsing fails, try a regular expression.
                pattern = r'"bbox":\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
                match = re.search(pattern, gt_text)
                if match:
                    return [int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))]
                print(f"Unable to extract bbox from ground_truth: {gt_text[:100]}")
                return [0, 0, 0, 0]
        
        def extract_bbox_from_prediction(pred_text):
            """Extract a bbox from prediction in normalized coordinates (0-1000)."""
            try:
                # Example prediction format: ["```json\n[\n\t{\"bbox_2d\": [337, 564, 367, 626], \"label\": \"cup\"}\n]\n```"]
                # Handle list values.
                if isinstance(pred_text, list):
                    pred_text = pred_text[0] if pred_text else ""
                
                pred_text = str(pred_text)
                
                # Remove markdown code block markers.
                pred_text = pred_text.replace('```json', '').replace('```', '').strip()
                
                # Try parsing JSON.
                try:
                    pred_data = json.loads(pred_text)
                    if isinstance(pred_data, list) and len(pred_data) > 0:
                        # Try to get bbox_2d or bbox.
                        bbox = pred_data[0].get('bbox_2d') or pred_data[0].get('bbox', [0, 0, 0, 0])
                        return bbox
                except:
                    pass
                
                # If JSON parsing fails, use regular expressions.
                # Prefer bbox_2d.
                pattern = r'"bbox_2d":\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
                match = re.search(pattern, pred_text)
                if match:
                    return [int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))]
                
                # Try matching a plain bbox.
                pattern = r'"bbox":\s*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
                match = re.search(pattern, pred_text)
                if match:
                    return [int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))]
                
                # Try matching any [x, y, x, y] format.
                pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
                match = re.search(pattern, pred_text)
                if match:
                    return [int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))]
                
                pattern = r'\[([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\]'
                match = re.search(pattern, pred_text)
                if match:
                    return [float(match.group(1)), float(match.group(2)), float(match.group(3)), float(match.group(4))]

                # Match <object> (x1, y1), (x2, y2) </object> format (e.g. RynnBrain models).
                pattern = r'<object>\s*\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\)\s*</object>'
                match = re.search(pattern, pred_text)
                if match:
                    return [int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))]

                print(f"Unable to extract bbox from prediction: {pred_text[:100]}")
                return [0, 0, 0, 0]
            except Exception as e:
                print(f"Error parsing prediction: {e}")
                return [0, 0, 0, 0]
                
        def calculate_iou(bbox1, bbox2):
            """
            Compute the intersection over union (IoU) of two bounding boxes.
            bbox format: [x1, y1, x2, y2] in absolute coordinates.
            """
            x1_1, y1_1, x2_1, y2_1 = bbox1
            x1_2, y1_2, x2_2, y2_2 = bbox2

            # Compute the intersection coordinates.
            x1_inter = max(x1_1, x1_2)
            y1_inter = max(y1_1, y1_2)
            x2_inter = min(x2_1, x2_2)
            y2_inter = min(y2_1, y2_2)

            # Compute the intersection area.
            inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

            # Compute the area of each bounding box.
            area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
            area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

            # Compute the union area.
            union_area = area1 + area2 - inter_area

            # Compute IoU.
            if union_area == 0:
                return 0.0
            return inter_area / union_area
        
        iou_threshold = 0.5
        total_iou = 0.0
        correct_iou_025 = 0  # Number correct at IoU > 0.25.
        correct_iou_075 = 0  # Number correct at IoU > 0.75.
        
        for pred in predictions:
            # Check whether this prediction has an error.
            if 'error' in pred:
                error_count += 1
                pred['correct'] = False
                pred['iou'] = 0.0
                detailed_results.append(pred)
                continue
            
            try:
                # Extract the ground-truth bbox in absolute coordinates.
                gt_bbox = extract_bbox_from_ground_truth(pred['ground_truth'])
                
                # Extract the prediction bbox in normalized coordinates (0-1000).
                pred_bbox_normalized = extract_bbox_from_prediction(pred['prediction'])
                
                # Get the image size.
                image_path = pred['image'][0] if isinstance(pred['image'], list) else pred['image']
                image_path = os.path.join(self.image_dir, os.path.basename(image_path))
                with Image.open(image_path) as img:
                    img_width, img_height = img.width, img.height
                
                # Convert normalized prediction coordinates to absolute coordinates.
                # Qwen3-VL models use 0-1000 normalized coordinates.
                if "Qwen3-VL" in self.model_name or "InternVL" in self.model_name or "qwen3-vl" in self.model_name or "RynnBrain" in self.model_name or "RoboBrain" in self.model_name:
                    pred_bbox_abs = [
                        pred_bbox_normalized[0] / 1000.0 * img_width, 
                        pred_bbox_normalized[1] / 1000.0 * img_height, 
                        pred_bbox_normalized[2] / 1000.0 * img_width, 
                        pred_bbox_normalized[3] / 1000.0 * img_height
                    ]
                else:
                    # Other models may already output absolute coordinates.
                    pred_bbox_abs = pred_bbox_normalized
                
                # Compute IoU.
                iou = calculate_iou(pred_bbox_abs, gt_bbox)
                total_iou += iou
                
                # Determine correctness (IoU > 0.5).
                is_correct = iou >= iou_threshold
                if is_correct:
                    correct += 1
                
                # Track accuracy at different IoU thresholds.
                if iou > 0.25:
                    correct_iou_025 += 1
                if iou > 0.75:
                    correct_iou_075 += 1
                
                # Draw the bbox on the image and save it.
                try:
                    # Re-open the image for drawing.
                    img = Image.open(image_path).convert('RGB')
                    draw = ImageDraw.Draw(img)
                    
                    # Try to load a font; fall back to the default font.
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
                    except:
                        font = ImageFont.load_default()
                    
                    # Draw the ground-truth bbox in green.
                    draw.rectangle(gt_bbox, outline='green', width=3)
                    draw.text((gt_bbox[0], gt_bbox[1] - 25), 'Ground Truth', fill='green', font=font)
                    
                    # Draw the predicted bbox in red.
                    draw.rectangle([round(x) for x in pred_bbox_abs], outline='red', width=3)
                    draw.text((pred_bbox_abs[0], pred_bbox_abs[1] - 50), f'Prediction (IOU: {iou:.3f})', fill='red', font=font)
                    
                    # Save the visualization result.
                    sample_id = pred.get('sample_id', 'unknown')
                    vis_filename = f"{sample_id}_iou_{iou:.3f}.jpg"
                    vis_path = vis_dir / vis_filename
                    img.save(vis_path)
                    
                    pred['visualization_path'] = str(vis_path)
                except Exception as vis_error:
                    print(f"Error saving visualization result: {vis_error}")
                    pred['visualization_error'] = str(vis_error)
                
                pred['correct'] = is_correct
                pred['iou'] = round(iou, 4)
                pred['extracted_prediction_normalized'] = pred_bbox_normalized
                pred['extracted_prediction_absolute'] = [round(x, 2) for x in pred_bbox_abs]
                pred['extracted_ground_truth'] = gt_bbox
                detailed_results.append(pred)
                
            except Exception as e:
                print(f"Error processing sample {pred.get('sample_id', 'unknown')}: {e}")
                error_count += 1
                pred['correct'] = False
                pred['iou'] = 0.0
                pred['error'] = str(e)
                detailed_results.append(pred)
        
        valid_count = total - error_count
        avg_iou = total_iou / valid_count if valid_count > 0 else 0.0
        
        metrics = {
            'task_type': 'visual_grounding',
            'total': total,
            'correct': correct,
            'valid': valid_count,
            'accuracy_iou_05': correct / total if total > 0 else 0.0,
            'accuracy_iou_025': correct_iou_025 / total if total > 0 else 0.0,
            'accuracy_iou_075': correct_iou_075 / total if total > 0 else 0.0,
            'average_iou': round(avg_iou, 4),
            'iou_threshold': iou_threshold,
            'visualization_dir': str(vis_dir)
        }
        
        print(f"Visualization results saved to: {vis_dir}")
        print(f"Accuracy (IoU > 0.5): {metrics['accuracy_iou_05']:.4f} ({correct}/{total})")
        print(f"Accuracy (IoU > 0.25): {metrics['accuracy_iou_025']:.4f} ({correct_iou_025}/{total})")
        print(f"Accuracy (IoU > 0.75): {metrics['accuracy_iou_075']:.4f} ({correct_iou_075}/{total})")
        print(f"Average IoU: {metrics['average_iou']:.4f}")
        
        return metrics, detailed_results
    
    def calculate_task_metrics(self, task_name: str, predictions: List[Dict]) -> tuple:
        """Calculate metrics for a single task."""
        if not predictions:
            return {'error': 'No prediction results'}, []
        
        # Determine the task type.
        task_type = predictions[0].get('task_type', 'unknown')
        
        # Check whether this is a spatial relationship task.
        if 'spatial_relationship' in task_name.lower():
            return self.calculate_multiple_choice_metrics(predictions)
        elif task_type == 'multiple_choice':
            return self.calculate_multiple_choice_metrics(predictions)
        elif task_type == 'true_or_false':
            return self.calculate_true_or_false_metrics(predictions)
        elif task_type == 'open_ended':
            return self.calculate_open_ended_metrics(predictions)
        elif task_type == 'visual_grounding':
            return self.calculate_visual_grounding_metrics(predictions)
        else:
            # Default to open-ended QA.
            return self.calculate_open_ended_metrics(predictions)
    
    def calculate_all_metrics(self):
        """Calculate metrics for all tasks."""
        print("\n" + "=" * 50)
        print("Calculating evaluation metrics")
        print("=" * 50)
        
        all_metrics = {}
        detailed_results = {}
        
        # Iterate over all prediction files.
        for pred_file in self.predictions_files:
            # Extract the task name from the filename.
            file_stem = pred_file.stem
            # Remove the model-name prefix and the _predictions suffix.
            task_name = file_stem.replace(f"{self.model_name}_", "").replace("_predictions", "")

            print(f"\nProcessing task: {task_name}")
            print(f"File: {pred_file.name}")
            try:
                # Load prediction results.
                with open(pred_file, 'r', encoding='utf-8') as f:
                    predictions = json.load(f)
                
                print(f"Sample count: {len(predictions)}")
                
                # Calculate metrics.
                metrics, detailed_preds = self.calculate_task_metrics(task_name, predictions)
                all_metrics[task_name] = metrics
                detailed_results[task_name] = detailed_preds
                
                # Print metrics.
                print(f"Metrics:")
                for key, value in metrics.items():
                    if key != 'note':
                        print(f"  {key}: {value}")
                
            except Exception as e:
                print(f"Error calculating metrics for {task_name}: {e}")
                import traceback
                traceback.print_exc()
                all_metrics[task_name] = {'error': str(e)}
                detailed_results[task_name] = []
        
        # Save results.
        self.save_results(all_metrics, detailed_results)
        
        # Print summary.
        # self.print_summary(all_metrics)
    
    def save_results(self, all_metrics: Dict, detailed_results: Dict):
        """Save evaluation results."""
        model_name_safe = self.model_name.replace('/', '_').replace('\\', '_')
        
        # Save the metrics summary.
        metrics_file = self.output_dir / f"metrics_{model_name_safe}.json"
        metrics_data = {
            'model_name': self.model_name,
            'timestamp': self.timestamp,
            'predictions_path': str(self.predictions_path),
            'metrics': all_metrics
        }
        
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nMetrics saved to: {metrics_file}")
        
        # Save detailed results, including correctness labels.
        detailed_file = self.output_dir / f"detailed_results_{model_name_safe}.json"
        detailed_data = {
            'model_name': self.model_name,
            'timestamp': self.timestamp,
            'predictions_path': str(self.predictions_path),
            'detailed_results': detailed_results
        }
        
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_data, f, ensure_ascii=False, indent=2)
        
        print(f"Detailed results saved to: {detailed_file}")
    
    def print_summary(self, all_metrics: Dict):
        """Print the evaluation summary."""
        print("\n" + "=" * 50)
        print("Evaluation Summary")
        print("=" * 50)
        
        # Summarize multiple-choice tasks.
        mc_tasks = []
        for task_name, metrics in all_metrics.items():
            if metrics.get('task_type') == 'multiple_choice':
                mc_tasks.append((task_name, metrics))
        
        if mc_tasks:
            print("\nMultiple-choice tasks:")
            total_acc = []
            for task_name, metrics in mc_tasks:
                acc = metrics.get('accuracy', 0.0)
                total_acc.append(acc)
                print(f"  {task_name}: {acc:.4f} ({metrics['correct']}/{metrics['total']})")
            
            if total_acc:
                avg_acc = sum(total_acc) / len(total_acc)
                print(f"\n  Average accuracy: {avg_acc:.4f}")

        # Summarize true/false tasks.
        tf_tasks = []
        for task_name, metrics in all_metrics.items():
            if metrics.get('task_type') == 'true_or_false':
                tf_tasks.append((task_name, metrics))
        
        if tf_tasks:
            print("\nTrue/false tasks:")
            for task_name, metrics in tf_tasks:
                print(f"  {task_name}:")
                print(json.dumps(metrics, ensure_ascii=False, indent=2))
        
        # Summarize visual grounding tasks.
        vg_tasks = []
        for task_name, metrics in all_metrics.items():
            if metrics.get('task_type') == 'visual_grounding':
                vg_tasks.append((task_name, metrics))
        
        if vg_tasks:
            print("\nVisual grounding tasks:")
            for task_name, metrics in vg_tasks:
                acc = metrics.get('accuracy', 0.0)
                avg_iou = metrics.get('average_iou', 0.0)
                iou_threshold = metrics.get('iou_threshold', 0.5)
                print(f"  {task_name}:")
                print(f"    Accuracy (IOU>{iou_threshold}): {acc:.4f} ({metrics['correct']}/{metrics['total']})")
                print(f"    Average IOU: {avg_iou:.4f}")
        
        # Summarize open-ended tasks.
        oe_tasks = []
        for task_name, metrics in all_metrics.items():
            if metrics.get('task_type') == 'open_ended':
                oe_tasks.append((task_name, metrics))
        
        if oe_tasks:
            print("\nOpen-ended tasks:")
            for task_name, metrics in oe_tasks:
                print(f"  {task_name}: {metrics['total']} samples")
        
        # Overall summary.
        total_samples = sum(m.get('total', 0) for m in all_metrics.values())
        print(f"\nTotal samples: {total_samples}")
        print(f"Total tasks: {len(all_metrics)}")


def main():
    parser = argparse.ArgumentParser(description='ERIBench evaluation metrics script')
    
    parser.add_argument('--predictions_path', type=str, required=True,
                       help='Path to a prediction file or directory')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (defaults to the predictions_path directory)')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Model name (default: inferred from the filename)')
    parser.add_argument('--image_dir', type=str, default=None,
                       help='Image directory (default: inferred from the filename)')
    parser.add_argument('--api_key', type=str, default=None,
                       help='API key')
    parser.add_argument('--base_url', type=str, default=None,
                       help='API base URL')
    parser.add_argument('--score_model', type=str, default=None,
                       help='Scoring model')
    args = parser.parse_args()
    
    # Validate arguments.
    if not os.path.exists(args.predictions_path):
        print(f"Error: path does not exist: {args.predictions_path}")
        return
    
    # If no output directory is provided, use the predictions_path directory.
    if args.output_dir is None:
        predictions_path = Path(args.predictions_path)
        if predictions_path.is_file():
            args.output_dir = predictions_path.parent / 'metrics'
        else:
            args.output_dir = predictions_path / 'metrics'
    
    # Create the metrics calculator and run evaluation.
    calculator = MetricsCalculator(
        predictions_path=args.predictions_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        image_dir=args.image_dir,
        api_key=args.api_key,
        base_url=args.base_url,
        score_model=args.score_model
    )
    
    calculator.calculate_all_metrics()


if __name__ == '__main__':
    main()
