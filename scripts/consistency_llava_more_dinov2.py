"""
Spatial Consistency Evaluation for LLaVA-MORE DINOv2 Model

This script evaluates spatial consistency of the aimagelab/LLaVA_MORE-gemma_2_9b-dinov2-finetuning
model using image transformations (flips, rotations) and checking if spatial reasoning
remains consistent.

Model: aimagelab/LLaVA_MORE-gemma_2_9b-dinov2-finetuning
- LLM backbone: gemma-2-9b-it
- Visual backbone: DINOv2

Based on the evaluation protocol from LLaVA-MORE repository:
https://github.com/aimagelab/LLaVA-MORE
"""

import sys
import os
import os.path as osp
from os.path import expanduser
from PIL import Image
import numpy as np
import re
from collections import defaultdict, Counter
import json
import torch
import glob
import argparse
from scipy.stats import wilcoxon
from itertools import combinations

# ============================================================================
# LLAVA-MORE MODEL LOADING
# ============================================================================

class LLaVAMoreDINOv2Model:
    """
    Wrapper for loading and running LLaVA-MORE DINOv2 model from HuggingFace.
    
    Model: aimagelab/LLaVA_MORE-gemma_2_9b-dinov2-finetuning
    """
    
    def __init__(self, model_name="aimagelab/LLaVA_MORE-gemma_2_9b-dinov2-finetuning", device='cuda'):
        """
        Initialize LLaVA-MORE DINOv2 model.
        
        Args:
            model_name: HuggingFace model name/path
            device: Device to run on ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        print(f"Loading LLaVA-MORE DINOv2 model: {model_name}")
        print(f"  Device: {self.device}")
        
        try:
            # Try loading with transformers LlavaForConditionalGeneration
            from transformers import LlavaForConditionalGeneration, AutoProcessor
            
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                device_map="auto" if self.device == 'cuda' else None,
                trust_remote_code=True,
            )
            self.model.eval()
            self.load_method = "transformers_llava"
            print(f"  ✓ Loaded with transformers LlavaForConditionalGeneration")
            
        except Exception as e:
            print(f"  Failed with LlavaForConditionalGeneration: {e}")
            try:
                # Try loading with AutoModelForVision2Seq
                from transformers import AutoModelForVision2Seq, AutoProcessor
                
                self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                    device_map="auto" if self.device == 'cuda' else None,
                    trust_remote_code=True,
                )
                self.model.eval()
                self.load_method = "transformers_vision2seq"
                print(f"  ✓ Loaded with transformers AutoModelForVision2Seq")
                
            except Exception as e2:
                print(f"  Failed with AutoModelForVision2Seq: {e2}")
                try:
                    # Try loading with LLaVA-MORE native loading
                    # This requires the LLaVA-MORE repo to be installed
                    from llava.model.builder import load_pretrained_model
                    from llava.mm_utils import get_model_name_from_path
                    
                    model_name_short = get_model_name_from_path(model_name)
                    self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                        model_path=model_name,
                        model_base=None,
                        model_name=model_name_short,
                        device_map="auto",
                        torch_dtype=torch.float16,
                    )
                    self.processor = None  # Use separate tokenizer and image_processor
                    self.model.eval()
                    self.load_method = "llava_native"
                    print(f"  ✓ Loaded with LLaVA native loader")
                    
                except Exception as e3:
                    print(f"  Failed with LLaVA native loader: {e3}")
                    raise RuntimeError(
                        f"Cannot load model {model_name}. "
                        f"Please ensure you have the correct dependencies installed:\n"
                        f"  pip install transformers>=4.36.0 accelerate\n"
                        f"Or clone and install LLaVA-MORE repo:\n"
                        f"  git clone https://github.com/aimagelab/LLaVA-MORE\n"
                        f"  cd LLaVA-MORE && pip install -e ."
                    )
    
    def generate(self, messages):
        """
        Generate response for given messages.
        
        Args:
            messages: List of dicts with 'type' and 'value' keys
                     e.g., [{'type': 'image', 'value': 'path.jpg'}, 
                            {'type': 'text', 'value': 'Question?'}]
        
        Returns:
            Generated text response
        """
        # Parse messages
        image_path = None
        text_prompt = None
        
        for msg in messages:
            if msg['type'] == 'image':
                image_path = msg['value']
            elif msg['type'] == 'text':
                text_prompt = msg['value']
        
        if image_path is None or text_prompt is None:
            return "Error: Missing image or text in messages"
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            if self.load_method in ["transformers_llava", "transformers_vision2seq"]:
                return self._generate_transformers(image, text_prompt)
            elif self.load_method == "llava_native":
                return self._generate_llava_native(image, text_prompt)
            else:
                return "Error: Unknown load method"
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error generating response: {str(e)}"
    
    def _generate_transformers(self, image, text_prompt):
        """Generate using transformers API."""
        # Format prompt for Gemma-based model
        # LLaVA-MORE uses a specific conversation format
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text_prompt}
                ]
            }
        ]
        
        # Apply chat template if available
        if hasattr(self.processor, 'apply_chat_template'):
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        else:
            # Fallback: simple prompt format
            prompt = f"<image>\nUser: {text_prompt}\nAssistant:"
        
        # Process inputs
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id if hasattr(self.processor, 'tokenizer') else None,
            )
        
        # Decode - only the generated part
        generated_ids = output_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response.strip()
    
    def _generate_llava_native(self, image, text_prompt):
        """Generate using LLaVA native API."""
        from llava.mm_utils import process_images, tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates
        
        # Process image
        image_tensor = process_images([image], self.image_processor, self.model.config)
        if isinstance(image_tensor, list):
            image_tensor = [img.to(self.model.device, dtype=torch.float16) for img in image_tensor]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
        
        # Build conversation - use gemma template
        conv_mode = "gemma_2"  # For Gemma-based models
        conv = conv_templates[conv_mode].copy()
        
        # Add image token to the question
        question = DEFAULT_IMAGE_TOKEN + '\n' + text_prompt
        
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        # Tokenize
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        input_ids = input_ids.unsqueeze(0).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image.size],
                do_sample=False,
                max_new_tokens=256,
                use_cache=True,
            )
        
        # Decode response
        response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        
        return response.strip()


# Global model cache
_model_cache = {}

def get_model(model_name="aimagelab/LLaVA_MORE-gemma_2_9b-dinov2-finetuning"):
    """Get or load the model."""
    if model_name not in _model_cache:
        _model_cache[model_name] = LLaVAMoreDINOv2Model(model_name)
    return _model_cache[model_name]


def CHECK(model_name, msg):
    """Run model inference on a message."""
    try:
        model = get_model(model_name)
        res = model.generate(msg)
        return (model_name, res)
    except Exception as e:
        return (model_name, f"Error: {str(e)}")


# ============================================================================
# TRANSFORMATIONS
# ============================================================================

def apply_transformation(image_path, transform_type):
    """
    Apply transformation to image and save to temp file.
    
    Args:
        image_path: Path to original image
        transform_type: One of ['original', 'hflip', 'vflip', 'rotate90', 'rotate180', 'rotate270']
    
    Returns:
        Path to transformed image
    """
    img = Image.open(image_path)
    
    if transform_type == 'original':
        return image_path
    elif transform_type == 'hflip':
        img_transformed = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif transform_type == 'vflip':
        img_transformed = img.transpose(Image.FLIP_TOP_BOTTOM)
    elif transform_type == 'rotate90':
        img_transformed = img.rotate(90, expand=True)
    elif transform_type == 'rotate180':
        img_transformed = img.rotate(180, expand=True)
    elif transform_type == 'rotate270':
        img_transformed = img.rotate(270, expand=True)
    else:
        raise ValueError(f"Unknown transform: {transform_type}")
    
    # Save to temp file with unique name based on original image
    base_name = os.path.basename(image_path)
    base_dir = os.path.dirname(image_path)
    name, ext = os.path.splitext(base_name)
    temp_path = os.path.join(base_dir, f'{name}_{transform_type}{ext}')
    img_transformed.save(temp_path)
    return temp_path


# ============================================================================
# ANSWER NORMALIZATION
# ============================================================================

def normalize_answer(raw_answer, question_type='spatial'):
    """
    Normalize model output to canonical form.
    
    Args:
        raw_answer: Raw string output from model
        question_type: Type of question ('spatial', 'count', 'multiple_choice')
    
    Returns:
        Normalized answer (int, str, or None if cannot parse)
    """
    if raw_answer is None:
        return None
    
    # Convert to lowercase and strip whitespace
    answer = str(raw_answer).lower().strip()
    
    if question_type == 'spatial':
        # Yes/No questions
        if re.search(r'\byes\b', answer):
            return 1
        elif re.search(r'\bno\b', answer):
            return 0
        
        # Look for directional answers
        if re.search(r'\bleft\b', answer):
            return 'left'
        elif re.search(r'\bright\b', answer):
            return 'right'
        elif re.search(r'\babove\b|\btop\b', answer):
            return 'above'
        elif re.search(r'\bbelow\b|\bbottom\b', answer):
            return 'below'
        
    elif question_type == 'count':
        # Extract first number from answer
        numbers = re.findall(r'\d+', answer)
        if numbers:
            return int(numbers[0])
    
    elif question_type == 'multiple_choice':
        # Extract A, B, C, D or 1, 2, 3, 4
        match = re.search(r'\b([a-d]|[1-4])\b', answer)
        if match:
            return match.group(1)
    
    # Return original if cannot normalize
    return answer


# ============================================================================
# QUESTION DEFINITION
# ============================================================================

class SpatialQuestion:
    """Defines a spatial question with expected answer transformations."""
    
    def __init__(self, text, question_type='spatial', spatial_axis='horizontal'):
        """
        Args:
            text: Question text
            question_type: Type for normalization
            spatial_axis: Which axis the question is about
                - 'horizontal': left/right questions
                - 'vertical': above/below questions
                - 'count': counting questions (answer should stay same)
                - 'yesno': yes/no questions about spatial relations
        """
        self.text = text
        self.question_type = question_type
        self.spatial_axis = spatial_axis
    
    def get_expected_answer(self, original_answer, transform_type):
        """
        Get expected answer after transformation.
        
        Transformation effects on spatial relations:
        - hflip: swaps left<->right, keeps above/below same
        - vflip: swaps above<->below, keeps left/right same
        - rotate90: left->above, right->below, above->right, below->left
        - rotate180: left<->right, above<->below
        - rotate270: left->below, right->above, above->left, below->right
        
        Args:
            original_answer: Normalized answer from original image
            transform_type: Type of transformation applied
        
        Returns:
            Expected normalized answer after transformation
        """
        if transform_type == 'original':
            return original_answer
        
        # For yes/no answers (1/0), determine based on axis and transform
        if original_answer in [0, 1]:
            return self._get_expected_yesno(original_answer, transform_type)
        
        # For directional answers
        if original_answer in ['left', 'right', 'above', 'below']:
            return self._get_expected_direction(original_answer, transform_type)
        
        # For count questions, answer should stay the same
        if self.spatial_axis == 'count':
            return original_answer
        
        return None  # Cannot determine expected answer
    
    def _get_expected_direction(self, original, transform):
        """Get expected directional answer after transform."""
        transform_maps = {
            'hflip': {'left': 'right', 'right': 'left', 'above': 'above', 'below': 'below'},
            'vflip': {'left': 'left', 'right': 'right', 'above': 'below', 'below': 'above'},
            'rotate90': {'left': 'above', 'right': 'below', 'above': 'right', 'below': 'left'},
            'rotate180': {'left': 'right', 'right': 'left', 'above': 'below', 'below': 'above'},
            'rotate270': {'left': 'below', 'right': 'above', 'above': 'left', 'below': 'right'},
        }
        
        if transform in transform_maps:
            return transform_maps[transform].get(original, None)
        return None
    
    def _get_expected_yesno(self, original, transform):
        """Get expected yes/no answer after transform."""
        if self.spatial_axis == 'horizontal':
            if transform == 'hflip':
                return 1 - original
            elif transform == 'vflip':
                return original
            elif transform == 'rotate180':
                return 1 - original
            elif transform in ['rotate90', 'rotate270']:
                return None
        
        elif self.spatial_axis == 'vertical':
            if transform == 'hflip':
                return original
            elif transform == 'vflip':
                return 1 - original
            elif transform == 'rotate180':
                return 1 - original
            elif transform in ['rotate90', 'rotate270']:
                return None
        
        return original


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_consistency(model_name, image_path, questions, transforms=['original', 'hflip']):
    """
    Evaluate model consistency across transformations.
    
    Args:
        model_name: Name of the model to test
        image_path: Path to test image
        questions: List of SpatialQuestion objects
        transforms: List of transformations to apply
    
    Returns:
        Dictionary with results and metrics
    """
    results = {
        'model': model_name,
        'questions': [],
        'metrics': {
            'total_questions': len(questions),
            'total_transforms': len(transforms) - 1,
            'consistent': 0,
            'inconsistent': 0,
            'undetermined': 0
        }
    }
    
    for question in questions:
        question_results = {
            'question': question.text,
            'type': question.question_type,
            'answers': {},
            'consistency': {}
        }
        
        # Get answer for original image
        original_path = image_path
        msg = [
            dict(type='image', value=original_path),
            dict(type='text', value=question.text)
        ]
        
        _, raw_answer = CHECK(model_name, msg)
        original_answer = normalize_answer(raw_answer, question.question_type)
        
        question_results['answers']['original'] = {
            'raw': raw_answer,
            'normalized': original_answer
        }
        
        # Test each transformation
        for transform in transforms:
            if transform == 'original':
                continue
            
            # Apply transformation
            transformed_path = apply_transformation(image_path, transform)
            
            msg = [
                dict(type='image', value=transformed_path),
                dict(type='text', value=question.text)
            ]
            
            _, raw_answer = CHECK(model_name, msg)
            transformed_answer = normalize_answer(raw_answer, question.question_type)
            
            question_results['answers'][transform] = {
                'raw': raw_answer,
                'normalized': transformed_answer
            }
            
            # Check consistency
            expected_answer = question.get_expected_answer(original_answer, transform)
            
            if expected_answer is None:
                consistency_status = 'undetermined'
                results['metrics']['undetermined'] += 1
            elif expected_answer == transformed_answer:
                consistency_status = 'consistent'
                results['metrics']['consistent'] += 1
            else:
                consistency_status = 'inconsistent'
                results['metrics']['inconsistent'] += 1
            
            question_results['consistency'][transform] = {
                'expected': expected_answer,
                'actual': transformed_answer,
                'status': consistency_status
            }
        
        results['questions'].append(question_results)
    
    # Calculate overall consistency rate
    total_checks = results['metrics']['consistent'] + results['metrics']['inconsistent']
    if total_checks > 0:
        results['metrics']['consistency_rate'] = results['metrics']['consistent'] / total_checks
    else:
        results['metrics']['consistency_rate'] = 0.0
    
    return results


def evaluate_model_on_images(model_name, image_paths, questions, transforms):
    """
    Evaluate a model across multiple images.
    
    Args:
        model_name: Name of model to test
        image_paths: List of image paths
        questions: List of SpatialQuestion objects
        transforms: List of transforms to apply
    
    Returns:
        Aggregated results dictionary
    """
    all_image_results = []
    total_consistent = 0
    total_inconsistent = 0
    total_undetermined = 0
    
    # Track per-transformation metrics
    transform_metrics = {t: {'consistent': 0, 'inconsistent': 0, 'undetermined': 0} 
                        for t in transforms if t != 'original'}
    
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"    ⚠️ Image not found: {image_path}")
            continue
            
        print(f"\n    Testing image: {os.path.basename(image_path)}")
        
        try:
            results = evaluate_consistency(
                model_name=model_name,
                image_path=image_path,
                questions=questions,
                transforms=transforms
            )
            results['image'] = image_path
            all_image_results.append(results)
            
            total_consistent += results['metrics']['consistent']
            total_inconsistent += results['metrics']['inconsistent']
            total_undetermined += results['metrics']['undetermined']
            
            # Aggregate per-transformation metrics
            for q_result in results.get('questions', []):
                for transform, cons_data in q_result.get('consistency', {}).items():
                    if transform in transform_metrics:
                        status = cons_data.get('status', 'undetermined')
                        if status == 'consistent':
                            transform_metrics[transform]['consistent'] += 1
                        elif status == 'inconsistent':
                            transform_metrics[transform]['inconsistent'] += 1
                        else:
                            transform_metrics[transform]['undetermined'] += 1
            
            print(f"      Consistency: {results['metrics']['consistency_rate']:.2%}")
            
        except Exception as e:
            print(f"      ❌ Error: {str(e)}")
            all_image_results.append({
                'image': image_path,
                'error': str(e)
            })
    
    # Calculate overall metrics
    total_checks = total_consistent + total_inconsistent
    overall_rate = total_consistent / total_checks if total_checks > 0 else 0.0
    
    # Calculate per-transformation rates
    for transform in transform_metrics:
        t_consistent = transform_metrics[transform]['consistent']
        t_inconsistent = transform_metrics[transform]['inconsistent']
        t_total = t_consistent + t_inconsistent
        transform_metrics[transform]['rate'] = t_consistent / t_total if t_total > 0 else 0.0
        transform_metrics[transform]['total'] = t_total
    
    return {
        'model': model_name,
        'num_images': len(image_paths),
        'image_results': all_image_results,
        'metrics': {
            'consistency_rate': overall_rate,
            'consistent': total_consistent,
            'inconsistent': total_inconsistent,
            'undetermined': total_undetermined,
            'total_checks': total_checks
        },
        'transform_metrics': transform_metrics
    }


def print_transform_breakdown(results):
    """Print consistency breakdown by transformation type with ASCII bar chart."""
    transform_metrics = results.get('transform_metrics', {})
    
    if not transform_metrics:
        return
    
    print(f"\n{'='*80}")
    print("CONSISTENCY BY TRANSFORMATION TYPE")
    print("="*80)
    print(f"\n{'Transform':<12} {'Consistent':<12} {'Inconsistent':<14} {'Rate':<10} {'Bar'}")
    print("-" * 70)
    
    for transform in ['hflip', 'vflip', 'rotate90', 'rotate180', 'rotate270']:
        if transform in transform_metrics:
            metrics = transform_metrics[transform]
            consistent = metrics['consistent']
            inconsistent = metrics['inconsistent']
            rate = metrics['rate']
            
            # Create ASCII bar (max 20 chars)
            bar_length = int(rate * 20)
            bar = '█' * bar_length + '░' * (20 - bar_length)
            
            print(f"{transform:<12} {consistent:<12} {inconsistent:<14} {rate*100:>6.2f}%   {bar}")
    
    print("-" * 70)
    
    # Overall
    overall_rate = results['metrics']['consistency_rate']
    overall_bar_length = int(overall_rate * 20)
    overall_bar = '█' * overall_bar_length + '░' * (20 - overall_bar_length)
    print(f"{'OVERALL':<12} {results['metrics']['consistent']:<12} {results['metrics']['inconsistent']:<14} {overall_rate*100:>6.2f}%   {overall_bar}")
    print()


def print_model_comparison_table(results):
    """Print a model comparison table (single model version)."""
    model_name = results.get('model', 'Unknown')
    transform_metrics = results.get('transform_metrics', {})
    
    # Shorten model name for display
    if '/' in model_name:
        display_name = model_name.split('/')[-1]
    else:
        display_name = model_name
    if len(display_name) > 30:
        display_name = display_name[:27] + '...'
    
    print(f"\n{'='*80}")
    print("MODEL PERFORMANCE SUMMARY")
    print("="*80)
    
    # Header
    transforms = ['hflip', 'vflip', 'rotate90', 'rotate180', 'rotate270']
    header = f"{'Model':<32}"
    for t in transforms:
        header += f" {t:<10}"
    header += f" {'OVERALL':<10}"
    print(header)
    print("-" * 95)
    
    # Data row
    row = f"{display_name:<32}"
    for t in transforms:
        if t in transform_metrics:
            rate = transform_metrics[t]['rate'] * 100
            row += f" {rate:>6.1f}%   "
        else:
            row += f" {'N/A':>6}    "
    
    overall = results['metrics']['consistency_rate'] * 100
    row += f" {overall:>6.1f}%   "
    print(row)
    print("-" * 95)
    print()


# ============================================================================
# JSON SERIALIZATION HELPERS
# ============================================================================

def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Evaluate LLaVA-MORE DINOv2 model for spatial consistency')
    parser.add_argument('--model', type=str, default='aimagelab/LLaVA_MORE-gemma_2_9b-dinov2-finetuning',
                        help='HuggingFace model name or path')
    parser.add_argument('--images_dir', type=str, default=None,
                        help='Directory containing test images')
    parser.add_argument('--image', type=str, default=None,
                        help='Single image path to test')
    parser.add_argument('--output', type=str, default='consistency_results_llava_more_dinov2.json',
                        help='Output JSON file path')
    parser.add_argument('--max_images', type=int, default=5,
                        help='Maximum number of images to test')
    args = parser.parse_args()
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    model_name = args.model
    
    # Find test images
    test_images = []
    if args.image:
        test_images = [args.image]
    elif args.images_dir:
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            test_images.extend(glob.glob(os.path.join(args.images_dir, ext)))
    else:
        # Default: look for images in current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            test_images.extend(glob.glob(os.path.join(current_dir, ext)))
    
    if not test_images:
        print("Error: No test images found. Please specify --images_dir or --image")
        print("\nUsage examples:")
        print(f"  python {os.path.basename(__file__)} --image /path/to/image.jpg")
        print(f"  python {os.path.basename(__file__)} --images_dir /path/to/images/")
        sys.exit(1)
    
    # Limit number of images
    test_images = sorted(test_images)[:args.max_images]
    
    # ========================================================================
    # SPATIAL QUESTIONS
    # ========================================================================
    
    questions = [
        # Left/Right questions (horizontal axis)
        SpatialQuestion(
            "Are the objects on the left or right side of the image? Answer with just 'left' or 'right'.",
            question_type='spatial',
            spatial_axis='horizontal'
        ),
        SpatialQuestion(
            "Is there anything on the left side of the image? Answer with just 'yes' or 'no'.",
            question_type='spatial',
            spatial_axis='horizontal'
        ),
        SpatialQuestion(
            "What objects are on the left side of the image?",
            question_type='spatial',
            spatial_axis='horizontal'
        ),
        
        # Above/Below questions (vertical axis)
        SpatialQuestion(
            "Is there anything at the top of the image? Answer with just 'yes' or 'no'.",
            question_type='spatial',
            spatial_axis='vertical'
        ),
        SpatialQuestion(
            "What is at the top of the image?",
            question_type='spatial',
            spatial_axis='vertical'
        ),
        
        # General spatial questions
        SpatialQuestion(
            "Describe the spatial arrangement of objects in this image.",
            question_type='spatial',
            spatial_axis='horizontal'
        ),
    ]
    
    # ========================================================================
    # TRANSFORMATIONS
    # ========================================================================
    
    transforms = [
        'original',
        'hflip',
        'vflip',
        'rotate90',
        'rotate180',
        'rotate270',
    ]
    
    # ========================================================================
    # RUN EVALUATION
    # ========================================================================
    
    print("\n" + "="*80)
    print("LLAVA-MORE DINOV2 SPATIAL CONSISTENCY EVALUATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Images: {len(test_images)}")
    print(f"  Questions: {len(questions)}")
    print(f"  Transforms: {len(transforms) - 1} (excluding original)")
    print(f"  Total inference calls: {len(test_images) * len(questions) * len(transforms)}")
    print(f"\nTest images:")
    for img in test_images:
        print(f"  - {os.path.basename(img)}")
    print(f"\nTransformations: {', '.join(transforms)}")
    print("="*80)
    
    try:
        results = evaluate_model_on_images(
            model_name=model_name,
            image_paths=test_images,
            questions=questions,
            transforms=transforms
        )
        
        # Print detailed breakdown by transformation
        print_transform_breakdown(results)
        
        # Print model comparison table
        print_model_comparison_table(results)
        
        # Print summary
        print(f"\n{'='*80}")
        print("SUMMARY")
        print("="*80)
        print(f"  Model: {model_name}")
        print(f"  Overall Consistency Rate: {results['metrics']['consistency_rate']:.2%}")
        print(f"  Consistent: {results['metrics']['consistent']}")
        print(f"  Inconsistent: {results['metrics']['inconsistent']}")
        print(f"  Undetermined: {results['metrics']['undetermined']}")
        print(f"  Total Checks: {results['metrics']['total_checks']}")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        results = {
            'model': model_name,
            'error': str(e),
            'metrics': {'consistency_rate': 0.0}
        }
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    output_file = args.output
    output_data = convert_numpy_types(results)
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print("="*80 + "\n")
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

