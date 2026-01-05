import sys
import os
import os.path as osp
from os.path import expanduser
from vlmeval.dataset import SUPPORTED_DATASETS
from vlmeval.config import *
from vlmeval.smp import *
from PIL import Image
import numpy as np
import re
from collections import defaultdict
import json
import torch
from scipy.stats import wilcoxon
from itertools import combinations

# Import for original LLaVA loading
try:
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates
    LLAVA_AVAILABLE = True
except ImportError:
    print("Warning: LLaVA library not found. Install with: pip install llava")
    LLAVA_AVAILABLE = False

PTH = osp.realpath(__file__)
IMAGE_PTH = '/home/ubuntu/VLMEvalKit/assets/022.jpg'

# Directory containing custom LLaVA models
CUSTOM_MODELS_DIR = expanduser('/home/ubuntu/custom_llava_models')

# ============================================================================
# CUSTOM MODEL LOADING (Using Original LLaVA Library)
# ============================================================================

class CustomLLaVAModel:
    """Wrapper for loading and running custom local LLaVA models using the original LLaVA library."""
    
    def __init__(self, model_path, device='cuda'):
        """
        Initialize custom LLaVA model using the LLaVA library's load_pretrained_model.
        
        Args:
            model_path: Path to the model directory
            device: Device to run on ('cuda' or 'cpu')
        """
        if not LLAVA_AVAILABLE:
            raise ImportError("LLaVA library required. Install with: pip install llava")
        
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        self.device = device
        
        print(f"Loading custom model: {self.model_name}")
        print(f"  Path: {model_path}")
        
        # Use LLaVA's native loading function
        # This properly handles the model architecture and config
        model_name = get_model_name_from_path(model_path)
        print(f"  Model name detected: {model_name}")
        
        try:
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path=model_path,
                model_base=None,  # For full fine-tuned models
                model_name=model_name,
                device_map="auto",
                torch_dtype=torch.float16,
            )
            print(f"  ✓ Loaded successfully using LLaVA library")
        except Exception as e:
            print(f"  Failed to load with model_base=None: {e}")
            # Try loading with base model (for LoRA fine-tuned models)
            try:
                print(f"  Trying with base model: liuhaotian/llava-v1.5-7b")
                self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                    model_path=model_path,
                    model_base="liuhaotian/llava-v1.5-7b",
                    model_name=model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
                print(f"  ✓ Loaded with base model")
            except Exception as e2:
                print(f"  Failed to load model: {e2}")
                raise RuntimeError(f"Cannot load model from {model_path}: {e2}")
        
        self.model.eval()
        
        # Determine conversation template
        if 'llama-2' in model_name.lower():
            self.conv_mode = "llava_llama_2"
        elif 'mistral' in model_name.lower():
            self.conv_mode = "mistral_instruct"
        elif 'v1.6' in model_name.lower() or 'v1.5' in model_name.lower():
            self.conv_mode = "llava_v1"
        else:
            self.conv_mode = "llava_v1"
        
        print(f"  Using conversation mode: {self.conv_mode}")
    
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
            # Load and process image
            image = Image.open(image_path).convert('RGB')
            image_tensor = process_images([image], self.image_processor, self.model.config)
            
            if isinstance(image_tensor, list):
                image_tensor = [img.to(self.model.device, dtype=torch.float16) for img in image_tensor]
            else:
                image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
            
            # Build conversation
            conv = conv_templates[self.conv_mode].copy()
            
            # Add image token to the question
            if self.model.config.mm_use_im_start_end:
                question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + text_prompt
            else:
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
            
            # Extract only the assistant's response
            response = response.strip()
            if conv.sep2 and conv.sep2 in response:
                response = response.split(conv.sep2)[-1].strip()
            
            return response
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error generating response: {str(e)}"


# Cache for loaded custom models
_custom_model_cache = {}

def get_custom_model(model_name):
    """Get or load a custom model by name."""
    if model_name not in _custom_model_cache:
        model_path = os.path.join(CUSTOM_MODELS_DIR, model_name)
        if os.path.exists(model_path):
            _custom_model_cache[model_name] = CustomLLaVAModel(model_path)
        else:
            raise FileNotFoundError(f"Custom model not found: {model_path}")
    return _custom_model_cache[model_name]


# List of custom model names
CUSTOM_MODEL_NAMES = [
    'llava-v1.5-7b-finetune-aimv2',
    'llava-v1.5-7b-finetune-mrope-aimv2',
    'llava-v1.5-7b-finetune-mrope-clip',
    'llava-v1.5-7b-finetune-mrope-siglip-base-patch16-256',
    'llava-v1.5-7b-finetune-mrope-siglip2-base-patch16-256',
    'llava-v1.5-7b-finetune-siglip',
    'llava-v1.5-7b-finetune-siglip2',
]


def CHECK(val, msg):
    """Run model inference on a message."""
    # First check if it's a custom model
    if val in CUSTOM_MODEL_NAMES:
        try:
            model = get_custom_model(val)
            res = model.generate(msg)
            return (val, res)
        except Exception as e:
            return (val, f"Error: {str(e)}")
    
    # Check VLMEval supported models
    if val in supported_VLM:
        model = supported_VLM[val]()
        res = model.generate(msg)
        return (val, res)
    elif val in models:
        results = []
        model_list = models[val]
        for m in model_list:
            results.append(CHECK(m, msg))
        return results
    else:
        return (val, "Model not found")


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
        # Look for yes/no in the answer
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
        # Mapping: transform -> {original -> expected}
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
        # For horizontal axis questions (left/right)
        if self.spatial_axis == 'horizontal':
            if transform == 'hflip':
                return 1 - original  # Flip the answer
            elif transform == 'vflip':
                return original  # No change
            elif transform == 'rotate180':
                return 1 - original  # Flip (left becomes right)
            elif transform in ['rotate90', 'rotate270']:
                return None  # Answer becomes about different axis
        
        # For vertical axis questions (above/below)
        elif self.spatial_axis == 'vertical':
            if transform == 'hflip':
                return original  # No change
            elif transform == 'vflip':
                return 1 - original  # Flip the answer
            elif transform == 'rotate180':
                return 1 - original  # Flip (above becomes below)
            elif transform in ['rotate90', 'rotate270']:
                return None  # Answer becomes about different axis
        
        return original  # Default: no change


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
            'total_transforms': len(transforms) - 1,  # Exclude original
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


# ============================================================================
# MULTI-IMAGE EVALUATION
# ============================================================================

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
        }
    }


# ============================================================================
# JSON SERIALIZATION HELPERS
# ============================================================================

def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    """
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
# ARCHITECTURE EXTRACTION
# ============================================================================

def extract_architecture(model_name):
    """
    Extract base architecture from model name.
    
    Examples:
        'llava-v1.5-7b-finetune-siglip' -> 'SigLIP'
        'llava-v1.5-7b-finetune-mrope-clip' -> 'CLIP'
        'llava-v1.5-7b-finetune-aimv2' -> 'AIMv2'
    """
    model_lower = model_name.lower()
    
    # Check for architecture patterns
    if 'siglip' in model_lower or 'siglip2' in model_lower:
        return 'SigLIP'
    elif 'clip' in model_lower:
        return 'CLIP'
    elif 'aimv2' in model_lower or 'aim-v2' in model_lower:
        return 'AIMv2'
    elif 'llava_v1.5_7b' in model_lower or 'llava-v1.5-7b' in model_lower:
        # Check if it's a base model without finetuning
        if 'finetune' not in model_lower:
            return 'Base'
        else:
            # Try to extract from finetune name
            if 'siglip' in model_lower:
                return 'SigLIP'
            elif 'clip' in model_lower:
                return 'CLIP'
            elif 'aimv2' in model_lower:
                return 'AIMv2'
            else:
                return 'Other'
    else:
        return 'Other'


def group_models_by_architecture(model_names):
    """Group model names by their base architecture."""
    arch_groups = defaultdict(list)
    for model_name in model_names:
        arch = extract_architecture(model_name)
        arch_groups[arch].append(model_name)
    return dict(arch_groups)


# ============================================================================
# PAIRED COMPARISON ANALYSIS
# ============================================================================

def perform_paired_comparisons(all_results, transforms):
    """
    Perform paired comparison analysis between all model pairs.
    
    For each pair of models, builds consistency vectors per image/question/transform
    and runs Wilcoxon signed-rank test with effect size calculation.
    
    Args:
        all_results: List of model evaluation results
        transforms: List of transformation types (excluding 'original')
    
    Returns:
        Dictionary with paired comparison results
    """
    # Filter out models with errors
    valid_results = {r['model']: r for r in all_results if 'error' not in r and 'image_results' in r}
    model_names = list(valid_results.keys())
    
    if len(model_names) < 2:
        print("  ⚠️ Need at least 2 valid models for paired comparison")
        return {}
    
    # Generate all pairs
    model_pairs = list(combinations(model_names, 2))
    paired_comparisons = {}
    
    print("\n" + "="*80)
    print("PAIRED MODEL COMPARISONS (Wilcoxon Signed-Rank Test)")
    print("="*80)
    
    transform_list = [t for t in transforms if t != 'original']
    
    for model_A, model_B in model_pairs:
        print(f"\n🔬 Comparing: {model_A} vs {model_B}")
        
        # Build consistency vectors per image/question/transform
        # Structure: {image: {question_idx: {transform: (cons_A, cons_B)}}}
        consistency_data = {}
        
        result_A = valid_results[model_A]
        result_B = valid_results[model_B]
        
        # Build index of image results by image path for both models
        img_results_A = {os.path.basename(r.get('image', '')): r for r in result_A['image_results'] if 'image' in r}
        img_results_B = {os.path.basename(r.get('image', '')): r for r in result_B['image_results'] if 'image' in r}
        
        # Find common images
        common_images = set(img_results_A.keys()) & set(img_results_B.keys())
        
        # Extract consistency vectors for common images
        for img_name in common_images:
            img_result_A = img_results_A[img_name]
            img_result_B = img_results_B[img_name]
            
            if 'questions' not in img_result_A or 'questions' not in img_result_B:
                continue
            
            if img_name not in consistency_data:
                consistency_data[img_name] = {}
            
            # Match questions by index (assuming same order and same questions)
            num_questions = min(len(img_result_A['questions']), len(img_result_B['questions']))
            for q_idx in range(num_questions):
                q_result_A = img_result_A['questions'][q_idx]
                q_result_B = img_result_B['questions'][q_idx]
                if q_idx not in consistency_data[img_name]:
                    consistency_data[img_name][q_idx] = {}
                
                # Get consistency for each transform
                for transform in transform_list:
                    cons_A = q_result_A.get('consistency', {}).get(transform, {})
                    cons_B = q_result_B.get('consistency', {}).get(transform, {})
                    
                    # Extract consistency status (1 if consistent, 0 if inconsistent, None if undetermined)
                    status_A = cons_A.get('status', '')
                    status_B = cons_B.get('status', '')
                    
                    cons_val_A = None
                    cons_val_B = None
                    
                    if status_A == 'consistent':
                        cons_val_A = 1
                    elif status_A == 'inconsistent':
                        cons_val_A = 0
                    # else: None (undetermined, skip)
                    
                    if status_B == 'consistent':
                        cons_val_B = 1
                    elif status_B == 'inconsistent':
                        cons_val_B = 0
                    # else: None (undetermined, skip)
                    
                    # Only include pairs where both have valid consistency values
                    if cons_val_A is not None and cons_val_B is not None:
                        consistency_data[img_name][q_idx][transform] = (cons_val_A, cons_val_B)
        
        # Build vectors for Wilcoxon test
        # Flatten all image/question/transform combinations into single vectors
        cons_model_A = []
        cons_model_B = []
        
        for img_name in sorted(consistency_data.keys()):
            for q_idx in sorted(consistency_data[img_name].keys()):
                for transform in transform_list:
                    if transform in consistency_data[img_name][q_idx]:
                        cons_A, cons_B = consistency_data[img_name][q_idx][transform]
                        cons_model_A.append(cons_A)
                        cons_model_B.append(cons_B)
        
        # Perform statistical tests
        comparison_result = {
            'model_A': model_A,
            'model_B': model_B,
            'n_pairs': len(cons_model_A),
            'mean_A': None,
            'mean_B': None,
            'delta': None,
            'wilcoxon_statistic': None,
            'wilcoxon_pvalue': None,
            'significant': None
        }
        
        if len(cons_model_A) > 0:
            cons_model_A = np.array(cons_model_A)
            cons_model_B = np.array(cons_model_B)
            
            # Compute means
            mean_A = np.mean(cons_model_A)
            mean_B = np.mean(cons_model_B)
            delta = mean_B - mean_A
            
            comparison_result['mean_A'] = float(mean_A)
            comparison_result['mean_B'] = float(mean_B)
            comparison_result['delta'] = float(delta)
            
            # Run Wilcoxon signed-rank test
            try:
                stat, p_value = wilcoxon(cons_model_A, cons_model_B, alternative='two-sided')
                comparison_result['wilcoxon_statistic'] = float(stat)
                comparison_result['wilcoxon_pvalue'] = float(p_value)
                comparison_result['significant'] = bool(p_value < 0.05)  # Convert numpy bool_ to Python bool
                
                print(f"  N pairs: {len(cons_model_A)}")
                print(f"  Mean consistency - {model_A}: {mean_A:.3f}")
                print(f"  Mean consistency - {model_B}: {mean_B:.3f}")
                print(f"  Effect size (Δ): {delta:.3f} ({'B better' if delta > 0 else 'A better' if delta < 0 else 'equal'})")
                print(f"  Wilcoxon statistic: {stat:.3f}")
                print(f"  Wilcoxon p-value: {p_value:.6f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''}")
                
            except Exception as e:
                print(f"  ⚠️ Error in Wilcoxon test: {str(e)}")
                comparison_result['error'] = str(e)
        else:
            print(f"  ⚠️ No valid pairs found for comparison")
            comparison_result['error'] = 'No valid pairs'
        
        # Store comparison result
        pair_key = f"{model_A}__vs__{model_B}"
        paired_comparisons[pair_key] = comparison_result
    
    return paired_comparisons


# ============================================================================
# ARCHITECTURE GROUP COMPARISONS AND REPORT GENERATION
# ============================================================================

def generate_architecture_comparison_reports(paired_comparisons, model_list):
    """
    Aggregate paired comparisons by architecture groups and generate natural language reports.
    
    Returns:
        Dictionary with architecture comparisons and list of natural language report sentences
    """
    # Group models by architecture
    arch_groups = group_models_by_architecture(model_list)
    
    # Filter to architectures with at least one model
    arch_groups = {arch: models for arch, models in arch_groups.items() if len(models) > 0}
    
    if len(arch_groups) < 2:
        return {'architecture_comparisons': {}, 'reports': []}
    
    # Build architecture comparison data
    arch_comparisons = {}
    reports = []
    
    # Generate all architecture pairs
    arch_pairs = list(combinations(sorted(arch_groups.keys()), 2))
    
    for arch_A, arch_B in arch_pairs:
        models_A = arch_groups[arch_A]
        models_B = arch_groups[arch_B]
        
        # Collect all individual model pair comparisons between these architectures
        all_deltas = []
        all_p_values = []
        all_n_pairs = []
        valid_comparisons = []
        
        for model_A in models_A:
            for model_B in models_B:
                pair_key = f"{model_A}__vs__{model_B}"
                reverse_key = f"{model_B}__vs__{model_A}"
                
                comp = paired_comparisons.get(pair_key)
                if comp and 'error' not in comp:
                    # Delta is already computed as model_B - model_A, which is arch_B - arch_A
                    valid_comparisons.append(comp)
                    if comp['delta'] is not None:
                        all_deltas.append(comp['delta'])
                    if comp['wilcoxon_pvalue'] is not None:
                        all_p_values.append(comp['wilcoxon_pvalue'])
                    if comp['n_pairs'] is not None:
                        all_n_pairs.append(comp['n_pairs'])
                else:
                    # Try reverse key and flip delta if found
                    comp = paired_comparisons.get(reverse_key)
                    if comp and 'error' not in comp:
                        # For reverse comparison, delta is model_A - model_B, so flip it
                        valid_comparisons.append(comp)
                        if comp['delta'] is not None:
                            all_deltas.append(-comp['delta'])  # Flip delta
                        if comp['wilcoxon_pvalue'] is not None:
                            all_p_values.append(comp['wilcoxon_pvalue'])
                        if comp['n_pairs'] is not None:
                            all_n_pairs.append(comp['n_pairs'])
        
        if len(valid_comparisons) == 0:
            continue
        
        # Aggregate statistics
        # Use meta-analysis approach: combine p-values using Fisher's method or average
        # For simplicity, we'll use the median/mean of individual comparisons
        mean_delta = np.mean(all_deltas) if all_deltas else None
        median_p_value = np.median(all_p_values) if all_p_values else None
        total_n = sum(all_n_pairs) if all_n_pairs else None
        
        # Determine which architecture performs better
        better_arch = None
        if mean_delta is not None:
            if mean_delta > 0:
                better_arch = arch_B
            elif mean_delta < 0:
                better_arch = arch_A
        
        # Generate natural language report
        report = None
        if mean_delta is not None and median_p_value is not None:
            # Format p-value
            if median_p_value < 0.001:
                p_str = "p < 0.001"
            elif median_p_value < 0.01:
                p_str = "p < 0.01"
            elif median_p_value < 0.05:
                p_str = "p < 0.05"
            else:
                p_str = f"p = {median_p_value:.3f}"
            
            # Format delta with sign
            delta_str = f"{mean_delta:+.2f}"
            
            # Generate sentence
            if median_p_value < 0.05:
                # Significant difference
                if better_arch == arch_B:
                    report = f"{arch_B}-based models exhibit significantly {'higher' if mean_delta > 0 else 'lower'} spatial consistency than {arch_A}-based models (Δ = {delta_str}, {p_str}, Wilcoxon signed-rank)."
                elif better_arch == arch_A:
                    report = f"{arch_A}-based models exhibit significantly {'higher' if mean_delta < 0 else 'lower'} spatial consistency than {arch_B}-based models (Δ = {-mean_delta:+.2f}, {p_str}, Wilcoxon signed-rank)."
                else:
                    report = f"{arch_A}-based and {arch_B}-based models show significantly different spatial consistency (Δ = {delta_str}, {p_str}, Wilcoxon signed-rank)."
                reports.append(report)
            else:
                # Non-significant difference - still generate report for the dictionary
                if better_arch == arch_B:
                    report = f"{arch_B}-based models show {'higher' if mean_delta > 0 else 'lower'} spatial consistency than {arch_A}-based models, though not statistically significant (Δ = {delta_str}, {p_str}, Wilcoxon signed-rank)."
                elif better_arch == arch_A:
                    report = f"{arch_A}-based models show {'higher' if mean_delta < 0 else 'lower'} spatial consistency than {arch_B}-based models, though not statistically significant (Δ = {-mean_delta:+.2f}, {p_str}, Wilcoxon signed-rank)."
                else:
                    report = f"{arch_A}-based and {arch_B}-based models show similar spatial consistency (Δ = {delta_str}, {p_str}, Wilcoxon signed-rank)."
            # Note: We only include significant findings in the main reports list
            
            arch_comparisons[f"{arch_A}_vs_{arch_B}"] = {
                'architecture_A': arch_A,
                'architecture_B': arch_B,
                'models_A': models_A,
                'models_B': models_B,
                'n_comparisons': len(valid_comparisons),
                'mean_delta': float(mean_delta) if mean_delta is not None else None,
                'median_p_value': float(median_p_value) if median_p_value is not None else None,
                'total_n_pairs': int(total_n) if total_n is not None else None,
                'better_architecture': better_arch,
                'report': report
            }
    
    return {
        'architecture_comparisons': arch_comparisons,
        'reports': reports,
        'architecture_groups': {arch: models for arch, models in arch_groups.items()}
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import glob
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    # Test images - add your image paths here
    # Will search for common image files in the assets directory
    ASSETS_DIR = '/home/ubuntu/VLMEvalKit/assets'
    
    # Find all jpg/png images in assets directory
    test_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        test_images.extend(glob.glob(os.path.join(ASSETS_DIR, ext)))
    
    # If no images found, fall back to the default
    if not test_images:
        test_images = [IMAGE_PTH]
    
    # Limit to first 5 images to keep runtime reasonable
    MAX_IMAGES = 5
    test_images = sorted(test_images)[:MAX_IMAGES]
    
    # ========================================================================
    # SPATIAL QUESTIONS
    # ========================================================================
    # Define questions for spatial consistency testing
    # Each question targets left/right or above/below relations
    
    questions = [
        # Left/Right questions (horizontal axis)
        SpatialQuestion(
            "Are the chopsticks to the left or right of the bowl? Answer with just 'left' or 'right'.",
            question_type='spatial',
            spatial_axis='horizontal'
        ),
        SpatialQuestion(
            "Is the spoon to the left of the bowl? Answer with just 'yes' or 'no'.",
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
            "Is the bowl above the chopsticks? Answer with just 'yes' or 'no'.",
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
    # MODELS TO TEST
    # ========================================================================
    
    # Original VLMEval models
    vlmeval_models = ["llava_v1.5_7b"]
    
    # Custom local models from ~/custom_llava_models
    custom_models = [
        'llava-v1.5-7b-finetune-aimv2',
        'llava-v1.5-7b-finetune-mrope-aimv2',
        'llava-v1.5-7b-finetune-mrope-clip',
        'llava-v1.5-7b-finetune-mrope-siglip-base-patch16-256',
        'llava-v1.5-7b-finetune-mrope-siglip2-base-patch16-256',
        'llava-v1.5-7b-finetune-siglip',
        'llava-v1.5-7b-finetune-siglip2',
    ]
    
    # Combine all models
    model_list = vlmeval_models + custom_models
    
    # ========================================================================
    # TRANSFORMATIONS
    # ========================================================================
    
    # All transformations for comprehensive testing
    transforms = [
        'original',   # No transformation (baseline)
        'hflip',      # Horizontal flip (left<->right swap)
        'vflip',      # Vertical flip (top<->bottom swap)
        'rotate90',   # 90° counterclockwise rotation
        'rotate180',  # 180° rotation
        'rotate270',  # 270° counterclockwise rotation (same as 90° clockwise)
    ]
    
    # ========================================================================
    # RUN EVALUATION
    # ========================================================================
    
    all_results = []
    
    print("\n" + "="*80)
    print("VLM SPATIAL CONSISTENCY TESTING - COMPREHENSIVE EVALUATION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Models: {len(model_list)} ({len(vlmeval_models)} VLMEval + {len(custom_models)} custom)")
    print(f"  Images: {len(test_images)}")
    print(f"  Questions: {len(questions)}")
    print(f"  Transforms: {len(transforms) - 1} (excluding original)")
    print(f"  Total inference calls per model: {len(test_images) * len(questions) * len(transforms)}")
    print(f"\nTest images:")
    for img in test_images:
        print(f"  - {os.path.basename(img)}")
    print(f"\nTransformations: {', '.join(transforms)}")
    print("="*80)
    
    for model_name in model_list:
        print(f"\n{'='*80}")
        print(f"TESTING MODEL: {model_name}")
        print("="*80)
        
        try:
            results = evaluate_model_on_images(
                model_name=model_name,
                image_paths=test_images,
                questions=questions,
                transforms=transforms
            )
            
            all_results.append(results)
            
            # Print summary for this model
            print(f"\n  📊 Model Summary:")
            print(f"     Overall Consistency Rate: {results['metrics']['consistency_rate']:.2%}")
            print(f"     Consistent: {results['metrics']['consistent']}")
            print(f"     Inconsistent: {results['metrics']['inconsistent']}")
            print(f"     Undetermined: {results['metrics']['undetermined']}")
            
        except Exception as e:
            print(f"  ❌ Error testing {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'model': model_name,
                'error': str(e),
                'metrics': {'consistency_rate': 0.0}
            })
        
        # Clear GPU memory and model cache between models
        if model_name in _custom_model_cache:
            del _custom_model_cache[model_name]
        
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    # ========================================================================
    # PAIRED COMPARISON ANALYSIS
    # ========================================================================
    
    paired_comparisons = perform_paired_comparisons(all_results, transforms)
    
    # ========================================================================
    # ARCHITECTURE GROUP COMPARISONS AND REPORT GENERATION
    # ========================================================================
    
    arch_comparison_results = generate_architecture_comparison_reports(paired_comparisons, model_list)
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    output_file = '/home/ubuntu/VLMEvalKit/consistency_results_comprehensive.json'
    
    # Combine all results with paired comparisons and architecture reports
    output_data = {
        'model_results': all_results,
        'paired_comparisons': paired_comparisons,
        'architecture_comparisons': arch_comparison_results['architecture_comparisons'],
        'architecture_reports': arch_comparison_results['reports'],
        'architecture_groups': arch_comparison_results['architecture_groups']
    }
    
    # Convert numpy types to native Python types for JSON serialization
    output_data = convert_numpy_types(output_data)
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # ========================================================================
    # PRINT SUMMARY TABLE
    # ========================================================================
    
    print("\n" + "="*80)
    print("SUMMARY TABLE - OVERALL CONSISTENCY RATES")
    print("="*80)
    print(f"{'Model':<55} {'Consistency':<15} {'Checks':<10}")
    print("-" * 80)
    
    for result in all_results:
        model_name = result['model']
        if 'error' in result and 'metrics' not in result:
            rate = 'ERROR'
            checks = '-'
        else:
            rate = f"{result['metrics']['consistency_rate']:.2%}"
            checks = str(result['metrics'].get('total_checks', 0))
        print(f"{model_name:<55} {rate:<15} {checks:<10}")
    
    # ========================================================================
    # PRINT PER-TRANSFORM BREAKDOWN
    # ========================================================================
    
    print("\n" + "="*80)
    print("CONSISTENCY BY TRANSFORMATION TYPE")
    print("="*80)
    
    # Calculate per-transform stats
    transform_stats = {t: {'consistent': 0, 'inconsistent': 0} for t in transforms if t != 'original'}
    
    for result in all_results:
        if 'image_results' not in result:
            continue
        for img_result in result['image_results']:
            if 'questions' not in img_result:
                continue
            for q_result in img_result['questions']:
                for t, cons in q_result.get('consistency', {}).items():
                    if cons['status'] == 'consistent':
                        transform_stats[t]['consistent'] += 1
                    elif cons['status'] == 'inconsistent':
                        transform_stats[t]['inconsistent'] += 1
    
    print(f"{'Transform':<15} {'Consistent':<12} {'Inconsistent':<12} {'Rate':<10}")
    print("-" * 50)
    for t, stats in transform_stats.items():
        total = stats['consistent'] + stats['inconsistent']
        rate = stats['consistent'] / total if total > 0 else 0
        print(f"{t:<15} {stats['consistent']:<12} {stats['inconsistent']:<12} {rate:.2%}")
    
    # ========================================================================
    # PRINT PER-MODEL PER-TRANSFORM BREAKDOWN
    # ========================================================================
    
    print("\n" + "="*80)
    print("CONSISTENCY BY MODEL AND TRANSFORMATION")
    print("="*80)
    
    # Build per-model per-transform stats
    model_transform_stats = {}  # {model: {transform: {consistent, inconsistent}}}
    
    for result in all_results:
        model_name = result['model']
        if model_name not in model_transform_stats:
            model_transform_stats[model_name] = {t: {'consistent': 0, 'inconsistent': 0} for t in transforms if t != 'original'}
        
        if 'image_results' not in result:
            continue
        
        for img_result in result['image_results']:
            if 'questions' not in img_result:
                continue
            for q_result in img_result['questions']:
                for t, cons in q_result.get('consistency', {}).items():
                    if t not in model_transform_stats[model_name]:
                        model_transform_stats[model_name][t] = {'consistent': 0, 'inconsistent': 0}
                    if cons['status'] == 'consistent':
                        model_transform_stats[model_name][t]['consistent'] += 1
                    elif cons['status'] == 'inconsistent':
                        model_transform_stats[model_name][t]['inconsistent'] += 1
    
    # Print header
    transform_list = [t for t in transforms if t != 'original']
    header = f"{'Model':<40}"
    for t in transform_list:
        header += f" {t:<12}"
    header += f" {'OVERALL':<12}"
    print(header)
    print("-" * len(header))
    
    # Print each model's per-transform stats
    for model_name in model_list:
        if model_name not in model_transform_stats:
            continue
        
        row = f"{model_name[:38]:<40}"
        total_consistent = 0
        total_checks = 0
        
        for t in transform_list:
            stats = model_transform_stats[model_name].get(t, {'consistent': 0, 'inconsistent': 0})
            total = stats['consistent'] + stats['inconsistent']
            rate = stats['consistent'] / total if total > 0 else 0
            row += f" {rate*100:>5.1f}%      "
            total_consistent += stats['consistent']
            total_checks += total
        
        # Overall rate for this model
        overall_rate = total_consistent / total_checks if total_checks > 0 else 0
        row += f" {overall_rate*100:>5.1f}%"
        print(row)
    
    # ========================================================================
    # PRINT PER-MODEL PER-TRANSFORM DETAILED TABLE
    # ========================================================================
    
    print("\n" + "="*80)
    print("DETAILED MODEL x TRANSFORM MATRIX (with counts)")
    print("="*80)
    
    for model_name in model_list:
        if model_name not in model_transform_stats:
            continue
        
        print(f"\n🤖 {model_name}")
        print("-" * 70)
        print(f"  {'Transform':<15} {'Consistent':<12} {'Inconsistent':<12} {'Rate':<12} {'Bar':<20}")
        print("  " + "-" * 66)
        
        model_total_consistent = 0
        model_total_checks = 0
        
        for t in transform_list:
            stats = model_transform_stats[model_name].get(t, {'consistent': 0, 'inconsistent': 0})
            total = stats['consistent'] + stats['inconsistent']
            rate = stats['consistent'] / total if total > 0 else 0
            bar = "█" * int(rate * 15) + "░" * (15 - int(rate * 15))
            print(f"  {t:<15} {stats['consistent']:<12} {stats['inconsistent']:<12} {rate:.2%}        {bar}")
            
            model_total_consistent += stats['consistent']
            model_total_checks += total
        
        # Model summary
        overall_rate = model_total_consistent / model_total_checks if model_total_checks > 0 else 0
        print("  " + "-" * 66)
        print(f"  {'TOTAL':<15} {model_total_consistent:<12} {model_total_checks - model_total_consistent:<12} {overall_rate:.2%}")
    
    # ========================================================================
    # PRINT TRANSFORM HEATMAP (Model x Transform)
    # ========================================================================
    
    print("\n" + "="*80)
    print("TRANSFORM CONSISTENCY HEATMAP (Model x Transform)")
    print("="*80)
    print("\nLegend: ██ >80% | ▓▓ 60-80% | ▒▒ 40-60% | ░░ 20-40% | ·· <20%")
    print()
    
    # Header
    header = f"{'Model':<42}"
    for t in transform_list:
        header += f" {t:<10}"
    print(header)
    print("-" * len(header))
    
    for model_name in model_list:
        if model_name not in model_transform_stats:
            continue
        
        row = f"{model_name[:40]:<42}"
        for t in transform_list:
            stats = model_transform_stats[model_name].get(t, {'consistent': 0, 'inconsistent': 0})
            total = stats['consistent'] + stats['inconsistent']
            rate = stats['consistent'] / total if total > 0 else 0
            
            if rate >= 0.8:
                symbol = "██"
            elif rate >= 0.6:
                symbol = "▓▓"
            elif rate >= 0.4:
                symbol = "▒▒"
            elif rate >= 0.2:
                symbol = "░░"
            else:
                symbol = "··"
            row += f" {symbol}{rate*100:>5.1f}%  "
        print(row)
    
    # ========================================================================
    # PRINT DETAILED PER-IMAGE BREAKDOWN
    # ========================================================================
    
    print("\n" + "="*80)
    print("DETAILED PER-IMAGE BREAKDOWN")
    print("="*80)
    
    # Collect all unique images
    all_images = set()
    for result in all_results:
        if 'image_results' in result:
            for img_result in result['image_results']:
                if 'image' in img_result:
                    all_images.add(os.path.basename(img_result['image']))
    all_images = sorted(all_images)
    
    # Build per-image stats for each model
    image_model_stats = {}  # {image: {model: {consistent, inconsistent, rate}}}
    
    for result in all_results:
        model_name = result['model']
        if 'image_results' not in result:
            continue
        
        for img_result in result['image_results']:
            if 'image' not in img_result or 'metrics' not in img_result:
                continue
            
            img_name = os.path.basename(img_result['image'])
            if img_name not in image_model_stats:
                image_model_stats[img_name] = {}
            
            metrics = img_result['metrics']
            total = metrics['consistent'] + metrics['inconsistent']
            rate = metrics['consistent'] / total if total > 0 else 0.0
            
            image_model_stats[img_name][model_name] = {
                'consistent': metrics['consistent'],
                'inconsistent': metrics['inconsistent'],
                'undetermined': metrics['undetermined'],
                'total': total,
                'rate': rate
            }
    
    # Print per-image table
    for img_name in sorted(image_model_stats.keys()):
        print(f"\n📷 Image: {img_name}")
        print("-" * 80)
        print(f"  {'Model':<53} {'Rate':<10} {'✓':<6} {'✗':<6} {'?':<6}")
        print("  " + "-" * 76)
        
        model_stats = image_model_stats[img_name]
        for model_name in model_list:
            if model_name in model_stats:
                stats = model_stats[model_name]
                print(f"  {model_name:<53} {stats['rate']:.2%}     {stats['consistent']:<6} {stats['inconsistent']:<6} {stats['undetermined']:<6}")
            else:
                print(f"  {model_name:<53} {'N/A':<10}")
    
    # ========================================================================
    # PRINT IMAGE RANKING (BEST TO WORST CONSISTENCY)
    # ========================================================================
    
    print("\n" + "="*80)
    print("IMAGE RANKING BY AVERAGE CONSISTENCY (across all models)")
    print("="*80)
    
    image_avg_rates = {}
    for img_name, model_stats in image_model_stats.items():
        rates = [s['rate'] for s in model_stats.values() if s['total'] > 0]
        if rates:
            image_avg_rates[img_name] = sum(rates) / len(rates)
    
    print(f"\n{'Rank':<6} {'Image':<40} {'Avg Consistency':<15}")
    print("-" * 65)
    
    for rank, (img_name, avg_rate) in enumerate(sorted(image_avg_rates.items(), key=lambda x: -x[1]), 1):
        print(f"{rank:<6} {img_name:<40} {avg_rate:.2%}")
    
    # ========================================================================
    # PRINT MODEL RANKING PER IMAGE
    # ========================================================================
    
    print("\n" + "="*80)
    print("MODEL RANKING BY IMAGE")
    print("="*80)
    
    for img_name in sorted(image_model_stats.keys()):
        print(f"\n📷 {img_name}")
        model_rates = [(m, s['rate'], s['total']) for m, s in image_model_stats[img_name].items() if s['total'] > 0]
        model_rates.sort(key=lambda x: -x[1])
        
        for rank, (model_name, rate, total) in enumerate(model_rates, 1):
            # Truncate model name for display
            display_name = model_name[:45] + "..." if len(model_name) > 48 else model_name
            bar = "█" * int(rate * 20) + "░" * (20 - int(rate * 20))
            print(f"   {rank}. {display_name:<48} {bar} {rate:.1%}")
    
    # ========================================================================
    # PRINT HEATMAP-STYLE SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print("CONSISTENCY HEATMAP (Model x Image)")
    print("="*80)
    print("\nLegend: ██ >80% | ▓▓ 60-80% | ▒▒ 40-60% | ░░ 20-40% | ·· <20%")
    print()
    
    # Header with image names (truncated)
    header = f"{'Model':<35}"
    for img_name in sorted(image_model_stats.keys()):
        short_name = img_name[:8] if len(img_name) > 8 else img_name
        header += f" {short_name:<10}"
    print(header)
    print("-" * len(header))
    
    # Rows for each model
    for model_name in model_list:
        row = f"{model_name[:33]:<35}"
        for img_name in sorted(image_model_stats.keys()):
            if img_name in image_model_stats and model_name in image_model_stats[img_name]:
                rate = image_model_stats[img_name][model_name]['rate']
                if rate >= 0.8:
                    symbol = "██"
                elif rate >= 0.6:
                    symbol = "▓▓"
                elif rate >= 0.4:
                    symbol = "▒▒"
                elif rate >= 0.2:
                    symbol = "░░"
                else:
                    symbol = "··"
                row += f" {symbol} {rate*100:>5.1f}% "
            else:
                row += f" {'N/A':^10}"
        print(row)
    
    # ========================================================================
    # PRINT PAIRED COMPARISON SUMMARY TABLE
    # ========================================================================
    
    if paired_comparisons:
        print("\n" + "="*80)
        print("PAIRED COMPARISON SUMMARY TABLE")
        print("="*80)
        print(f"{'Model A':<40} {'Model B':<40} {'N':<6} {'Mean A':<8} {'Mean B':<8} {'Δ':<8} {'p-value':<12} {'Sig':<6}")
        print("-" * 130)
        
        for pair_key, comp in sorted(paired_comparisons.items()):
            if 'error' in comp:
                continue
            
            model_A = comp['model_A'][:38]
            model_B = comp['model_B'][:38]
            n = comp['n_pairs']
            mean_A = comp['mean_A']
            mean_B = comp['mean_B']
            delta = comp['delta']
            p_val = comp['wilcoxon_pvalue']
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
            
            print(f"{model_A:<40} {model_B:<40} {n:<6} {mean_A:<8.3f} {mean_B:<8.3f} {delta:<8.3f} {p_val:<12.6f} {sig:<6}")
    
    # ========================================================================
    # PRINT ARCHITECTURE GROUP COMPARISON REPORTS
    # ========================================================================
    
    if arch_comparison_results['reports']:
        print("\n" + "="*80)
        print("ARCHITECTURE GROUP COMPARISON REPORTS")
        print("="*80)
        print("\nNatural Language Findings:\n")
        
        for i, report in enumerate(arch_comparison_results['reports'], 1):
            print(f"{i}. {report}\n")
        
        # Also print architecture groups for reference
        print("\nArchitecture Groups:")
        for arch, models in sorted(arch_comparison_results['architecture_groups'].items()):
            print(f"  {arch}: {len(models)} model(s)")
            for model in models:
                print(f"    - {model}")
    
    # ========================================================================
    # PRINT ARCHITECTURE COMPARISON DETAILS
    # ========================================================================
    
    if arch_comparison_results['architecture_comparisons']:
        print("\n" + "="*80)
        print("ARCHITECTURE COMPARISON DETAILS")
        print("="*80)
        print(f"{'Architecture A':<20} {'Architecture B':<20} {'N Comp':<8} {'Mean Δ':<10} {'Median p':<12} {'Better':<15}")
        print("-" * 100)
        
        for comp_key, comp in sorted(arch_comparison_results['architecture_comparisons'].items()):
            arch_A = comp['architecture_A']
            arch_B = comp['architecture_B']
            n_comp = comp['n_comparisons']
            mean_delta = comp['mean_delta']
            median_p = comp['median_p_value']
            better = comp['better_architecture'] or 'Equal'
            
            delta_str = f"{mean_delta:.3f}" if mean_delta is not None else "N/A"
            p_str = f"{median_p:.6f}" if median_p is not None else "N/A"
            
            print(f"{arch_A:<20} {arch_B:<20} {n_comp:<8} {delta_str:<10} {p_str:<12} {better:<15}")
    
    print("\n" + "="*80)
    print(f"Results saved to: {output_file}")
    print("="*80 + "\n")


