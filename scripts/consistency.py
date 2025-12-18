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
    # SAVE RESULTS
    # ========================================================================
    
    output_file = '/home/ubuntu/VLMEvalKit/consistency_results_comprehensive.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
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
    
    print("\n" + "="*80)
    print(f"Results saved to: {output_file}")
    print("="*80 + "\n")


