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
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from transformers import LlavaForConditionalGeneration, LlavaProcessor

PTH = osp.realpath(__file__)
IMAGE_PTH = '/home/ubuntu/VLMEvalKit/assets/022.jpg'

# Directory containing custom LLaVA models
CUSTOM_MODELS_DIR = expanduser('~/custom_llava_models')

# Base LLaVA model to load processor from (since fine-tuned models don't have processor files)
BASE_LLAVA_MODEL = "llava-hf/llava-1.5-7b-hf"

# ============================================================================
# CUSTOM MODEL LOADING
# ============================================================================

class CustomLLaVAModel:
    """Wrapper for loading and running custom local LLaVA models."""
    
    def __init__(self, model_path, device='cuda'):
        """
        Initialize custom LLaVA model.
        
        Args:
            model_path: Path to the model directory
            device: Device to run on ('cuda' or 'cpu')
        """
        self.model_path = model_path
        self.model_name = os.path.basename(model_path)
        self.device = device
        
        print(f"Loading custom model: {self.model_name}")
        print(f"  Path: {model_path}")
        
        # Load processor from base LLaVA model (fine-tuned models don't have processor files)
        print(f"  Loading processor from base model: {BASE_LLAVA_MODEL}")
        try:
            self.processor = LlavaProcessor.from_pretrained(BASE_LLAVA_MODEL)
        except Exception as e:
            print(f"  Failed to load processor from {BASE_LLAVA_MODEL}: {e}")
            # Try alternative base model
            alt_base = "llava-hf/llava-v1.6-mistral-7b-hf"
            print(f"  Trying alternative: {alt_base}")
            self.processor = LlavaProcessor.from_pretrained(alt_base)
        
        # Load model weights from local checkpoint
        print(f"  Loading model weights from: {model_path}")
        try:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
            self.model_type = 'llava'
            print(f"  ✓ Loaded successfully as LLaVA model")
        except Exception as e:
            print(f"  Failed to load as LLaVA: {e}")
            # Try loading with AutoModelForCausalLM
            try:
                print(f"  Trying AutoModelForCausalLM...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                self.model_type = 'auto'
                print(f"  ✓ Loaded as AutoModelForCausalLM")
            except Exception as e2:
                print(f"  Failed to load model: {e2}")
                raise RuntimeError(f"Cannot load model from {model_path}")
        
        self.model.eval()
    
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
            
            if self.model_type == 'llava':
                # Format for LLaVA
                prompt = f"USER: <image>\n{text_prompt}\nASSISTANT:"
                inputs = self.processor(text=prompt, images=image, return_tensors="pt")
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False,
                        pad_token_id=self.processor.tokenizer.pad_token_id
                    )
                
                # Decode only the new tokens
                input_len = inputs['input_ids'].shape[1]
                response = self.processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
                
            else:
                # Generic approach for AutoModel
                inputs = self.processor(text=text_prompt, images=image, return_tensors="pt")
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False
                    )
                
                response = self.processor.decode(output_ids[0], skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
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
        transform_type: One of ['original', 'hflip', 'rotate90', 'rotate180', 'rotate270']
    
    Returns:
        Path to transformed image
    """
    img = Image.open(image_path)
    
    if transform_type == 'original':
        return image_path
    elif transform_type == 'hflip':
        img_transformed = img.transpose(Image.FLIP_LEFT_RIGHT)
    elif transform_type == 'rotate90':
        img_transformed = img.rotate(90, expand=True)
    elif transform_type == 'rotate180':
        img_transformed = img.rotate(180, expand=True)
    elif transform_type == 'rotate270':
        img_transformed = img.rotate(270, expand=True)
    else:
        raise ValueError(f"Unknown transform: {transform_type}")
    
    # Save to temp file
    temp_path = image_path.replace('.jpg', f'_{transform_type}.jpg')
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
    
    def __init__(self, text, question_type='spatial', expected_flip_behavior='opposite'):
        """
        Args:
            text: Question text
            question_type: Type for normalization
            expected_flip_behavior: How answer should change with horizontal flip
                - 'opposite': left<->right swap (e.g., "Is X to the left of Y?")
                - 'same': answer should not change (e.g., "Is X above Y?")
                - 'custom': use custom logic
        """
        self.text = text
        self.question_type = question_type
        self.expected_flip_behavior = expected_flip_behavior
    
    def get_expected_answer(self, original_answer, transform_type):
        """
        Get expected answer after transformation.
        
        Args:
            original_answer: Normalized answer from original image
            transform_type: Type of transformation applied
        
        Returns:
            Expected normalized answer after transformation
        """
        if transform_type == 'original':
            return original_answer
        
        if transform_type == 'hflip':
            if self.expected_flip_behavior == 'opposite':
                # Swap left/right
                if original_answer == 'left':
                    return 'right'
                elif original_answer == 'right':
                    return 'left'
            elif self.expected_flip_behavior == 'same':
                return original_answer
        
        # For rotations, logic depends on question
        # Can be extended later
        
        return None  # Cannot determine expected answer


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
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Define spatial questions
    questions = [
        SpatialQuestion(
            "Are the chopsticks to the left or right of the bowl?",
            question_type='spatial',
            expected_flip_behavior='opposite'
        ),
        SpatialQuestion(
            "Is the spoon to the left of the bowl?",
            question_type='spatial',
            expected_flip_behavior='opposite'
        ),
        SpatialQuestion(
            "Is the bowl above the chopsticks?",
            question_type='spatial',
            expected_flip_behavior='same'
        ),
    ]
    
    # Define models to test
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
    
    # Define transformations to test
    transforms = ['original', 'hflip']  # Start with just horizontal flip
    # transforms = ['original', 'hflip', 'rotate90', 'rotate180', 'rotate270']  # Add more later
    
    # Run evaluation
    all_results = []
    
    print("\n" + "="*80)
    print("VLM CONSISTENCY TESTING WITH IMAGE TRANSFORMATIONS")
    print("="*80)
    print(f"\nTesting {len(vlmeval_models)} VLMEval models + {len(custom_models)} custom models")
    print(f"Custom models directory: {CUSTOM_MODELS_DIR}")
    print(f"Test image: {IMAGE_PTH}")
    
    for model_name in model_list:
        print(f"\n{'='*80}")
        print(f"Testing model: {model_name}")
        print("-" * 80)
        
        try:
            results = evaluate_consistency(
                model_name=model_name,
                image_path=IMAGE_PTH,
                questions=questions,
                transforms=transforms
            )
            
            all_results.append(results)
            
            # Print results
            print(f"\nOverall Metrics:")
            print(f"  Consistency Rate: {results['metrics']['consistency_rate']:.2%}")
            print(f"  Consistent: {results['metrics']['consistent']}")
            print(f"  Inconsistent: {results['metrics']['inconsistent']}")
            print(f"  Undetermined: {results['metrics']['undetermined']}")
            
            print(f"\nDetailed Results:")
            for q_result in results['questions']:
                print(f"\n  Question: {q_result['question']}")
                print(f"    Original answer: {q_result['answers']['original']['normalized']} " +
                      f"(raw: {q_result['answers']['original']['raw'][:100] if q_result['answers']['original']['raw'] else 'None'}...)")
                
                for transform in transforms:
                    if transform == 'original':
                        continue
                    
                    ans = q_result['answers'][transform]
                    cons = q_result['consistency'][transform]
                    
                    print(f"    {transform.capitalize()} answer: {ans['normalized']} " +
                          f"(expected: {cons['expected']}, status: {cons['status']})")
        
        except Exception as e:
            print(f"  ❌ Error testing {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'model': model_name,
                'error': str(e),
                'metrics': {'consistency_rate': 0.0}
            })
            continue
        
        # Clear GPU memory between models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save results to JSON
    output_file = '/home/ubuntu/VLMEvalKit/consistency_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Model':<55} {'Consistency Rate':<20}")
    print("-" * 80)
    
    for result in all_results:
        model_name = result['model']
        if 'error' in result:
            rate = 'ERROR'
        else:
            rate = f"{result['metrics']['consistency_rate']:.2%}"
        print(f"{model_name:<55} {rate:<20}")
    
    print("\n" + "="*80)
    print(f"Results saved to: {output_file}")
    print("="*80 + "\n")


