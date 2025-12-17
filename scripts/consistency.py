import sys
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

PTH = osp.realpath(__file__)
IMAGE_PTH = expanduser('~/VLMEvalKit/assets/022.jpg')

def CHECK(val, msg):
    """Run model inference on a message."""
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
    model_list = ["llava_v1.5_7b"]
    
    # Define transformations to test
    transforms = ['original', 'hflip']  # Start with just horizontal flip
    # transforms = ['original', 'hflip', 'rotate90', 'rotate180', 'rotate270']  # Add more later
    
    # Run evaluation
    all_results = []
    
    print("\n" + "="*80)
    print("VLM CONSISTENCY TESTING WITH IMAGE TRANSFORMATIONS")
    print("="*80)
    
    for model_name in model_list:
        print(f"\nTesting model: {model_name}")
        print("-" * 80)
        
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
                  f"(raw: {q_result['answers']['original']['raw']})")
            
            for transform in transforms:
                if transform == 'original':
                    continue
                
                ans = q_result['answers'][transform]
                cons = q_result['consistency'][transform]
                
                print(f"    {transform.capitalize()} answer: {ans['normalized']} " +
                      f"(expected: {cons['expected']}, status: {cons['status']})")
    
    # Save results to JSON
    output_file = expanduser('~/VLMEvalKit/consistency_results.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*80)
    print(f"Results saved to: {output_file}")
    print("="*80 + "\n")
