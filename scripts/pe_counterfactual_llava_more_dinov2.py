"""
PE Counterfactual Test for LLaVA-MORE DINOv2 Model

This script tests the sensitivity of the aimagelab/LLaVA_MORE-gemma_2_9b-dinov2-finetuning
model to positional encoding (PE) modifications. It applies counterfactual interventions
(shuffled, constant PE) to understand how much the model relies on positional information
for spatial reasoning.

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
import torch
import numpy as np
from PIL import Image
from scipy.stats import wilcoxon, binom
import json
import glob
import argparse
import gc
from collections import defaultdict


# ============================================================================
# LLAVA-MORE MODEL LOADING
# ============================================================================

class LLaVAMoreDINOv2Model:
    """
    Wrapper for loading and running LLaVA-MORE DINOv2 model from HuggingFace.
    Includes hooks for PE modification experiments.
    
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
        self.hooks = []
        
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
                    self.processor = None
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
    
    def get_vision_encoder(self):
        """Find and return the vision encoder module for hook injection."""
        vision_encoder = None
        
        # Try different paths to find the vision encoder
        vision_paths = [
            'vision_tower',
            'vision_model',
            'model.vision_tower',
            'model.vision_model',
            'visual',
            'model.visual',
        ]
        
        for path in vision_paths:
            try:
                parts = path.split('.')
                obj = self.model
                for part in parts:
                    obj = getattr(obj, part, None)
                    if obj is None:
                        break
                if obj is not None:
                    vision_encoder = obj
                    print(f"  ✓ Found vision encoder at: {path}")
                    break
            except:
                continue
        
        if vision_encoder is None:
            # Search all named modules
            for name, module in self.model.named_modules():
                if 'vision' in name.lower() or 'visual' in name.lower() or 'dino' in name.lower():
                    print(f"    Found potential vision module: {name}")
                    vision_encoder = module
                    break
        
        return vision_encoder
    
    def generate(self, messages):
        """
        Generate response for given messages.
        
        Args:
            messages: List of dicts with 'type' and 'value' keys
        
        Returns:
            Generated text response
        """
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
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text_prompt}
                ]
            }
        ]
        
        if hasattr(self.processor, 'apply_chat_template'):
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        else:
            prompt = f"<image>\nUser: {text_prompt}\nAssistant:"
        
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id if hasattr(self.processor, 'tokenizer') else None,
            )
        
        generated_ids = output_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response.strip()
    
    def _generate_llava_native(self, image, text_prompt):
        """Generate using LLaVA native API."""
        from llava.mm_utils import process_images, tokenizer_image_token
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates
        
        image_tensor = process_images([image], self.image_processor, self.model.config)
        if isinstance(image_tensor, list):
            image_tensor = [img.to(self.model.device, dtype=torch.float16) for img in image_tensor]
        else:
            image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)
        
        conv_mode = "gemma_2"  # For Gemma-2 based models
        conv = conv_templates[conv_mode].copy()
        
        question = DEFAULT_IMAGE_TOKEN + '\n' + text_prompt
        
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        input_ids = input_ids.unsqueeze(0).to(self.model.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[image.size],
                do_sample=False,
                max_new_tokens=256,
                use_cache=True,
            )
        
        response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        
        return response.strip()


# Global model cache
_model_cache = {}

def get_model(model_name="aimagelab/LLaVA_MORE-gemma_2_9b-dinov2-finetuning"):
    """Get or load the model."""
    if model_name not in _model_cache:
        _model_cache[model_name] = LLaVAMoreDINOv2Model(model_name)
    return _model_cache[model_name]


# ============================================================================
# PE MODIFIER
# ============================================================================

class PEModifier:
    """Handles PE modification at different architectural levels"""
    
    def __init__(self, mode='normal', seed=None):
        self.mode = mode
        self.seed = seed
        self.hooks = []
    
    def modify_embeddings(self, embeddings):
        """Apply PE modification to embeddings tensor"""
        if self.mode == 'normal':
            return embeddings
        
        # Handle different tensor shapes
        if embeddings.dim() == 3:  # [batch, seq_len, hidden]
            batch_size, seq_len, hidden_dim = embeddings.shape
        elif embeddings.dim() == 2:  # [seq_len, hidden]
            embeddings = embeddings.unsqueeze(0)
            batch_size, seq_len, hidden_dim = embeddings.shape
        else:
            return embeddings
        
        if self.mode == 'shuffled':
            # Use seed for reproducibility if provided
            if self.seed is not None:
                generator = torch.Generator(device=embeddings.device)
                generator.manual_seed(self.seed)
            else:
                generator = None
            
            for b in range(batch_size):
                if generator:
                    perm = torch.randperm(seq_len, generator=generator, device=embeddings.device)
                else:
                    perm = torch.randperm(seq_len, device=embeddings.device)
                embeddings[b] = embeddings[b][perm]
        
        elif self.mode == 'constant':
            # Replace with mean (removes positional information)
            mean_embed = embeddings.mean(dim=1, keepdim=True)
            embeddings = mean_embed.expand_as(embeddings)
        
        return embeddings
    
    def create_forward_hook(self):
        """Create hook function"""
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                return self.modify_embeddings(output)
            elif isinstance(output, tuple):
                modified = self.modify_embeddings(output[0])
                return (modified,) + output[1:]
            return output
        return hook_fn
    
    def create_forward_pre_hook(self):
        """Create pre-hook for modifying inputs"""
        def pre_hook_fn(module, input):
            if isinstance(input, tuple) and len(input) > 0:
                if isinstance(input[0], torch.Tensor):
                    modified = self.modify_embeddings(input[0])
                    return (modified,) + input[1:]
            return input
        return pre_hook_fn


# ============================================================================
# HOOK INJECTION
# ============================================================================

def find_and_hook_vision_encoder(model_wrapper, pe_modifier):
    """
    Find vision encoder in the model and inject PE modification hooks.
    
    Args:
        model_wrapper: LLaVAMoreDINOv2Model instance
        pe_modifier: PEModifier instance
    
    Returns:
        (hooks, hooked_modules) - list of hook handles and names of hooked modules
    """
    hooks = []
    hooked_modules = []
    
    model = model_wrapper.model
    
    # Find vision encoder
    vision_encoder = model_wrapper.get_vision_encoder()
    
    if vision_encoder is None:
        print("  ✗ Warning: Could not locate vision encoder. PE modification will not work.")
        return hooks, hooked_modules
    
    # Hook injection strategies
    hook_targets = []
    
    # Strategy 1: Hook embeddings/patch_embed layer (earliest)
    for attr_name in ['embeddings', 'patch_embed', 'patch_embedding', 'embed_patches', 'cls_token']:
        if hasattr(vision_encoder, attr_name):
            module = getattr(vision_encoder, attr_name)
            if module is not None:
                hook_targets.append((f'vision_encoder.{attr_name}', module))
    
    # Strategy 2: Hook encoder layers (for DINOv2)
    if hasattr(vision_encoder, 'encoder'):
        encoder = vision_encoder.encoder
        if hasattr(encoder, 'layer') and len(encoder.layer) > 0:
            hook_targets.append(('vision_encoder.encoder.layer[0]', encoder.layer[0]))
        elif hasattr(encoder, 'layers') and len(encoder.layers) > 0:
            hook_targets.append(('vision_encoder.encoder.layers[0]', encoder.layers[0]))
    
    # Strategy 3: For DINOv2 specific architecture
    if hasattr(vision_encoder, 'blocks'):
        if len(vision_encoder.blocks) > 0:
            hook_targets.append(('vision_encoder.blocks[0]', vision_encoder.blocks[0]))
    
    # Strategy 4: Hook the entire vision encoder output
    hook_targets.append(('vision_encoder', vision_encoder))
    
    # Register hooks
    for name, module in hook_targets:
        try:
            hook = module.register_forward_hook(pe_modifier.create_forward_hook())
            hooks.append(hook)
            hooked_modules.append(name)
            print(f"  ✓ Registered hook on: {name} ({type(module).__name__})")
        except Exception as e:
            print(f"  ✗ Failed to hook {name}: {e}")
    
    if not hooks:
        print("  ✗ Warning: No hooks were successfully registered")
    
    return hooks, hooked_modules


# ============================================================================
# INFERENCE WITH PE MODIFICATION
# ============================================================================

def CHECK_with_PE(model_name, msg, pe_mode='normal', seed=None):
    """Run inference with specified PE mode"""
    
    print(f"\n  Loading model: {model_name}")
    
    try:
        model_wrapper = get_model(model_name)
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        return (model_name, f"Error loading model: {str(e)}", pe_mode, [])
    
    # Create PE modifier
    pe_modifier = PEModifier(mode=pe_mode, seed=seed)
    
    # Inject hooks
    print(f"  Injecting PE modification hooks (mode={pe_mode})...")
    hooks, hooked_modules = find_and_hook_vision_encoder(model_wrapper, pe_modifier)
    
    if not hooks:
        print(f"  ⚠ WARNING: No hooks injected - results may not reflect PE modifications!")
    
    try:
        print(f"  Running inference...")
        res = model_wrapper.generate(msg)
        result = (model_name, res, pe_mode, hooked_modules)
    except Exception as e:
        print(f"  ✗ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        result = (model_name, f"Error: {str(e)}", pe_mode, hooked_modules)
    finally:
        # Clean up hooks
        for hook in hooks:
            hook.remove()
        print(f"  Cleaned up {len(hooks)} hooks")
    
    return result


def run_pe_experiment(model_name, messages, pe_modes=['normal', 'shuffled', 'constant'], 
                      num_trials=3):
    """Run counterfactual PE test across multiple conditions"""
    
    results = {mode: [] for mode in pe_modes}
    
    print(f"\n{'='*70}")
    print(f"Running PE Experiment on: {model_name}")
    print(f"PE Modes: {pe_modes}")
    print(f"Queries: {len(messages)}")
    print(f"Trials per shuffled mode: {num_trials}")
    print(f"{'='*70}")
    
    for mode in pe_modes:
        print(f"\n{'─'*70}")
        print(f"PE Mode: {mode.upper()}")
        print(f"{'─'*70}")
        
        # Run multiple trials for shuffled (due to randomness)
        trials = num_trials if mode == 'shuffled' else 1
        
        for trial in range(trials):
            if trials > 1:
                print(f"\n  Trial {trial + 1}/{trials}")
            
            trial_results = []
            for idx, msg in enumerate(messages):
                query_text = msg[1]['value']
                print(f"\n  Query {idx+1}: {query_text[:60]}...")
                
                # Use different seed for each trial
                seed = trial * 1000 + idx if mode == 'shuffled' else None
                result = CHECK_with_PE(model_name, msg, pe_mode=mode, seed=seed)
                
                trial_results.append(result)
                print(f"  → Output: {result[1][:100]}{'...' if len(result[1]) > 100 else ''}")
                if len(result) > 3 and result[3]:
                    print(f"  → Hooked modules: {result[3]}")
            
            results[mode].append(trial_results)
    
    return results


# ============================================================================
# METRICS AND STATISTICAL TESTS
# ============================================================================

def bootstrap_ci(data, n_bootstrap=10000, confidence=0.95):
    """Compute bootstrap confidence interval for a statistic."""
    if len(data) == 0:
        return (None, None)
    
    data = np.array(data)
    n = len(data)
    bootstrap_means = []
    
    np.random.seed(42)
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    bootstrap_means = np.array(bootstrap_means)
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_means, lower_percentile)
    upper_bound = np.percentile(bootstrap_means, upper_percentile)
    
    return (float(lower_bound), float(upper_bound))


def exact_sign_test(acc_normal, acc_const):
    """
    Exact sign test for binary outcomes.
    Tests H0: P(worse under Const) = 0.5 vs H1: P(worse under Const) > 0.5
    """
    if len(acc_normal) != len(acc_const):
        return (None, None, None)
    
    acc_normal = np.array(acc_normal)
    acc_const = np.array(acc_const)
    
    worse = (acc_normal == 1) & (acc_const == 0)
    n_worse = np.sum(worse)
    n_total = len(acc_normal)
    
    if n_total == 0:
        return (0, 0, None)
    
    p_value = 1 - binom.cdf(n_worse - 1, n_total, 0.5)
    
    return (int(n_worse), int(n_total), float(p_value))


def compute_pe_sensitivity(results, ground_truth=None):
    """Compute ΔPE metric and detailed analysis with per-sample statistical tests"""
    
    metrics = {}
    
    # Extract outputs for comparison
    normal_outputs = [r[1] for r in results['normal'][0]]
    
    # For shuffled, compare across trials
    shuffled_all_trials = [[r[1] for r in trial] for trial in results['shuffled']]
    shuffled_first = shuffled_all_trials[0] if shuffled_all_trials else []
    
    constant_outputs = [r[1] for r in results['constant'][0]]
    
    def exact_match(outputs1, outputs2):
        """Exact string match (case-insensitive, stripped)"""
        if len(outputs1) != len(outputs2):
            return 0.0
        matches = sum(1 for o1, o2 in zip(outputs1, outputs2) 
                     if o1.strip().lower() == o2.strip().lower())
        return matches / len(outputs1)
    
    def token_overlap(outputs1, outputs2):
        """Token-level Jaccard similarity"""
        scores = []
        for o1, o2 in zip(outputs1, outputs2):
            tokens1 = set(o1.lower().split())
            tokens2 = set(o2.lower().split())
            if not tokens1 and not tokens2:
                scores.append(1.0)
            elif not tokens1 or not tokens2:
                scores.append(0.0)
            else:
                intersection = len(tokens1 & tokens2)
                union = len(tokens1 | tokens2)
                scores.append(intersection / union if union > 0 else 0.0)
        return sum(scores) / len(scores) if scores else 0.0
    
    def token_overlap_single(o1, o2):
        """Token-level Jaccard similarity for a single pair"""
        tokens1 = set(o1.lower().split())
        tokens2 = set(o2.lower().split())
        if not tokens1 and not tokens2:
            return 1.0
        elif not tokens1 or not tokens2:
            return 0.0
        else:
            intersection = len(tokens1 & tokens2)
            union = len(tokens1 | tokens2)
            return intersection / union if union > 0 else 0.0
    
    # Build per-sample accuracy vectors
    acc_normal = np.ones(len(normal_outputs))
    acc_shuf = np.array([1 if o1.strip().lower() == o2.strip().lower() 
                         else 0 for o1, o2 in zip(normal_outputs, shuffled_first)])
    acc_const = np.array([1 if o1.strip().lower() == o2.strip().lower() 
                          else 0 for o1, o2 in zip(normal_outputs, constant_outputs)])
    
    # Per-sample ΔPE values
    delta_pe_shuf_samples = np.array([1 - token_overlap_single(o1, o2) 
                                       for o1, o2 in zip(normal_outputs, shuffled_first)])
    delta_pe_const_samples = np.array([1 - token_overlap_single(o1, o2) 
                                        for o1, o2 in zip(normal_outputs, constant_outputs)])
    
    # Compute aggregate metrics
    metrics['exact_match_shuffled'] = exact_match(normal_outputs, shuffled_first)
    metrics['exact_match_constant'] = exact_match(normal_outputs, constant_outputs)
    
    metrics['token_overlap_shuffled'] = token_overlap(normal_outputs, shuffled_first)
    metrics['token_overlap_constant'] = token_overlap(normal_outputs, constant_outputs)
    
    # ΔPE based on token overlap
    metrics['delta_PE_shuffled'] = 1 - metrics['token_overlap_shuffled']
    metrics['delta_PE_constant'] = 1 - metrics['token_overlap_constant']
    
    # Bootstrap CIs for ΔPE
    ci_shuf = bootstrap_ci(delta_pe_shuf_samples)
    ci_const = bootstrap_ci(delta_pe_const_samples)
    metrics['delta_PE_shuffled_ci_lower'] = ci_shuf[0]
    metrics['delta_PE_shuffled_ci_upper'] = ci_shuf[1]
    metrics['delta_PE_constant_ci_lower'] = ci_const[0]
    metrics['delta_PE_constant_ci_upper'] = ci_const[1]
    
    # Per-sample accuracies
    metrics['acc_normal'] = acc_normal.tolist()
    metrics['acc_shuf'] = acc_shuf.tolist()
    metrics['acc_const'] = acc_const.tolist()
    
    # Statistical tests
    try:
        # Wilcoxon signed-rank test: normal vs shuffled
        if len(acc_normal) == len(acc_shuf) and len(acc_normal) > 0:
            stat_shuf, p_shuf = wilcoxon(acc_normal, acc_shuf, alternative='two-sided')
            metrics['wilcoxon_stat_shuf'] = float(stat_shuf)
            metrics['wilcoxon_pvalue_shuf'] = float(p_shuf)
        else:
            metrics['wilcoxon_stat_shuf'] = None
            metrics['wilcoxon_pvalue_shuf'] = None
        
        # Exact sign test: normal vs constant
        if len(acc_normal) == len(acc_const) and len(acc_normal) > 0:
            n_worse, n_total, p_const = exact_sign_test(acc_normal, acc_const)
            metrics['sign_test_n_worse_const'] = n_worse
            metrics['sign_test_n_total_const'] = n_total
            metrics['sign_test_pvalue_const'] = p_const
        else:
            metrics['sign_test_n_worse_const'] = None
            metrics['sign_test_n_total_const'] = None
            metrics['sign_test_pvalue_const'] = None
    except Exception as e:
        print(f"  ⚠️ Error in statistical tests: {str(e)}")
        metrics['wilcoxon_stat_shuf'] = None
        metrics['wilcoxon_pvalue_shuf'] = None
        metrics['sign_test_n_worse_const'] = None
        metrics['sign_test_n_total_const'] = None
        metrics['sign_test_pvalue_const'] = None
    
    # Inter-trial consistency for shuffled
    if len(shuffled_all_trials) > 1:
        trial_pairs = []
        for i in range(len(shuffled_all_trials)):
            for j in range(i+1, len(shuffled_all_trials)):
                consistency = token_overlap(shuffled_all_trials[i], shuffled_all_trials[j])
                trial_pairs.append(consistency)
        metrics['shuffled_trial_consistency'] = sum(trial_pairs) / len(trial_pairs) if trial_pairs else 1.0
    else:
        metrics['shuffled_trial_consistency'] = 1.0
    
    return metrics


def format_p_value(p_value):
    """Format p-value for display."""
    if p_value is None:
        return "N/A"
    if p_value < 0.001:
        return "<0.001"
    elif p_value < 0.01:
        return "<0.01"
    elif p_value < 0.05:
        return "<0.05"
    else:
        return f"{p_value:.3f}"


def print_pe_metrics(metrics, model_name):
    """Print detailed PE sensitivity metrics."""
    print(f"\n{'='*70}")
    print(f"PE SENSITIVITY METRICS - {model_name}")
    print(f"{'='*70}")
    print(f"\nExact Match Scores:")
    print(f"  Normal vs Shuffled: {metrics['exact_match_shuffled']:.3f}")
    print(f"  Normal vs Constant: {metrics['exact_match_constant']:.3f}")
    print(f"\nToken Overlap Scores:")
    print(f"  Normal vs Shuffled: {metrics['token_overlap_shuffled']:.3f}")
    print(f"  Normal vs Constant: {metrics['token_overlap_constant']:.3f}")
    print(f"\nΔPE (Sensitivity to PE modifications):")
    delta_s = metrics['delta_PE_shuffled']
    delta_c = metrics['delta_PE_constant']
    ci_s_lower = metrics.get('delta_PE_shuffled_ci_lower')
    ci_s_upper = metrics.get('delta_PE_shuffled_ci_upper')
    ci_c_lower = metrics.get('delta_PE_constant_ci_lower')
    ci_c_upper = metrics.get('delta_PE_constant_ci_upper')
    
    if ci_s_lower is not None and ci_s_upper is not None:
        print(f"  ΔPE(Shuffled):  {delta_s:.3f} [{ci_s_lower:.3f}, {ci_s_upper:.3f}]")
    else:
        print(f"  ΔPE(Shuffled):  {delta_s:.3f}")
    
    if ci_c_lower is not None and ci_c_upper is not None:
        print(f"  ΔPE(Constant):  {delta_c:.3f} [{ci_c_lower:.3f}, {ci_c_upper:.3f}]")
    else:
        print(f"  ΔPE(Constant):  {delta_c:.3f}")
    
    # Statistical tests
    print(f"\nStatistical Tests:")
    p_shuf = metrics.get('wilcoxon_pvalue_shuf')
    p_const = metrics.get('sign_test_pvalue_const')
    n_worse = metrics.get('sign_test_n_worse_const')
    n_total = metrics.get('sign_test_n_total_const')
    
    if p_shuf is not None:
        p_str_shuf = format_p_value(p_shuf)
        print(f"  Normal vs Shuffled (Wilcoxon): p = {p_str_shuf}")
    else:
        print(f"  Normal vs Shuffled (Wilcoxon): p = N/A")
    
    if p_const is not None and n_worse is not None and n_total is not None:
        p_str_const = format_p_value(p_const)
        print(f"  Normal vs Constant (Exact sign test): {n_worse}/{n_total} worse, p = {p_str_const}")
    else:
        print(f"  Normal vs Constant (Exact sign test): p = N/A")
    
    print(f"\nShuffled Trial Consistency: {metrics['shuffled_trial_consistency']:.3f}")


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
    parser = argparse.ArgumentParser(description='PE Counterfactual Test for LLaVA-MORE DINOv2 model')
    parser.add_argument('--model', type=str, default='aimagelab/LLaVA_MORE-gemma_2_9b-dinov2-finetuning',
                        help='HuggingFace model name or path')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to test image')
    parser.add_argument('--output', type=str, default='pe_counterfactual_results_llava_more_dinov2.json',
                        help='Output JSON file path')
    parser.add_argument('--num_trials', type=int, default=3,
                        help='Number of trials for shuffled PE mode')
    args = parser.parse_args()
    
    model_name = args.model
    image_path = args.image
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    # ========================================================================
    # SPATIAL QUERIES
    # ========================================================================
    
    spatial_queries = [
        [
            dict(type='image', value=image_path),
            dict(type='text', value="What objects are on the left side of the image?")
        ],
        [
            dict(type='image', value=image_path),
            dict(type='text', value="What object is closest to the top of the image?")
        ],
        [
            dict(type='image', value=image_path),
            dict(type='text', value="Describe the spatial arrangement of objects.")
        ],
        [
            dict(type='image', value=image_path),
            dict(type='text', value="Is there anything on the right side of the image?")
        ],
        [
            dict(type='image', value=image_path),
            dict(type='text', value="What is at the bottom of the image?")
        ],
    ]
    
    # ========================================================================
    # RUN EXPERIMENT
    # ========================================================================
    
    print("\n" + "="*80)
    print("PE COUNTERFACTUAL TEST - LLAVA-MORE DINOV2")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Image: {image_path}")
    print(f"  Queries: {len(spatial_queries)}")
    print(f"  PE Modes: normal, shuffled, constant")
    print(f"  Trials per shuffled: {args.num_trials}")
    print("="*80)
    
    try:
        results = run_pe_experiment(
            model_name, 
            spatial_queries,
            pe_modes=['normal', 'shuffled', 'constant'],
            num_trials=args.num_trials
        )
        
        metrics = compute_pe_sensitivity(results)
        print_pe_metrics(metrics, model_name)
        
    except Exception as e:
        print(f"✗ Error during experiment: {str(e)}")
        import traceback
        traceback.print_exc()
        metrics = {
            'error': str(e),
            'delta_PE_shuffled': -1,
            'delta_PE_constant': -1,
        }
        results = {}
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    output_file = args.output
    
    # Prepare serializable results
    serializable_results = {}
    for mode, trials in results.items():
        serializable_results[mode] = [[r[1] for r in trial] for trial in trials]
    
    output_data = {
        'model': model_name,
        'image': image_path,
        'metrics': convert_numpy_types(metrics),
        'outputs': serializable_results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    if 'error' not in metrics:
        delta_s = metrics.get('delta_PE_shuffled', -1)
        delta_c = metrics.get('delta_PE_constant', -1)
        ci_s_lower = metrics.get('delta_PE_shuffled_ci_lower')
        ci_s_upper = metrics.get('delta_PE_shuffled_ci_upper')
        ci_c_lower = metrics.get('delta_PE_constant_ci_lower')
        ci_c_upper = metrics.get('delta_PE_constant_ci_upper')
        p_shuf = metrics.get('wilcoxon_pvalue_shuf')
        p_const = metrics.get('sign_test_pvalue_const')
        n_worse = metrics.get('sign_test_n_worse_const')
        n_total = metrics.get('sign_test_n_total_const')
        
        p_str_shuf = format_p_value(p_shuf)
        p_str_const = format_p_value(p_const)
        
        print(f"\n{'Metric':<30} {'Value':<25} {'p-value':<15} {'Significant?':<15}")
        print(f"{'-'*85}")
        
        if ci_s_lower is not None and ci_s_upper is not None:
            delta_s_str = f"{delta_s:.3f} [{ci_s_lower:.3f}, {ci_s_upper:.3f}]"
        else:
            delta_s_str = f"{delta_s:.3f}"
        
        if ci_c_lower is not None and ci_c_upper is not None:
            delta_c_str = f"{delta_c:.3f} [{ci_c_lower:.3f}, {ci_c_upper:.3f}]"
        else:
            delta_c_str = f"{delta_c:.3f}"
        
        sig_shuf = "Yes*" if p_shuf is not None and p_shuf < 0.05 else ("Marginal" if p_shuf is not None and p_shuf < 0.1 else "No")
        sig_const = "Yes*" if p_const is not None and p_const < 0.05 else "No"
        
        print(f"{'ΔPE (Shuffled)':<30} {delta_s_str:<25} {p_str_shuf:<15} {sig_shuf:<15}")
        print(f"{'ΔPE (Constant)':<30} {delta_c_str:<25} {p_str_const:<15} {sig_const:<15}")
        
        # ================================================================
        # P-VALUE INTERPRETATION
        # ================================================================
        print(f"\n{'='*80}")
        print("P-VALUE INTERPRETATION")
        print(f"{'='*80}")
        print("""
A p-value tells you how likely your observed result would happen by random 
chance, assuming no real effect (null hypothesis: PE modifications don't matter).

  • p < 0.05  : Statistically significant - strong evidence PE matters
  • p < 0.10  : Marginally significant - suggestive evidence
  • p >= 0.10 : Not significant - could be due to chance

Tests used:
  • Wilcoxon signed-rank test: Compares normal vs shuffled PE outputs
  • Exact sign test: Counts how many samples got worse with constant PE
""")
        
        print(f"{'Test':<40} {'Result':<40}")
        print(f"{'-'*80}")
        
        # Wilcoxon interpretation
        if p_shuf is not None:
            if p_shuf < 0.05:
                shuf_interp = f"p={p_shuf:.3f} < 0.05 → Shuffling PE significantly changes outputs"
            elif p_shuf < 0.1:
                shuf_interp = f"p={p_shuf:.3f} < 0.10 → Marginally significant effect"
            else:
                shuf_interp = f"p={p_shuf:.3f} → Not significant (could be chance)"
        else:
            shuf_interp = "Test could not be performed"
        print(f"{'Wilcoxon (Normal vs Shuffled)':<40} {shuf_interp:<40}")
        
        # Sign test interpretation
        if p_const is not None and n_worse is not None and n_total is not None:
            if p_const < 0.05:
                const_interp = f"{n_worse}/{n_total} worse, p={p_const:.3f} → Constant PE significantly degrades outputs"
            else:
                const_interp = f"{n_worse}/{n_total} worse, p={p_const:.3f} → Effect not statistically significant"
        else:
            const_interp = "Test could not be performed"
        print(f"{'Sign Test (Normal vs Constant)':<40} {const_interp:<40}")
        
        # ================================================================
        # OVERALL INTERPRETATION
        # ================================================================
        print(f"\n{'='*80}")
        print("OVERALL INTERPRETATION")
        print(f"{'='*80}")
        
        avg_delta = (delta_s + delta_c) / 2
        
        # Determine statistical support
        sig_shuf_bool = p_shuf is not None and p_shuf < 0.05
        sig_const_bool = p_const is not None and p_const < 0.05
        
        print(f"\nSensitivity Level: ", end="")
        if avg_delta > 0.5:
            print(f"STRONG (avg ΔPE = {avg_delta:.3f})")
            sensitivity = "strong"
        elif avg_delta > 0.2:
            print(f"MODERATE (avg ΔPE = {avg_delta:.3f})")
            sensitivity = "moderate"
        else:
            print(f"LOW (avg ΔPE = {avg_delta:.3f})")
            sensitivity = "low"
        
        print(f"Statistical Support: ", end="")
        if sig_shuf_bool and sig_const_bool:
            print("STRONG (both tests significant at p < 0.05)")
            stat_support = "strong"
        elif sig_shuf_bool or sig_const_bool:
            print("PARTIAL (one test significant at p < 0.05)")
            stat_support = "partial"
        elif p_shuf is not None and p_shuf < 0.1:
            print("WEAK (marginally significant)")
            stat_support = "weak"
        else:
            print("NONE (no significant effects)")
            stat_support = "none"
        
        # Final conclusion
        print(f"\nConclusion:")
        if sensitivity == "strong" and stat_support in ["strong", "partial"]:
            print(f"  ✓ The model HEAVILY RELIES on positional encoding for spatial reasoning.")
            print(f"  ✓ Disrupting PE significantly changes model outputs.")
            print(f"  ✓ This is statistically supported (p < 0.05).")
        elif sensitivity == "moderate":
            print(f"  ~ The model uses BOTH positional encoding AND visual features.")
            print(f"  ~ PE contributes to spatial reasoning but isn't the only factor.")
        else:
            print(f"  ✗ The model's spatial reasoning is primarily based on VISUAL FEATURES.")
            print(f"  ✗ Positional encoding has minimal impact on outputs.")
        
        # Visual summary bar
        print(f"\nPE Dependence: ", end="")
        bar_length = int(avg_delta * 20)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        print(f"[{bar}] {avg_delta*100:.1f}%")
        
    else:
        print(f"  Error: {metrics.get('error', 'Unknown error')}")
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print("="*80 + "\n")
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

