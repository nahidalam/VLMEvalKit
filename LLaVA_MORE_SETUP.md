# LLaVA-MORE DINOv2 Setup Guide

This document summarizes the steps required to set up and run the LLaVA-MORE DINOv2 model (`aimagelab/LLaVA_MORE-gemma_2_9b-dinov2-finetuning`) for spatial coherence analysis.

## Prerequisites

- Ubuntu with NVIDIA GPU (tested on L40S with CUDA 12.7)
- Conda package manager
- ~20GB disk space for model weights

## Step 1: Clone LLaVA-MORE Repository

```bash
cd /home/ubuntu
git clone https://github.com/aimagelab/LLaVA-MORE.git
cd LLaVA-MORE
```

## Step 2: Create Conda Environment

```bash
conda create -n llava-more python=3.10 -y
conda activate llava-more
```

## Step 3: Install PyTorch (CUDA 12.x)

```bash
# For CUDA 12.x (adjust for your CUDA version)
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu121
```

## Step 4: Install Dependencies

```bash
# Core dependencies
pip install transformers==4.45.0 accelerate safetensors huggingface-hub
pip install einops timm sentencepiece tokenizers pillow scipy numpy
pip install protobuf pydantic

# Optional: flash-attention (can skip if installation fails)
pip install psutil ninja packaging wheel setuptools
pip install flash-attn --no-build-isolation  # May fail, that's OK
```

## Step 5: Download and Configure the Model

```bash
# Create models directory
mkdir -p /home/ubuntu/models

# Download model from HuggingFace
huggingface-cli download aimagelab/LLaVA_MORE-gemma_2_9b-dinov2-finetuning \
    --local-dir /home/ubuntu/models/llava-more-dinov2

# Fix the hardcoded vision tower path in config.json
sed -i 's|/leonardo_scratch/large/userexternal/fcocchi0/rag_mlmm/hf_models/cvprw/visual/models--facebook--dinov2-large/snapshots/47b73eefe95e8d44ec3623f8890bd894b6ea2d6c|facebook/dinov2-large|g' /home/ubuntu/models/llava-more-dinov2/config.json
```

## Step 6: Fix LLaVA-MORE Code Compatibility Issues

### 6.1 Remove `logits_to_keep` argument (transformers 4.45.0 compatibility)

The `logits_to_keep` parameter is not supported in transformers 4.45.0. Remove it from `llava_gemma.py`:

```bash
sed -i '/logits_to_keep=logits_to_keep/d' /home/ubuntu/LLaVA-MORE/src/llava/model/language_model/llava_gemma.py
```

Verify the change:
```bash
sed -n '100,115p' /home/ubuntu/LLaVA-MORE/src/llava/model/language_model/llava_gemma.py
```

The `super().forward()` call should NOT contain `logits_to_keep=logits_to_keep`.

## Step 7: Set Environment Variables

Every time you run the model, set these environment variables:

```bash
cd /home/ubuntu/LLaVA-MORE
export PYTHONPATH=/home/ubuntu/LLaVA-MORE/src:$PYTHONPATH
export TOKENIZER_PATH=/home/ubuntu/models/llava-more-dinov2
```

**Tip:** Add these to `~/.bashrc` for persistence:
```bash
echo 'export PYTHONPATH="/home/ubuntu/LLaVA-MORE/src:$PYTHONPATH"' >> ~/.bashrc
echo 'export TOKENIZER_PATH="/home/ubuntu/models/llava-more-dinov2"' >> ~/.bashrc
```

## Step 8: Verify Installation

```bash
conda activate llava-more
cd /home/ubuntu/LLaVA-MORE
export PYTHONPATH=/home/ubuntu/LLaVA-MORE/src:$PYTHONPATH
export TOKENIZER_PATH=/home/ubuntu/models/llava-more-dinov2

# Test imports
python -c 'import torch; print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")'
python -c 'import llava; print("llava module OK")'
python -c 'from llava.conversation import conv_templates; print("conv_templates OK")'
```

## Running the Evaluation Scripts

### Important: Conversation Mode

The scripts must use `conv_mode = "gemma_2"` (not `"gemma"`). If you see `KeyError: 'gemma'`, fix it:

```bash
# In consistency script
sed -i 's/conv_mode = "gemma"/conv_mode = "gemma_2"/' /path/to/consistency_llava_more_dinov2.py

# In PE counterfactual script
sed -i 's/conv_mode = "gemma"/conv_mode = "gemma_2"/' /path/to/pe_counterfactual_llava_more_dinov2.py
```

### Run Consistency Evaluation

```bash
conda activate llava-more
cd /home/ubuntu/LLaVA-MORE
export PYTHONPATH=/home/ubuntu/LLaVA-MORE/src:$PYTHONPATH
export TOKENIZER_PATH=/home/ubuntu/models/llava-more-dinov2

python /home/ubuntu/VLMEvalKit/scripts/consistency_llava_more_dinov2.py \
    --model /home/ubuntu/models/llava-more-dinov2 \
    --image /home/ubuntu/VLMEvalKit/assets/022.jpg \
    --output /home/ubuntu/VLMEvalKit/consistency_results_llava_more_dinov2.json
```

### Run PE Counterfactual Evaluation

```bash
python /home/ubuntu/VLMEvalKit/scripts/pe_counterfactual_llava_more_dinov2.py \
    --model /home/ubuntu/models/llava-more-dinov2 \
    --image /home/ubuntu/VLMEvalKit/assets/022.jpg \
    --output /home/ubuntu/VLMEvalKit/pe_counterfactual_results_llava_more_dinov2.json \
    --num_trials 3
```

## Summary of Issues and Fixes

| Issue | Fix |
|-------|-----|
| `llava_gemma` not recognized by transformers | Use LLaVA-MORE native loader with correct PYTHONPATH |
| Hardcoded vision tower path in config.json | Replace with `facebook/dinov2-large` |
| `No module named 'llava'` | Set `PYTHONPATH=/home/ubuntu/LLaVA-MORE/src` |
| `tokenizer_path = None` error | Set `TOKENIZER_PATH` environment variable |
| `KeyError: 'gemma'` | Use `conv_mode = "gemma_2"` |
| `logits_to_keep` not supported | Remove from `llava_gemma.py` forward() call |
| Missing protobuf | `pip install protobuf` |

## Model Information

- **Model**: `aimagelab/LLaVA_MORE-gemma_2_9b-dinov2-finetuning`
- **LLM Backbone**: Gemma-2-9B
- **Vision Backbone**: DINOv2-Large
- **HuggingFace**: https://huggingface.co/aimagelab/LLaVA_MORE-gemma_2_9b-dinov2-finetuning
- **Repository**: https://github.com/aimagelab/LLaVA-MORE

