import sys
import os.path as osp
from os.path import expanduser
from vlmeval.dataset import SUPPORTED_DATASETS
from vlmeval.config import *
from vlmeval.smp import *

PTH = osp.realpath(__file__)
#IMAGE_PTH = osp.join(osp.dirname(PTH), '~/VLMEvalKit/assets/022.jpg')
# Use expanduser to expand ~ or use absolute path
IMAGE_PTH = expanduser('~/VLMEvalKit/assets/022.jpg')

def CHECK(val, msg):
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

if __name__ == "__main__":
    # Default text prompt
    text_prompt = "Are the chop sticks to the left or right of the bowl?"

    # Allow override from command line
    if len(sys.argv) > 1:
        text_prompt = sys.argv[1]

    msg = [
        dict(type='image', value=IMAGE_PTH),
        dict(type='text', value=text_prompt)
    ]
    ''' 
    model_list = [
        "llava_v1.5_7b",
        "llava_v1.5_7b_finetune_mrope_clip",
        "llava_v1.5_7b_finetune_siglip",
        "llava_v1.5_7b_finetune_mrope_siglip_base_patch16_256",
        "llava_v1.5_7b_finetune_siglip2",
        "llava_v1.5_7b_finetune_mrope_siglip2_base_patch16_256",
        "llava_v1.5_7b_finetune_aimv2",
        "llava_v1.5_7b_finetune_mrope_aimv2",
        "Qwen2.5-VL-7B-Instruct"
    ]
    '''
    model_list = ["llava_v1.5_7b"]

    results = []
    for m in model_list:
        res = CHECK(m, msg)
        if isinstance(res, list):
            results.extend(res)
        else:
            results.append(res)

    # Print prompt and results table
    print("\n====================================")
    print(f"Prompt used: \"{text_prompt}\"")
    print("Results (model name, output):")
    print(results)
    print("====================================\n")

