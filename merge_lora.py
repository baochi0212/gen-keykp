import argparse
import bitsandbytes as bnb
from datasets import load_dataset
from functools import partial
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset
import  sys
## --lora_path --save_dir --model_path
path = sys.argv[1]
save_dir = sys.argv[2]
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
model = AutoPeftModelForCausalLM.from_pretrained(f"./{path}", torch_dtype=torch.float16)
model = model.merge_and_unload()
print("Loaded model!!!")
output_merged_dir = "llama2_merged_checkpoint"
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir, safe_serialization=True)
print("Merging")
# save tokenizer for easy inference
model_path = sys.argv[3]
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.save_pretrained(save_dir)
print("Tokenizer saved!!!")
