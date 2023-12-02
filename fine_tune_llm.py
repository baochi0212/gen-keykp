import os

from glob import glob
from copy import deepcopy
from random import randrange
from functools import partial
from trl import SFTTrainer

import torch
from accelerate import Accelerator
import bitsandbytes as bnb
from trainer import CustomTrainer, CustomArgs

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
    PeftModel
)
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# torch.cuda.set_device(2)
# model_name = "HuggingFaceH4/zephyr-7b-beta"
# model_name = "/home/speechmt/chitb/SimCKP/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93"


from llm_dataset import *

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./")
if 'llama' in model_name:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    cache_dir="./",
    device_map={'': 0},
    # device_map="auto"
)


model.config.use_cache = False
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    # lm_head is often excluded.
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)
modules = find_all_linear_names(model)


lora_alpha = 16
lora_dropout = 0.1
lora_r = 8


peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=modules,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM"
)


model = get_peft_model(model, peft_config)
training_args = CustomArgs(
    model_type=model_type,
    model_name=model_name,
    output_dir=f"outputs_{model_type}",
    output_merged_dir=f"lora_{model_type}",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=1e-4,
    max_grad_norm=1.0,  
    # max_steps=500,
    lr_scheduler_type="linear",
    warmup_steps=50,
    fp16=True,
    logging_strategy="steps",
    logging_steps=300,
    save_strategy="steps",
    save_steps=9000,
    #evaluation_strategy="epoch",
    #per_device_eval_batch_size=8,
    optim="paged_adamw_8bit",
    num_train_epochs=3, 
    # report_to="wandb"
    report_to="tensorboard"
)


trainer = CustomTrainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    train_dataset=dataset,
)

# trainer = SFTTrainer(
#     model=model,
#     train_dataset=dataset,
#     peft_config=peft_config,
#     dataset_text_field="text",
#     max_seq_length=1024,
#     tokenizer=tokenizer,
#     args=training_args,
#     packing=False,
# )
def eval(type, path):
    step_num = path.split("_")[-1]
    os.system(f"python infer_valid_set_llm.py {type} {path} ")
    os.system(f"bash ./evaluate_llm_valid.sh {type} {step_num}")
def test(type, path):
    step_num = path.split("_")[-1]
    os.system(f"python infer_llm.py {type} {path}")
    os.system(f"bash ./evaluate_llm.sh {type} {step_num}")
    

#train
trainer.train()  # Now we just run train()!
#eval
for eval_path in glob(f"./lora_{model_type}*"):
    print("Evaluating checkpoint: ", eval_path)
    #eval(model_type, eval_path)
    test(model_type, eval_path)

#print("Args checkig", training_args.model_type, training_args.model_name)


