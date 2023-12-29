import os
print("Debug torch: ", os.environ["TORCH_DISTRIBUTED_DEBUG"])
#modules = find_all_linear_names(model)
from glob import glob
from copy import deepcopy
from random import randrange
from functools import partial
#from trl import SFTTrainer

import torch
from accelerate import Accelerator
import bitsandbytes as bnb
from trainer import CustomTrainer, CustomArgs
from dataclasses import dataclass, field
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    LlamaForCausalLM, 
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    #prepare_model_for_kbit_training,
    get_peft_model,
    PeftModel
)
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
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
#model = AutoModelForCausalLM.from_pretrained(
#    model_name,
#    quantization_config=bnb_config,
#    cache_dir="./",
#    device_map = {"":int(os.environ.get("LOCAL_RANK") or 0)}
#    #device_map={'': 0},
#    # device_map="auto"
#)
def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True):
    r"""
    This method wraps the entire protocol for preparing a model before running a training. This includes:
        1- Cast the layernorm in fp32 2- making output embedding layer require grads 3- Add the upcasting of the lm
        head to fp32

    Args:
        model, (`transformers.PreTrainedModel`):
            The loaded model from `transformers`
    """
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)
    print("Load in kbit is ", loaded_in_kbit)

    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    # cast all non INT8/INT4 parameters to fp32
    for param in model.parameters():
        if ((param.dtype == torch.float16) or (param.dtype == torch.bfloat16)) and loaded_in_kbit:
            param.data = param.data.to(torch.float32)

    for name, module in model.named_modules():
        if 'norm' in name:
            module = module.to(torch.float32)

    if loaded_in_kbit and use_gradient_checkpointing:
        # For backward compatibility
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, _input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

    return model
model = LlamaForCausalLM.from_pretrained(
        model_name,
        cache_dir="./",
        revision="main",
        #use_auth_token=True if model_args.use_auth_token else None,
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True,
        device_map={"":int(os.environ.get("LOCAL_RANK") or 0)},
        load_in_4bit=True,
        quantization_config=bnb_config,
        #torch_dtype="float16"
    )
model.config.use_cache = False
#model.gradient_checkpointing_enable()
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
    inference_mode=False,
    task_type="CAUSAL_LM"
)
#peft_config = LoraConfig(
#                task_type="CAUSAL_LM",
#                target_modules=target_modules,
#                inference_mode=False,
#                r=lora_rank, lora_alpha=lora_alpha,
#                lora_dropout=lora_dropout,
#                modules_to_save=modules_to_save)
#
model = get_peft_model(model, peft_config)
training_args = CustomArgs(
    model_type=model_type,
    model_name=model_name,
    output_dir=f"outputs_{model_type}",
    output_merged_dir=f"lora_{model_type}",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    learning_rate=1e-4,
    max_grad_norm=1.0,  
    # max_steps=500,
    lr_scheduler_type="linear",
    warmup_steps=50,
    fp16=True,
    logging_strategy="steps",
    logging_steps=50,
    save_strategy="steps",
    save_steps=6000,
    #max_steps=600,
    #evaluation_strategy="epoch",
    #per_device_eval_batch_size=8,
    optim="paged_adamw_8bit",
    num_train_epochs=3, 
    # report_to="wandb"
    report_to="tensorboard",
    ddp_find_unused_parameters=False
)
#@dataclass
#class MyTrainingArguments(TrainingArguments):
#    trainable : Optional[str] = field(default="q_proj,v_proj")
#    lora_rank : Optional[int] = field(default=8)
#    lora_dropout : Optional[float] = field(default=0.1)
#    lora_alpha : Optional[float] = field(default=32.)
#    modules_to_save : Optional[str] = field(default=None)
#    peft_path : Optional[str] = field(default=None)
#    use_flash_attention_2 : Optional[bool] = field(default=False)
#    double_quant: Optional[bool] = field(default=True)
#    quant_type: Optional[str] = field(default="nf4")
#    load_in_kbits: Optional[int] = field(default=16)
#    full_finetuning : Optional[bool] = field(default=False)
#    output_merged_dir: Optional[str] = field(default=None)
#    optim: Optional[str] = field(default="paged_adamw_8
#
#parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))
#training_args = parser.parse_args_into_dataclasses()
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
#for eval_path in glob(f"./lora_{model_type}*"):
#    print("Evaluating checkpoint: ", eval_path)
#    #eval(model_type, eval_path)
#    test(model_type, eval_path)
#
#print("Args checkig", training_args.model_type, training_args.model_name)
