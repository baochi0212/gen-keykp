from transformers.trainer import *
import argparse
import bitsandbytes as bnb
#from datasets import load_dataset
from functools import partial
import os
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, Trainer, TrainingArguments, BitsAndBytesConfig, \
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
#from datasets import load_datase
import  sys
import os
import subprocess
class CustomArgs(TrainingArguments):
    def __init__(self, model_type, model_name, output_merged_dir, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_type = model_type
        self.model_name = model_name 
        self.output_merged_dir = output_merged_dir
        self.step_count = 0

class CustomTrainer(Trainer):
    def _save(self, output_dir, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        #supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        
        if True:
            self.model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)
        
        # Merge lora
        print("MERGING LORA....")
        step_num = self.args.step_count + self.args.save_steps
        self.args.step_count = step_num
        ## --lora_path --save_dir --model_path
        path = f"{output_dir}"
        #os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        model = AutoPeftModelForCausalLM.from_pretrained(f"./{path}", torch_dtype=torch.float16)
        model = model.merge_and_unload()
        print(f"Merged lora with model!!!, Step: {step_num}")
        save_dir = f"{self.args.output_merged_dir}_{step_num}"
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir, safe_serialization=True)
        tokenizer = AutoTokenizer.from_pretrained(self.args.model_name)
        tokenizer.save_pretrained(save_dir)

        
        # Evaluate: infer -> evaluate
        # infer --model_type --save_dir
        
        #print("Mem gpu checking...")
        #device1 = torch.device('cuda:0')
        #device2 = torch.device('cuda:1')
        # Check the memory usage
        #print('Max memory allocated:', torch.cuda.max_memory_allocated(device) / 1024 / 1024, 'MB')
        #print('Current memory 0 allocated:', torch.cuda.memory_allocated(device1) / 1024 / 1024, 'MB')
        #print('Current memory 1 allocated:', torch.cuda.memory_allocated(device2) / 1024 / 1024, 'MB')
        #print("Infer LLM...")
        #os.system("python infer_valid_set_llm.py llama-last lora_llama_2_last")
        #(["python infer_valid_set_llm.py llama-last lora_llama_2_last"])
        #os.system(f"python ./infer_valid_set_llm.py {self.args.model_type} {save_dir}")
        # evaluate  --model_type --step_num
        #print("Evaluate") 
        #os.system(f"bash ./evaluate_llm_valid.sh {self.args.model_type} {step_num}")

        
if __name__ == "__main__":
    os.system("python infer_valid_set_llm.py llama-last lora_llama_2_last")

