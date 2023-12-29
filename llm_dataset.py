from copy import deepcopy
from random import randrange
from functools import partial

import jsonlines
import torch
import accelerate
import bitsandbytes as bnb


from datasets import load_dataset, Dataset
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
import sys
#model_name, model_type, num_samples = sys.argv[1], sys.argv[2], int(sys.argv[3])
#print("Training samples: ", num_samples)
#model_name = "HuggingFaceH4/zephyr-7b-beta"
#model_type = "zephyr_fine_tune"
model_type = "llama_fine_tune"
model_name = "./models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93"
num_samples = 10000
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./")
def make_simple_prompt(sample):
    
    document, present, absent = sample['document'], sample['present'], sample['absent']
#     # chats = []
#     PROMPT_TEMPLATE = """{}
# Present keyphrase: {}
# Absent keyphrase: {}"""
#     prompt = """Extract present and absent keyphrase in the given document. Given that present keyphrases are phrases that appear in the Document and absent ones do not appear in Document. Both absent and present keyphrases represent Document's main idea. 
# Document: {}
# Below are present and absent keyphrases capturing document's information:"""
    PROMPT_TEMPLATE = """### Instruction:
Extract present and absent keyphrases in the Input. Given that present keyphrases are phrases that appear in the Input and absent ones do not appear in Input. Both absent and present keyphrases represent main idea of Input's Document. 
### Input: {}
### Response:
Present: {}
Absent: {}
### End"""
    # chat = [
    # {'role': 'user', 'content': prompt.format(document)}
    # ]
    # tokenizer.use_default_system_prompt = False
    # chat = tokenizer.apply_chat_template(chat, tokenize=False).strip()
    # prompt = PROMPT_TEMPLATE.format(chat, present, absent)
    # sample['chat'] = chat
    # sample['text'] = prompt
    sample['text'] = PROMPT_TEMPLATE.format(document, present, absent)
    # sample['text'] = tokenizer.apply_chat_template(chat, tokenize=False)


    return sample

def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, dataset: str, seed: int = 42):
    # Format each prompt.
    print("Preprocessing dataset...")
    dataset = dataset.map(make_simple_prompt)


    def preprocess_batch(batch, tokenizer, max_length):
        return tokenizer(
            batch["text"],
            max_length=max_length,
            truncation=True,
        )


    # Apply preprocessing to each batch of the dataset & and remove "conversations" and "text" fields.
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        # remove_columns=["conversations", "text"],
    )


    # Shuffle dataset.
    dataset = dataset.shuffle(seed=seed)


    return dataset



max_length = 1024
def gen_sample_kp():
    #os.system("rm -rf ./hf_data/*")
    for i, line in enumerate(jsonlines.open("./data/kp20k_train_llm.jsonl")):
        #500 steps batch 8
        if i < num_samples:
            yield line
dataset = Dataset.from_generator(gen_sample_kp, cache_dir='./hf_data')
dataset = preprocess_dataset(tokenizer, max_length, dataset)
#print("Num samples:", len(dataset))
print(dataset[10]['text'])
# for i, line in enumerate(jsonlines.open("/home/speechmt/chitb/SimCKP/data/kp20k_train_llm.jsonl")):
#     if i < 1:
#         print(make_simple_prompt(line))
