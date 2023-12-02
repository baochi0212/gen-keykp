import os


from copy import deepcopy
from random import randrange
from functools import partial


import torch
import accelerate
import bitsandbytes as bnb


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
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
model_name = "HuggingFaceH4/zephyr-7b-beta"
tokenizer = AutoTokenizer.from_pretrained(model_name)
dataset = load_dataset("THUDM/AgentInstruct")


def format_prompt(sample):
    """Given a sample dictionary with key "conversations", format the conversation into a prompt.


    Args:
      sample: A sample dictionary from a Hugging Face dataset.


    Returns:
      sample: sample dictionary with "text" key for the formatted prompt.
    """


    INTRO = "Below is a conversation between a user and you."
    END = "Instruction: Write a response appropriate to the conversation."


    conversations = ""
    for response in sample["conversations"]:
      from_, value = response["from"], response["value"]
      conversations += f"<{from_}>: " + value + "\n"


    sample["text"] = "\n\n".join([INTRO, conversations, END])


    return sample


def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, dataset: str, seed: int = 42):
    # Format each prompt.
    print("Preprocessing dataset...")
    dataset = dataset.map(format_prompt)


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
        remove_columns=["conversations", "text"],
    )


    # Shuffle dataset.
    dataset = dataset.shuffle(seed=seed)


    return dataset


max_length = 3000
dataset = preprocess_dataset(tokenizer, max_length, dataset)
